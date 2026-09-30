// K2.1 IBGDA — GPU warp posts mlx5 RDMA READ WQEs and bumps the doorbell record.
//
// WQE payload is 48 bytes — control, remote-address, then data segment, the
// order the HCA parses — at a 64-byte basic-block stride. WQE index is u16
// (matches host IbgdaQueue). Threads claim indices with an atomic, write
// their WQE, then publish in claim order: each waits until every lower index
// is out before writing the doorbell record, so the HCA never fetches a slot
// still being written (as host IbgdaQueue::post_rdma_read does). The host
// launcher keeps claims within the free depth of the queue.
//
// TODO(HARDWARE): compare against NVSHMEM IBGDA on ConnectX.

#define MLX5_WQE_BB 64
#define MLX5_OPCODE_RDMA_READ 0x10u
#define MLX5_WQE_CTRL_CQ_UPDATE (2u << 2)

struct Mlx5RdmaReadWqe {
    unsigned int ctrl_opmod_idx_opcode;
    unsigned int ctrl_qpn_ds;
    unsigned char ctrl_signature;
    unsigned char ctrl_rsvd[2];
    unsigned char ctrl_fm_ce_se;
    unsigned int ctrl_imm;
    unsigned long long remote_address;
    unsigned int rkey;
    unsigned int remote_reserved;
    unsigned int byte_count;
    unsigned int lkey;
    unsigned long long local_address;
};

__device__ __forceinline__ unsigned int bswap32(unsigned int x) {
    return __byte_perm(x, 0, 0x0123);
}

__device__ __forceinline__ unsigned long long bswap64(unsigned long long x) {
    return ((unsigned long long)bswap32((unsigned int)x) << 32)
        | bswap32((unsigned int)(x >> 32));
}

extern "C" __global__ void ibgda_post_rdma_read(
    unsigned char* ring,          // depth * 64 bytes
    unsigned int* dbr,            // doorbell record: [recv, send], big-endian
    unsigned int qpn,
    unsigned int depth_mask,      // depth - 1
    unsigned int* claimed,        // next index to claim
    unsigned int* ready,          // count of WQEs published through `dbr`
    const unsigned long long* local_addrs,
    const unsigned int* lkeys,
    const unsigned int* byte_counts,
    const unsigned long long* remote_addrs,
    const unsigned int* rkeys,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const unsigned int idx32 = atomicAdd(claimed, 1u);
    const unsigned int idx = idx32 & 0xffffu;
    const unsigned int slot = idx & depth_mask;

    Mlx5RdmaReadWqe wqe;
    const unsigned int opmod_idx_opcode = (idx << 8) | MLX5_OPCODE_RDMA_READ;
    const unsigned int qpn_ds = (qpn << 8) | 3u;
    wqe.ctrl_opmod_idx_opcode = bswap32(opmod_idx_opcode);
    wqe.ctrl_qpn_ds = bswap32(qpn_ds);
    wqe.ctrl_signature = 0;
    wqe.ctrl_rsvd[0] = wqe.ctrl_rsvd[1] = 0;
    wqe.ctrl_fm_ce_se = (unsigned char)MLX5_WQE_CTRL_CQ_UPDATE;
    wqe.ctrl_imm = 0;
    wqe.remote_address = bswap64(remote_addrs[i]);
    wqe.rkey = bswap32(rkeys[i]);
    wqe.remote_reserved = 0;
    wqe.byte_count = bswap32(byte_counts[i]);
    wqe.lkey = bswap32(lkeys[i]);
    wqe.local_address = bswap64(local_addrs[i]);

    Mlx5RdmaReadWqe* dst =
        (Mlx5RdmaReadWqe*)(ring + (unsigned long long)slot * MLX5_WQE_BB);
    *dst = wqe;
    __threadfence_system();

    // Publish in claim order. Whoever holds a lower index already claimed
    // it, so it is resident and will publish (independent thread
    // scheduling, sm_70+, lets a spinning lane yield to it).
    while (*((volatile unsigned int*)ready) != idx32) {
        __nanosleep(32);
    }
    __threadfence_system();
    ((volatile unsigned int*)dbr)[1] = bswap32((idx32 + 1u) & 0xffffu);
    __threadfence_system();
    atomicExch(ready, idx32 + 1u);
}
