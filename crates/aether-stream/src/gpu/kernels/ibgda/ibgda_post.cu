// K2.1 IBGDA — GPU warp posts mlx5 RDMA READ WQEs and bumps the doorbell record.
//
// WQE payload is 48 bytes; SQ stride is 64-byte basic blocks (pad to BB).
// WQE index is u16 (matches host IbgdaQueue). Doorbell advances with a CAS
// loop that only stores a strictly newer index (wrapping half-range), matching
// host Mlx5DoorbellRecord::ring_send_monotonic.
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
    unsigned int byte_count;
    unsigned int lkey;
    unsigned long long local_address;
    unsigned long long remote_address;
    unsigned int rkey;
    unsigned int remote_reserved;
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
    unsigned int* send_db,        // host-endian logical send index (low 16 used)
    unsigned int qpn,
    unsigned int depth_mask,      // depth - 1
    unsigned int* wqe_counter,    // low 16 bits used as u16 counter
    const unsigned long long* local_addrs,
    const unsigned int* lkeys,
    const unsigned int* byte_counts,
    const unsigned long long* remote_addrs,
    const unsigned int* rkeys,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const unsigned int idx32 = atomicAdd(wqe_counter, 1u);
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
    wqe.byte_count = bswap32(byte_counts[i]);
    wqe.lkey = bswap32(lkeys[i]);
    wqe.local_address = bswap64(local_addrs[i]);
    wqe.remote_address = bswap64(remote_addrs[i]);
    wqe.rkey = bswap32(rkeys[i]);
    wqe.remote_reserved = 0;

    Mlx5RdmaReadWqe* dst =
        (Mlx5RdmaReadWqe*)(ring + (unsigned long long)slot * MLX5_WQE_BB);
    *dst = wqe;
    __threadfence_system();

    // Monotonic doorbell with wrapping half-range compare (matches host).
    const unsigned int next = (idx + 1u) & 0xffffu;
    unsigned int old;
    do {
        old = *send_db;
        const unsigned int ahead = (next - (old & 0xffffu)) & 0xffffu;
        if (ahead == 0u || ahead > 0x8000u)
            break;
    } while (atomicCAS(send_db, old, next) != old);
}
