// K2.1 IBGDA — GPU warp posts mlx5 RDMA READ WQEs and bumps the doorbell record.
//
// WQE layout matches aethergraph_core::Mlx5RdmaReadWqe (48 bytes). Host maps
// the DBR and BlueFlame separately; this kernel only fills the ring + DBR.
//
// TODO(HARDWARE): compare against NVSHMEM IBGDA on ConnectX.

struct Mlx5RdmaReadWqe {
    unsigned int ctrl_opmod_idx_opcode;
    unsigned int ctrl_qpn_ds;
    unsigned char ctrl_signature;
    unsigned char ctrl_reserved[3];
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
    Mlx5RdmaReadWqe* ring,
    unsigned int* send_db, // points at DBR send half (host-endian u16 stored as u32)
    unsigned int qpn,
    unsigned int depth_mask, // depth - 1
    unsigned int* wqe_counter, // atomic host/device counter
    const unsigned long long* local_addrs,
    const unsigned int* lkeys,
    const unsigned int* byte_counts,
    const unsigned long long* remote_addrs,
    const unsigned int* rkeys,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const unsigned int idx = atomicAdd(wqe_counter, 1u);
    const unsigned int slot = idx & depth_mask;

    Mlx5RdmaReadWqe wqe;
    const unsigned int opmod_idx_opcode = (idx << 8) | 0x10u; // RDMA_READ
    const unsigned int qpn_ds = (qpn << 8) | 3u;
    wqe.ctrl_opmod_idx_opcode = bswap32(opmod_idx_opcode);
    wqe.ctrl_qpn_ds = bswap32(qpn_ds);
    wqe.ctrl_signature = 0;
    wqe.ctrl_reserved[0] = wqe.ctrl_reserved[1] = wqe.ctrl_reserved[2] = 0;
    wqe.ctrl_imm = 0;
    wqe.byte_count = bswap32(byte_counts[i]);
    wqe.lkey = bswap32(lkeys[i]);
    wqe.local_address = bswap64(local_addrs[i]);
    wqe.remote_address = bswap64(remote_addrs[i]);
    wqe.rkey = bswap32(rkeys[i]);
    wqe.remote_reserved = 0;

    ring[slot] = wqe;
    __threadfence_system();
    // Publish send doorbell (wqe index + 1), host-endian for the CPU oracle;
    // BlueFlame path byte-swaps at the MMIO edge on the ConnectX rig.
    atomicExch(send_db, idx + 1u);
}
