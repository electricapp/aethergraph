// Single-snapshot seqlock reader for device-resident FeatureTable slots.
//
// Slot layout comes from common.cuh: u64 head @ 0, f32 features @
// AETHER_FEATURE_OFFSET, u64 tail at feature_tail_offset(feature_dim). A
// valid row has equal, nonzero, even versions before the copy and the same
// head after it.
//
// Reader protocol, mirroring FeatureTable::read_node: acquire head and tail,
// copy the payload, fence, re-acquire head. The writer fences between its
// head->odd RMW and its payload stores, so a payload word from a write that
// began after the first head load implies the re-load sees that write's odd
// head (or a later version). Without the re-load a writer starting after the
// version loads is invisible and a torn row passes.
//
// One warp per row, like the two-snapshot validator: thread-per-row puts
// lanes a slot apart, 32 sector fetches where one coalesced read would do.
// Every lane checks its own loads and the warp votes, so no lane's payload
// read escapes the re-check.
//
// Litmus sources live in crates/aether-stream/litmus/k5_3/.
// TODO(HARDWARE): run herd7 + compute-sanitizer racecheck on a real GPU for
// acquire/release claims.

// `.acquire` and `fence.acq_rel` are PTX ISA 6.0, so this unit needs sm_70 —
// which gpu::kernels::NVRTC_ARCH_FLOOR and compile_for_device both
// guarantee. Below that the driver JIT rejects it as CUDA_ERROR_INVALID_PTX,
// so say so at compile time instead.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#error "seqlock_reader needs sm_70 for ld.acquire.sys; see NVRTC_ARCH_FLOOR"
#endif

__device__ __forceinline__ unsigned long long load_acquire_sys_u64(
    const unsigned long long* ptr
) {
    unsigned long long value;
    asm volatile("ld.acquire.sys.u64 %0, [%1];" : "=l"(value) : "l"(ptr) : "memory");
    return value;
}

__device__ __forceinline__ void fence_acq_rel_sys() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

extern "C" __global__ void seqlock_snapshot_rows(
    const char* slots,
    float* output,
    int* valid_mask,
    int feature_dim,
    int row_count,
    long long slot_size
) {
    const unsigned int full = 0xffffffffu;
    const int lane = (int)(threadIdx.x & 31);
    const long long warp =
        ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const long long warps = ((long long)gridDim.x * blockDim.x) >> 5;
    if (warps == 0) return;

    const int tail_offset = feature_tail_offset(feature_dim);

    for (long long idx = warp; idx < row_count; idx += warps) {
        const char* slot = slots + idx * slot_size;
        const unsigned long long* head_ptr = (const unsigned long long*)slot;
        const unsigned long long head = load_acquire_sys_u64(head_ptr);
        const unsigned long long tail =
            load_acquire_sys_u64((const unsigned long long*)(slot + tail_offset));
        const bool stable = head == tail && head != 0 && (head & 1) == 0;
        if (!__all_sync(full, stable)) {
            if (lane == 0) valid_mask[idx] = 0;
            continue;
        }

        const unsigned int* src = (const unsigned int*)(slot + AETHER_FEATURE_OFFSET);
        unsigned int* dst = (unsigned int*)(output + idx * feature_dim);

        int i = 0;
        if ((feature_dim & 3) == 0) {
            const int per_round = 128;
            const int rounds = feature_dim / per_round;
            const int lane_off = lane * 4;
            for (int r = 0; r < rounds; ++r) {
                const int base = r * per_round + lane_off;
                st_cs_v4u32(dst + base, ld_cs_v4u32(src + base));
            }
            i = rounds * per_round;
        }
        for (int e = i + lane; e < feature_dim; e += 32) {
            st_cs_u32(dst + e, ld_cs_u32(src + e));
        }

        // Order this lane's payload loads before its head re-load.
        fence_acq_rel_sys();
        const bool unchanged = load_acquire_sys_u64(head_ptr) == head;
        const bool valid = __all_sync(full, unchanged);
        if (lane == 0) valid_mask[idx] = valid ? 1 : 0;
    }
}
