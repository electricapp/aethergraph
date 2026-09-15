// Single-snapshot seqlock reader for device-resident FeatureTable slots.
//
// Slot layout comes from common.cuh: u64 head @ 0, f32 features @
// AETHER_FEATURE_OFFSET, u64 tail at feature_tail_offset(feature_dim). A
// valid row has equal, nonzero, even versions. Producers publish the tail
// with release ordering.
//
// One warp per row, like the two-snapshot validator: thread-per-row puts
// lanes a slot apart, 32 sector fetches where one coalesced read would do.
//
// Litmus sources live in crates/aether-stream/litmus/k5_3/.
// TODO(HARDWARE): run herd7 + compute-sanitizer racecheck on a real GPU for
// acquire/release claims.

// `.acquire` is PTX ISA 6.0, so this unit needs sm_70 — which
// gpu::kernels::NVRTC_ARCH_FLOOR and compile_for_device both guarantee.
// Below that the driver JIT rejects it as CUDA_ERROR_INVALID_PTX, so say so
// at compile time instead.
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

extern "C" __global__ void seqlock_snapshot_rows(
    const char* slots,
    float* output,
    int* valid_mask,
    int feature_dim,
    int row_count,
    int slot_size
) {
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int warps = (int)((gridDim.x * blockDim.x) >> 5);
    if (warps == 0) return;

    const int tail_offset = feature_tail_offset(feature_dim);

    for (int idx = warp; idx < row_count; idx += warps) {
        const char* slot = slots + (long long)idx * slot_size;
        const unsigned long long head =
            load_acquire_sys_u64((const unsigned long long*)slot);
        const unsigned long long tail =
            load_acquire_sys_u64((const unsigned long long*)(slot + tail_offset));
        const bool valid = head == tail && head != 0 && (head & 1) == 0;
        if (lane == 0) valid_mask[idx] = valid ? 1 : 0;
        if (!valid) continue;

        const unsigned int* src = (const unsigned int*)(slot + AETHER_FEATURE_OFFSET);
        unsigned int* dst = (unsigned int*)(output + (long long)idx * feature_dim);

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
    }
}
