// Single-snapshot seqlock reader for device-resident FeatureTable slots.
//
// Slot layout: u64 head @ 0, f32 features @ 8, and u64 tail at
// align_up(8 + feature_dim * 4, 8).  A valid row has equal, nonzero, even
// versions.  Producers publish the tail with release ordering.
//
// Litmus sources live in crates/aether-stream/litmus/k5_3/.
// TODO(HARDWARE): run herd7 + compute-sanitizer racecheck on a real GPU for
// acquire/release claims.

__device__ __forceinline__ unsigned long long load_acquire_sys_u64(
    const unsigned long long* ptr
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    unsigned long long value;
    asm volatile("ld.acquire.sys.u64 %0, [%1];" : "=l"(value) : "l"(ptr) : "memory");
    return value;
#else
    // NVRTC targets predating Volta do not accept ld.acquire.  A normal load
    // followed by a system acquire/release fence is the conservative fallback.
    unsigned long long value = *ptr;
    asm volatile("fence.acq_rel.sys;" ::: "memory");
    return value;
#endif
}

extern "C" __global__ void seqlock_snapshot_rows(
    const char* slots,
    float* output,
    int* valid_mask,
    int feature_dim,
    int row_count,
    int slot_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= row_count) return;

    const char* slot = slots + (long long)idx * slot_size;
    const int tail_offset = (8 + feature_dim * 4 + 7) & ~7;
    const unsigned long long head =
        load_acquire_sys_u64((const unsigned long long*)slot);
    const unsigned long long tail =
        load_acquire_sys_u64((const unsigned long long*)(slot + tail_offset));
    const bool valid = head == tail && head != 0 && (head & 1) == 0;
    valid_mask[idx] = valid ? 1 : 0;
    if (!valid) return;

    const float* features = (const float*)(slot + 8);
    float* dst = output + (long long)idx * feature_dim;
    for (int feature = 0; feature < feature_dim; ++feature) {
        dst[feature] = __ldg(features + feature);
    }
}
