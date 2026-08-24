// K5.4 dense aggregation — bandwidth-roofed GEMV (not sparse gather).
//
// Shape: out[row] += sum_k a[row,k] * b[k]
//   - B staged into shared memory once per CTA (reused across rows)
//   - A consumed with ld.global.cs.v4.f32 (streaming, once-touched)
//   - multiple independent FMA accumulators for ILP
//
// Hopper TMA + WGMMA / Blackwell tcgen05 need host-side tensor-map
// descriptors (cuTensorMapEncodeTiled) which NVRTC string kernels cannot
// build alone. This path targets the GEMV DRAM roof; the ISA path stays
// gated behind require_sm(90) host checks + TODO(HARDWARE) descriptor wire-up.

extern "C" __global__ void dense_tile_accumulate(
    const float* a,
    const float* b,
    float* out,
    int rows,
    int cols
) {
    extern __shared__ float sb[];

    // Cooperative load of B. Host sizes dynamic smem to cols * sizeof(float)
    // (capped in the Rust launcher).
    for (int k = (int)threadIdx.x; k < cols; k += (int)blockDim.x) {
        sb[k] = b[k];
    }
    __syncthreads();

    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;

    const float* a_row = a + (long long)row * cols;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    int k = 0;
    // 128-bit streaming loads of A; B hits smem.
    for (; k + 3 < cols; k += 4) {
        const float4 av = ld_cs_v4f32(a_row + k);
        acc0 += av.x * sb[k];
        acc1 += av.y * sb[k + 1];
        acc2 += av.z * sb[k + 2];
        acc3 += av.w * sb[k + 3];
    }
    float acc = acc0 + acc1 + acc2 + acc3;
    for (; k < cols; ++k) {
        acc += ld_cs_f32(a_row + k) * sb[k];
    }
    out[row] = out[row] + acc;
}
