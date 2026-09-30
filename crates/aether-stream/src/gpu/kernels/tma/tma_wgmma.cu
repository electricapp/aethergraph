// K5.4 dense aggregation — bandwidth-roofed GEMV (not sparse gather).
//
// Shape: out[row] += sum_k a[row,k] * b[k]
//   - B staged into shared memory once per CTA (reused across rows)
//   - one warp per row: consecutive lanes read consecutive columns, so a
//     row streams as one coalesced run instead of 32 rows a stride apart
//   - A consumed with ld.global.cs.v4.f32 when every row base is 16-byte
//     aligned (cols % 4 == 0 over a 16-byte-aligned A); scalar otherwise
//   - lane partials reduced with a shuffle tree
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

    const unsigned int full = 0xffffffffu;
    const int lane = (int)(threadIdx.x & 31);
    const long long warp =
        ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const long long warps = ((long long)gridDim.x * blockDim.x) >> 5;
    const bool vector =
        (cols & 3) == 0 && (((unsigned long long)a) & 15ULL) == 0;

    for (long long row = warp; row < rows; row += warps) {
        const float* a_row = a + row * cols;
        float acc = 0.f;
        int k = 0;
        if (vector) {
            // 128 columns per round: lane l takes [4l, 4l + 4).
            const int rounds = cols / 128;
            for (int r = 0; r < rounds; ++r) {
                const int base = r * 128 + lane * 4;
                const float4 av = ld_cs_v4f32(a_row + base);
                acc += av.x * sb[base] + av.y * sb[base + 1] + av.z * sb[base + 2]
                     + av.w * sb[base + 3];
            }
            k = rounds * 128;
        }
        for (int c = k + lane; c < cols; c += 32) {
            acc += ld_cs_f32(a_row + c) * sb[c];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(full, acc, offset);
        }
        if (lane == 0) {
            out[row] = out[row] + acc;
        }
    }
}
