// K5.5 feature-payload decode: block-scaled int8 rows expanded in VRAM, so
// the link carries 1.125 bytes per feature instead of 4.
//
// Layout is aethergraph_core::BlockScaledI8: `codes` one i8 per element
// row-major, `scales` one f32 per AETHER_QUANT_BLOCK elements per row. A
// lane's 16-code vector load sits inside one block, so it never straddles
// two scales.
//
// Decode is `(float)(int8)code * scale`, one IEEE multiply, matching
// BlockScaledI8::decode bit for bit — the oracle test asserts equality, not
// a tolerance.
//
// One warp per row: consecutive lanes read consecutive codes, one 512-byte
// run rather than 32 scattered sectors.

__device__ __forceinline__ float dequant_lane(unsigned int packed, int byte, float scale) {
    const int q = (int)(signed char)((packed >> (byte * 8)) & 0xffu);
    return (float)q * scale;
}

extern "C" __global__ void dequant_block_i8_rows(
    const unsigned char* codes,  // rows * feature_dim int8, row-major
    const float* scales,         // rows * blocks_per_row, row-major
    float* output,               // rows * feature_dim f32
    int feature_dim,
    int row_count
) {
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int warps = (int)((gridDim.x * blockDim.x) >> 5);
    if (warps == 0 || feature_dim <= 0) return;

    const int blocks_per_row =
        (feature_dim + AETHER_QUANT_BLOCK - 1) / AETHER_QUANT_BLOCK;

    for (int row = warp; row < row_count; row += warps) {
        const unsigned char* row_codes = codes + (long long)row * feature_dim;
        const float* row_scales = scales + (long long)row * blocks_per_row;
        float* dst = output + (long long)row * feature_dim;

        int done = 0;
        // Vector body, 512 codes per round. Needs both rows 16B-aligned,
        // which holds when feature_dim is a multiple of 16.
        if ((feature_dim & 15) == 0) {
            const int per_round = 32 * 16;
            const int rounds = feature_dim / per_round;
            const int lane_off = lane * 16;
            for (int r = 0; r < rounds; ++r) {
                const int base = r * per_round + lane_off;
                const uint4 packed = ld_cs_v4u32((const unsigned int*)(row_codes + base));
                // One scale for all 16: AETHER_QUANT_BLOCK is a multiple of
                // 16 and `base` is 16-aligned.
                const float scale = row_scales[base / AETHER_QUANT_BLOCK];
                const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    float4 v;
                    v.x = dequant_lane(words[w], 0, scale);
                    v.y = dequant_lane(words[w], 1, scale);
                    v.z = dequant_lane(words[w], 2, scale);
                    v.w = dequant_lane(words[w], 3, scale);
                    st_cs_v4f32(dst + base + w * 4, v);
                }
            }
            done = rounds * per_round;
        }

        // Scalar remainder, and the whole row when feature_dim % 16 != 0.
        for (int e = done + lane; e < feature_dim; e += 32) {
            const float scale = row_scales[e / AETHER_QUANT_BLOCK];
            const int q = (int)(signed char)row_codes[e];
            st_cs_f32(dst + e, (float)q * scale);
        }
    }
}
