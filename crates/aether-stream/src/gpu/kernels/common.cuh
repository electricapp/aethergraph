// Shared device helpers for aether-stream GPU kernels (KERNELS.md Tier A).
// Included via string concat / paste into NVRTC units that need Philox or
// FeatureTable slot layout constants. Keep this header freestanding — no
// host includes.

#pragma once

// FeatureTable slot: u64 head @ 0, f32 features @ 8, u64 tail at
// align_up(8 + feature_dim * 4, 8).
__device__ __forceinline__ int feature_tail_offset(int feature_dim) {
    return (8 + feature_dim * 4 + 7) & ~7;
}

// K5.6: streaming / read-only global loads. `.cs` = cache-streaming
// (evict-first); `.nc` / __ldg = read-only path. Use on once-touched
// payload and neighbor streams so working-set lines stay in L1/L2.

__device__ __forceinline__ unsigned int ld_cs_u32(const unsigned int* p) {
    unsigned int v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned long long ld_cs_u64(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.global.cs.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float ld_cs_f32(const float* p) {
    float v;
    asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float4 ld_cs_v4f32(const float* p) {
    float4 v;
    asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 ld_cs_v4u32(const unsigned int* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// Warp inclusive scan (Hillis–Steele) over the calling warp's 32 lanes.
__device__ __forceinline__ unsigned int warp_inclusive_scan_u32(unsigned int v) {
    const unsigned int mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const unsigned int n = __shfl_up_sync(mask, v, offset);
        if (lane >= offset) v += n;
    }
    return v;
}

// Warp exclusive scan: inclusive(v) - v.
__device__ __forceinline__ unsigned int warp_exclusive_scan_u32(unsigned int v) {
    return warp_inclusive_scan_u32(v) - v;
}

// Philox4x32-10 (Random123). Counter (node_lo, node_hi, layer, draw_group)
// matches aethergraph_core::philox4x32_10 for draw_group 0.
__device__ __forceinline__ uint4 aether_philox4x32_10(
    unsigned long long seed, unsigned int layer, unsigned long long node,
    unsigned int draw_group
) {
    unsigned int c0 = (unsigned int)node, c1 = (unsigned int)(node >> 32);
    unsigned int c2 = layer, c3 = draw_group;
    unsigned int k0 = (unsigned int)seed, k1 = (unsigned int)(seed >> 32);
#pragma unroll
    for (int round = 0; round < 10; ++round) {
        unsigned int hi0 = __umulhi(c0, 0xD2511F53U);
        unsigned int lo0 = c0 * 0xD2511F53U;
        unsigned int hi1 = __umulhi(c2, 0xCD9E8D57U);
        unsigned int lo1 = c2 * 0xCD9E8D57U;
        unsigned int next0 = hi1 ^ c1 ^ k0;
        unsigned int next2 = hi0 ^ c3 ^ k1;
        c0 = next0; c1 = lo1; c2 = next2; c3 = lo0;
        if (round != 9) {
            k0 += 0x9E3779B9U;
            k1 += 0xBB67AE85U;
        }
    }
    return make_uint4(c0, c1, c2, c3);
}

__device__ __forceinline__ unsigned int aether_philox_u32(
    unsigned long long seed, unsigned int layer, unsigned long long node,
    unsigned int draw_index
) {
    const uint4 r = aether_philox4x32_10(seed, layer, node, draw_index >> 2);
    switch (draw_index & 3) {
        case 1: return r.y;
        case 2: return r.z;
        case 3: return r.w;
        default: return r.x;
    }
}
