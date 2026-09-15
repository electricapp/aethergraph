// Host-side stand-ins for the CUDA builtins the Tier A kernels use, so a
// plain clang++ can parse `common.cuh` + a `.cu` unit with `-fsyntax-only`.
//
// A syntax and type gate only: whether a kernel declares what it uses and
// type-checks. ptxas judges the inline PTX and a GPU judges the answer.

#pragma once

#define __device__
#define __global__
#define __host__
#define __forceinline__ inline
#define __shared__
#define __restrict__

struct uint4 {
    unsigned int x, y, z, w;
};
struct float4 {
    float x, y, z, w;
};
struct dim3_shim {
    unsigned int x, y, z;
};

static dim3_shim threadIdx, blockIdx, blockDim, gridDim;

static inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) {
    return uint4{x, y, z, w};
}

// `__syncthreads` and friends come from clang's own nvptx builtins; only
// what it does not declare is stubbed below.
static inline void __syncwarp(unsigned int = 0xffffffffu) {}
static inline void __nanosleep(unsigned int) {}
static inline void __threadfence_block() {}
static inline void __threadfence_system() {}

static inline unsigned int __ballot_sync(unsigned int, bool) { return 0; }
static inline bool __any_sync(unsigned int, bool) { return false; }
static inline bool __all_sync(unsigned int, bool) { return false; }
static inline unsigned int __shfl_sync(unsigned int, unsigned int v, int, int = 32) { return v; }
static inline float __shfl_sync(unsigned int, float v, int, int = 32) { return v; }
static inline unsigned int __shfl_up_sync(unsigned int, unsigned int v, unsigned int, int = 32) {
    return v;
}
static inline unsigned long long __shfl_sync(
    unsigned int, unsigned long long v, int, int = 32
) {
    return v;
}

static inline int __ffsll(long long) { return 0; }
static inline int __popcll(unsigned long long) { return 0; }
static inline int __popc(unsigned int) { return 0; }
static inline unsigned int __umulhi(unsigned int, unsigned int) { return 0; }
static inline float __expf(float v) { return v; }
static inline float __ldg(const float* p) { return *p; }
static inline unsigned int __ldg(const unsigned int* p) { return *p; }
static inline unsigned long long __ldg(const unsigned long long* p) { return *p; }
static inline unsigned int __float_as_uint(float) { return 0; }
static inline float __uint_as_float(unsigned int) { return 0.0f; }

static inline unsigned int __byte_perm(unsigned int, unsigned int, unsigned int) { return 0; }

template <typename T>
static inline T atomicAdd(T* p, T v) {
    T old = *p;
    *p = static_cast<T>(*p + v);
    return old;
}
template <typename T>
static inline T atomicCAS(T* p, T expected, T desired) {
    T old = *p;
    if (old == expected) *p = desired;
    return old;
}
template <typename T>
static inline T atomicExch(T* p, T v) {
    T old = *p;
    *p = v;
    return old;
}
