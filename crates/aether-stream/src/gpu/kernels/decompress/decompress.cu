// K5.5 GPU decompression — StreamVByte + Elias-Fano (roofline attempt).
//
// StreamVByte: one warp cooperatively decodes waves of ≤32 deltas.
//   1) each lane reads its 2-bit length tag
//   2) warp exclusive scan → byte offsets into the data stream
//   3) gather deltas, warp inclusive scan → absolute values
// Prefix dependence prevents full grid parallelism; bandwidth wins come
// from coalesced control/data traffic inside the warp.
//
// Elias-Fano: one CTA expands the high-bits bitmap in parallel when
// high_words ≤ blockDim (≤256). Larger bitmaps fall back to a linear
// pass (same result as EliasFano::to_vec).
//
// TODO(HARDWARE): Blackwell hardware decompress crossover curve.

__device__ __forceinline__ unsigned long long read_low_bits(
    const unsigned long long* low, unsigned int low_bits, unsigned int index
) {
    if (low_bits == 0) return 0;
    const unsigned long long bit = (unsigned long long)index * low_bits;
    const unsigned int word = (unsigned int)(bit / 64);
    const unsigned int offset = (unsigned int)(bit % 64);
    if (offset + low_bits <= 64) {
        const unsigned long long mask =
            low_bits == 64 ? ~0ULL : ((1ULL << low_bits) - 1ULL);
        return (ld_cs_u64(low + word) >> offset) & mask;
    }
    const unsigned int lo_bits = 64 - offset;
    const unsigned long long lo_mask = (1ULL << lo_bits) - 1ULL;
    const unsigned long long hi_mask = (1ULL << (low_bits - lo_bits)) - 1ULL;
    const unsigned long long w0 = ld_cs_u64(low + word);
    const unsigned long long w1 = ld_cs_u64(low + word + 1);
    return ((w0 >> offset) & lo_mask) | ((w1 & hi_mask) << lo_bits);
}

__device__ void ef_linear_decode(
    const unsigned long long* low,
    const unsigned long long* high,
    unsigned long long* output,
    int len,
    int high_words,
    unsigned int low_bits
) {
    int i = 0;
    for (int word_idx = 0; word_idx < high_words && i < len; ++word_idx) {
        unsigned long long w = ld_cs_u64(high + word_idx);
        while (w != 0 && i < len) {
            const int bit = __ffsll((long long)w) - 1;
            const unsigned long long set_pos =
                (unsigned long long)word_idx * 64ULL + (unsigned long long)bit;
            w &= w - 1;
            const unsigned long long high_part = set_pos - (unsigned long long)i;
            const unsigned long long low_part =
                read_low_bits(low, low_bits, (unsigned int)i);
            output[i] = (high_part << low_bits) | low_part;
            ++i;
        }
    }
}

// Inclusive Hillis–Steele over the first `n` entries of `s` (n ≤ blockDim.x).
__device__ void block_inclusive_scan_u32(unsigned int* s, int n) {
    for (int offset = 1; offset < n; offset <<= 1) {
        unsigned int v = 0;
        if ((int)threadIdx.x < n) v = s[threadIdx.x];
        __syncthreads();
        if ((int)threadIdx.x < n && (int)threadIdx.x >= offset) {
            v += s[threadIdx.x - offset];
        }
        __syncthreads();
        if ((int)threadIdx.x < n) s[threadIdx.x] = v;
        __syncthreads();
    }
}

extern "C" __global__ void streamvbyte_delta_decode(
    const unsigned char* control,
    const unsigned char* data,
    unsigned int* output,
    int len,
    unsigned int first
) {
    if (blockIdx.x != 0 || (threadIdx.x >> 5) != 0 || len <= 0) return;

    const int lane = threadIdx.x & 31;
    if (lane == 0) {
        output[0] = first;
    }
    __syncwarp();

    unsigned int acc = first;
    int data_pos = 0;
    int produced = 1;

    while (produced < len) {
        const int remaining = len - produced;
        const int wave = remaining < 32 ? remaining : 32;
        const int di = produced - 1 + lane;
        unsigned int nbytes = 0;
        if (lane < wave) {
            nbytes = (unsigned int)(((control[di / 4] >> ((di % 4) * 2)) & 3) + 1);
        }
        const unsigned int excl = warp_exclusive_scan_u32(nbytes);
        const unsigned int wave_bytes =
            __shfl_sync(0xffffffffu, excl + nbytes, wave > 0 ? wave - 1 : 0);

        unsigned int delta = 0;
        if (lane < wave) {
            const unsigned char* p = data + data_pos + (int)excl;
            delta = (unsigned int)p[0];
            if (nbytes >= 2) delta |= (unsigned int)p[1] << 8;
            if (nbytes >= 3) delta |= (unsigned int)p[2] << 16;
            if (nbytes >= 4) delta |= (unsigned int)p[3] << 24;
        }

        unsigned int scan = lane < wave ? delta : 0u;
        scan = warp_inclusive_scan_u32(scan);
        if (lane < wave) {
            output[produced + lane] = acc + scan;
        }
        if (wave > 0) {
            acc = __shfl_sync(0xffffffffu, acc + scan, wave - 1);
        }
        data_pos += (int)wave_bytes;
        produced += wave;
        __syncwarp();
    }
}

extern "C" __global__ void elias_fano_decode_all(
    const unsigned long long* low,
    const unsigned long long* high,
    unsigned long long* output,
    int len,
    int high_words,
    unsigned int low_bits
) {
    if (blockIdx.x != 0 || len <= 0) return;

    extern __shared__ unsigned int scratch[];
    unsigned int* pops = scratch;
    unsigned int* excl = scratch + blockDim.x;

    const int tid = (int)threadIdx.x;
    const int nthreads = (int)blockDim.x;

    if (high_words > nthreads) {
        if (tid == 0) {
            ef_linear_decode(low, high, output, len, high_words, low_bits);
        }
        return;
    }

    pops[tid] = 0u;
    if (tid < high_words) {
        pops[tid] = (unsigned int)__popcll(ld_cs_u64(high + tid));
    }
    __syncthreads();

    // Inclusive scan of popcounts into `excl`, then exclusive base = incl[i-1].
    excl[tid] = pops[tid];
    __syncthreads();
    block_inclusive_scan_u32(excl, high_words);
    unsigned int base = 0;
    if (tid < high_words) {
        base = (tid == 0) ? 0u : excl[tid - 1];
    }
    __syncthreads();

    if (tid < high_words) {
        unsigned long long w = ld_cs_u64(high + tid);
        unsigned int local = 0;
        while (w != 0) {
            const int bit = __ffsll((long long)w) - 1;
            w &= w - 1;
            const unsigned int i = base + local;
            if ((int)i >= len) break;
            const unsigned long long set_pos =
                (unsigned long long)tid * 64ULL + (unsigned long long)bit;
            const unsigned long long high_part = set_pos - (unsigned long long)i;
            const unsigned long long low_part = read_low_bits(low, low_bits, i);
            output[i] = (high_part << low_bits) | low_part;
            ++local;
        }
    }
}
