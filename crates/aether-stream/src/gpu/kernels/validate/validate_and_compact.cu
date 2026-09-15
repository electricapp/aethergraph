// Two-snapshot seqlock validation and feature compaction.
//
// The client READs each slot twice into separate staging regions, the second
// only after the first completed. Completions within one READ can land in any
// order, so a single snapshot's head/tail pair can be stale against its own
// payload; two ordered snapshots are what make acceptance sound (reader
// contract in feature_table.rs).
//
// Accept iff snap1.head == snap1.tail == snap2.head == snap2.tail, that
// version is even and nonzero, and the payloads are byte-identical. Accepted
// rows compact from snapshot 1; the rest flag for retry, and their output
// bytes are undefined — the compare aborts on the first bad chunk, leaving
// earlier ones written.
//
// One warp per row: thread-per-row puts lanes `slot_size` apart, 32 sectors
// per load where 4 would do. Payload traffic is once-touched `ld.global.cs`
// (`.v4` where the aligned base licenses it), AETHER_GATHER_IN_FLIGHT rounds
// in flight (K5.6).

// Compare two 16B-aligned payloads lane-cooperatively, writing snapshot 1 to
// `dst` as it goes. Bit patterns, not floats: a torn write can produce NaN,
// which a float compare would call equal. Returns false warp-uniformly on the
// first differing chunk.
__device__ __forceinline__ bool warp_compare_and_compact(
    const unsigned int* f1,
    const unsigned int* f2,
    unsigned int* dst,
    int feature_dim,
    int lane
) {
    const unsigned int all = 0xffffffffu;
    int i = 0;

    // Vector body, 128 elements per round. Needs a 16B-aligned destination
    // row, which holds when feature_dim is a multiple of 4; the source is
    // aligned by slot geometry.
    if ((feature_dim & 3) == 0) {
        const int per_round = 128;
        const int rounds = feature_dim / per_round;
        const int lane_off = lane * 4;
        int r = 0;

        for (; r + AETHER_GATHER_IN_FLIGHT <= rounds; r += AETHER_GATHER_IN_FLIGHT) {
            uint4 a[AETHER_GATHER_IN_FLIGHT];
            uint4 b[AETHER_GATHER_IN_FLIGHT];
#pragma unroll
            for (int k = 0; k < AETHER_GATHER_IN_FLIGHT; ++k) {
                const int base = (r + k) * per_round + lane_off;
                a[k] = ld_cs_v4u32(f1 + base);
                b[k] = ld_cs_v4u32(f2 + base);
            }
            bool bad = false;
#pragma unroll
            for (int k = 0; k < AETHER_GATHER_IN_FLIGHT; ++k) {
                bad |= a[k].x != b[k].x || a[k].y != b[k].y || a[k].z != b[k].z
                    || a[k].w != b[k].w;
            }
            if (__any_sync(all, bad)) return false;
#pragma unroll
            for (int k = 0; k < AETHER_GATHER_IN_FLIGHT; ++k) {
                st_cs_v4u32(dst + (r + k) * per_round + lane_off, a[k]);
            }
        }

        for (; r < rounds; ++r) {
            const int base = r * per_round + lane_off;
            const uint4 a = ld_cs_v4u32(f1 + base);
            const uint4 b = ld_cs_v4u32(f2 + base);
            const bool bad = a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
            if (__any_sync(all, bad)) return false;
            st_cs_v4u32(dst + base, a);
        }
        i = rounds * per_round;
    }

    // Scalar remainder, and the whole row when feature_dim % 4 != 0.
    // Lane-strided, so the warp still reads consecutive words.
    for (int base = i; base < feature_dim; base += 32) {
        const int e = base + lane;
        bool bad = false;
        unsigned int v = 0;
        if (e < feature_dim) {
            v = ld_cs_u32(f1 + e);
            bad = v != ld_cs_u32(f2 + e);
        }
        if (__any_sync(all, bad)) return false;
        if (e < feature_dim) st_cs_u32(dst + e, v);
    }
    return true;
}

extern "C" __global__ void validate_and_compact(
    const char* staging1,
    const char* staging2,
    float* output,
    int* retry_mask,
    int* retry_count,
    int feature_dim,
    int batch_size,
    int slot_size
) {
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int warps = (int)((gridDim.x * blockDim.x) >> 5);
    if (warps == 0) return;

    const int tail_offset = feature_tail_offset(feature_dim);
    int retries = 0;

    // Grid-stride: the launch sizes the grid to the device, not the batch.
    for (int idx = warp; idx < batch_size; idx += warps) {
        const char* slot1 = staging1 + (long long)idx * slot_size;
        const char* slot2 = staging2 + (long long)idx * slot_size;

        // Ordinary loads — ordering matters for the seqlock — and warp-uniform,
        // so 32 lanes broadcast from one sector.
        const unsigned long long head1 = *(const unsigned long long*)(slot1);
        const unsigned long long head2 = *(const unsigned long long*)(slot2);
        const unsigned long long tail1 = *(const unsigned long long*)(slot1 + tail_offset);
        const unsigned long long tail2 = *(const unsigned long long*)(slot2 + tail_offset);

        bool ok = head1 == tail1 && head1 == head2 && head1 == tail2
            && (head1 & 1) == 0 && head1 != 0;

        if (ok) {
            ok = warp_compare_and_compact(
                (const unsigned int*)(slot1 + AETHER_FEATURE_OFFSET),
                (const unsigned int*)(slot2 + AETHER_FEATURE_OFFSET),
                (unsigned int*)(output + (long long)idx * feature_dim),
                feature_dim,
                lane
            );
        }

        if (lane == 0) {
            retry_mask[idx] = ok ? 0 : 1;
            retries += ok ? 0 : 1;
        }
    }

    // One atomic per warp for the whole run, not one per torn row.
    if (lane == 0 && retries != 0) {
        atomicAdd(retry_count, retries);
    }
}
