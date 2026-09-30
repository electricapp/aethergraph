// K5.2 warp-cooperative neighbor sampler (roofline attempt).
//
// Roofline shape for sparse sampling is memory-bound on the neighbor stream:
//   - one warp per seed (Algorithm R is sequential in t — cannot parallelize
//     the reservoir decisions without breaking Philox bit-parity)
//   - neighbors arrive in 128-bit ld.global.cs chunks into a small shared
//     window so the sequential consumer stays fed from L1/smem
//   - reservoir lives in registers; live-lane ballot + lane == j replace
//   - degree <= fanout: coalesced vector copy with ld.cs
//   - fanout > 32: with-replacement draws, 4 Philox lanes per round
//
// Every row reports how many of its `fanout` slots hold a sample in
// `counts`, and the unused tail is filled with AETHER_SAMPLE_PAD, so a short
// row (degree < fanout, degree 0, or a seed outside the graph) is never read
// as edges to node 0.
//
// C-tree leaves are ≤15 u32s; the same Algorithm R over an in-order leaf
// stream is bit-identical to CSR. A tagged-offset arena walker is the
// next wire-up (same kernel body over a chunk iterator).
//
// TODO(HARDWARE): bandwidth counters vs. DRAM roof; C-tree arena on-device.

// Mirrors SAMPLE_PAD in sampler/mod.rs.
#define AETHER_SAMPLE_PAD 0xffffffffu

extern "C" __global__ void warp_sample_neighbors(
    const unsigned long long* offsets,   // num_nodes + 1, validated CSR
    const unsigned int* neighbors,
    unsigned long long num_nodes,
    const unsigned long long* nodes,
    unsigned int* output,                // node_count * fanout
    unsigned int* counts,                // node_count
    int node_count,
    int fanout,
    unsigned long long seed,
    unsigned int layer
) {
    // Shared prefetch window: 32 warps/block × 32 u32 = 4 KiB.
    __shared__ unsigned int win[32 * 32];

    const unsigned int full = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    unsigned int* wbuf = win + warp_in_block * 32;
    const long long first_row =
        ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const long long row_stride = ((long long)gridDim.x * blockDim.x) >> 5;

    for (long long row = first_row; row < node_count; row += row_stride) {
        unsigned int* row_out = output + row * fanout;
        const unsigned long long node = nodes[row];
        unsigned long long degree = 0;
        unsigned long long begin = 0;
        if (node < num_nodes) {
            begin = ld_cs_u64(offsets + node);
            degree = ld_cs_u64(offsets + node + 1) - begin;
        }

        unsigned int taken;
        if (degree == 0) {
            taken = 0;
        } else if (fanout > 32) {
            // ---- with-replacement (large fanout): indexed loads ----
            for (int draw = lane; draw < fanout; draw += 32) {
                // One Philox round yields 4 u32s; burn the matching lane.
                const unsigned int r =
                    aether_philox_u32(seed, layer, node, (unsigned int)draw);
                row_out[draw] = ld_cs_u32(neighbors + begin + (r % degree));
            }
            taken = (unsigned int)fanout;
        } else if ((unsigned long long)fanout >= degree) {
            // ---- take-all: coalesced streaming copy ----
            for (unsigned long long i = (unsigned long long)lane; i < degree; i += 32) {
                row_out[i] = ld_cs_u32(neighbors + begin + i);
            }
            taken = (unsigned int)degree;
        } else {
            // ---- Algorithm R, fanout <= 32 ----
            // Seed the reservoir from the first fanout neighbors (streaming).
            unsigned int slot = 0;
            if (lane < fanout) {
                slot = ld_cs_u32(neighbors + begin + (unsigned long long)lane);
            }
            const unsigned int live = __ballot_sync(full, lane < fanout);

            // Prefetch + consume the remainder in 32-wide windows.
            for (unsigned long long base = (unsigned long long)fanout; base < degree; ) {
                const unsigned long long remain = degree - base;
                const unsigned long long nload = remain < 32ULL ? remain : 32ULL;
                if ((unsigned long long)lane < nload) {
                    wbuf[lane] = ld_cs_u32(neighbors + begin + base + (unsigned long long)lane);
                }
                __syncwarp();

                for (unsigned long long k = 0; k < nload; ++k) {
                    const unsigned long long t = base + k;
                    const unsigned int r =
                        aether_philox_u32(seed, layer, node, (unsigned int)t);
                    const unsigned int j = r % (unsigned int)(t + 1);
                    if (j < (unsigned int)fanout && ((live >> j) & 1u) && lane == (int)j) {
                        slot = wbuf[k];
                    }
                    __syncwarp();
                }
                base += nload;
            }

            if (lane < fanout) {
                row_out[lane] = slot;
            }
            taken = (unsigned int)fanout;
        }

        for (int i = (int)taken + lane; i < fanout; i += 32) {
            row_out[i] = AETHER_SAMPLE_PAD;
        }
        if (lane == 0) {
            counts[row] = taken;
        }
        // The next row reuses this warp's window.
        __syncwarp();
    }
}
