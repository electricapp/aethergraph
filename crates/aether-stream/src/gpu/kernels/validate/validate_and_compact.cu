// Two-snapshot seqlock validation and feature compaction kernel.
//
// The client issues each slot READ twice, sequentially, into two staging
// regions (snapshot 2 is posted only after snapshot 1's completions were
// observed). PCIe completions within one READ can land in any order, so a
// single snapshot's head/tail pair can be stale relative to its payload
// bytes; cross-checking two fully-ordered snapshots is what makes
// acceptance sound (see the RDMA reader contract in feature_table.rs).
//
// A row is accepted iff:
//   - snap1.head == snap1.tail == snap2.head == snap2.tail,
//   - the common version is even and nonzero, and
//   - the payload bytes of the two snapshots are identical.
// Accepted rows are compacted from snapshot 1 into the output tensor;
// everything else is flagged for retry (both snapshots re-read).
//
// K5.6: payload traffic is once-touched — ld.global.cs (and v4) so version
// working set is not evicted by feature bytes.

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const char* slot1 = staging1 + ((long long)idx * slot_size);
    const char* slot2 = staging2 + ((long long)idx * slot_size);

    // Version words stay ordinary loads (ordering matters for the seqlock).
    unsigned long long head1 = *(const unsigned long long*)(slot1);
    unsigned long long head2 = *(const unsigned long long*)(slot2);

    int feature_bytes = feature_dim * 4;
    int tail_offset = 8 + feature_bytes;
    tail_offset = (tail_offset + 7) & ~7;
    unsigned long long tail1 = *(const unsigned long long*)(slot1 + tail_offset);
    unsigned long long tail2 = *(const unsigned long long*)(slot2 + tail_offset);

    bool ok = head1 == tail1 && head1 == head2 && head1 == tail2
        && (head1 & 1) == 0 && head1 != 0;

    if (ok) {
        const unsigned int* feat1 = (const unsigned int*)(slot1 + 8);
        const unsigned int* feat2 = (const unsigned int*)(slot2 + 8);
        int i = 0;
        for (; i + 3 < feature_dim; i += 4) {
            const uint4 a = ld_cs_v4u32(feat1 + i);
            const uint4 b = ld_cs_v4u32(feat2 + i);
            if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) {
                ok = false;
                break;
            }
        }
        for (; ok && i < feature_dim; ++i) {
            if (ld_cs_u32(feat1 + i) != ld_cs_u32(feat2 + i)) {
                ok = false;
            }
        }
    }

    if (ok) {
        const float* feat = (const float*)(slot1 + 8);
        float* dst = output + ((long long)idx * feature_dim);
        int i = 0;
        for (; i + 3 < feature_dim; i += 4) {
            const float4 v = ld_cs_v4f32(feat + i);
            dst[i] = v.x;
            dst[i + 1] = v.y;
            dst[i + 2] = v.z;
            dst[i + 3] = v.w;
        }
        for (; i < feature_dim; ++i) {
            dst[i] = ld_cs_f32(feat + i);
        }
        retry_mask[idx] = 0;
    } else {
        retry_mask[idx] = 1;
        atomicAdd(retry_count, 1);
    }
}
