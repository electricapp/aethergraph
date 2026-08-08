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

    // Head version is at offset 0
    unsigned long long head1 = *(const unsigned long long*)(slot1);
    unsigned long long head2 = *(const unsigned long long*)(slot2);

    // Tail version is at offset 8 + feature_dim * 4, rounded to 8-byte align
    int feature_bytes = feature_dim * 4;
    int tail_offset = 8 + feature_bytes;
    tail_offset = (tail_offset + 7) & ~7;
    unsigned long long tail1 = *(const unsigned long long*)(slot1 + tail_offset);
    unsigned long long tail2 = *(const unsigned long long*)(slot2 + tail_offset);

    bool ok = head1 == tail1 && head1 == head2 && head1 == tail2
        && (head1 & 1) == 0 && head1 != 0;

    if (ok) {
        // Payload bytes must match between the snapshots. Word-wise compare
        // is exact: the payload is feature_dim 4-byte elements.
        const unsigned int* feat1 = (const unsigned int*)(slot1 + 8);
        const unsigned int* feat2 = (const unsigned int*)(slot2 + 8);
        for (int i = 0; i < feature_dim; i++) {
            if (feat1[i] != feat2[i]) {
                ok = false;
                break;
            }
        }
    }

    if (ok) {
        // Consistent across both snapshots — compact snapshot 1 to output.
        const float* feat = (const float*)(slot1 + 8);
        float* dst = output + ((long long)idx * feature_dim);
        for (int i = 0; i < feature_dim; i++) {
            dst[i] = feat[i];
        }
        retry_mask[idx] = 0;
    } else {
        // Torn, in-progress, or uninitialized — mark for retry.
        retry_mask[idx] = 1;
        atomicAdd(retry_count, 1);
    }
}
