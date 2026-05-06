// Seqlock validation and feature compaction kernel.
//
// After RDMA READs land in the VRAM staging buffer, this kernel
// validates head == tail (no torn read) and compacts valid features
// into a contiguous output tensor for downstream GNN consumption.

extern "C" __global__ void validate_and_compact(
    const char* staging,
    float* output,
    int* retry_mask,
    int* retry_count,
    int feature_dim,
    int batch_size,
    int slot_size,
    unsigned long long epoch_version
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const char* slot = staging + ((long long)idx * slot_size);

    // Head version is at offset 0
    unsigned long long head = *(const unsigned long long*)(slot);

    // Tail version is at offset 8 + feature_dim * 4, rounded to 8-byte align
    int feature_bytes = feature_dim * 4;
    int tail_offset = 8 + feature_bytes;
    tail_offset = (tail_offset + 7) & ~7;
    unsigned long long tail = *(const unsigned long long*)(slot + tail_offset);

    // epoch_version == 0 means no epoch pinning (accept any consistent read).
    // epoch_version > 0 means MVCC: only accept features written before this epoch.
    if (head == tail && (head & 1) == 0 && head != 0
        && (epoch_version == 0 || head <= epoch_version)) {
        // Consistent read — compact features to output tensor
        const float* feat = (const float*)(slot + 8);
        float* dst = output + ((long long)idx * feature_dim);
        for (int i = 0; i < feature_dim; i++) {
            dst[i] = feat[i];
        }
        retry_mask[idx] = 0;
    } else {
        // Torn read or uninitialized — mark for retry
        retry_mask[idx] = 1;
        atomicAdd(retry_count, 1);
    }
}
