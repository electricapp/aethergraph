//! TEST_PLAN T2.1 — CUDA seqlock validation kernel.
//!
//! Stages three slots in VRAM and runs the `validate_and_compact` kernel
//! directly (no RDMA, no GpuGatherBuffer). Asserts that:
//!   - the valid slot is compacted into the output tensor,
//!   - the two torn slots (head odd, head ≠ tail) are flagged for retry,
//!   - `retry_count` equals 2.
//!
//! Requires CUDA + `gpudirect` feature so the kernel module compiles. Run on
//! a GPU box:
//!
//!     cargo test --release -p aether-stream --features gpudirect \
//!         --test t2_1_cuda_seqlock -- --nocapture

#![cfg(all(target_os = "linux", feature = "gpudirect"))]

use aether_stream::gpu::kernel::SeqlockValidator;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtrMut};

/// feature_dim chosen small so the test is easy to read; the kernel's layout
/// assumes `slot_size = 8 + feature_dim*4 (8-aligned) + 8`.
const FEATURE_DIM: usize = 4;
const SLOT_SIZE: usize = 8 + FEATURE_DIM * 4 + 8; // head + features + tail = 32

fn pack_slot(head: u64, features: [f32; FEATURE_DIM], tail: u64) -> [u8; SLOT_SIZE] {
    let mut out = [0u8; SLOT_SIZE];
    out[0..8].copy_from_slice(&head.to_le_bytes());
    for (i, f) in features.iter().enumerate() {
        out[8 + i * 4..8 + (i + 1) * 4].copy_from_slice(&f.to_le_bytes());
    }
    out[SLOT_SIZE - 8..SLOT_SIZE].copy_from_slice(&tail.to_le_bytes());
    out
}

#[test]
fn seqlock_kernel_flags_torn_and_compacts_valid() {
    let ctx = CudaContext::new(0).expect("CUDA init — check nvidia-smi + driver");
    let stream = ctx.default_stream();

    // Craft three slots: one valid, two torn for different reasons.
    let slot_valid = pack_slot(2, [1.0, 2.0, 3.0, 4.0], 2);
    let slot_head_odd = pack_slot(3, [9.0, 9.0, 9.0, 9.0], 3);
    let slot_head_ne_tail = pack_slot(4, [7.0, 7.0, 7.0, 7.0], 5);

    let mut host = Vec::with_capacity(SLOT_SIZE * 3);
    host.extend_from_slice(&slot_valid);
    host.extend_from_slice(&slot_head_odd);
    host.extend_from_slice(&slot_head_ne_tail);

    // Stage in VRAM. Plain CudaSlice<u8>, no RDMA registration — the kernel
    // only needs the raw device pointer.
    let mut staging: CudaSlice<u8> = stream
        .alloc_zeros(SLOT_SIZE * 3)
        .expect("alloc VRAM staging");
    stream.memcpy_htod(&host, &mut staging).expect("H2D memcpy");
    let staging_ptr = {
        let (p, _g) = staging.device_ptr_mut(&stream);
        p as u64
    };

    let mut validator = SeqlockValidator::new(&ctx, &stream, 3, FEATURE_DIM)
        .expect("SeqlockValidator nvrtc compile");

    // epoch_version = 0 → accept any consistent read.
    let retry_count = validator
        .validate(staging_ptr, SLOT_SIZE, 3, 0)
        .expect("kernel launch");
    assert_eq!(retry_count, 2, "expected 2 torn slots flagged");

    let retry_indices = validator.retry_indices(3).expect("retry_indices readback");
    assert_eq!(
        retry_indices,
        vec![1, 2],
        "expected slots 1 (head odd) and 2 (head != tail) in retry set"
    );

    // Slot 0 should have been compacted into output[0..feature_dim]. Torn
    // slots leave their output rows untouched (they stay at alloc_zeros()'s
    // initial zeros), so we only assert the valid one to avoid depending on
    // undefined kernel behavior for torn rows.
    let output = validator.output();
    let mut host_out = vec![0.0f32; FEATURE_DIM * 3];
    stream
        .memcpy_dtoh(output, &mut host_out)
        .expect("D2H output readback");
    assert_eq!(
        &host_out[..FEATURE_DIM],
        &[1.0, 2.0, 3.0, 4.0],
        "valid slot not compacted correctly"
    );
}
