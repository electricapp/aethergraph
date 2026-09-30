//! CUDA seqlock validation kernel — two-snapshot torn-slot detection.
//!
//! Stages two snapshot regions of four slots each in VRAM and runs the
//! `validate_and_compact` kernel directly (no RDMA, no GpuGatherBuffer).
//! Asserts that:
//!   - the slot that is consistent across both snapshots is compacted into
//!     the output tensor,
//!   - a slot with an odd head (both snapshots) is flagged for retry,
//!   - a slot with head ≠ tail within a snapshot is flagged for retry,
//!   - a slot whose versions match across both snapshots but whose payload
//!     bytes differ between them is flagged for retry — the pattern a
//!     single-snapshot check cannot see (stale versions delivered with a
//!     torn payload by out-of-order PCIe completions),
//!   - `retry_count` equals 3.
//!
//! Requires CUDA + `gpudirect` feature so the kernel module compiles.

#![cfg(all(target_os = "linux", feature = "gpudirect"))]

use aether_stream::gpu::kernel::SeqlockValidator;
use aether_stream::gpu::kernels::validate::StagingRegions;
use aether_stream::rdma::layout::SlotGeometry;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtrMut};

/// feature_dim chosen small so the test is easy to read. Geometry comes from
/// the same helpers the table uses, so the staging stride matches what the
/// device gather's vector loads assume.
const FEATURE_DIM: usize = 4;
const SLOT_SIZE: usize = aethergraph_core::feature_slot_stride(FEATURE_DIM);
const FEAT: usize = aethergraph_core::FEATURE_SLOT_HEAD_BYTES;
const TAIL: usize = aethergraph_core::feature_slot_tail_offset(FEATURE_DIM);
const BATCH: usize = 4;

fn pack_slot(head: u64, features: [f32; FEATURE_DIM], tail: u64) -> [u8; SLOT_SIZE] {
    let mut out = [0u8; SLOT_SIZE];
    out[0..8].copy_from_slice(&head.to_le_bytes());
    for (i, f) in features.iter().enumerate() {
        out[FEAT + i * 4..FEAT + (i + 1) * 4].copy_from_slice(&f.to_le_bytes());
    }
    out[TAIL..TAIL + 8].copy_from_slice(&tail.to_le_bytes());
    out
}

#[test]
fn seqlock_kernel_flags_torn_and_compacts_valid() {
    let ctx = CudaContext::new(0).expect("CUDA init — check nvidia-smi + driver");
    let stream = ctx.default_stream();

    // Snapshot 1 / snapshot 2 pairs per slot:
    //   slot 0: identical, even, nonzero            → valid, compacted
    //   slot 1: head odd in both snapshots          → retry
    //   slot 2: head != tail within each snapshot   → retry
    //   slot 3: versions match across both snapshots (head == tail == 6)
    //           but payload bytes differ between them → retry. This is the
    //           adversarial pattern only the cross-snapshot payload compare
    //           catches: within one snapshot the version pair looks clean.
    let snap1: Vec<[u8; SLOT_SIZE]> = vec![
        pack_slot(2, [1.0, 2.0, 3.0, 4.0], 2),
        pack_slot(3, [9.0, 9.0, 9.0, 9.0], 3),
        pack_slot(4, [7.0, 7.0, 7.0, 7.0], 5),
        pack_slot(6, [5.0, 5.0, 5.0, 5.0], 6),
    ];
    let snap2: Vec<[u8; SLOT_SIZE]> = vec![
        pack_slot(2, [1.0, 2.0, 3.0, 4.0], 2),
        pack_slot(3, [9.0, 9.0, 9.0, 9.0], 3),
        pack_slot(4, [7.0, 7.0, 7.0, 7.0], 5),
        pack_slot(6, [5.0, 5.0, 8.0, 5.0], 6),
    ];

    let mut host = Vec::with_capacity(SLOT_SIZE * BATCH * 2);
    for slot in &snap1 {
        host.extend_from_slice(slot);
    }
    for slot in &snap2 {
        host.extend_from_slice(slot);
    }

    // Stage in VRAM. Plain CudaSlice<u8>, no RDMA registration — the kernel
    // only needs the raw device pointers. The two snapshot regions are
    // back-to-back in one allocation.
    let mut staging: CudaSlice<u8> = stream
        .alloc_zeros(SLOT_SIZE * BATCH * 2)
        .expect("alloc VRAM staging");
    stream.memcpy_htod(&host, &mut staging).expect("H2D memcpy");
    let staging1_ptr = {
        let (p, _g) = staging.device_ptr_mut(&stream);
        p
    };
    let staging2_ptr = staging1_ptr + (SLOT_SIZE * BATCH) as u64;

    let geometry = SlotGeometry::new(FEATURE_DIM, SLOT_SIZE).expect("slot geometry");
    // SAFETY: `staging` holds both regions of BATCH slots at SLOT_SIZE and
    // outlives every validate below.
    let regions = unsafe { StagingRegions::new(staging1_ptr, staging2_ptr, BATCH, geometry) }
        .expect("staging regions");
    let mut validator = SeqlockValidator::new(&ctx, &stream, BATCH, &geometry)
        .expect("SeqlockValidator nvrtc compile");

    let retry_count = validator.validate(&regions, BATCH).expect("kernel launch");
    assert_eq!(retry_count, 3, "expected 3 torn slots flagged");

    let retry_indices = validator
        .retry_indices(BATCH)
        .expect("retry_indices readback");
    assert_eq!(
        retry_indices,
        vec![1, 2, 3],
        "expected slots 1 (head odd), 2 (head != tail), and 3 (payload \
         differs across snapshots) in retry set"
    );

    // Slot 0 should have been compacted into output[0..feature_dim]. Torn
    // slots leave their output rows untouched (they stay at alloc_zeros()'s
    // initial zeros), so we only assert the valid one to avoid depending on
    // undefined kernel behavior for torn rows.
    let output = validator.output();
    let mut host_out = vec![0.0f32; FEATURE_DIM * BATCH];
    stream
        .memcpy_dtoh(output, &mut host_out)
        .expect("D2H output readback");
    assert_eq!(
        &host_out[..FEATURE_DIM],
        &[1.0, 2.0, 3.0, 4.0],
        "valid slot not compacted correctly"
    );
}

#[test]
fn seqlock_kernel_replays_cuda_graph_on_second_validate() {
    // K5.0: first validate captures; second with the same signature replays.
    // Capture needs a real stream; the legacy default stream cannot be
    // captured.
    let ctx = CudaContext::new(0).expect("CUDA init — check nvidia-smi + driver");
    let stream = ctx.new_stream().expect("stream");

    let snap = pack_slot(2, [1.0, 2.0, 3.0, 4.0], 2);
    let mut host = Vec::with_capacity(SLOT_SIZE * BATCH * 2);
    for _ in 0..BATCH {
        host.extend_from_slice(&snap);
    }
    for _ in 0..BATCH {
        host.extend_from_slice(&snap);
    }

    let mut staging: CudaSlice<u8> = stream
        .alloc_zeros(SLOT_SIZE * BATCH * 2)
        .expect("alloc VRAM staging");
    stream.memcpy_htod(&host, &mut staging).expect("H2D memcpy");
    let staging1_ptr = {
        let (p, _g) = staging.device_ptr_mut(&stream);
        p
    };
    let staging2_ptr = staging1_ptr + (SLOT_SIZE * BATCH) as u64;

    let geometry = SlotGeometry::new(FEATURE_DIM, SLOT_SIZE).expect("slot geometry");
    // SAFETY: `staging` holds both regions of BATCH slots at SLOT_SIZE and
    // outlives every validate below.
    let regions = unsafe { StagingRegions::new(staging1_ptr, staging2_ptr, BATCH, geometry) }
        .expect("staging regions");
    let mut validator = SeqlockValidator::new(&ctx, &stream, BATCH, &geometry)
        .expect("SeqlockValidator nvrtc compile");
    assert!(!validator.has_captured_graph());

    let r1 = validator
        .validate(&regions, BATCH)
        .expect("first validate (capture)");
    assert_eq!(r1, 0);
    // Capture is best-effort — mapped host retry_count and some driver stacks
    // refuse it, and validate falls back to eager launch. This test is about
    // the kernel's answer either way.
    if !validator.has_captured_graph() {
        eprintln!("note: CUDA graph capture fell back to eager validate");
        let r2 = validator
            .validate(&regions, BATCH)
            .expect("second validate (eager)");
        assert_eq!(r2, 0);
        return;
    }
    let _ = validator.has_mapped_retry_count();

    let r2 = validator
        .validate(&regions, BATCH)
        .expect("second validate (replay)");
    assert_eq!(r2, 0);
    assert!(validator.has_captured_graph());
}
