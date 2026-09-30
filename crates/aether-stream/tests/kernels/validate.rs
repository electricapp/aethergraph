use aether_stream::gpu::kernels::SeqlockValidator;
use aether_stream::gpu::kernels::harness::cuda_or_skip;
use aether_stream::gpu::kernels::validate::StagingRegions;
use aether_stream::rdma::layout::SlotGeometry;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut};
use std::sync::Arc;

const FEAT: usize = aethergraph_core::FEATURE_SLOT_HEAD_BYTES;

/// One slot's bytes at the production stride, so staging matches what the
/// gather's vector loads assume.
fn pack_slot(head: u64, features: &[f32], tail: u64) -> Vec<u8> {
    let dim = features.len();
    let mut out = vec![0u8; aethergraph_core::feature_slot_stride(dim)];
    out[0..8].copy_from_slice(&head.to_le_bytes());
    for (i, f) in features.iter().enumerate() {
        out[FEAT + i * 4..FEAT + (i + 1) * 4].copy_from_slice(&f.to_le_bytes());
    }
    let tail_at = aethergraph_core::feature_slot_tail_offset(dim);
    out[tail_at..tail_at + 8].copy_from_slice(&tail.to_le_bytes());
    out
}

fn row(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim).map(|i| (seed * dim + i) as f32 * 0.5).collect()
}

/// Stage `snap1` then `snap2` back to back in one allocation, returning both
/// base pointers. The returned slice keeps the allocation alive.
fn stage(
    stream: &Arc<CudaStream>,
    snap1: &[Vec<u8>],
    snap2: &[Vec<u8>],
) -> (CudaSlice<u8>, u64, u64) {
    let mut host: Vec<u8> = snap1.concat();
    let split = host.len();
    host.extend_from_slice(&snap2.concat());
    let mut dev = stream.alloc_zeros::<u8>(host.len()).expect("alloc");
    stream.memcpy_htod(&host, &mut dev).expect("H2D");
    let base = {
        let (p, _) = dev.device_ptr_mut(stream);
        p
    };
    (dev, base, base + split as u64)
}

/// Round-trip clean rows of `dim` features and check every value. Callers
/// pick `dim` to hit each of the gather's three shapes: whole in-flight
/// groups, a vector remainder, and the scalar-only path.
fn compacts_exactly(dim: usize) {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    const ROWS: usize = 6;
    let want: Vec<Vec<f32>> = (0..ROWS).map(|r| row(r, dim)).collect();
    let snap: Vec<Vec<u8>> = want.iter().map(|f| pack_slot(2, f, 2)).collect();
    let stride = aethergraph_core::feature_slot_stride(dim);

    let (_dev, p1, p2) = stage(&stream, &snap, &snap);
    let geometry = SlotGeometry::new(dim, stride).unwrap();
    let mut v = SeqlockValidator::new(&ctx, &stream, ROWS, &geometry).expect("nvrtc");
    // SAFETY: `_dev` holds both regions of ROWS slots at `stride`.
    let staging = unsafe { StagingRegions::new(p1, p2, ROWS, geometry) }.unwrap();
    assert_eq!(v.validate(&staging, ROWS).unwrap(), 0, "dim {dim}");

    stream.synchronize().unwrap();
    let mut got = vec![0f32; ROWS * dim];
    stream
        .memcpy_dtoh(&v.output().slice(0..ROWS * dim), &mut got)
        .unwrap();
    for r in 0..ROWS {
        assert_eq!(
            &got[r * dim..(r + 1) * dim],
            &want[r][..],
            "dim {dim} row {r}"
        );
    }
}

#[test]
fn compacts_whole_in_flight_groups() {
    // 384 = 3 rounds of 128, exactly one AETHER_GATHER_IN_FLIGHT group.
    compacts_exactly(384);
}

#[test]
fn compacts_vector_body_plus_remainder() {
    // 200 = one 128-element round then a 72-element scalar tail; 4 | 200 so
    // the destination row is still 16-byte aligned.
    compacts_exactly(200);
}

#[test]
fn compacts_scalar_only_row() {
    // 130 % 4 != 0, so output rows are not 16-aligned and the whole row
    // takes the scalar path.
    compacts_exactly(130);
}

#[test]
fn compacts_row_shorter_than_one_round() {
    compacts_exactly(4);
}

#[test]
fn flags_payload_mismatch_in_the_vector_body() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    const DIM: usize = 384;
    const ROWS: usize = 4;
    let stride = aethergraph_core::feature_slot_stride(DIM);

    let clean: Vec<Vec<f32>> = (0..ROWS).map(|r| row(r, DIM)).collect();
    let snap1: Vec<Vec<u8>> = clean.iter().map(|f| pack_slot(2, f, 2)).collect();
    // Row 1 differs in its last element, caught by the final in-flight group;
    // row 2 in its first, caught by the opening one.
    let snap2: Vec<Vec<u8>> = clean
        .iter()
        .enumerate()
        .map(|(r, f)| {
            let mut f = f.clone();
            match r {
                1 => *f.last_mut().unwrap() = -1.0,
                2 => f[0] = -1.0,
                _ => {}
            }
            pack_slot(2, &f, 2)
        })
        .collect();

    let (_dev, p1, p2) = stage(&stream, &snap1, &snap2);
    let geometry = SlotGeometry::new(DIM, stride).unwrap();
    let mut v = SeqlockValidator::new(&ctx, &stream, ROWS, &geometry).expect("nvrtc");
    // SAFETY: `_dev` holds both regions of ROWS slots at `stride`.
    let staging = unsafe { StagingRegions::new(p1, p2, ROWS, geometry) }.unwrap();
    assert_eq!(v.validate(&staging, ROWS).unwrap(), 2);
    assert_eq!(v.retry_indices(ROWS).unwrap(), vec![1, 2]);
}

#[test]
fn rejects_staging_that_breaks_the_vector_alignment_contract() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    const DIM: usize = 128;
    let snap = vec![pack_slot(2, &row(0, DIM), 2)];
    let (_dev, p1, p2) = stage(&stream, &snap, &snap);
    // A compact slot size is 8-aligned but not 16-aligned, so slot n's
    // payload would land off a vector boundary: no geometry admits it.
    let compact = aethergraph_core::feature_slot_size(DIM);
    assert_eq!(compact % 16, 8, "the case this contract exists for");
    assert!(SlotGeometry::new(DIM, compact).is_err());

    let geometry = SlotGeometry::packed(DIM).unwrap();
    // SAFETY: construction is refused before the pointer is ever used.
    assert!(unsafe { StagingRegions::new(p1 + 8, p2, 1, geometry) }.is_err());

    // Staging laid out for another dim, or shorter than the batch, is
    // refused at the launch site rather than read out of bounds.
    let mut v = SeqlockValidator::new(&ctx, &stream, 4, &geometry).expect("nvrtc");
    let other = SlotGeometry::packed(DIM + 4).unwrap();
    // SAFETY: `_dev` holds one slot per region; neither staging is launched.
    let wrong_dim = unsafe { StagingRegions::new(p1, p2, 1, other) }.unwrap();
    assert!(v.validate(&wrong_dim, 1).is_err());
    // SAFETY: as above.
    let one_row = unsafe { StagingRegions::new(p1, p2, 1, geometry) }.unwrap();
    assert!(v.validate(&one_row, 2).is_err());
}

#[test]
fn validate_graph_replay_skips_without_cuda() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    const DIM: usize = 4;
    const BATCH: usize = 4;
    let stride = aethergraph_core::feature_slot_stride(DIM);
    let snap = vec![pack_slot(2, &[1.0, 2.0, 3.0, 4.0], 2); BATCH];
    let (_dev, p1, p2) = stage(&stream, &snap, &snap);
    let geometry = SlotGeometry::new(DIM, stride).unwrap();
    let mut v = SeqlockValidator::new(&ctx, &stream, BATCH, &geometry).expect("nvrtc");
    // SAFETY: `_dev` holds both regions of BATCH slots at `stride`.
    let staging = unsafe { StagingRegions::new(p1, p2, BATCH, geometry) }.unwrap();
    assert_eq!(v.validate(&staging, BATCH).unwrap(), 0);
    // Capture is best-effort; a stack that refuses it falls back to eager
    // launch and must still validate.
    assert_eq!(v.validate(&staging, BATCH).unwrap(), 0);
}
