use aether_stream::gpu::kernels::TmaAggregator;
use aether_stream::gpu::kernels::harness::{cuda_or_skip, require_sm};

/// Run one tile against a host GEMV. Values are small integers so every
/// partial sum is exact in f32 and reduction order cannot matter.
fn accumulate_matches_host(rows: u32, cols: u32) {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Hopper ISA path needs SM90+; the baseline runs everywhere.
    if require_sm(&ctx, 90) {
        eprintln!("SM90+ present — baseline still used until TMA builders land");
    }
    let (r, c) = (rows as usize, cols as usize);
    let a: Vec<f32> = (0..r * c).map(|i| (i % 7) as f32).collect();
    let b: Vec<f32> = (0..c).map(|i| 1.0 + (i % 5) as f32).collect();
    let mut expect = vec![1f32; r];
    for (row, e) in expect.iter_mut().enumerate() {
        for k in 0..c {
            *e += a[row * c + k] * b[k];
        }
    }
    let mut d_a = stream.alloc_zeros::<f32>(a.len()).unwrap();
    let mut d_b = stream.alloc_zeros::<f32>(b.len()).unwrap();
    let mut d_out = stream.alloc_zeros::<f32>(r).unwrap();
    stream.memcpy_htod(&a, &mut d_a).unwrap();
    stream.memcpy_htod(&b, &mut d_b).unwrap();
    stream.memcpy_htod(&vec![1f32; r], &mut d_out).unwrap();
    let agg = TmaAggregator::new(&ctx, &stream).unwrap();
    agg.accumulate(&d_a, &d_b, &mut d_out, rows, cols).unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f32; r];
    stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    assert_eq!(got, expect, "rows {rows} cols {cols}");
}

#[test]
fn dense_tile_accumulate_smoke() {
    accumulate_matches_host(4, 8);
}

/// Column counts that are not a multiple of 4 leave row bases off 16-byte
/// alignment, which the vector path must not touch.
#[test]
fn dense_tile_accumulate_unaligned_rows() {
    accumulate_matches_host(9, 5);
    accumulate_matches_host(33, 602);
    accumulate_matches_host(17, 1433);
}

/// A full vector round plus a scalar tail, and more rows than the grid has
/// warps, so the grid-stride loop runs.
#[test]
fn dense_tile_accumulate_vector_and_tail() {
    accumulate_matches_host(40_000, 132);
}

#[test]
fn dense_tile_accumulate_rejects_short_buffers() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let d_a = stream.alloc_zeros::<f32>(15).unwrap();
    let d_b = stream.alloc_zeros::<f32>(4).unwrap();
    let mut d_out = stream.alloc_zeros::<f32>(4).unwrap();
    let agg = TmaAggregator::new(&ctx, &stream).unwrap();
    assert!(agg.accumulate(&d_a, &d_b, &mut d_out, 4, 4).is_err());
}
