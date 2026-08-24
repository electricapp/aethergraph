use aether_stream::gpu::kernels::TmaAggregator;
use aether_stream::gpu::kernels::harness::{cuda_or_skip, require_sm};

#[test]
fn dense_tile_accumulate_smoke() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Hopper ISA path needs SM90+; the scalar baseline runs everywhere.
    if require_sm(&ctx, 90) {
        eprintln!("SM90+ present — baseline still used until TMA builders land");
    }
    let rows = 4u32;
    let cols = 8u32;
    let a: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..cols).map(|i| 1.0 + i as f32).collect();
    let mut expect = vec![0f32; rows as usize];
    for r in 0..rows as usize {
        for k in 0..cols as usize {
            expect[r] += a[r * cols as usize + k] * b[k];
        }
    }
    let mut d_a = stream.alloc_zeros::<f32>(a.len()).unwrap();
    let mut d_b = stream.alloc_zeros::<f32>(b.len()).unwrap();
    let mut d_out = stream.alloc_zeros::<f32>(rows as usize).unwrap();
    stream.memcpy_htod(&a, &mut d_a).unwrap();
    stream.memcpy_htod(&b, &mut d_b).unwrap();
    let agg = TmaAggregator::new(&ctx, &stream).unwrap();
    agg.accumulate(&d_a, &d_b, &mut d_out, rows, cols).unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0f32; rows as usize];
    stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    for (g, e) in got.iter().zip(expect.iter()) {
        assert!((g - e).abs() < 1e-4, "{g} vs {e}");
    }
}
