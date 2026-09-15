use aether_stream::gpu::kernels::harness::cuda_or_skip;
use aether_stream::gpu::kernels::{FeatureDequantizer, QuantizedRowsDevice};
use aethergraph_core::BlockScaledI8;

fn rows(count: usize, dim: usize) -> Vec<f32> {
    // Mixed magnitudes across blocks so a per-row scale would visibly lose
    // the small ones, and the per-block scale has something to prove.
    (0..count * dim)
        .map(|i| {
            let block = (i % dim) / BlockScaledI8::BLOCK;
            let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
            sign * ((i % 61) as f32 + 1.0) * 10f32.powi(block as i32 % 4 - 2)
        })
        .collect()
}

/// Device decode must equal the CPU oracle bit for bit — both are one IEEE
/// multiply per element, so anything but equality means the layout or the
/// sign extension is wrong.
fn device_matches_cpu(count: usize, dim: usize) {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let src = rows(count, dim);
    let enc = BlockScaledI8::encode_rows(&src, dim).expect("encode");
    let expect = enc.decode();

    let dev = QuantizedRowsDevice::upload(&stream, &enc).expect("upload");
    let mut dq = FeatureDequantizer::new(&ctx, &stream, count * dim).expect("nvrtc");
    dq.decode(&dev).expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; count * dim];
    stream
        .memcpy_dtoh(&dq.output().slice(0..count * dim), &mut got)
        .expect("D2H");
    assert_eq!(got.len(), expect.len());
    for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "dim {dim} element {i}: {g} vs {e}"
        );
    }
}

#[test]
fn dequant_matches_cpu_on_a_whole_vector_round() {
    // 512 codes = one full 32-lane x 16-code round.
    device_matches_cpu(4, 512);
}

#[test]
fn dequant_matches_cpu_with_a_scalar_remainder() {
    // 16 | 544, so the vector body runs one round and 32 codes fall to the
    // scalar tail.
    device_matches_cpu(3, 544);
}

#[test]
fn dequant_matches_cpu_on_the_scalar_only_path() {
    // 100 % 16 != 0: no vector body at all, and the trailing block is
    // partial (100 = 3 blocks of 32 plus 4).
    device_matches_cpu(5, 100);
}

#[test]
fn dequant_matches_cpu_when_rows_outnumber_warps() {
    // Forces the grid-stride loop rather than one row per warp.
    device_matches_cpu(40_000, 128);
}

#[test]
fn dequant_reproduces_zero_blocks() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    const DIM: usize = 128;
    let mut src = vec![0f32; 2 * DIM];
    src[0] = 4.0; // only the first block of the first row is nonzero
    let enc = BlockScaledI8::encode_rows(&src, DIM).expect("encode");
    let dev = QuantizedRowsDevice::upload(&stream, &enc).expect("upload");
    let mut dq = FeatureDequantizer::new(&ctx, &stream, src.len()).expect("nvrtc");
    dq.decode(&dev).expect("launch");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; src.len()];
    stream
        .memcpy_dtoh(&dq.output().slice(0..src.len()), &mut got)
        .expect("D2H");
    assert_eq!(got, enc.decode());
    assert!(got[1..].iter().all(|v| *v == 0.0));
}
