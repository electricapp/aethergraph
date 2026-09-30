use aether_stream::gpu::kernels::harness::cuda_or_skip;
use aether_stream::gpu::kernels::{DeviceCsr, SAMPLE_PAD, SampleBatch, SampleRequest, WarpSampler};
use aethergraph_core::reservoir_sample;

#[test]
fn sampler_matches_cpu_reservoir_oracle() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };

    // One node with degree 10; fanout 4 → Algorithm R.
    let offsets: Vec<u64> = vec![0, 10];
    let neighbors: Vec<u32> = (100..110).collect();
    let fanout = 4usize;
    let req = SampleRequest { seed: 7, layer: 1 };

    let expect = reservoir_sample(&neighbors, fanout, req.seed, req.layer, 0);

    let csr = DeviceCsr::upload(&stream, &offsets, &neighbors).unwrap();
    let mut d_seeds = stream.alloc_zeros::<u64>(1).unwrap();
    stream.memcpy_htod(&[0u64], &mut d_seeds).unwrap();
    let mut out = SampleBatch::new(&stream, 1, fanout).unwrap();

    let sampler = WarpSampler::new(&ctx, &stream).expect("nvrtc");
    sampler.sample(&csr, &d_seeds, &mut out, req).unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0u32; fanout];
    stream.memcpy_dtoh(out.neighbors(), &mut got).unwrap();
    assert_eq!(
        got, expect,
        "GPU reservoir must match CPU Philox Algorithm R"
    );
    let mut count = [0u32];
    stream.memcpy_dtoh(out.counts(), &mut count).unwrap();
    assert_eq!(count, [fanout as u32]);
}

/// Short rows report how much they hold and pad the rest, so neither a
/// low-degree seed, an isolated one, nor one outside the graph reads as
/// edges to node 0.
#[test]
fn sampler_counts_and_pads_short_rows() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Node 0: degree 2. Node 1: degree 0. Node 2: degree 6.
    let offsets: Vec<u64> = vec![0, 2, 2, 8];
    let neighbors: Vec<u32> = vec![5, 6, 10, 11, 12, 13, 14, 15];
    let fanout = 4usize;
    let seeds: Vec<u64> = vec![0, 1, 2, 99];

    let csr = DeviceCsr::upload(&stream, &offsets, &neighbors).unwrap();
    let mut d_seeds = stream.alloc_zeros::<u64>(seeds.len()).unwrap();
    stream.memcpy_htod(&seeds, &mut d_seeds).unwrap();
    let mut out = SampleBatch::new(&stream, seeds.len(), fanout).unwrap();
    let sampler = WarpSampler::new(&ctx, &stream).expect("nvrtc");
    sampler
        .sample(
            &csr,
            &d_seeds,
            &mut out,
            SampleRequest { seed: 3, layer: 0 },
        )
        .unwrap();
    stream.synchronize().unwrap();

    let mut counts = vec![0u32; seeds.len()];
    stream.memcpy_dtoh(out.counts(), &mut counts).unwrap();
    assert_eq!(counts, vec![2, 0, 4, 0]);
    let mut got = vec![0u32; seeds.len() * fanout];
    stream.memcpy_dtoh(out.neighbors(), &mut got).unwrap();
    assert_eq!(&got[0..4], &[5, 6, SAMPLE_PAD, SAMPLE_PAD]);
    assert_eq!(&got[4..8], &[SAMPLE_PAD; 4]);
    assert!(got[8..12].iter().all(|n| (10..16).contains(n)));
    assert_eq!(&got[12..16], &[SAMPLE_PAD; 4]);
}

#[test]
fn device_csr_rejects_malformed_offsets() {
    let Some((_ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let neighbors = [1u32, 2, 3];
    assert!(DeviceCsr::upload(&stream, &[], &neighbors).is_err());
    assert!(DeviceCsr::upload(&stream, &[1, 3], &neighbors).is_err());
    assert!(DeviceCsr::upload(&stream, &[0, 2, 1, 3], &neighbors).is_err());
    assert!(DeviceCsr::upload(&stream, &[0, 4], &neighbors).is_err());
    assert!(DeviceCsr::upload(&stream, &[0, 1, 3], &neighbors).is_ok());
}
