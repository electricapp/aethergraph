use aether_stream::gpu::kernels::WarpSampler;
use aether_stream::gpu::kernels::harness::cuda_or_skip;
use aethergraph_core::reservoir_sample;
use cudarc::driver::DevicePtr;

#[test]
fn sampler_matches_cpu_reservoir_oracle() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };

    // One node with degree 10; fanout 4 → Algorithm R.
    let offsets: Vec<u64> = vec![0, 10];
    let neighbors: Vec<u32> = (100..110).collect();
    let nodes: Vec<u64> = vec![0];
    let fanout = 4usize;
    let seed = 7u64;
    let layer = 1u32;

    let expect = reservoir_sample(&neighbors, fanout, seed, layer, 0);

    let mut d_off = stream.alloc_zeros::<u64>(offsets.len()).unwrap();
    let mut d_nbr = stream.alloc_zeros::<u32>(neighbors.len()).unwrap();
    let mut d_nodes = stream.alloc_zeros::<u64>(nodes.len()).unwrap();
    let mut d_out = stream.alloc_zeros::<u32>(fanout).unwrap();
    stream.memcpy_htod(&offsets, &mut d_off).unwrap();
    stream.memcpy_htod(&neighbors, &mut d_nbr).unwrap();
    stream.memcpy_htod(&nodes, &mut d_nodes).unwrap();

    let sampler = WarpSampler::new(&ctx, &stream).expect("nvrtc");
    sampler
        .sample(&d_off, &d_nbr, &d_nodes, &mut d_out, 1, fanout, seed, layer)
        .unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0u32; fanout];
    stream.memcpy_dtoh(&d_out, &mut got).unwrap();
    assert_eq!(
        got, expect,
        "GPU reservoir must match CPU Philox Algorithm R"
    );
}
