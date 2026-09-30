//! Criterion benches for KERNELS.md Tier A (requires `--features gpudirect`).

#![cfg(all(target_os = "linux", feature = "gpudirect"))]

use aether_stream::gpu::kernels::harness::cuda_or_skip;
use aether_stream::gpu::kernels::validate::StagingRegions;
use aether_stream::gpu::kernels::{
    DeviceCsr, FeatureDequantizer, PersistentWork, PersistentWorkKind, PersistentWorker,
    QuantizedRowsDevice, SampleBatch, SampleRequest, SeqlockValidator, WarpSampler,
};
use aether_stream::rdma::layout::SlotGeometry;
use aethergraph_core::BlockScaledI8;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use cudarc::driver::DevicePtrMut;

fn validate_graph_replay(c: &mut Criterion) {
    let Some((ctx, _)) = cuda_or_skip() else {
        eprintln!("bench skip: no CUDA");
        return;
    };
    // Graph capture needs a real stream, not the legacy default.
    let stream = ctx.new_stream().unwrap();
    // 384 exercises the vector gather's unrolled in-flight group; 32 would
    // sit entirely in the scalar tail and measure the wrong path.
    const FEATURE_DIM: usize = 384;
    const SLOT_SIZE: usize = aethergraph_core::feature_slot_stride(FEATURE_DIM);
    const TAIL: usize = aethergraph_core::feature_slot_tail_offset(FEATURE_DIM);
    const BATCH: usize = 256;
    let mut host = vec![0u8; SLOT_SIZE * BATCH * 2];
    for slot in 0..BATCH * 2 {
        let base = slot * SLOT_SIZE;
        host[base..base + 8].copy_from_slice(&2u64.to_le_bytes());
        host[base + TAIL..base + TAIL + 8].copy_from_slice(&2u64.to_le_bytes());
    }
    let mut staging = stream.alloc_zeros::<u8>(host.len()).unwrap();
    stream.memcpy_htod(&host, &mut staging).unwrap();
    let s1 = {
        let (p, _) = staging.device_ptr_mut(&stream);
        p
    };
    let s2 = s1 + (SLOT_SIZE * BATCH) as u64;
    let geometry = SlotGeometry::new(FEATURE_DIM, SLOT_SIZE).unwrap();
    // SAFETY: `staging` holds both regions of BATCH slots and outlives the bench.
    let regions = unsafe { StagingRegions::new(s1, s2, BATCH, geometry) }.unwrap();
    let mut v = SeqlockValidator::new(&ctx, &stream, BATCH, &geometry).unwrap();
    // Capture once outside the bench.
    let _ = v.validate(&regions, BATCH).unwrap();
    c.bench_function("validate_graph_replay", |b| {
        b.iter(|| {
            let n = v.validate(&regions, BATCH).unwrap();
            std::hint::black_box(n);
        })
    });
}

fn sampler_reservoir(c: &mut Criterion) {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("bench skip: no CUDA");
        return;
    };
    let n_nodes = 1024usize;
    let degree = 64u64;
    let fanout = 16usize;
    let mut offsets = Vec::with_capacity(n_nodes + 1);
    let mut neighbors = Vec::new();
    offsets.push(0);
    for node in 0..n_nodes {
        for d in 0..degree {
            neighbors.push((node as u32) * 1000 + d as u32);
        }
        offsets.push(neighbors.len() as u64);
    }
    let nodes: Vec<u64> = (0..n_nodes as u64).collect();
    let csr = DeviceCsr::upload(&stream, &offsets, &neighbors).unwrap();
    let mut d_nodes = stream.alloc_zeros::<u64>(nodes.len()).unwrap();
    stream.memcpy_htod(&nodes, &mut d_nodes).unwrap();
    let mut out = SampleBatch::new(&stream, n_nodes, fanout).unwrap();
    let sampler = WarpSampler::new(&ctx, &stream).unwrap();
    c.bench_function("sampler_reservoir", |b| {
        b.iter(|| {
            sampler
                .sample(
                    &csr,
                    &d_nodes,
                    &mut out,
                    SampleRequest { seed: 1, layer: 0 },
                )
                .unwrap();
            stream.synchronize().unwrap();
        })
    });
}

fn persistent_drain(c: &mut Criterion) {
    let Some((ctx, _stream)) = cuda_or_skip() else {
        eprintln!("bench skip: no CUDA");
        return;
    };
    // The ring is single-shot, so each iteration needs a fresh worker, and
    // building one runs NVRTC — which dwarfs the drain unless it is setup.
    c.bench_function("persistent_drain_64", |b| {
        b.iter_batched(
            || PersistentWorker::new(&ctx, 128).unwrap(),
            |mut w| {
                w.start().unwrap();
                for i in 0..64u64 {
                    let _ = w.post(PersistentWork::new(PersistentWorkKind::Complete, i, 1));
                }
                let n = w.stop_and_join().unwrap();
                std::hint::black_box(n);
            },
            BatchSize::PerIteration,
        )
    });
}

/// Decode a batch-sized block of quantized rows. Against
/// `validate_graph_replay` the number to watch is bytes moved, not wall
/// clock: 1.125 B/feature here against 4 B/feature there.
fn feature_dequant(c: &mut Criterion) {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("bench skip: no CUDA");
        return;
    };
    const DIM: usize = 384;
    const ROWS: usize = 4096;
    let src: Vec<f32> = (0..ROWS * DIM).map(|i| (i % 251) as f32 - 125.0).collect();
    let enc = BlockScaledI8::encode_rows(&src, DIM).unwrap();
    let dev = QuantizedRowsDevice::upload(&stream, &enc).unwrap();
    let mut dq = FeatureDequantizer::new(&ctx, &stream, ROWS * DIM).unwrap();
    c.bench_function("feature_dequant_4096x384", |b| {
        b.iter(|| {
            dq.decode(&dev).unwrap();
            stream.synchronize().unwrap();
        })
    });
}

criterion_group!(
    kernels,
    validate_graph_replay,
    sampler_reservoir,
    persistent_drain,
    feature_dequant
);
criterion_main!(kernels);
