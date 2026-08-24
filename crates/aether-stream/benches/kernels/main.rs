//! Criterion benches for KERNELS.md Tier A (requires `--features gpudirect`).

#![cfg(all(target_os = "linux", feature = "gpudirect"))]

use aether_stream::gpu::kernels::harness::cuda_or_skip;
use aether_stream::gpu::kernels::{
    PersistentWork, PersistentWorkKind, PersistentWorker, SeqlockValidator, WarpSampler,
};
use criterion::{Criterion, criterion_group, criterion_main};
use cudarc::driver::DevicePtrMut;

fn validate_graph_replay(c: &mut Criterion) {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("bench skip: no CUDA");
        return;
    };
    const FEATURE_DIM: usize = 32;
    const SLOT_SIZE: usize = 8 + FEATURE_DIM * 4 + 8;
    const BATCH: usize = 256;
    let mut host = vec![0u8; SLOT_SIZE * BATCH * 2];
    for slot in 0..BATCH * 2 {
        let base = slot * SLOT_SIZE;
        host[base..base + 8].copy_from_slice(&2u64.to_le_bytes());
        host[base + SLOT_SIZE - 8..base + SLOT_SIZE].copy_from_slice(&2u64.to_le_bytes());
    }
    let mut staging = stream.alloc_zeros::<u8>(host.len()).unwrap();
    stream.memcpy_htod(&host, &mut staging).unwrap();
    let s1 = {
        let (p, _) = staging.device_ptr_mut(&stream);
        p
    };
    let s2 = s1 + (SLOT_SIZE * BATCH) as u64;
    let mut v = SeqlockValidator::new(&ctx, &stream, BATCH, FEATURE_DIM).unwrap();
    // Capture once outside the bench.
    let _ = v.validate(s1, s2, SLOT_SIZE, BATCH).unwrap();
    c.bench_function("validate_graph_replay", |b| {
        b.iter(|| {
            let n = v.validate(s1, s2, SLOT_SIZE, BATCH).unwrap();
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
    let mut d_off = stream.alloc_zeros::<u64>(offsets.len()).unwrap();
    let mut d_nbr = stream.alloc_zeros::<u32>(neighbors.len()).unwrap();
    let mut d_nodes = stream.alloc_zeros::<u64>(nodes.len()).unwrap();
    let mut d_out = stream.alloc_zeros::<u32>(n_nodes * fanout).unwrap();
    stream.memcpy_htod(&offsets, &mut d_off).unwrap();
    stream.memcpy_htod(&neighbors, &mut d_nbr).unwrap();
    stream.memcpy_htod(&nodes, &mut d_nodes).unwrap();
    let sampler = WarpSampler::new(&ctx, &stream).unwrap();
    c.bench_function("sampler_reservoir", |b| {
        b.iter(|| {
            sampler
                .sample(&d_off, &d_nbr, &d_nodes, &mut d_out, n_nodes, fanout, 1, 0)
                .unwrap();
            stream.synchronize().unwrap();
        })
    });
}

fn persistent_drain(c: &mut Criterion) {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("bench skip: no CUDA");
        return;
    };
    c.bench_function("persistent_drain_64", |b| {
        b.iter(|| {
            let mut w = PersistentWorker::new(&ctx, &stream, 128).unwrap();
            w.start().unwrap();
            for i in 0..64u64 {
                let _ = w.post(PersistentWork::new(PersistentWorkKind::Complete, i, 1));
            }
            let n = w.stop_and_join().unwrap();
            std::hint::black_box(n);
        })
    });
}

criterion_group!(
    kernels,
    validate_graph_replay,
    sampler_reservoir,
    persistent_drain
);
criterion_main!(kernels);
