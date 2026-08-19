//! Random-row feature gather: the memory-bound half of a training step.
//!
//! Sweeps feature dimension because the row size decides how many cache
//! lines each gathered row spans, and therefore how far ahead the gather
//! may prefetch before it exceeds the core's outstanding-miss budget.

use aethergraph_core::{FeatureStore, NodeId, SyncFeatureStore, save_features};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use tempfile::TempDir;

/// Payload bytes per store, held constant across feature dimensions so the
/// sweep isolates row size rather than also varying how much memory the
/// gather walks. Far past LLC, so every row is a genuine miss.
const PAYLOAD_BYTES: usize = 128 << 20;
const BATCH: usize = 4096;

fn num_nodes(feature_dim: usize) -> usize {
    PAYLOAD_BYTES / (feature_dim * 4)
}

fn store_path(dir: &TempDir, feature_dim: usize) -> std::path::PathBuf {
    let path = dir.path().join(format!("feat_{feature_dim}.bin"));
    if !path.exists() {
        let nodes = num_nodes(feature_dim);
        let mut rng = StdRng::seed_from_u64(0xFEA7);
        let features: Vec<f32> = (0..nodes * feature_dim)
            .map(|_| rng.random::<f32>())
            .collect();
        save_features(&path, features, nodes, feature_dim).unwrap();
    }
    path
}

fn build_store(dir: &TempDir, feature_dim: usize) -> FeatureStore {
    FeatureStore::load(store_path(dir, feature_dim)).unwrap()
}

fn random_nodes(count: usize, num_nodes: usize) -> Vec<NodeId> {
    let mut rng = StdRng::seed_from_u64(0xB47C);
    (0..count)
        .map(|_| rng.random_range(0..num_nodes as NodeId))
        .collect()
}

fn bench_gather(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();

    let mut group = c.benchmark_group("feature_gather");
    for feature_dim in [32usize, 128, 512, 1024] {
        let store = build_store(&dir, feature_dim);
        let nodes = random_nodes(BATCH, num_nodes(feature_dim));
        let row_bytes = feature_dim * 4;
        group.throughput(Throughput::Bytes((BATCH * row_bytes) as u64));

        group.bench_with_input(
            BenchmarkId::new("get_batch", feature_dim),
            &feature_dim,
            |b, _| b.iter(|| black_box(store.get_batch(black_box(&nodes)).unwrap())),
        );

        let mut out = vec![0f32; BATCH * feature_dim];
        group.bench_with_input(
            BenchmarkId::new("get_batch_into", feature_dim),
            &feature_dim,
            |b, _| {
                b.iter(|| {
                    store
                        .get_batch_into(black_box(&nodes), black_box(&mut out))
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

/// The positional-read path taken wherever io_uring is unavailable, so the
/// per-row decode sits behind one `pread` per row.
fn bench_sync_store(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();

    let mut group = c.benchmark_group("sync_store_gather");
    // One row per pread makes this syscall-heavy; a smaller batch keeps the
    // benchmark's wall clock reasonable without changing what it measures.
    let batch = 512;
    for feature_dim in [128usize, 1024] {
        let mut store = SyncFeatureStore::load(store_path(&dir, feature_dim)).unwrap();
        let nodes = random_nodes(batch, num_nodes(feature_dim));
        group.throughput(Throughput::Bytes((batch * feature_dim * 4) as u64));
        group.bench_with_input(
            BenchmarkId::new("get_batch", feature_dim),
            &feature_dim,
            |b, _| b.iter(|| black_box(store.get_batch(black_box(&nodes)).unwrap())),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_gather, bench_sync_store);
criterion_main!(benches);
