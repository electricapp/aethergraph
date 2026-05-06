//! Benchmarks for aether-graph C-tree dynamic graph.

use aether_graph::DynamicGraph;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;

fn build_graph(num_vertices: usize, num_edges: usize) -> DynamicGraph {
    let mut rng = SmallRng::seed_from_u64(42);
    let arena_bytes = num_edges * 128;
    let g = DynamicGraph::new(num_vertices, arena_bytes);
    for _ in 0..num_edges {
        let src = rng.random_range(0..num_vertices as u32);
        let dst = rng.random_range(0..num_vertices as u32);
        let _ = g.insert_edge(src, dst);
    }
    g
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_edge");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));

    for n in [10_000, 100_000] {
        let arena_bytes = 1 << 28;
        let g = DynamicGraph::new(n, arena_bytes);
        let mut rng = SmallRng::seed_from_u64(99);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let src = rng.random_range(0..n as u32);
                let dst = rng.random_range(0..n as u32);
                let _ = black_box(g.insert_edge(src, dst));
            });
        });
    }
    group.finish();
}

fn bench_neighbor_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors_into");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));

    for (n, e) in [(10_000, 100_000), (100_000, 1_000_000)] {
        let g = build_graph(n, e);
        let mut rng = SmallRng::seed_from_u64(99);
        let mut buf = Vec::with_capacity(256);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("nodes", n), &n, |b, _| {
            b.iter(|| {
                let v = rng.random_range(0..n as u32);
                g.neighbors_into(v, &mut buf);
                black_box(buf.len());
            });
        });
    }
    group.finish();
}

fn bench_has_edge(c: &mut Criterion) {
    let mut group = c.benchmark_group("has_edge");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));

    let g = build_graph(100_000, 1_000_000);
    let mut rng = SmallRng::seed_from_u64(99);

    group.throughput(Throughput::Elements(1));
    group.bench_function("100k_1m", |b| {
        b.iter(|| {
            let src = rng.random_range(0..100_000u32);
            let dst = rng.random_range(0..100_000u32);
            black_box(g.has_edge(src, dst));
        });
    });
    group.finish();
}

fn bench_degree(c: &mut Criterion) {
    let mut group = c.benchmark_group("degree");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));

    let g = build_graph(100_000, 1_000_000);
    let mut rng = SmallRng::seed_from_u64(99);

    group.throughput(Throughput::Elements(1));
    group.bench_function("100k_1m", |b| {
        b.iter(|| {
            let v = rng.random_range(0..100_000u32);
            black_box(g.degree(v));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_neighbor_read,
    bench_has_edge,
    bench_degree
);
criterion_main!(benches);
