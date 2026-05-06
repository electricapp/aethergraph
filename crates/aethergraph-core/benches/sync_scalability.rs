use aethergraph_core::{Graph, NodeId};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;

fn generate_graph(num_nodes: usize, avg_degree: usize) -> Graph {
    let mut rng = StdRng::seed_from_u64(42);
    let mut edges = Vec::new();

    for src in 0..num_nodes as NodeId {
        let degree = if rng.random::<f64>() < 0.1 {
            avg_degree * 10 // 10% hubs
        } else {
            rng.random_range(1..=avg_degree * 2)
        };

        for _ in 0..degree {
            let dst = rng.random_range(0..num_nodes as NodeId);
            if src != dst {
                edges.push((src, dst));
            }
        }
    }

    Graph::from_edges(num_nodes, &edges, None).unwrap()
}

/// Test neighbor access scales linearly with graph size
fn bench_graph_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_size_scaling");

    for &num_nodes in &[10_000, 100_000, 1_000_000] {
        let graph = generate_graph(num_nodes, 20);
        let mut rng = StdRng::seed_from_u64(42);

        // Sample 1000 random nodes
        let nodes: Vec<NodeId> = (0..1000)
            .map(|_| rng.random_range(0..num_nodes as NodeId))
            .collect();

        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(
            BenchmarkId::new("neighbors", num_nodes),
            &num_nodes,
            |b, _| {
                b.iter(|| {
                    for &node in &nodes {
                        black_box(graph.neighbors(node));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Test batch size scaling
fn bench_batch_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size_scaling");
    let graph = generate_graph(100_000, 20);
    let mut rng = StdRng::seed_from_u64(42);

    for &batch_size in &[10, 100, 1_000, 10_000] {
        let nodes: Vec<NodeId> = (0..batch_size)
            .map(|_| rng.random_range(0..100_000))
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("lookups", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    for &node in &nodes {
                        black_box(graph.neighbors(node));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Test degree() calls (just offset arithmetic)
fn bench_degree_queries(c: &mut Criterion) {
    let graph = generate_graph(1_000_000, 20);
    let mut rng = StdRng::seed_from_u64(42);
    let nodes: Vec<NodeId> = (0..10_000)
        .map(|_| rng.random_range(0..1_000_000))
        .collect();

    c.bench_function("degree_10k_nodes", |b| {
        b.iter(|| {
            for &node in &nodes {
                black_box(graph.degree(node));
            }
        });
    });
}

/// Simulate real GNN sampling pattern: random access to many nodes
fn bench_gnn_sampling_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("gnn_sampling_pattern");

    for &num_nodes in &[100_000, 1_000_000] {
        let graph = generate_graph(num_nodes, 20);
        let mut rng = StdRng::seed_from_u64(42);

        // Simulate: 64 seed nodes, sample 25 neighbors each, then 10 from each of those
        let seeds: Vec<NodeId> = (0..64)
            .map(|_| rng.random_range(0..num_nodes as NodeId))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("2hop_sampling", num_nodes),
            &num_nodes,
            |b, _| {
                b.iter(|| {
                    // Hop 1: Get neighbors of seeds
                    let mut hop1_neighbors = Vec::new();
                    for &seed in &seeds {
                        let neighbors = graph.neighbors(seed);
                        hop1_neighbors.extend(neighbors.iter().take(25).copied());
                    }

                    // Hop 2: Get neighbors of hop1 neighbors
                    let mut total = 0;
                    for &node in &hop1_neighbors {
                        let neighbors = graph.neighbors(node);
                        total += neighbors.len().min(10);
                    }
                    black_box(total);
                });
            },
        );
    }

    group.finish();
}

/// Test sequential vs random access patterns
fn bench_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("access_patterns");
    let graph = generate_graph(100_000, 20);

    // Sequential access
    group.bench_function("sequential_1k", |b| {
        b.iter(|| {
            for node in 0..1000 {
                black_box(graph.neighbors(node));
            }
        });
    });

    // Random access
    let mut rng = StdRng::seed_from_u64(42);
    let random_nodes: Vec<NodeId> = (0..1000).map(|_| rng.random_range(0..100_000)).collect();

    group.bench_function("random_1k", |b| {
        b.iter(|| {
            for &node in &random_nodes {
                black_box(graph.neighbors(node));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_graph_size_scaling,
    bench_batch_size_scaling,
    bench_degree_queries,
    bench_gnn_sampling_pattern,
    bench_access_patterns,
);
criterion_main!(benches);
