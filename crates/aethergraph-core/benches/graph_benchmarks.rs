use aethergraph_core::{
    Graph, NeighborSampler, NodeId, ParallelBatchSampler, SamplingConfig, TemporalStrategy,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;

// Generate a synthetic graph for benchmarking
fn generate_power_law_graph(num_nodes: usize, avg_degree: usize) -> Graph {
    let mut rng = StdRng::seed_from_u64(42);
    let mut edges = Vec::new();

    // Generate edges with power-law-like degree distribution
    for src in 0..num_nodes as NodeId {
        let degree = if rng.random::<f64>() < 0.1 {
            // 10% of nodes are high-degree hubs
            avg_degree * 10
        } else {
            // 90% have average or below-average degree
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

/// Generate a graph with edge weights for weighted sampling benchmarks.
fn generate_weighted_graph(num_nodes: usize, avg_degree: usize) -> Graph {
    let mut rng = StdRng::seed_from_u64(42);
    let mut edges = Vec::new();
    let mut weights = Vec::new();

    for src in 0..num_nodes as NodeId {
        let degree = rng.random_range(1..=avg_degree * 2);
        for _ in 0..degree {
            let dst = rng.random_range(0..num_nodes as NodeId);
            if src != dst {
                edges.push((src, dst));
                weights.push(rng.random::<f32>() * 10.0);
            }
        }
    }

    Graph::from_edges(num_nodes, &edges, Some(&weights)).unwrap()
}

/// Generate a graph with edge timestamps for temporal sampling benchmarks.
fn generate_timestamped_graph(num_nodes: usize, avg_degree: usize) -> Graph {
    let mut rng = StdRng::seed_from_u64(42);
    let mut edges = Vec::new();

    for src in 0..num_nodes as NodeId {
        let degree = rng.random_range(1..=avg_degree * 2);
        for _ in 0..degree {
            let dst = rng.random_range(0..num_nodes as NodeId);
            if src != dst {
                edges.push((src, dst));
            }
        }
    }

    let mut graph = Graph::from_edges(num_nodes, &edges, None).unwrap();
    let timestamps: Vec<f64> = (0..graph.num_edges())
        .map(|i| (i as f64) * 0.001) // monotonically increasing timestamps
        .collect();
    graph.set_timestamps(timestamps);
    graph
}

fn bench_csr_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_construction");

    for num_nodes in [1_000, 10_000, 100_000].iter() {
        let num_edges = num_nodes * 10;

        // Generate random edges
        let mut rng = StdRng::seed_from_u64(42);
        let edges: Vec<(NodeId, NodeId)> = (0..num_edges)
            .map(|_| {
                let src = rng.random_range(0..*num_nodes as NodeId);
                let dst = rng.random_range(0..*num_nodes as NodeId);
                (src, dst)
            })
            .collect();

        group.throughput(Throughput::Elements(num_edges as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_nodes),
            num_nodes,
            |b, &num_nodes| {
                b.iter(|| Graph::from_edges(num_nodes, black_box(&edges), None).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_neighbor_access(c: &mut Criterion) {
    let graph = generate_power_law_graph(10_000, 20);
    let mut rng = StdRng::seed_from_u64(42);

    let nodes: Vec<NodeId> = (0..1000).map(|_| rng.random_range(0..10_000)).collect();

    c.bench_function("neighbor_access_random_1k", |b| {
        b.iter(|| {
            for &node in &nodes {
                black_box(graph.neighbors(node));
            }
        });
    });
}

fn bench_degree_queries(c: &mut Criterion) {
    let graph = generate_power_law_graph(10_000, 20);

    c.bench_function("degree_query_all_nodes", |b| {
        b.iter(|| {
            for node in 0..10_000 {
                black_box(graph.degree(node));
            }
        });
    });
}

fn bench_neighbor_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_sampling");
    let graph = generate_power_law_graph(10_000, 20);

    // Single-hop sampling
    for &fanout in &[5, 10, 25] {
        let config = SamplingConfig {
            fanout: vec![fanout],
            replace: false,
            seed: Some(42),
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::new("single_hop", fanout), &fanout, |b, _| {
            let mut sampler = NeighborSampler::new(&graph, config.clone());
            let seeds = vec![0, 100, 200, 300, 400]; // 5 seed nodes

            b.iter(|| black_box(sampler.sample_neighbors(black_box(&seeds))));
        });
    }

    // Multi-hop sampling (GraphSAGE-style)
    let config = SamplingConfig {
        fanout: vec![25, 10],
        replace: false,
        seed: Some(42),
        ..Default::default()
    };

    group.bench_function("two_hop_graphsage", |b| {
        let mut sampler = NeighborSampler::new(&graph, config.clone());
        let seeds = vec![0, 100, 200, 300, 400];

        b.iter(|| black_box(sampler.sample_neighbors(black_box(&seeds))));
    });

    group.finish();
}

fn bench_parallel_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_sampling");
    let graph = generate_power_law_graph(10_000, 20);

    let config = SamplingConfig {
        fanout: vec![25, 10],
        replace: false,
        seed: Some(42),
        ..Default::default()
    };

    // Generate multiple batches
    let mut rng = StdRng::seed_from_u64(42);
    let batches: Vec<Vec<NodeId>> = (0..100)
        .map(|_| (0..32).map(|_| rng.random_range(0..10_000)).collect())
        .collect();

    group.throughput(Throughput::Elements((batches.len() * 32) as u64));
    group.bench_function("parallel_100_batches", |b| {
        let sampler = ParallelBatchSampler::new(&graph, config.clone());

        b.iter(|| black_box(sampler.sample_batches(black_box(&batches))));
    });

    group.finish();
}

fn bench_sampling_with_vs_without_replacement(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_strategy");
    let graph = generate_power_law_graph(10_000, 20);
    let seeds = vec![0, 100, 200, 300, 400];

    // With replacement
    let config_with = SamplingConfig {
        fanout: vec![25],
        replace: true,
        seed: Some(42),
        ..Default::default()
    };

    group.bench_function("with_replacement", |b| {
        let mut sampler = NeighborSampler::new(&graph, config_with.clone());
        b.iter(|| black_box(sampler.sample_neighbors(black_box(&seeds))));
    });

    // Without replacement
    let config_without = SamplingConfig {
        fanout: vec![25],
        replace: false,
        seed: Some(42),
        ..Default::default()
    };

    group.bench_function("without_replacement", |b| {
        let mut sampler = NeighborSampler::new(&graph, config_without.clone());
        b.iter(|| black_box(sampler.sample_neighbors(black_box(&seeds))));
    });

    group.finish();
}

fn bench_weighted_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_sampling");
    let graph = generate_weighted_graph(10_000, 20);
    let seeds: Vec<NodeId> = vec![0, 100, 200, 300, 400];

    // Baseline: unweighted
    let config_unweighted = SamplingConfig {
        fanout: vec![25, 10],
        replace: false,
        seed: Some(42),
        weighted: false,
        ..Default::default()
    };
    group.bench_function("unweighted_baseline", |b| {
        let mut sampler = NeighborSampler::new(&graph, config_unweighted.clone());
        b.iter(|| black_box(sampler.sample_neighbors(black_box(&seeds))));
    });

    // Weighted without replacement (Efraimidis-Spirakis)
    let config_weighted = SamplingConfig {
        fanout: vec![25, 10],
        replace: false,
        seed: Some(42),
        weighted: true,
        ..Default::default()
    };
    group.bench_function("weighted_no_replace", |b| {
        let mut sampler = NeighborSampler::new(&graph, config_weighted.clone());
        b.iter(|| black_box(sampler.sample_neighbors(black_box(&seeds))));
    });

    // Weighted with replacement (CDF binary search)
    let config_weighted_replace = SamplingConfig {
        fanout: vec![25, 10],
        replace: true,
        seed: Some(42),
        weighted: true,
        ..Default::default()
    };
    group.bench_function("weighted_replace", |b| {
        let mut sampler = NeighborSampler::new(&graph, config_weighted_replace.clone());
        b.iter(|| black_box(sampler.sample_neighbors(black_box(&seeds))));
    });

    group.finish();
}

fn bench_temporal_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_sampling");
    let graph = generate_timestamped_graph(10_000, 20);
    let seeds: Vec<NodeId> = vec![0, 100, 200, 300, 400];
    // Seed times: allow ~50% of edges through the temporal filter
    let mid_time = (graph.num_edges() as f64) * 0.001 * 0.5;
    let input_times: Vec<f64> = seeds.iter().map(|_| mid_time).collect();

    // Baseline: non-temporal
    let config_baseline = SamplingConfig {
        fanout: vec![25, 10],
        replace: false,
        seed: Some(42),
        ..Default::default()
    };
    group.bench_function("non_temporal_baseline", |b| {
        let mut sampler = NeighborSampler::new(&graph, config_baseline.clone());
        b.iter(|| black_box(sampler.sample_neighbors(black_box(&seeds))));
    });

    // Temporal uniform
    let config_uniform = SamplingConfig {
        fanout: vec![25, 10],
        replace: false,
        seed: Some(42),
        temporal_strategy: Some(TemporalStrategy::Uniform),
        ..Default::default()
    };
    group.bench_function("temporal_uniform", |b| {
        let mut sampler = NeighborSampler::new(&graph, config_uniform.clone());
        b.iter(|| {
            black_box(sampler.sample_neighbors_temporal(black_box(&seeds), black_box(&input_times)))
        });
    });

    // Temporal last
    let config_last = SamplingConfig {
        fanout: vec![25, 10],
        replace: false,
        seed: Some(42),
        temporal_strategy: Some(TemporalStrategy::Last),
        ..Default::default()
    };
    group.bench_function("temporal_last", |b| {
        let mut sampler = NeighborSampler::new(&graph, config_last.clone());
        b.iter(|| {
            black_box(sampler.sample_neighbors_temporal(black_box(&seeds), black_box(&input_times)))
        });
    });

    group.finish();
}

fn bench_disjoint_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("disjoint_sampling");
    let graph = generate_power_law_graph(10_000, 20);

    for &batch_size in &[5, 32, 128] {
        let mut rng = StdRng::seed_from_u64(42);
        let seeds: Vec<NodeId> = (0..batch_size)
            .map(|_| rng.random_range(0..10_000))
            .collect();

        // Baseline: shared dedup (normal)
        let config = SamplingConfig {
            fanout: vec![15, 10],
            replace: false,
            seed: Some(42),
            ..Default::default()
        };
        group.bench_with_input(
            BenchmarkId::new("shared", batch_size),
            &batch_size,
            |b, _| {
                let mut sampler = NeighborSampler::new(&graph, config.clone());
                b.iter(|| black_box(sampler.sample_neighbors(black_box(&seeds))));
            },
        );

        // Disjoint: per-seed isolation
        group.bench_with_input(
            BenchmarkId::new("disjoint", batch_size),
            &batch_size,
            |b, _| {
                let mut sampler = NeighborSampler::new(&graph, config.clone());
                b.iter(|| black_box(sampler.sample_neighbors_disjoint(black_box(&seeds), None)));
            },
        );
    }

    group.finish();
}

fn bench_rabbit_reorder(c: &mut Criterion) {
    let mut group = c.benchmark_group("rabbit_reorder");

    for &num_nodes in &[10_000, 100_000] {
        let graph = generate_power_law_graph(num_nodes, 20);

        group.throughput(Throughput::Elements(graph.num_edges() as u64));
        group.bench_with_input(
            BenchmarkId::new("compute_perm", num_nodes),
            &num_nodes,
            |b, _| {
                b.iter(|| black_box(graph.reorder_rabbit()));
            },
        );

        let perm = graph.reorder_rabbit();
        group.bench_with_input(
            BenchmarkId::new("apply_perm", num_nodes),
            &num_nodes,
            |b, _| {
                b.iter(|| black_box(graph.permute(black_box(&perm)).unwrap()));
            },
        );
    }

    group.finish();
}

fn bench_sampling_reordered(c: &mut Criterion) {
    let mut group = c.benchmark_group("reorder_sampling_speedup");
    let graph = generate_power_law_graph(100_000, 20);
    let perm = graph.reorder_rabbit();
    let reordered = graph.permute(&perm).unwrap();

    let config = SamplingConfig {
        fanout: vec![25, 10],
        replace: false,
        seed: Some(42),
        ..Default::default()
    };

    let mut rng = StdRng::seed_from_u64(99);
    let seeds: Vec<NodeId> = (0..512)
        .map(|_| rng.random_range(0..100_000))
        .collect();

    // Remap seeds for reordered graph
    let inv_perm: Vec<u32> = {
        let mut inv = vec![0u32; perm.len()];
        for (new_id, &old_id) in perm.iter().enumerate() {
            inv[old_id as usize] = new_id as u32;
        }
        inv
    };
    let reordered_seeds: Vec<NodeId> = seeds.iter().map(|&s| inv_perm[s as usize]).collect();

    group.bench_function("original", |b| {
        let mut sampler = NeighborSampler::new(&graph, config.clone());
        b.iter(|| black_box(sampler.sample_neighbors(black_box(&seeds))));
    });

    group.bench_function("rabbit_reordered", |b| {
        let mut sampler = NeighborSampler::new(&reordered, config.clone());
        b.iter(|| black_box(sampler.sample_neighbors(black_box(&reordered_seeds))));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_csr_construction,
    bench_neighbor_access,
    bench_degree_queries,
    bench_neighbor_sampling,
    bench_parallel_sampling,
    bench_sampling_with_vs_without_replacement,
    bench_weighted_sampling,
    bench_temporal_sampling,
    bench_disjoint_sampling,
    bench_rabbit_reorder,
    bench_sampling_reordered,
);
criterion_main!(benches);
