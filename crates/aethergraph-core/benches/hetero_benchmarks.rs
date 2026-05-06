//! Benchmarks for heterogeneous graph sampling.
//!
//! Default: finishes in ~10 seconds. Set BENCH_SCALE=large for heavier runs.
//! ```
//! cargo bench -p aethergraph-core --bench hetero_benchmarks
//! BENCH_SCALE=large cargo bench -p aethergraph-core --bench hetero_benchmarks
//! ```

use aethergraph_core::graph::hetero::HeteroGraph;
use aethergraph_core::loader::hetero_sampler::{
    HeteroNeighborSampler, HeteroSamplingConfig,
};
use aethergraph_core::{Graph, NodeId};
use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Graph generators
// ---------------------------------------------------------------------------

fn is_large() -> bool {
    std::env::var("BENCH_SCALE").ok().as_deref() == Some("large")
}

/// Build a Reddit-shaped heterogeneous graph.
/// Small: 10K users, 50K posts, 200K comments, 1K subreddits (~1.5M edges)
/// Large: 100K users, 500K posts, 2M comments, 10K subreddits (~15M edges)
fn reddit_graph() -> HeteroGraph {
    let (users, posts, comments, subs) = if is_large() {
        (100_000, 500_000, 2_000_000, 10_000)
    } else {
        (10_000, 50_000, 200_000, 1_000)
    };

    let mut rng = StdRng::seed_from_u64(42);

    let make_edges = |rng: &mut StdRng, count: usize, max_src: usize, max_dst: usize| -> (Vec<NodeId>, Vec<NodeId>) {
        let src: Vec<NodeId> = (0..count).map(|_| rng.random_range(0..max_src as NodeId)).collect();
        let dst: Vec<NodeId> = (0..count).map(|_| rng.random_range(0..max_dst as NodeId)).collect();
        (src, dst)
    };

    let votes_count = users * 20; // avg 20 votes per user
    let writes_count = users * 30;
    let replies_count = comments / 2;
    let belongs_count = posts;

    let (v_src, v_dst) = make_edges(&mut rng, votes_count, users, posts);
    let (w_src, w_dst) = make_edges(&mut rng, writes_count, users, comments);
    let (r_src, r_dst) = make_edges(&mut rng, replies_count, comments, comments);
    let (b_src, b_dst) = make_edges(&mut rng, belongs_count, posts, subs);

    let build_csr = |src: &[NodeId], dst: &[NodeId], max_src: usize, max_dst: usize| -> Graph {
        let num_nodes = max_src.max(max_dst);
        let edges: Vec<(NodeId, NodeId)> = src.iter().zip(dst.iter()).map(|(&s, &d)| (s, d)).collect();
        Graph::from_edges(num_nodes, &edges, None).unwrap()
    };

    let votes_csr = build_csr(&v_src, &v_dst, users, posts);
    let writes_csr = build_csr(&w_src, &w_dst, users, comments);
    let replies_csr = build_csr(&r_src, &r_dst, comments, comments);
    let belongs_csr = build_csr(&b_src, &b_dst, posts, subs);

    HeteroGraph::from_parts(
        vec![
            ("user".into(), users),
            ("post".into(), posts),
            ("comment".into(), comments),
            ("subreddit".into(), subs),
        ],
        vec![
            ("user".into(), "votes".into(), "post".into(), votes_csr),
            ("user".into(), "writes".into(), "comment".into(), writes_csr),
            ("comment".into(), "reply_to".into(), "comment".into(), replies_csr),
            ("post".into(), "belongs_to".into(), "subreddit".into(), belongs_csr),
        ],
    )
}

fn make_config(graph: &HeteroGraph, fanout: &[usize]) -> HeteroSamplingConfig {
    let mut per_type: Vec<Vec<usize>> = Vec::new();
    for _ in 0..graph.edge_type_count() {
        per_type.push(fanout.to_vec());
    }
    HeteroSamplingConfig {
        fanout: per_type,
        replace: false,
        seed: Some(42),
        max_degree: Some(10_000),
        num_hops: fanout.len(),
    }
}

fn random_seeds(count: usize, max: usize) -> Vec<NodeId> {
    let mut rng = StdRng::seed_from_u64(99);
    (0..count).map(|_| rng.random_range(0..max as NodeId)).collect()
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_hetero_sample_1hop(c: &mut Criterion) {
    let graph = reddit_graph();
    let config = make_config(&graph, &[15]);
    let user_type = graph.node_type_id("user").unwrap();
    let users = graph.num_nodes(user_type);

    let mut group = c.benchmark_group("hetero_sample_1hop");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));

    for batch_size in [32, 128, 512] {
        let seeds = random_seeds(batch_size, users);
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &seeds,
            |b, seeds| {
                let mut sampler = HeteroNeighborSampler::new(&graph, config.clone());
                b.iter(|| {
                    black_box(sampler.sample_neighbors(user_type, seeds));
                });
            },
        );
    }
    group.finish();
}

fn bench_hetero_sample_2hop(c: &mut Criterion) {
    let graph = reddit_graph();
    let config = make_config(&graph, &[15, 10]);
    let user_type = graph.node_type_id("user").unwrap();
    let users = graph.num_nodes(user_type);

    let mut group = c.benchmark_group("hetero_sample_2hop");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));

    for batch_size in [32, 128, 512] {
        let seeds = random_seeds(batch_size, users);
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &seeds,
            |b, seeds| {
                let mut sampler = HeteroNeighborSampler::new(&graph, config.clone());
                b.iter(|| {
                    black_box(sampler.sample_neighbors(user_type, seeds));
                });
            },
        );
    }
    group.finish();
}

fn bench_hetero_sample_3hop(c: &mut Criterion) {
    let graph = reddit_graph();
    let config = make_config(&graph, &[15, 10, 5]);
    let user_type = graph.node_type_id("user").unwrap();
    let users = graph.num_nodes(user_type);

    let mut group = c.benchmark_group("hetero_sample_3hop");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));

    for batch_size in [32, 128] {
        let seeds = random_seeds(batch_size, users);
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &seeds,
            |b, seeds| {
                let mut sampler = HeteroNeighborSampler::new(&graph, config.clone());
                b.iter(|| {
                    black_box(sampler.sample_neighbors(user_type, seeds));
                });
            },
        );
    }
    group.finish();
}

fn bench_hetero_access_edges(c: &mut Criterion) {
    let graph = reddit_graph();
    let config = make_config(&graph, &[15, 10]);
    let user_type = graph.node_type_id("user").unwrap();
    let seeds = random_seeds(128, graph.num_nodes(user_type));
    let mut sampler = HeteroNeighborSampler::new(&graph, config);
    let sub = sampler.sample_neighbors(user_type, &seeds);

    let mut group = c.benchmark_group("hetero_access_edges");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));

    // Edge access is now O(1) — just reading pre-computed local indices
    for eid in 0..graph.edge_type_count() as u8 {
        let meta = graph.edge_type_meta(eid);
        let label = format!("{}_{}", meta.relation, eid);
        group.bench_function(&label, |b| {
            b.iter(|| {
                let et = eid as usize;
                black_box((&sub.edge_src_local[et], &sub.edge_dst_local[et]));
            });
        });
    }
    group.finish();
}

fn bench_hetero_graph_construction(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let num_nodes = 10_000usize;
    let num_edges = 100_000usize;

    let src: Vec<NodeId> = (0..num_edges).map(|_| rng.random_range(0..num_nodes as NodeId)).collect();
    let dst: Vec<NodeId> = (0..num_edges).map(|_| rng.random_range(0..num_nodes as NodeId)).collect();

    let mut group = c.benchmark_group("hetero_construction");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.throughput(Throughput::Elements(num_edges as u64));

    group.bench_function("single_edge_type_10k", |b| {
        b.iter(|| {
            let edges: Vec<(NodeId, NodeId)> = src.iter().zip(dst.iter()).map(|(&s, &d)| (s, d)).collect();
            let csr = Graph::from_edges(num_nodes, &edges, None).unwrap();
            let g = HeteroGraph::from_parts(
                vec![("a".into(), num_nodes), ("b".into(), num_nodes)],
                vec![("a".into(), "rel".into(), "b".into(), csr)],
            );
            black_box(g);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_hetero_sample_1hop,
    bench_hetero_sample_2hop,
    bench_hetero_sample_3hop,
    bench_hetero_access_edges,
    bench_hetero_graph_construction,
);
criterion_main!(benches);
