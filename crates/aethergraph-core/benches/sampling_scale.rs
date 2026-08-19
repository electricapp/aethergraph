//! Neighbor sampling on a graph that does not fit in cache.
//!
//! The other sampling benchmarks build a 10k-node graph: 80 KB of offsets and
//! 800 KB of edges, which sits entirely in L2. There, every `offsets[node]`
//! probe hits, so software prefetch has nothing to hide and a locality
//! regression costs nothing measurable. Production graphs are three to four
//! orders of magnitude larger and every frontier probe is a random miss.
//!
//! This builds a CSR whose offsets array alone (32 MB) exceeds the last-level
//! cache of the machines this runs on, so the frontier walk is bound by memory
//! latency and TLB reach — the regime the prefetch distance is tuned for.
//! Graph construction dominates the runtime of this file; the graph is built
//! once and shared by every benchmark in it.

use aethergraph_core::{Graph, NeighborSampler, NodeId, SamplingConfig};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use std::time::Duration;

/// 4M nodes x avg degree 12 => 32 MB of offsets, ~192 MB of edges.
const NUM_NODES: usize = 4_000_000;
const AVG_DEGREE: usize = 12;
/// Seeds per sampled batch, in the range a GNN trainer actually uses.
const BATCH: usize = 1024;

/// Skewed out-degrees (a tenth of the nodes carry 10x the average) over
/// uniformly random destinations, so frontier probes land unpredictably.
fn build_graph() -> Graph {
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let mut src: Vec<NodeId> = Vec::with_capacity(NUM_NODES * AVG_DEGREE);
    let mut dst: Vec<NodeId> = Vec::with_capacity(NUM_NODES * AVG_DEGREE);

    for u in 0..NUM_NODES as NodeId {
        let degree = if rng.random::<f32>() < 0.1 {
            AVG_DEGREE * 10
        } else {
            rng.random_range(1..=AVG_DEGREE)
        };
        for _ in 0..degree {
            src.push(u);
            dst.push(rng.random_range(0..NUM_NODES as NodeId));
        }
    }

    Graph::from_src_dst(NUM_NODES, &src, &dst, None).expect("csr build")
}

fn seed_batch(rng: &mut StdRng) -> Vec<NodeId> {
    (0..BATCH)
        .map(|_| rng.random_range(0..NUM_NODES as NodeId))
        .collect()
}

/// Random `neighbor_range` probes with no sampling work attached — the raw
/// cost of one frontier lookup once the offsets array has left the cache.
fn bench_cold_probe(c: &mut Criterion, graph: &Graph) {
    let mut rng = StdRng::seed_from_u64(0xB33F);
    let nodes: Vec<NodeId> = (0..BATCH * 16)
        .map(|_| rng.random_range(0..NUM_NODES as NodeId))
        .collect();

    let mut group = c.benchmark_group("scale_probe");
    group.sample_size(30);
    group.throughput(Throughput::Elements(nodes.len() as u64));
    group.bench_function("neighbor_range_random", |b| {
        let csr = graph.csr_view();
        b.iter(|| {
            let mut acc = 0usize;
            for &n in &nodes {
                acc += csr.neighbor_range(black_box(n)).len();
            }
            acc
        });
    });
    group.finish();
}

/// Multi-hop sampling at batch sizes a trainer issues. Fanouts are the
/// GraphSAGE and PinSAGE shapes; the two-hop case is the one that grows the
/// frontier past any cache.
fn bench_scale_sampling(c: &mut Criterion, graph: &Graph) {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let seeds = seed_batch(&mut rng);

    let mut group = c.benchmark_group("scale_sampling");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(12));

    for fanout in [vec![25], vec![15, 10], vec![10, 10, 5]] {
        let label = fanout
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join("x");
        let config = SamplingConfig {
            fanout: fanout.clone(),
            replace: false,
            seed: Some(42),
            ..Default::default()
        };
        group.bench_with_input(
            BenchmarkId::new("multi_hop", &label),
            &config,
            |b, config| {
                let mut sampler = NeighborSampler::new(graph, config.clone());
                b.iter(|| black_box(sampler.sample_neighbors(black_box(&seeds))));
            },
        );
    }

    group.finish();
}

/// A batch small enough that the sample touches a negligible fraction of the
/// graph. This is the case a dense per-node dedup table is meant to lose: it
/// reserves eight bytes for every node in the graph to track a few hundred.
fn bench_sparse_batch(c: &mut Criterion, graph: &Graph) {
    let mut rng = StdRng::seed_from_u64(0xD00D);
    let seeds: Vec<NodeId> = (0..16)
        .map(|_| rng.random_range(0..NUM_NODES as NodeId))
        .collect();

    let config = SamplingConfig {
        fanout: vec![10],
        replace: false,
        seed: Some(42),
        ..Default::default()
    };

    let mut group = c.benchmark_group("scale_sparse");
    group.sample_size(30);
    group.bench_function("sixteen_seeds_one_hop", |b| {
        let mut sampler = NeighborSampler::new(graph, config.clone());
        b.iter(|| black_box(sampler.sample_neighbors(black_box(&seeds))));
    });
    group.finish();
}

fn bench_all(c: &mut Criterion) {
    let graph = build_graph();
    bench_cold_probe(c, &graph);
    bench_scale_sampling(c, &graph);
    bench_sparse_batch(c, &graph);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
