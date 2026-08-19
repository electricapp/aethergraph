//! What node ordering is worth on the memory-bound sampling path.
//!
//! `sampling_scale` establishes that frontier probes miss cache on a graph
//! this size. It cannot show what a better layout would buy, because its
//! destinations are drawn uniformly at random: a graph with no community
//! structure has no locality to recover, so every ordering is equally bad.
//!
//! This builds an R-MAT graph instead, which has both the degree skew and the
//! recursive community structure of a real one, and compares two orderings of
//! it. The two graphs are **isomorphic** — same degree sequence, same
//! adjacency up to renaming, same number of sampled edges per call — so the
//! sampler does the same amount of work on both and the only difference
//! between them is which addresses that work touches. Any gap is locality and
//! nothing else.
//!
//! The baseline is a randomly permuted ordering rather than R-MAT's native
//! one, because R-MAT numbers nodes by recursive quadrant and so arrives
//! pre-clustered. Node ids in a delivered dataset carry no such promise.
//!
//! Seeds are drawn uniformly, which is the ordering-unfriendly case: at the
//! first hop, consecutive frontier entries are unrelated nodes whatever the
//! numbering, so only the later hops — whose frontier is one hop's neighbors,
//! and therefore correlated — have locality available to win back. Batching
//! seeds by community (`partition_aligned_batches`) would raise the numbers
//! further, but it also samples a denser subgraph, so it is a change in what
//! gets sampled and not a like-for-like layout comparison.
//!
//! Reading it: the multi-hop arms hold hundreds of thousands of output
//! elements and dedup slots, which makes them sensitive to memory pressure and
//! to competing load. Trust a difference only when repeated runs agree on its
//! sign and their intervals stay apart; a single run's interval understates
//! the true spread on a busy machine.

use aethergraph_core::{Graph, NeighborSampler, NodeId, SamplingConfig};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use std::time::Duration;

/// 2^21 nodes x avg degree 12 => 16 MB of offsets, ~96 MB of edges.
///
/// Both orderings are held live at once for the A/B, so the scale is set by
/// what two copies plus the build scratch fit in without provoking memory
/// pressure — which perturbs timings far more than the effect being measured.
/// 16 MB of offsets still exceeds any last-level cache here, and at 16 KiB
/// pages the two arrays span ~7000 pages against an L2 TLB an order of
/// magnitude smaller, so the walk stays TLB-bound.
const SCALE: u32 = 21;
const NUM_NODES: usize = 1 << SCALE;
const AVG_DEGREE: usize = 12;
const NUM_EDGES: usize = NUM_NODES * AVG_DEGREE;
/// Seeds per sampled batch, in the range a GNN trainer actually uses.
const BATCH: usize = 1024;

/// Graph500 R-MAT quadrant probabilities.
const RMAT_A: f32 = 0.57;
const RMAT_B: f32 = 0.19;
const RMAT_C: f32 = 0.19;

/// One R-MAT edge: descend the adjacency matrix `SCALE` times, picking a
/// quadrant at each level. The skew between quadrants is what produces both
/// the heavy-tailed degree distribution and the nested community structure.
#[inline]
fn rmat_edge(rng: &mut StdRng) -> (NodeId, NodeId) {
    let (mut src, mut dst) = (0u32, 0u32);
    for level in 0..SCALE {
        let bit = 1u32 << level;
        let r = rng.random::<f32>();
        if r < RMAT_A {
            // top-left: neither endpoint advances
        } else if r < RMAT_A + RMAT_B {
            dst |= bit;
        } else if r < RMAT_A + RMAT_B + RMAT_C {
            src |= bit;
        } else {
            src |= bit;
            dst |= bit;
        }
    }
    (src, dst)
}

fn build_rmat() -> Graph {
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let mut src: Vec<NodeId> = Vec::with_capacity(NUM_EDGES);
    let mut dst: Vec<NodeId> = Vec::with_capacity(NUM_EDGES);
    for _ in 0..NUM_EDGES {
        let (u, v) = rmat_edge(&mut rng);
        src.push(u);
        dst.push(v);
    }
    Graph::from_src_dst(NUM_NODES, &src, &dst, None).expect("csr build")
}

/// A uniformly random permutation, in `permute`'s `perm[new_id] = old_id`
/// convention.
fn shuffled_perm(rng: &mut StdRng) -> Vec<NodeId> {
    let mut perm: Vec<NodeId> = (0..NUM_NODES as NodeId).collect();
    for i in (1..NUM_NODES).rev() {
        perm.swap(i, rng.random_range(0..=i));
    }
    perm
}

/// Invert `perm[new_id] = old_id` into `inv[old_id] = new_id`, so a seed
/// chosen against one ordering can be restated against the other.
fn invert(perm: &[NodeId]) -> Vec<NodeId> {
    let mut inv = vec![0 as NodeId; perm.len()];
    for (new_id, &old_id) in perm.iter().enumerate() {
        inv[old_id as usize] = new_id as NodeId;
    }
    inv
}

/// Assert the two orderings really do give the sampler the same amount of
/// work, so a timing gap can only be layout.
///
/// Isomorphism guarantees equal degrees, and therefore an equal count of
/// emitted edges and visited nodes, for every batch. If these ever diverge
/// the benchmark is comparing two different workloads and its numbers mean
/// nothing — so it refuses to run rather than report them.
fn assert_equal_work(
    baseline: &Graph,
    reordered: &Graph,
    seeds: &[NodeId],
    mapped: &[NodeId],
    config: &SamplingConfig,
) {
    assert_eq!(
        baseline.num_edges(),
        reordered.num_edges(),
        "permutation changed the edge count"
    );
    let a = NeighborSampler::new(baseline, config.clone()).sample_neighbors(seeds);
    let b = NeighborSampler::new(reordered, config.clone()).sample_neighbors(mapped);
    assert_eq!(
        (a.num_nodes(), a.num_edges()),
        (b.num_nodes(), b.num_edges()),
        "orderings sampled different subgraph sizes for fanout {:?}",
        config.fanout
    );
}

fn fanouts() -> Vec<(String, SamplingConfig)> {
    [vec![25], vec![15, 10]]
        .into_iter()
        .map(|fanout| {
            let label = fanout
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join("x");
            let config = SamplingConfig {
                fanout,
                replace: false,
                seed: Some(42),
                ..Default::default()
            };
            (label, config)
        })
        .collect()
}

/// Time one ordering, with only that ordering's graph resident.
fn bench_arm(c: &mut Criterion, arm: &str, graph: &Graph, seeds: &[NodeId]) {
    let mut group = c.benchmark_group("locality");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(15));

    for (label, config) in fanouts() {
        group.bench_with_input(BenchmarkId::new(arm, &label), &config, |b, config| {
            let mut sampler = NeighborSampler::new(graph, config.clone());
            b.iter(|| black_box(sampler.sample_neighbors(black_box(seeds))));
        });
    }

    group.finish();
}

/// Each ordering is timed with only its own graph resident.
///
/// Two graphs of this size held at once is enough to push a modest machine
/// into memory pressure, and pressure moves the multi-hop timings by far more
/// than layout does — enough to flip the sign of the comparison between runs.
/// Both graphs overlap only while the equal-work check runs, never while the
/// timer does.
fn bench_all(c: &mut Criterion) {
    let base = build_rmat();
    let mut seed_rng = StdRng::seed_from_u64(0xC0FFEE);
    let seeds: Vec<NodeId> = (0..BATCH)
        .map(|_| seed_rng.random_range(0..NUM_NODES as NodeId))
        .collect();

    let mut rng = StdRng::seed_from_u64(0xA11CE);
    let baseline = base
        .permute(&shuffled_perm(&mut rng))
        .expect("shuffle permute");
    drop(base);

    bench_arm(c, "shuffled", &baseline, &seeds);

    let perm = baseline.reorder_rabbit();
    let reordered = baseline.permute(&perm).expect("rabbit permute");
    // The same logical nodes, named as the reordered graph names them.
    let mapped: Vec<NodeId> = {
        let inv = invert(&perm);
        seeds.iter().map(|&s| inv[s as usize]).collect()
    };
    for (_, config) in fanouts() {
        assert_equal_work(&baseline, &reordered, &seeds, &mapped, &config);
    }
    drop(baseline);

    bench_arm(c, "rabbit", &reordered, &mapped);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
