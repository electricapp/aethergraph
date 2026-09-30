//! Heterogeneous neighborhood sampling for multi-relational GNN training.
//!
//! Sampling follows PyG's heterogeneous neighbor sampler: a node of type `T`
//! is expanded along every edge type whose destination is `T`, drawing from
//! the sources with an edge into it. Each sampled edge therefore keeps its
//! stored direction, with the expanded node as destination, so messages
//! passed along `edge_index` flow toward the seeds.
//!
//! Internal scratch (frontiers, dedup maps, sample buffers) is pre-allocated at
//! sampler construction and reset across calls via `clear()`. The per-type node
//! and edge buffers handed back in each `HeteroSampledSubgraph` are freshly
//! allocated per call (swapped out of the sampler).
//!
//! Local node indices are assigned during sampling — no post-sort, no
//! binary search. Edges are stored with local indices directly.

use crate::graph::NodeId;
use crate::graph::hetero::{EdgeTypeId, HeteroGraph, InAdjacency, NodeTypeId};
use crate::internal::genstamp::{FRONTIER_PREFETCH_DIST, FloydStamps, GenDedup, GenSlots, WyRand};
use crate::loader::planned_capacity;
use crate::loader::sampler::{SampleError, Seeds};
use rustc_hash::{FxHashMap, FxHashSet};

/// Node types with at most this many nodes get the dense dedup table
/// (8 bytes per node, so at most 8 MB per type) — one random array load per
/// probe, where the hash map pays hash + bucket walk (~3-5x slower and the
/// dominant per-edge cost of the sampler).
const DENSE_DEDUP_MAX_NODES: usize = 1 << 20;

#[derive(Debug, Clone)]
pub struct HeteroSamplingConfig {
    /// `fanout[edge_type][hop]`: sources drawn per expanded destination node.
    pub fanout: Vec<Vec<usize>>,
    pub replace: bool,
    pub seed: Option<u64>,
    /// Accepted for parity with [`crate::loader::SamplingConfig`]. Every
    /// heterogeneous draw is uniform and O(fanout) at any degree, so no cap
    /// applies.
    pub max_degree: Option<usize>,
    pub num_hops: usize,
}

/// Result of heterogeneous neighborhood sampling.
///
/// Edges are stored with **local indices** (into the per-type `nodes` vecs)
/// in the edge type's stored direction: `edge_src_local[et]` indexes
/// `nodes[src_type]` (the sampled neighbor) and `edge_dst_local[et]` indexes
/// `nodes[dst_type]` (the expanded node). Stacked as `[src; dst]` they are
/// PyG's `edge_index` for that edge type.
#[derive(Debug, Clone)]
pub struct HeteroSampledSubgraph {
    /// Per-node-type sampled node IDs (global, in discovery order).
    pub nodes: Vec<Vec<NodeId>>,
    /// Per-edge-type source local indices (index into `nodes[src_type]`).
    pub edge_src_local: Vec<Vec<u32>>,
    /// Per-edge-type destination local indices (index into `nodes[dst_type]`).
    pub edge_dst_local: Vec<Vec<u32>>,
    pub seed_type: NodeTypeId,
    pub seeds: Vec<NodeId>,
    /// Local index of each seed into `nodes[seed_type]`, one entry per
    /// input seed (duplicates preserved). Matches homogeneous
    /// [`crate::loader::SampledSubgraph::seed_indices_local`].
    pub seed_indices: Vec<u32>,
}

/// Heterogeneous neighborhood sampler.
///
/// Internal scratch buffers are pre-allocated at construction and reset across
/// calls; the per-call output buffers are freshly allocated and swapped out
/// into the returned `HeteroSampledSubgraph`.
pub struct HeteroNeighborSampler<'a> {
    graph: &'a HeteroGraph,
    /// Incoming adjacency per edge type, cached on the graph.
    in_adj: &'a [InAdjacency],
    config: HeteroSamplingConfig,
    rng: WyRand,
    /// Per-type dedup: dense generation-stamped table for small types,
    /// hash map for large ones.
    dedup: Vec<GenDedup>,
    /// Per-type: nodes in discovery order (parallel to the dedup state).
    node_vecs: Vec<Vec<NodeId>>,
    /// Per-edge-type edge buffers (local indices).
    edge_src_buf: Vec<Vec<u32>>,
    edge_dst_buf: Vec<Vec<u32>>,
    /// Double-buffered frontiers carrying (type, global id, local index) —
    /// the local index was assigned at discovery, so re-deriving it per
    /// hop with a dedup probe would be pure waste.
    frontier: Vec<(NodeTypeId, NodeId, u32)>,
    next_frontier: Vec<(NodeTypeId, NodeId, u32)>,
    /// Precomputed: edge types into each node type (stack-copied in hot loop).
    in_edge_types: Vec<Vec<EdgeTypeId>>,
    /// Precomputed: source type per edge type.
    src_types: Vec<NodeTypeId>,
    /// Reusable Floyd scratch (small degree ≤256): generation stamps, so
    /// per-node reuse is a counter bump instead of an O(n) clear.
    floyd: FloydStamps,
    /// Reusable Floyd set (large degree).
    floyd_set: FxHashSet<usize>,
    /// Local indices of the seeds, captured at registration.
    seed_indices_buf: Vec<u32>,
}

impl<'a> HeteroNeighborSampler<'a> {
    /// Build a sampler. The first sampler on a graph builds the graph's
    /// incoming adjacency (see [`HeteroGraph::in_adjacency`]); later ones
    /// share it.
    pub fn new(graph: &'a HeteroGraph, config: HeteroSamplingConfig) -> Self {
        let seed = config.seed.unwrap_or_else(rand::random::<u64>);
        let num_nt = graph.node_type_count();
        let num_et = graph.edge_type_count();

        let max_fanout = config
            .fanout
            .iter()
            .flat_map(|f| f.iter())
            .max()
            .copied()
            .unwrap_or(25);

        let dedup: Vec<GenDedup> = (0..num_nt)
            .map(|nt| {
                let n = graph.num_nodes(nt as NodeTypeId);
                if n <= DENSE_DEDUP_MAX_NODES {
                    GenDedup::Dense(GenSlots::new(n))
                } else {
                    GenDedup::Map(FxHashMap::with_capacity_and_hasher(256, Default::default()))
                }
            })
            .collect();
        let node_vecs: Vec<Vec<NodeId>> = (0..num_nt).map(|_| Vec::with_capacity(256)).collect();
        let edge_src_buf: Vec<Vec<u32>> = (0..num_et).map(|_| Vec::with_capacity(1024)).collect();
        let edge_dst_buf: Vec<Vec<u32>> = (0..num_et).map(|_| Vec::with_capacity(1024)).collect();

        let in_edge_types = (0..num_nt)
            .map(|nt| graph.edge_types_for_dst(nt as NodeTypeId))
            .collect();
        let src_types = (0..num_et)
            .map(|et| graph.edge_type_meta(et as EdgeTypeId).src_type)
            .collect();

        Self {
            graph,
            in_adj: graph.in_adjacency(),
            config,
            rng: WyRand::new(seed),
            dedup,
            node_vecs,
            edge_src_buf,
            edge_dst_buf,
            frontier: Vec::with_capacity(512),
            next_frontier: Vec::with_capacity(4096),
            in_edge_types,
            src_types,
            floyd: FloydStamps::new(),
            floyd_set: FxHashSet::with_capacity_and_hasher(max_fanout * 2, Default::default()),
            seed_indices_buf: Vec::with_capacity(512),
        }
    }

    /// Reset the RNG without rebuilding scratch (see [`super::batch_seed`]).
    #[inline]
    pub fn reseed(&mut self, seed: u64) {
        self.rng = WyRand::new(seed);
    }

    /// Sample from checked seeds of `seed_type`.
    ///
    /// # Errors
    /// [`SampleError::SeedsExceedGraph`] when `seeds` were checked against
    /// more nodes than `seed_type` has.
    ///
    /// # Panics
    /// When `seed_type` is not a node type of the graph.
    pub fn sample(
        &mut self,
        seed_type: NodeTypeId,
        seeds: &Seeds,
    ) -> Result<HeteroSampledSubgraph, SampleError> {
        let num_nodes = self.graph.num_nodes(seed_type);
        if seeds.num_nodes() > num_nodes {
            return Err(SampleError::SeedsExceedGraph {
                checked_against: seeds.num_nodes(),
                num_nodes,
            });
        }
        Ok(self.sample_neighbors(seed_type, seeds.ids()))
    }

    /// Sample from seeds of `seed_type`. Every seed must be below
    /// `graph.num_nodes(seed_type)`; an out-of-range seed panics.
    /// [`Self::sample`] takes range-checked [`Seeds`].
    pub fn sample_neighbors(
        &mut self,
        seed_type: NodeTypeId,
        seeds: &[NodeId],
    ) -> HeteroSampledSubgraph {
        // Reset reusable state. Dense dedup tables reset with a generation
        // bump (a fill only on the u32 wrap); maps clear normally.
        for d in &mut self.dedup {
            d.begin();
        }
        for v in &mut self.node_vecs {
            v.clear();
        }
        for v in &mut self.edge_src_buf {
            v.clear();
        }
        for v in &mut self.edge_dst_buf {
            v.clear();
        }
        self.frontier.clear();
        self.seed_indices_buf.clear();

        // Register seeds with local indices. A duplicate seed keeps its own
        // seed_indices entry, pointing at the same local slot, and is
        // expanded once.
        for &seed in seeds {
            let (idx, is_new) = self.insert_node(seed, seed_type);
            self.seed_indices_buf.push(idx);
            if is_new {
                self.frontier.push((seed_type, seed, idx));
            }
        }

        let in_adj = self.in_adj;

        // Hop-by-hop sampling
        for hop in 0..self.config.num_hops {
            self.next_frontier.clear();
            let frontier_len = self.frontier.len();

            for fi in 0..frontier_len {
                // Prefetch the first incoming edge type's offset word for an
                // upcoming frontier entry — the probes are random-access and
                // serially dependent without the hint.
                if fi + FRONTIER_PREFETCH_DIST < frontier_len {
                    let (nt_ahead, id_ahead, _) = self.frontier[fi + FRONTIER_PREFETCH_DIST];
                    if let Some(&et0) = self.in_edge_types[nt_ahead as usize].first() {
                        let offsets = in_adj[et0 as usize].offsets();
                        if (id_ahead as usize) < offsets.len() {
                            crate::internal::prefetch::prefetch_read(&offsets[id_ahead as usize]);
                        }
                    }
                }

                let (node_type, node_id, dst_local) = self.frontier[fi];

                // Copy the node type's incoming edge-type IDs out so the inner
                // loop can call &mut self sampling methods without holding a
                // borrow of self.in_edge_types. Common case (<= 32 types) uses
                // a stack buffer; a node type with more types falls back to a
                // heap Vec rather than overflowing the fixed array.
                let in_ets = &self.in_edge_types[node_type as usize];
                let et_count = in_ets.len();
                let mut et_stack = [0u8; 32];
                let mut et_heap: Vec<EdgeTypeId> = Vec::new();
                let et_slice: &[EdgeTypeId] = if et_count <= et_stack.len() {
                    et_stack[..et_count].copy_from_slice(in_ets);
                    &et_stack[..et_count]
                } else {
                    et_heap.extend_from_slice(in_ets);
                    &et_heap[..]
                };

                for &et_id in et_slice {
                    let et = et_id as usize;
                    let fanout = self.config.fanout[et][hop];
                    if fanout == 0 {
                        continue;
                    }

                    let sources = in_adj[et].sources(node_id);
                    if sources.is_empty() {
                        continue;
                    }

                    let src_type = self.src_types[et];
                    if self.config.replace {
                        self.sample_replace(sources, fanout, dst_local, et, src_type);
                    } else if sources.len() <= fanout {
                        self.take_all(sources, dst_local, et, src_type);
                    } else {
                        self.sample_floyd(sources, fanout, dst_local, et, src_type);
                    }
                }
            }

            std::mem::swap(&mut self.frontier, &mut self.next_frontier);
        }

        // Build output: hand the filled per-type buffers to the caller and
        // swap fresh ones back in, each sized at the length its buffer just
        // reached. A steady stream of same-shaped batches therefore allocates
        // each per-type array once instead of growing into it every call, and
        // a type left untouched asks for capacity 0 — which `Vec` serves
        // without allocating, so wide schemas still pay nothing for the types
        // a batch never visits.
        let num_nt = self.node_vecs.len();
        let num_et = self.edge_src_buf.len();

        let mut nodes = Vec::with_capacity(num_nt);
        for i in 0..num_nt {
            let mut swap = Vec::with_capacity(planned_capacity(self.node_vecs[i].len(), 0));
            std::mem::swap(&mut self.node_vecs[i], &mut swap);
            nodes.push(swap);
        }

        let mut edge_src = Vec::with_capacity(num_et);
        let mut edge_dst = Vec::with_capacity(num_et);
        for i in 0..num_et {
            let planned = planned_capacity(self.edge_src_buf[i].len(), 0);
            let (mut s, mut d) = (Vec::with_capacity(planned), Vec::with_capacity(planned));
            std::mem::swap(&mut self.edge_src_buf[i], &mut s);
            std::mem::swap(&mut self.edge_dst_buf[i], &mut d);
            edge_src.push(s);
            edge_dst.push(d);
        }

        let mut seed_indices = Vec::with_capacity(seeds.len());
        std::mem::swap(&mut self.seed_indices_buf, &mut seed_indices);

        HeteroSampledSubgraph {
            nodes,
            edge_src_local: edge_src,
            edge_dst_local: edge_dst,
            seed_type,
            seeds: seeds.to_vec(),
            seed_indices,
        }
    }

    /// Insert a node of `node_type`, assigning a local index if new.
    /// Returns `(local index, is_new)`.
    #[inline(always)]
    fn insert_node(&mut self, id: NodeId, node_type: NodeTypeId) -> (u32, bool) {
        let nt = node_type as usize;
        let (idx, is_new) = self.dedup[nt].probe_or_insert(id, self.node_vecs[nt].len() as u32);
        if is_new {
            self.node_vecs[nt].push(id);
        }
        (idx, is_new)
    }

    /// Register a sampled source node and push its edge into `dst_local`.
    /// Sources come from the graph's incoming adjacency, which holds only
    /// in-range IDs.
    #[inline(always)]
    fn emit(&mut self, src_id: NodeId, src_type: NodeTypeId, dst_local: u32, et: usize) {
        let (src_local, is_new) = self.insert_node(src_id, src_type);
        if is_new {
            self.next_frontier.push((src_type, src_id, src_local));
        }
        self.edge_src_buf[et].push(src_local);
        self.edge_dst_buf[et].push(dst_local);
    }

    /// Uniform integer in `[0, s)` for `s <= 2^32` (multiply-shift).
    #[inline(always)]
    fn rand_below(&mut self, s: usize) -> usize {
        (u64::from(self.rng.next_u32()).wrapping_mul(s as u64) >> 32) as usize
    }

    #[inline]
    fn take_all(&mut self, sources: &[NodeId], dst_local: u32, et: usize, src_type: NodeTypeId) {
        for &src_id in sources {
            self.emit(src_id, src_type, dst_local, et);
        }
    }

    #[inline]
    fn sample_replace(
        &mut self,
        sources: &[NodeId],
        k: usize,
        dst_local: u32,
        et: usize,
        src_type: NodeTypeId,
    ) {
        for _ in 0..k {
            let idx = self.rand_below(sources.len());
            self.emit(sources[idx], src_type, dst_local, et);
        }
    }

    #[inline]
    fn sample_floyd(
        &mut self,
        sources: &[NodeId],
        k: usize,
        dst_local: u32,
        et: usize,
        src_type: NodeTypeId,
    ) {
        let n = sources.len();

        if n <= 256 {
            // Stamp reuse is a counter bump, not an O(n) clear per node.
            self.floyd.begin();
            for i in (n - k)..n {
                let j = self.rand_below(i + 1);
                let pick = if self.floyd.test_and_set(j) {
                    j
                } else {
                    self.floyd.test_and_set(i);
                    i
                };
                self.emit(sources[pick], src_type, dst_local, et);
            }
        } else {
            self.floyd_set.clear();
            for i in (n - k)..n {
                let j = self.rand_below(i + 1);
                let pick = if self.floyd_set.insert(j) {
                    j
                } else {
                    self.floyd_set.insert(i);
                    i
                };
                self.emit(sources[pick], src_type, dst_local, et);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::graph::hetero::HeteroGraph;

    /// Reddit-shaped graph whose relations point into users, so user seeds
    /// have neighbors to draw.
    ///
    /// Node types: user(100), post(100), comment(100), subreddit(20)
    /// Edge types:
    ///   - (post, voted_by, user): users 0..50 each voted on posts 0..4
    ///   - (comment, written_by, user): users 0..30 each wrote comments 0..3
    ///   - (subreddit, contains, post): post p sits in subreddit p % 20
    fn build_reddit_graph() -> HeteroGraph {
        let mut voted_by = Vec::new();
        for user in 0u32..50 {
            for post in 0u32..4 {
                voted_by.push((post, user));
            }
        }
        voted_by.sort_unstable();
        let voted_by_csr = Graph::from_edges(100, &voted_by, None).unwrap();

        let mut written_by = Vec::new();
        for user in 0u32..30 {
            for comment in 0u32..3 {
                written_by.push((comment, user));
            }
        }
        written_by.sort_unstable();
        let written_by_csr = Graph::from_edges(100, &written_by, None).unwrap();

        let mut contains: Vec<(u32, u32)> = (0u32..100).map(|post| (post % 20, post)).collect();
        contains.sort_unstable();
        let contains_csr = Graph::from_edges(100, &contains, None).unwrap();

        HeteroGraph::from_parts(
            vec![
                ("user".into(), 100),
                ("post".into(), 100),
                ("comment".into(), 100),
                ("subreddit".into(), 20),
            ],
            vec![
                (
                    "post".into(),
                    "voted_by".into(),
                    "user".into(),
                    voted_by_csr,
                ),
                (
                    "comment".into(),
                    "written_by".into(),
                    "user".into(),
                    written_by_csr,
                ),
                (
                    "subreddit".into(),
                    "contains".into(),
                    "post".into(),
                    contains_csr,
                ),
            ],
        )
    }

    fn config(fanout: Vec<Vec<usize>>, replace: bool, num_hops: usize) -> HeteroSamplingConfig {
        HeteroSamplingConfig {
            fanout,
            replace,
            seed: Some(42),
            max_degree: None,
            num_hops,
        }
    }

    #[test]
    fn basic_2hop_sampling() {
        let graph = build_reddit_graph();
        let mut sampler = HeteroNeighborSampler::new(&graph, config(vec![vec![5, 3]; 3], false, 2));
        let user_type = graph.node_type_id("user").unwrap();
        let post_type = graph.node_type_id("post").unwrap();
        let subreddit_type = graph.node_type_id("subreddit").unwrap();

        let sub = sampler.sample_neighbors(user_type, &[0, 1, 2]);
        assert!(sub.nodes[user_type as usize].len() >= 3);
        assert!(!sub.nodes[post_type as usize].is_empty());
        // Hop 2 expands posts along (subreddit, contains, post).
        assert!(!sub.nodes[subreddit_type as usize].is_empty());
    }

    /// Every sampled edge is a stored edge whose destination is the expanded
    /// node: messages along it flow toward the seeds.
    #[test]
    fn edges_point_at_the_expanded_node() {
        let graph = build_reddit_graph();
        let mut sampler = HeteroNeighborSampler::new(&graph, config(vec![vec![5]; 3], false, 1));
        let user_type = graph.node_type_id("user").unwrap();
        let post_type = graph.node_type_id("post").unwrap();
        let voted_by = graph.edge_type_id("post", "voted_by", "user").unwrap();

        let sub = sampler.sample_neighbors(user_type, &[0, 1, 2]);
        let users = &sub.nodes[user_type as usize];
        let posts = &sub.nodes[post_type as usize];
        let src = &sub.edge_src_local[voted_by as usize];
        let dst = &sub.edge_dst_local[voted_by as usize];
        assert!(!src.is_empty());
        for (&s, &d) in src.iter().zip(dst) {
            let post = posts[s as usize];
            let user = users[d as usize];
            assert!(
                [0, 1, 2].contains(&user),
                "hop-1 edge must end at a seed, got user {user}"
            );
            assert!(
                graph.csr(voted_by).neighbors(post).contains(&user),
                "edge {post}->{user} not in the stored relation"
            );
        }
    }

    #[test]
    fn edges_use_local_indices() {
        let graph = build_reddit_graph();
        let mut sampler = HeteroNeighborSampler::new(&graph, config(vec![vec![5, 5]; 3], false, 2));
        let user_type = graph.node_type_id("user").unwrap();

        let sub = sampler.sample_neighbors(user_type, &[0, 1, 2]);
        for et in 0..graph.edge_type_count() {
            let meta = graph.edge_type_meta(et as EdgeTypeId);
            let num_src = sub.nodes[meta.src_type as usize].len();
            let num_dst = sub.nodes[meta.dst_type as usize].len();
            for &s in &sub.edge_src_local[et] {
                assert!((s as usize) < num_src, "src local {s} >= {num_src}");
            }
            for &d in &sub.edge_dst_local[et] {
                assert!((d as usize) < num_dst, "dst local {d} >= {num_dst}");
            }
        }
    }

    #[test]
    fn reuse_across_calls() {
        let graph = build_reddit_graph();
        let mut sampler = HeteroNeighborSampler::new(&graph, config(vec![vec![5, 3]; 3], false, 2));
        let user_type = graph.node_type_id("user").unwrap();

        let sub1 = sampler.sample_neighbors(user_type, &[0, 1, 2]);
        let sub2 = sampler.sample_neighbors(user_type, &[3, 4, 5]);

        assert!(sub1.nodes[user_type as usize].len() >= 3);
        assert!(sub2.nodes[user_type as usize].len() >= 3);
        assert_ne!(sub1.seeds, sub2.seeds);
    }

    #[test]
    fn empty_seeds() {
        let graph = build_reddit_graph();
        let mut sampler = HeteroNeighborSampler::new(&graph, config(vec![vec![5]; 3], false, 1));
        let user_type = graph.node_type_id("user").unwrap();
        let sub = sampler.sample_neighbors(user_type, &[]);
        assert!(sub.seeds.is_empty());
    }

    #[test]
    fn with_replacement() {
        let graph = build_reddit_graph();
        let mut sampler = HeteroNeighborSampler::new(&graph, config(vec![vec![20]; 3], true, 1));
        let user_type = graph.node_type_id("user").unwrap();
        let voted_by = graph.edge_type_id("post", "voted_by", "user").unwrap();

        let sub = sampler.sample_neighbors(user_type, &[0]);
        // With replacement, fanout=20, 4 voted posts → 20 edges
        assert_eq!(sub.edge_src_local[voted_by as usize].len(), 20);
        let unique: FxHashSet<u32> = sub.edge_src_local[voted_by as usize]
            .iter()
            .copied()
            .collect();
        assert!(unique.len() <= 4);
    }

    #[test]
    fn deterministic_with_seed() {
        let graph = build_reddit_graph();
        let cfg = config(vec![vec![5, 3]; 3], false, 2);
        let user_type = graph.node_type_id("user").unwrap();

        let mut s1 = HeteroNeighborSampler::new(&graph, cfg.clone());
        let sub1 = s1.sample_neighbors(user_type, &[0, 1, 2]);

        let mut s2 = HeteroNeighborSampler::new(&graph, cfg);
        let sub2 = s2.sample_neighbors(user_type, &[0, 1, 2]);

        for i in 0..graph.node_type_count() {
            assert_eq!(sub1.nodes[i], sub2.nodes[i]);
        }
    }

    /// Node types: author(50), paper(100), venue(20)
    /// Edge types:
    ///   - (paper, written_by, author): authors 0..30 each wrote papers 0..3
    ///   - (paper, cites, paper): paper p cites paper (p+1)%100 for p < 50
    ///   - (venue, publishes, paper): paper p appears at venue p%20
    ///   - (paper, reviewed_by, author): authors 0..20 each reviewed papers 5..8
    fn build_multi_edge_graph() -> HeteroGraph {
        let mut written_by = Vec::new();
        for author in 0u32..30 {
            for paper in 0u32..3 {
                written_by.push((paper, author));
            }
        }
        written_by.sort_unstable();
        let written_by_csr = Graph::from_edges(100, &written_by, None).unwrap();

        let cites: Vec<(u32, u32)> = (0u32..50).map(|p| (p, (p + 1) % 100)).collect();
        let cites_csr = Graph::from_edges(100, &cites, None).unwrap();

        let mut publishes: Vec<(u32, u32)> = (0u32..100).map(|p| (p % 20, p)).collect();
        publishes.sort_unstable();
        let publishes_csr = Graph::from_edges(100, &publishes, None).unwrap();

        let mut reviewed_by = Vec::new();
        for author in 0u32..20 {
            for paper in 5u32..8 {
                reviewed_by.push((paper, author));
            }
        }
        reviewed_by.sort_unstable();
        let reviewed_by_csr = Graph::from_edges(100, &reviewed_by, None).unwrap();

        HeteroGraph::from_parts(
            vec![
                ("author".into(), 50),
                ("paper".into(), 100),
                ("venue".into(), 20),
            ],
            vec![
                (
                    "paper".into(),
                    "written_by".into(),
                    "author".into(),
                    written_by_csr,
                ),
                ("paper".into(), "cites".into(), "paper".into(), cites_csr),
                (
                    "venue".into(),
                    "publishes".into(),
                    "paper".into(),
                    publishes_csr,
                ),
                (
                    "paper".into(),
                    "reviewed_by".into(),
                    "author".into(),
                    reviewed_by_csr,
                ),
            ],
        )
    }

    #[test]
    fn test_multi_edge_type_sampling() {
        let graph = build_multi_edge_graph();
        let author_type = graph.node_type_id("author").unwrap();
        let paper_type = graph.node_type_id("paper").unwrap();
        let venue_type = graph.node_type_id("venue").unwrap();

        let written_by = graph.edge_type_id("paper", "written_by", "author").unwrap();
        let cites = graph.edge_type_id("paper", "cites", "paper").unwrap();
        let publishes = graph.edge_type_id("venue", "publishes", "paper").unwrap();
        let reviewed_by = graph
            .edge_type_id("paper", "reviewed_by", "author")
            .unwrap();

        // 2-hop: author seeds <- (written_by, reviewed_by) <- paper
        //        <- (cites, publishes) <- paper/venue
        let mut sampler = HeteroNeighborSampler::new(&graph, config(vec![vec![3, 2]; 4], false, 2));
        let sub = sampler.sample_neighbors(author_type, &[0, 1, 2]);

        assert!(sub.nodes[author_type as usize].len() >= 3);
        assert!(
            !sub.edge_src_local[written_by as usize].is_empty()
                || !sub.edge_src_local[reviewed_by as usize].is_empty(),
            "at least one paper->author edge type should be populated"
        );
        assert!(
            !sub.edge_src_local[cites as usize].is_empty()
                || !sub.edge_src_local[publishes as usize].is_empty(),
            "at least one X->paper edge type should be populated in hop 2"
        );
        assert!(!sub.nodes[paper_type as usize].is_empty());
        assert!(!sub.nodes[venue_type as usize].is_empty());
    }

    #[test]
    fn test_seed_type_filtering() {
        let graph = build_reddit_graph();
        let user_type = graph.node_type_id("user").unwrap();
        let post_type = graph.node_type_id("post").unwrap();
        let cfg = config(vec![vec![3]; 3], false, 1);

        let mut sampler = HeteroNeighborSampler::new(&graph, cfg.clone());
        let sub_user = sampler.sample_neighbors(user_type, &[0, 1, 2]);
        assert_eq!(sub_user.seed_type, user_type);
        assert_eq!(sub_user.seeds, vec![0, 1, 2]);
        assert_eq!(&sub_user.nodes[user_type as usize][..3], &[0, 1, 2]);

        let mut sampler2 = HeteroNeighborSampler::new(&graph, cfg);
        let sub_post = sampler2.sample_neighbors(post_type, &[10, 20]);
        assert_eq!(sub_post.seed_type, post_type);
        assert_eq!(sub_post.seeds, vec![10, 20]);
        assert_eq!(&sub_post.nodes[post_type as usize][..2], &[10, 20]);
    }

    #[test]
    fn test_fanout_per_edge_type() {
        let graph = build_reddit_graph();
        let user_type = graph.node_type_id("user").unwrap();
        let voted_by = graph.edge_type_id("post", "voted_by", "user").unwrap();
        let written_by = graph.edge_type_id("comment", "written_by", "user").unwrap();
        let contains = graph.edge_type_id("subreddit", "contains", "post").unwrap();

        let mut fanout = vec![vec![0]; 3];
        fanout[voted_by as usize] = vec![2];
        fanout[written_by as usize] = vec![1];
        let mut sampler = HeteroNeighborSampler::new(&graph, config(fanout, false, 1));

        // User 0 voted on 4 posts and wrote 3 comments.
        let sub = sampler.sample_neighbors(user_type, &[0]);
        assert_eq!(sub.edge_src_local[voted_by as usize].len(), 2);
        assert_eq!(sub.edge_src_local[written_by as usize].len(), 1);
        assert!(sub.edge_src_local[contains as usize].is_empty());
    }

    #[test]
    fn test_hetero_with_replacement() {
        let graph = build_reddit_graph();
        let user_type = graph.node_type_id("user").unwrap();
        let voted_by = graph.edge_type_id("post", "voted_by", "user").unwrap();
        let written_by = graph.edge_type_id("comment", "written_by", "user").unwrap();

        let mut sampler =
            HeteroNeighborSampler::new(&graph, config(vec![vec![50], vec![50], vec![0]], true, 1));
        let sub = sampler.sample_neighbors(user_type, &[0]);

        assert_eq!(sub.edge_src_local[voted_by as usize].len(), 50);
        assert_eq!(sub.edge_src_local[written_by as usize].len(), 50);

        let src_locals = &sub.edge_src_local[voted_by as usize];
        let unique: FxHashSet<u32> = src_locals.iter().copied().collect();
        assert!(unique.len() <= 4);
        assert!(src_locals.len() > unique.len());
    }

    /// A hub's whole in-neighborhood stays eligible, whatever `max_degree`.
    #[test]
    fn test_hetero_high_degree_node() {
        // B-node 0 has 1500 in-neighbors of type A.
        let edges: Vec<(u32, u32)> = (0u32..1500).map(|a| (a, 0)).collect();
        let csr = Graph::from_edges(2000, &edges, None).unwrap();
        let graph = HeteroGraph::from_parts(
            vec![("A".into(), 2000), ("B".into(), 2000)],
            vec![("A".into(), "connects".into(), "B".into(), csr)],
        );
        let a_type = graph.node_type_id("A").unwrap();
        let b_type = graph.node_type_id("B").unwrap();
        let connects = graph.edge_type_id("A", "connects", "B").unwrap();

        let mut cfg = config(vec![vec![5]], false, 1);
        cfg.max_degree = Some(100);
        let mut sampler = HeteroNeighborSampler::new(&graph, cfg);
        let mut max_src = 0;
        for _ in 0..50 {
            let sub = sampler.sample_neighbors(b_type, &[0]);
            assert_eq!(sub.edge_src_local[connects as usize].len(), 5);
            for &s in &sub.edge_src_local[connects as usize] {
                max_src = max_src.max(sub.nodes[a_type as usize][s as usize]);
            }
        }
        assert!(
            max_src >= 100,
            "draws confined to the first max_degree in-neighbors"
        );
    }

    #[test]
    fn test_hetero_disconnected_type() {
        let writes: Vec<(u32, u32)> = (0u32..10).map(|u| (u, u)).collect();
        let writes_csr = Graph::from_edges(20, &writes, None).unwrap();
        let empty_csr = Graph::from_edges(20, &[], None).unwrap();

        let graph = HeteroGraph::from_parts(
            vec![("user".into(), 20), ("post".into(), 20), ("tag".into(), 20)],
            vec![
                (
                    "post".into(),
                    "written_by".into(),
                    "user".into(),
                    writes_csr,
                ),
                ("tag".into(), "tags".into(), "user".into(), empty_csr),
            ],
        );

        let user_type = graph.node_type_id("user").unwrap();
        let tag_type = graph.node_type_id("tag").unwrap();
        let tags = graph.edge_type_id("tag", "tags", "user").unwrap();
        let written_by = graph.edge_type_id("post", "written_by", "user").unwrap();

        let mut sampler = HeteroNeighborSampler::new(&graph, config(vec![vec![5]; 2], false, 1));
        let sub = sampler.sample_neighbors(user_type, &[0, 1, 2]);

        assert!(sub.edge_src_local[tags as usize].is_empty());
        assert!(sub.edge_dst_local[tags as usize].is_empty());
        assert!(sub.nodes[tag_type as usize].is_empty());
        assert!(!sub.edge_src_local[written_by as usize].is_empty());
    }

    #[test]
    fn test_hetero_single_node_type() {
        let mut edges = Vec::new();
        for src in 0u32..20 {
            for offset in 1u32..4 {
                edges.push((src, (src + offset) % 50));
            }
        }
        let csr = Graph::from_edges(50, &edges, None).unwrap();
        let graph = HeteroGraph::from_parts(
            vec![("node".into(), 50)],
            vec![("node".into(), "link".into(), "node".into(), csr)],
        );
        let node_type = graph.node_type_id("node").unwrap();
        let link = graph.edge_type_id("node", "link", "node").unwrap();

        let mut sampler = HeteroNeighborSampler::new(&graph, config(vec![vec![2, 2]], false, 2));
        let sub = sampler.sample_neighbors(node_type, &[3, 5, 10]);

        assert_eq!(&sub.nodes[node_type as usize][..3], &[3, 5, 10]);
        assert!(!sub.edge_src_local[link as usize].is_empty());

        let num_nodes = sub.nodes[node_type as usize].len();
        for (&s, &d) in sub.edge_src_local[link as usize]
            .iter()
            .zip(&sub.edge_dst_local[link as usize])
        {
            assert!((s as usize) < num_nodes && (d as usize) < num_nodes);
            let src_global = sub.nodes[node_type as usize][s as usize];
            let dst_global = sub.nodes[node_type as usize][d as usize];
            assert!(
                graph.csr(link).neighbors(src_global).contains(&dst_global),
                "edge {src_global}->{dst_global} not in CSR"
            );
        }
    }

    #[test]
    fn test_hetero_large_batch() {
        let graph = build_reddit_graph();
        let user_type = graph.node_type_id("user").unwrap();
        let post_type = graph.node_type_id("post").unwrap();
        let voted_by = graph.edge_type_id("post", "voted_by", "user").unwrap();

        let seeds: Vec<u32> = (0u32..100).collect();
        let mut sampler = HeteroNeighborSampler::new(&graph, config(vec![vec![5, 3]; 3], false, 2));
        let sub = sampler.sample_neighbors(user_type, &seeds);

        assert_eq!(sub.seeds.len(), 100);
        assert_eq!(&sub.nodes[user_type as usize][..100], &seeds[..]);
        assert!(!sub.nodes[post_type as usize].is_empty());
        assert!(!sub.edge_src_local[voted_by as usize].is_empty());
    }

    /// A duplicate seed is expanded once, and both entries point at its slot.
    #[test]
    fn duplicate_seeds_expand_once() {
        let graph = build_reddit_graph();
        let user_type = graph.node_type_id("user").unwrap();
        let voted_by = graph.edge_type_id("post", "voted_by", "user").unwrap();
        let mut sampler = HeteroNeighborSampler::new(&graph, config(vec![vec![10]; 3], false, 1));
        let sub = sampler.sample_neighbors(user_type, &[0, 0]);
        assert_eq!(sub.seed_indices, vec![0, 0]);
        // User 0 voted on 4 posts; expanding it twice would emit 8 edges.
        assert_eq!(sub.edge_src_local[voted_by as usize].len(), 4);
    }

    #[test]
    fn sample_checks_seed_bound_against_the_seed_type() {
        let graph = build_reddit_graph();
        let user_type = graph.node_type_id("user").unwrap();
        let subreddit_type = graph.node_type_id("subreddit").unwrap();
        let mut sampler = HeteroNeighborSampler::new(&graph, config(vec![vec![2]; 3], false, 1));

        let users = Seeds::new(vec![0, 99], graph.num_nodes(user_type)).unwrap();
        assert!(sampler.sample(user_type, &users).is_ok());
        assert_eq!(
            sampler.sample(subreddit_type, &users).unwrap_err(),
            SampleError::SeedsExceedGraph {
                checked_against: 100,
                num_nodes: 20
            }
        );
    }

    /// The incoming adjacency is the exact transpose of each relation and is
    /// built once per graph.
    #[test]
    fn in_adjacency_transposes_each_relation_once() {
        let graph = build_multi_edge_graph();
        let first = graph.in_adjacency().as_ptr();
        assert_eq!(graph.in_adjacency().as_ptr(), first, "rebuilt per call");
        for et in 0..graph.edge_type_count() {
            let meta = graph.edge_type_meta(et as EdgeTypeId);
            let csr = graph.csr(et as EdgeTypeId);
            let adj = &graph.in_adjacency()[et];
            let mut forward = Vec::new();
            for s in 0..graph.num_nodes(meta.src_type) as NodeId {
                for &d in csr.neighbors(s) {
                    forward.push((s, d));
                }
            }
            let mut transposed = Vec::new();
            for d in 0..graph.num_nodes(meta.dst_type) as NodeId {
                for &s in adj.sources(d) {
                    transposed.push((s, d));
                }
            }
            forward.sort_unstable();
            transposed.sort_unstable();
            assert_eq!(forward, transposed, "edge type {et}");
        }
    }
}
