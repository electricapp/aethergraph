//! Heterogeneous neighborhood sampling for multi-relational GNN training.
//!
//! Internal scratch (frontiers, dedup maps, sample buffers) is pre-allocated at
//! sampler construction and reset across calls via `clear()`. The per-type node
//! and edge buffers handed back in each `HeteroSampledSubgraph` are freshly
//! allocated per call (swapped out of the sampler).
//!
//! Local node indices are assigned during sampling — no post-sort, no
//! binary search. Edges are stored with local indices directly.

use crate::graph::hetero::{EdgeTypeId, HeteroGraph, NodeTypeId};
use crate::graph::{CsrView, NodeId};
use crate::internal::genstamp::{FRONTIER_PREFETCH_DIST, FloydStamps, GenDedup, GenSlots, WyRand};
use crate::loader::planned_capacity;
use rustc_hash::{FxHashMap, FxHashSet};

/// Node types with at most this many nodes get the dense dedup table
/// (8 bytes per node, so at most 8 MB per type) — one random array load per
/// probe, where the hash map pays hash + bucket walk (~3-5x slower and the
/// dominant per-edge cost of the sampler).
const DENSE_DEDUP_MAX_NODES: usize = 1 << 20;

#[derive(Debug, Clone)]
pub struct HeteroSamplingConfig {
    pub fanout: Vec<Vec<usize>>,
    pub replace: bool,
    pub seed: Option<u64>,
    pub max_degree: Option<usize>,
    pub num_hops: usize,
}

/// Result of heterogeneous neighborhood sampling.
///
/// Edges are stored with **local indices** (into the per-type `nodes` vecs).
/// No post-processing needed — pass `edge_src_local`/`edge_dst_local` directly
/// to PyG's `edge_index`.
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
/// `sample_neighbors` calls; the per-call output buffers are freshly allocated
/// and swapped out into the returned `HeteroSampledSubgraph`.
pub struct HeteroNeighborSampler<'a> {
    graph: &'a HeteroGraph,
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
    /// Precomputed: src edge types per node type (stack-copied in hot loop).
    src_edge_types: Vec<Vec<EdgeTypeId>>,
    /// Precomputed: dst type per edge type.
    dst_types: Vec<NodeTypeId>,
    /// Reusable Floyd scratch (small degree ≤256): generation stamps, so
    /// per-node reuse is a counter bump instead of an O(n) clear.
    floyd: FloydStamps,
    /// Reusable Floyd set (large degree).
    floyd_set: FxHashSet<usize>,
    /// Local indices of the seeds, captured at registration.
    seed_indices_buf: Vec<u32>,
}

impl<'a> HeteroNeighborSampler<'a> {
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

        let mut src_edge_types = vec![Vec::new(); num_nt];
        for (nt, entry) in src_edge_types.iter_mut().enumerate() {
            *entry = graph.edge_types_for_src(nt as NodeTypeId);
        }
        let mut dst_types = Vec::with_capacity(num_et);
        for et in 0..num_et {
            dst_types.push(graph.edge_type_meta(et as EdgeTypeId).dst_type);
        }

        Self {
            graph,
            config,
            rng: WyRand::new(seed),
            dedup,
            node_vecs,
            edge_src_buf,
            edge_dst_buf,
            frontier: Vec::with_capacity(512),
            next_frontier: Vec::with_capacity(4096),
            src_edge_types,
            dst_types,
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

        // Register seeds with local indices (duplicates keep distinct
        // seed_indices entries pointing at the same local slot).
        for &seed in seeds {
            let (idx, _) = self.insert_node(seed, seed_type);
            self.seed_indices_buf.push(idx);
            self.frontier.push((seed_type, seed, idx));
        }

        // Hoist every edge type's CSR view once per call: the inner loop
        // then slices raw arrays instead of re-deriving storage per
        // (node, edge type), and the offsets arrays double as prefetch
        // targets.
        let csrs: Vec<CsrView<'_>> = (0..self.dst_types.len())
            .map(|et| self.graph.csr(et as EdgeTypeId).csr_view())
            .collect();

        // Hop-by-hop sampling
        for hop in 0..self.config.num_hops {
            self.next_frontier.clear();
            let frontier_len = self.frontier.len();

            for fi in 0..frontier_len {
                // Prefetch the first source edge type's offset word for an
                // upcoming frontier entry — the probes are random-access
                // and serially dependent without the hint.
                if fi + FRONTIER_PREFETCH_DIST < frontier_len {
                    let (nt_ahead, id_ahead, _) = self.frontier[fi + FRONTIER_PREFETCH_DIST];
                    if let Some(&et0) = self.src_edge_types[nt_ahead as usize].first() {
                        let offsets = csrs[et0 as usize].offsets();
                        if (id_ahead as usize) < offsets.len() {
                            crate::internal::prefetch::prefetch_read(&offsets[id_ahead as usize]);
                        }
                    }
                }

                let (node_type, local_id, src_local_idx) = self.frontier[fi];

                // Copy the source node type's edge-type IDs out so the inner
                // loop can call &mut self sampling methods without holding a
                // borrow of self.src_edge_types. Common case (<= 32 types) uses
                // a stack buffer; a node type with more types falls back to a
                // heap Vec rather than overflowing the fixed array.
                let src_ets = &self.src_edge_types[node_type as usize];
                let et_count = src_ets.len();
                let mut et_stack = [0u8; 32];
                let mut et_heap: Vec<EdgeTypeId> = Vec::new();
                let et_slice: &[EdgeTypeId] = if et_count <= et_stack.len() {
                    et_stack[..et_count].copy_from_slice(src_ets);
                    &et_stack[..et_count]
                } else {
                    et_heap.extend_from_slice(src_ets);
                    &et_heap[..]
                };

                for &et_id in et_slice {
                    let fanout = self.config.fanout[et_id as usize][hop];
                    if fanout == 0 {
                        continue;
                    }

                    let neighbors = csrs[et_id as usize].neighbors(local_id);
                    if neighbors.is_empty() {
                        continue;
                    }

                    let effective = if let Some(max_deg) = self.config.max_degree {
                        if neighbors.len() > max_deg {
                            &neighbors[..max_deg]
                        } else {
                            neighbors
                        }
                    } else {
                        neighbors
                    };

                    let dst_type = self.dst_types[et_id as usize];
                    let et = et_id as usize;

                    if self.config.replace {
                        self.sample_replace(effective, fanout, src_local_idx, et, dst_type);
                    } else if effective.len() <= fanout {
                        self.take_all(effective, src_local_idx, et, dst_type);
                    } else {
                        self.sample_floyd(effective, fanout, src_local_idx, et, dst_type);
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
    /// Returns `(local index, is_new)`. Caller must ensure `id` is in range
    /// for `node_type` (seeds are range-checked at the loader edge;
    /// [`insert_dst`] checks destinations).
    #[inline(always)]
    fn insert_node(&mut self, id: NodeId, node_type: NodeTypeId) -> (u32, bool) {
        let nt = node_type as usize;
        let (idx, is_new) = self.dedup[nt].probe_or_insert(id, self.node_vecs[nt].len() as u32);
        if is_new {
            self.node_vecs[nt].push(id);
        }
        (idx, is_new)
    }

    /// Insert a destination node, assigning a local index if new.
    /// Returns `None` when `dst_id` is out of range for `dst_type`.
    #[inline(always)]
    fn insert_dst(&mut self, dst_id: NodeId, dst_type: NodeTypeId) -> Option<u32> {
        if (dst_id as usize) >= self.graph.num_nodes(dst_type) {
            return None;
        }
        let (idx, is_new) = self.insert_node(dst_id, dst_type);
        if is_new {
            self.next_frontier.push((dst_type, dst_id, idx));
        }
        Some(idx)
    }

    #[inline]
    fn take_all(&mut self, neighbors: &[NodeId], src_local: u32, et: usize, dst_type: NodeTypeId) {
        for &dst_id in neighbors {
            let Some(dst_local) = self.insert_dst(dst_id, dst_type) else {
                continue;
            };
            self.edge_src_buf[et].push(src_local);
            self.edge_dst_buf[et].push(dst_local);
        }
    }

    #[inline]
    fn sample_replace(
        &mut self,
        neighbors: &[NodeId],
        k: usize,
        src_local: u32,
        et: usize,
        dst_type: NodeTypeId,
    ) {
        let n = neighbors.len() as u64;
        for _ in 0..k {
            let idx = (u64::from(self.rng.next_u32()).wrapping_mul(n) >> 32) as usize;
            let dst_id = neighbors[idx];
            let Some(dst_local) = self.insert_dst(dst_id, dst_type) else {
                continue;
            };
            self.edge_src_buf[et].push(src_local);
            self.edge_dst_buf[et].push(dst_local);
        }
    }

    #[inline]
    fn sample_floyd(
        &mut self,
        neighbors: &[NodeId],
        k: usize,
        src_local: u32,
        et: usize,
        dst_type: NodeTypeId,
    ) {
        let n = neighbors.len();

        if n <= 256 {
            // Stamp reuse is a counter bump, not an O(n) clear per node.
            self.floyd.begin();
            for i in (n - k)..n {
                let s = (i + 1) as u64;
                let x = self.rng.next_u32();
                let j = (u64::from(x).wrapping_mul(s) >> 32) as usize;
                let pick = if self.floyd.test_and_set(j) {
                    j
                } else {
                    self.floyd.test_and_set(i);
                    i
                };
                let Some(dst_local) = self.insert_dst(neighbors[pick], dst_type) else {
                    continue;
                };
                self.edge_src_buf[et].push(src_local);
                self.edge_dst_buf[et].push(dst_local);
            }
        } else {
            self.floyd_set.clear();
            for i in (n - k)..n {
                let s = (i + 1) as u64;
                let x = self.rng.next_u32();
                let j = (u64::from(x).wrapping_mul(s) >> 32) as usize;
                let pick = if self.floyd_set.insert(j) {
                    j
                } else {
                    self.floyd_set.insert(i);
                    i
                };
                let Some(dst_local) = self.insert_dst(neighbors[pick], dst_type) else {
                    continue;
                };
                self.edge_src_buf[et].push(src_local);
                self.edge_dst_buf[et].push(dst_local);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::graph::hetero::HeteroGraph;

    fn build_reddit_graph() -> HeteroGraph {
        let mut votes_edges = Vec::new();
        for user in 0u32..50 {
            for post in 0u32..4 {
                votes_edges.push((user, post));
            }
        }
        let votes_csr = Graph::from_edges(100, &votes_edges, None).unwrap();

        let mut writes_edges = Vec::new();
        for user in 0u32..30 {
            for comment in 0u32..3 {
                writes_edges.push((user, comment));
            }
        }
        let writes_csr = Graph::from_edges(100, &writes_edges, None).unwrap();

        let mut belongs_edges = Vec::new();
        for post in 0u32..100 {
            belongs_edges.push((post, post % 20));
        }
        let belongs_csr = Graph::from_edges(100, &belongs_edges, None).unwrap();

        HeteroGraph::from_parts(
            vec![
                ("user".into(), 100),
                ("post".into(), 100),
                ("comment".into(), 100),
                ("subreddit".into(), 20),
            ],
            vec![
                ("user".into(), "votes".into(), "post".into(), votes_csr),
                ("user".into(), "writes".into(), "comment".into(), writes_csr),
                (
                    "post".into(),
                    "belongs_to".into(),
                    "subreddit".into(),
                    belongs_csr,
                ),
            ],
        )
    }

    #[test]
    fn basic_2hop_sampling() {
        let graph = build_reddit_graph();
        let config = HeteroSamplingConfig {
            fanout: vec![vec![5, 3]; 3],
            replace: false,
            seed: Some(42),
            max_degree: None,
            num_hops: 2,
        };
        let mut sampler = HeteroNeighborSampler::new(&graph, config);
        let user_type = graph.node_type_id("user").unwrap();
        let post_type = graph.node_type_id("post").unwrap();

        let sub = sampler.sample_neighbors(user_type, &[0, 1, 2]);
        assert!(sub.nodes[user_type as usize].len() >= 3);
        assert!(!sub.nodes[post_type as usize].is_empty());
    }

    #[test]
    fn edges_use_local_indices() {
        let graph = build_reddit_graph();
        let config = HeteroSamplingConfig {
            fanout: vec![vec![5]; 3],
            replace: false,
            seed: Some(42),
            max_degree: None,
            num_hops: 1,
        };
        let mut sampler = HeteroNeighborSampler::new(&graph, config);
        let user_type = graph.node_type_id("user").unwrap();
        let post_type = graph.node_type_id("post").unwrap();
        let votes_et = graph.edge_type_id("user", "votes", "post").unwrap();

        let sub = sampler.sample_neighbors(user_type, &[0, 1, 2]);

        // Edge src/dst should be LOCAL indices, not global IDs
        let num_users = sub.nodes[user_type as usize].len();
        let num_posts = sub.nodes[post_type as usize].len();
        for &s in &sub.edge_src_local[votes_et as usize] {
            assert!((s as usize) < num_users, "src local {s} >= {num_users}");
        }
        for &d in &sub.edge_dst_local[votes_et as usize] {
            assert!((d as usize) < num_posts, "dst local {d} >= {num_posts}");
        }
    }

    #[test]
    fn local_indices_map_back_correctly() {
        let graph = build_reddit_graph();
        let config = HeteroSamplingConfig {
            fanout: vec![vec![5]; 3],
            replace: false,
            seed: Some(42),
            max_degree: None,
            num_hops: 1,
        };
        let mut sampler = HeteroNeighborSampler::new(&graph, config);
        let user_type = graph.node_type_id("user").unwrap();
        let post_type = graph.node_type_id("post").unwrap();
        let votes_et = graph.edge_type_id("user", "votes", "post").unwrap();

        let sub = sampler.sample_neighbors(user_type, &[0, 1, 2]);

        // Verify: local_index → global_id → CSR neighbors contains the dst
        for (&src_local, &dst_local) in sub.edge_src_local[votes_et as usize]
            .iter()
            .zip(&sub.edge_dst_local[votes_et as usize])
        {
            let src_global = sub.nodes[user_type as usize][src_local as usize];
            let dst_global = sub.nodes[post_type as usize][dst_local as usize];
            let neighbors = graph.csr(votes_et).neighbors(src_global);
            assert!(
                neighbors.contains(&dst_global),
                "edge {src_global}->{dst_global} not in CSR"
            );
        }
    }

    #[test]
    fn reuse_across_calls() {
        let graph = build_reddit_graph();
        let config = HeteroSamplingConfig {
            fanout: vec![vec![5, 3]; 3],
            replace: false,
            seed: Some(42),
            max_degree: None,
            num_hops: 2,
        };
        let mut sampler = HeteroNeighborSampler::new(&graph, config);
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
        let config = HeteroSamplingConfig {
            fanout: vec![vec![5]; 3],
            replace: false,
            seed: Some(42),
            max_degree: None,
            num_hops: 1,
        };
        let mut sampler = HeteroNeighborSampler::new(&graph, config);
        let user_type = graph.node_type_id("user").unwrap();
        let sub = sampler.sample_neighbors(user_type, &[]);
        assert!(sub.seeds.is_empty());
    }

    #[test]
    fn with_replacement() {
        let graph = build_reddit_graph();
        let config = HeteroSamplingConfig {
            fanout: vec![vec![20]; 3],
            replace: true,
            seed: Some(42),
            max_degree: None,
            num_hops: 1,
        };
        let mut sampler = HeteroNeighborSampler::new(&graph, config);
        let user_type = graph.node_type_id("user").unwrap();
        let votes_et = graph.edge_type_id("user", "votes", "post").unwrap();

        let sub = sampler.sample_neighbors(user_type, &[0]);
        // With replacement, fanout=20, 4 neighbors → 20 edges
        assert_eq!(sub.edge_src_local[votes_et as usize].len(), 20);
    }

    #[test]
    fn deterministic_with_seed() {
        let graph = build_reddit_graph();
        let config = HeteroSamplingConfig {
            fanout: vec![vec![5, 3]; 3],
            replace: false,
            seed: Some(999),
            max_degree: None,
            num_hops: 2,
        };
        let user_type = graph.node_type_id("user").unwrap();

        let mut s1 = HeteroNeighborSampler::new(&graph, config.clone());
        let sub1 = s1.sample_neighbors(user_type, &[0, 1, 2]);

        let mut s2 = HeteroNeighborSampler::new(&graph, config);
        let sub2 = s2.sample_neighbors(user_type, &[0, 1, 2]);

        for i in 0..graph.node_type_count() {
            assert_eq!(sub1.nodes[i], sub2.nodes[i]);
        }
    }

    /// Build a graph with 4 edge types across 3 node types for multi-edge-type tests.
    ///
    /// Node types: author(50), paper(100), venue(20)
    /// Edge types:
    ///   - (author, writes, paper): authors 0..30 each write papers 0..2
    ///   - (paper, cites, paper): papers 0..50 each cite paper (i+1)%100
    ///   - (paper, published_at, venue): papers 0..100 each published at venue i%20
    ///   - (author, reviews, paper): authors 0..20 each review papers 5..8
    fn build_multi_edge_graph() -> HeteroGraph {
        let mut writes_edges = Vec::new();
        for author in 0u32..30 {
            for paper in 0u32..3 {
                writes_edges.push((author, paper));
            }
        }
        let writes_csr = Graph::from_edges(50, &writes_edges, None).unwrap();

        let mut cites_edges = Vec::new();
        for paper in 0u32..50 {
            cites_edges.push((paper, (paper + 1) % 100));
        }
        let cites_csr = Graph::from_edges(100, &cites_edges, None).unwrap();

        let mut published_edges = Vec::new();
        for paper in 0u32..100 {
            published_edges.push((paper, paper % 20));
        }
        let published_csr = Graph::from_edges(100, &published_edges, None).unwrap();

        let mut reviews_edges = Vec::new();
        for author in 0u32..20 {
            for paper in 5u32..8 {
                reviews_edges.push((author, paper));
            }
        }
        let reviews_csr = Graph::from_edges(50, &reviews_edges, None).unwrap();

        HeteroGraph::from_parts(
            vec![
                ("author".into(), 50),
                ("paper".into(), 100),
                ("venue".into(), 20),
            ],
            vec![
                ("author".into(), "writes".into(), "paper".into(), writes_csr),
                ("paper".into(), "cites".into(), "paper".into(), cites_csr),
                (
                    "paper".into(),
                    "published_at".into(),
                    "venue".into(),
                    published_csr,
                ),
                (
                    "author".into(),
                    "reviews".into(),
                    "paper".into(),
                    reviews_csr,
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

        let writes_et = graph.edge_type_id("author", "writes", "paper").unwrap();
        let cites_et = graph.edge_type_id("paper", "cites", "paper").unwrap();
        let published_et = graph
            .edge_type_id("paper", "published_at", "venue")
            .unwrap();
        let reviews_et = graph.edge_type_id("author", "reviews", "paper").unwrap();

        // 2-hop: author seeds -> (writes, reviews) -> paper -> (cites, published_at) -> paper/venue
        let config = HeteroSamplingConfig {
            fanout: vec![vec![3, 2]; 4],
            replace: false,
            seed: Some(42),
            max_degree: None,
            num_hops: 2,
        };
        let mut sampler = HeteroNeighborSampler::new(&graph, config);
        let sub = sampler.sample_neighbors(author_type, &[0, 1, 2]);

        // Seeds are authors
        assert!(sub.nodes[author_type as usize].len() >= 3);

        // Hop 1 produces paper edges via writes and reviews
        let has_writes = !sub.edge_src_local[writes_et as usize].is_empty();
        let has_reviews = !sub.edge_src_local[reviews_et as usize].is_empty();
        assert!(
            has_writes || has_reviews,
            "at least one author->paper edge type should be populated"
        );

        // Hop 2 explores from papers via cites and published_at
        let has_cites = !sub.edge_src_local[cites_et as usize].is_empty();
        let has_published = !sub.edge_src_local[published_et as usize].is_empty();
        assert!(
            has_cites || has_published,
            "at least one paper->X edge type should be populated in hop 2"
        );

        // Papers should be discovered
        assert!(
            !sub.nodes[paper_type as usize].is_empty(),
            "papers should be sampled"
        );

        // Venues should be discovered in hop 2
        assert!(
            !sub.nodes[venue_type as usize].is_empty(),
            "venues should be sampled in hop 2"
        );
    }

    #[test]
    fn test_seed_type_filtering() {
        let graph = build_reddit_graph();
        let user_type = graph.node_type_id("user").unwrap();
        let post_type = graph.node_type_id("post").unwrap();

        // Sample from user seeds
        let config = HeteroSamplingConfig {
            fanout: vec![vec![3]; 3],
            replace: false,
            seed: Some(42),
            max_degree: None,
            num_hops: 1,
        };
        let mut sampler = HeteroNeighborSampler::new(&graph, config.clone());
        let sub_user = sampler.sample_neighbors(user_type, &[0, 1, 2]);
        assert_eq!(sub_user.seed_type, user_type);
        assert_eq!(sub_user.seeds, vec![0, 1, 2]);
        // First 3 entries in user node vec should be our seeds
        assert_eq!(sub_user.nodes[user_type as usize][0], 0);
        assert_eq!(sub_user.nodes[user_type as usize][1], 1);
        assert_eq!(sub_user.nodes[user_type as usize][2], 2);

        // Sample from post seeds
        let mut sampler2 = HeteroNeighborSampler::new(&graph, config);
        let sub_post = sampler2.sample_neighbors(post_type, &[10, 20]);
        assert_eq!(sub_post.seed_type, post_type);
        assert_eq!(sub_post.seeds, vec![10, 20]);
        // First 2 entries in post node vec should be our seeds
        assert_eq!(sub_post.nodes[post_type as usize][0], 10);
        assert_eq!(sub_post.nodes[post_type as usize][1], 20);
        // User type should not contain seeds (users are NOT seeds here)
        // but may contain discovered nodes through edges
    }

    #[test]
    fn test_fanout_per_edge_type() {
        let graph = build_reddit_graph();
        let user_type = graph.node_type_id("user").unwrap();
        let votes_et = graph.edge_type_id("user", "votes", "post").unwrap();
        let writes_et = graph.edge_type_id("user", "writes", "comment").unwrap();

        // Set different fanouts per edge type:
        // votes (et 0): fanout 2, writes (et 1): fanout 1, belongs_to (et 2): fanout 0
        let config = HeteroSamplingConfig {
            fanout: vec![vec![2], vec![1], vec![0]],
            replace: false,
            seed: Some(42),
            max_degree: None,
            num_hops: 1,
        };
        let mut sampler = HeteroNeighborSampler::new(&graph, config);

        // Use user 0 which has 4 vote-neighbors and 3 write-neighbors
        let sub = sampler.sample_neighbors(user_type, &[0]);

        // votes fanout=2: user 0 has 4 neighbors, sample 2
        assert_eq!(
            sub.edge_src_local[votes_et as usize].len(),
            2,
            "votes should have exactly 2 edges (fanout=2)"
        );

        // writes fanout=1: user 0 has 3 neighbors, sample 1
        assert_eq!(
            sub.edge_src_local[writes_et as usize].len(),
            1,
            "writes should have exactly 1 edge (fanout=1)"
        );

        // belongs_to fanout=0: no edges should be sampled at all
        let belongs_et = graph
            .edge_type_id("post", "belongs_to", "subreddit")
            .unwrap();
        assert!(
            sub.edge_src_local[belongs_et as usize].is_empty(),
            "belongs_to should have 0 edges (fanout=0)"
        );
    }

    #[test]
    fn test_hetero_with_replacement() {
        let graph = build_reddit_graph();
        let user_type = graph.node_type_id("user").unwrap();
        let votes_et = graph.edge_type_id("user", "votes", "post").unwrap();
        let writes_et = graph.edge_type_id("user", "writes", "comment").unwrap();

        // With replacement, fanout=50, but user 0 only has 4 vote-neighbors
        // and 3 write-neighbors. We should get exactly 50 edges per type,
        // meaning many duplicates.
        let config = HeteroSamplingConfig {
            fanout: vec![vec![50], vec![50], vec![0]],
            replace: true,
            seed: Some(42),
            max_degree: None,
            num_hops: 1,
        };
        let mut sampler = HeteroNeighborSampler::new(&graph, config);
        let sub = sampler.sample_neighbors(user_type, &[0]);

        // With replacement always produces exactly fanout edges
        assert_eq!(
            sub.edge_src_local[votes_et as usize].len(),
            50,
            "with replacement should produce exactly fanout edges for votes"
        );
        assert_eq!(
            sub.edge_src_local[writes_et as usize].len(),
            50,
            "with replacement should produce exactly fanout edges for writes"
        );

        // Since user 0 has only 4 unique vote-neighbors, with 50 samples
        // there MUST be duplicate dst local indices.
        let dst_locals: Vec<u32> = sub.edge_dst_local[votes_et as usize].clone();
        let unique: FxHashSet<u32> = dst_locals.iter().copied().collect();
        assert!(
            unique.len() <= 4,
            "only 4 unique vote neighbors exist, but got {} unique dst locals",
            unique.len()
        );
        assert!(
            dst_locals.len() > unique.len(),
            "with replacement should produce duplicate neighbors"
        );
    }

    #[test]
    fn test_hetero_high_degree_node() {
        // Create a graph where node 0 has 1500 neighbors of one edge type
        let mut edges = Vec::new();
        for dst in 0u32..1500 {
            edges.push((0u32, dst));
        }
        let csr = Graph::from_edges(2000, &edges, None).unwrap();

        let graph = HeteroGraph::from_parts(
            vec![("A".into(), 2000), ("B".into(), 2000)],
            vec![("A".into(), "connects".into(), "B".into(), csr)],
        );

        let a_type = graph.node_type_id("A").unwrap();
        let connects_et = graph.edge_type_id("A", "connects", "B").unwrap();

        // Sample with small fanout from the high-degree node
        let config = HeteroSamplingConfig {
            fanout: vec![vec![5]],
            replace: false,
            seed: Some(42),
            max_degree: None,
            num_hops: 1,
        };
        let mut sampler = HeteroNeighborSampler::new(&graph, config);
        let sub = sampler.sample_neighbors(a_type, &[0]);

        // Should sample exactly 5 edges without panic
        assert_eq!(
            sub.edge_src_local[connects_et as usize].len(),
            5,
            "should sample exactly fanout=5 edges from high-degree node"
        );

        // All dst local indices should be valid
        let num_b = sub.nodes[graph.node_type_id("B").unwrap() as usize].len();
        for &d in &sub.edge_dst_local[connects_et as usize] {
            assert!(
                (d as usize) < num_b,
                "dst local index out of bounds: {d} >= {num_b}"
            );
        }

        // Also test with max_degree limiting the effective neighborhood
        let config_limited = HeteroSamplingConfig {
            fanout: vec![vec![5]],
            replace: false,
            seed: Some(42),
            max_degree: Some(100),
            num_hops: 1,
        };
        let mut sampler2 = HeteroNeighborSampler::new(&graph, config_limited);
        let sub2 = sampler2.sample_neighbors(a_type, &[0]);
        assert_eq!(
            sub2.edge_src_local[connects_et as usize].len(),
            5,
            "should still sample 5 edges with max_degree capping"
        );
        // With max_degree=100, sampled dst global IDs should all be < 100
        let b_type = graph.node_type_id("B").unwrap();
        for &d in &sub2.edge_dst_local[connects_et as usize] {
            let global_id = sub2.nodes[b_type as usize][d as usize];
            assert!(
                global_id < 100,
                "with max_degree=100, sampled neighbor {global_id} should be < 100"
            );
        }
    }

    #[test]
    fn test_hetero_disconnected_type() {
        // Build a graph where one edge type has no edges at all
        let mut writes_edges = Vec::new();
        for user in 0u32..10 {
            writes_edges.push((user, user));
        }
        let writes_csr = Graph::from_edges(20, &writes_edges, None).unwrap();

        // Empty edge type: no edges
        let empty_csr = Graph::from_edges(20, &[], None).unwrap();

        let graph = HeteroGraph::from_parts(
            vec![("user".into(), 20), ("post".into(), 20), ("tag".into(), 20)],
            vec![
                ("user".into(), "writes".into(), "post".into(), writes_csr),
                ("user".into(), "tags".into(), "tag".into(), empty_csr),
            ],
        );

        let user_type = graph.node_type_id("user").unwrap();
        let tags_et = graph.edge_type_id("user", "tags", "tag").unwrap();
        let writes_et = graph.edge_type_id("user", "writes", "post").unwrap();

        let config = HeteroSamplingConfig {
            fanout: vec![vec![5]; 2],
            replace: false,
            seed: Some(42),
            max_degree: None,
            num_hops: 1,
        };
        let mut sampler = HeteroNeighborSampler::new(&graph, config);
        let sub = sampler.sample_neighbors(user_type, &[0, 1, 2]);

        // The empty edge type should produce no edges and not crash
        assert!(
            sub.edge_src_local[tags_et as usize].is_empty(),
            "disconnected edge type should have no edges"
        );
        assert!(
            sub.edge_dst_local[tags_et as usize].is_empty(),
            "disconnected edge type should have no dst edges"
        );

        // Tag nodes should not be discovered
        let tag_type = graph.node_type_id("tag").unwrap();
        assert!(
            sub.nodes[tag_type as usize].is_empty(),
            "tag nodes should not appear since no edges exist"
        );

        // The other edge type should still work fine
        assert!(
            !sub.edge_src_local[writes_et as usize].is_empty(),
            "writes edge type should have edges"
        );
    }

    #[test]
    fn test_hetero_single_node_type() {
        // Degenerate case: one node type with self-referencing edges (like homogeneous)
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
        let link_et = graph.edge_type_id("node", "link", "node").unwrap();

        let config = HeteroSamplingConfig {
            fanout: vec![vec![2, 2]],
            replace: false,
            seed: Some(42),
            max_degree: None,
            num_hops: 2,
        };
        let mut sampler = HeteroNeighborSampler::new(&graph, config);
        let sub = sampler.sample_neighbors(node_type, &[0, 5, 10]);

        // Seeds should be present
        assert!(sub.nodes[node_type as usize].len() >= 3);
        assert_eq!(sub.nodes[node_type as usize][0], 0);
        assert_eq!(sub.nodes[node_type as usize][1], 5);
        assert_eq!(sub.nodes[node_type as usize][2], 10);

        // Should have some link edges
        assert!(
            !sub.edge_src_local[link_et as usize].is_empty(),
            "self-referencing edge type should produce edges"
        );

        // All local indices should be into the same node vec
        let num_nodes = sub.nodes[node_type as usize].len();
        for &s in &sub.edge_src_local[link_et as usize] {
            assert!(
                (s as usize) < num_nodes,
                "src local {s} out of bounds (num_nodes={num_nodes})"
            );
        }
        for &d in &sub.edge_dst_local[link_et as usize] {
            assert!(
                (d as usize) < num_nodes,
                "dst local {d} out of bounds (num_nodes={num_nodes})"
            );
        }

        // Verify edges are valid by checking CSR
        for (&src_local, &dst_local) in sub.edge_src_local[link_et as usize]
            .iter()
            .zip(&sub.edge_dst_local[link_et as usize])
        {
            let src_global = sub.nodes[node_type as usize][src_local as usize];
            let dst_global = sub.nodes[node_type as usize][dst_local as usize];
            let neighbors = graph.csr(link_et).neighbors(src_global);
            assert!(
                neighbors.contains(&dst_global),
                "edge {src_global}->{dst_global} not in CSR"
            );
        }
    }

    #[test]
    fn test_hetero_large_batch() {
        let graph = build_reddit_graph();
        let user_type = graph.node_type_id("user").unwrap();
        let post_type = graph.node_type_id("post").unwrap();
        let votes_et = graph.edge_type_id("user", "votes", "post").unwrap();

        // 100 seed nodes
        let seeds: Vec<u32> = (0u32..100).collect();

        let config = HeteroSamplingConfig {
            fanout: vec![vec![5, 3]; 3],
            replace: false,
            seed: Some(42),
            max_degree: None,
            num_hops: 2,
        };
        let mut sampler = HeteroNeighborSampler::new(&graph, config);
        let sub = sampler.sample_neighbors(user_type, &seeds);

        // All 100 seeds should be in the output
        assert!(
            sub.nodes[user_type as usize].len() >= 100,
            "should have at least 100 user nodes (the seeds), got {}",
            sub.nodes[user_type as usize].len()
        );
        assert_eq!(sub.seeds.len(), 100);

        // Verify first 100 entries are our seeds (deduplication may reorder if
        // seeds are unique, which they are here)
        let user_nodes = &sub.nodes[user_type as usize];
        for i in 0..100 {
            assert_eq!(
                user_nodes[i], seeds[i],
                "seed {} should be at position {}",
                seeds[i], i
            );
        }

        // Should have discovered posts
        assert!(
            !sub.nodes[post_type as usize].is_empty(),
            "should have sampled some post nodes"
        );

        // Should have edges
        assert!(
            !sub.edge_src_local[votes_et as usize].is_empty(),
            "should have votes edges from 100 seeds"
        );

        // All local indices should be valid
        let num_users = sub.nodes[user_type as usize].len();
        let num_posts = sub.nodes[post_type as usize].len();
        for &s in &sub.edge_src_local[votes_et as usize] {
            assert!(
                (s as usize) < num_users,
                "src local {s} out of bounds (num_users={num_users})"
            );
        }
        for &d in &sub.edge_dst_local[votes_et as usize] {
            assert!(
                (d as usize) < num_posts,
                "dst local {d} out of bounds (num_posts={num_posts})"
            );
        }
    }
}
