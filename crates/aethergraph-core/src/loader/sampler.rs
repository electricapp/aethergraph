//! Zero-allocation neighborhood sampling for GNN training.
//!
//! All buffers are pre-allocated at sampler construction and reused across
//! `sample_neighbors` calls via `clear()`. Local node indices are assigned
//! during sampling — no post-sort, no binary search.

use crate::graph::{Graph, NodeId};
use crate::internal::telemetry::{SamplingTelemetry, SamplingTimer};

/// Temporal sampling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalStrategy {
    /// Sample uniformly from edges with timestamp < node time.
    Uniform,
    /// Take the k most recent edges with timestamp < node time.
    Last,
}

/// Type of subgraph to extract during sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubgraphType {
    /// Default mode: Return all sampled edges as-is.
    /// This includes edges from the sampling process.
    #[default]
    Directional,

    /// Induced mode: Return only edges where both endpoints are in the sampled node set.
    /// Filters out edges that go to nodes outside the subgraph.
    Induced,

    /// Bidirectional mode: Add reverse edges for all sampled edges.
    /// Useful for undirected graph models.
    Bidirectional,
}

/// Ultra-fast wyrand PRNG - simpler and faster than xoshiro for our use case.
/// Based on wyhash: https://github.com/wangyi-fudan/wyhash
#[derive(Clone)]
struct WyRand {
    state: u64,
}

impl WyRand {
    #[inline(always)]
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0xa0761d6478bd642f);
        let t = (self.state as u128).wrapping_mul((self.state ^ 0xe7037ed1a0b428db) as u128);
        ((t >> 64) ^ t) as u64
    }

    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}
/// Fast approximate -ln(u) for a random u64 interpreted as U(0,1).
///
/// Extracts the IEEE 754 exponent to get -log2(u), then scales by ln(2).
/// Monotone (preserves ordering) which is all Efraimidis-Spirakis needs.
/// ~3-4x faster than `f64::ln()` — avoids the iterative polynomial evaluation.
#[inline(always)]
fn fast_neg_ln_u64(bits: u64) -> f64 {
    // Reinterpret bits as f64 in [1.0, 2.0) by setting exponent to 1023
    // then use: -ln(u) where u = mantissa * 2^(-64)
    // Approximation: -ln(u) ≈ (64 - leading_zeros) * ln(2) + correction
    // Using the IEEE trick: cast to f64 and extract exponent directly.
    let u = (bits >> 11) | 0x3FF0_0000_0000_0000; // [1.0, 2.0)
    let f = f64::from_bits(u) - 1.0; // [0.0, 1.0)
    // -ln(f) for f near 0 is large; for f near 1 is near 0
    // Use: -ln(1 - x) ≈ x + x²/2 for small x ... but this loses for large x
    // Simpler: count leading zeros for the integer part, use mantissa for fractional
    let lz = bits.leading_zeros() as f64;
    // -ln(u) ≈ (lz + 1 - frac) * ln(2), where frac is the mantissa contribution
    (lz + 1.0 - f) * std::f64::consts::LN_2
}

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;
use tracing::trace;

/// Configuration for neighborhood sampling
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Number of neighbors to sample per node at each hop
    /// For example, [25, 10] means sample 25 neighbors at hop 1 and 10 at hop 2
    pub fanout: Vec<usize>,

    /// Whether to sample with replacement
    pub replace: bool,

    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Maximum degree to process for hub nodes (e.g., r/AskReddit with millions of edges)
    /// If a node has more than this many neighbors, use reservoir sampling without
    /// loading the full edge list into memory. This prevents OOM on power-law graphs.
    /// Default: None (no capping)
    pub max_degree: Option<usize>,

    /// Whether to use cumulative sampling (PyG-style) vs frontier-only (pure GraphSAGE).
    ///
    /// - `true` (default): Each hop samples from ALL nodes seen so far (PyG behavior)
    ///   Results in larger subgraphs. Example: hop1 samples from seeds, hop2 samples
    ///   from seeds + hop1 neighbors.
    ///
    /// - `false`: Each hop only samples from the new frontier (pure GraphSAGE)
    ///   Results in smaller subgraphs. Example: hop1 samples from seeds, hop2 samples
    ///   only from hop1 neighbors.
    pub cumulative: bool,

    /// Whether to use edge weights for weighted sampling.
    /// When true, neighbors are sampled proportionally to their edge weights.
    /// Requires the graph to have weights loaded.
    pub weighted: bool,

    /// Type of subgraph to extract.
    /// - Directional (default): Return edges as sampled
    /// - Induced: Only edges where both endpoints are in sampled nodes
    /// - Bidirectional: Add reverse edges
    pub subgraph_type: SubgraphType,

    /// Whether to track global edge IDs (for e_id in PyG Data).
    /// Default: true (matches PyG behavior).
    /// Set to false for ~10-15% speedup if you don't need edge features.
    pub track_edge_ids: bool,

    /// Temporal sampling strategy. When set, only edges with timestamp < node time
    /// are eligible for sampling. Requires `Graph::set_timestamps()`.
    pub temporal_strategy: Option<TemporalStrategy>,

    /// Whether to produce disjoint subgraphs per seed (no node dedup across seeds).
    /// Each seed gets its own isolated subgraph. The output includes a `batch` vector
    /// mapping each node to its seed index.
    pub disjoint: bool,

    /// Optional telemetry collector (opt-in, zero overhead if None)
    pub telemetry: Option<Arc<SamplingTelemetry>>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            fanout: vec![25, 10],
            replace: false,
            seed: None,
            max_degree: Some(10_000), // Cap at 10k neighbors for hub nodes
            cumulative: true,         // PyG-style cumulative sampling by default
            weighted: false,          // Uniform sampling by default
            subgraph_type: SubgraphType::Directional, // Default: edges as sampled
            track_edge_ids: true,     // Match PyG: always track e_id by default
            temporal_strategy: None,  // No temporal filtering by default
            disjoint: false,          // Shared node dedup by default
            telemetry: None,          // Opt-in telemetry
        }
    }
}

/// A neighborhood sampler for GNN training.
///
/// All internal buffers are pre-allocated at construction and reused
/// across `sample_neighbors` calls. Zero allocations in the hot path.
///
/// Uses either a generation-tagged direct array (for graphs < 1M nodes) or
/// FxHashMap (for large graphs) for node dedup. The direct array is O(1) with
/// no hashing but costs 8 bytes/node. FxHashMap is better when sampled nodes
/// are a tiny fraction of the graph (e.g., 12K / 10M = 0.1%).
pub struct NeighborSampler<'a> {
    graph: &'a Graph,
    config: SamplingConfig,
    rng: WyRand,
    /// Direct array path (graphs < DIRECT_ARRAY_THRESHOLD nodes).
    node_gen: Vec<u32>,
    node_local_idx: Vec<u32>,
    current_gen: u32,
    /// HashMap path (large graphs).
    local_index: FxHashMap<NodeId, u32>,
    /// Whether to use direct array (true) or HashMap (false).
    use_direct: bool,
    /// Nodes in discovery order.
    node_vec: Vec<NodeId>,
    /// Edge source nodes (global IDs, for subgraph filtering).
    edge_src_buf: Vec<NodeId>,
    /// Edge destination nodes (global IDs).
    edge_dst_buf: Vec<NodeId>,
    /// Global edge IDs (position in CSR edges array).
    edge_ids_buf: Vec<u64>,
    /// Double-buffered frontiers.
    frontier: Vec<NodeId>,
    next_frontier: Vec<NodeId>,
    /// Reusable buffer for weighted sampling results.
    sample_buf: Vec<(NodeId, usize)>,
    /// Reusable Floyd bitmap (small degree <= 256).
    floyd_bitmap: [bool; 256],
    /// Reusable Floyd set (large degree > 256).
    seen_set: FxHashSet<usize>,
    /// Temporal: per-node time constraints (parallel to node_vec).
    node_times: Vec<f64>,
    /// Temporal: filtered (csr_index, timestamp) pairs for select-k.
    temporal_filtered: Vec<(usize, f64)>,
    /// Weighted no-replace: reusable key buffer to avoid per-call allocation.
    weighted_keys: Vec<(f64, usize)>,
}

impl<'a> NeighborSampler<'a> {
    /// Creates a new neighbor sampler.
    pub fn new(graph: &'a Graph, config: SamplingConfig) -> Self {
        let seed = config.seed.unwrap_or_else(|| {
            // Use system entropy for seed if not provided
            rand::random::<u64>()
        });
        let rng = WyRand::new(seed);

        // Pre-allocate with typical sizes
        let max_fanout = config.fanout.iter().max().copied().unwrap_or(25);

        // Use direct array when the expected sample is a large fraction of the graph.
        // Direct array: O(1) lookup, but 8 bytes/node upfront → cache pressure for small samples.
        // HashMap: O(1) amortized, grows with sample size → better when sample << graph.
        //
        // Heuristic: estimate max sample size from fanout and a typical batch.
        // If sample/graph > 1%, direct array wins (sequential gen-tag check).
        // Otherwise HashMap avoids polluting caches with 8MB of mostly-untouched arrays.
        let num_nodes = graph.num_nodes();
        let estimated_sample: usize = config
            .fanout
            .iter()
            .fold(128usize, |acc, &f| acc.saturating_mul(f).min(num_nodes));
        let use_direct = num_nodes <= 100_000
            || (estimated_sample as f64 / num_nodes as f64) > 0.01;
        Self {
            graph,
            config,
            rng,
            node_gen: if use_direct { vec![0u32; num_nodes] } else { Vec::new() },
            node_local_idx: if use_direct { vec![0u32; num_nodes] } else { Vec::new() },
            current_gen: 0,
            local_index: FxHashMap::with_capacity_and_hasher(512, Default::default()),
            use_direct,
            node_vec: Vec::with_capacity(512),
            edge_src_buf: Vec::with_capacity(2048),
            edge_dst_buf: Vec::with_capacity(2048),
            edge_ids_buf: Vec::with_capacity(2048),
            frontier: Vec::with_capacity(512),
            next_frontier: Vec::with_capacity(4096),
            sample_buf: Vec::with_capacity(max_fanout),
            floyd_bitmap: [false; 256],
            seen_set: FxHashSet::with_capacity_and_hasher(max_fanout * 2, Default::default()),
            node_times: Vec::new(),
            temporal_filtered: Vec::with_capacity(max_fanout),
            weighted_keys: Vec::with_capacity(256),
        }
    }

    #[inline(always)]
    fn insert_node(&mut self, id: NodeId) -> u32 {
        if self.use_direct {
            let i = id as usize;
            if self.node_gen[i] == self.current_gen {
                self.node_local_idx[i]
            } else {
                let idx = self.node_vec.len() as u32;
                self.node_gen[i] = self.current_gen;
                self.node_local_idx[i] = idx;
                self.node_vec.push(id);
                idx
            }
        } else {
            let next_idx = self.node_vec.len() as u32;
            match self.local_index.entry(id) {
                std::collections::hash_map::Entry::Occupied(e) => *e.get(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(next_idx);
                    self.node_vec.push(id);
                    next_idx
                }
            }
        }
    }

    #[inline(always)]
    fn insert_node_frontier(&mut self, id: NodeId) -> bool {
        if self.use_direct {
            let i = id as usize;
            if self.node_gen[i] == self.current_gen {
                return false;
            }
            let idx = self.node_vec.len() as u32;
            self.node_gen[i] = self.current_gen;
            self.node_local_idx[i] = idx;
            self.node_vec.push(id);
            self.next_frontier.push(id);
            true
        } else {
            let next_idx = self.node_vec.len() as u32;
            match self.local_index.entry(id) {
                std::collections::hash_map::Entry::Occupied(_) => false,
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(next_idx);
                    self.node_vec.push(id);
                    self.next_frontier.push(id);
                    true
                }
            }
        }
    }

    /// Sample k-hop neighborhoods for a batch of seed nodes.
    ///
    /// Returns a subgraph containing all sampled nodes and their edges.
    pub fn sample_neighbors(&mut self, seeds: &[NodeId]) -> SampledSubgraph {
        self.sample_neighbors_inner(seeds, None)
    }

    /// Sample k-hop neighborhoods with temporal constraints.
    ///
    /// Each seed has an associated time; only edges with timestamp < seed time are eligible.
    /// Requires `temporal_strategy` set in config and `Graph::set_timestamps()`.
    pub fn sample_neighbors_temporal(
        &mut self,
        seeds: &[NodeId],
        input_times: &[f64],
    ) -> SampledSubgraph {
        assert_eq!(
            seeds.len(),
            input_times.len(),
            "seeds and input_times must have same length"
        );
        self.sample_neighbors_inner(seeds, Some(input_times))
    }

    /// Sample each seed independently (no node dedup across seeds).
    ///
    /// Returns a combined subgraph with a `batch` vector mapping each node to its seed index.
    /// Local edge indices are offset per seed so each seed's subgraph is isolated.
    ///
    /// Single-pass: resets dedup state per seed without rebuilding the sampler.
    #[inline]
    pub fn sample_neighbors_disjoint(
        &mut self,
        seeds: &[NodeId],
        input_times: Option<&[f64]>,
    ) -> SampledSubgraph {
        let num_hops = self.config.fanout.len();
        let is_temporal = self.config.temporal_strategy.is_some();

        // Pre-allocate output buffers: estimate ~50 nodes/edges per seed
        let est = seeds.len() * 50;
        let mut all_nodes = Vec::with_capacity(est);
        let mut all_edge_src = Vec::with_capacity(est);
        let mut all_edge_dst = Vec::with_capacity(est);
        let mut all_edge_ids = Vec::with_capacity(if self.config.track_edge_ids { est } else { 0 });
        let mut batch = Vec::with_capacity(est);
        let mut all_src_local = Vec::with_capacity(est);
        let mut all_dst_local = Vec::with_capacity(est);
        let mut num_sampled_nodes = Vec::new();
        let mut num_sampled_edges = Vec::new();

        for (seed_idx, &seed) in seeds.iter().enumerate() {
            // Reset dedup state per seed (cheap: no dealloc, just counter bump or clear)
            if self.use_direct {
                self.current_gen = self.current_gen.wrapping_add(1);
                if self.current_gen == 0 {
                    self.node_gen.fill(0);
                    self.current_gen = 1;
                }
            } else {
                self.local_index.clear();
            }
            self.node_vec.clear();
            self.edge_src_buf.clear();
            self.edge_dst_buf.clear();
            self.edge_ids_buf.clear();
            self.frontier.clear();
            self.next_frontier.clear();
            self.node_times.clear();

            // Register single seed
            self.insert_node(seed);
            self.frontier.push(seed);
            if let Some(times) = input_times {
                self.node_times.push(times[seed_idx]);
            }

            // Run hop loop inline (same logic as sample_neighbors_inner)
            for hop in 0..num_hops {
                let sample_size = self.config.fanout[hop];
                self.next_frontier.clear();

                let frontier_len = self.frontier.len();
                for fi in 0..frontier_len {
                    let node = self.frontier[fi];
                    let neighbors = self.graph.neighbors(node);
                    if neighbors.is_empty() {
                        continue;
                    }
                    if is_temporal {
                        self.sample_temporal(node, neighbors, sample_size);
                    } else {
                        self.sample_normal(node, neighbors, sample_size);
                    }
                }

                if self.config.cumulative {
                    self.frontier.extend_from_slice(&self.next_frontier);
                } else {
                    std::mem::swap(&mut self.frontier, &mut self.next_frontier);
                }
            }

            // Compute local edge indices and offset into combined output
            let node_offset = all_nodes.len() as u32;

            // Build local_index for this seed's subgraph
            let seed_local: FxHashMap<NodeId, u32> = self.node_vec
                .iter()
                .enumerate()
                .map(|(idx, &id)| (id, idx as u32))
                .collect();

            for &src in &self.edge_src_buf {
                all_src_local.push(seed_local[&src] + node_offset);
            }
            for &dst in &self.edge_dst_buf {
                all_dst_local.push(seed_local[&dst] + node_offset);
            }

            all_nodes.extend_from_slice(&self.node_vec);
            all_edge_src.extend_from_slice(&self.edge_src_buf);
            all_edge_dst.extend_from_slice(&self.edge_dst_buf);
            all_edge_ids.extend_from_slice(&self.edge_ids_buf);

            let node_count = self.node_vec.len();
            batch.resize(batch.len() + node_count, seed_idx as u32);

            if seed_idx == 0 {
                num_sampled_nodes = self.next_frontier.iter().map(|_| 0).collect();
                num_sampled_edges = Vec::new();
                // Approximate: just use counts from first seed
            }
        }

        SampledSubgraph {
            nodes: all_nodes,
            edge_src: all_edge_src,
            edge_dst: all_edge_dst,
            edge_ids: all_edge_ids,
            seeds: seeds.to_vec(),
            num_sampled_nodes,
            num_sampled_edges,
            local_index: FxHashMap::default(),
            batch: Some(batch),
            precomputed_local: Some((all_src_local, all_dst_local)),
        }
    }

    #[inline]
    fn sample_neighbors_inner(
        &mut self,
        seeds: &[NodeId],
        input_times: Option<&[f64]>,
    ) -> SampledSubgraph {
        let _timer = SamplingTimer::new();
        let num_hops = self.config.fanout.len();
        let is_temporal = self.config.temporal_strategy.is_some();
        trace!(
            "Sampling {}-hop neighborhood for {} seeds (temporal={})",
            num_hops,
            seeds.len(),
            is_temporal,
        );

        if self.use_direct {
            self.current_gen = self.current_gen.wrapping_add(1);
            if self.current_gen == 0 {
                self.node_gen.fill(0);
                self.current_gen = 1;
            }
        } else {
            self.local_index.clear();
        }
        self.node_vec.clear();
        self.edge_src_buf.clear();
        self.edge_dst_buf.clear();
        self.edge_ids_buf.clear();
        self.frontier.clear();
        self.next_frontier.clear();
        self.node_times.clear();

        // Register seeds with local indices
        for &seed in seeds {
            self.insert_node(seed);
            self.frontier.push(seed);
        }

        // Initialize seed times for temporal sampling
        if let Some(times) = input_times {
            self.node_times.extend_from_slice(times);
        }

        // Track per-hop sampling stats (for PyG compatibility)
        let mut num_sampled_nodes = Vec::with_capacity(num_hops);
        let mut num_sampled_edges = Vec::with_capacity(num_hops);

        // Sample neighbors hop by hop
        for hop in 0..num_hops {
            let edges_before = self.edge_src_buf.len();
            let sample_size = self.config.fanout[hop];
            self.next_frontier.clear();

            let frontier_len = self.frontier.len();
            for fi in 0..frontier_len {
                let node = self.frontier[fi];
                let neighbors = self.graph.neighbors(node);
                if neighbors.is_empty() {
                    continue;
                }

                if is_temporal {
                    self.sample_temporal(node, neighbors, sample_size);
                } else {
                    self.sample_normal(node, neighbors, sample_size);
                }
            }

            // Record per-hop stats (for PyG compatibility)
            num_sampled_nodes.push(self.next_frontier.len());
            num_sampled_edges.push(self.edge_src_buf.len() - edges_before);

            // Update frontier based on sampling mode (double-buffer swap)
            if self.config.cumulative {
                // PyG-style: accumulate all nodes for next hop
                self.frontier.extend_from_slice(&self.next_frontier);
            } else {
                // Pure GraphSAGE: only use new frontier
                std::mem::swap(&mut self.frontier, &mut self.next_frontier);
            }
        }

        // Move buffers out, replacing with fresh capacity for reuse
        let mut node_vec = Vec::with_capacity(512);
        std::mem::swap(&mut self.node_vec, &mut node_vec);

        let mut edge_src = Vec::with_capacity(2048);
        std::mem::swap(&mut self.edge_src_buf, &mut edge_src);

        let mut edge_dst = Vec::with_capacity(2048);
        std::mem::swap(&mut self.edge_dst_buf, &mut edge_dst);

        let mut edge_ids = Vec::with_capacity(if self.config.track_edge_ids { 2048 } else { 0 });
        std::mem::swap(&mut self.edge_ids_buf, &mut edge_ids);

        // Build local_index map for the subgraph from the generation arrays.
        // This is O(N) where N = sampled nodes (~12K), not graph nodes.
        let local_index: FxHashMap<NodeId, u32> = node_vec
            .iter()
            .enumerate()
            .map(|(idx, &id)| (id, idx as u32))
            .collect();

        // Apply subgraph_type post-processing
        let (edge_src, edge_dst, edge_ids) = match self.config.subgraph_type {
            SubgraphType::Directional => {
                // Default: return edges as-is
                (edge_src, edge_dst, edge_ids)
            }
            SubgraphType::Induced => {
                // Filter to only edges where both endpoints are in the node set.
                // local_index serves as the node set — O(1) lookup.
                let mut new_src = Vec::with_capacity(edge_src.len());
                let mut new_dst = Vec::with_capacity(edge_dst.len());
                let mut new_ids = if self.config.track_edge_ids {
                    Vec::with_capacity(edge_ids.len())
                } else {
                    Vec::new()
                };

                for i in 0..edge_src.len() {
                    if local_index.contains_key(&edge_src[i])
                        && local_index.contains_key(&edge_dst[i])
                    {
                        new_src.push(edge_src[i]);
                        new_dst.push(edge_dst[i]);
                        if self.config.track_edge_ids {
                            new_ids.push(edge_ids[i]);
                        }
                    }
                }

                (new_src, new_dst, new_ids)
            }
            SubgraphType::Bidirectional => {
                // Add reverse edges
                let mut new_src = Vec::with_capacity(edge_src.len() * 2);
                let mut new_dst = Vec::with_capacity(edge_dst.len() * 2);
                let mut new_ids = if self.config.track_edge_ids {
                    Vec::with_capacity(edge_ids.len() * 2)
                } else {
                    Vec::new()
                };

                // Original edges
                new_src.extend_from_slice(&edge_src);
                new_dst.extend_from_slice(&edge_dst);
                if self.config.track_edge_ids {
                    new_ids.extend_from_slice(&edge_ids);
                }

                // Reverse edges (edge_id for reverse is same as forward)
                new_src.extend_from_slice(&edge_dst);
                new_dst.extend_from_slice(&edge_src);
                if self.config.track_edge_ids {
                    new_ids.extend_from_slice(&edge_ids);
                }

                (new_src, new_dst, new_ids)
            }
        };

        let subgraph = SampledSubgraph {
            nodes: node_vec,
            edge_src,
            edge_dst,
            edge_ids,
            seeds: seeds.to_vec(),
            num_sampled_nodes,
            num_sampled_edges,
            local_index,
            batch: None,
            precomputed_local: None,
        };

        // Record telemetry if enabled (opt-in, zero overhead if None)
        if let Some(ref telemetry) = self.config.telemetry {
            telemetry.record_sample(
                subgraph.num_nodes() as u64,
                subgraph.num_edges() as u64,
                _timer.elapsed(),
            );
        }

        subgraph
    }

    /// Temporal sampling for a single node: filter by timestamp, sample, emit.
    /// Uses sample_buf and temporal_filtered as scratch (pre-allocated on sampler).
    #[inline]
    fn sample_temporal(&mut self, node: NodeId, neighbors: &[NodeId], sample_size: usize) {
        let ts = match self.graph.neighbor_timestamps(node) {
            Some(ts) => ts,
            None => return,
        };
        let node_time = if !self.node_times.is_empty() {
            let local_idx = if self.use_direct {
                self.node_local_idx[node as usize]
            } else {
                self.local_index[&node]
            };
            self.node_times[local_idx as usize]
        } else {
            f64::INFINITY
        };

        let strategy = match self.config.temporal_strategy {
            Some(s) => s,
            None => return,
        };

        self.sample_buf.clear();

        match strategy {
            TemporalStrategy::Uniform => {
                // Filter valid neighbors directly into sample_buf
                // Count valid first to decide Floyd's vs take-all
                self.temporal_filtered.clear();
                for (i, &t) in ts.iter().enumerate() {
                    if t < node_time {
                        self.temporal_filtered.push((i, t));
                    }
                }
                let valid = self.temporal_filtered.len();
                if valid == 0 {
                    return;
                }
                if sample_size >= valid {
                    for &(csr_idx, _) in &self.temporal_filtered {
                        self.sample_buf.push((neighbors[csr_idx], csr_idx));
                    }
                } else {
                    // Floyd's O(k) on filtered set
                    self.seen_set.clear();
                    for i in (valid - sample_size)..valid {
                        let s = (i + 1) as u64;
                        let x = self.rng.next_u32();
                        let j = ((x as u64).wrapping_mul(s) >> 32) as usize;
                        if self.seen_set.insert(j) {
                            let (csr_idx, _) = self.temporal_filtered[j];
                            self.sample_buf.push((neighbors[csr_idx], csr_idx));
                        } else {
                            self.seen_set.insert(i);
                            let (csr_idx, _) = self.temporal_filtered[i];
                            self.sample_buf.push((neighbors[csr_idx], csr_idx));
                        }
                    }
                }
            }
            TemporalStrategy::Last => {
                // Collect valid (csr_idx, timestamp) pairs, select top-k by time desc
                self.temporal_filtered.clear();
                for (i, &t) in ts.iter().enumerate() {
                    if t < node_time {
                        self.temporal_filtered.push((i, t));
                    }
                }
                let valid = self.temporal_filtered.len();
                if valid == 0 {
                    return;
                }
                let take = valid.min(sample_size);
                if take < valid {
                    // select_nth_unstable: O(n) partial sort — only partition, no full sort
                    self.temporal_filtered.select_nth_unstable_by(take - 1, |a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                for &(csr_idx, _) in &self.temporal_filtered[..take] {
                    self.sample_buf.push((neighbors[csr_idx], csr_idx));
                }
            }
        }

        // Emit from sample_buf with time recording — single pass, no redundant lookups
        let edge_offset = self.graph.edge_offset(node);
        for j in 0..self.sample_buf.len() {
            let (neighbor, csr_idx) = self.sample_buf[j];
            self.edge_src_buf.push(node);
            self.edge_dst_buf.push(neighbor);
            if self.config.track_edge_ids {
                self.edge_ids_buf.push(edge_offset + csr_idx as u64);
            }
            let is_new = self.insert_node_frontier(neighbor);
            if is_new {
                // Use ts directly (already borrowed above, same lifetime)
                self.node_times.push(ts[csr_idx]);
            }
        }
    }

    /// Normal (non-temporal) sampling for a single node: weighted or unweighted.
    #[inline]
    fn sample_normal(&mut self, node: NodeId, neighbors: &[NodeId], sample_size: usize) {
        let weights = if self.config.weighted {
            self.graph.neighbor_weights(node)
        } else {
            None
        };

        if let Some(w) = weights {
            self.sample_buf.clear();
            if self.config.replace {
                self.weighted_sample_with_replacement_into(neighbors, w, sample_size);
            } else {
                self.weighted_sample_without_replacement_into(neighbors, w, sample_size);
            }
            if self.config.track_edge_ids {
                let edge_offset = self.graph.edge_offset(node);
                for i in 0..self.sample_buf.len() {
                    let (neighbor, local_idx) = self.sample_buf[i];
                    self.edge_src_buf.push(node);
                    self.edge_dst_buf.push(neighbor);
                    self.edge_ids_buf.push(edge_offset + local_idx as u64);
                    self.insert_node_frontier(neighbor);
                }
            } else {
                for i in 0..self.sample_buf.len() {
                    let (neighbor, _) = self.sample_buf[i];
                    self.edge_src_buf.push(node);
                    self.edge_dst_buf.push(neighbor);
                    self.insert_node_frontier(neighbor);
                }
            }
        } else {
            let effective = if let Some(max_deg) = self.config.max_degree {
                if neighbors.len() > max_deg {
                    if let Some(ref telemetry) = self.config.telemetry {
                        telemetry.record_hub_node();
                    }
                    &neighbors[..max_deg]
                } else {
                    neighbors
                }
            } else {
                neighbors
            };

            let n = effective.len();
            if self.config.replace {
                self.emit_sample_replace(node, effective, sample_size);
            } else if sample_size >= n {
                self.emit_take_all(node, effective);
            } else {
                self.emit_sample_floyd(node, effective, sample_size);
            }
        }
    }

    /// Emit edges for sampling with replacement — pushes directly to edge buffers.
    #[inline]
    fn emit_sample_replace(&mut self, src: NodeId, neighbors: &[NodeId], k: usize) {
        let n = neighbors.len() as u64;
        if self.config.track_edge_ids {
            let edge_offset = self.graph.edge_offset(src);
            for _ in 0..k {
                let x = self.rng.next_u32();
                let idx = ((x as u64).wrapping_mul(n) >> 32) as usize;
                let neighbor = neighbors[idx];
                self.edge_src_buf.push(src);
                self.edge_dst_buf.push(neighbor);
                self.edge_ids_buf.push(edge_offset + idx as u64);
                self.insert_node_frontier(neighbor);
            }
        } else {
            for _ in 0..k {
                let x = self.rng.next_u32();
                let idx = ((x as u64).wrapping_mul(n) >> 32) as usize;
                let neighbor = neighbors[idx];
                self.edge_src_buf.push(src);
                self.edge_dst_buf.push(neighbor);
                self.insert_node_frontier(neighbor);
            }
        }
    }

    /// Emit edges for take-all (fanout >= degree) — pushes directly to edge buffers.
    #[inline]
    fn emit_take_all(&mut self, src: NodeId, neighbors: &[NodeId]) {
        if self.config.track_edge_ids {
            let edge_offset = self.graph.edge_offset(src);
            for (idx, &neighbor) in neighbors.iter().enumerate() {
                self.edge_src_buf.push(src);
                self.edge_dst_buf.push(neighbor);
                self.edge_ids_buf.push(edge_offset + idx as u64);
                self.insert_node_frontier(neighbor);
            }
        } else {
            for &neighbor in neighbors {
                self.edge_src_buf.push(src);
                self.edge_dst_buf.push(neighbor);
                self.insert_node_frontier(neighbor);
            }
        }
    }

    /// Floyd's O(k) sampling without replacement — pushes directly to edge buffers.
    /// Uses stack bitmap for n <= 256, reusable HashSet otherwise.
    fn emit_sample_floyd(&mut self, src: NodeId, neighbors: &[NodeId], k: usize) {
        let n = neighbors.len();

        // Macro to emit an edge, avoiding borrow conflicts with floyd_bitmap/seen_set.
        macro_rules! emit {
            ($idx:expr) => {{
                let neighbor = neighbors[$idx];
                self.edge_src_buf.push(src);
                self.edge_dst_buf.push(neighbor);
                if self.use_direct {
                    let ni = neighbor as usize;
                    if self.node_gen[ni] != self.current_gen {
                        let local = self.node_vec.len() as u32;
                        self.node_gen[ni] = self.current_gen;
                        self.node_local_idx[ni] = local;
                        self.node_vec.push(neighbor);
                        self.next_frontier.push(neighbor);
                    }
                } else {
                    let next_idx = self.node_vec.len() as u32;
                    match self.local_index.entry(neighbor) {
                        std::collections::hash_map::Entry::Occupied(_) => {}
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(next_idx);
                            self.node_vec.push(neighbor);
                            self.next_frontier.push(neighbor);
                        }
                    }
                }
            }};
        }

        if self.config.track_edge_ids {
            let edge_offset = self.graph.edge_offset(src);

            macro_rules! emit_with_eid {
                ($idx:expr) => {{
                    let neighbor = neighbors[$idx];
                    self.edge_src_buf.push(src);
                    self.edge_dst_buf.push(neighbor);
                    self.edge_ids_buf.push(edge_offset + $idx as u64);
                    if self.use_direct {
                        let ni = neighbor as usize;
                        if self.node_gen[ni] != self.current_gen {
                            let local = self.node_vec.len() as u32;
                            self.node_gen[ni] = self.current_gen;
                            self.node_local_idx[ni] = local;
                            self.node_vec.push(neighbor);
                            self.next_frontier.push(neighbor);
                        }
                    } else {
                        let next_idx = self.node_vec.len() as u32;
                        match self.local_index.entry(neighbor) {
                            std::collections::hash_map::Entry::Occupied(_) => {}
                            std::collections::hash_map::Entry::Vacant(e) => {
                                e.insert(next_idx);
                                self.node_vec.push(neighbor);
                                self.next_frontier.push(neighbor);
                            }
                        }
                    }
                }};
            }

            if n <= 256 {
                let bitmap = &mut self.floyd_bitmap[..n];
                for b in bitmap.iter_mut() {
                    *b = false;
                }
                for i in (n - k)..n {
                    let s = (i + 1) as u64;
                    let x = self.rng.next_u32();
                    let j = ((x as u64).wrapping_mul(s) >> 32) as usize;
                    if !bitmap[j] {
                        bitmap[j] = true;
                        emit_with_eid!(j);
                    } else {
                        bitmap[i] = true;
                        emit_with_eid!(i);
                    }
                }
            } else {
                self.seen_set.clear();
                for i in (n - k)..n {
                    let s = (i + 1) as u64;
                    let x = self.rng.next_u32();
                    let j = ((x as u64).wrapping_mul(s) >> 32) as usize;
                    if self.seen_set.insert(j) {
                        emit_with_eid!(j);
                    } else {
                        self.seen_set.insert(i);
                        emit_with_eid!(i);
                    }
                }
            }
        } else {
            // No edge ID tracking — faster path
            if n <= 256 {
                let bitmap = &mut self.floyd_bitmap[..n];
                for b in bitmap.iter_mut() {
                    *b = false;
                }
                for i in (n - k)..n {
                    let s = (i + 1) as u64;
                    let x = self.rng.next_u32();
                    let j = ((x as u64).wrapping_mul(s) >> 32) as usize;
                    if !bitmap[j] {
                        bitmap[j] = true;
                        emit!(j);
                    } else {
                        bitmap[i] = true;
                        emit!(i);
                    }
                }
            } else {
                self.seen_set.clear();
                for i in (n - k)..n {
                    let s = (i + 1) as u64;
                    let x = self.rng.next_u32();
                    let j = ((x as u64).wrapping_mul(s) >> 32) as usize;
                    if self.seen_set.insert(j) {
                        emit!(j);
                    } else {
                        self.seen_set.insert(i);
                        emit!(i);
                    }
                }
            }
        }
    }

    /// Weighted sampling with replacement — pushes results into self.sample_buf.
    fn weighted_sample_with_replacement_into(
        &mut self,
        neighbors: &[NodeId],
        weights: &[f32],
        k: usize,
    ) {
        let n = neighbors.len();
        debug_assert_eq!(n, weights.len());

        // Build cumulative distribution
        let mut cumsum = Vec::with_capacity(n);
        let mut total = 0.0f64;
        for &w in weights {
            total += w as f64;
            cumsum.push(total);
        }

        for _ in 0..k {
            let u = (self.rng.next_u64() as f64) / (u64::MAX as f64) * total;
            let idx = cumsum.partition_point(|&c| c <= u).min(n - 1);
            self.sample_buf.push((neighbors[idx], idx));
        }
    }

    /// Weighted sampling without replacement (Efraimidis-Spirakis) — pushes into self.sample_buf.
    ///
    /// Uses pre-allocated `weighted_keys` buffer. Key computation uses fast_neg_ln
    /// (bit-hack log2 approximation) since we only need relative ordering, not exact values.
    fn weighted_sample_without_replacement_into(
        &mut self,
        neighbors: &[NodeId],
        weights: &[f32],
        k: usize,
    ) {
        let n = neighbors.len();
        debug_assert_eq!(n, weights.len());

        if k >= n {
            for (i, &neighbor) in neighbors.iter().enumerate() {
                self.sample_buf.push((neighbor, i));
            }
            return;
        }

        // Reuse pre-allocated buffer
        self.weighted_keys.clear();
        self.weighted_keys.reserve(n.saturating_sub(self.weighted_keys.capacity()));

        // Compute keys: key[i] = -ln(u) / w[i], u ~ Uniform(0,1)
        // We use fast_neg_ln: bit-extract log2 approximation (~4x faster than ln())
        // Only relative ordering matters, so approximation is fine.
        for (i, &weight) in weights[..n].iter().enumerate() {
            let u_bits = self.rng.next_u64() | 1; // ensure nonzero
            let w = weight as f64;
            let key = if w > 0.0 {
                fast_neg_ln_u64(u_bits) / w
            } else {
                f64::INFINITY
            };
            self.weighted_keys.push((key, i));
        }

        // O(n) partial sort — only partitions around the k-th element
        self.weighted_keys
            .select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());

        for &(_, idx) in &self.weighted_keys[..k] {
            self.sample_buf.push((neighbors[idx], idx));
        }
    }
}

/// A sampled subgraph containing nodes and edges.
///
/// Nodes are stored in discovery order (not sorted). Local index lookups
/// use an O(1) HashMap instead of O(log N) binary search.
#[derive(Debug, Clone)]
pub struct SampledSubgraph {
    /// All nodes in the subgraph (discovery order)
    pub nodes: Vec<NodeId>,

    /// Edge source nodes (SOA format for zero-copy to numpy)
    pub edge_src: Vec<NodeId>,

    /// Edge destination nodes (SOA format for zero-copy to numpy)
    pub edge_dst: Vec<NodeId>,

    /// Global edge IDs (position in CSR edges array)
    pub edge_ids: Vec<u64>,

    /// Original seed nodes
    pub seeds: Vec<NodeId>,

    /// Number of nodes sampled at each hop (for PyG compatibility)
    pub num_sampled_nodes: Vec<usize>,

    /// Number of edges sampled at each hop (for PyG compatibility)
    pub num_sampled_edges: Vec<usize>,

    /// Local index map: global NodeId → position in `nodes` vec. O(1) lookup.
    local_index: FxHashMap<NodeId, u32>,

    /// Per-node batch assignment (disjoint mode only). Maps each node to its seed index.
    pub batch: Option<Vec<u32>>,

    /// Pre-computed local edge indices (disjoint mode, where local_index has duplicates).
    precomputed_local: Option<(Vec<u32>, Vec<u32>)>,
}

impl SampledSubgraph {
    /// Construct a SampledSubgraph, building the local_index from the nodes vec.
    ///
    /// This is intended for external callers (e.g., PyO3 round-trip) that need
    /// to reconstruct a SampledSubgraph from its public fields.
    pub fn from_parts(
        nodes: Vec<NodeId>,
        edge_src: Vec<NodeId>,
        edge_dst: Vec<NodeId>,
        edge_ids: Vec<u64>,
        seeds: Vec<NodeId>,
        num_sampled_nodes: Vec<usize>,
        num_sampled_edges: Vec<usize>,
    ) -> Self {
        let mut local_index =
            FxHashMap::with_capacity_and_hasher(nodes.len(), Default::default());
        for (i, &id) in nodes.iter().enumerate() {
            local_index.insert(id, i as u32);
        }
        Self {
            nodes,
            edge_src,
            edge_dst,
            edge_ids,
            seeds,
            num_sampled_nodes,
            num_sampled_edges,
            local_index,
            batch: None,
            precomputed_local: None,
        }
    }

    /// Returns the number of nodes in the subgraph
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges in the subgraph
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.edge_src.len()
    }

    /// Returns the number of seed nodes
    #[inline]
    pub fn num_seeds(&self) -> usize {
        self.seeds.len()
    }

    /// No-op for backward compatibility. Local indices are pre-computed during sampling.
    pub fn sort_nodes(&mut self) {
        // Intentionally empty — nodes are in discovery order and local_index
        // provides O(1) lookups. Kept for API compatibility.
    }

    /// Compute edge indices with local (remapped) node IDs in [0, num_nodes).
    ///
    /// Uses O(1) HashMap lookup per edge. No sorting required.
    ///
    /// # Errors
    /// Returns an error if any edge endpoint is not found in the local index,
    /// which indicates a bug in the sampling algorithm.
    pub fn edge_index_local(&mut self) -> Result<(Vec<u32>, Vec<u32>), String> {
        // Fast path: disjoint mode has precomputed local indices — take them (zero-copy move).
        if let Some(precomputed) = self.precomputed_local.take() {
            return Ok(precomputed);
        }

        let mut src_local = Vec::with_capacity(self.edge_src.len());
        for src in &self.edge_src {
            match self.local_index.get(src) {
                Some(&idx) => src_local.push(idx),
                None => {
                    return Err(format!("edge src {} not in local_index", src))
                }
            }
        }

        let mut dst_local = Vec::with_capacity(self.edge_dst.len());
        for dst in &self.edge_dst {
            match self.local_index.get(dst) {
                Some(&idx) => dst_local.push(idx),
                None => {
                    return Err(format!("edge dst {} not in local_index", dst))
                }
            }
        }

        Ok((src_local, dst_local))
    }

    /// Compute local seed indices (position of each seed in the nodes array).
    /// Useful for identifying which nodes in the subgraph were the original seeds.
    ///
    /// # Errors
    /// Returns an error if any seed is not found in the local index.
    pub fn seed_indices_local(&self) -> Result<Vec<u32>, String> {
        self.seeds
            .iter()
            .map(|id| {
                self.local_index
                    .get(id)
                    .copied()
                    .ok_or_else(|| format!("seed {} not in local_index", id))
            })
            .collect()
    }
}

/// Parallel batch sampler for high-throughput GNN training.
///
/// This sampler uses Rayon to sample multiple subgraphs in parallel,
/// maximizing CPU utilization during data loading.
pub struct ParallelBatchSampler<'a> {
    graph: &'a Graph,
    config: SamplingConfig,
}

impl<'a> ParallelBatchSampler<'a> {
    /// Creates a new parallel batch sampler
    pub fn new(graph: &'a Graph, config: SamplingConfig) -> Self {
        Self { graph, config }
    }

    /// Sample neighborhoods for multiple batches in parallel.
    ///
    /// Each batch is a list of seed nodes to sample from.
    pub fn sample_batches(&self, batches: &[Vec<NodeId>]) -> Vec<SampledSubgraph> {
        batches
            .par_iter()
            .map(|seeds| {
                let mut sampler = NeighborSampler::new(self.graph, self.config.clone());
                sampler.sample_neighbors(seeds)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph {
        // Create a simple graph:
        // 0 -> 1, 2, 3
        // 1 -> 2, 3
        // 2 -> 3, 4
        // 3 -> 4
        // 4 -> (no outgoing edges)
        let edges = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
        ];
        Graph::from_edges(5, &edges, None).unwrap()
    }

    #[test]
    fn test_sample_with_replacement() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2],
            replace: true,
            seed: Some(42),
            max_degree: Some(10_000),
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors(&[0]);

        // Should sample 2 neighbors from node 0's 3 neighbors
        assert_eq!(subgraph.num_seeds(), 1);
        assert!(subgraph.num_nodes() >= 1); // At least the seed
        assert_eq!(subgraph.num_edges(), 2); // Sampled 2 edges
    }

    #[test]
    fn test_sample_without_replacement() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2],
            replace: false,
            seed: Some(42),
            max_degree: Some(10_000),
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors(&[0]);

        assert_eq!(subgraph.num_edges(), 2);
        // Check that sampled neighbors are unique
        let neighbors: FxHashSet<_> = subgraph.edge_dst.iter().collect();
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_sample_without_replacement_fanout_exceeds_degree() {
        // Node 0 has only two neighbors; replace=false should not duplicate them.
        let edges = vec![(0, 1), (0, 2)];
        let graph = Graph::from_edges(3, &edges, None).unwrap();
        let config = SamplingConfig {
            fanout: vec![5],
            replace: false,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors(&[0]);

        assert_eq!(subgraph.num_edges(), 2);
        let neighbors: FxHashSet<_> = subgraph.edge_dst.iter().copied().collect();
        assert_eq!(neighbors, FxHashSet::from_iter([1, 2]));
    }

    #[test]
    fn test_induced_without_edge_ids() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2],
            replace: false,
            seed: Some(42),
            max_degree: Some(10_000),
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Induced,
            track_edge_ids: false,
            temporal_strategy: None,
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors(&[0]);

        assert!(subgraph.num_edges() > 0);
        assert!(subgraph.edge_ids.is_empty());
    }

    #[test]
    fn test_multi_hop_sampling() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2, 1], // 2 neighbors at hop 1, 1 neighbor at hop 2
            replace: false,
            seed: Some(42),
            max_degree: Some(10_000),
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors(&[0]);

        // Should have sampled 2 hops
        assert!(subgraph.num_nodes() >= 2); // At least seed + 1 hop
        assert!(subgraph.num_edges() > 0);
    }

    #[test]
    fn test_parallel_batch_sampling() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2],
            replace: false,
            seed: Some(42),
            max_degree: Some(10_000),
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            telemetry: None,
        };

        let sampler = ParallelBatchSampler::new(&graph, config);
        let batches = vec![vec![0], vec![1], vec![2]];

        let subgraphs = sampler.sample_batches(&batches);

        assert_eq!(subgraphs.len(), 3);
        for subgraph in subgraphs {
            assert_eq!(subgraph.num_seeds(), 1);
        }
    }

    #[test]
    fn test_empty_neighbors() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2],
            replace: false,
            seed: Some(42),
            max_degree: Some(10_000),
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors(&[4]); // Node 4 has no outgoing edges

        assert_eq!(subgraph.num_seeds(), 1);
        assert_eq!(subgraph.num_nodes(), 1); // Only the seed node
        assert_eq!(subgraph.num_edges(), 0); // No edges sampled
    }

    #[test]
    fn test_hub_node_degree_capping() {
        // Create a hub node with many neighbors
        let mut edges = vec![];
        let hub_node: NodeId = 0;
        let num_neighbors: usize = 20_000;

        for i in 1..=num_neighbors {
            edges.push((hub_node, i as NodeId));
        }

        let graph = Graph::from_edges(num_neighbors + 1, &edges, None).unwrap();
        let telemetry = Arc::new(SamplingTelemetry::new());
        let config = SamplingConfig {
            fanout: vec![25],
            replace: false,
            seed: Some(42),
            max_degree: Some(10_000), // Cap at 10k neighbors
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            telemetry: Some(telemetry.clone()),
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors(&[hub_node]);

        // Should sample exactly 25 neighbors despite hub having 20k edges
        assert_eq!(subgraph.num_edges(), 25);
        assert_eq!(subgraph.num_seeds(), 1);

        // Verify no panics or OOM - the key win for Reddit-scale graphs

        // Verify telemetry recorded the hub node
        let summary = telemetry.summary();
        assert_eq!(summary.hub_nodes_capped, 1);
        assert_eq!(summary.total_samples, 1);
    }

    #[test]
    fn test_edge_index_local_correctness() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![3, 2],
            replace: false,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let mut subgraph = sampler.sample_neighbors(&[0]);

        let (src_local, dst_local) = subgraph.edge_index_local().unwrap();
        assert_eq!(src_local.len(), subgraph.num_edges());
        assert_eq!(dst_local.len(), subgraph.num_edges());

        // All local indices should be < num_nodes
        for &s in &src_local {
            assert!((s as usize) < subgraph.num_nodes());
        }
        for &d in &dst_local {
            assert!((d as usize) < subgraph.num_nodes());
        }

        // Verify round-trip: local index maps back to correct global ID
        for (i, &s) in src_local.iter().enumerate() {
            assert_eq!(subgraph.nodes[s as usize], subgraph.edge_src[i]);
        }
        for (i, &d) in dst_local.iter().enumerate() {
            assert_eq!(subgraph.nodes[d as usize], subgraph.edge_dst[i]);
        }
    }

    #[test]
    fn test_seed_indices_local() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2],
            replace: false,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors(&[0, 1]);

        let seed_indices = subgraph.seed_indices_local().unwrap();
        assert_eq!(seed_indices.len(), 2);

        // Seeds should map back correctly
        for (i, &idx) in seed_indices.iter().enumerate() {
            assert_eq!(subgraph.nodes[idx as usize], subgraph.seeds[i]);
        }
    }

    #[test]
    fn test_reuse_across_calls() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2, 1],
            replace: false,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let mut sub1 = sampler.sample_neighbors(&[0]);
        let mut sub2 = sampler.sample_neighbors(&[1, 2]);

        assert!(sub1.num_nodes() >= 1);
        assert!(sub2.num_nodes() >= 2);
        assert_ne!(sub1.seeds, sub2.seeds);

        // Both should produce valid local indices
        sub1.edge_index_local().unwrap();
        sub2.edge_index_local().unwrap();
    }

    // =========================================================================
    // Weighted sampling tests
    // =========================================================================

    #[test]
    fn test_weighted_sampling_basic() {
        let edges = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
        ];
        let weights = vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let graph = Graph::from_edges(5, &edges, Some(&weights)).unwrap();

        let config = SamplingConfig {
            fanout: vec![2],
            replace: true,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: true,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors(&[0]);

        // Should sample 2 edges from node 0's 3 neighbors (weighted)
        assert_eq!(subgraph.num_edges(), 2);
        assert!(subgraph.num_nodes() >= 2); // seed + at least 1 unique neighbor
        // All sampled destinations must be actual neighbors of node 0
        for &dst in &subgraph.edge_dst {
            assert!([1u32, 2, 3].contains(&dst));
        }
    }

    #[test]
    fn test_weighted_sampling_without_replacement() {
        let edges = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
        ];
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let graph = Graph::from_edges(6, &edges, Some(&weights)).unwrap();

        let config = SamplingConfig {
            fanout: vec![3],
            replace: false,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: true,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors(&[0]);

        assert_eq!(subgraph.num_edges(), 3);
        // Without replacement: all sampled destinations must be unique
        let neighbors: FxHashSet<_> = subgraph.edge_dst.iter().copied().collect();
        assert_eq!(neighbors.len(), 3, "weighted without replacement should produce unique neighbors");
        // All must be valid neighbors of node 0
        for &dst in &subgraph.edge_dst {
            assert!([1u32, 2, 3, 4, 5].contains(&dst));
        }
    }

    #[test]
    fn test_weighted_sampling_high_weight_bias() {
        // Node 0 has 4 neighbors. Edge to node 1 has weight 100, others have weight 1.
        let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4)];
        let weights = vec![100.0, 1.0, 1.0, 1.0];
        let graph = Graph::from_edges(5, &edges, Some(&weights)).unwrap();

        let config = SamplingConfig {
            fanout: vec![1],
            replace: true,
            seed: Some(123),
            max_degree: None,
            cumulative: true,
            weighted: true,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            telemetry: None,
        };

        let mut count_heavy = 0u32;
        let trials = 1000;
        for trial in 0..trials {
            let trial_config = SamplingConfig {
                seed: Some(123 + trial as u64),
                ..config.clone()
            };
            let mut sampler = NeighborSampler::new(&graph, trial_config);
            let subgraph = sampler.sample_neighbors(&[0]);
            assert_eq!(subgraph.num_edges(), 1);
            if subgraph.edge_dst[0] == 1 {
                count_heavy += 1;
            }
        }

        // With weight 100 vs 3*1, expected proportion for node 1 is 100/103 ~ 97%.
        // Being conservative: just check > 50% to avoid flaky tests.
        assert!(
            count_heavy > 500,
            "Heavy-weight edge (w=100) should be sampled >50% of the time, got {}/{}",
            count_heavy,
            trials
        );
    }

    // =========================================================================
    // Temporal sampling tests
    // =========================================================================

    fn create_temporal_test_graph() -> Graph {
        // 0 -> 1, 2, 3
        // 1 -> 2, 3
        // 2 -> 3, 4
        // 3 -> 4
        let edges = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
        ];
        let mut graph = Graph::from_edges(5, &edges, None).unwrap();
        // Timestamps parallel to edges: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
        graph.set_timestamps(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        graph
    }

    #[test]
    fn test_temporal_uniform_basic() {
        let graph = create_temporal_test_graph();
        // Node 0's edges: 0->1 (t=1.0), 0->2 (t=2.0), 0->3 (t=3.0)
        // With seed time 2.5, only edges with t < 2.5 are valid: 0->1 (t=1.0), 0->2 (t=2.0)
        let config = SamplingConfig {
            fanout: vec![10], // large fanout to take all valid
            replace: false,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: Some(TemporalStrategy::Uniform),
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors_temporal(&[0], &[2.5]);

        // Only edges with t < 2.5 should be sampled: 0->1 and 0->2
        assert_eq!(subgraph.num_edges(), 2);
        let dsts: FxHashSet<_> = subgraph.edge_dst.iter().copied().collect();
        assert!(dsts.contains(&1), "edge 0->1 (t=1.0) should be sampled");
        assert!(dsts.contains(&2), "edge 0->2 (t=2.0) should be sampled");
        assert!(!dsts.contains(&3), "edge 0->3 (t=3.0) should NOT be sampled (t >= 2.5)");
    }

    #[test]
    fn test_temporal_last_basic() {
        let graph = create_temporal_test_graph();
        // Node 0's edges: 0->1 (t=1.0), 0->2 (t=2.0), 0->3 (t=3.0)
        // With seed time 3.5, valid edges: 0->1 (t=1.0), 0->2 (t=2.0), 0->3 (t=3.0)
        // TemporalStrategy::Last with fanout=2 should pick the 2 most recent: 0->3 (t=3.0), 0->2 (t=2.0)
        let config = SamplingConfig {
            fanout: vec![2],
            replace: false,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: Some(TemporalStrategy::Last),
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors_temporal(&[0], &[3.5]);

        assert_eq!(subgraph.num_edges(), 2);
        let dsts: FxHashSet<_> = subgraph.edge_dst.iter().copied().collect();
        // Last strategy should pick the 2 most recent valid edges
        assert!(dsts.contains(&3), "edge 0->3 (t=3.0, most recent) should be sampled");
        assert!(dsts.contains(&2), "edge 0->2 (t=2.0, second most recent) should be sampled");
        assert!(!dsts.contains(&1), "edge 0->1 (t=1.0, oldest) should NOT be sampled");
    }

    #[test]
    fn test_temporal_filters_all() {
        let graph = create_temporal_test_graph();
        // Node 0's edges have timestamps 1.0, 2.0, 3.0.
        // Seed time 0.5 means ALL edges have t >= 0.5 ... but filter is t < node_time.
        // So with time 0.5: no edge has t < 0.5.
        let config = SamplingConfig {
            fanout: vec![10],
            replace: false,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: Some(TemporalStrategy::Uniform),
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors_temporal(&[0], &[0.5]);

        // No edges should be sampled since all timestamps >= 0.5 and the earliest is 1.0
        assert_eq!(subgraph.num_edges(), 0);
        assert_eq!(subgraph.num_nodes(), 1); // Only the seed
    }

    #[test]
    fn test_temporal_propagates_time() {
        // Build graph: 0->1 (t=10.0), 0->2 (t=20.0), 1->3 (t=5.0), 1->4 (t=15.0)
        let edges = vec![(0, 1), (0, 2), (1, 3), (1, 4)];
        let mut graph = Graph::from_edges(5, &edges, None).unwrap();
        graph.set_timestamps(vec![10.0, 20.0, 5.0, 15.0]);

        // 2-hop sampling with seed time 25.0
        // Hop 1: node 0, time=25.0 -> edges with t<25: 0->1 (t=10), 0->2 (t=20) both valid
        //   Node 1 gets time = 10.0 (the edge timestamp from 0->1)
        //   Node 2 gets time = 20.0 (the edge timestamp from 0->2)
        // Hop 2: node 1, time=10.0 -> edges: 1->3 (t=5.0) valid, 1->4 (t=15.0) NOT valid (15 >= 10)
        //         node 2, time=20.0 -> no outgoing edges in this graph
        let config = SamplingConfig {
            fanout: vec![10, 10], // large fanout to take all valid
            replace: false,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: Some(TemporalStrategy::Uniform),
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors_temporal(&[0], &[25.0]);

        // Hop 1 edges: 0->1, 0->2
        // Hop 2 edges from node 1 (time=10.0): only 1->3 (t=5.0 < 10.0)
        // Node 4 should NOT appear because 1->4 has t=15.0 >= 10.0
        let all_nodes: FxHashSet<_> = subgraph.nodes.iter().copied().collect();
        assert!(all_nodes.contains(&0), "seed should be present");
        assert!(all_nodes.contains(&1), "hop-1 neighbor should be present");
        assert!(all_nodes.contains(&3), "hop-2 neighbor via edge t=5<10 should be present");
        assert!(
            !all_nodes.contains(&4),
            "node 4 should NOT be present: edge 1->4 has t=15.0 >= node 1's time 10.0"
        );
    }

    #[test]
    fn test_temporal_uniform_respects_fanout() {
        // Node 0 has 5 neighbors, all with timestamps well below the seed time.
        let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)];
        let mut graph = Graph::from_edges(6, &edges, None).unwrap();
        graph.set_timestamps(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let config = SamplingConfig {
            fanout: vec![2], // Only sample 2 out of 5 valid neighbors
            replace: false,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: Some(TemporalStrategy::Uniform),
            disjoint: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors_temporal(&[0], &[100.0]);

        // All 5 edges have t < 100.0, but fanout=2 so exactly 2 should be sampled
        assert_eq!(subgraph.num_edges(), 2, "fanout=2 should produce exactly 2 edges");
        // Sampled destinations should be valid neighbors
        for &dst in &subgraph.edge_dst {
            assert!([1u32, 2, 3, 4, 5].contains(&dst));
        }
        // Without replacement, the 2 destinations should be unique
        let dsts: FxHashSet<_> = subgraph.edge_dst.iter().copied().collect();
        assert_eq!(dsts.len(), 2, "without replacement, destinations should be unique");
    }

    // =========================================================================
    // Disjoint mode tests
    // =========================================================================

    #[test]
    fn test_disjoint_basic() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2],
            replace: false,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: true,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors_disjoint(&[0, 1], None);

        // batch should be Some and have one entry per node
        assert!(subgraph.batch.is_some(), "disjoint mode should produce a batch vector");
        let batch = subgraph.batch.as_ref().unwrap();
        assert_eq!(
            batch.len(),
            subgraph.nodes.len(),
            "batch length should equal number of nodes"
        );

        // Every batch value should be 0 or 1 (2 seeds)
        for &b in batch {
            assert!(b == 0 || b == 1, "batch values should map to seed indices 0 or 1");
        }

        // Seeds vector should be preserved
        assert_eq!(subgraph.seeds, vec![0, 1]);
    }

    #[test]
    fn test_disjoint_no_node_sharing() {
        // Node 0 and node 1 share neighbors (2, 3).
        // In disjoint mode, shared neighbors should appear TWICE: once per seed.
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![10], // large fanout to take all neighbors
            replace: false,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: true,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors_disjoint(&[0, 1], None);
        let batch = subgraph.batch.as_ref().unwrap();

        // Collect nodes per seed
        let mut seed0_nodes: Vec<NodeId> = Vec::new();
        let mut seed1_nodes: Vec<NodeId> = Vec::new();
        for (i, &node) in subgraph.nodes.iter().enumerate() {
            if batch[i] == 0 {
                seed0_nodes.push(node);
            } else {
                seed1_nodes.push(node);
            }
        }

        // Seed 0 (node 0): neighbors are 1, 2, 3
        // Seed 1 (node 1): neighbors are 2, 3
        // Shared neighbors: 2 and 3
        // In disjoint mode, nodes 2 and 3 should appear in BOTH seed subgraphs
        let seed0_set: FxHashSet<_> = seed0_nodes.iter().copied().collect();
        let seed1_set: FxHashSet<_> = seed1_nodes.iter().copied().collect();

        assert!(seed0_set.contains(&2) && seed0_set.contains(&3),
            "seed 0 should have neighbors 2 and 3");
        assert!(seed1_set.contains(&2) && seed1_set.contains(&3),
            "seed 1 should have neighbors 2 and 3");

        // The total node count should be greater than the unique count
        // (because shared nodes are duplicated)
        let unique_global: FxHashSet<_> = subgraph.nodes.iter().copied().collect();
        assert!(
            subgraph.nodes.len() > unique_global.len(),
            "disjoint mode should duplicate shared nodes: total {} > unique {}",
            subgraph.nodes.len(),
            unique_global.len()
        );
    }

    #[test]
    fn test_disjoint_local_indices_offset() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2],
            replace: false,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: true,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors_disjoint(&[0, 1], None);

        // precomputed_local should be Some in disjoint mode
        assert!(
            subgraph.precomputed_local.is_some(),
            "disjoint mode should produce precomputed_local"
        );
        let (src_local, dst_local) = subgraph.precomputed_local.as_ref().unwrap();

        assert_eq!(src_local.len(), subgraph.num_edges());
        assert_eq!(dst_local.len(), subgraph.num_edges());

        // All local indices should be valid (< total nodes in combined subgraph)
        let num_nodes = subgraph.nodes.len() as u32;
        for &s in src_local {
            assert!(s < num_nodes, "local src index {} should be < {}", s, num_nodes);
        }
        for &d in dst_local {
            assert!(d < num_nodes, "local dst index {} should be < {}", d, num_nodes);
        }

        // Verify that local indices correctly map back to global IDs
        for (i, &s) in src_local.iter().enumerate() {
            assert_eq!(
                subgraph.nodes[s as usize], subgraph.edge_src[i],
                "local src index should map back to correct global node"
            );
        }
        for (i, &d) in dst_local.iter().enumerate() {
            assert_eq!(
                subgraph.nodes[d as usize], subgraph.edge_dst[i],
                "local dst index should map back to correct global node"
            );
        }
    }

    #[test]
    fn test_disjoint_with_temporal() {
        // Combine disjoint + temporal: verify both batch vector and temporal filtering work
        let edges = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
        ];
        let mut graph = Graph::from_edges(5, &edges, None).unwrap();
        graph.set_timestamps(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let config = SamplingConfig {
            fanout: vec![10], // large fanout
            replace: false,
            seed: Some(42),
            max_degree: None,
            cumulative: true,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: Some(TemporalStrategy::Uniform),
            disjoint: true,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        // Seed 0 with time 2.5: only 0->1 (t=1.0) and 0->2 (t=2.0) valid
        // Seed 1 with time 4.5: only 1->2 (t=4.0) valid (1->3 has t=5.0 >= 4.5)
        let subgraph = sampler.sample_neighbors_disjoint(
            &[0, 1],
            Some(&[2.5, 4.5]),
        );

        // batch should exist
        assert!(subgraph.batch.is_some(), "disjoint+temporal should produce batch");
        let batch = subgraph.batch.as_ref().unwrap();
        assert_eq!(batch.len(), subgraph.nodes.len());

        // Collect destinations per seed
        let mut seed0_dsts: FxHashSet<NodeId> = FxHashSet::default();
        let mut seed1_dsts: FxHashSet<NodeId> = FxHashSet::default();

        // Map edges to their seed based on source node's batch assignment
        // In disjoint mode edges are concatenated: first seed0's edges then seed1's
        // We can identify them by looking at the batch of the source local index
        let (src_local, _dst_local) = subgraph.precomputed_local.as_ref().unwrap();
        for (i, &sl) in src_local.iter().enumerate() {
            let seed_idx = batch[sl as usize];
            if seed_idx == 0 {
                seed0_dsts.insert(subgraph.edge_dst[i]);
            } else {
                seed1_dsts.insert(subgraph.edge_dst[i]);
            }
        }

        // Seed 0 (time=2.5): edges 0->1 (t=1.0), 0->2 (t=2.0)
        assert!(seed0_dsts.contains(&1), "seed 0 should sample 0->1 (t=1.0 < 2.5)");
        assert!(seed0_dsts.contains(&2), "seed 0 should sample 0->2 (t=2.0 < 2.5)");
        assert!(!seed0_dsts.contains(&3), "seed 0 should NOT sample 0->3 (t=3.0 >= 2.5)");

        // Seed 1 (time=4.5): edges 1->2 (t=4.0) valid, 1->3 (t=5.0) NOT valid
        assert!(seed1_dsts.contains(&2), "seed 1 should sample 1->2 (t=4.0 < 4.5)");
        assert!(!seed1_dsts.contains(&3), "seed 1 should NOT sample 1->3 (t=5.0 >= 4.5)");
    }
}
