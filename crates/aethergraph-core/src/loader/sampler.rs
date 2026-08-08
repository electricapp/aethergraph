//! Neighborhood sampling for GNN training.
//!
//! Internal scratch (frontiers, dedup arrays, sample buffers) is pre-allocated
//! at sampler construction and reset across `sample_neighbors` calls via
//! `clear()`. The output buffers handed back in each `SampledSubgraph` are
//! freshly allocated per call (swapped out of the sampler). Local node indices
//! are assigned during sampling — no post-sort, no binary search.

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
/// Exact `-ln(u)` for a random u64 mapped to a uniform `u` in (0, 1].
///
/// The top 53 bits of `bits` form a uniform integer in `[0, 2^53)`; adding 1
/// and scaling by `2^-53` yields `u` in `(0, 1]`, so `-u.ln()` is finite and
/// non-negative for every input — `bits == 0` gives `u = 2^-53` (a large but
/// finite key) and `bits == u64::MAX` gives `u = 1.0` (key `0.0`). Used as the
/// Efraimidis-Spirakis key exponent, where exactness keeps weighted
/// sampling unbiased.
#[inline(always)]
fn fast_neg_ln_u64(bits: u64) -> f64 {
    let u = ((bits >> 11) as f64 + 1.0) * (1.0 / (1u64 << 53) as f64);
    -u.ln()
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

    /// Bit-deterministic mode.
    ///
    /// When `true`, parallel sampling paths fall back to serial execution
    /// so that the same `seed` produces byte-identical subgraphs across
    /// runs, machines, and core counts. When `false` (default), the
    /// `seed` parameter still controls per-thread RNG streams but Rayon's
    /// thread-pool scheduling can permute the order in which batches are
    /// emitted — meaning two runs with the same seed are *statistically*
    /// equivalent but not bit-identical.
    ///
    /// Use `true` for regression tests, audit logs, and any reproducibility
    /// claim. Pay the throughput cost (typically 2–8×) only when needed.
    pub deterministic: bool,

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
            // Default to non-deterministic parallel path for throughput;
            // opt in to `deterministic = true` for tests / audit / regression.
            deterministic: false,
            telemetry: None, // Opt-in telemetry
        }
    }
}

/// Errors returned by [`NeighborSampler::sample_neighbors_temporal`].
///
/// Each variant distinguishes a misconfigured graph or sampler from an
/// honestly-empty neighborhood.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalSamplingError {
    /// `seeds.len() != input_times.len()`.
    LengthMismatch { seeds: usize, times: usize },
    /// The graph has no edge timestamps attached.
    TimestampsMissing,
    /// `SamplingConfig::temporal_strategy` is `None`.
    StrategyMissing,
}

impl std::fmt::Display for TemporalSamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { seeds, times } => write!(
                f,
                "temporal sampling: seeds.len() ({seeds}) != input_times.len() ({times})"
            ),
            Self::TimestampsMissing => f.write_str(
                "temporal sampling: graph has no edge timestamps — call Graph::set_timestamps() first",
            ),
            Self::StrategyMissing => f.write_str(
                "temporal sampling: SamplingConfig::temporal_strategy is None",
            ),
        }
    }
}

impl std::error::Error for TemporalSamplingError {}

/// A neighborhood sampler for GNN training.
///
/// Internal scratch buffers are pre-allocated at construction and reset across
/// `sample_neighbors` calls; the per-call output buffers are freshly allocated
/// and swapped out into the returned `SampledSubgraph`.
///
/// Dedup uses either a generation-tagged direct array (for graphs with
/// <= 100K nodes, or when the estimated sample exceeds 1% of the graph;
/// O(1) lookup, ~8 bytes/node memory) or an FxHashMap-backed index for sparser
/// sampling patterns (e.g. 12K nodes touched out of 10M).
pub struct NeighborSampler<'a> {
    graph: &'a Graph,
    config: SamplingConfig,
    rng: WyRand,
    /// Direct array path (selected by `use_direct`: <= 100K nodes, or a
    /// sample estimated to touch > 1% of the graph). Each entry packs the
    /// generation tag in the high 32 bits and the local index in the low 32,
    /// so a dedup probe touches exactly one cache line instead of two
    /// parallel arrays.
    node_slot: Vec<u64>,
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
    /// Local (remapped) endpoint indices, filled at emit time — the local
    /// index of every endpoint is already known when the edge is pushed, so
    /// no post-pass hashmap lookup is ever needed.
    src_local_buf: Vec<u32>,
    dst_local_buf: Vec<u32>,
    /// Local indices of the seeds, captured at registration.
    seed_locals_buf: Vec<u32>,
    /// Double-buffered frontiers carrying (global id, local index).
    frontier: Vec<(NodeId, u32)>,
    next_frontier: Vec<(NodeId, u32)>,
    /// Reusable buffer for weighted sampling results.
    sample_buf: Vec<(NodeId, usize)>,
    /// Reusable Floyd scratch (small degree <= 256): generation stamps, so
    /// per-node reuse is a counter bump instead of an O(n) clear.
    floyd_stamp: [u32; 256],
    floyd_gen: u32,
    /// Reusable Floyd set (large degree > 256).
    seen_set: FxHashSet<usize>,
    /// Temporal: per-node time constraints (parallel to node_vec).
    node_times: Vec<f64>,
    /// Temporal: filtered (csr_index, timestamp) pairs for select-k.
    temporal_filtered: Vec<(usize, f64)>,
    /// Weighted no-replace: reusable key buffer to avoid per-call allocation.
    weighted_keys: Vec<(f64, usize)>,
    /// Weighted with-replace: reusable cumulative-distribution buffer.
    cumsum_buf: Vec<f64>,
}

/// How many frontier entries ahead the hop loop prefetches the CSR offsets
/// (and, at half this distance, the first edge line). The dedup probes and
/// neighbor-list reads are random-access and serially dependent without
/// these hints.
const FRONTIER_PREFETCH_DIST: usize = 4;

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
        let use_direct =
            num_nodes <= 100_000 || (estimated_sample as f64 / num_nodes as f64) > 0.01;
        Self {
            graph,
            config,
            rng,
            node_slot: if use_direct {
                vec![0u64; num_nodes]
            } else {
                Vec::new()
            },
            current_gen: 0,
            local_index: FxHashMap::with_capacity_and_hasher(512, Default::default()),
            use_direct,
            node_vec: Vec::with_capacity(512),
            edge_src_buf: Vec::with_capacity(2048),
            edge_dst_buf: Vec::with_capacity(2048),
            edge_ids_buf: Vec::with_capacity(2048),
            src_local_buf: Vec::with_capacity(2048),
            dst_local_buf: Vec::with_capacity(2048),
            seed_locals_buf: Vec::with_capacity(512),
            frontier: Vec::with_capacity(512),
            next_frontier: Vec::with_capacity(4096),
            sample_buf: Vec::with_capacity(max_fanout),
            floyd_stamp: [0u32; 256],
            floyd_gen: 0,
            seen_set: FxHashSet::with_capacity_and_hasher(max_fanout * 2, Default::default()),
            node_times: Vec::new(),
            temporal_filtered: Vec::with_capacity(max_fanout),
            weighted_keys: Vec::with_capacity(256),
            cumsum_buf: Vec::with_capacity(256),
        }
    }

    /// Pack a generation tag and local index into one dedup-slot word.
    #[inline(always)]
    const fn pack_slot(generation: u32, idx: u32) -> u64 {
        ((generation as u64) << 32) | idx as u64
    }

    #[inline(always)]
    fn insert_node(&mut self, id: NodeId) -> u32 {
        if self.use_direct {
            let i = id as usize;
            let slot = self.node_slot[i];
            if (slot >> 32) as u32 == self.current_gen {
                slot as u32
            } else {
                let idx = self.node_vec.len() as u32;
                self.node_slot[i] = Self::pack_slot(self.current_gen, idx);
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

    /// Register `id`, pushing it onto the next frontier when new. Returns
    /// its local index either way — the caller records it for the edge's
    /// endpoint without any later lookup.
    #[inline(always)]
    fn insert_node_frontier(&mut self, id: NodeId) -> (u32, bool) {
        if self.use_direct {
            let i = id as usize;
            let slot = self.node_slot[i];
            if (slot >> 32) as u32 == self.current_gen {
                return (slot as u32, false);
            }
            let idx = self.node_vec.len() as u32;
            self.node_slot[i] = Self::pack_slot(self.current_gen, idx);
            self.node_vec.push(id);
            self.next_frontier.push((id, idx));
            (idx, true)
        } else {
            let next_idx = self.node_vec.len() as u32;
            match self.local_index.entry(id) {
                std::collections::hash_map::Entry::Occupied(e) => (*e.get(), false),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(next_idx);
                    self.node_vec.push(id);
                    self.next_frontier.push((id, next_idx));
                    (next_idx, true)
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
    /// Each seed has an associated time; only edges with timestamp < seed time
    /// are eligible.
    ///
    /// # Errors
    /// Returns [`TemporalSamplingError::TimestampsMissing`] if the graph has
    /// no edge timestamps attached (call [`crate::Graph::set_timestamps`]
    /// first). Returns [`TemporalSamplingError::StrategyMissing`] if no
    /// `temporal_strategy` is set on the `SamplingConfig`. Returns
    /// [`TemporalSamplingError::LengthMismatch`] if `seeds.len() !=
    /// input_times.len()`.
    pub fn sample_neighbors_temporal(
        &mut self,
        seeds: &[NodeId],
        input_times: &[f64],
    ) -> Result<SampledSubgraph, TemporalSamplingError> {
        if seeds.len() != input_times.len() {
            return Err(TemporalSamplingError::LengthMismatch {
                seeds: seeds.len(),
                times: input_times.len(),
            });
        }
        if !self.graph.has_timestamps() {
            return Err(TemporalSamplingError::TimestampsMissing);
        }
        if self.config.temporal_strategy.is_none() {
            return Err(TemporalSamplingError::StrategyMissing);
        }
        Ok(self.sample_neighbors_inner(seeds, Some(input_times)))
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
        // Per-hop counts accumulated across every seed (PyG compatibility).
        let mut num_sampled_nodes = vec![0usize; num_hops];
        let mut num_sampled_edges = vec![0usize; num_hops];

        for (seed_idx, &seed) in seeds.iter().enumerate() {
            // Reset dedup state per seed (cheap: no dealloc, just counter bump or clear)
            if self.use_direct {
                self.current_gen = self.current_gen.wrapping_add(1);
                if self.current_gen == 0 {
                    self.node_slot.fill(0);
                    self.current_gen = 1;
                }
            } else {
                self.local_index.clear();
            }
            self.node_vec.clear();
            self.edge_src_buf.clear();
            self.edge_dst_buf.clear();
            self.edge_ids_buf.clear();
            self.src_local_buf.clear();
            self.dst_local_buf.clear();
            self.frontier.clear();
            self.next_frontier.clear();
            self.node_times.clear();

            // Register single seed
            let seed_local = self.insert_node(seed);
            self.frontier.push((seed, seed_local));
            if let Some(times) = input_times {
                self.node_times.push(times[seed_idx]);
            }

            // Run hop loop inline (same logic as sample_neighbors_inner)
            self.run_hops(num_hops, is_temporal, |hop, new_nodes, new_edges| {
                // Accumulate this seed's per-hop counts into the combined totals.
                num_sampled_nodes[hop] += new_nodes;
                num_sampled_edges[hop] += new_edges;
            });

            // Endpoint locals were recorded at emit time; only the per-seed
            // block offset needs applying while concatenating.
            let node_offset = all_nodes.len() as u32;
            all_src_local.extend(self.src_local_buf.iter().map(|&l| l + node_offset));
            all_dst_local.extend(self.dst_local_buf.iter().map(|&l| l + node_offset));

            all_nodes.extend_from_slice(&self.node_vec);
            all_edge_src.extend_from_slice(&self.edge_src_buf);
            all_edge_dst.extend_from_slice(&self.edge_dst_buf);
            all_edge_ids.extend_from_slice(&self.edge_ids_buf);

            let node_count = self.node_vec.len();
            batch.resize(batch.len() + node_count, seed_idx as u32);
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
            seed_locals: None,
        }
    }

    /// Run the hop loop over the current frontier, invoking `record` with
    /// `(hop, new_frontier_nodes, new_edges)` after each hop.
    ///
    /// CSR slices are hoisted once per call so the inner loop indexes raw
    /// arrays (no per-node storage dispatch), and upcoming frontier entries'
    /// offset words and first edge lines are prefetched ahead of use — the
    /// probes are random-access and otherwise serially dependent.
    fn run_hops(
        &mut self,
        num_hops: usize,
        is_temporal: bool,
        mut record: impl FnMut(usize, usize, usize),
    ) {
        use crate::internal::prefetch::prefetch_read;

        let offsets = self.graph.offsets();
        let edges = self.graph.edges();
        let num_nodes = self.graph.num_nodes();
        let num_edges = edges.len();
        let weights = if self.config.weighted {
            self.graph.weights()
        } else {
            None
        };
        let timestamps = if is_temporal {
            self.graph.timestamps()
        } else {
            None
        };

        for hop in 0..num_hops {
            let edges_before = self.edge_src_buf.len();
            let sample_size = self.config.fanout[hop];
            self.next_frontier.clear();

            let frontier_len = self.frontier.len();
            for fi in 0..frontier_len {
                if fi + FRONTIER_PREFETCH_DIST < frontier_len {
                    let ahead = self.frontier[fi + FRONTIER_PREFETCH_DIST].0 as usize;
                    if ahead < num_nodes {
                        prefetch_read(&offsets[ahead]);
                    }
                }
                if fi + FRONTIER_PREFETCH_DIST / 2 < frontier_len {
                    let ahead = self.frontier[fi + FRONTIER_PREFETCH_DIST / 2].0 as usize;
                    if ahead < num_nodes {
                        let s = offsets[ahead] as usize;
                        if s < num_edges {
                            prefetch_read(&edges[s]);
                        }
                    }
                }

                let (node, node_local) = self.frontier[fi];
                let i = node as usize;
                if i >= num_nodes {
                    continue;
                }
                let start = offsets[i] as usize;
                let end = offsets[i + 1] as usize;
                if start >= end || end > num_edges {
                    continue;
                }
                let neighbors = &edges[start..end];
                let edge_offset = start as u64;

                if is_temporal {
                    let Some(ts_all) = timestamps else { continue };
                    self.sample_temporal(
                        node,
                        node_local,
                        neighbors,
                        &ts_all[start..end],
                        edge_offset,
                        sample_size,
                    );
                } else {
                    let w = weights.map(|w| &w[start..end]);
                    self.sample_normal(node, node_local, neighbors, w, edge_offset, sample_size);
                }
            }

            record(
                hop,
                self.next_frontier.len(),
                self.edge_src_buf.len() - edges_before,
            );

            // Update frontier based on sampling mode (double-buffer swap)
            if self.config.cumulative {
                // PyG-style: accumulate all nodes for next hop. By design this
                // re-processes the whole accumulated frontier each hop, so the
                // total work is O(Σ frontier sizes) rather than O(new nodes).
                self.frontier.extend_from_slice(&self.next_frontier);
            } else {
                // Pure GraphSAGE: only use new frontier
                std::mem::swap(&mut self.frontier, &mut self.next_frontier);
            }
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
                self.node_slot.fill(0);
                self.current_gen = 1;
            }
        } else {
            self.local_index.clear();
        }
        self.node_vec.clear();
        self.edge_src_buf.clear();
        self.edge_dst_buf.clear();
        self.edge_ids_buf.clear();
        self.src_local_buf.clear();
        self.dst_local_buf.clear();
        self.seed_locals_buf.clear();
        self.frontier.clear();
        self.next_frontier.clear();
        self.node_times.clear();

        // Register seeds with local indices
        for &seed in seeds {
            let local = self.insert_node(seed);
            self.seed_locals_buf.push(local);
            self.frontier.push((seed, local));
        }

        // Initialize seed times for temporal sampling
        if let Some(times) = input_times {
            self.node_times.extend_from_slice(times);
        }

        // Track per-hop sampling stats (for PyG compatibility)
        let mut num_sampled_nodes = Vec::with_capacity(num_hops);
        let mut num_sampled_edges = Vec::with_capacity(num_hops);

        self.run_hops(num_hops, is_temporal, |_hop, new_nodes, new_edges| {
            num_sampled_nodes.push(new_nodes);
            num_sampled_edges.push(new_edges);
        });

        // Hand the filled buffers to the caller and swap fresh ones back in.
        // The returned buffers (nodes, the three edge arrays, and the local
        // endpoint indices) are therefore freshly allocated on every call;
        // only the internal scratch (frontiers, dedup arrays, sample buffers)
        // is reset and reused.
        let mut node_vec = Vec::with_capacity(512);
        std::mem::swap(&mut self.node_vec, &mut node_vec);

        let mut edge_src = Vec::with_capacity(2048);
        std::mem::swap(&mut self.edge_src_buf, &mut edge_src);

        let mut edge_dst = Vec::with_capacity(2048);
        std::mem::swap(&mut self.edge_dst_buf, &mut edge_dst);

        let mut edge_ids = Vec::with_capacity(if self.config.track_edge_ids { 2048 } else { 0 });
        std::mem::swap(&mut self.edge_ids_buf, &mut edge_ids);

        let mut src_local = Vec::with_capacity(2048);
        std::mem::swap(&mut self.src_local_buf, &mut src_local);

        let mut dst_local = Vec::with_capacity(2048);
        std::mem::swap(&mut self.dst_local_buf, &mut dst_local);

        let mut seed_locals = Vec::with_capacity(512);
        std::mem::swap(&mut self.seed_locals_buf, &mut seed_locals);

        // Apply subgraph_type post-processing. Local endpoint indices were
        // recorded at emit time, so no per-edge map lookup happens on any
        // path; the Induced filter builds its membership map alone.
        let (edge_src, edge_dst, edge_ids, src_local, dst_local, local_index) =
            match self.config.subgraph_type {
                SubgraphType::Directional => (
                    edge_src,
                    edge_dst,
                    edge_ids,
                    src_local,
                    dst_local,
                    FxHashMap::default(),
                ),
                SubgraphType::Induced => {
                    // Filter to only edges where both endpoints are in the
                    // node set. local_index serves as the node set — O(1)
                    // lookup. The local arrays are filtered in lockstep.
                    let local_index: FxHashMap<NodeId, u32> = node_vec
                        .iter()
                        .enumerate()
                        .map(|(idx, &id)| (id, idx as u32))
                        .collect();

                    let mut new_src = Vec::with_capacity(edge_src.len());
                    let mut new_dst = Vec::with_capacity(edge_dst.len());
                    let mut new_ids = if self.config.track_edge_ids {
                        Vec::with_capacity(edge_ids.len())
                    } else {
                        Vec::new()
                    };
                    let mut new_src_local = Vec::with_capacity(src_local.len());
                    let mut new_dst_local = Vec::with_capacity(dst_local.len());

                    for i in 0..edge_src.len() {
                        if local_index.contains_key(&edge_src[i])
                            && local_index.contains_key(&edge_dst[i])
                        {
                            new_src.push(edge_src[i]);
                            new_dst.push(edge_dst[i]);
                            if self.config.track_edge_ids {
                                new_ids.push(edge_ids[i]);
                            }
                            new_src_local.push(src_local[i]);
                            new_dst_local.push(dst_local[i]);
                        }
                    }

                    (
                        new_src,
                        new_dst,
                        new_ids,
                        new_src_local,
                        new_dst_local,
                        local_index,
                    )
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
                    let mut new_src_local = Vec::with_capacity(src_local.len() * 2);
                    let mut new_dst_local = Vec::with_capacity(dst_local.len() * 2);

                    // Original edges
                    new_src.extend_from_slice(&edge_src);
                    new_dst.extend_from_slice(&edge_dst);
                    if self.config.track_edge_ids {
                        new_ids.extend_from_slice(&edge_ids);
                    }
                    new_src_local.extend_from_slice(&src_local);
                    new_dst_local.extend_from_slice(&dst_local);

                    // Reverse edges (edge_id for reverse is same as forward)
                    new_src.extend_from_slice(&edge_dst);
                    new_dst.extend_from_slice(&edge_src);
                    if self.config.track_edge_ids {
                        new_ids.extend_from_slice(&edge_ids);
                    }
                    new_src_local.extend_from_slice(&dst_local);
                    new_dst_local.extend_from_slice(&src_local);

                    (
                        new_src,
                        new_dst,
                        new_ids,
                        new_src_local,
                        new_dst_local,
                        FxHashMap::default(),
                    )
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
            precomputed_local: Some((src_local, dst_local)),
            seed_locals: Some(seed_locals),
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
    fn sample_temporal(
        &mut self,
        node: NodeId,
        node_local: u32,
        neighbors: &[NodeId],
        ts: &[f64],
        edge_offset: u64,
        sample_size: usize,
    ) {
        let node_time = if !self.node_times.is_empty() {
            self.node_times[node_local as usize]
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
                if take == 0 {
                    return;
                }
                if take < valid {
                    // select_nth_unstable: O(n) partial sort — only partition, no full sort
                    self.temporal_filtered
                        .select_nth_unstable_by(take - 1, |a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                }
                for &(csr_idx, _) in &self.temporal_filtered[..take] {
                    self.sample_buf.push((neighbors[csr_idx], csr_idx));
                }
            }
        }

        // Emit from sample_buf with time recording — single pass, no redundant lookups
        for j in 0..self.sample_buf.len() {
            let (neighbor, csr_idx) = self.sample_buf[j];
            self.edge_src_buf.push(node);
            self.edge_dst_buf.push(neighbor);
            self.src_local_buf.push(node_local);
            if self.config.track_edge_ids {
                self.edge_ids_buf.push(edge_offset + csr_idx as u64);
            }
            let (dst_local, is_new) = self.insert_node_frontier(neighbor);
            self.dst_local_buf.push(dst_local);
            if is_new {
                // Use ts directly (already borrowed above, same lifetime)
                self.node_times.push(ts[csr_idx]);
            }
        }
    }

    /// Normal (non-temporal) sampling for a single node: weighted or unweighted.
    #[inline]
    fn sample_normal(
        &mut self,
        node: NodeId,
        node_local: u32,
        neighbors: &[NodeId],
        weights: Option<&[f32]>,
        edge_offset: u64,
        sample_size: usize,
    ) {
        if let Some(w) = weights {
            self.sample_buf.clear();
            if self.config.replace {
                self.weighted_sample_with_replacement_into(neighbors, w, sample_size);
            } else {
                self.weighted_sample_without_replacement_into(neighbors, w, sample_size);
            }
            let track = self.config.track_edge_ids;
            for i in 0..self.sample_buf.len() {
                let (neighbor, local_idx) = self.sample_buf[i];
                self.edge_src_buf.push(node);
                self.edge_dst_buf.push(neighbor);
                self.src_local_buf.push(node_local);
                if track {
                    self.edge_ids_buf.push(edge_offset + local_idx as u64);
                }
                let (dst_local, _) = self.insert_node_frontier(neighbor);
                self.dst_local_buf.push(dst_local);
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
                self.emit_sample_replace(node, node_local, effective, edge_offset, sample_size);
            } else if sample_size >= n {
                self.emit_take_all(node, node_local, effective, edge_offset);
            } else {
                self.emit_sample_floyd(node, node_local, effective, edge_offset, sample_size);
            }
        }
    }

    /// Push one sampled edge (with its emit-time local endpoint indices) to
    /// the output buffers.
    #[inline(always)]
    fn emit_edge(
        &mut self,
        src: NodeId,
        src_local: u32,
        neighbor: NodeId,
        edge_offset: u64,
        idx: usize,
        track: bool,
    ) {
        self.edge_src_buf.push(src);
        self.edge_dst_buf.push(neighbor);
        self.src_local_buf.push(src_local);
        if track {
            self.edge_ids_buf.push(edge_offset + idx as u64);
        }
        let (dst_local, _) = self.insert_node_frontier(neighbor);
        self.dst_local_buf.push(dst_local);
    }

    /// Emit edges for sampling with replacement — pushes directly to edge buffers.
    #[inline]
    fn emit_sample_replace(
        &mut self,
        src: NodeId,
        src_local: u32,
        neighbors: &[NodeId],
        edge_offset: u64,
        k: usize,
    ) {
        let n = neighbors.len() as u64;
        let track = self.config.track_edge_ids;
        for _ in 0..k {
            let x = self.rng.next_u32();
            let idx = ((x as u64).wrapping_mul(n) >> 32) as usize;
            self.emit_edge(src, src_local, neighbors[idx], edge_offset, idx, track);
        }
    }

    /// Emit edges for take-all (fanout >= degree) — pushes directly to edge buffers.
    #[inline]
    fn emit_take_all(
        &mut self,
        src: NodeId,
        src_local: u32,
        neighbors: &[NodeId],
        edge_offset: u64,
    ) {
        let track = self.config.track_edge_ids;
        for (idx, &neighbor) in neighbors.iter().enumerate() {
            self.emit_edge(src, src_local, neighbor, edge_offset, idx, track);
        }
    }

    /// Floyd's O(k) sampling without replacement — pushes directly to edge buffers.
    /// Uses generation-stamped scratch for n <= 256, reusable HashSet otherwise.
    fn emit_sample_floyd(
        &mut self,
        src: NodeId,
        src_local: u32,
        neighbors: &[NodeId],
        edge_offset: u64,
        k: usize,
    ) {
        let n = neighbors.len();
        let track = self.config.track_edge_ids;

        if n <= 256 {
            // Stamp reuse is a counter bump, not an O(n) clear per node.
            self.floyd_gen = self.floyd_gen.wrapping_add(1);
            if self.floyd_gen == 0 {
                self.floyd_stamp.fill(0);
                self.floyd_gen = 1;
            }
            let stamp = self.floyd_gen;
            for i in (n - k)..n {
                let s = (i + 1) as u64;
                let x = self.rng.next_u32();
                let j = ((x as u64).wrapping_mul(s) >> 32) as usize;
                let pick = if self.floyd_stamp[j] != stamp {
                    self.floyd_stamp[j] = stamp;
                    j
                } else {
                    self.floyd_stamp[i] = stamp;
                    i
                };
                self.emit_edge(src, src_local, neighbors[pick], edge_offset, pick, track);
            }
        } else {
            self.seen_set.clear();
            for i in (n - k)..n {
                let s = (i + 1) as u64;
                let x = self.rng.next_u32();
                let j = ((x as u64).wrapping_mul(s) >> 32) as usize;
                let pick = if self.seen_set.insert(j) {
                    j
                } else {
                    self.seen_set.insert(i);
                    i
                };
                self.emit_edge(src, src_local, neighbors[pick], edge_offset, pick, track);
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

        // Build cumulative distribution in the reusable buffer.
        self.cumsum_buf.clear();
        let mut total = 0.0f64;
        for &w in weights {
            total += w as f64;
            self.cumsum_buf.push(total);
        }

        for _ in 0..k {
            let u = (self.rng.next_u64() as f64) / (u64::MAX as f64) * total;
            let idx = self.cumsum_buf.partition_point(|&c| c <= u).min(n - 1);
            self.sample_buf.push((neighbors[idx], idx));
        }
    }

    /// Weighted sampling without replacement (Efraimidis-Spirakis) — pushes into self.sample_buf.
    ///
    /// Uses the pre-allocated `weighted_keys` buffer. Key computation uses
    /// `fast_neg_ln_u64`, which computes `-ln(u)` exactly so the sampling is unbiased.
    fn weighted_sample_without_replacement_into(
        &mut self,
        neighbors: &[NodeId],
        weights: &[f32],
        k: usize,
    ) {
        let n = neighbors.len();
        debug_assert_eq!(n, weights.len());

        if k == 0 {
            return;
        }
        if k >= n {
            for (i, &neighbor) in neighbors.iter().enumerate() {
                self.sample_buf.push((neighbor, i));
            }
            return;
        }

        // Reuse pre-allocated buffer
        self.weighted_keys.clear();
        self.weighted_keys.reserve(n);

        // Compute keys: key[i] = -ln(u) / w[i], u ~ Uniform(0,1).
        // fast_neg_ln_u64 computes -ln(u) exactly, so the keys are unbiased.
        //
        // Guard against non-finite weights (NaN, ±inf): NaN > 0.0 is false so
        // it routes to INFINITY, but +inf > 0.0 is true and would produce
        // INFINITY / INFINITY = NaN, which makes `partial_cmp` return None and
        // poisons the sort. Require strictly finite, strictly positive weight.
        for (i, &weight) in weights[..n].iter().enumerate() {
            let u_bits = self.rng.next_u64();
            let w = weight as f64;
            let key = if w.is_finite() && w > 0.0 {
                fast_neg_ln_u64(u_bits) / w
            } else {
                f64::INFINITY
            };
            self.weighted_keys.push((key, i));
        }

        // O(n) partial sort — only partitions around the k-th element.
        // Fall back to Equal on any unexpected NaN so the comparator stays a
        // total order; the finite-weight guard above should already prevent it.
        self.weighted_keys.select_nth_unstable_by(k - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });

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

    /// Local index map: global NodeId → position in `nodes` vec. Only
    /// populated for subgraphs built via [`SampledSubgraph::from_parts`]
    /// (external reconstruction) — sampler-produced subgraphs carry
    /// pre-computed local indices instead.
    local_index: FxHashMap<NodeId, u32>,

    /// Per-node batch assignment (disjoint mode only). Maps each node to its seed index.
    pub batch: Option<Vec<u32>>,

    /// Pre-computed local edge indices, recorded at emit time during
    /// sampling. Always present on sampler-produced subgraphs.
    precomputed_local: Option<(Vec<u32>, Vec<u32>)>,

    /// Local indices of the seeds, captured at registration. Present on
    /// non-disjoint sampler-produced subgraphs.
    seed_locals: Option<Vec<u32>>,
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
        let mut local_index = FxHashMap::with_capacity_and_hasher(nodes.len(), Default::default());
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
            seed_locals: None,
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
    /// Sampler-produced subgraphs carry local indices recorded at emit time,
    /// so this is a zero-copy move; the per-edge HashMap fallback only runs
    /// for subgraphs reconstructed via [`SampledSubgraph::from_parts`].
    ///
    /// # Errors
    /// Returns an error if any edge endpoint is not found in the local index,
    /// which indicates a bug in the sampling algorithm.
    pub fn edge_index_local(&mut self) -> Result<(Vec<u32>, Vec<u32>), String> {
        // Fast path: local indices were recorded during sampling — take them
        // (zero-copy move).
        if let Some(precomputed) = self.precomputed_local.take() {
            return Ok(precomputed);
        }

        let mut src_local = Vec::with_capacity(self.edge_src.len());
        for src in &self.edge_src {
            match self.local_index.get(src) {
                Some(&idx) => src_local.push(idx),
                None => return Err(format!("edge src {} not in local_index", src)),
            }
        }

        let mut dst_local = Vec::with_capacity(self.edge_dst.len());
        for dst in &self.edge_dst {
            match self.local_index.get(dst) {
                Some(&idx) => dst_local.push(idx),
                None => return Err(format!("edge dst {} not in local_index", dst)),
            }
        }

        Ok((src_local, dst_local))
    }

    /// Compute local seed indices (position of each seed in the nodes array).
    /// Useful for identifying which nodes in the subgraph were the original seeds.
    ///
    /// Sampler-produced subgraphs return the indices captured at seed
    /// registration; the HashMap fallback only runs for subgraphs built via
    /// [`SampledSubgraph::from_parts`].
    ///
    /// # Errors
    /// Returns an error if any seed is not found in the local index.
    pub fn seed_indices_local(&self) -> Result<Vec<u32>, String> {
        if let Some(ref locals) = self.seed_locals {
            return Ok(locals.clone());
        }
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

    /// Sample neighborhoods for multiple batches.
    ///
    /// When `config.deterministic == false` (default), batches are sampled
    /// in parallel via Rayon. The same seed produces statistically
    /// equivalent results across runs but not bit-identical output —
    /// rayon's work-stealing scheduler can permute order.
    ///
    /// When `config.deterministic == true`, batches are sampled
    /// **serially** in the order given. Output is byte-identical across
    /// runs and machines. Expect 2–8× lower throughput; use only for
    /// regression tests, audit logs, or strict reproducibility.
    pub fn sample_batches(&self, batches: &[Vec<NodeId>]) -> Vec<SampledSubgraph> {
        if self.config.deterministic {
            // Serial path stays byte-identical: one sampler reused in order.
            let mut sampler = NeighborSampler::new(self.graph, self.config.clone());
            batches
                .iter()
                .map(|seeds| sampler.sample_neighbors(seeds))
                .collect()
        } else {
            // One sampler per chunk, processed serially within the chunk.
            // `map_init` runs its init once per rayon *split* (which
            // work-stealing multiplies), and each construction allocates and
            // zeroes the num_nodes-sized dedup array in direct mode; chunking
            // caps constructions at the thread count while keeping order.
            let threads = rayon::current_num_threads().max(1);
            let chunk = batches.len().div_ceil(threads).max(1);
            let grouped: Vec<Vec<SampledSubgraph>> = batches
                .par_chunks(chunk)
                .map(|group| {
                    let mut sampler = NeighborSampler::new(self.graph, self.config.clone());
                    group
                        .iter()
                        .map(|seeds| sampler.sample_neighbors(seeds))
                        .collect()
                })
                .collect();
            grouped.into_iter().flatten().collect()
        }
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
            deterministic: false,
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
            deterministic: false,
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
            deterministic: false,
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
            deterministic: false,
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
            deterministic: false,
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
            deterministic: false,
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
            deterministic: false,
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
            deterministic: false,
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
            deterministic: false,
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
            deterministic: false,
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
            deterministic: false,
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
            deterministic: false,
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
        let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)];
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
            deterministic: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors(&[0]);

        assert_eq!(subgraph.num_edges(), 3);
        // Without replacement: all sampled destinations must be unique
        let neighbors: FxHashSet<_> = subgraph.edge_dst.iter().copied().collect();
        assert_eq!(
            neighbors.len(),
            3,
            "weighted without replacement should produce unique neighbors"
        );
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
            deterministic: false,
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
        graph
            .set_timestamps(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
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
            deterministic: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors_temporal(&[0], &[2.5]).unwrap();

        // Only edges with t < 2.5 should be sampled: 0->1 and 0->2
        assert_eq!(subgraph.num_edges(), 2);
        let dsts: FxHashSet<_> = subgraph.edge_dst.iter().copied().collect();
        assert!(dsts.contains(&1), "edge 0->1 (t=1.0) should be sampled");
        assert!(dsts.contains(&2), "edge 0->2 (t=2.0) should be sampled");
        assert!(
            !dsts.contains(&3),
            "edge 0->3 (t=3.0) should NOT be sampled (t >= 2.5)"
        );
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
            deterministic: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors_temporal(&[0], &[3.5]).unwrap();

        assert_eq!(subgraph.num_edges(), 2);
        let dsts: FxHashSet<_> = subgraph.edge_dst.iter().copied().collect();
        // Last strategy should pick the 2 most recent valid edges
        assert!(
            dsts.contains(&3),
            "edge 0->3 (t=3.0, most recent) should be sampled"
        );
        assert!(
            dsts.contains(&2),
            "edge 0->2 (t=2.0, second most recent) should be sampled"
        );
        assert!(
            !dsts.contains(&1),
            "edge 0->1 (t=1.0, oldest) should NOT be sampled"
        );
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
            deterministic: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors_temporal(&[0], &[0.5]).unwrap();

        // No edges should be sampled since all timestamps >= 0.5 and the earliest is 1.0
        assert_eq!(subgraph.num_edges(), 0);
        assert_eq!(subgraph.num_nodes(), 1); // Only the seed
    }

    #[test]
    fn test_temporal_propagates_time() {
        // Build graph: 0->1 (t=10.0), 0->2 (t=20.0), 1->3 (t=5.0), 1->4 (t=15.0)
        let edges = vec![(0, 1), (0, 2), (1, 3), (1, 4)];
        let mut graph = Graph::from_edges(5, &edges, None).unwrap();
        graph.set_timestamps(vec![10.0, 20.0, 5.0, 15.0]).unwrap();

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
            deterministic: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors_temporal(&[0], &[25.0]).unwrap();

        // Hop 1 edges: 0->1, 0->2
        // Hop 2 edges from node 1 (time=10.0): only 1->3 (t=5.0 < 10.0)
        // Node 4 should NOT appear because 1->4 has t=15.0 >= 10.0
        let all_nodes: FxHashSet<_> = subgraph.nodes.iter().copied().collect();
        assert!(all_nodes.contains(&0), "seed should be present");
        assert!(all_nodes.contains(&1), "hop-1 neighbor should be present");
        assert!(
            all_nodes.contains(&3),
            "hop-2 neighbor via edge t=5<10 should be present"
        );
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
        graph.set_timestamps(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

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
            deterministic: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors_temporal(&[0], &[100.0]).unwrap();

        // All 5 edges have t < 100.0, but fanout=2 so exactly 2 should be sampled
        assert_eq!(
            subgraph.num_edges(),
            2,
            "fanout=2 should produce exactly 2 edges"
        );
        // Sampled destinations should be valid neighbors
        for &dst in &subgraph.edge_dst {
            assert!([1u32, 2, 3, 4, 5].contains(&dst));
        }
        // Without replacement, the 2 destinations should be unique
        let dsts: FxHashSet<_> = subgraph.edge_dst.iter().copied().collect();
        assert_eq!(
            dsts.len(),
            2,
            "without replacement, destinations should be unique"
        );
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
            deterministic: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors_disjoint(&[0, 1], None);

        // batch should be Some and have one entry per node
        assert!(
            subgraph.batch.is_some(),
            "disjoint mode should produce a batch vector"
        );
        let batch = subgraph.batch.as_ref().unwrap();
        assert_eq!(
            batch.len(),
            subgraph.nodes.len(),
            "batch length should equal number of nodes"
        );

        // Every batch value should be 0 or 1 (2 seeds)
        for &b in batch {
            assert!(
                b == 0 || b == 1,
                "batch values should map to seed indices 0 or 1"
            );
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
            deterministic: false,
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

        assert!(
            seed0_set.contains(&2) && seed0_set.contains(&3),
            "seed 0 should have neighbors 2 and 3"
        );
        assert!(
            seed1_set.contains(&2) && seed1_set.contains(&3),
            "seed 1 should have neighbors 2 and 3"
        );

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
            deterministic: false,
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
            assert!(
                s < num_nodes,
                "local src index {} should be < {}",
                s,
                num_nodes
            );
        }
        for &d in dst_local {
            assert!(
                d < num_nodes,
                "local dst index {} should be < {}",
                d,
                num_nodes
            );
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
        graph
            .set_timestamps(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();

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
            deterministic: false,
            telemetry: None,
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        // Seed 0 with time 2.5: only 0->1 (t=1.0) and 0->2 (t=2.0) valid
        // Seed 1 with time 4.5: only 1->2 (t=4.0) valid (1->3 has t=5.0 >= 4.5)
        let subgraph = sampler.sample_neighbors_disjoint(&[0, 1], Some(&[2.5, 4.5]));

        // batch should exist
        assert!(
            subgraph.batch.is_some(),
            "disjoint+temporal should produce batch"
        );
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
        assert!(
            seed0_dsts.contains(&1),
            "seed 0 should sample 0->1 (t=1.0 < 2.5)"
        );
        assert!(
            seed0_dsts.contains(&2),
            "seed 0 should sample 0->2 (t=2.0 < 2.5)"
        );
        assert!(
            !seed0_dsts.contains(&3),
            "seed 0 should NOT sample 0->3 (t=3.0 >= 2.5)"
        );

        // Seed 1 (time=4.5): edges 1->2 (t=4.0) valid, 1->3 (t=5.0) NOT valid
        assert!(
            seed1_dsts.contains(&2),
            "seed 1 should sample 1->2 (t=4.0 < 4.5)"
        );
        assert!(
            !seed1_dsts.contains(&3),
            "seed 1 should NOT sample 1->3 (t=5.0 >= 4.5)"
        );
    }

    #[test]
    fn parallel_batch_sampler_deterministic_mode_is_bit_identical() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2, 2],
            seed: Some(0xCAFE_F00D),
            deterministic: true,
            ..Default::default()
        };
        let batches: Vec<Vec<NodeId>> = vec![vec![0, 1], vec![2, 3], vec![4]];
        let s = ParallelBatchSampler::new(&graph, config.clone());

        let run1 = s.sample_batches(&batches);
        let run2 = s.sample_batches(&batches);
        assert_eq!(run1.len(), run2.len());
        for (a, b) in run1.iter().zip(run2.iter()) {
            assert_eq!(a.nodes, b.nodes, "deterministic mode: nodes diverge");
            assert_eq!(
                a.edge_src, b.edge_src,
                "deterministic mode: edge_src diverges"
            );
            assert_eq!(
                a.edge_dst, b.edge_dst,
                "deterministic mode: edge_dst diverges"
            );
        }
    }
}
