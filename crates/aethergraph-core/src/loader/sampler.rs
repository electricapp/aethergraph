//! Neighborhood sampling for GNN training.
//!
//! Internal scratch (frontiers, dedup arrays, sample buffers) is pre-allocated
//! at sampler construction and reset across `sample_neighbors` calls via
//! `clear()`. The output buffers handed back in each `SampledSubgraph` are
//! freshly allocated per call (swapped out of the sampler). Local node indices
//! are assigned during sampling — no post-sort, no binary search.

use crate::graph::{Graph, NodeId};
use crate::internal::genstamp::{FRONTIER_PREFETCH_DIST, FloydStamps, GenDedup, GenSlots, WyRand};
use crate::internal::telemetry::{SamplingTelemetry, SamplingTimer};
use crate::loader::planned_capacity;

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
use std::borrow::Cow;
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
    /// Node dedup: dense generation-stamped slots (<= 100K nodes, or a
    /// sample estimated to touch > 1% of the graph — one cache line per
    /// probe) or an FxHashMap for sparser sampling patterns.
    dedup: GenDedup,
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
    floyd: FloydStamps,
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

/// Floor capacities for the output buffers swapped in after a sampling call
/// hands its filled ones to the caller. Above these floors the replacement is
/// sized from the count the call just produced.
const MIN_NODE_CAPACITY: usize = 512;
const MIN_EDGE_CAPACITY: usize = 2048;

/// Largest graph that gets the dense per-node dedup table, at one `u64` slot
/// per node — a 64 MiB allocation at this bound. Past it the sampler switches
/// to a hash map, whose footprint follows the sample rather than the graph.
const DENSE_DEDUP_MAX_NODES: usize = 8 << 20;

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

        // Dense dedup probes one u64 slot with a single load and resets with a
        // generation bump; the map pays a hash and a control-byte scan per
        // probe. The dense table wins at every sample density measured,
        // including a 16-seed one-hop batch on a 4M-node graph — the case a
        // density rule would hand to the map, where dense tracks a few hundred
        // nodes inside a 32 MB allocation and is still faster. The table is
        // allocated zero-filled, so slots the sample never touches are never
        // faulted in: what an oversized table costs is address space, not
        // resident memory.
        //
        // The bound is therefore a memory decision rather than a speed one.
        // Scattered probes into a large table fault a full page per distinct
        // node touched, so worst-case residency is the whole table, and a
        // sampler is constructed per rayon chunk — the table is multiplied by
        // the thread count.
        let num_nodes = graph.num_nodes();
        let use_direct = num_nodes <= DENSE_DEDUP_MAX_NODES;
        Self {
            graph,
            config,
            rng,
            dedup: if use_direct {
                GenDedup::Dense(GenSlots::new(num_nodes))
            } else {
                GenDedup::Map(FxHashMap::with_capacity_and_hasher(512, Default::default()))
            },
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
            floyd: FloydStamps::new(),
            seen_set: FxHashSet::with_capacity_and_hasher(max_fanout * 2, Default::default()),
            node_times: Vec::new(),
            temporal_filtered: Vec::with_capacity(max_fanout),
            weighted_keys: Vec::with_capacity(256),
            cumsum_buf: Vec::with_capacity(256),
        }
    }

    #[inline(always)]
    fn insert_node(&mut self, id: NodeId) -> u32 {
        let (idx, is_new) = self.dedup.probe_or_insert(id, self.node_vec.len() as u32);
        if is_new {
            self.node_vec.push(id);
        }
        idx
    }

    /// Register `id`, pushing it onto the next frontier when new. Returns
    /// its local index either way — the caller records it for the edge's
    /// endpoint without any later lookup.
    #[inline(always)]
    fn insert_node_frontier(&mut self, id: NodeId) -> (u32, bool) {
        let (idx, is_new) = self.dedup.probe_or_insert(id, self.node_vec.len() as u32);
        if is_new {
            self.node_vec.push(id);
            self.next_frontier.push((id, idx));
        }
        (idx, is_new)
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
        let mut all_seed_locals = Vec::with_capacity(seeds.len());
        // Per-hop counts accumulated across every seed (PyG compatibility).
        let mut num_sampled_nodes = vec![0usize; num_hops];
        let mut num_sampled_edges = vec![0usize; num_hops];

        for (seed_idx, &seed) in seeds.iter().enumerate() {
            // Reset dedup state per seed (cheap: no dealloc, just counter bump or clear)
            self.dedup.begin();
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
            // block offset needs applying while concatenating. The seed is
            // the first node registered in its block, so its combined-array
            // local index is the block offset plus its within-block index.
            let node_offset = all_nodes.len() as u32;
            all_seed_locals.push(node_offset + seed_local);
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
            locals: Locals::Recorded {
                src: all_src_local,
                dst: all_dst_local,
                seeds: all_seed_locals,
            },
            batch: Some(batch),
        }
    }

    /// Run the hop loop over the current frontier, invoking `record` with
    /// `(hop, new_frontier_nodes, new_edges)` after each hop.
    ///
    /// A `CsrView` is hoisted once per call so the inner loop indexes raw
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

        let csr = self.graph.csr_view();
        let offsets = csr.offsets();
        let edges = csr.edges();
        let num_nodes = csr.num_nodes();
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
                let range = csr.neighbor_range(node);
                if range.is_empty() {
                    continue;
                }
                let (start, end) = (range.start, range.end);
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

        self.dedup.begin();
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
        //
        // Each replacement is sized from what this call just produced — the
        // buffer's own length, read before it is swapped away. A sampler is
        // reused across batches of near-identical shape, so that count
        // predicts the next call far better than a fixed constant does: a
        // three-hop batch emits hundreds of thousands of edges, and starting
        // each of the five parallel edge arrays at a small constant climbs a
        // realloc-and-copy ladder on every call.
        let node_capacity = planned_capacity(self.node_vec.len(), MIN_NODE_CAPACITY);
        let edge_capacity = planned_capacity(self.edge_src_buf.len(), MIN_EDGE_CAPACITY);

        let mut node_vec = Vec::with_capacity(node_capacity);
        std::mem::swap(&mut self.node_vec, &mut node_vec);

        let mut edge_src = Vec::with_capacity(edge_capacity);
        std::mem::swap(&mut self.edge_src_buf, &mut edge_src);

        let mut edge_dst = Vec::with_capacity(edge_capacity);
        std::mem::swap(&mut self.edge_dst_buf, &mut edge_dst);

        let mut edge_ids = Vec::with_capacity(if self.config.track_edge_ids {
            edge_capacity
        } else {
            0
        });
        std::mem::swap(&mut self.edge_ids_buf, &mut edge_ids);

        let mut src_local = Vec::with_capacity(edge_capacity);
        std::mem::swap(&mut self.src_local_buf, &mut src_local);

        let mut dst_local = Vec::with_capacity(edge_capacity);
        std::mem::swap(&mut self.dst_local_buf, &mut dst_local);

        // Exactly one local index per seed, so this size is known, not predicted.
        let mut seed_locals = Vec::with_capacity(seeds.len());
        std::mem::swap(&mut self.seed_locals_buf, &mut seed_locals);

        // Apply subgraph_type post-processing. Local endpoint indices were
        // recorded at emit time, so no per-edge map lookup happens on any
        // path; the Induced filter builds a transient membership map for
        // itself alone and drops it once the edges are filtered.
        let (edge_src, edge_dst, edge_ids, src_local, dst_local) = match self.config.subgraph_type {
            // Induced keeps edges whose endpoints are both in the sampled
            // node set, which every emitted edge already satisfies: a source
            // is a frontier node, and a destination is registered by
            // `insert_node_frontier` in the same step that pushes the edge.
            // The set membership test is therefore decided at emit time, and
            // re-deciding it here would cost a node-set build plus two hash
            // probes per edge to retain every edge. `induced_keeps_every_edge`
            // pins the invariant.
            SubgraphType::Directional | SubgraphType::Induced => {
                (edge_src, edge_dst, edge_ids, src_local, dst_local)
            }
            SubgraphType::Bidirectional => {
                // Append the reverse run onto the forward one, rather than
                // allocating five fresh vectors and copying every array
                // twice. Each append reads the *other* array's forward
                // prefix, which stays put: the new elements land past it, so
                // `..n` still names the forward run on the second read.
                let n = edge_src.len();
                edge_src.reserve(n);
                edge_dst.reserve(n);
                src_local.reserve(n);
                dst_local.reserve(n);

                edge_src.extend_from_slice(&edge_dst[..n]);
                edge_dst.extend_from_slice(&edge_src[..n]);
                src_local.extend_from_slice(&dst_local[..n]);
                dst_local.extend_from_slice(&src_local[..n]);
                if self.config.track_edge_ids {
                    // A reverse edge carries its forward edge's id.
                    edge_ids.extend_from_within(..n);
                }

                (edge_src, edge_dst, edge_ids, src_local, dst_local)
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
            locals: Locals::Recorded {
                src: src_local,
                dst: dst_local,
                seeds: seed_locals,
            },
            batch: None,
        };

        // Record telemetry if enabled (opt-in, zero overhead if None)
        if let Some(ref telemetry) = self.config.telemetry {
            telemetry.record_sample(
                subgraph.num_nodes() as u64,
                subgraph.num_edges() as u64,
                _timer.elapsed(),
            );
        }

        // Unlike the telemetry above, this fires whether or not anything
        // is configured: an unattached probe is one nop.
        crate::probe!(
            sample_batch_done,
            subgraph.num_seeds(),
            subgraph.num_nodes(),
            subgraph.num_edges(),
        );

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
                        let j = (u64::from(x).wrapping_mul(s) >> 32) as usize;
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
                    crate::probe!(hub_node_capped, neighbors.len(), max_deg);
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
            let idx = (u64::from(x).wrapping_mul(n) >> 32) as usize;
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
                self.emit_edge(src, src_local, neighbors[pick], edge_offset, pick, track);
            }
        } else {
            self.seen_set.clear();
            for i in (n - k)..n {
                let s = (i + 1) as u64;
                let x = self.rng.next_u32();
                let j = (u64::from(x).wrapping_mul(s) >> 32) as usize;
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
            total += f64::from(w);
            self.cumsum_buf.push(total);
        }

        // One reciprocal for the whole node instead of a divide per draw:
        // `total / u64::MAX` is loop-invariant, and a divide costs several
        // times a multiply on every target here.
        let scale = total / (u64::MAX as f64);
        for _ in 0..k {
            let u = (self.rng.next_u64() as f64) * scale;
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
            let w = f64::from(weight);
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

/// Local endpoint index pair returned by
/// [`SampledSubgraph::edge_index_local`]: `(src, dst)`, parallel to
/// `edge_src`/`edge_dst`. Recorded indices are borrowed from the subgraph;
/// indices derived for [`SampledSubgraph::from_parts`] subgraphs are owned.
pub type LocalEdgeIndex<'a> = (Cow<'a, [u32]>, Cow<'a, [u32]>);

/// Local-index data carried by a [`SampledSubgraph`], discriminated by
/// provenance.
///
/// Every sampler path (directional, induced, bidirectional, disjoint)
/// records local indices at emit time and produces [`Locals::Recorded`];
/// [`SampledSubgraph::from_parts`] produces [`Locals::Lazy`], whose
/// accessors derive the indices transiently per call from the `nodes`
/// array. Accessors behave identically on repeated calls for both variants.
#[derive(Debug, Clone)]
enum Locals {
    /// Local indices recorded during sampling: per-edge endpoint indices
    /// (`src`/`dst`, parallel to `edge_src`/`edge_dst`) and per-seed
    /// indices (`seeds`, parallel to the public `seeds` array).
    Recorded {
        src: Vec<u32>,
        dst: Vec<u32>,
        seeds: Vec<u32>,
    },
    /// No recorded indices; accessors build the global-to-local mapping
    /// inside the call and store nothing.
    Lazy,
}

/// A sampled subgraph containing nodes and edges.
///
/// Nodes are stored in discovery order (not sorted). Sampler-produced
/// subgraphs carry local endpoint and seed indices recorded at emit time;
/// subgraphs rebuilt via [`SampledSubgraph::from_parts`] derive them on
/// demand.
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

    /// Local-index data, discriminated by provenance (see [`Locals`]).
    locals: Locals,

    /// Per-node batch assignment (disjoint mode only). Maps each node to its seed index.
    pub batch: Option<Vec<u32>>,
}

impl SampledSubgraph {
    /// Construct a SampledSubgraph from its public fields.
    ///
    /// This is intended for external callers (e.g., PyO3 round-trip) that
    /// need to reconstruct a SampledSubgraph. The result carries no recorded
    /// local indices (`Locals::Lazy`); accessors derive them per call.
    pub fn from_parts(
        nodes: Vec<NodeId>,
        edge_src: Vec<NodeId>,
        edge_dst: Vec<NodeId>,
        edge_ids: Vec<u64>,
        seeds: Vec<NodeId>,
        num_sampled_nodes: Vec<usize>,
        num_sampled_edges: Vec<usize>,
    ) -> Self {
        Self {
            nodes,
            edge_src,
            edge_dst,
            edge_ids,
            seeds,
            num_sampled_nodes,
            num_sampled_edges,
            locals: Locals::Lazy,
            batch: None,
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

    /// No-op for backward compatibility. Nodes are in discovery order and
    /// local indices are recorded at emit time (or derived on demand for
    /// [`SampledSubgraph::from_parts`] subgraphs). Kept for API compatibility.
    pub fn sort_nodes(&mut self) {
        // Intentionally empty.
    }

    /// Transient global-to-local map for [`Locals::Lazy`] subgraphs, built
    /// inside the accessor call and dropped with it.
    fn lazy_local_index(&self) -> FxHashMap<NodeId, u32> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i as u32))
            .collect()
    }

    /// Edge indices with local (remapped) node IDs in `[0, num_nodes)`.
    ///
    /// Sampler-produced subgraphs return borrowed slices of the indices
    /// recorded at emit time; subgraphs reconstructed via
    /// [`SampledSubgraph::from_parts`] derive owned vectors from a transient
    /// per-call map. Repeated calls return the same answer on every
    /// provenance.
    ///
    /// # Errors
    /// Only [`SampledSubgraph::from_parts`] subgraphs can fail, when an edge
    /// endpoint is missing from `nodes` — inconsistent reconstruction input.
    pub fn edge_index_local(&self) -> Result<LocalEdgeIndex<'_>, String> {
        match &self.locals {
            Locals::Recorded { src, dst, .. } => {
                Ok((Cow::Borrowed(&src[..]), Cow::Borrowed(&dst[..])))
            }
            Locals::Lazy => {
                let local_index = self.lazy_local_index();

                let mut src_local = Vec::with_capacity(self.edge_src.len());
                for src in &self.edge_src {
                    match local_index.get(src) {
                        Some(&idx) => src_local.push(idx),
                        None => return Err(format!("edge src {src} not in subgraph nodes")),
                    }
                }

                let mut dst_local = Vec::with_capacity(self.edge_dst.len());
                for dst in &self.edge_dst {
                    match local_index.get(dst) {
                        Some(&idx) => dst_local.push(idx),
                        None => return Err(format!("edge dst {dst} not in subgraph nodes")),
                    }
                }

                Ok((Cow::Owned(src_local), Cow::Owned(dst_local)))
            }
        }
    }

    /// Local seed indices (position of each seed in the nodes array).
    /// Useful for identifying which nodes in the subgraph were the original seeds.
    ///
    /// Sampler-produced subgraphs (disjoint included) return the indices
    /// recorded at seed registration; subgraphs built via
    /// [`SampledSubgraph::from_parts`] derive them from a transient per-call
    /// map. Repeated calls return the same answer on every provenance.
    ///
    /// # Errors
    /// Only [`SampledSubgraph::from_parts`] subgraphs can fail, when a seed
    /// is missing from `nodes` — inconsistent reconstruction input.
    pub fn seed_indices_local(&self) -> Result<Vec<u32>, String> {
        match &self.locals {
            Locals::Recorded { seeds, .. } => Ok(seeds.clone()),
            Locals::Lazy => {
                let local_index = self.lazy_local_index();
                self.seeds
                    .iter()
                    .map(|id| {
                        local_index
                            .get(id)
                            .copied()
                            .ok_or_else(|| format!("seed {id} not in subgraph nodes"))
                    })
                    .collect()
            }
        }
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

    /// Every emitted endpoint is registered in the node set as the edge is
    /// pushed, so the Induced membership filter can never drop an edge.
    /// `sample_neighbors_inner` relies on this to skip the filter entirely;
    /// if an emit path ever pushes an edge without registering both
    /// endpoints, this fails instead of silently returning stale edges.
    #[test]
    fn induced_keeps_every_edge() {
        let mut edges = Vec::new();
        for src in 0..200u32 {
            for step in 1..=7u32 {
                edges.push((src, (src * 7 + step * 13) % 200));
            }
        }
        let graph = Graph::from_edges(200, &edges, None).unwrap();

        let base = SamplingConfig {
            fanout: vec![4, 3, 2],
            replace: false,
            seed: Some(7),
            max_degree: None,
            cumulative: false,
            weighted: false,
            subgraph_type: SubgraphType::Induced,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            deterministic: false,
            telemetry: None,
        };
        let seeds: Vec<NodeId> = (0..16).collect();

        let mut induced = NeighborSampler::new(&graph, base.clone());
        let induced = induced.sample_neighbors(&seeds);

        let member: FxHashSet<NodeId> = induced.nodes.iter().copied().collect();
        assert!(!induced.edge_src.is_empty());
        for (&src, &dst) in induced.edge_src.iter().zip(induced.edge_dst.iter()) {
            assert!(member.contains(&src), "source {src} missing from node set");
            assert!(member.contains(&dst), "dest {dst} missing from node set");
        }

        // Same seed, same RNG stream: the two modes must agree edge for edge.
        let directional_cfg = SamplingConfig {
            subgraph_type: SubgraphType::Directional,
            ..base
        };
        let mut directional = NeighborSampler::new(&graph, directional_cfg);
        let directional = directional.sample_neighbors(&seeds);

        assert_eq!(induced.edge_src, directional.edge_src);
        assert_eq!(induced.edge_dst, directional.edge_dst);
        assert_eq!(induced.edge_ids, directional.edge_ids);
    }

    /// Bidirectional had no test at all, and it is the one subgraph mode that
    /// rewrites the emitted arrays rather than passing them through. Length
    /// alone proves nothing here — an append that read the wrong array, or
    /// left the locals unmirrored, still doubles every array. So this pins the
    /// values: the forward run survives untouched, the reverse run is it with
    /// endpoints swapped, ids are shared, and the local indices mirror so
    /// `edge_index_local` stays consistent with `edge_src`/`edge_dst`.
    #[test]
    fn bidirectional_mirrors_every_forward_edge() {
        let mut edges = Vec::new();
        for src in 0..120u32 {
            for step in 1..=5u32 {
                edges.push((src, (src * 11 + step * 7) % 120));
            }
        }
        let graph = Graph::from_edges(120, &edges, None).unwrap();

        let base = SamplingConfig {
            fanout: vec![4, 3],
            replace: false,
            seed: Some(11),
            max_degree: None,
            cumulative: false,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            deterministic: false,
            telemetry: None,
        };
        let seeds: Vec<NodeId> = (0..12).collect();

        let mut forward = NeighborSampler::new(&graph, base.clone());
        let forward = forward.sample_neighbors(&seeds);

        let bidi_cfg = SamplingConfig {
            subgraph_type: SubgraphType::Bidirectional,
            ..base
        };
        let mut bidi = NeighborSampler::new(&graph, bidi_cfg);
        let bidi = bidi.sample_neighbors(&seeds);

        let n = forward.edge_src.len();
        assert!(n > 0, "fixture produced no edges");
        assert_eq!(bidi.edge_src.len(), n * 2);
        assert_eq!(bidi.edge_dst.len(), n * 2);
        assert_eq!(bidi.edge_ids.len(), n * 2);

        // The forward run is untouched.
        assert_eq!(&bidi.edge_src[..n], &forward.edge_src[..]);
        assert_eq!(&bidi.edge_dst[..n], &forward.edge_dst[..]);
        assert_eq!(&bidi.edge_ids[..n], &forward.edge_ids[..]);

        // The reverse run is the forward run with endpoints swapped, sharing
        // each forward edge's id.
        assert_eq!(&bidi.edge_src[n..], &forward.edge_dst[..]);
        assert_eq!(&bidi.edge_dst[n..], &forward.edge_src[..]);
        assert_eq!(&bidi.edge_ids[n..], &forward.edge_ids[..]);

        // Local endpoint indices mirror the same way, so edge_index_local
        // stays consistent with edge_src/edge_dst.
        let (fwd_src_local, fwd_dst_local) = forward.edge_index_local().unwrap();
        let (bidi_src_local, bidi_dst_local) = bidi.edge_index_local().unwrap();
        assert_eq!(&bidi_src_local[..n], &fwd_src_local[..]);
        assert_eq!(&bidi_dst_local[..n], &fwd_dst_local[..]);
        assert_eq!(&bidi_src_local[n..], &fwd_dst_local[..]);
        assert_eq!(&bidi_dst_local[n..], &fwd_src_local[..]);
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
        let subgraph = sampler.sample_neighbors(&[0]);

        let (src_local, dst_local) = subgraph.edge_index_local().unwrap();
        assert_eq!(src_local.len(), subgraph.num_edges());
        assert_eq!(dst_local.len(), subgraph.num_edges());

        // All local indices should be < num_nodes
        for &s in src_local.iter() {
            assert!((s as usize) < subgraph.num_nodes());
        }
        for &d in dst_local.iter() {
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
        let sub1 = sampler.sample_neighbors(&[0]);
        let sub2 = sampler.sample_neighbors(&[1, 2]);

        assert!(sub1.num_nodes() >= 1);
        assert!(sub2.num_nodes() >= 2);
        assert_ne!(sub1.seeds, sub2.seeds);

        // Both should produce valid local indices
        sub1.edge_index_local().unwrap();
        sub2.edge_index_local().unwrap();
    }

    #[test]
    fn test_edge_index_local_repeated_calls_directional() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![3, 2],
            seed: Some(42),
            subgraph_type: SubgraphType::Directional,
            ..Default::default()
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let subgraph = sampler.sample_neighbors(&[0]);

        let first = subgraph.edge_index_local().unwrap();
        let first = (first.0.into_owned(), first.1.into_owned());
        let second = subgraph.edge_index_local().unwrap();
        let second = (second.0.into_owned(), second.1.into_owned());
        assert_eq!(
            first, second,
            "edge_index_local must return the same answer on every call"
        );
        assert_eq!(first.0.len(), subgraph.num_edges());
    }

    #[test]
    fn test_edge_index_local_repeated_calls_from_parts() {
        let subgraph = SampledSubgraph::from_parts(
            vec![10, 20, 30], // nodes
            vec![10, 10, 20], // edge_src
            vec![20, 30, 30], // edge_dst
            vec![0, 1, 2],    // edge_ids
            vec![10],         // seeds
            vec![2],
            vec![3],
        );

        let first = subgraph.edge_index_local().unwrap();
        let first = (first.0.into_owned(), first.1.into_owned());
        let second = subgraph.edge_index_local().unwrap();
        let second = (second.0.into_owned(), second.1.into_owned());
        assert_eq!(
            first, second,
            "edge_index_local must return the same answer on every call"
        );
        assert_eq!(first.0, vec![0, 0, 1]);
        assert_eq!(first.1, vec![1, 2, 2]);

        // Seed indices are likewise repeatable and correct.
        assert_eq!(subgraph.seed_indices_local().unwrap(), vec![0]);
        assert_eq!(subgraph.seed_indices_local().unwrap(), vec![0]);
    }

    #[test]
    fn test_disjoint_recorded_seed_locals_match_batch() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2],
            seed: Some(42),
            disjoint: true,
            ..Default::default()
        };

        let mut sampler = NeighborSampler::new(&graph, config);
        let seeds = [0u32, 1, 2];
        let subgraph = sampler.sample_neighbors_disjoint(&seeds, None);
        let batch = subgraph.batch.as_ref().unwrap();

        let seed_locals = subgraph.seed_indices_local().unwrap();
        assert_eq!(seed_locals.len(), seeds.len());
        for (i, &idx) in seed_locals.iter().enumerate() {
            assert_eq!(
                subgraph.nodes[idx as usize], seeds[i],
                "recorded seed local must point at the seed's node entry"
            );
            assert_eq!(
                batch[idx as usize], i as u32,
                "recorded seed local must land in the seed's own batch block"
            );
            let first_in_block = batch
                .iter()
                .position(|&b| b == i as u32)
                .expect("every seed has a batch block");
            assert_eq!(
                idx as usize, first_in_block,
                "seed must be the first node of its batch block"
            );
        }
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
            "Heavy-weight edge (w=100) should be sampled >50% of the time, got {count_heavy}/{trials}"
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

        let (src_local, dst_local) = subgraph.edge_index_local().unwrap();

        assert_eq!(src_local.len(), subgraph.num_edges());
        assert_eq!(dst_local.len(), subgraph.num_edges());

        // All local indices should be valid (< total nodes in combined subgraph)
        let num_nodes = subgraph.nodes.len() as u32;
        for &s in src_local.iter() {
            assert!(s < num_nodes, "local src index {s} should be < {num_nodes}");
        }
        for &d in dst_local.iter() {
            assert!(d < num_nodes, "local dst index {d} should be < {num_nodes}");
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
        let (src_local, _dst_local) = subgraph.edge_index_local().unwrap();
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
        let s = ParallelBatchSampler::new(&graph, config);

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
