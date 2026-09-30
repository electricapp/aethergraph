//! Neighborhood sampling for GNN training.
//!
//! Internal scratch (frontiers, dedup arrays, sample buffers) is pre-allocated
//! at sampler construction and reset across sampling calls via `clear()`. The
//! output buffers handed back in each `SampledSubgraph` are freshly allocated
//! per call (swapped out of the sampler). Local node indices are assigned
//! during sampling — no post-sort, no binary search.
//!
//! Edges are recorded in stored (CSR) direction: `edge_src` is the node whose
//! row was expanded, `edge_dst` the neighbor drawn from that row.

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
    /// Edges exactly as sampled.
    #[default]
    Directional,

    /// Every graph edge whose endpoints are both in the sampled node set,
    /// sampled or not. Under temporal sampling an edge is kept only when it
    /// predates its source node's time.
    Induced,

    /// Sampled edges plus their reverses, coalesced so each ordered endpoint
    /// pair appears once. A reverse edge carries its forward edge's id.
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
use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use tracing::trace;

/// Derive a per-batch RNG seed so multi-worker prefetch yields the same
/// subgraph for `(base_seed, batch_idx, seeds)` regardless of which worker
/// thread claims the work item.
#[inline]
pub fn batch_seed(base: u64, batch_idx: usize) -> u64 {
    // SplitMix64 finalizer over base XOR index-scaled golden ratio.
    let mut z = base ^ (batch_idx as u64).wrapping_mul(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

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

    /// Degree above which a node counts as a hub.
    ///
    /// Uniform sampling draws in O(fanout) at any degree and never caps.
    /// Weighted and uniform-temporal sampling, whose cost is linear in the
    /// degree, first draw `max_degree` positions uniformly at random from a
    /// hub's row and sample among those. Temporal `Last` always scans the
    /// whole row: the most recent edges can sit anywhere in it.
    /// Default: `None` (never cap).
    pub max_degree: Option<usize>,

    /// Whether every hop re-expands all nodes seen so far rather than only
    /// the previous hop's new nodes.
    ///
    /// - `false` (default): each hop expands only the nodes the previous hop
    ///   discovered — PyG's neighbor-sampling semantics.
    /// - `true`: each hop also re-expands every earlier node, adding only
    ///   edges no earlier hop emitted.
    pub cumulative: bool,

    /// Whether to use edge weights for weighted sampling.
    /// When true, neighbors are sampled proportionally to their edge weights.
    /// Requires the graph to have weights loaded.
    pub weighted: bool,

    /// Type of subgraph to extract (see [`SubgraphType`]).
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

    /// Has no effect on output. [`ParallelBatchSampler`] and the prefetch
    /// loaders reseed per batch from `seed` and the batch's position, so a
    /// fixed `seed` already yields bit-identical output across runs,
    /// machines, and thread counts.
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
            max_degree: None,
            cumulative: false,
            weighted: false,
            subgraph_type: SubgraphType::Directional,
            track_edge_ids: true,
            temporal_strategy: None,
            disjoint: false,
            deterministic: false,
            telemetry: None,
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

/// A seed ID at or past the node count it was checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeedOutOfRange {
    /// Position of the offending ID in the input.
    pub position: usize,
    pub seed: NodeId,
    pub num_nodes: usize,
}

impl std::fmt::Display for SeedOutOfRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "seed node {} (position {}) is out of range for {} nodes",
            self.seed, self.position, self.num_nodes
        )
    }
}

impl std::error::Error for SeedOutOfRange {}

/// Seed node IDs proven to lie in `[0, num_nodes)`.
///
/// Built once where seeds enter the system ([`Seeds::new`]); samplers take it
/// by reference and index their node-sized tables without re-checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seeds {
    ids: Vec<NodeId>,
    num_nodes: usize,
}

impl Seeds {
    /// Check every ID against `num_nodes` — the graph's node count, or the
    /// seed node type's count for heterogeneous sampling.
    ///
    /// # Errors
    /// [`SeedOutOfRange`] naming the first ID at or past `num_nodes`.
    pub fn new(ids: Vec<NodeId>, num_nodes: usize) -> Result<Self, SeedOutOfRange> {
        if let Some(position) = ids.iter().position(|&id| id as usize >= num_nodes) {
            return Err(SeedOutOfRange {
                position,
                seed: ids[position],
                num_nodes,
            });
        }
        Ok(Self { ids, num_nodes })
    }

    /// The checked IDs, in input order (duplicates preserved).
    #[inline]
    pub fn ids(&self) -> &[NodeId] {
        &self.ids
    }

    /// The node count the IDs were checked against.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    #[inline]
    pub fn into_ids(self) -> Vec<NodeId> {
        self.ids
    }
}

/// Errors returned by [`NeighborSampler::sample`] and
/// [`ParallelBatchSampler::sample_batches`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SampleError {
    /// The seeds were checked against more nodes than the sampler's graph has.
    SeedsExceedGraph {
        checked_against: usize,
        num_nodes: usize,
    },
    /// The per-seed times do not fit the sampler's configuration or graph.
    Temporal(TemporalSamplingError),
}

impl std::fmt::Display for SampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SeedsExceedGraph {
                checked_against,
                num_nodes,
            } => write!(
                f,
                "seeds were checked against {checked_against} nodes, but the graph has {num_nodes}"
            ),
            Self::Temporal(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for SampleError {}

impl From<TemporalSamplingError> for SampleError {
    fn from(e: TemporalSamplingError) -> Self {
        Self::Temporal(e)
    }
}

/// A [`SamplingConfig`] that asks for edge data the graph does not carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerConfigError {
    /// `weighted` is set but the graph has no edge weights.
    WeightsMissing,
    /// `temporal_strategy` is set but the graph has no edge timestamps.
    TimestampsMissing,
}

impl std::fmt::Display for SamplerConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WeightsMissing => {
                f.write_str("weighted sampling requested, but the graph has no edge weights")
            }
            Self::TimestampsMissing => f.write_str(
                "temporal sampling requested, but the graph has no edge timestamps — call \
                 Graph::set_timestamps() first",
            ),
        }
    }
}

impl std::error::Error for SamplerConfigError {}

/// Reject a config whose weighted or temporal mode the graph cannot serve.
pub(crate) fn check_config(
    graph: &Graph,
    config: &SamplingConfig,
) -> Result<(), SamplerConfigError> {
    if config.weighted && graph.weights().is_none() {
        return Err(SamplerConfigError::WeightsMissing);
    }
    if config.temporal_strategy.is_some() && !graph.has_timestamps() {
        return Err(SamplerConfigError::TimestampsMissing);
    }
    Ok(())
}

/// A neighborhood sampler for GNN training.
///
/// Internal scratch buffers are pre-allocated at construction and reset across
/// calls; the per-call output buffers are freshly allocated and swapped out
/// into the returned `SampledSubgraph`.
///
/// Dedup uses a generation-tagged direct array for graphs up to
/// `DENSE_DEDUP_MAX_NODES` nodes and an FxHashMap-backed index above it.
pub struct NeighborSampler<'a> {
    graph: &'a Graph,
    config: SamplingConfig,
    rng: WyRand,
    /// Node dedup: dense generation-stamped slots or an FxHashMap.
    dedup: GenDedup,
    /// Nodes in discovery order.
    node_vec: Vec<NodeId>,
    /// Temporal: per-node time bound, pushed in the same step as `node_vec`
    /// whenever `track_times` is set, so the two stay index-parallel.
    node_times: Vec<f64>,
    /// Whether the current pass records per-node times.
    track_times: bool,
    /// Hop the current pass is expanding.
    hop: u32,
    /// Edge source nodes (global IDs).
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
    /// Reusable buffer for weighted and temporal sampling results.
    sample_buf: Vec<(NodeId, usize)>,
    /// Reusable Floyd scratch (small degree <= 256): generation stamps, so
    /// per-node reuse is a counter bump instead of an O(n) clear.
    floyd: FloydStamps,
    /// Reusable Floyd set (large degree > 256).
    seen_set: FxHashSet<usize>,
    /// Temporal: filtered (csr_index, timestamp) pairs for select-k.
    temporal_filtered: Vec<(usize, f64)>,
    /// Weighted no-replace: reusable key buffer to avoid per-call allocation.
    weighted_keys: Vec<(f64, usize)>,
    /// Weighted with-replace: reusable cumulative-distribution buffer.
    cumsum_buf: Vec<f64>,
    /// Hub cap: positions drawn from a capped row, and the neighbors and
    /// weights gathered at them.
    cap_idx: Vec<usize>,
    cap_nbrs: Vec<NodeId>,
    cap_w: Vec<f32>,
    /// Cumulative mode: hop at which each edge id was first emitted.
    emitted_at: FxHashMap<u64, u32>,
    /// Bidirectional: local endpoint pairs already present.
    pair_set: FxHashSet<u64>,
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

/// One `u64` key for an ordered pair of local endpoint indices.
#[inline(always)]
fn pair_key(src: u32, dst: u32) -> u64 {
    (u64::from(src) << 32) | u64::from(dst)
}

impl<'a> NeighborSampler<'a> {
    /// Creates a new neighbor sampler.
    ///
    /// Does not check the config against the graph: a weighted config on an
    /// unweighted graph samples uniformly, and a temporal config on a graph
    /// without timestamps expands no edges. [`Self::try_new`] rejects both.
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
        // node touched, so worst-case residency is the whole table, and one
        // sampler lives per worker thread — the table is multiplied by the
        // thread count.
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
            node_times: Vec::new(),
            track_times: false,
            hop: 0,
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
            temporal_filtered: Vec::with_capacity(max_fanout),
            weighted_keys: Vec::with_capacity(256),
            cumsum_buf: Vec::with_capacity(256),
            cap_idx: Vec::new(),
            cap_nbrs: Vec::new(),
            cap_w: Vec::new(),
            emitted_at: FxHashMap::default(),
            pair_set: FxHashSet::default(),
        }
    }

    /// Creates a sampler after checking that the graph carries the edge data
    /// the config asks for.
    ///
    /// # Errors
    /// [`SamplerConfigError`] when `weighted` is set on a graph without
    /// weights, or `temporal_strategy` on a graph without timestamps.
    pub fn try_new(graph: &'a Graph, config: SamplingConfig) -> Result<Self, SamplerConfigError> {
        check_config(graph, &config)?;
        Ok(Self::new(graph, config))
    }

    /// Reset the RNG to `seed` without rebuilding scratch buffers.
    ///
    /// Prefetch workers call this with [`batch_seed`] so each `(base_seed,
    /// batch_idx)` pair draws the same samples regardless of which worker
    /// thread claims the work item.
    #[inline]
    pub fn reseed(&mut self, seed: u64) {
        self.rng = WyRand::new(seed);
    }

    /// Sample k-hop neighborhoods for checked seeds, routing to the disjoint
    /// path when `config.disjoint` is set.
    ///
    /// `input_times` gives each seed's time bound under temporal sampling;
    /// `None` leaves every seed unbounded (only edges before an expanded
    /// node's own time are eligible past the seeds).
    ///
    /// # Errors
    /// [`SampleError::SeedsExceedGraph`] when `seeds` were checked against a
    /// larger graph; [`SampleError::Temporal`] when `input_times` is given
    /// without a temporal strategy or with the wrong length, or a temporal
    /// strategy is set on a graph without timestamps.
    pub fn sample(
        &mut self,
        seeds: &Seeds,
        input_times: Option<&[f64]>,
    ) -> Result<SampledSubgraph, SampleError> {
        let num_nodes = self.graph.num_nodes();
        if seeds.num_nodes() > num_nodes {
            return Err(SampleError::SeedsExceedGraph {
                checked_against: seeds.num_nodes(),
                num_nodes,
            });
        }
        if let Some(times) = input_times {
            if times.len() != seeds.len() {
                return Err(TemporalSamplingError::LengthMismatch {
                    seeds: seeds.len(),
                    times: times.len(),
                }
                .into());
            }
            if self.config.temporal_strategy.is_none() {
                return Err(TemporalSamplingError::StrategyMissing.into());
            }
        }
        if self.config.temporal_strategy.is_some() && !self.graph.has_timestamps() {
            return Err(TemporalSamplingError::TimestampsMissing.into());
        }
        Ok(if self.config.disjoint {
            self.sample_neighbors_disjoint(seeds.ids(), input_times)
        } else {
            self.sample_neighbors_inner(seeds.ids(), input_times)
        })
    }

    /// Sample k-hop neighborhoods for a batch of seed nodes (shared dedup,
    /// no per-seed times).
    ///
    /// Every seed must be below `graph.num_nodes()`; an out-of-range seed
    /// panics. [`Self::sample`] takes range-checked [`Seeds`] and also routes
    /// disjoint configs.
    pub fn sample_neighbors(&mut self, seeds: &[NodeId]) -> SampledSubgraph {
        self.sample_neighbors_inner(seeds, None)
    }

    /// Sample k-hop neighborhoods with temporal constraints.
    ///
    /// Each seed has an associated time; only edges with timestamp < seed time
    /// are eligible. A seed listed more than once takes its earliest time.
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
    /// Returns a combined subgraph with a `batch` vector mapping each node to
    /// its seed index. Local edge indices are offset per seed so each seed's
    /// subgraph is isolated; `subgraph_type` applies within each block.
    ///
    /// Single-pass: resets dedup state per seed without rebuilding the sampler.
    ///
    /// # Panics
    /// When `input_times` is given with a length other than `seeds.len()`
    /// ([`Self::sample`] reports that as an error instead).
    #[inline]
    pub fn sample_neighbors_disjoint(
        &mut self,
        seeds: &[NodeId],
        input_times: Option<&[f64]>,
    ) -> SampledSubgraph {
        if let Some(times) = input_times {
            assert_eq!(
                times.len(),
                seeds.len(),
                "disjoint sampling: one input time per seed"
            );
        }
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
        // PyG layout: every block contributes its one seed node up front,
        // then per-hop counts accumulate across blocks.
        let mut num_sampled_nodes = vec![0usize; num_hops + 1];
        num_sampled_nodes[0] = seeds.len();
        let mut num_sampled_edges = vec![0usize; num_hops];

        for (seed_idx, &seed) in seeds.iter().enumerate() {
            self.begin_pass(is_temporal);

            let time = input_times.map_or(f64::INFINITY, |t| t[seed_idx]);
            let (seed_local, _) = self.register(seed, time);
            self.frontier.push((seed, seed_local));

            self.run_hops(num_hops, is_temporal, |hop, new_nodes, new_edges| {
                num_sampled_nodes[hop + 1] += new_nodes;
                num_sampled_edges[hop] += new_edges;
            });
            self.apply_subgraph_type();

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

    /// Reset per-pass state: dedup, node list and times, edge buffers,
    /// frontiers.
    fn begin_pass(&mut self, is_temporal: bool) {
        self.dedup.begin();
        self.node_vec.clear();
        self.node_times.clear();
        self.track_times = is_temporal;
        self.edge_src_buf.clear();
        self.edge_dst_buf.clear();
        self.edge_ids_buf.clear();
        self.src_local_buf.clear();
        self.dst_local_buf.clear();
        self.frontier.clear();
        self.next_frontier.clear();
        self.emitted_at.clear();
    }

    /// Register `id` with time bound `time`, returning `(local index, is_new)`.
    ///
    /// The one place a node enters `node_vec`; when the pass tracks times
    /// the time is pushed in the same step, and a node reached again keeps
    /// the earliest bound any path gave it.
    #[inline(always)]
    fn register(&mut self, id: NodeId, time: f64) -> (u32, bool) {
        let (idx, is_new) = self.dedup.probe_or_insert(id, self.node_vec.len() as u32);
        if is_new {
            self.node_vec.push(id);
            if self.track_times {
                self.node_times.push(time);
            }
        } else if self.track_times {
            let bound = &mut self.node_times[idx as usize];
            if time < *bound {
                *bound = time;
            }
        }
        (idx, is_new)
    }

    /// Register a sampled neighbor, pushing it onto the next frontier when
    /// new. Returns `None` when `id` is outside the graph (corrupt edge body
    /// under `OffsetsOnly` loads) so callers skip the edge instead of
    /// panicking the dense dedup table.
    #[inline(always)]
    fn insert_node_frontier(&mut self, id: NodeId, time: f64) -> Option<u32> {
        if (id as usize) >= self.graph.num_nodes() {
            return None;
        }
        let (idx, is_new) = self.register(id, time);
        if is_new {
            self.next_frontier.push((id, idx));
        }
        Some(idx)
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
            self.hop = hop as u32;
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
                // Re-expand everything seen so far next hop; `emit_edge`
                // drops edges an earlier hop already emitted.
                self.frontier.extend_from_slice(&self.next_frontier);
            } else {
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

        self.begin_pass(is_temporal);
        self.seed_locals_buf.clear();

        // Register seeds with local indices. A repeated seed keeps one node
        // and one frontier entry; its seed_locals entry repeats.
        for (i, &seed) in seeds.iter().enumerate() {
            let time = input_times.map_or(f64::INFINITY, |t| t[i]);
            let (local, is_new) = self.register(seed, time);
            self.seed_locals_buf.push(local);
            if is_new {
                self.frontier.push((seed, local));
            }
        }

        // Per-stage counts in PyG's layout: seed nodes, then each hop's new
        // nodes; one edge count per hop.
        let mut num_sampled_nodes = Vec::with_capacity(num_hops + 1);
        num_sampled_nodes.push(self.node_vec.len());
        let mut num_sampled_edges = Vec::with_capacity(num_hops);

        self.run_hops(num_hops, is_temporal, |_hop, new_nodes, new_edges| {
            num_sampled_nodes.push(new_nodes);
            num_sampled_edges.push(new_edges);
        });
        self.apply_subgraph_type();

        // Hand the filled buffers to the caller and swap fresh ones back in.
        // The returned buffers (nodes, the three edge arrays, and the local
        // endpoint indices) are therefore freshly allocated on every call;
        // only the internal scratch (frontiers, dedup arrays, sample buffers)
        // is reset and reused.
        //
        // Each replacement is sized from what this call just produced — the
        // buffer's own final length, subgraph-type rewrite included, read
        // before it is swapped away. A sampler is reused across batches of
        // near-identical shape, so that count predicts the next call far
        // better than a fixed constant does.
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

    /// Rewrite the current pass's edge buffers for `config.subgraph_type`.
    fn apply_subgraph_type(&mut self) {
        match self.config.subgraph_type {
            SubgraphType::Directional => {}
            SubgraphType::Induced => self.induce(),
            SubgraphType::Bidirectional => self.mirror(),
        }
    }

    /// Replace the sampled edges with every graph edge between sampled nodes.
    ///
    /// Scans each sampled node's full row — O(sum of their degrees) — and
    /// keeps a neighbor when the pass's dedup table holds it. Temporal passes
    /// keep only edges older than the source node's time bound, the same rule
    /// sampling applied.
    fn induce(&mut self) {
        let graph = self.graph;
        let csr = graph.csr_view();
        let edges = csr.edges();
        let timestamps = if self.track_times {
            graph.timestamps()
        } else {
            None
        };

        self.edge_src_buf.clear();
        self.edge_dst_buf.clear();
        self.edge_ids_buf.clear();
        self.src_local_buf.clear();
        self.dst_local_buf.clear();

        for u_local in 0..self.node_vec.len() {
            let u = self.node_vec[u_local];
            let range = csr.neighbor_range(u);
            let start = range.start;
            let bound = if self.track_times {
                self.node_times[u_local]
            } else {
                f64::INFINITY
            };
            for (j, &w) in edges[range].iter().enumerate() {
                let pos = start + j;
                if timestamps.is_some_and(|ts| ts[pos] >= bound) {
                    continue;
                }
                if let Some(w_local) = self.dedup.get(w) {
                    self.push_edge(u, u_local as u32, w, w_local, pos as u64);
                }
            }
        }
    }

    /// Append the reverse of every sampled edge whose reverse is not already
    /// present, so each ordered endpoint pair appears exactly once.
    fn mirror(&mut self) {
        let n = self.edge_src_buf.len();
        self.pair_set.clear();
        for i in 0..n {
            self.pair_set
                .insert(pair_key(self.src_local_buf[i], self.dst_local_buf[i]));
        }
        let track = self.config.track_edge_ids;
        for i in 0..n {
            let (s, d) = (self.src_local_buf[i], self.dst_local_buf[i]);
            if !self.pair_set.insert(pair_key(d, s)) {
                continue;
            }
            let (src, dst) = (self.edge_src_buf[i], self.edge_dst_buf[i]);
            self.edge_src_buf.push(dst);
            self.edge_dst_buf.push(src);
            self.src_local_buf.push(d);
            self.dst_local_buf.push(s);
            if track {
                let eid = self.edge_ids_buf[i];
                self.edge_ids_buf.push(eid);
            }
        }
    }

    /// `Some(max_degree)` when a row of `degree` exceeds it (a hub), after
    /// recording the hub in telemetry.
    #[inline]
    fn hub_cap(&self, degree: usize) -> Option<usize> {
        let cap = self.config.max_degree?;
        if degree <= cap {
            return None;
        }
        if let Some(ref telemetry) = self.config.telemetry {
            telemetry.record_hub_node();
        }
        crate::probe!(hub_node_capped, degree, cap);
        Some(cap)
    }

    /// Uniform integer in `[0, s)` for `s <= 2^32` (multiply-shift).
    #[inline(always)]
    fn rand_below(&mut self, s: usize) -> usize {
        (u64::from(self.rng.next_u32()).wrapping_mul(s as u64) >> 32) as usize
    }

    /// Fill `cap_idx` with `k` distinct positions drawn uniformly from
    /// `[0, n)` (Floyd), ascending so gathers walk the row in order.
    fn draw_positions(&mut self, n: usize, k: usize) {
        self.cap_idx.clear();
        self.seen_set.clear();
        for i in (n - k)..n {
            let j = self.rand_below(i + 1);
            let pick = if self.seen_set.insert(j) {
                j
            } else {
                self.seen_set.insert(i);
                i
            };
            self.cap_idx.push(pick);
        }
        self.cap_idx.sort_unstable();
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
        let Some(strategy) = self.config.temporal_strategy else {
            return;
        };
        let node_time = self.node_times[node_local as usize];
        let cap = self.hub_cap(neighbors.len());

        self.sample_buf.clear();
        self.temporal_filtered.clear();
        match (strategy, cap) {
            (TemporalStrategy::Uniform, Some(cap)) => {
                self.draw_positions(neighbors.len(), cap);
                for &i in &self.cap_idx {
                    if ts[i] < node_time {
                        self.temporal_filtered.push((i, ts[i]));
                    }
                }
            }
            _ => {
                for (i, &t) in ts.iter().enumerate() {
                    if t < node_time {
                        self.temporal_filtered.push((i, t));
                    }
                }
            }
        }
        let valid = self.temporal_filtered.len();
        if valid == 0 {
            return;
        }

        match strategy {
            TemporalStrategy::Uniform => {
                if sample_size >= valid {
                    for &(csr_idx, _) in &self.temporal_filtered {
                        self.sample_buf.push((neighbors[csr_idx], csr_idx));
                    }
                } else {
                    // Floyd's O(k) on the filtered set.
                    self.seen_set.clear();
                    for i in (valid - sample_size)..valid {
                        let j = self.rand_below(i + 1);
                        let pick = if self.seen_set.insert(j) {
                            j
                        } else {
                            self.seen_set.insert(i);
                            i
                        };
                        let (csr_idx, _) = self.temporal_filtered[pick];
                        self.sample_buf.push((neighbors[csr_idx], csr_idx));
                    }
                }
            }
            TemporalStrategy::Last => {
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

        // A neighbor's time bound is the timestamp of the edge that reached it.
        for j in 0..self.sample_buf.len() {
            let (neighbor, csr_idx) = self.sample_buf[j];
            self.emit_edge(
                node,
                node_local,
                neighbor,
                edge_offset + csr_idx as u64,
                ts[csr_idx],
            );
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
        let cap = self.hub_cap(neighbors.len());
        if let Some(w) = weights {
            self.sample_buf.clear();
            match cap {
                Some(cap) => {
                    // Weighted sampling is O(degree); on a hub it runs over
                    // `cap` uniformly drawn positions instead.
                    self.draw_positions(neighbors.len(), cap);
                    let mut nbrs = std::mem::take(&mut self.cap_nbrs);
                    let mut ws = std::mem::take(&mut self.cap_w);
                    nbrs.clear();
                    ws.clear();
                    for &i in &self.cap_idx {
                        nbrs.push(neighbors[i]);
                        ws.push(w[i]);
                    }
                    self.weighted_into(&nbrs, &ws, sample_size);
                    // Results index the drawn subset; map back to row positions.
                    for pick in &mut self.sample_buf {
                        pick.1 = self.cap_idx[pick.1];
                    }
                    self.cap_nbrs = nbrs;
                    self.cap_w = ws;
                }
                None => self.weighted_into(neighbors, w, sample_size),
            }
            for i in 0..self.sample_buf.len() {
                let (neighbor, idx) = self.sample_buf[i];
                self.emit_edge(
                    node,
                    node_local,
                    neighbor,
                    edge_offset + idx as u64,
                    f64::INFINITY,
                );
            }
        } else {
            // Floyd and replacement draws are O(fanout) at any degree, so a
            // hub's whole row stays eligible.
            let n = neighbors.len();
            if self.config.replace {
                self.emit_sample_replace(node, node_local, neighbors, edge_offset, sample_size);
            } else if sample_size >= n {
                self.emit_take_all(node, node_local, neighbors, edge_offset);
            } else {
                self.emit_sample_floyd(node, node_local, neighbors, edge_offset, sample_size);
            }
        }
    }

    #[inline]
    fn weighted_into(&mut self, neighbors: &[NodeId], weights: &[f32], k: usize) {
        if self.config.replace {
            self.weighted_sample_with_replacement_into(neighbors, weights, k);
        } else {
            self.weighted_sample_without_replacement_into(neighbors, weights, k);
        }
    }

    /// Whether edge `eid` may be emitted this hop. Cumulative mode re-expands
    /// earlier nodes, so an edge an earlier hop emitted is refused; repeats
    /// within one hop (sampling with replacement) pass.
    #[inline]
    fn fresh_edge(&mut self, eid: u64) -> bool {
        match self.emitted_at.entry(eid) {
            Entry::Occupied(e) => *e.get() == self.hop,
            Entry::Vacant(e) => {
                e.insert(self.hop);
                true
            }
        }
    }

    /// Push one edge with its local endpoint indices to the output buffers.
    #[inline(always)]
    fn push_edge(&mut self, src: NodeId, src_local: u32, dst: NodeId, dst_local: u32, eid: u64) {
        self.edge_src_buf.push(src);
        self.edge_dst_buf.push(dst);
        self.src_local_buf.push(src_local);
        self.dst_local_buf.push(dst_local);
        if self.config.track_edge_ids {
            self.edge_ids_buf.push(eid);
        }
    }

    /// Register a sampled neighbor and push its edge. Destination IDs outside
    /// `[0, num_nodes)` are skipped — `OffsetsOnly` loads leave edge bodies
    /// unchecked, and a corrupt destination must not panic the dense dedup
    /// table or invent phantom nodes in map mode.
    #[inline(always)]
    fn emit_edge(&mut self, src: NodeId, src_local: u32, neighbor: NodeId, eid: u64, time: f64) {
        if self.config.cumulative && !self.fresh_edge(eid) {
            return;
        }
        let Some(dst_local) = self.insert_node_frontier(neighbor, time) else {
            return;
        };
        self.push_edge(src, src_local, neighbor, dst_local, eid);
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
        let n = neighbors.len();
        for _ in 0..k {
            let idx = self.rand_below(n);
            self.emit_edge(
                src,
                src_local,
                neighbors[idx],
                edge_offset + idx as u64,
                f64::INFINITY,
            );
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
        for (idx, &neighbor) in neighbors.iter().enumerate() {
            self.emit_edge(
                src,
                src_local,
                neighbor,
                edge_offset + idx as u64,
                f64::INFINITY,
            );
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
                self.emit_edge(
                    src,
                    src_local,
                    neighbors[pick],
                    edge_offset + pick as u64,
                    f64::INFINITY,
                );
            }
        } else {
            self.seen_set.clear();
            for i in (n - k)..n {
                let j = self.rand_below(i + 1);
                let pick = if self.seen_set.insert(j) {
                    j
                } else {
                    self.seen_set.insert(i);
                    i
                };
                self.emit_edge(
                    src,
                    src_local,
                    neighbors[pick],
                    edge_offset + pick as u64,
                    f64::INFINITY,
                );
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
        if n == 0 {
            return;
        }

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

/// Local seed indices into [`SampledSubgraph::nodes`], borrowed when the
/// sampler recorded them at emit time.
pub type SeedIndicesLocal<'a> = Cow<'a, [u32]>;

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
/// Nodes are stored in discovery order (not sorted): seeds first, then each
/// hop's new nodes. Sampler-produced subgraphs carry local endpoint and seed
/// indices recorded at emit time; subgraphs rebuilt via
/// [`SampledSubgraph::from_parts`] derive them on demand.
#[derive(Debug, Clone)]
pub struct SampledSubgraph {
    /// All nodes in the subgraph (discovery order)
    pub nodes: Vec<NodeId>,

    /// Edge sources in stored direction: the node whose row was expanded.
    pub edge_src: Vec<NodeId>,

    /// Edge destinations in stored direction: the neighbor drawn from the
    /// source's row.
    pub edge_dst: Vec<NodeId>,

    /// Global edge IDs (position in CSR edges array)
    pub edge_ids: Vec<u64>,

    /// Original seed nodes
    pub seeds: Vec<NodeId>,

    /// Nodes each stage added, in PyG's layout: `[seed nodes, hop 1, ...,
    /// hop k]` (length `fanout.len() + 1`).
    pub num_sampled_nodes: Vec<usize>,

    /// Edges each hop sampled (length `fanout.len()`). Induced and
    /// bidirectional subgraphs rewrite the edge arrays afterwards; these
    /// counts still describe the sampling pass.
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

    /// Edge indices with local (remapped) node IDs in `[0, num_nodes)`, in
    /// stored direction (parallel to `edge_src`/`edge_dst`).
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
    /// Sampler-produced subgraphs (disjoint included) return a borrowed slice of
    /// the indices recorded at seed registration; subgraphs built via
    /// [`SampledSubgraph::from_parts`] derive an owned vector from a transient
    /// per-call map. Repeated calls return the same answer on every provenance.
    ///
    /// # Errors
    /// Only [`SampledSubgraph::from_parts`] subgraphs can fail, when a seed
    /// is missing from `nodes` — inconsistent reconstruction input.
    pub fn seed_indices_local(&self) -> Result<SeedIndicesLocal<'_>, String> {
        match &self.locals {
            Locals::Recorded { seeds, .. } => Ok(Cow::Borrowed(seeds)),
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
                    .collect::<Result<Vec<_>, _>>()
                    .map(Cow::Owned)
            }
        }
    }
}

/// Parallel batch sampler for high-throughput GNN training.
///
/// Batches are sampled in parallel with Rayon. Before each batch the sampler
/// reseeds from the configured seed and the batch's position in the stream of
/// every batch this sampler has handled (see [`batch_seed`]), so a fixed
/// `seed` gives bit-identical output at any thread count, and successive
/// calls continue the stream instead of replaying it. Per-thread samplers —
/// and their node-sized dedup tables — persist across calls.
pub struct ParallelBatchSampler<'a> {
    graph: &'a Graph,
    config: SamplingConfig,
    base_seed: u64,
    /// Stream position of the next batch.
    next_batch: AtomicU64,
    /// Idle samplers, checked out one per rayon chunk.
    pool: Mutex<Vec<NeighborSampler<'a>>>,
}

impl<'a> ParallelBatchSampler<'a> {
    /// Creates a new parallel batch sampler (see [`NeighborSampler::new`]
    /// for what is not checked).
    pub fn new(graph: &'a Graph, config: SamplingConfig) -> Self {
        let base_seed = config.seed.unwrap_or_else(rand::random::<u64>);
        Self {
            graph,
            config,
            base_seed,
            next_batch: AtomicU64::new(0),
            pool: Mutex::new(Vec::new()),
        }
    }

    /// Creates a sampler after checking the config against the graph.
    ///
    /// # Errors
    /// See [`NeighborSampler::try_new`].
    pub fn try_new(graph: &'a Graph, config: SamplingConfig) -> Result<Self, SamplerConfigError> {
        check_config(graph, &config)?;
        Ok(Self::new(graph, config))
    }

    /// Sample neighborhoods for multiple batches, in input order.
    ///
    /// # Errors
    /// The first [`SampleError`] any batch raised (see
    /// [`NeighborSampler::sample`]).
    pub fn sample_batches(&self, batches: &[Seeds]) -> Result<Vec<SampledSubgraph>, SampleError> {
        let first = self
            .next_batch
            .fetch_add(batches.len() as u64, Ordering::Relaxed);
        // One sampler per chunk, processed serially within the chunk, so
        // sampler checkouts are capped at the thread count.
        let threads = rayon::current_num_threads().max(1);
        let chunk = batches.len().div_ceil(threads).max(1);
        let grouped: Vec<Result<Vec<SampledSubgraph>, SampleError>> = batches
            .par_chunks(chunk)
            .enumerate()
            .map(|(c, group)| {
                let mut sampler = self.checkout();
                let out = group
                    .iter()
                    .enumerate()
                    .map(|(j, seeds)| {
                        let position = first + (c * chunk + j) as u64;
                        sampler.reseed(batch_seed(self.base_seed, position as usize));
                        sampler.sample(seeds, None)
                    })
                    .collect();
                self.checkin(sampler);
                out
            })
            .collect();

        let mut out = Vec::with_capacity(batches.len());
        for group in grouped {
            out.extend(group?);
        }
        Ok(out)
    }

    fn checkout(&self) -> NeighborSampler<'a> {
        let idle = self
            .pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        idle.unwrap_or_else(|| NeighborSampler::new(self.graph, self.config.clone()))
    }

    fn checkin(&self, sampler: NeighborSampler<'a>) {
        self.pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(sampler);
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

    /// Induced returns every graph edge between sampled nodes — a superset
    /// of the sampled edges — over the same node set the directional pass
    /// discovers.
    #[test]
    fn induced_returns_every_edge_between_sampled_nodes() {
        let mut edges = Vec::new();
        for src in 0..200u32 {
            for step in 1..=7u32 {
                edges.push((src, (src * 7 + step * 13) % 200));
            }
        }
        let graph = Graph::from_edges(200, &edges, None).unwrap();

        let base = SamplingConfig {
            fanout: vec![4, 3, 2],
            seed: Some(7),
            subgraph_type: SubgraphType::Induced,
            ..Default::default()
        };
        let seeds: Vec<NodeId> = (0..16).collect();

        let mut induced = NeighborSampler::new(&graph, base.clone());
        let induced = induced.sample_neighbors(&seeds);

        let directional_cfg = SamplingConfig {
            subgraph_type: SubgraphType::Directional,
            ..base
        };
        let mut directional = NeighborSampler::new(&graph, directional_cfg);
        let directional = directional.sample_neighbors(&seeds);

        // Same seed, same RNG stream: identical node discovery.
        assert_eq!(induced.nodes, directional.nodes);

        let member: FxHashSet<NodeId> = induced.nodes.iter().copied().collect();
        let expected: FxHashSet<u64> = (0..graph.num_edges() as u64)
            .filter(|&e| {
                let (s, d) = edges_at(&graph, e);
                member.contains(&s) && member.contains(&d)
            })
            .collect();
        let got: FxHashSet<u64> = induced.edge_ids.iter().copied().collect();
        assert_eq!(got.len(), induced.edge_ids.len(), "induced edge repeated");
        assert_eq!(got, expected);
        for id in &directional.edge_ids {
            assert!(
                got.contains(id),
                "sampled edge {id} missing from induced set"
            );
        }
        assert!(got.len() > directional.edge_ids.len());

        let (src_local, dst_local) = induced.edge_index_local().unwrap();
        for i in 0..induced.num_edges() {
            assert_eq!(induced.nodes[src_local[i] as usize], induced.edge_src[i]);
            assert_eq!(induced.nodes[dst_local[i] as usize], induced.edge_dst[i]);
            let (s, d) = edges_at(&graph, induced.edge_ids[i]);
            assert_eq!((s, d), (induced.edge_src[i], induced.edge_dst[i]));
        }
    }

    /// `(src, dst)` of the edge at CSR position `eid`.
    fn edges_at(graph: &Graph, eid: u64) -> (NodeId, NodeId) {
        let src = (0..graph.num_nodes() as NodeId)
            .find(|&n| {
                let r = graph.csr_view().neighbor_range(n);
                (r.start as u64..r.end as u64).contains(&eid)
            })
            .expect("edge id inside the CSR");
        let start = graph.csr_view().neighbor_range(src).start as u64;
        (src, graph.neighbors(src)[(eid - start) as usize])
    }

    /// Bidirectional keeps the forward run untouched and appends the reverse
    /// of each forward edge whose reverse is not already there, so every
    /// ordered endpoint pair appears exactly once and a reverse edge shares
    /// its forward edge's id. Local indices mirror, keeping
    /// `edge_index_local` consistent with `edge_src`/`edge_dst`.
    #[test]
    fn bidirectional_mirrors_and_coalesces() {
        let mut edges = Vec::new();
        for src in 0..120u32 {
            for step in 1..=5u32 {
                edges.push((src, (src * 11 + step * 7) % 120));
            }
        }
        // Mutual pairs, so some forward edges already have their reverse.
        for src in 0..60u32 {
            edges.push(((src * 11 + 7) % 120, src));
        }
        edges.sort_unstable();
        edges.dedup();
        let graph = Graph::from_edges(120, &edges, None).unwrap();

        let base = SamplingConfig {
            fanout: vec![6, 6],
            seed: Some(11),
            ..Default::default()
        };
        let seeds: Vec<NodeId> = (0..40).collect();

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
        assert_eq!(&bidi.edge_src[..n], &forward.edge_src[..]);
        assert_eq!(&bidi.edge_dst[..n], &forward.edge_dst[..]);
        assert_eq!(&bidi.edge_ids[..n], &forward.edge_ids[..]);

        let fwd_pairs: FxHashSet<(NodeId, NodeId)> = forward
            .edge_src
            .iter()
            .copied()
            .zip(forward.edge_dst.iter().copied())
            .collect();
        let expected_reverse: Vec<(NodeId, NodeId, u64)> = (0..n)
            .filter(|&i| !fwd_pairs.contains(&(forward.edge_dst[i], forward.edge_src[i])))
            .map(|i| {
                (
                    forward.edge_dst[i],
                    forward.edge_src[i],
                    forward.edge_ids[i],
                )
            })
            .collect();
        assert!(
            expected_reverse.len() < n,
            "fixture has no mutual pair to coalesce"
        );
        let got_reverse: Vec<(NodeId, NodeId, u64)> = (n..bidi.num_edges())
            .map(|i| (bidi.edge_src[i], bidi.edge_dst[i], bidi.edge_ids[i]))
            .collect();
        assert_eq!(got_reverse, expected_reverse);

        let all_pairs: FxHashSet<(NodeId, NodeId)> = bidi
            .edge_src
            .iter()
            .copied()
            .zip(bidi.edge_dst.iter().copied())
            .collect();
        assert_eq!(all_pairs.len(), bidi.num_edges(), "ordered pair repeated");
        for &(s, d) in &all_pairs {
            assert!(all_pairs.contains(&(d, s)), "({s},{d}) has no reverse");
        }

        let (src_local, dst_local) = bidi.edge_index_local().unwrap();
        for i in 0..bidi.num_edges() {
            assert_eq!(bidi.nodes[src_local[i] as usize], bidi.edge_src[i]);
            assert_eq!(bidi.nodes[dst_local[i] as usize], bidi.edge_dst[i]);
        }
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
        let batches = seeds_for(&graph, &[vec![0], vec![1], vec![2]]);

        let subgraphs = sampler.sample_batches(&batches).unwrap();

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
    fn corrupt_destination_is_skipped_not_panic() {
        // Simulate an OffsetsOnly-loaded CSR whose edge body has an
        // out-of-range destination. Sampling must skip it, not panic the
        // dense dedup table.
        let graph = Graph::from_csr_vecs(
            3,
            3,
            vec![0u64, 2, 2, 2],
            vec![1u32, 99u32],
            None,
            crate::GraphValidationMode::OffsetsOnly,
        )
        .unwrap();
        let config = SamplingConfig {
            fanout: vec![10],
            replace: false,
            seed: Some(1),
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
        let sub = sampler.sample_neighbors(&[0]);
        assert!(sub.nodes.iter().all(|&n| (n as usize) < 3));
        assert!(sub.edge_dst.iter().all(|&d| (d as usize) < 3));
        // Only the in-range neighbor (1) should have been emitted.
        assert_eq!(sub.num_edges(), 1);
    }

    #[test]
    fn batch_seed_is_stable_and_distinct() {
        assert_eq!(batch_seed(42, 0), batch_seed(42, 0));
        assert_ne!(batch_seed(42, 0), batch_seed(42, 1));
        assert_ne!(batch_seed(1, 0), batch_seed(2, 0));
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

        // Uniform draws are O(fanout), so the whole row stays eligible: with
        // 25 draws the chance that none lands past position 10k is 2^-25.
        assert!(
            subgraph.edge_ids.iter().any(|&e| e >= 10_000),
            "hub samples confined to the first max_degree positions: {:?}",
            subgraph.edge_ids
        );

        // Verify telemetry recorded the hub node
        let summary = telemetry.summary();
        assert_eq!(summary.hub_nodes_capped, 1);
        assert_eq!(summary.total_samples, 1);
    }

    /// Uniform sampling from a hub is uniform over its entire row.
    #[test]
    fn hub_uniform_sampling_covers_whole_row() {
        let edges: Vec<(NodeId, NodeId)> = (1..=2_000).map(|i| (0, i)).collect();
        let graph = Graph::from_edges(2_001, &edges, None).unwrap();
        let config = SamplingConfig {
            fanout: vec![10],
            seed: Some(3),
            max_degree: Some(100),
            ..Default::default()
        };
        let mut sampler = NeighborSampler::new(&graph, config);
        let mut halves = [0usize; 2];
        for _ in 0..400 {
            for &e in &sampler.sample_neighbors(&[0]).edge_ids {
                halves[usize::from(e >= 1_000)] += 1;
            }
        }
        let total = halves[0] + halves[1];
        assert_eq!(total, 4_000);
        // Each half expects 2000; allow ±10%.
        for h in halves {
            assert!((1_800..=2_200).contains(&h), "row halves drawn {halves:?}");
        }
    }

    /// Weighted sampling on a hub runs over `max_degree` random positions,
    /// not a prefix of the row.
    #[test]
    fn hub_weighted_cap_draws_random_positions() {
        let n = 5_000u32;
        let edges: Vec<(NodeId, NodeId)> = (1..=n).map(|i| (0, i)).collect();
        let weights = vec![1.0f32; n as usize];
        let graph = Graph::from_edges(n as usize + 1, &edges, Some(&weights)).unwrap();
        for replace in [false, true] {
            let config = SamplingConfig {
                fanout: vec![20],
                replace,
                seed: Some(9),
                weighted: true,
                max_degree: Some(50),
                ..Default::default()
            };
            let mut sampler = NeighborSampler::try_new(&graph, config).unwrap();
            let mut max_eid = 0;
            for _ in 0..20 {
                let sub = sampler.sample_neighbors(&[0]);
                assert_eq!(sub.num_edges(), 20);
                for (i, &e) in sub.edge_ids.iter().enumerate() {
                    // The id maps back to the neighbor it names.
                    assert_eq!(graph.neighbors(0)[e as usize], sub.edge_dst[i]);
                    max_eid = max_eid.max(e);
                }
            }
            assert!(max_eid >= 50, "weighted cap stayed in the row prefix");
        }
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

    fn seeds_for(graph: &Graph, batches: &[Vec<NodeId>]) -> Vec<Seeds> {
        batches
            .iter()
            .map(|b| Seeds::new(b.clone(), graph.num_nodes()).unwrap())
            .collect()
    }

    /// A ring where every node has `degree` out-neighbors, large enough that
    /// fanouts below the degree actually draw.
    fn ring_graph(num_nodes: u32, degree: u32) -> Graph {
        let edges: Vec<(NodeId, NodeId)> = (0..num_nodes)
            .flat_map(|s| (1..=degree).map(move |k| (s, (s + k * 7) % num_nodes)))
            .collect();
        Graph::from_edges(num_nodes as usize, &edges, None).unwrap()
    }

    fn fingerprint(subs: &[SampledSubgraph]) -> Vec<(Vec<NodeId>, Vec<u64>)> {
        subs.iter()
            .map(|s| (s.nodes.clone(), s.edge_ids.clone()))
            .collect()
    }

    /// Output depends on the seed and batch positions only: two samplers
    /// with the same seed agree at any thread count, and a sampler's second
    /// call continues the stream instead of replaying the first.
    #[test]
    fn parallel_batch_sampler_is_seed_deterministic_across_thread_counts() {
        let graph = ring_graph(500, 12);
        let config = SamplingConfig {
            fanout: vec![4, 3],
            seed: Some(0xCAFE_F00D),
            ..Default::default()
        };
        let batches = seeds_for(
            &graph,
            &(0..24)
                .map(|b| (0..8).map(|i| (b * 17 + i * 5) % 500).collect())
                .collect::<Vec<Vec<NodeId>>>(),
        );

        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let s = ParallelBatchSampler::new(&graph, config.clone());
                let first = s.sample_batches(&batches).unwrap();
                let second = s.sample_batches(&batches).unwrap();
                (fingerprint(&first), fingerprint(&second))
            })
        };
        let (one_a, one_b) = run(1);
        let (four_a, four_b) = run(4);
        assert_eq!(one_a, four_a, "output depends on the thread count");
        assert_eq!(one_b, four_b, "output depends on the thread count");
        assert_ne!(one_a, one_b, "second call replayed the first");
    }

    /// Two equal batches in different chunks draw independently.
    #[test]
    fn parallel_batch_sampler_reseeds_each_batch() {
        let graph = ring_graph(500, 30);
        let config = SamplingConfig {
            fanout: vec![5],
            seed: Some(1),
            ..Default::default()
        };
        let batches = seeds_for(&graph, &vec![vec![3, 9, 27]; 16]);
        let s = ParallelBatchSampler::new(&graph, config);
        let subs = s.sample_batches(&batches).unwrap();
        let distinct: FxHashSet<Vec<u64>> = subs.iter().map(|s| s.edge_ids.clone()).collect();
        assert!(distinct.len() > 1, "every batch drew the same sample");
    }

    #[test]
    fn parallel_batch_sampler_routes_disjoint() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![3],
            seed: Some(5),
            disjoint: true,
            ..Default::default()
        };
        let s = ParallelBatchSampler::new(&graph, config);
        let subs = s
            .sample_batches(&seeds_for(&graph, &[vec![0, 1], vec![2]]))
            .unwrap();
        assert_eq!(
            subs[0].batch.as_ref().map(Vec::len),
            Some(subs[0].num_nodes())
        );
        assert!(subs.iter().all(|s| s.batch.is_some()));
    }

    #[test]
    fn seeds_reject_ids_at_or_past_num_nodes() {
        assert_eq!(
            Seeds::new(vec![0, 4, 5, 9], 5),
            Err(SeedOutOfRange {
                position: 2,
                seed: 5,
                num_nodes: 5
            })
        );
        let ok = Seeds::new(vec![4, 0, 4], 5).unwrap();
        assert_eq!(ok.ids(), &[4, 0, 4]);
        assert_eq!(ok.num_nodes(), 5);
    }

    #[test]
    fn sample_rejects_seeds_checked_against_a_larger_graph() {
        let graph = create_test_graph();
        let mut sampler = NeighborSampler::new(&graph, SamplingConfig::default());
        let seeds = Seeds::new(vec![0], 100).unwrap();
        assert_eq!(
            sampler.sample(&seeds, None).unwrap_err(),
            SampleError::SeedsExceedGraph {
                checked_against: 100,
                num_nodes: 5
            }
        );
    }

    #[test]
    fn sample_validates_temporal_input() {
        let graph = create_temporal_test_graph();
        let seeds = Seeds::new(vec![0, 1], graph.num_nodes()).unwrap();

        let mut plain = NeighborSampler::new(&graph, SamplingConfig::default());
        assert_eq!(
            plain.sample(&seeds, Some(&[1.0, 2.0])).unwrap_err(),
            SampleError::Temporal(TemporalSamplingError::StrategyMissing)
        );

        let temporal = SamplingConfig {
            temporal_strategy: Some(TemporalStrategy::Uniform),
            ..Default::default()
        };
        let mut sampler = NeighborSampler::new(&graph, temporal.clone());
        assert_eq!(
            sampler.sample(&seeds, Some(&[1.0])).unwrap_err(),
            SampleError::Temporal(TemporalSamplingError::LengthMismatch { seeds: 2, times: 1 })
        );

        let bare = create_test_graph();
        let mut untimed = NeighborSampler::new(&bare, temporal.clone());
        assert_eq!(
            untimed
                .sample(&Seeds::new(vec![0], 5).unwrap(), None)
                .unwrap_err(),
            SampleError::Temporal(TemporalSamplingError::TimestampsMissing)
        );
        assert_eq!(
            NeighborSampler::try_new(&bare, temporal).err(),
            Some(SamplerConfigError::TimestampsMissing)
        );
        let weighted = SamplingConfig {
            weighted: true,
            ..Default::default()
        };
        assert_eq!(
            NeighborSampler::try_new(&bare, weighted).err(),
            Some(SamplerConfigError::WeightsMissing)
        );
    }

    /// With no per-seed times every seed is unbounded, however many seeds
    /// the batch holds and whatever times earlier seeds' discoveries carry.
    #[test]
    fn temporal_without_times_leaves_every_seed_unbounded() {
        // Seed 0's edges are old, seed 1's are recent.
        let edges = vec![(0, 2), (0, 3), (1, 4), (1, 5)];
        let mut graph = Graph::from_edges(6, &edges, None).unwrap();
        graph.set_timestamps(vec![1.0, 2.0, 100.0, 200.0]).unwrap();
        let config = SamplingConfig {
            fanout: vec![10],
            seed: Some(1),
            temporal_strategy: Some(TemporalStrategy::Uniform),
            ..Default::default()
        };
        let mut sampler = NeighborSampler::try_new(&graph, config).unwrap();
        let seeds = Seeds::new(vec![0, 1], 6).unwrap();
        let sub = sampler.sample(&seeds, None).unwrap();
        let dsts: FxHashSet<NodeId> = sub.edge_dst.iter().copied().collect();
        assert_eq!(dsts, FxHashSet::from_iter([2, 3, 4, 5]));

        // The prefetch path's entry point behaves the same.
        let sub = sampler.sample_neighbors(&[0, 1]);
        assert_eq!(sub.num_edges(), 4);
    }

    /// Many seeds, few discoveries: every node's time bound is its own, so
    /// nothing indexes past the recorded times.
    #[test]
    fn temporal_many_seeds_few_discoveries() {
        let edges: Vec<(NodeId, NodeId)> = (0..64).map(|s| (s, 64 + s % 4)).collect();
        let mut graph = Graph::from_edges(68, &edges, None).unwrap();
        graph
            .set_timestamps((0..64).map(f64::from).collect())
            .unwrap();
        let config = SamplingConfig {
            fanout: vec![2, 2],
            seed: Some(2),
            temporal_strategy: Some(TemporalStrategy::Last),
            ..Default::default()
        };
        let mut sampler = NeighborSampler::try_new(&graph, config).unwrap();
        let seeds: Vec<NodeId> = (0..64).collect();
        let times: Vec<f64> = (0..64).map(|t| f64::from(t) + 0.5).collect();
        let sub = sampler
            .sample(&Seeds::new(seeds, 68).unwrap(), Some(&times))
            .unwrap();
        // Every seed's single edge predates its own time.
        assert_eq!(sub.num_edges(), 64);
    }

    /// Per-seed times bound each seed independently, and a node reached from
    /// several seeds expands under the earliest bound.
    #[test]
    fn temporal_per_seed_times_and_duplicate_seeds() {
        // 0 -> 1 (t=3), 0 -> 2 (t=10); 3 -> 4 (t=1), 3 -> 5 (t=6)
        let edges = vec![(0, 1), (0, 2), (3, 4), (3, 5)];
        let mut graph = Graph::from_edges(6, &edges, None).unwrap();
        graph.set_timestamps(vec![3.0, 10.0, 1.0, 6.0]).unwrap();
        let config = SamplingConfig {
            fanout: vec![10],
            seed: Some(4),
            temporal_strategy: Some(TemporalStrategy::Uniform),
            ..Default::default()
        };
        let mut sampler = NeighborSampler::try_new(&graph, config).unwrap();

        let sub = sampler
            .sample(&Seeds::new(vec![3, 0], 6).unwrap(), Some(&[5.0, 20.0]))
            .unwrap();
        let dsts: FxHashSet<NodeId> = sub.edge_dst.iter().copied().collect();
        assert_eq!(dsts, FxHashSet::from_iter([4, 1, 2]));

        // Seed 0 twice: the earlier bound (5.0) wins, and the node expands
        // once, so no edge repeats.
        let sub = sampler
            .sample(&Seeds::new(vec![0, 0], 6).unwrap(), Some(&[50.0, 5.0]))
            .unwrap();
        assert_eq!(sub.edge_dst, vec![1]);
        let seed_locals = sub.seed_indices_local().unwrap();
        assert_eq!(&seed_locals[..], &[0, 0]);
        assert_eq!(sub.num_sampled_nodes[0], 1);
    }

    /// Duplicate seeds expand once, so no edge is emitted twice.
    #[test]
    fn duplicate_seeds_expand_once() {
        let graph = create_test_graph();
        let mut sampler = NeighborSampler::new(
            &graph,
            SamplingConfig {
                fanout: vec![10],
                seed: Some(1),
                ..Default::default()
            },
        );
        let sub = sampler.sample_neighbors(&[0, 0, 1]);
        let ids: FxHashSet<u64> = sub.edge_ids.iter().copied().collect();
        assert_eq!(ids.len(), sub.num_edges());
        assert_eq!(sub.num_edges(), 5);
    }

    /// Cumulative mode re-expands earlier nodes but never emits an edge
    /// twice; frontier-only mode matches PyG.
    #[test]
    fn cumulative_mode_emits_each_edge_once() {
        let graph = ring_graph(300, 8);
        for replace in [false, true] {
            let config = SamplingConfig {
                fanout: vec![25, 10, 5],
                replace,
                seed: Some(6),
                cumulative: true,
                ..Default::default()
            };
            let mut sampler = NeighborSampler::new(&graph, config);
            let sub = sampler.sample_neighbors(&[0, 1, 2]);
            if !replace {
                let ids: FxHashSet<u64> = sub.edge_ids.iter().copied().collect();
                assert_eq!(
                    ids.len(),
                    sub.num_edges(),
                    "cumulative mode repeated an edge"
                );
            }
            // With replacement, repeats only come from one hop's draws: the
            // edges each hop added never reappear in a later hop.
            let mut seen: FxHashSet<u64> = FxHashSet::default();
            let mut at = 0;
            for &count in &sub.num_sampled_edges {
                let hop: FxHashSet<u64> = sub.edge_ids[at..at + count].iter().copied().collect();
                assert!(seen.is_disjoint(&hop), "a later hop re-emitted an edge");
                seen.extend(hop);
                at += count;
            }
        }
    }

    /// `num_sampled_nodes` follows PyG's layout: seed nodes first, one entry
    /// per hop after, summing to the node count.
    #[test]
    fn num_sampled_nodes_leads_with_seed_count() {
        let graph = ring_graph(400, 6);
        let config = SamplingConfig {
            fanout: vec![3, 2],
            seed: Some(8),
            ..Default::default()
        };
        let mut sampler = NeighborSampler::new(&graph, config.clone());
        let sub = sampler.sample_neighbors(&[5, 6, 5, 7]);
        assert_eq!(sub.num_sampled_nodes.len(), 3);
        assert_eq!(sub.num_sampled_nodes[0], 3);
        assert_eq!(sub.num_sampled_nodes.iter().sum::<usize>(), sub.num_nodes());
        assert_eq!(sub.num_sampled_edges.len(), 2);
        assert_eq!(sub.num_sampled_edges.iter().sum::<usize>(), sub.num_edges());

        let disjoint = SamplingConfig {
            disjoint: true,
            ..config
        };
        let mut sampler = NeighborSampler::new(&graph, disjoint);
        let sub = sampler.sample_neighbors_disjoint(&[5, 6], None);
        assert_eq!(sub.num_sampled_nodes[0], 2);
        assert_eq!(sub.num_sampled_nodes.iter().sum::<usize>(), sub.num_nodes());
    }

    /// Frontier-only (the default) expands each node in exactly one hop.
    #[test]
    fn frontier_mode_expands_each_node_once() {
        let graph = ring_graph(400, 6);
        let mut sampler = NeighborSampler::new(
            &graph,
            SamplingConfig {
                fanout: vec![10, 10, 10],
                seed: Some(12),
                ..Default::default()
            },
        );
        let sub = sampler.sample_neighbors(&[0]);
        let ids: FxHashSet<u64> = sub.edge_ids.iter().copied().collect();
        assert_eq!(ids.len(), sub.num_edges());
        let srcs: FxHashSet<NodeId> = sub.edge_src.iter().copied().collect();
        for s in srcs {
            let degree = graph.degree(s);
            let from_s = sub.edge_src.iter().filter(|&&x| x == s).count();
            assert_eq!(from_s, degree, "node {s} expanded more than once");
        }
    }

    /// `sample` routes disjoint configs, applying the subgraph type within
    /// each seed's block.
    #[test]
    fn disjoint_applies_subgraph_type_per_block() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![3],
            seed: Some(3),
            disjoint: true,
            subgraph_type: SubgraphType::Bidirectional,
            ..Default::default()
        };
        let mut sampler = NeighborSampler::new(&graph, config);
        let sub = sampler
            .sample(&Seeds::new(vec![0, 1], 5).unwrap(), None)
            .unwrap();
        let batch = sub.batch.as_ref().unwrap();
        let (src_local, dst_local) = sub.edge_index_local().unwrap();
        // Seed 0 has 3 out-edges and seed 1 has 2; each gets its reverse.
        assert_eq!(sub.num_edges(), 2 * (3 + 2));
        for i in 0..sub.num_edges() {
            assert_eq!(
                batch[src_local[i] as usize], batch[dst_local[i] as usize],
                "edge crosses seed blocks"
            );
        }
    }

    /// Induced edges under temporal sampling respect each source's time bound.
    #[test]
    fn induced_temporal_keeps_only_edges_before_the_source_bound() {
        // 0 -> 1 (t=1), 0 -> 2 (t=2), 1 -> 2 (t=5), 2 -> 1 (t=0.5)
        let edges = vec![(0, 1), (0, 2), (1, 2), (2, 1)];
        let mut graph = Graph::from_edges(3, &edges, None).unwrap();
        graph.set_timestamps(vec![1.0, 2.0, 5.0, 0.5]).unwrap();
        let config = SamplingConfig {
            fanout: vec![10],
            seed: Some(1),
            temporal_strategy: Some(TemporalStrategy::Uniform),
            subgraph_type: SubgraphType::Induced,
            ..Default::default()
        };
        let mut sampler = NeighborSampler::try_new(&graph, config).unwrap();
        let sub = sampler
            .sample(&Seeds::new(vec![0], 3).unwrap(), Some(&[3.0]))
            .unwrap();
        // Node 1's bound is 1.0 (reached at t=1), so 1 -> 2 (t=5) is out;
        // node 2's bound is 2.0, so 2 -> 1 (t=0.5) is in.
        let pairs: FxHashSet<(NodeId, NodeId)> = sub
            .edge_src
            .iter()
            .copied()
            .zip(sub.edge_dst.iter().copied())
            .collect();
        assert_eq!(pairs, FxHashSet::from_iter([(0, 1), (0, 2), (2, 1)]));
    }
}
