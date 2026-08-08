//! Compressed Sparse Row (CSR) graph representation optimized for GNN neighborhood sampling.
//!
//! CSR format stores a graph using two arrays:
//! - `offsets`: offsets[i] = start index in `edges` array for node i's neighbors
//! - `edges`: flat array of all destination nodes, grouped by source
//!
//! This layout provides O(1) access to any node's neighbor list and excellent cache locality.

use anyhow::Result;
use memmap2::Mmap;
use rayon::prelude::*;
use std::ops::Range;
use std::sync::Arc;
use tracing::trace;

/// Node ID type. u32 supports graphs up to 4 billion nodes.
pub type NodeId = u32;

/// Edge offset type. u64 supports graphs with up to 18 quintillion edges.
pub type EdgeOffset = u64;

/// Edge timestamp length did not match `Graph::num_edges`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimestampLengthMismatch {
    pub got: usize,
    pub expected: usize,
}

impl std::fmt::Display for TimestampLengthMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "timestamps length {} != num_edges {}",
            self.got, self.expected
        )
    }
}

impl std::error::Error for TimestampLengthMismatch {}

/// Graph in CSR (Compressed Sparse Row) format.
///
/// Memory layout is optimized for sequential scans and random access patterns
/// common in GNN sampling.
///
/// Uses HPC-optimized custom binary format with bytemuck for zero-copy I/O.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of nodes in the graph
    num_nodes: usize,

    /// Number of edges in the graph
    num_edges: usize,

    /// Backing storage for CSR arrays.
    storage: GraphStorage,

    /// Optional edge timestamps (parallel to edges array, set separately).
    /// Used for temporal sampling: only edges with timestamp < seed time are eligible.
    timestamps: Option<Arc<[f64]>>,
}

#[derive(Debug, Clone)]
enum GraphStorage {
    Owned {
        offsets: Arc<[EdgeOffset]>,
        edges: Arc<[NodeId]>,
        weights: Option<Arc<[f32]>>,
    },
    Mapped {
        mmap: Arc<Mmap>,
        offsets_range: Range<usize>,
        edges_range: Range<usize>,
        weights_range: Option<Range<usize>>,
    },
}

/// Validation modes for graph loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphValidationMode {
    /// Validate only metadata/length checks.
    HeaderOnly,
    /// Validate metadata + offsets invariants.
    OffsetsOnly,
    /// Full validation including destination node range checks.
    Full,
}

impl Graph {
    /// Create an empty placeholder graph.
    ///
    /// Used for NVMe-backed samplers where the graph is file-backed
    /// and we don't need to keep it in memory.
    pub fn empty() -> Self {
        Self {
            num_nodes: 0,
            num_edges: 0,
            storage: GraphStorage::Owned {
                offsets: Arc::from(vec![0u64].into_boxed_slice()),
                edges: Arc::from(Vec::<NodeId>::new().into_boxed_slice()),
                weights: None,
            },
            timestamps: None,
        }
    }

    /// Creates a graph directly from CSR arrays.
    ///
    /// This is the most efficient way to construct a graph when you already
    /// have the CSR representation (e.g., loading from a binary file).
    ///
    /// # Arguments
    /// * `num_nodes` - Number of nodes
    /// * `offsets` - CSR offset array (length = num_nodes + 1)
    /// * `edges` - CSR edges array (destination nodes)
    /// * `weights` - Optional edge weights
    ///
    /// # Safety
    /// Caller must ensure arrays are valid CSR format. These invariants are
    /// checked with `debug_assert!` (active in debug builds, compiled out in
    /// release):
    /// - offsets.len() == num_nodes + 1
    /// - offsets[0] == 0
    /// - offsets is monotonically increasing
    /// - offsets[num_nodes] == edges.len()
    ///
    /// Edge-destination range (all destinations < num_nodes) is not asserted
    /// here; use [`Graph::validate`] for that.
    pub fn from_csr_arrays(
        num_nodes: usize,
        offsets: Vec<EdgeOffset>,
        edges: Vec<NodeId>,
        weights: Option<Vec<f32>>,
    ) -> Self {
        let num_edges = edges.len();
        debug_assert_eq!(
            offsets.len(),
            num_nodes + 1,
            "offsets length must be num_nodes + 1"
        );
        debug_assert!(
            offsets.first().is_none_or(|&first| first == 0),
            "offsets[0] must be 0"
        );
        debug_assert!(
            offsets.windows(2).all(|w| w[0] <= w[1]),
            "offsets must be monotonically increasing"
        );
        debug_assert!(
            offsets
                .last()
                .is_none_or(|&last| last == num_edges as EdgeOffset),
            "offsets[num_nodes] must equal edges.len()"
        );
        Self {
            num_nodes,
            num_edges,
            storage: GraphStorage::Owned {
                offsets: Arc::from(offsets.into_boxed_slice()),
                edges: Arc::from(edges.into_boxed_slice()),
                weights: weights.map(|w| Arc::from(w.into_boxed_slice())),
            },
            timestamps: None,
        }
    }

    /// Creates a graph from pre-built owned CSR arrays.
    ///
    /// Used by reordering and other transformations that construct new CSR data.
    /// Caller must ensure arrays are consistent (offsets length = num_nodes + 1,
    /// edges/weights length = num_edges).
    pub(crate) fn from_owned_parts(
        num_nodes: usize,
        num_edges: usize,
        offsets: Arc<[EdgeOffset]>,
        edges: Arc<[NodeId]>,
        weights: Option<Arc<[f32]>>,
    ) -> Self {
        Self {
            num_nodes,
            num_edges,
            storage: GraphStorage::Owned {
                offsets,
                edges,
                weights,
            },
            timestamps: None,
        }
    }

    /// Creates a graph view directly from mmap-backed CSR bytes.
    ///
    /// Caller must ensure ranges are valid and correctly typed. Alignment
    /// and length divisibility are asserted here, once, so the per-call
    /// slice accessors can reconstruct typed slices without re-validating
    /// on every `neighbors()` in the sampling hot loop.
    pub(crate) fn from_mapped_parts(
        num_nodes: usize,
        num_edges: usize,
        mmap: Arc<Mmap>,
        offsets_range: Range<usize>,
        edges_range: Range<usize>,
        weights_range: Option<Range<usize>>,
    ) -> Self {
        let base = mmap.as_ptr() as usize;
        assert!(
            (base + offsets_range.start).is_multiple_of(std::mem::align_of::<EdgeOffset>())
                && offsets_range
                    .len()
                    .is_multiple_of(std::mem::size_of::<EdgeOffset>()),
            "mmap offsets range misaligned for u64"
        );
        assert!(
            (base + edges_range.start).is_multiple_of(std::mem::align_of::<NodeId>())
                && edges_range
                    .len()
                    .is_multiple_of(std::mem::size_of::<NodeId>()),
            "mmap edges range misaligned for u32"
        );
        if let Some(ref w) = weights_range {
            assert!(
                (base + w.start).is_multiple_of(std::mem::align_of::<f32>())
                    && w.len().is_multiple_of(std::mem::size_of::<f32>()),
                "mmap weights range misaligned for f32"
            );
        }
        Self {
            num_nodes,
            num_edges,
            storage: GraphStorage::Mapped {
                mmap,
                offsets_range,
                edges_range,
                weights_range,
            },
            timestamps: None,
        }
    }

    /// Creates a new CSR graph from an edge list.
    ///
    /// # Arguments
    /// * `num_nodes` - Total number of nodes
    /// * `edges` - List of (source, destination) tuples
    /// * `weights` - Optional edge weights (must match edges length if provided)
    ///
    /// # Performance
    /// This uses counting sort which is O(V + E) time and O(V) extra space.
    /// For large graphs (>100K edges), uses parallel processing with Rayon.
    pub fn from_edges(
        num_nodes: usize,
        edges: &[(NodeId, NodeId)],
        weights: Option<&[f32]>,
    ) -> Result<Self> {
        Self::build_csr(
            num_nodes,
            edges.len(),
            |i| edges[i].0,
            |i| edges[i].1,
            weights,
        )
    }

    /// Creates a new CSR graph from separate source and destination arrays.
    ///
    /// Structure-of-arrays entry point for callers that already hold `src`
    /// and `dst` as parallel arrays (numpy bindings, columnar imports) — it
    /// avoids materializing an interleaved `(src, dst)` tuple copy of the
    /// whole edge list before the build.
    pub fn from_src_dst(
        num_nodes: usize,
        src: &[NodeId],
        dst: &[NodeId],
        weights: Option<&[f32]>,
    ) -> Result<Self> {
        anyhow::ensure!(
            src.len() == dst.len(),
            "src length {} doesn't match dst length {}",
            src.len(),
            dst.len()
        );
        Self::build_csr(num_nodes, src.len(), |i| src[i], |i| dst[i], weights)
    }

    /// Shared counting-sort CSR builder over an indexed edge accessor.
    ///
    /// Degree counting uses one shared array of relaxed atomic counters:
    /// O(V) memory total, where per-chunk local counts would be
    /// O(V x threads) (1.28 GB of temporaries for a 10M-node graph on 16
    /// threads) plus an O(V x threads) reduce.
    fn build_csr(
        num_nodes: usize,
        num_edges: usize,
        src_at: impl Fn(usize) -> NodeId + Sync,
        dst_at: impl Fn(usize) -> NodeId + Sync,
        weights: Option<&[f32]>,
    ) -> Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};

        trace!(
            "Building CSR graph: {} nodes, {} edges",
            num_nodes, num_edges
        );

        if let Some(w) = weights {
            anyhow::ensure!(
                w.len() == num_edges,
                "weights length {} doesn't match edges length {}",
                w.len(),
                num_edges
            );
        }

        // Validate all source nodes up front so the counting pass is
        // branch-free on the bounds.
        let parallel = num_edges > 100_000;
        let all_src_valid = if parallel {
            (0..num_edges)
                .into_par_iter()
                .all(|i| (src_at(i) as usize) < num_nodes)
        } else {
            (0..num_edges).all(|i| (src_at(i) as usize) < num_nodes)
        };
        if !all_src_valid {
            let bad = (0..num_edges)
                .find(|&i| (src_at(i) as usize) >= num_nodes)
                .expect("a source failed validation");
            anyhow::bail!(
                "source node {} exceeds num_nodes {}",
                src_at(bad),
                num_nodes
            );
        }

        // Count outgoing edges per node.
        let degree: Vec<u64> = if parallel {
            let counters: Vec<AtomicU64> = (0..num_nodes).map(|_| AtomicU64::new(0)).collect();
            (0..num_edges).into_par_iter().for_each(|i| {
                counters[src_at(i) as usize].fetch_add(1, Ordering::Relaxed);
            });
            counters.into_iter().map(AtomicU64::into_inner).collect()
        } else {
            let mut degree = vec![0u64; num_nodes];
            for i in 0..num_edges {
                degree[src_at(i) as usize] += 1;
            }
            degree
        };

        // Build offsets using prefix sum
        let mut offsets: Vec<EdgeOffset> = Vec::with_capacity(num_nodes + 1);
        offsets.push(0);
        for &deg in &degree {
            let last = *offsets.last().unwrap();
            let next = last.checked_add(deg).ok_or_else(|| {
                anyhow::anyhow!("CSR offset overflow: {} + {} exceeds u64::MAX", last, deg)
            })?;
            offsets.push(next);
        }

        // Allocate edges array
        let mut csr_edges = vec![0; num_edges];
        let mut csr_weights = weights.map(|_| vec![0.0; num_edges]);

        // Fill edges using offsets as insertion cursors. Serial: per-source
        // arrival order of neighbors is part of the observable layout.
        let mut cursors = offsets[..num_nodes].to_vec();
        for edge_idx in 0..num_edges {
            let src = src_at(edge_idx);
            let dst = dst_at(edge_idx);
            anyhow::ensure!(
                (dst as usize) < num_nodes,
                "destination node {} exceeds num_nodes {}",
                dst,
                num_nodes
            );

            let pos = cursors[src as usize] as usize;
            csr_edges[pos] = dst;

            if let (Some(csr_w), Some(w)) = (&mut csr_weights, weights) {
                csr_w[pos] = w[edge_idx];
            }

            cursors[src as usize] += 1;
        }

        Ok(Self {
            num_nodes,
            num_edges,
            storage: GraphStorage::Owned {
                offsets: Arc::from(offsets.into_boxed_slice()),
                edges: Arc::from(csr_edges.into_boxed_slice()),
                weights: csr_weights.map(|w| Arc::from(w.into_boxed_slice())),
            },
            timestamps: None,
        })
    }

    /// Returns the number of nodes in the graph.
    #[inline(always)]
    pub const fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of edges in the graph.
    #[inline(always)]
    pub const fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Raw offsets array (for serialization).
    #[inline(always)]
    pub(crate) fn offsets_raw(&self) -> &[EdgeOffset] {
        self.offsets_slice()
    }

    /// Raw edges array (for serialization).
    #[inline(always)]
    pub(crate) fn edges_raw(&self) -> &[NodeId] {
        self.edges_slice()
    }

    /// Returns the degree (number of outgoing edges) of a node.
    ///
    /// # Performance
    /// O(1) - single array lookup.
    #[inline(always)]
    pub fn degree(&self, node: NodeId) -> usize {
        let idx = node as usize;
        if idx >= self.num_nodes {
            return 0;
        }
        let offsets = self.offsets_slice();
        // saturating_sub guards against non-monotonic offsets that slipped past
        // HeaderOnly validation — without it the fuzz target trips a panic.
        (offsets[idx + 1].saturating_sub(offsets[idx])) as usize
    }

    /// Returns the degree of every node in one pass over the offsets array.
    ///
    /// Equivalent to calling [`Graph::degree`] for each node in
    /// `0..num_nodes`, but amortizes the per-call overhead — useful for FFI
    /// callers that would otherwise cross the boundary once per node.
    ///
    /// # Performance
    /// O(num_nodes) - a single sequential scan over the (possibly mmap'd)
    /// offsets array.
    pub fn degrees(&self) -> Vec<u32> {
        self.offsets_slice()
            .windows(2)
            .map(|w| w[1].saturating_sub(w[0]) as u32)
            .collect()
    }

    /// Returns the neighbor IDs for a given node as a slice.
    ///
    /// # Performance
    /// O(1) - just a slice lookup, no allocations.
    #[inline(always)]
    pub fn neighbors(&self, node: NodeId) -> &[NodeId] {
        let range = self.neighbor_range(node);
        let edges = self.edges_slice();
        // Safe indexing with bounds check for correctness
        &edges[range]
    }

    /// Returns the neighbor range for a node in the edges array.
    #[inline(always)]
    fn neighbor_range(&self, node: NodeId) -> Range<usize> {
        let idx = node as usize;
        if idx >= self.num_nodes {
            return 0..0;
        }
        let offsets = self.offsets_slice();
        // Safe indexing: bounds are verified by construction (offsets.len() == num_nodes + 1)
        let start = offsets[idx] as usize;
        let end = offsets[idx + 1] as usize;
        if start > end || end > self.num_edges {
            return 0..0;
        }
        start..end
    }

    /// Returns the starting edge offset for a node (for computing global edge IDs).
    #[inline(always)]
    pub fn edge_offset(&self, node: NodeId) -> EdgeOffset {
        let idx = node as usize;
        if idx >= self.num_nodes {
            return 0;
        }
        let offsets = self.offsets_slice();
        // Safe indexing: bounds are verified by construction
        offsets[idx]
    }

    /// Returns edge weights for a node's neighbors, if weights are available.
    #[inline(always)]
    pub fn neighbor_weights(&self, node: NodeId) -> Option<&[f32]> {
        self.weights_slice().map(|w| {
            let range = self.neighbor_range(node);
            // Safe indexing with bounds check
            &w[range]
        })
    }

    /// Sets edge timestamps (parallel to the edges array).
    ///
    /// Returns `Err(TimestampLengthMismatch)` if the slice length does not
    /// equal `num_edges`. This is the single public way to attach timestamps;
    /// there is no panicking variant.
    pub fn set_timestamps(&mut self, timestamps: Vec<f64>) -> Result<(), TimestampLengthMismatch> {
        if timestamps.len() != self.num_edges {
            return Err(TimestampLengthMismatch {
                got: timestamps.len(),
                expected: self.num_edges,
            });
        }
        self.timestamps = Some(Arc::from(timestamps.into_boxed_slice()));
        Ok(())
    }

    /// Returns edge timestamps for a node's neighbors, if timestamps are set.
    #[inline(always)]
    pub fn neighbor_timestamps(&self, node: NodeId) -> Option<&[f64]> {
        self.timestamps.as_ref().map(|ts| {
            let range = self.neighbor_range(node);
            &ts[range]
        })
    }

    /// Returns the full timestamps array, if set.
    #[inline(always)]
    pub fn timestamps(&self) -> Option<&[f64]> {
        self.timestamps.as_deref()
    }

    /// Returns whether timestamps are available.
    #[inline(always)]
    pub fn has_timestamps(&self) -> bool {
        self.timestamps.is_some()
    }

    /// Returns an iterator over (neighbor_id, optional_weight) pairs for a node.
    #[inline]
    pub fn neighbors_with_weights(
        &self,
        node: NodeId,
    ) -> impl Iterator<Item = (NodeId, Option<f32>)> + '_ {
        let neighbors = self.neighbors(node);
        let weights = self.neighbor_weights(node);

        neighbors.iter().enumerate().map(move |(i, &neighbor)| {
            let weight = weights.map(|w| w[i]);
            (neighbor, weight)
        })
    }

    /// Returns raw access to offsets array (for zero-copy serialization).
    #[inline(always)]
    pub fn offsets(&self) -> &[EdgeOffset] {
        self.offsets_slice()
    }

    /// Returns raw access to edges array (for zero-copy serialization).
    #[inline(always)]
    pub fn edges(&self) -> &[NodeId] {
        self.edges_slice()
    }

    /// Returns raw access to weights array (for zero-copy serialization).
    #[inline(always)]
    pub fn weights(&self) -> Option<&[f32]> {
        self.weights_slice()
    }

    /// Validates graph structure invariants.
    ///
    /// # Performance
    /// Uses parallel validation for large graphs (>100K edges) with rayon.
    pub fn validate(&self) -> Result<()> {
        self.validate_with_mode(GraphValidationMode::Full)
    }

    /// Validates graph structure invariants with configurable strictness.
    pub fn validate_with_mode(&self, mode: GraphValidationMode) -> Result<()> {
        trace!("Validating graph structure");
        let offsets = self.offsets_slice();
        let edges = self.edges_slice();
        let weights = self.weights_slice();

        anyhow::ensure!(
            offsets.len() == self.num_nodes + 1,
            "offsets length {} should be num_nodes + 1 = {}",
            offsets.len(),
            self.num_nodes + 1
        );

        anyhow::ensure!(
            offsets[0] == 0,
            "first offset should be 0, got {}",
            offsets[0]
        );

        anyhow::ensure!(
            *offsets.last().unwrap() as usize == self.num_edges,
            "last offset should equal num_edges"
        );

        if mode != GraphValidationMode::HeaderOnly {
            // Check offsets are monotonically increasing (parallel for large graphs)
            if self.num_nodes > 10_000 {
                // Parallel validation for large graphs
                let is_monotonic = offsets.par_windows(2).all(|w| w[1] >= w[0]);
                anyhow::ensure!(is_monotonic, "offsets not monotonically increasing");
            } else {
                // Sequential for small graphs (less overhead)
                for i in 1..offsets.len() {
                    anyhow::ensure!(
                        offsets[i] >= offsets[i - 1],
                        "offsets not monotonic at index {}: {} < {}",
                        i,
                        offsets[i],
                        offsets[i - 1]
                    );
                }
            }
        }

        anyhow::ensure!(
            edges.len() == self.num_edges,
            "edges array length {} doesn't match num_edges {}",
            edges.len(),
            self.num_edges
        );

        if let Some(w) = weights {
            anyhow::ensure!(
                w.len() == self.num_edges,
                "weights length {} doesn't match num_edges {}",
                w.len(),
                self.num_edges
            );
        }

        if mode != GraphValidationMode::Full {
            trace!(?mode, "Graph validation passed");
            return Ok(());
        }

        // Validate all destination nodes are in range (parallel for large graphs)
        if self.num_edges > 100_000 {
            let all_valid = edges.par_iter().all(|&dst| (dst as usize) < self.num_nodes);

            anyhow::ensure!(
                all_valid,
                "some edge destinations are out of range [0, {})",
                self.num_nodes
            );
        } else {
            for &dst in edges {
                anyhow::ensure!(
                    (dst as usize) < self.num_nodes,
                    "edge destination {} out of range [0, {})",
                    dst,
                    self.num_nodes
                );
            }
        }

        trace!("Graph validation passed");
        Ok(())
    }

    /// Returns statistics about the graph structure.
    ///
    /// # Performance
    /// Uses parallel computation for large graphs (>10K nodes).
    pub fn stats(&self) -> GraphStats {
        let (max_degree, degree_sum) = if self.num_nodes > 10_000 {
            // Parallel stats computation for large graphs
            (0..self.num_nodes as NodeId)
                .into_par_iter()
                .map(|node| {
                    let deg = self.degree(node);
                    (deg, deg as u64)
                })
                .reduce(
                    || (0, 0u64),
                    |(max1, sum1), (max2, sum2)| (max1.max(max2), sum1 + sum2),
                )
        } else {
            // Sequential for small graphs
            let mut max_degree = 0;
            let mut degree_sum = 0u64;

            for node in 0..self.num_nodes as NodeId {
                let deg = self.degree(node);
                max_degree = max_degree.max(deg);
                degree_sum += deg as u64;
            }

            (max_degree, degree_sum)
        };

        let avg_degree = if self.num_nodes > 0 {
            degree_sum as f64 / self.num_nodes as f64
        } else {
            0.0
        };

        GraphStats {
            num_nodes: self.num_nodes,
            num_edges: self.num_edges,
            max_degree,
            avg_degree,
            has_weights: self.weights().is_some(),
        }
    }

    #[inline(always)]
    fn offsets_slice(&self) -> &[EdgeOffset] {
        match &self.storage {
            GraphStorage::Owned { offsets, .. } => offsets,
            GraphStorage::Mapped {
                mmap,
                offsets_range,
                ..
            } => {
                let bytes = &mmap[offsets_range.start..offsets_range.end];
                // SAFETY: alignment and length divisibility were asserted
                // once in `from_mapped_parts`; the mmap is immutable and
                // outlives `self` via the Arc. Re-checking per call would
                // put an alignment test and division in the sampler's
                // per-node path.
                unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const EdgeOffset,
                        bytes.len() / std::mem::size_of::<EdgeOffset>(),
                    )
                }
            }
        }
    }

    #[inline(always)]
    fn edges_slice(&self) -> &[NodeId] {
        match &self.storage {
            GraphStorage::Owned { edges, .. } => edges,
            GraphStorage::Mapped {
                mmap, edges_range, ..
            } => {
                let bytes = &mmap[edges_range.start..edges_range.end];
                // SAFETY: see `offsets_slice` — validated at construction.
                unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const NodeId,
                        bytes.len() / std::mem::size_of::<NodeId>(),
                    )
                }
            }
        }
    }

    #[inline(always)]
    fn weights_slice(&self) -> Option<&[f32]> {
        match &self.storage {
            GraphStorage::Owned { weights, .. } => weights.as_deref(),
            GraphStorage::Mapped {
                mmap,
                weights_range,
                ..
            } => weights_range.as_ref().map(|range| {
                let bytes = &mmap[range.start..range.end];
                // SAFETY: see `offsets_slice` — validated at construction.
                unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const f32,
                        bytes.len() / std::mem::size_of::<f32>(),
                    )
                }
            }),
        }
    }
}

/// Graph statistics for debugging and profiling.
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub max_degree: usize,
    pub avg_degree: f64,
    pub has_weights: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph() {
        // Graph: 0 -> 1, 2
        //        1 -> 2
        //        2 -> (no outgoing edges)
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let graph = Graph::from_edges(3, &edges, None).unwrap();

        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.num_edges(), 3);

        assert_eq!(graph.degree(0), 2);
        assert_eq!(graph.degree(1), 1);
        assert_eq!(graph.degree(2), 0);

        let empty: &[NodeId] = &[];
        assert_eq!(graph.neighbors(0), &[1u32, 2u32][..]);
        assert_eq!(graph.neighbors(1), &[2u32][..]);
        assert_eq!(graph.neighbors(2), empty);

        graph.validate().unwrap();
    }

    #[test]
    fn test_degrees_bulk() {
        let edges = vec![(0, 1), (0, 2), (1, 2), (3, 0)];
        let graph = Graph::from_edges(4, &edges, None).unwrap();

        assert_eq!(graph.degrees(), vec![2, 1, 0, 1]);
        let per_node: Vec<u32> = (0..graph.num_nodes() as NodeId)
            .map(|n| graph.degree(n) as u32)
            .collect();
        assert_eq!(graph.degrees(), per_node);

        let empty = Graph::from_edges(0, &[], None).unwrap();
        assert!(empty.degrees().is_empty());
    }

    #[test]
    fn test_weighted_graph() {
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let weights = vec![0.5, 1.0, 0.3];
        let graph = Graph::from_edges(3, &edges, Some(&weights)).unwrap();

        assert_eq!(graph.neighbor_weights(0), Some(&[0.5, 1.0][..]));
        assert_eq!(graph.neighbor_weights(1), Some(&[0.3][..]));

        graph.validate().unwrap();
    }

    #[test]
    fn test_empty_graph() {
        let graph = Graph::from_edges(0, &[], None).unwrap();
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_edges(), 0);
        graph.validate().unwrap();
    }

    #[test]
    fn test_disconnected_nodes() {
        // Nodes 0 and 2 have no edges
        let edges = vec![(1, 3), (3, 1)];
        let graph = Graph::from_edges(4, &edges, None).unwrap();

        assert_eq!(graph.degree(0), 0);
        assert_eq!(graph.degree(1), 1);
        assert_eq!(graph.degree(2), 0);
        assert_eq!(graph.degree(3), 1);

        graph.validate().unwrap();
    }

    #[test]
    fn test_single_node_graph() {
        // Single node with no edges
        let graph = Graph::from_edges(1, &[], None).unwrap();
        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(graph.num_edges(), 0);
        assert_eq!(graph.degree(0), 0);
        assert!(graph.neighbors(0).is_empty());
        graph.validate().unwrap();
    }

    #[test]
    fn test_single_node_self_loop() {
        // Single node with self-loop
        let edges = vec![(0, 0)];
        let graph = Graph::from_edges(1, &edges, None).unwrap();
        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.degree(0), 1);
        assert_eq!(graph.neighbors(0), &[0u32][..]);
        graph.validate().unwrap();
    }

    #[test]
    fn test_hub_node() {
        // Node 0 is a hub with many outgoing edges
        let num_targets = 1000;
        let edges: Vec<(u32, u32)> = (1..=num_targets).map(|i| (0, i)).collect();
        let graph = Graph::from_edges(num_targets as usize + 1, &edges, None).unwrap();

        assert_eq!(graph.degree(0), num_targets as usize);
        assert_eq!(graph.neighbors(0).len(), num_targets as usize);

        // Other nodes have no outgoing edges
        for i in 1..=num_targets {
            assert_eq!(graph.degree(i), 0);
        }

        graph.validate().unwrap();
    }

    #[test]
    fn test_out_of_bounds_access() {
        let edges = vec![(0, 1), (1, 0)];
        let graph = Graph::from_edges(2, &edges, None).unwrap();

        // Access node >= num_nodes should return 0/empty, not panic
        assert_eq!(graph.degree(2), 0);
        assert_eq!(graph.degree(100), 0);
        assert_eq!(graph.degree(u32::MAX), 0);
        assert!(graph.neighbors(2).is_empty());
        assert!(graph.neighbors(100).is_empty());
        assert!(graph.neighbors(u32::MAX).is_empty());
        assert_eq!(graph.edge_offset(100), 0);
        assert!(graph.neighbor_weights(100).is_none()); // No weights on this graph
    }

    #[test]
    fn test_invalid_source_returns_error() {
        let edges = vec![(5, 0)];
        let result = Graph::from_edges(3, &edges, None);
        assert!(result.is_err());
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("source node 5 exceeds num_nodes 3")
        );
    }

    #[test]
    fn test_out_of_bounds_with_weights() {
        let edges = vec![(0, 1)];
        let weights = vec![1.0];
        let graph = Graph::from_edges(2, &edges, Some(&weights)).unwrap();

        // Out of bounds with weights should return empty slice
        assert!(graph.neighbor_weights(100).unwrap().is_empty());
    }

    #[test]
    fn test_set_timestamps_basic() {
        // Graph: 0 -> 1, 2; 1 -> 2; 2 -> (none)
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let mut graph = Graph::from_edges(3, &edges, None).unwrap();

        let timestamps = vec![100.0, 200.0, 300.0];
        graph.set_timestamps(timestamps).unwrap();

        assert!(graph.has_timestamps());
        assert_eq!(graph.neighbor_timestamps(0), Some(&[100.0, 200.0][..]));
        assert_eq!(graph.neighbor_timestamps(1), Some(&[300.0][..]));
        assert_eq!(graph.neighbor_timestamps(2), Some(&[][..]));
    }

    #[test]
    fn test_timestamps_parallel_to_edges() {
        // Build a graph with known CSR layout and verify timestamps align
        // Node 0: edges to 1, 2 (offsets 0..2)
        // Node 1: edge to 0 (offset 2..3)
        // Node 2: edges to 0, 1 (offsets 3..5)
        let edges = vec![(0, 1), (0, 2), (1, 0), (2, 0), (2, 1)];
        let mut graph = Graph::from_edges(3, &edges, None).unwrap();

        let timestamps = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        graph.set_timestamps(timestamps).unwrap();

        // Node 0 has 2 neighbors -> timestamps at positions 0, 1
        let ts0 = graph.neighbor_timestamps(0).unwrap();
        assert_eq!(ts0.len(), graph.degree(0));
        assert_eq!(ts0, &[1.0, 2.0]);

        // Node 1 has 1 neighbor -> timestamp at position 2
        let ts1 = graph.neighbor_timestamps(1).unwrap();
        assert_eq!(ts1.len(), graph.degree(1));
        assert_eq!(ts1, &[3.0]);

        // Node 2 has 2 neighbors -> timestamps at positions 3, 4
        let ts2 = graph.neighbor_timestamps(2).unwrap();
        assert_eq!(ts2.len(), graph.degree(2));
        assert_eq!(ts2, &[4.0, 5.0]);
    }

    #[test]
    fn test_set_timestamps_returns_error_on_mismatch() {
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let mut graph = Graph::from_edges(3, &edges, None).unwrap();
        let err = graph.set_timestamps(vec![1.0, 2.0]).unwrap_err();
        assert_eq!(err.got, 2);
        assert_eq!(err.expected, 3);
    }

    #[test]
    fn test_timestamps_none_by_default() {
        let edges = vec![(0, 1), (1, 0)];
        let graph = Graph::from_edges(2, &edges, None).unwrap();

        assert!(!graph.has_timestamps());
        assert!(graph.neighbor_timestamps(0).is_none());
        assert!(graph.neighbor_timestamps(1).is_none());
    }
}
