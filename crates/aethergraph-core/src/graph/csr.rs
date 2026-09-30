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
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use tracing::trace;

/// Node ID type. u32 supports graphs up to 4 billion nodes.
pub type NodeId = u32;

/// Edge offset type. u64 supports graphs with up to 18 quintillion edges.
pub type EdgeOffset = u64;

/// Largest node count a graph may declare.
///
/// Node ids are `u32`; `u32::MAX` itself is never a valid id, so code that
/// walks ids can use it as a sentinel and `0..num_nodes as NodeId` never
/// wraps.
pub const MAX_NODES: usize = u32::MAX as usize;

/// Edge count above which scans and fills fan out across rayon.
const PARALLEL_EDGES: usize = 100_000;

/// Node count above which offsets scans fan out across rayon.
const PARALLEL_NODES: usize = 10_000;

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
/// Every value satisfies the shape invariant — `offsets.len() == num_nodes +
/// 1`, `offsets[0] == 0`, `offsets[num_nodes] == num_edges`, edge and weight
/// arrays of length `num_edges`, `num_nodes <= MAX_NODES`. Stronger
/// invariants are recorded as a [`GraphValidationMode`] proof level (see
/// [`Graph::validated`]): monotone offsets from `OffsetsOnly` up, every
/// destination below [`Graph::num_dst_nodes`] at `Full`.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of nodes (CSR rows).
    num_nodes: usize,

    /// Size of the destination id space. Equals `num_nodes` for a
    /// homogeneous graph; a per-edge-type CSR of a heterogeneous graph
    /// carries its destination type's count.
    num_dst_nodes: usize,

    /// Number of edges in the graph
    num_edges: usize,

    /// Backing storage for CSR arrays.
    storage: GraphStorage,

    /// Optional edge timestamps (parallel to edges array, set separately).
    /// Used for temporal sampling: only edges with timestamp < seed time are eligible.
    timestamps: Option<Arc<Vec<f64>>>,

    /// Strongest validation level this value has passed.
    proof: Proof,
}

/// Owned arrays are `Arc<Vec<_>>`: wrapping a filled `Vec` is free, where
/// `Arc<[_]>::from(vec)` would allocate and copy the whole array.
#[derive(Debug, Clone)]
enum GraphStorage {
    Owned {
        offsets: Arc<Vec<EdgeOffset>>,
        edges: Arc<Vec<NodeId>>,
        weights: Option<Arc<Vec<f32>>>,
    },
    Mapped {
        mmap: Arc<Mmap>,
        offsets_range: Range<usize>,
        edges_range: Range<usize>,
        weights_range: Option<Range<usize>>,
    },
}

/// How much of a graph's structure has been proven. Ordered: each level
/// implies every level below it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GraphValidationMode {
    /// Array lengths and the first/last offsets. Neighbor lookups stay
    /// memory-safe on a corrupt body (a bad offset pair reads as an empty
    /// list), but degrees and neighbor lists are then unspecified.
    HeaderOnly,
    /// Also monotone offsets: every node's neighbor range lies inside the
    /// edge array and the ranges tile it exactly.
    OffsetsOnly,
    /// Also every destination below [`Graph::num_dst_nodes`].
    Full,
}

impl GraphValidationMode {
    const fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::HeaderOnly,
            1 => Self::OffsetsOnly,
            _ => Self::Full,
        }
    }
}

/// Monotone proof level. Graph data is immutable, so a raised level stays
/// true; clones carry the level they were taken at.
#[derive(Debug)]
struct Proof(AtomicU8);

impl Proof {
    fn new(mode: GraphValidationMode) -> Self {
        Self(AtomicU8::new(mode as u8))
    }

    fn get(&self) -> GraphValidationMode {
        GraphValidationMode::from_u8(self.0.load(Ordering::Relaxed))
    }

    fn raise(&self, mode: GraphValidationMode) {
        self.0.fetch_max(mode as u8, Ordering::Relaxed);
    }
}

impl Clone for Proof {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

/// Smallest array worth a huge-page hint: one 2 MiB page.
const HUGEPAGE_HINT_MIN_BYTES: usize = 2 << 20;

fn hint_hugepage<T>(slice: &[T]) {
    let bytes = std::mem::size_of_val(slice);
    if bytes >= HUGEPAGE_HINT_MIN_BYTES {
        crate::internal::hint::advise_hugepage(slice.as_ptr().cast::<u8>(), bytes);
    }
}

/// A zero-filled array for `len` elements with the huge-page hint issued
/// before any page is touched.
///
/// `vec![0; n]` is a zeroed allocation, so a large array arrives as
/// untouched lazily-zeroed pages; advising then lets the first-touch faults
/// of the fill that follows take huge pages directly. Hinting a filled
/// array only queues it for khugepaged.
pub(crate) fn alloc_hinted<T: bytemuck::Zeroable + Clone>(len: usize) -> Vec<T> {
    let v = vec![T::zeroed(); len];
    hint_hugepage(&v);
    v
}

/// Shape checks every `Graph` satisfies, whatever its proof level. O(1).
fn check_shape(
    num_nodes: usize,
    num_dst_nodes: usize,
    offsets: &[EdgeOffset],
    num_edges: usize,
    weights_len: Option<usize>,
) -> Result<()> {
    anyhow::ensure!(
        num_nodes <= MAX_NODES,
        "num_nodes {num_nodes} exceeds the u32 node-id limit {MAX_NODES}"
    );
    anyhow::ensure!(
        num_dst_nodes <= MAX_NODES,
        "destination id space {num_dst_nodes} exceeds the u32 node-id limit {MAX_NODES}"
    );
    anyhow::ensure!(
        offsets.len() == num_nodes + 1,
        "offsets length {} should be num_nodes + 1 = {}",
        offsets.len(),
        num_nodes + 1
    );
    anyhow::ensure!(
        offsets[0] == 0,
        "first offset should be 0, got {}",
        offsets[0]
    );
    anyhow::ensure!(
        offsets[num_nodes] == num_edges as EdgeOffset,
        "last offset {} should equal num_edges {num_edges}",
        offsets[num_nodes]
    );
    if let Some(w) = weights_len {
        anyhow::ensure!(
            w == num_edges,
            "weights length {w} doesn't match num_edges {num_edges}"
        );
    }
    Ok(())
}

fn check_monotone(offsets: &[EdgeOffset]) -> Result<()> {
    let ok = if offsets.len() > PARALLEL_NODES {
        offsets.par_windows(2).all(|w| w[0] <= w[1])
    } else {
        offsets.windows(2).all(|w| w[0] <= w[1])
    };
    if !ok {
        let i = offsets
            .windows(2)
            .position(|w| w[0] > w[1])
            .expect("a window failed the monotone scan");
        anyhow::bail!(
            "offsets not monotonic at index {}: {} < {}",
            i + 1,
            offsets[i + 1],
            offsets[i]
        );
    }
    Ok(())
}

fn check_destinations(edges: &[NodeId], bound: usize) -> Result<()> {
    let ok = if edges.len() > PARALLEL_EDGES {
        edges.par_iter().all(|&d| (d as usize) < bound)
    } else {
        edges.iter().all(|&d| (d as usize) < bound)
    };
    if !ok {
        let i = edges
            .iter()
            .position(|&d| (d as usize) >= bound)
            .expect("an edge failed the destination scan");
        anyhow::bail!(
            "edge destination {} at edge {i} out of range [0, {bound})",
            edges[i]
        );
    }
    Ok(())
}

/// View a `u64` array as atomics for a parallel counting pass.
pub(crate) fn as_atomic(slice: &mut [u64]) -> &[AtomicU64] {
    const {
        assert!(std::mem::align_of::<AtomicU64>() == std::mem::align_of::<u64>());
        assert!(std::mem::size_of::<AtomicU64>() == std::mem::size_of::<u64>());
    };
    // SAFETY: AtomicU64 has u64's size and bit validity, and the const
    // assertion above pins its alignment to u64's on this target. The
    // exclusive borrow means no non-atomic access overlaps the view.
    unsafe { &*(std::ptr::from_mut::<[u64]>(slice) as *const [AtomicU64]) }
}

/// Fill `out` row by row in parallel. `row_offsets` (monotone, first 0,
/// last `out.len()`) partitions `out` into one slice per row, and
/// `fill(row, slice)` writes row `row`'s slice.
///
/// Rows are grouped into contiguous chunks, and `out` is split into one
/// disjoint `&mut` subslice per chunk before the parallel pass, so each task
/// owns its output outright.
pub(crate) fn par_fill_rows<T: Send>(
    out: &mut [T],
    row_offsets: &[EdgeOffset],
    fill: impl Fn(usize, &mut [T]) + Sync,
) {
    const ROWS_PER_CHUNK: usize = 4096;
    let rows = row_offsets.len().saturating_sub(1);
    assert_eq!(
        row_offsets.last().copied().unwrap_or(0),
        out.len() as EdgeOffset,
        "row offsets must span the output"
    );
    let mut chunks = Vec::with_capacity(rows.div_ceil(ROWS_PER_CHUNK));
    let mut rest = out;
    let mut start = 0;
    while start < rows {
        let end = (start + ROWS_PER_CHUNK).min(rows);
        let len = (row_offsets[end] - row_offsets[start]) as usize;
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(len);
        chunks.push((start, end, head));
        rest = tail;
        start = end;
    }
    chunks.into_par_iter().for_each(|(start, end, chunk)| {
        let base = row_offsets[start];
        for row in start..end {
            let lo = (row_offsets[row] - base) as usize;
            let hi = (row_offsets[row + 1] - base) as usize;
            fill(row, &mut chunk[lo..hi]);
        }
    });
}

/// The canonical guarded neighbor-range lookup over hoisted CSR arrays.
///
/// Every neighbor accessor — [`Graph::neighbors`], [`Graph::degree`], and
/// the sampler hop loops via [`CsrView`] — routes through this one
/// function, so an out-of-range node or a corrupt offset pair on an
/// unproven graph yields an empty range everywhere, with identical
/// semantics.
#[inline(always)]
fn neighbor_range_guarded(
    offsets: &[EdgeOffset],
    num_nodes: usize,
    num_edges: usize,
    node: NodeId,
) -> Range<usize> {
    let idx = node as usize;
    if idx >= num_nodes {
        return 0..0;
    }
    // Safe indexing: bounds are verified by construction (offsets.len() == num_nodes + 1)
    let start = offsets[idx] as usize;
    let end = offsets[idx + 1] as usize;
    if start > end || end > num_edges {
        return 0..0;
    }
    start..end
}

/// Borrowed CSR view with the storage dispatch already resolved.
///
/// The sampler hop loops hoist one view per call (per edge type for the
/// hetero sampler) so the per-node inner loop indexes raw slices with no
/// per-access storage `match`, while the neighbor-range guard stays the
/// single canonical one shared with [`Graph::neighbors`].
#[derive(Clone, Copy)]
pub struct CsrView<'a> {
    offsets: &'a [EdgeOffset],
    edges: &'a [NodeId],
    num_nodes: usize,
    num_edges: usize,
}

impl<'a> CsrView<'a> {
    /// Number of nodes covered by the view.
    #[inline(always)]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// The hoisted offsets array (`num_nodes + 1` entries) — exposed for
    /// software prefetch of upcoming offset words.
    #[inline(always)]
    pub fn offsets(&self) -> &'a [EdgeOffset] {
        self.offsets
    }

    /// The hoisted edges array — exposed for software prefetch of upcoming
    /// neighbor lines and for slicing a [`CsrView::neighbor_range`].
    #[inline(always)]
    pub fn edges(&self) -> &'a [NodeId] {
        self.edges
    }

    /// Guarded neighbor range for `node`; empty for out-of-range nodes or
    /// corrupt offsets. Semantics identical to [`Graph::neighbors`].
    #[inline(always)]
    pub fn neighbor_range(&self, node: NodeId) -> Range<usize> {
        neighbor_range_guarded(self.offsets, self.num_nodes, self.num_edges, node)
    }

    /// Guarded neighbor slice for `node`; empty for out-of-range nodes or
    /// corrupt offsets. Semantics identical to [`Graph::neighbors`].
    #[inline(always)]
    pub fn neighbors(&self, node: NodeId) -> &'a [NodeId] {
        &self.edges[self.neighbor_range(node)]
    }
}

impl Graph {
    /// Create an empty placeholder graph.
    ///
    /// Used for NVMe-backed samplers where the graph is file-backed
    /// and we don't need to keep it in memory.
    pub fn empty() -> Self {
        Self::from_trusted_parts(0, 0, vec![0], Vec::new(), None)
    }

    /// Creates a graph from CSR arrays, proving every invariant — shape,
    /// monotone offsets, destinations below `num_nodes` — before the graph
    /// exists. The result is `Full`-validated.
    ///
    /// # Arguments
    /// * `num_nodes` - Number of nodes
    /// * `offsets` - CSR offset array (length = num_nodes + 1)
    /// * `edges` - CSR edges array (destination nodes)
    /// * `weights` - Optional edge weights
    pub fn from_csr_arrays(
        num_nodes: usize,
        offsets: Vec<EdgeOffset>,
        edges: Vec<NodeId>,
        weights: Option<Vec<f32>>,
    ) -> Result<Self> {
        let graph = Self::from_csr_vecs(
            num_nodes,
            num_nodes,
            offsets,
            edges,
            weights,
            GraphValidationMode::Full,
        )?;
        // The caller filled these arrays, so the hint can only queue them
        // for khugepaged.
        hint_hugepage(graph.offsets_slice());
        hint_hugepage(graph.edges_slice());
        Ok(graph)
    }

    /// Validate owned CSR arrays up to `mode`, then build the graph.
    pub(crate) fn from_csr_vecs(
        num_nodes: usize,
        num_dst_nodes: usize,
        offsets: Vec<EdgeOffset>,
        edges: Vec<NodeId>,
        weights: Option<Vec<f32>>,
        mode: GraphValidationMode,
    ) -> Result<Self> {
        anyhow::ensure!(
            !offsets.is_empty(),
            "offsets array is empty; it needs num_nodes + 1 entries"
        );
        check_shape(
            num_nodes,
            num_dst_nodes,
            &offsets,
            edges.len(),
            weights.as_ref().map(Vec::len),
        )?;
        if mode >= GraphValidationMode::OffsetsOnly {
            check_monotone(&offsets)?;
        }
        if mode == GraphValidationMode::Full {
            check_destinations(&edges, num_dst_nodes)?;
        }
        Ok(Self {
            num_nodes,
            num_dst_nodes,
            num_edges: edges.len(),
            storage: GraphStorage::Owned {
                offsets: Arc::new(offsets),
                edges: Arc::new(edges),
                weights: weights.map(Arc::new),
            },
            timestamps: None,
            proof: Proof::new(mode),
        })
    }

    /// Build a graph from arrays whose construction already proves every
    /// invariant (a builder or a permutation of a `Full` graph).
    pub(crate) fn from_trusted_parts(
        num_nodes: usize,
        num_dst_nodes: usize,
        offsets: Vec<EdgeOffset>,
        edges: Vec<NodeId>,
        weights: Option<Vec<f32>>,
    ) -> Self {
        debug_assert!(
            check_shape(
                num_nodes,
                num_dst_nodes,
                &offsets,
                edges.len(),
                weights.as_ref().map(Vec::len)
            )
            .is_ok()
        );
        Self {
            num_nodes,
            num_dst_nodes,
            num_edges: edges.len(),
            storage: GraphStorage::Owned {
                offsets: Arc::new(offsets),
                edges: Arc::new(edges),
                weights: weights.map(Arc::new),
            },
            timestamps: None,
            proof: Proof::new(GraphValidationMode::Full),
        }
    }

    /// Creates a graph view over mmap-backed CSR bytes, validated up to
    /// `mode` before it exists.
    ///
    /// Range alignment and length divisibility are checked here, once, so
    /// the per-call slice accessors can reconstruct typed slices without
    /// re-validating on every `neighbors()` in the sampling hot loop.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_mapped_parts(
        num_nodes: usize,
        num_dst_nodes: usize,
        num_edges: usize,
        mmap: Arc<Mmap>,
        offsets_range: Range<usize>,
        edges_range: Range<usize>,
        weights_range: Option<Range<usize>>,
        mode: GraphValidationMode,
    ) -> Result<Self> {
        let base = mmap.as_ptr() as usize;
        check_range::<EdgeOffset>(base, mmap.len(), &offsets_range, "offsets")?;
        check_range::<NodeId>(base, mmap.len(), &edges_range, "edges")?;
        if let Some(ref w) = weights_range {
            check_range::<f32>(base, mmap.len(), w, "weights")?;
            anyhow::ensure!(
                w.len() == edges_range.len(),
                "weights section length {} doesn't match edges section {}",
                w.len(),
                edges_range.len()
            );
        }
        anyhow::ensure!(
            edges_range.len() == num_edges * std::mem::size_of::<NodeId>(),
            "edges section holds {} bytes for {num_edges} edges",
            edges_range.len()
        );
        anyhow::ensure!(
            !offsets_range.is_empty(),
            "offsets section is empty; it needs num_nodes + 1 entries"
        );
        let graph = Self {
            num_nodes,
            num_dst_nodes,
            num_edges,
            storage: GraphStorage::Mapped {
                mmap,
                offsets_range,
                edges_range,
                weights_range,
            },
            timestamps: None,
            proof: Proof::new(GraphValidationMode::HeaderOnly),
        };
        check_shape(
            num_nodes,
            num_dst_nodes,
            graph.offsets_slice(),
            num_edges,
            graph.weights_slice().map(<[f32]>::len),
        )?;
        graph.validate_with_mode(mode)?;
        Ok(graph)
    }

    /// Creates a new CSR graph from an edge list.
    ///
    /// # Arguments
    /// * `num_nodes` - Total number of nodes
    /// * `edges` - List of (source, destination) tuples
    /// * `weights` - Optional edge weights (must match edges length if provided)
    ///
    /// # Performance
    /// Counting sort, O(V + E) time. Neighbors keep their input order within
    /// each source. Validation and degree counting share one parallel pass;
    /// input already grouped by source fills with a parallel copy.
    pub fn from_edges(
        num_nodes: usize,
        edges: &[(NodeId, NodeId)],
        weights: Option<&[f32]>,
    ) -> Result<Self> {
        Self::build_csr(
            num_nodes,
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
        Self::from_bipartite_src_dst(num_nodes, num_nodes, src, dst, weights)
    }

    /// Creates a bipartite CSR: rows are sources in `0..num_src`,
    /// destinations lie in `0..num_dst`. The shape of one edge type in a
    /// heterogeneous graph, where the two endpoint types have their own id
    /// spaces — sizing the rows by `num_src` alone keeps a 1M-source,
    /// 1B-destination relation at 8 MB of offsets instead of 8 GB.
    pub fn from_bipartite_src_dst(
        num_src: usize,
        num_dst: usize,
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
        Self::build_csr(num_src, num_dst, src.len(), |i| src[i], |i| dst[i], weights)
    }

    /// Shared counting-sort CSR builder over an indexed edge accessor.
    ///
    /// Pass 1 validates both endpoints, counts degrees straight into the
    /// offsets array (as relaxed atomics, O(V) memory rather than per-thread
    /// counts), and notes whether the input is already grouped by source.
    /// Pass 2 fills: a parallel copy for grouped input, otherwise a serial
    /// stable scatter that walks the edges backwards and decrements the
    /// row ends in place, so it needs no cursor array beside the offsets.
    fn build_csr(
        num_nodes: usize,
        num_dst_nodes: usize,
        num_edges: usize,
        src_at: impl Fn(usize) -> NodeId + Sync,
        dst_at: impl Fn(usize) -> NodeId + Sync,
        weights: Option<&[f32]>,
    ) -> Result<Self> {
        trace!(
            "Building CSR graph: {} nodes, {} edges",
            num_nodes, num_edges
        );
        anyhow::ensure!(
            num_nodes <= MAX_NODES,
            "num_nodes {num_nodes} exceeds the u32 node-id limit {MAX_NODES}"
        );
        anyhow::ensure!(
            num_dst_nodes <= MAX_NODES,
            "destination id space {num_dst_nodes} exceeds the u32 node-id limit {MAX_NODES}"
        );
        if let Some(w) = weights {
            anyhow::ensure!(
                w.len() == num_edges,
                "weights length {} doesn't match edges length {}",
                w.len(),
                num_edges
            );
        }

        // offsets[s + 1] accumulates the degree of s; the prefix sum below
        // turns it into the row end in place.
        let mut offsets: Vec<EdgeOffset> = alloc_hinted(num_nodes + 1);
        let (first_bad, grouped) = {
            let counts = &as_atomic(&mut offsets)[1..];
            let scan = |range: Range<usize>| -> (usize, bool) {
                let mut grouped =
                    range.start == 0 || src_at(range.start - 1) <= src_at(range.start);
                let mut prev = 0;
                for i in range {
                    let s = src_at(i);
                    let d = dst_at(i);
                    if (s as usize) >= num_nodes || (d as usize) >= num_dst_nodes {
                        return (i, false);
                    }
                    grouped &= prev <= s;
                    prev = s;
                    counts[s as usize].fetch_add(1, Ordering::Relaxed);
                }
                (usize::MAX, grouped)
            };
            if num_edges > PARALLEL_EDGES {
                let chunk = num_edges.div_ceil(rayon::current_num_threads() * 4).max(1);
                (0..num_edges.div_ceil(chunk))
                    .into_par_iter()
                    .map(|c| scan(c * chunk..((c + 1) * chunk).min(num_edges)))
                    .reduce(|| (usize::MAX, true), |a, b| (a.0.min(b.0), a.1 && b.1))
            } else {
                scan(0..num_edges)
            }
        };
        if first_bad != usize::MAX {
            let (s, d) = (src_at(first_bad), dst_at(first_bad));
            if (s as usize) >= num_nodes {
                anyhow::bail!("source node {s} exceeds num_nodes {num_nodes} (edge {first_bad})");
            }
            anyhow::bail!(
                "destination node {d} exceeds num_nodes {num_dst_nodes} (edge {first_bad})"
            );
        }
        for i in 1..=num_nodes {
            offsets[i] += offsets[i - 1];
        }
        debug_assert_eq!(offsets[num_nodes], num_edges as EdgeOffset);

        let mut csr_edges: Vec<NodeId> = alloc_hinted(num_edges);
        let mut csr_weights: Option<Vec<f32>> = weights.map(|_| alloc_hinted(num_edges));

        if grouped {
            // Grouped by source: CSR order is input order.
            let fill = |i: usize, out: &mut NodeId| *out = dst_at(i);
            if num_edges > PARALLEL_EDGES {
                csr_edges
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, out)| fill(i, out));
            } else {
                csr_edges
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, out)| fill(i, out));
            }
            if let (Some(out), Some(w)) = (&mut csr_weights, weights) {
                out.copy_from_slice(w);
            }
        } else {
            // Backward stable scatter: offsets[s + 1] starts as row s's end
            // and is decremented per edge, so each row fills back to front
            // and keeps input order. It ends at row s's start, one slot to
            // the right of where the offsets array keeps it.
            for i in (0..num_edges).rev() {
                let s = src_at(i) as usize;
                offsets[s + 1] -= 1;
                let pos = offsets[s + 1] as usize;
                csr_edges[pos] = dst_at(i);
                if let (Some(out), Some(w)) = (&mut csr_weights, weights) {
                    out[pos] = w[i];
                }
            }
            offsets.copy_within(1.., 0);
            offsets[num_nodes] = num_edges as EdgeOffset;
        }

        Ok(Self::from_trusted_parts(
            num_nodes,
            num_dst_nodes,
            offsets,
            csr_edges,
            csr_weights,
        ))
    }

    /// Returns the number of nodes in the graph.
    #[inline(always)]
    pub const fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Size of the destination id space: once `Full`-validated, every edge
    /// destination is below it. Equal to [`Self::num_nodes`] for a
    /// homogeneous graph.
    #[inline(always)]
    pub const fn num_dst_nodes(&self) -> usize {
        self.num_dst_nodes
    }

    /// Whether sources and destinations share one id space — required by
    /// operations that relabel both ends with one permutation.
    #[inline(always)]
    pub const fn is_homogeneous(&self) -> bool {
        self.num_dst_nodes == self.num_nodes
    }

    /// Returns the number of edges in the graph.
    #[inline(always)]
    pub const fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Strongest validation level this graph has passed.
    ///
    /// At `Full`, every destination is below [`Self::num_dst_nodes`], so
    /// consumers need no per-edge range check. [`Self::validate_with_mode`]
    /// raises the level (once per graph value) when a caller needs more.
    #[inline(always)]
    pub fn validated(&self) -> GraphValidationMode {
        self.proof.get()
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

    /// Returns the degree (number of outgoing edges) of a node — always
    /// `self.neighbors(node).len()`.
    ///
    /// # Performance
    /// O(1) - single array lookup.
    #[inline(always)]
    pub fn degree(&self, node: NodeId) -> usize {
        self.neighbor_range(node).len()
    }

    /// Returns the degree of every node in one pass over the offsets array.
    ///
    /// Equivalent to calling [`Graph::degree`] for each node in
    /// `0..num_nodes`, but amortizes the per-call overhead — useful for FFI
    /// callers that would otherwise cross the boundary once per node. A
    /// degree past `u32::MAX` saturates.
    ///
    /// # Performance
    /// O(num_nodes) - a single sequential scan over the (possibly mmap'd)
    /// offsets array.
    pub fn degrees(&self) -> Vec<u32> {
        let num_edges = self.num_edges as EdgeOffset;
        self.offsets_slice()
            .windows(2)
            .map(|w| {
                if w[0] <= w[1] && w[1] <= num_edges {
                    u32::try_from(w[1] - w[0]).unwrap_or(u32::MAX)
                } else {
                    0
                }
            })
            .collect()
    }

    /// Returns the degree of each node in `nodes`, in input order.
    ///
    /// Unlike [`Self::degrees`], this touches only the requested nodes —
    /// the right shape for a sampling frontier or any batch of scattered
    /// IDs, where materializing all `num_nodes` degrees would dominate the
    /// work. The offsets lookups are a hardware gather where the CPU
    /// supports one, so the batch's cache misses overlap instead of
    /// serializing.
    ///
    /// Out-of-range nodes report degree 0, matching [`Self::degree`].
    pub fn degrees_of(&self, nodes: &[NodeId]) -> Vec<u32> {
        if self.validated() < GraphValidationMode::OffsetsOnly {
            // The gather reads raw offset pairs; without monotone offsets
            // it could disagree with the guarded `degree`.
            return nodes
                .iter()
                .map(|&n| u32::try_from(self.degree(n)).unwrap_or(u32::MAX))
                .collect();
        }
        let mut out = vec![0u32; nodes.len()];
        crate::internal::simd::gather_degrees(self.offsets_slice(), nodes, &mut out);
        out
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
        neighbor_range_guarded(self.offsets_slice(), self.num_nodes, self.num_edges, node)
    }

    /// Returns a borrowed [`CsrView`] with the storage dispatch resolved
    /// once, for loops that access many nodes' neighbor lists.
    #[inline(always)]
    pub fn csr_view(&self) -> CsrView<'_> {
        CsrView {
            offsets: self.offsets_slice(),
            edges: self.edges_slice(),
            num_nodes: self.num_nodes,
            num_edges: self.num_edges,
        }
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
        self.timestamps = Some(Arc::new(timestamps));
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
        self.timestamps.as_deref().map(Vec::as_slice)
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

    /// Validates every graph invariant (`Full`).
    ///
    /// # Performance
    /// O(1) once the graph is `Full`-validated; otherwise one parallel pass
    /// over whatever is not yet proven.
    pub fn validate(&self) -> Result<()> {
        self.validate_with_mode(GraphValidationMode::Full)
    }

    /// Proves the graph up to `mode`, recording the result so a later call
    /// at or below the proven level returns immediately. Only the levels
    /// above the current proof are checked.
    pub fn validate_with_mode(&self, mode: GraphValidationMode) -> Result<()> {
        let proven = self.proof.get();
        if proven >= mode {
            return Ok(());
        }
        trace!(?proven, ?mode, "Validating graph structure");
        if proven < GraphValidationMode::OffsetsOnly {
            check_monotone(self.offsets_slice())?;
            self.proof.raise(GraphValidationMode::OffsetsOnly);
        }
        if mode == GraphValidationMode::Full {
            check_destinations(self.edges_slice(), self.num_dst_nodes)?;
            self.proof.raise(GraphValidationMode::Full);
        }
        Ok(())
    }

    /// Returns statistics about the graph structure.
    ///
    /// # Performance
    /// One scan over the offsets array, parallel for large graphs.
    pub fn stats(&self) -> GraphStats {
        let num_edges = self.num_edges as EdgeOffset;
        let degree = |w: &[EdgeOffset]| -> u64 {
            if w[0] <= w[1] && w[1] <= num_edges {
                w[1] - w[0]
            } else {
                0
            }
        };
        let offsets = self.offsets_slice();
        let (max_degree, degree_sum) = if self.num_nodes > PARALLEL_NODES {
            offsets
                .par_windows(2)
                .map(|w| {
                    let d = degree(w);
                    (d, d)
                })
                .reduce(|| (0, 0), |(m1, s1), (m2, s2)| (m1.max(m2), s1 + s2))
        } else {
            offsets
                .windows(2)
                .map(degree)
                .fold((0, 0), |(m, s), d| (m.max(d), s + d))
        };

        let avg_degree = if self.num_nodes > 0 {
            degree_sum as f64 / self.num_nodes as f64
        } else {
            0.0
        };

        GraphStats {
            num_nodes: self.num_nodes,
            num_edges: self.num_edges,
            max_degree: max_degree as usize,
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
                // SAFETY: `from_mapped_parts` checked this range aligned,
                // in bounds, and sized for `EdgeOffset` (see `typed_slice`'s
                // contract); the mmap is immutable and outlives `self` via
                // the Arc.
                unsafe { typed_slice::<EdgeOffset>(&mmap[offsets_range.start..offsets_range.end]) }
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
                // SAFETY: `from_mapped_parts` checked this range aligned,
                // in bounds, and sized for `NodeId` (see `typed_slice`'s
                // contract); the mmap is immutable and outlives `self` via
                // the Arc.
                unsafe { typed_slice::<NodeId>(&mmap[edges_range.start..edges_range.end]) }
            }
        }
    }

    #[inline(always)]
    fn weights_slice(&self) -> Option<&[f32]> {
        match &self.storage {
            GraphStorage::Owned { weights, .. } => weights.as_deref().map(Vec::as_slice),
            GraphStorage::Mapped {
                mmap,
                weights_range,
                ..
            } => weights_range.as_ref().map(|range| {
                // SAFETY: `from_mapped_parts` checked this range aligned,
                // in bounds, and sized for `f32` (see `typed_slice`'s
                // contract); the mmap is immutable and outlives `self` via
                // the Arc.
                unsafe { typed_slice::<f32>(&mmap[range.start..range.end]) }
            }),
        }
    }
}

/// Check that an mmap byte range lies inside the mapping, starts aligned for
/// `T`, and spans a whole number of `T`s. Runs once per range at
/// [`Graph::from_mapped_parts`] so [`typed_slice`] can skip per-call
/// re-validation — re-checking on every access would put an alignment test
/// and division in the sampler's per-node path.
fn check_range<T>(base: usize, map_len: usize, range: &Range<usize>, what: &str) -> Result<()> {
    anyhow::ensure!(
        range.start <= range.end && range.end <= map_len,
        "mmap {what} range {range:?} lies outside the {map_len}-byte mapping"
    );
    anyhow::ensure!(
        (base + range.start).is_multiple_of(std::mem::align_of::<T>())
            && range.len().is_multiple_of(std::mem::size_of::<T>()),
        "mmap {what} range misaligned for {}",
        std::any::type_name::<T>()
    );
    Ok(())
}

/// Reinterpret an mmap byte range as a typed slice.
///
/// # Safety
/// - `bytes` must start aligned for `T` and its length must be a multiple of
///   `size_of::<T>()` — [`check_range`] establishes both, once per range, in
///   [`Graph::from_mapped_parts`].
/// - The backing memory must be immutable and outlive the returned slice
///   (the `Graph` holds the mmap via `Arc`, and the slice borrows `bytes`).
///
/// The `bytemuck::Pod` bound guarantees every bit pattern is a valid `T`.
#[inline(always)]
unsafe fn typed_slice<T: bytemuck::Pod>(bytes: &[u8]) -> &[T] {
    // SAFETY: alignment, size divisibility, immutability, and lifetime are
    // the caller's contract, stated above.
    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr().cast::<T>(),
            bytes.len() / std::mem::size_of::<T>(),
        )
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
    fn test_invalid_destination_names_the_edge() {
        let err = Graph::from_edges(3, &[(0, 1), (1, 7)], None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("destination node 7"), "got: {err}");
        assert!(err.contains("edge 1"), "got: {err}");
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

    /// Unsorted input takes the backward-scatter path, sorted input the
    /// parallel copy; both keep each source's neighbors in input order.
    #[test]
    fn test_build_keeps_input_order_on_both_fill_paths() {
        let unsorted = vec![(2, 5), (0, 3), (2, 1), (0, 4), (1, 0), (0, 1)];
        let w: Vec<f32> = (0..unsorted.len()).map(|i| i as f32).collect();
        let g = Graph::from_edges(6, &unsorted, Some(&w)).unwrap();
        assert_eq!(g.offsets(), &[0, 3, 4, 6, 6, 6, 6]);
        assert_eq!(g.neighbors(0), &[3, 4, 1]);
        assert_eq!(g.neighbor_weights(0), Some(&[1.0, 3.0, 5.0][..]));
        assert_eq!(g.neighbors(2), &[5, 1]);
        assert_eq!(g.neighbor_weights(2), Some(&[0.0, 2.0][..]));

        let mut sorted = unsorted.clone();
        sorted.sort_by_key(|&(s, _)| s);
        let g2 = Graph::from_edges(6, &sorted, None).unwrap();
        assert_eq!(g2.offsets(), g.offsets());
        assert_eq!(g2.edges(), g.edges());
    }

    /// A large unsorted edge list runs the parallel counting pass and must
    /// agree with a naive per-source bucketing.
    #[test]
    fn test_build_parallel_pass_matches_reference() {
        let n = 5_000u32;
        let edges: Vec<(u32, u32)> = (0..300_000u64)
            .map(|i| {
                let s = (i.wrapping_mul(2_654_435_761) % u64::from(n)) as u32;
                let d = (i.wrapping_mul(40_503) % u64::from(n)) as u32;
                (s, d)
            })
            .collect();
        let g = Graph::from_edges(n as usize, &edges, None).unwrap();
        let mut buckets = vec![Vec::new(); n as usize];
        for &(s, d) in &edges {
            buckets[s as usize].push(d);
        }
        for s in 0..n {
            assert_eq!(g.neighbors(s), &buckets[s as usize][..]);
        }
    }

    #[test]
    fn test_bipartite_rows_and_destination_space() {
        let g = Graph::from_bipartite_src_dst(2, 1_000, &[0, 1, 1], &[999, 5, 0], None).unwrap();
        assert_eq!(g.num_nodes(), 2);
        assert_eq!(g.num_dst_nodes(), 1_000);
        assert!(!g.is_homogeneous());
        assert_eq!(g.offsets().len(), 3);
        assert_eq!(g.neighbors(0), &[999]);
        assert_eq!(g.validated(), GraphValidationMode::Full);
        assert!(Graph::from_bipartite_src_dst(2, 1_000, &[2], &[0], None).is_err());
        assert!(Graph::from_bipartite_src_dst(2, 1_000, &[0], &[1_000], None).is_err());
    }

    #[test]
    fn test_from_csr_arrays_checks_every_invariant() {
        // Valid.
        let g = Graph::from_csr_arrays(3, vec![0, 2, 2, 3], vec![1, 2, 0], None).unwrap();
        assert_eq!(g.validated(), GraphValidationMode::Full);
        // Non-monotone offsets.
        assert!(Graph::from_csr_arrays(3, vec![0, 3, 1, 3], vec![1, 2, 0], None).is_err());
        // Destination out of range.
        assert!(Graph::from_csr_arrays(3, vec![0, 2, 2, 2], vec![1, 99], None).is_err());
        // Last offset disagrees with the edge count.
        assert!(Graph::from_csr_arrays(3, vec![0, 1, 1, 1], vec![1, 2], None).is_err());
        // Offsets length disagrees with num_nodes.
        assert!(Graph::from_csr_arrays(3, vec![0, 2], vec![1, 2], None).is_err());
        // Nonzero first offset.
        assert!(Graph::from_csr_arrays(1, vec![1, 1], vec![], None).is_err());
        // Empty offsets.
        assert!(Graph::from_csr_arrays(0, vec![], vec![], None).is_err());
        // Weights of the wrong length.
        assert!(Graph::from_csr_arrays(2, vec![0, 1, 1], vec![1], Some(vec![])).is_err());
    }

    #[test]
    fn test_node_count_limit() {
        assert!(Graph::from_src_dst(MAX_NODES + 1, &[], &[], None).is_err());
    }

    #[test]
    fn test_validation_records_and_raises_proof() {
        let g = Graph::from_csr_vecs(
            3,
            3,
            vec![0, 2, 2, 3],
            vec![1, 2, 0],
            None,
            GraphValidationMode::HeaderOnly,
        )
        .unwrap();
        assert_eq!(g.validated(), GraphValidationMode::HeaderOnly);
        g.validate_with_mode(GraphValidationMode::OffsetsOnly)
            .unwrap();
        assert_eq!(g.validated(), GraphValidationMode::OffsetsOnly);
        let cloned = g.clone();
        g.validate().unwrap();
        assert_eq!(g.validated(), GraphValidationMode::Full);
        assert_eq!(cloned.validated(), GraphValidationMode::OffsetsOnly);
    }

    /// Guarded accessors agree with each other on an unproven, corrupt
    /// body: degree is the neighbor list's length and the bulk variants
    /// match it.
    #[test]
    fn test_degree_agrees_with_neighbors_on_corrupt_offsets() {
        let g = Graph::from_csr_vecs(
            3,
            3,
            vec![0, 4, 0, 12],
            vec![0; 12],
            None,
            GraphValidationMode::HeaderOnly,
        )
        .unwrap();
        for n in 0..3 {
            assert_eq!(g.degree(n), g.neighbors(n).len());
        }
        let per_node: Vec<u32> = (0..3).map(|n| g.degree(n) as u32).collect();
        assert_eq!(g.degrees(), per_node);
        assert_eq!(g.degrees_of(&[0, 1, 2]), per_node);
        assert!(
            g.validate_with_mode(GraphValidationMode::OffsetsOnly)
                .is_err()
        );
        assert_eq!(g.validated(), GraphValidationMode::HeaderOnly);
    }

    #[test]
    fn test_par_fill_rows_covers_every_row() {
        let mut prefix = vec![0u64; 10_001];
        for i in 1..prefix.len() {
            prefix[i] = prefix[i - 1] + (i as u64 % 5);
        }
        let total = *prefix.last().unwrap() as usize;
        let mut out = vec![u32::MAX; total];
        par_fill_rows(&mut out, &prefix, |row, slice| {
            assert_eq!(slice.len() as u64, prefix[row + 1] - prefix[row]);
            slice.fill(row as u32);
        });
        for row in 0..prefix.len() - 1 {
            let (lo, hi) = (prefix[row] as usize, prefix[row + 1] as usize);
            assert!(out[lo..hi].iter().all(|&v| v == row as u32));
        }
        assert!(out.iter().all(|&v| v != u32::MAX));
    }
}
