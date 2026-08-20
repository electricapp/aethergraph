//! Dynamic graph: vertex array + per-vertex C-tree neighbor lists.
//!
//! Single-writer, multi-reader via functional C-trees. Edge inserts
//! path-copy the affected C-tree; the old version stays intact for any
//! reader holding a [`ReadGuard`](crate::ReadGuard), which is what defers
//! recycling of the slots the writer retired. The vertex array uses atomic
//! stores for the root pointer swap.
//!
//! Read path (sampling): zero allocations, lock-free.
//! Write path (edge insert): arena bump-alloc only, no locks.
//!
//! The single-writer invariant is enforced at runtime by [`Writer`] —
//! `DynamicGraph::writer()` returns `None` if a writer already exists.

use crate::arena::Arena;
use crate::chunk::Chunk;
use crate::ctree::CTree;
use crate::dirty::DirtyBitmap;
use crate::pad::CachePadded;
use crate::snapshot::Snapshot;
use crate::writer::InsertError;
#[cfg(feature = "wal")]
use crate::writer::Writer;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use aether_epoch::{Epoch, EpochClock};

#[cfg(feature = "wal")]
use crate::wal::{WalError, WalWriter};
#[cfg(feature = "wal")]
use std::path::Path;

/// Lock-free dynamic graph with O(log degree) edge insert and O(degree)
/// neighbor access.
///
/// Each vertex's neighbor list is a C-tree (balanced tree of sorted chunks)
/// stored in a bump-allocating arena. Edge inserts create new tree nodes
/// via path copying; the superseded tree stays intact for any reader holding
/// a [`ReadGuard`](crate::ReadGuard), which is what defers slot recycling.
pub struct DynamicGraph {
    /// Per-vertex C-tree root offsets. Atomic for lock-free root swaps.
    /// NULL (u32::MAX) means no neighbors.
    pub(crate) roots: Vec<AtomicU32>,
    /// Arena for all C-tree nodes (chunks + interior nodes).
    pub(crate) arena: Arena,
    /// Number of vertices (fixed at construction).
    pub(crate) num_vertices: usize,
    /// Total edge count (informational, not used for correctness).
    /// Updated once per committed writer guard, not per edge; padded so
    /// the update doesn't share a line with fields readers poll.
    pub(crate) num_edges: CachePadded<AtomicU64>,
    /// Dirty-node bitmap for historical embedding tracking.
    /// Set on writer commit, cleared at epoch boundary.
    pub(crate) dirty: DirtyBitmap,
    /// Single-writer guard. `true` while a `Writer` exists.
    pub(crate) writer_locked: AtomicBool,
    /// Poison flag — set when a `Writer` is dropped during unwinding (a
    /// panic mid-insert) or when a WAL append/fsync failed during the
    /// guard's lifetime. Once poisoned, bookkeeping (num_edges counter,
    /// dirty bitmap, WAL contents) may be out of step with the published
    /// roots, so no further writes are allowed: new `writer()` calls
    /// return [`crate::WriterError::Poisoned`]. Every published root points at a
    /// fully-built tree — roots are stored only after their nodes are
    /// written and WAL-appended — so reads (`degree`, `neighbors_into`,
    /// `has_edge`) keep working at their last consistent state.
    pub(crate) poisoned: AtomicBool,
    /// Shared monotonic version clock. Advanced once per successful writer
    /// drop (skipped on panic-poisoned drops). Readers pin the current
    /// epoch to coordinate consistent multi-source reads with subsystems
    /// like the feature store; today it's an opaque counter, but the same
    /// `Arc<EpochClock>` is the join point for future MVCC.
    pub(crate) epoch: Arc<EpochClock>,
    /// Latest committed snapshot, replaced at each writer-guard commit.
    /// Always live, so its epoch is always pinned — any snapshot
    /// [`acquire`](Self::acquire) hands out is protected from publication
    /// onward.
    pub(crate) latest: Mutex<Snapshot>,
    /// Optional write-ahead log. When present, every successful
    /// `Writer::insert_edge` appends a record; `Writer::drop` fsyncs.
    /// The `Mutex` is uncontended in practice (the surrounding `Writer`
    /// guard already enforces single-writer), but we still need interior
    /// mutability since the inserts go through `&self`.
    #[cfg(feature = "wal")]
    pub(crate) wal: Option<Mutex<WalWriter>>,
}

impl DynamicGraph {
    /// Create a graph with `num_vertices` vertices and no edges.
    ///
    /// `arena_bytes` controls the arena capacity, at most
    /// [`Arena::MAX_CAPACITY`] (32 GiB — offsets are u32 slot indices
    /// with a tag bit). Each insert path-copies its tree path, but the
    /// superseded nodes are recycled once no concurrent reader can still
    /// observe them, so steady-state consumption tracks live edges plus a
    /// bounded recycling lag rather than total inserts. [`compact`]
    /// remains the escape hatch that repacks everything perfectly
    /// (~6–10 bytes per live edge) and clears any leaked slots.
    ///
    /// [`compact`]: Self::compact
    ///
    /// The graph owns a private [`EpochClock`]. Use [`new_with_epoch`] to
    /// share a clock with another subsystem (e.g. a feature store).
    ///
    /// [`new_with_epoch`]: Self::new_with_epoch
    pub fn new(num_vertices: usize, arena_bytes: usize) -> Self {
        Self::new_with_epoch(num_vertices, arena_bytes, Arc::new(EpochClock::new()))
    }

    /// Create a graph that shares the given [`EpochClock`] with other
    /// subsystems. Each successful writer-guard drop advances the clock.
    pub fn new_with_epoch(num_vertices: usize, arena_bytes: usize, epoch: Arc<EpochClock>) -> Self {
        // Vertex IDs are u32 throughout the API.
        assert!(
            num_vertices <= u32::MAX as usize,
            "num_vertices {num_vertices} exceeds u32::MAX"
        );
        let mut roots = Vec::with_capacity(num_vertices);
        for _ in 0..num_vertices {
            roots.push(AtomicU32::new(crate::ctree::NULL));
        }
        let arena = Arena::new(arena_bytes);
        let latest = Mutex::new(crate::snapshot::empty_snapshot(
            num_vertices,
            arena.pins(),
            epoch.current().as_u64(),
        ));
        Self {
            roots,
            arena,
            num_vertices,
            num_edges: CachePadded(AtomicU64::new(0)),
            dirty: DirtyBitmap::new(num_vertices),
            writer_locked: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            epoch,
            latest,
            #[cfg(feature = "wal")]
            wal: None,
        }
    }

    /// Open a graph backed by an append-only write-ahead log. If the WAL
    /// at `path` already contains records (a previous run's edges), they
    /// are replayed into the in-memory state before this returns; if the
    /// WAL ends in a torn record (mid-write crash), the file is
    /// truncated to the last clean record.
    ///
    /// Replay runs under one writer guard, so `current_epoch()` advances
    /// exactly once for the whole recovery — treat the post-recovery
    /// epoch as a fresh starting point, not as comparable to any epoch
    /// value from before the restart.
    #[cfg(feature = "wal")]
    pub fn open_with_wal(
        path: impl AsRef<Path>,
        num_vertices: usize,
        arena_bytes: usize,
    ) -> Result<Self, WalError> {
        Self::open_with_wal_and_epoch(path, num_vertices, arena_bytes, Arc::new(EpochClock::new()))
    }

    /// Like [`open_with_wal`], but the graph shares the provided clock
    /// rather than minting a private one.
    ///
    /// [`open_with_wal`]: Self::open_with_wal
    #[cfg(feature = "wal")]
    pub fn open_with_wal_and_epoch(
        path: impl AsRef<Path>,
        num_vertices: usize,
        arena_bytes: usize,
        epoch: Arc<EpochClock>,
    ) -> Result<Self, WalError> {
        let path = path.as_ref();

        // Build a fresh in-memory graph, then replay records into it
        // before opening the WalWriter that will accept new appends. We
        // deliberately leave `graph.wal = None` during replay — we do
        // NOT want to re-log records we're reading from the log.
        //
        // One writer guard spans the whole replay: recovery is one
        // logical commit, so it costs one acquire/release/epoch-advance
        // total instead of five atomics per record across a
        // potentially-multi-GB log.
        let mut graph = Self::new_with_epoch(num_vertices, arena_bytes, Arc::clone(&epoch));

        // The WAL does not record the graph dimensions it was written
        // against, so a mismatched `num_vertices` or an undersized arena
        // surfaces here, record by record. Both must abort recovery: a
        // silently partial replay would present itself as a smaller but
        // valid graph.
        let outcome = {
            // Lazy guard: an empty WAL applies no records and must not
            // advance the epoch (the guard's drop commits once).
            let mut writer: Option<Writer<'_>> = None;
            crate::wal::replay(path, |rec| {
                let w = writer.get_or_insert_with(|| graph.writer_or_panic());
                match w.insert_edge(rec.src, rec.dst) {
                    Ok(_) => Ok(()),
                    Err(InsertError::VertexOutOfRange { src, dst }) => {
                        Err(WalError::RecordOutOfRange {
                            src,
                            dst,
                            num_vertices: num_vertices as u64,
                        })
                    }
                    Err(InsertError::ArenaFull) => Err(WalError::ReplayArenaFull),
                    // `graph.wal` is None during replay, so no append happens.
                    Err(InsertError::WalAppend) => unreachable!("no WAL attached during replay"),
                }
            })?
        };

        // If the WAL ended in a torn record, truncate so future appends
        // sit on top of clean data. The truncation is fsynced before any
        // new appends: an unsynced set_len could be undone by a later
        // crash, resurrecting the discarded torn bytes mid-log.
        if let Some(off) = outcome.truncate_to {
            std::fs::OpenOptions::new()
                .write(true)
                .open(path)
                .and_then(|f| {
                    f.set_len(off)?;
                    f.sync_data()
                })
                .map_err(WalError::Io)?;
        }

        let writer = WalWriter::create_or_open(path)?;
        graph.wal = Some(Mutex::new(writer));
        Ok(graph)
    }

    /// Read the current epoch. Use this to pin a version before issuing a
    /// multi-source read; later calls into subsystems sharing the same
    /// [`EpochClock`] can be range-checked against this pin.
    #[inline]
    pub fn current_epoch(&self) -> Epoch {
        self.epoch.current()
    }

    /// Access the shared epoch clock. Other subsystems clone the `Arc` to
    /// observe writer commits.
    #[inline]
    pub fn epoch_clock(&self) -> &Arc<EpochClock> {
        &self.epoch
    }

    /// Has this graph been poisoned by a panicking writer?
    #[inline]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Number of vertices.
    #[inline(always)]
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// Total edge count. Reflects committed writer guards: a live guard's
    /// inserts are folded in when it drops.
    #[inline(always)]
    pub fn num_edges(&self) -> u64 {
        self.num_edges.0.load(Ordering::Relaxed)
    }

    /// Get the C-tree root for a vertex. Out-of-range vertices read as
    /// empty rather than panicking — the sampler's inner loops shouldn't
    /// carry a panic edge for a bounds condition the caller can't hit
    /// with valid IDs.
    #[inline(always)]
    fn tree_for(&self, vertex: u32) -> CTree {
        let root = match self.roots.get(vertex as usize) {
            Some(r) => r.load(Ordering::Acquire),
            None => crate::ctree::NULL,
        };
        CTree { root }
    }

    /// Degree of a vertex.
    #[inline]
    pub fn degree(&self, vertex: u32) -> usize {
        let _gate = self.arena.read_guard();
        self.tree_for(vertex).count(&self.arena)
    }

    /// Iterate a vertex's neighbors in sorted order. Calls `f` for each chunk.
    ///
    /// Zero allocations. The chunks are read directly from the arena.
    /// Safe to call concurrently with edge inserts — readers see a
    /// consistent snapshot (the tree that was current when they read the
    /// root), and the traversal holds the arena's reader gate so recycled
    /// slots can never be rewritten underneath it. Keep the callback
    /// short: gate time delays slot reuse for the writer.
    #[inline]
    pub fn for_each_chunk(&self, vertex: u32, f: impl FnMut(&Chunk)) {
        let _gate = self.arena.read_guard();
        self.tree_for(vertex).for_each_chunk(&self.arena, f);
    }

    /// Collect all neighbors of `vertex` into `buf` (sorted).
    ///
    /// Clears `buf` first. For GNN sampling, use this to get a flat
    /// slice suitable for Floyd's algorithm.
    #[inline]
    pub fn neighbors_into(&self, vertex: u32, buf: &mut Vec<u32>) {
        buf.clear();
        let _gate = self.arena.read_guard();
        self.tree_for(vertex).collect_into(&self.arena, buf);
    }

    /// Check if edge (src → dst) exists.
    #[inline]
    pub fn has_edge(&self, src: u32, dst: u32) -> bool {
        let _gate = self.arena.read_guard();
        self.tree_for(src).contains(&self.arena, dst)
    }

    /// Arena usage stats.
    pub fn arena_used(&self) -> usize {
        self.arena.used()
    }

    pub fn arena_capacity(&self) -> usize {
        self.arena.capacity()
    }

    /// Snapshot the dynamic graph into raw CSR arrays.
    ///
    /// Returns `(offsets, edges)` where `offsets[v]` is the start of vertex
    /// `v`'s neighbors in `edges`. `offsets` has length `num_vertices + 1`.
    ///
    /// This is O(V + E) time and allocates fresh vectors for the CSR data.
    /// Call once per epoch, not per batch — the returned arrays can be used
    /// to construct a static `Graph` in the PyO3 layer.
    ///
    /// # Concurrency
    ///
    /// Safe to call concurrently with edge inserts. Each vertex's neighbor
    /// list is read from the C-tree root that was current at read time, so
    /// the snapshot is a consistent (though not globally atomic) view.
    pub fn snapshot_csr(&self) -> (Vec<u64>, Vec<u32>) {
        let mut offsets = Vec::with_capacity(self.num_vertices + 1);
        let mut edges = Vec::with_capacity(self.num_edges() as usize);

        offsets.push(0);
        // Iterate via usize to avoid u32 truncation when num_vertices is large.
        // Each vertex's neighbors are appended straight onto `edges`; the
        // running length after each append is the next CSR offset. The
        // reader gate is entered per vertex, not for the whole O(V + E)
        // snapshot, so the writer's slot recycling keeps making progress
        // while the snapshot runs.
        for v in 0..self.num_vertices {
            let _gate = self.arena.read_guard();
            self.tree_for(v as u32)
                .collect_into(&self.arena, &mut edges);
            offsets.push(edges.len() as u64);
        }

        (offsets, edges)
    }

    /// Drain all dirty node IDs and atomically clear the bitmap.
    ///
    /// Returns the set of vertices that had edges inserted since the last
    /// drain or clear. Order is ascending by vertex ID within each word.
    pub fn drain_dirty(&self) -> Vec<u32> {
        self.dirty.drain_dirty()
    }

    /// Number of vertices currently marked dirty.
    pub fn dirty_count(&self) -> usize {
        self.dirty.count_dirty()
    }

    /// Check if a vertex is dirty (received new edges since last clear/drain).
    #[inline]
    pub fn is_dirty(&self, vertex: u32) -> bool {
        self.dirty.is_dirty(vertex)
    }

    /// Clear all dirty bits without collecting them.
    pub fn clear_dirty(&self) {
        self.dirty.clear_all();
    }

    /// Build from a batch of edges.
    ///
    /// Convenience for trusted input: duplicate edges are skipped and
    /// edges referencing vertices ≥ `num_vertices` are dropped. Use
    /// [`Writer::insert_edge`] directly when those need to be surfaced,
    /// or compare `num_edges()` against the input length afterwards.
    /// Panics if the arena fills up.
    pub fn from_edges(num_vertices: usize, edges: &[(u32, u32)], arena_bytes: usize) -> Self {
        let graph = Self::new(num_vertices, arena_bytes);
        {
            let mut writer = graph.writer_or_panic();
            for &(src, dst) in edges {
                match writer.insert_edge(src, dst) {
                    Ok(_) | Err(InsertError::VertexOutOfRange { .. }) => {}
                    Err(InsertError::ArenaFull) => {
                        panic!("DynamicGraph::from_edges: arena full ({arena_bytes} bytes)")
                    }
                    // A graph built by `Self::new` carries no WAL.
                    #[cfg(feature = "wal")]
                    Err(InsertError::WalAppend) => unreachable!("no WAL attached"),
                }
            }
        }
        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompactError, WriterError};

    #[test]
    fn empty_graph() {
        let g = DynamicGraph::new(100, 1 << 20);
        assert_eq!(g.num_vertices(), 100);
        assert_eq!(g.num_edges(), 0);
        assert_eq!(g.degree(0), 0);
        assert!(!g.has_edge(0, 1));
    }

    #[test]
    fn insert_and_query() {
        let g = DynamicGraph::new(100, 1 << 20);
        let mut w = g.writer_or_panic();
        assert!(w.insert_edge(0, 10).unwrap());
        assert!(w.insert_edge(0, 20).unwrap());
        assert!(w.insert_edge(0, 5).unwrap());
        drop(w);

        assert_eq!(g.degree(0), 3);
        assert!(g.has_edge(0, 10));
        assert!(g.has_edge(0, 20));
        assert!(g.has_edge(0, 5));
        assert!(!g.has_edge(0, 15));

        let mut buf = Vec::new();
        g.neighbors_into(0, &mut buf);
        assert_eq!(buf, vec![5, 10, 20]);
    }

    #[test]
    fn duplicate_edge() {
        let g = DynamicGraph::new(100, 1 << 20);
        let mut w = g.writer_or_panic();
        assert!(w.insert_edge(0, 10).unwrap());
        assert!(!w.insert_edge(0, 10).unwrap());
        drop(w);
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.num_edges(), 1);
    }

    #[test]
    fn multiple_vertices() {
        let g = DynamicGraph::new(100, 1 << 20);
        let mut w = g.writer_or_panic();
        w.insert_edge(0, 10).unwrap();
        w.insert_edge(0, 20).unwrap();
        w.insert_edge(1, 30).unwrap();
        w.insert_edge(1, 40).unwrap();
        w.insert_edge(1, 50).unwrap();
        drop(w);

        assert_eq!(g.degree(0), 2);
        assert_eq!(g.degree(1), 3);
        assert_eq!(g.degree(2), 0);
    }

    #[test]
    fn high_degree_vertex() {
        let g = DynamicGraph::new(1000, 1 << 20);
        let mut w = g.writer_or_panic();
        for i in 0..500u32 {
            assert!(w.insert_edge(0, i).unwrap());
        }
        drop(w);
        assert_eq!(g.degree(0), 500);

        let mut buf = Vec::new();
        g.neighbors_into(0, &mut buf);
        assert_eq!(buf.len(), 500);
        for win in buf.windows(2) {
            assert!(win[0] < win[1]);
        }
    }

    #[test]
    fn from_edges_batch() {
        let edges: Vec<(u32, u32)> = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 3),
        ];
        let g = DynamicGraph::from_edges(4, &edges, 1 << 20);
        assert_eq!(g.degree(0), 3);
        assert_eq!(g.degree(1), 2);
        assert_eq!(g.degree(2), 3);
        assert_eq!(g.degree(3), 0);
    }

    #[test]
    fn snapshot_csr_basic() {
        let g = DynamicGraph::new(4, 1 << 20);
        {
            let mut w = g.writer_or_panic();
            w.insert_edge(0, 1).unwrap();
            w.insert_edge(0, 2).unwrap();
            w.insert_edge(0, 3).unwrap();
            w.insert_edge(1, 0).unwrap();
            w.insert_edge(1, 2).unwrap();
            w.insert_edge(2, 0).unwrap();
        }

        let (offsets, edges) = g.snapshot_csr();
        assert_eq!(offsets.len(), 5);
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1] - offsets[0], 3);
        assert_eq!(&edges[offsets[0] as usize..offsets[1] as usize], &[1, 2, 3]);
        assert_eq!(offsets[2] - offsets[1], 2);
        assert_eq!(&edges[offsets[1] as usize..offsets[2] as usize], &[0, 2]);
        assert_eq!(offsets[3] - offsets[2], 1);
        assert_eq!(&edges[offsets[2] as usize..offsets[3] as usize], &[0]);
        assert_eq!(offsets[4] - offsets[3], 0);
    }

    #[test]
    fn snapshot_csr_empty() {
        let g = DynamicGraph::new(3, 1 << 20);
        let (offsets, edges) = g.snapshot_csr();
        assert_eq!(offsets, vec![0, 0, 0, 0]);
        assert!(edges.is_empty());
    }

    #[test]
    fn dirty_tracking_basic() {
        let g = DynamicGraph::new(100, 1024);
        assert_eq!(g.dirty_count(), 0);
        let mut w = g.writer_or_panic();
        w.insert_edge(0, 1).unwrap();
        drop(w);
        assert!(g.is_dirty(0));
        assert!(g.is_dirty(1));
        assert!(!g.is_dirty(2));
        assert_eq!(g.dirty_count(), 2);
    }

    #[test]
    fn dirty_drain() {
        let g = DynamicGraph::new(100, 1024);
        {
            let mut w = g.writer_or_panic();
            w.insert_edge(0, 1).unwrap();
            w.insert_edge(2, 3).unwrap();
        }
        let mut dirty = g.drain_dirty();
        dirty.sort_unstable();
        assert_eq!(dirty, vec![0, 1, 2, 3]);
        assert_eq!(g.dirty_count(), 0);
    }

    #[test]
    fn dirty_duplicate_edge() {
        let g = DynamicGraph::new(100, 1024);
        let mut w = g.writer_or_panic();
        w.insert_edge(0, 1).unwrap();
        drop(w);
        g.clear_dirty();
        let mut w = g.writer_or_panic();
        w.insert_edge(0, 1).unwrap();
        drop(w);
        assert_eq!(g.dirty_count(), 0);
    }

    #[test]
    fn dirty_clear() {
        let g = DynamicGraph::new(100, 1024);
        let mut w = g.writer_or_panic();
        w.insert_edge(0, 1).unwrap();
        drop(w);
        g.clear_dirty();
        assert_eq!(g.dirty_count(), 0);
        assert!(!g.is_dirty(0));
    }

    #[test]
    fn writer_is_exclusive() {
        let g = DynamicGraph::new(10, 1024);
        let w1 = g.writer().expect("first writer should succeed");
        assert!(
            matches!(g.writer(), Err(WriterError::Busy)),
            "second writer must be rejected with Busy"
        );
        drop(w1);
        assert!(g.writer().is_ok(), "writer slot should be free again");
    }

    #[test]
    fn writer_panic_poisons_graph() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        use std::sync::Arc;

        let g = Arc::new(DynamicGraph::new(10, 1024));
        let g_for_panic = Arc::clone(&g);

        // Run an insert inside a writer that panics — the Writer's Drop
        // must observe std::thread::panicking() and poison the graph.
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut w = g_for_panic.writer().unwrap();
            let _ = w.insert_edge(0, 1);
            panic!("simulated mid-insert panic");
        }));
        assert!(result.is_err(), "the panic should have propagated");

        assert!(
            g.is_poisoned(),
            "graph must be poisoned after panicking writer"
        );
        assert!(
            matches!(g.writer(), Err(WriterError::Poisoned)),
            "no further writers should be issued after poisoning"
        );

        // Reads still work — they just observe whatever state existed at
        // last-consistent snapshot. `degree`/`has_edge` must not panic.
        let _ = g.degree(0);
        let _ = g.has_edge(0, 1);
    }

    #[test]
    fn ascending_high_degree_insert_completes() {
        // Streaming-ingest worst case: one hub vertex receiving
        // monotonically increasing neighbor IDs. Must stay balanced
        // (bounded depth, no stack overflow, sub-quadratic arena use).
        let g = DynamicGraph::new(100_000, 128 << 20);
        {
            let mut w = g.writer_or_panic();
            for i in 0..50_000u32 {
                assert!(w.insert_edge(0, i).unwrap());
            }
        }
        assert_eq!(g.degree(0), 50_000);
        let mut buf = Vec::new();
        g.neighbors_into(0, &mut buf);
        assert_eq!(buf.len(), 50_000);
        for w in buf.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn compact_reclaims_garbage_and_preserves_graph() {
        let mut g = DynamicGraph::new(5000, 16 << 20);
        {
            let mut w = g.writer_or_panic();
            for v in 0..64u32 {
                for d in 0..200u32 {
                    let _ = w.insert_edge(v, (d * 7 + v) % 5000);
                }
            }
        }
        let edges_before = g.num_edges();
        let mut snapshot_before: Vec<Vec<u32>> = Vec::new();
        for v in 0..64u32 {
            let mut buf = Vec::new();
            g.neighbors_into(v, &mut buf);
            snapshot_before.push(buf);
        }
        let used_before = g.arena_used();

        g.compact().unwrap();

        assert!(
            g.arena_used() < used_before / 4,
            "compact reclaimed too little: {} -> {}",
            used_before,
            g.arena_used()
        );
        assert_eq!(g.num_edges(), edges_before);
        for v in 0..64u32 {
            let mut buf = Vec::new();
            g.neighbors_into(v, &mut buf);
            assert_eq!(buf, snapshot_before[v as usize], "vertex {v} changed");
        }

        // Graph keeps accepting inserts after compaction.
        {
            let mut w = g.writer_or_panic();
            assert!(w.insert_edge(0, 4999).unwrap());
        }
        assert!(g.has_edge(0, 4999));
    }

    #[test]
    fn compact_with_capacity_grows_arena() {
        let mut g = DynamicGraph::new(1000, 1 << 20);
        {
            let mut w = g.writer_or_panic();
            for d in 0..1000u32 {
                let _ = w.insert_edge(0, d);
            }
        }
        g.compact_with_capacity(4 << 20).unwrap();
        assert_eq!(g.arena_capacity(), 4 << 20);
        assert_eq!(g.degree(0), 1000);
    }

    #[test]
    fn compact_too_small_fails_and_preserves_graph() {
        let mut g = DynamicGraph::new(2000, 1 << 20);
        {
            let mut w = g.writer_or_panic();
            for d in 0..2000u32 {
                let _ = w.insert_edge(0, d);
            }
        }
        // 2000 edges cannot fit in 256 bytes.
        assert_eq!(g.compact_with_capacity(256), Err(CompactError::ArenaFull));
        assert_eq!(g.degree(0), 2000, "failed compact must not lose data");
        assert_eq!(g.arena_capacity(), 1 << 20);
    }

    #[test]
    fn dirty_out_of_range_is_safe() {
        let g = DynamicGraph::new(10, 1024);
        // Out-of-range queries must not panic.
        assert!(!g.is_dirty(u32::MAX));
        assert!(!g.is_dirty(10));
    }

    #[test]
    fn insert_out_of_range_is_an_error() {
        let g = DynamicGraph::new(10, 1024);
        let mut w = g.writer_or_panic();
        assert_eq!(
            w.insert_edge(0, 10),
            Err(InsertError::VertexOutOfRange { src: 0, dst: 10 })
        );
        assert_eq!(
            w.insert_edge(10, 0),
            Err(InsertError::VertexOutOfRange { src: 10, dst: 0 })
        );
        drop(w);
        assert_eq!(g.num_edges(), 0);
        assert_eq!(g.dirty_count(), 0);
    }

    #[test]
    fn concurrent_read_during_write() {
        use std::sync::Arc;
        use std::thread;

        let g = Arc::new(DynamicGraph::new(1000, 1 << 22));

        let g_writer = Arc::clone(&g);
        let writer = thread::spawn(move || {
            let mut w = g_writer.writer_or_panic();
            for i in 0..1000u32 {
                let _ = w.insert_edge(0, i);
            }
        });

        let g_reader = Arc::clone(&g);
        let reader = thread::spawn(move || {
            let mut buf = Vec::new();
            for _ in 0..100 {
                g_reader.neighbors_into(0, &mut buf);
                for win in buf.windows(2) {
                    assert!(win[0] < win[1], "unsorted: {:?}", &buf[..20.min(buf.len())]);
                }
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();
    }
}
