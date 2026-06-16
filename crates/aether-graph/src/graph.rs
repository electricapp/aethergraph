//! Dynamic graph: vertex array + per-vertex C-tree neighbor lists.
//!
//! Single-writer, multi-reader via functional C-trees. Edge inserts
//! path-copy the affected C-tree; the old version remains valid for
//! concurrent readers. The vertex array uses atomic stores for the
//! root pointer swap.
//!
//! Read path (sampling): zero allocations, lock-free.
//! Write path (edge insert): arena bump-alloc only, no locks.
//!
//! The single-writer invariant is enforced at runtime by [`Writer`] —
//! `DynamicGraph::writer()` returns `None` if a writer already exists.

use crate::arena::Arena;
use crate::chunk::Chunk;
use crate::ctree::{CTree, InsertResult};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use aether_epoch::{Epoch, EpochClock};

#[cfg(feature = "wal")]
use crate::wal::{EdgeRecord, WalError, WalWriter};
#[cfg(feature = "wal")]
use std::path::Path;
#[cfg(feature = "wal")]
use std::sync::Mutex;

/// Dirty-node bitmap for historical embedding tracking.
///
/// One bit per vertex. Set on edge insert, cleared at epoch boundary.
/// Lock-free: writers set bits with `fetch_or`, readers load with Acquire.
struct DirtyBitmap {
    words: Vec<AtomicU64>,
    /// Number of vertices addressable. Caps the per-vertex bit index.
    num_vertices: usize,
}

impl DirtyBitmap {
    fn new(num_vertices: usize) -> Self {
        let num_words = num_vertices.div_ceil(64).max(1);
        let mut words = Vec::with_capacity(num_words);
        for _ in 0..num_words {
            words.push(AtomicU64::new(0));
        }
        Self {
            words,
            num_vertices,
        }
    }

    /// Mark a vertex as dirty. Lock-free (atomic fetch_or).
    /// Returns `false` if `vertex` is out of bounds (no bit was set).
    #[inline]
    fn mark(&self, vertex: u32) -> bool {
        let v = vertex as usize;
        if v >= self.num_vertices {
            return false;
        }
        let word = v / 64;
        let bit = v % 64;
        self.words[word].fetch_or(1u64 << bit, Ordering::Release);
        true
    }

    /// Check if a vertex is dirty. Returns `false` if `vertex` is out of bounds.
    #[inline]
    fn is_dirty(&self, vertex: u32) -> bool {
        let v = vertex as usize;
        if v >= self.num_vertices {
            return false;
        }
        let word = v / 64;
        let bit = v % 64;
        self.words[word].load(Ordering::Acquire) & (1u64 << bit) != 0
    }

    /// Clear all dirty bits (epoch boundary).
    ///
    /// Uses `swap(0, AcqRel)` per word so a `mark()` racing on the same word
    /// is not silently lost: the `fetch_or` either lands before our swap (its
    /// bit is included in the discarded value) or after (its bit survives in
    /// the cleared word). A plain `store(0)` would clobber the racing bit.
    fn clear_all(&self) {
        for w in &self.words {
            w.swap(0, Ordering::AcqRel);
        }
    }

    /// Collect all dirty node IDs and atomically clear the bitmap.
    /// Uses swap per word so concurrent writers don't lose marks.
    fn drain_dirty(&self) -> Vec<u32> {
        let mut result = Vec::new();
        for (i, w) in self.words.iter().enumerate() {
            let bits = w.swap(0, Ordering::AcqRel);
            if bits == 0 {
                continue;
            }
            let base = (i as u64) * 64;
            let mut remaining = bits;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as u64;
                let id = base + bit;
                // Mask off bits past num_vertices (last word can over-allocate).
                if (id as usize) < self.num_vertices {
                    result.push(id as u32);
                }
                remaining &= remaining - 1; // clear lowest set bit
            }
        }
        result
    }

    /// Count set bits across all words.
    fn count_dirty(&self) -> usize {
        self.words
            .iter()
            .map(|w| w.load(Ordering::Acquire).count_ones() as usize)
            .sum()
    }
}

/// Lock-free dynamic graph with O(1) edge insert and O(degree) neighbor access.
///
/// Each vertex's neighbor list is a C-tree (balanced tree of sorted chunks)
/// stored in a bump-allocating arena. Edge inserts create new tree nodes
/// via path copying; old trees remain valid for concurrent readers.
pub struct DynamicGraph {
    /// Per-vertex C-tree root offsets. Atomic for lock-free root swaps.
    /// NULL (u32::MAX) means no neighbors.
    roots: Vec<AtomicU32>,
    /// Arena for all C-tree nodes (chunks + interior nodes).
    arena: Arena,
    /// Number of vertices (fixed at construction).
    num_vertices: usize,
    /// Total edge count (informational, not used for correctness).
    num_edges: AtomicU64,
    /// Dirty-node bitmap for historical embedding tracking.
    /// Set on edge insert, cleared at epoch boundary.
    dirty: DirtyBitmap,
    /// Single-writer guard. `true` while a `Writer` exists.
    writer_locked: AtomicBool,
    /// Poison flag — set when a `Writer` is dropped during unwinding (a
    /// panic mid-insert). Once poisoned, the graph's internal invariants
    /// (per-vertex root pointers, dirty bitmap, num_edges counter, arena
    /// cursor) may all be in inconsistent states. New `writer()` calls
    /// return [`WriterError::Poisoned`]; reads (`degree`, `neighbors_into`,
    /// `has_edge`) keep working at their last consistent state.
    poisoned: AtomicBool,
    /// Shared monotonic version clock. Advanced once per successful writer
    /// drop (skipped on panic-poisoned drops). Readers pin the current
    /// epoch to coordinate consistent multi-source reads with subsystems
    /// like the feature store; today it's an opaque counter, but the same
    /// `Arc<EpochClock>` is the join point for future MVCC.
    epoch: Arc<EpochClock>,
    /// Optional write-ahead log. When present, every successful
    /// `Writer::insert_edge` appends a record; `Writer::drop` fsyncs.
    /// The `Mutex` is uncontended in practice (the surrounding `Writer`
    /// guard already enforces single-writer), but we still need interior
    /// mutability since the inserts go through `&self`.
    #[cfg(feature = "wal")]
    wal: Option<Mutex<WalWriter>>,
}

impl DynamicGraph {
    /// Create a graph with `num_vertices` vertices and no edges.
    ///
    /// `arena_bytes` controls the arena capacity, at most
    /// [`Arena::MAX_CAPACITY`] (2 GiB — offsets are u32 with a tag bit).
    /// Each insert path-copies its tree path (~100–250 bytes including
    /// superseded nodes), so arena consumption tracks total inserts, not
    /// live edges. Long-running ingest should call [`compact`] when
    /// usage approaches capacity; compacted storage is ~6–10 bytes per
    /// live edge.
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
        Self {
            roots,
            arena: Arena::new(arena_bytes),
            num_vertices,
            num_edges: AtomicU64::new(0),
            dirty: DirtyBitmap::new(num_vertices),
            writer_locked: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            epoch,
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
    /// After this returns, `current_epoch()` equals the number of
    /// replayed records — replay advances the clock once per record,
    /// while a live run advances once per writer-guard drop. The two
    /// agree only when every guard inserted exactly one edge, so do not
    /// compare epoch values across a restart; treat the post-recovery
    /// epoch as a fresh, deterministic starting point.
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
        // One writer-guard per record: cheap (a single CAS to acquire,
        // a single CAS to release, plus an epoch advance). The
        // per-record advance is exactly what we want — after replay
        // `current_epoch()` equals the number of records recovered,
        // giving callers a deterministic "where did we end up" signal.
        let mut graph = Self::new_with_epoch(num_vertices, arena_bytes, Arc::clone(&epoch));

        // The WAL does not record the graph dimensions it was written
        // against, so a mismatched `num_vertices` or an undersized arena
        // surfaces here, record by record. Both must abort recovery: a
        // silently partial replay would present itself as a smaller but
        // valid graph.
        let mut replay_err: Option<WalError> = None;
        let outcome = crate::wal::replay(path, |rec| {
            if replay_err.is_some() {
                return;
            }
            let mut w = graph.writer_or_panic();
            match w.insert_edge(rec.src, rec.dst) {
                Ok(_) => {}
                Err(InsertError::VertexOutOfRange { src, dst }) => {
                    replay_err = Some(WalError::RecordOutOfRange {
                        src,
                        dst,
                        num_vertices: num_vertices as u64,
                    });
                }
                Err(InsertError::ArenaFull) => {
                    replay_err = Some(WalError::ReplayArenaFull);
                }
            }
            let _ = rec.epoch; // reserved for future MVCC pinning
        })?;
        if let Some(e) = replay_err {
            return Err(e);
        }

        // If the WAL ended in a torn record, truncate so future appends
        // sit on top of clean data.
        if let Some(off) = outcome.truncate_to {
            std::fs::OpenOptions::new()
                .write(true)
                .open(path)
                .and_then(|f| f.set_len(off))
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

    /// Acquire the single-writer guard.
    ///
    /// Holding a `Writer` is the only way to insert edges. The guard releases
    /// on drop, allowing another acquirer.
    ///
    /// # Errors
    /// - [`WriterError::Busy`] if another `Writer` is currently held.
    /// - [`WriterError::Poisoned`] if a previous writer was dropped during
    ///   a panic. Once poisoned, the graph is read-only forever — the
    ///   internal arena cursor / per-vertex roots / dirty bitmap may all
    ///   be in mutually-inconsistent states. Recovery requires destroying
    ///   the graph and rebuilding from a checkpoint (see the WAL story).
    pub fn writer(&self) -> Result<Writer<'_>, WriterError> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(WriterError::Poisoned);
        }
        self.writer_locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|_| WriterError::Busy)?;
        // Re-check poison: a previous Writer may have set the flag between
        // our first check and our CAS. Without this we would briefly hold
        // `writer_locked=true` on a poisoned graph; we release immediately.
        if self.poisoned.load(Ordering::Acquire) {
            self.writer_locked.store(false, Ordering::Release);
            return Err(WriterError::Poisoned);
        }
        Ok(Writer {
            graph: self,
            rebalance_scratch: Vec::new(),
            #[cfg(feature = "wal")]
            wal_failed: false,
        })
    }

    /// Acquire the writer guard, panicking on any error.
    ///
    /// Convenience for tests and bulk-load paths that should not contend
    /// AND don't worry about poisoning. Production callers should match
    /// on [`Self::writer`].
    pub fn writer_or_panic(&self) -> Writer<'_> {
        match self.writer() {
            Ok(w) => w,
            Err(e) => panic!("DynamicGraph::writer failed: {e}"),
        }
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

    /// Total edge count.
    #[inline(always)]
    pub fn num_edges(&self) -> u64 {
        self.num_edges.load(Ordering::Relaxed)
    }

    /// Get the C-tree root for a vertex.
    #[inline(always)]
    fn tree_for(&self, vertex: u32) -> CTree {
        CTree {
            root: self.roots[vertex as usize].load(Ordering::Acquire),
        }
    }

    /// Degree of a vertex.
    #[inline]
    pub fn degree(&self, vertex: u32) -> usize {
        self.tree_for(vertex).count(&self.arena)
    }

    /// Iterate a vertex's neighbors in sorted order. Calls `f` for each chunk.
    ///
    /// Zero allocations. The chunks are read directly from the arena.
    /// Safe to call concurrently with edge inserts — readers see a
    /// consistent snapshot (the tree that was current when they read the root).
    #[inline]
    pub fn for_each_chunk(&self, vertex: u32, f: impl FnMut(&Chunk)) {
        self.tree_for(vertex).for_each_chunk(&self.arena, f);
    }

    /// Collect all neighbors of `vertex` into `buf` (sorted).
    ///
    /// Clears `buf` first. For GNN sampling, use this to get a flat
    /// slice suitable for Floyd's algorithm.
    #[inline]
    pub fn neighbors_into(&self, vertex: u32, buf: &mut Vec<u32>) {
        buf.clear();
        self.tree_for(vertex).collect_into(&self.arena, buf);
    }

    /// Check if edge (src → dst) exists.
    #[inline]
    pub fn has_edge(&self, src: u32, dst: u32) -> bool {
        self.tree_for(src).contains(&self.arena, dst)
    }

    /// Arena usage stats.
    pub fn arena_used(&self) -> usize {
        self.arena.used()
    }

    pub fn arena_capacity(&self) -> usize {
        self.arena.capacity()
    }

    /// Rebuild every vertex's C-tree into a fresh arena, reclaiming the
    /// garbage that path-copying inserts leave behind.
    ///
    /// Arena consumption grows with every insert (superseded nodes are
    /// never reused), so a long-running ingest eventually fills the
    /// arena. Compacting rebuilds each adjacency list perfectly balanced
    /// from a CSR snapshot — afterwards usage is proportional to live
    /// edges (~6–10 bytes each). Use [`compact_with_capacity`] to grow
    /// (or shrink) the arena at the same time.
    ///
    /// `&mut self` guarantees no concurrent readers or writers, so the
    /// old arena can be dropped safely. Peak transient memory is the new
    /// arena plus the CSR snapshot (~12 bytes per live edge): the old
    /// arena is freed only after the rebuild succeeds, which also means
    /// a failed compaction leaves the graph untouched.
    ///
    /// [`compact_with_capacity`]: Self::compact_with_capacity
    pub fn compact(&mut self) -> Result<(), CompactError> {
        self.compact_with_capacity(self.arena.capacity())
    }

    /// Like [`compact`](Self::compact), but the rebuilt trees go into an
    /// arena of `new_capacity` bytes. The escape hatch for recovering
    /// from [`ArenaFull`]: compact into a larger arena and keep
    /// ingesting.
    ///
    /// # Panics
    /// Panics if `new_capacity` is 0 or exceeds [`Arena::MAX_CAPACITY`]
    /// (the [`Arena::new`] contract).
    pub fn compact_with_capacity(&mut self, new_capacity: usize) -> Result<(), CompactError> {
        if self.is_poisoned() {
            return Err(CompactError::Poisoned);
        }

        let (offsets, edges) = self.snapshot_csr();
        let new_arena = Arena::new(new_capacity);

        let mut new_roots = Vec::with_capacity(self.num_vertices);
        for v in 0..self.num_vertices {
            let start = offsets[v] as usize;
            let end = offsets[v + 1] as usize;
            // SAFETY: `&mut self` makes this thread the only one touching
            // the new arena.
            let tree = unsafe { CTree::from_sorted(&new_arena, &edges[start..end]) }
                .ok_or(CompactError::ArenaFull)?;
            new_roots.push(AtomicU32::new(tree.root));
        }

        // Rebuild succeeded in full — only now replace the live state.
        self.arena = new_arena;
        self.roots = new_roots;
        Ok(())
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
        let mut edges = Vec::new();

        offsets.push(0);
        // Iterate via usize to avoid u32 truncation when num_vertices is large.
        // Each vertex's neighbors are appended straight onto `edges`; the
        // running length after each append is the next CSR offset.
        for v in 0..self.num_vertices {
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
                }
            }
        }
        graph
    }
}

/// Single-writer guard for [`DynamicGraph`].
///
/// Created by [`DynamicGraph::writer`]. Releases the writer slot on drop.
/// Only one `Writer` may exist at a time — this is enforced at runtime by
/// a CAS-based flag and is the primary safety invariant of the crate.
pub struct Writer<'a> {
    graph: &'a DynamicGraph,
    /// Scratch buffer reused across inserts to hold a scapegoat subtree's
    /// elements during a rebalance. Owned by the guard so the write path
    /// allocates at most once per writer instead of once per rebalance.
    rebalance_scratch: Vec<u32>,
    /// True if any WAL append failed during this writer's lifetime. The
    /// drop path consults this to poison the graph rather than advance the
    /// epoch on data that isn't durable.
    #[cfg(feature = "wal")]
    wal_failed: bool,
}

impl<'a> Writer<'a> {
    /// Insert a directed edge from `src` to `dst`.
    ///
    /// Returns `Ok(true)` if the edge was new, `Ok(false)` if it already
    /// existed (no allocation occurred). Errors distinguish a full arena
    /// (compact or grow, then retry) from an out-of-range vertex (the
    /// edge is invalid for this graph and was not inserted).
    pub fn insert_edge(&mut self, src: u32, dst: u32) -> Result<bool, InsertError> {
        if (src as usize) >= self.graph.num_vertices || (dst as usize) >= self.graph.num_vertices {
            return Err(InsertError::VertexOutOfRange { src, dst });
        }

        let current_root = self.graph.roots[src as usize].load(Ordering::Acquire);
        let tree = CTree { root: current_root };

        // SAFETY: Writer existence guarantees single-writer access to the arena.
        match unsafe {
            tree.insert_with_scratch(&self.graph.arena, dst, &mut self.rebalance_scratch)
        } {
            InsertResult::Inserted(new_tree) => {
                // Release ordering ensures all arena writes (new nodes) are
                // visible before the root pointer becomes visible to readers.
                // The edge is observable to readers from here on; the
                // counter and dirty bits are published only after the WAL
                // append below succeeds so they never count an insert that
                // recovery will discard.
                self.graph.roots[src as usize].store(new_tree.root, Ordering::Release);

                // WAL append (buffered; fsync happens in `Writer::drop`).
                // Errors here mean the in-memory state is ahead of the log
                // — we record the error so `Writer::drop` can poison the
                // graph instead of silently bumping the epoch on lossy
                // state.
                #[cfg(feature = "wal")]
                if let Some(wal_mu) = self.graph.wal.as_ref() {
                    let rec = EdgeRecord {
                        epoch: self.graph.epoch.current().as_u64(),
                        src,
                        dst,
                    };
                    // The Writer guard already enforces single-writer, so
                    // the mutex is uncontended. Take the guard even if a
                    // prior holder poisoned it — the buffered writer behind
                    // the lock is still usable, and panicking here (in a
                    // path that may run during unwinding) would abort.
                    let mut w = wal_mu.lock().unwrap_or_else(|e| e.into_inner());
                    if let Err(e) = w.append_edge(rec) {
                        // Record the failure so `Writer::drop` can poison.
                        self.wal_failed = true;
                        tracing::error!(error = %e, "WAL append failed");
                    }
                }

                // Bookkeeping runs only when the edge is durable-pending:
                // no WAL attached, or the append above (and every earlier
                // append in this guard) succeeded. A failed append leaves
                // the log torn, so the guard's edges are dropped on
                // recovery — counting them here would over-report.
                #[cfg(feature = "wal")]
                let durable = !self.wal_failed;
                #[cfg(not(feature = "wal"))]
                let durable = true;
                if durable {
                    self.graph.num_edges.fetch_add(1, Ordering::Relaxed);
                    // Out-of-range src/dst are already rejected by the bounds
                    // check at the top of `insert_edge`, so both marks succeed.
                    let _ = self.graph.dirty.mark(src);
                    let _ = self.graph.dirty.mark(dst);
                }

                Ok(true)
            }
            InsertResult::Duplicate => Ok(false),
            InsertResult::ArenaFull => Err(InsertError::ArenaFull),
        }
    }
}

impl Drop for Writer<'_> {
    fn drop(&mut self) {
        // If we're unwinding from a panic mid-insert, the graph's invariants
        // may be partially updated: arena cursor advanced past a node that
        // was never linked into the tree, a root pointer that points at a
        // freshly-allocated but uninitialized chunk, num_edges incremented
        // for an insert that aborted, etc. Poison the graph so no further
        // writer can run; readers keep working at their last consistent
        // state.
        let panicking = std::thread::panicking();
        if panicking {
            self.graph.poisoned.store(true, Ordering::Release);
            tracing::warn!("DynamicGraph writer panicked — graph poisoned");
            self.graph.writer_locked.store(false, Ordering::Release);
            return;
        }

        // Clean drop path. The order matters: WAL sync FIRST, then advance
        // the epoch only if sync succeeded. If sync fails, we've got
        // in-memory edges that aren't durable — poison instead.
        #[cfg(feature = "wal")]
        let mut durable_failure = self.wal_failed;
        #[cfg(not(feature = "wal"))]
        let durable_failure = false;

        #[cfg(feature = "wal")]
        if let Some(wal_mu) = self.graph.wal.as_ref() {
            // Take the guard even if poisoned — the buffered writer behind
            // it is still usable, and a panic in `Drop` during unwinding
            // would abort the process.
            let mut w = wal_mu.lock().unwrap_or_else(|e| e.into_inner());
            if let Err(e) = w.sync() {
                tracing::error!(error = %e, "WAL fsync failed; poisoning graph");
                durable_failure = true;
            }
        }

        if durable_failure {
            self.graph.poisoned.store(true, Ordering::Release);
        } else {
            // Publish a new epoch so readers pinning the clock see this
            // writer's edits. Done before releasing the writer lock so
            // the next writer can't bump the clock first.
            let new_epoch = self.graph.epoch.advance().as_u64();
            tracing::trace!(epoch = new_epoch, "DynamicGraph writer committed");
        }
        self.graph.writer_locked.store(false, Ordering::Release);
    }
}

/// Reason [`DynamicGraph::writer`] cannot hand out a writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriterError {
    /// Another `Writer` is currently held by some thread.
    Busy,
    /// A prior `Writer` was dropped during a panic; the graph's internal
    /// state may be inconsistent and no further writes are allowed.
    Poisoned,
}

impl std::fmt::Display for WriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy => f.write_str("DynamicGraph already has a writer"),
            Self::Poisoned => f.write_str(
                "DynamicGraph is poisoned (a previous writer panicked); rebuild from a checkpoint",
            ),
        }
    }
}

impl std::error::Error for WriterError {}

/// Reason [`DynamicGraph::compact`] refused to run or failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactError {
    /// The graph is poisoned (a previous writer panicked); its state may
    /// be inconsistent, so compacting it would persist the damage.
    Poisoned,
    /// The rebuilt trees did not fit in the requested capacity. The
    /// graph is unchanged; retry with a larger capacity.
    ArenaFull,
}

impl std::fmt::Display for CompactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Poisoned => f.write_str("cannot compact a poisoned graph"),
            Self::ArenaFull => f.write_str("compacted trees exceed the requested arena capacity"),
        }
    }
}

impl std::error::Error for CompactError {}

/// Error from [`Writer::insert_edge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertError {
    /// The arena has no remaining capacity. Recover with
    /// [`DynamicGraph::compact`] / [`DynamicGraph::compact_with_capacity`].
    ArenaFull,
    /// `src` or `dst` is not a vertex of this graph
    /// (≥ `num_vertices`). The edge was not inserted.
    VertexOutOfRange { src: u32, dst: u32 },
}

impl std::fmt::Display for InsertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArenaFull => write!(f, "C-tree arena is full"),
            Self::VertexOutOfRange { src, dst } => {
                write!(f, "edge ({src}, {dst}) references a vertex out of range")
            }
        }
    }
}

impl std::error::Error for InsertError {}

#[cfg(test)]
mod tests {
    use super::*;

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
        dirty.sort();
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
