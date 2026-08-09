//! The single-writer guard: edge inserts, WAL appends, retirement.

use crate::arena::{ArenaWriter, RecycleStats, RetireLog};
use crate::ctree::{CTree, InsertResult};
use crate::graph::DynamicGraph;
use std::sync::atomic::Ordering;

#[cfg(feature = "wal")]
use crate::wal::{EdgeRecord, WalWriter};

impl DynamicGraph {
    /// Acquire the single-writer guard.
    ///
    /// Holding a `Writer` is the only way to insert edges. The guard releases
    /// on drop, allowing another acquirer.
    ///
    /// # Errors
    /// - [`WriterError::Busy`] if another `Writer` is currently held.
    /// - [`WriterError::Poisoned`] if a previous writer was dropped during
    ///   a panic or hit a WAL failure. Once poisoned, the graph is
    ///   read-only forever — bookkeeping (num_edges, dirty bitmap, WAL)
    ///   may be out of step with the published roots, though reads stay
    ///   consistent. Recovery requires destroying the graph and
    ///   rebuilding from a checkpoint (see the WAL story).
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
            // SAFETY: the CAS above admitted exactly one `Writer`, which
            // owns this handle for its lifetime; `compact*` takes
            // `&mut self` and so cannot overlap a live guard.
            arena: unsafe { self.arena.writer() },
            rebalance_scratch: Vec::new(),
            merge_scratch: Vec::new(),
            new_dsts: Vec::new(),
            pending_edges: 0,
            dirty_buf: Vec::with_capacity(DIRTY_BUF_FLUSH_THRESHOLD),
            retire_log: RetireLog::new(),
            #[cfg(feature = "wal")]
            wal_guard: self
                .wal
                .as_ref()
                .map(|m| m.lock().unwrap_or_else(|e| e.into_inner())),
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
}

/// Once the guard's dirty buffer reaches this many entries it is flushed
/// to the bitmap eagerly, keeping the buffer at a fixed capacity so the
/// per-edge insert path never grows it (heap-free steady state, enforced
/// by `tests/zero_alloc.rs`).
const DIRTY_BUF_FLUSH_THRESHOLD: usize = 8192;

/// Single-writer guard for [`DynamicGraph`].
///
/// Created by [`DynamicGraph::writer`]. Releases the writer slot on drop.
/// Only one `Writer` may exist at a time — this is enforced at runtime by
/// a CAS-based flag and is the primary safety invariant of the crate.
pub struct Writer<'a> {
    graph: &'a DynamicGraph,
    /// The arena's write handle. Its construction in
    /// [`DynamicGraph::writer`] is the one point where the single-writer
    /// invariant is proven; allocation through it is safe code.
    arena: ArenaWriter<'a>,
    /// Scratch buffer reused across inserts to hold a scapegoat subtree's
    /// elements during a rebalance. Owned by the guard so the write path
    /// allocates at most once per writer instead of once per rebalance.
    rebalance_scratch: Vec<u32>,
    /// Scratch for the bulk insert path's merged neighbor list.
    merge_scratch: Vec<u32>,
    /// Scratch for the bulk insert path's genuinely-new destinations —
    /// the WAL records and dirty marks a batch produces.
    new_dsts: Vec<u32>,
    /// Edges inserted by this guard, folded into the shared counter once
    /// at drop — a per-edge atomic RMW on a shared line would buy nothing
    /// from a provably single writer.
    pending_edges: u64,
    /// Vertices touched by this guard's inserts. Sorted, deduplicated, and
    /// flushed to the dirty bitmap in one sequential pass at commit —
    /// marking per edge would cost two random-line atomic RMWs per insert
    /// (each a likely DRAM + TLB miss on a multi-hundred-MB bitmap), and
    /// consumers only drain dirtiness at epoch boundaries anyway.
    dirty_buf: Vec<u32>,
    /// Slots superseded by this guard's inserts, awaiting their gate
    /// stamp. Stamped (handed to the arena's grace ring) at the fixed
    /// watermark and at commit — always after the root stores that made
    /// the slots unreachable, which is what makes the stamp's grace
    /// reasoning sound.
    retire_log: RetireLog,
    /// The WAL, locked once for the guard's lifetime — the guard already
    /// enforces single-writer, so per-edge lock traffic would be pure tax.
    #[cfg(feature = "wal")]
    wal_guard: Option<std::sync::MutexGuard<'a, WalWriter>>,
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
    /// edge is invalid for this graph and was not inserted). With a WAL
    /// attached, the record is appended before the edge becomes
    /// reader-visible; a failed append returns `InsertError::WalAppend`
    /// and publishes nothing.
    pub fn insert_edge(&mut self, src: u32, dst: u32) -> Result<bool, InsertError> {
        if (src as usize) >= self.graph.num_vertices || (dst as usize) >= self.graph.num_vertices {
            return Err(InsertError::VertexOutOfRange { src, dst });
        }
        match self.insert_edge_inner(src, dst) {
            Err(InsertError::ArenaFull) => {
                // Both cursors are spent, but slots superseded by earlier
                // (already published) inserts may be waiting in the log.
                // Stamp them so the exhausted allocator can reclaim any
                // grace-cleared batch, then retry once.
                // SAFETY: everything in the log was unpublished by a
                // prior root store.
                unsafe { self.arena.retire(&mut self.retire_log) };
                self.insert_edge_inner(src, dst)
            }
            other => other,
        }
    }

    fn insert_edge_inner(&mut self, src: u32, dst: u32) -> Result<bool, InsertError> {
        // Relaxed: this thread is the only root-storer (single-writer
        // guard), so it reads back its own prior store; the initial state
        // was published by the guard-acquisition synchronization.
        let current_root = self.graph.roots[src as usize].load(Ordering::Relaxed);
        let tree = CTree { root: current_root };

        // Marks for rolling back retire entries if the WAL append fails
        // below: a failed append leaves the old tree live, so its nodes
        // must not stay logged for reuse.
        let chunks_mark = self.retire_log.chunks.len();
        let interiors_mark = self.retire_log.interiors.len();

        match tree.insert_with_scratch(
            &mut self.arena,
            dst,
            &mut self.rebalance_scratch,
            &mut self.retire_log,
        ) {
            InsertResult::Inserted(new_tree) => {
                // WAL append first (buffered; fsync happens in
                // `Writer::drop`). The freshly allocated tree nodes are
                // invisible to readers until the root store below, so a
                // failed append leaves readers on the old root — the edge
                // exists in neither memory nor the log. The arena bytes
                // allocated for the failed insert leak until `compact`.
                #[cfg(feature = "wal")]
                if let Some(w) = self.wal_guard.as_mut() {
                    let rec = EdgeRecord { src, dst };
                    if let Err(e) = w.append_edge(rec) {
                        // Record the failure so `Writer::drop` can poison,
                        // and un-log the still-live old nodes.
                        self.retire_log.chunks.truncate(chunks_mark);
                        self.retire_log.interiors.truncate(interiors_mark);
                        self.wal_failed = true;
                        tracing::error!(error = %e, "WAL append failed");
                        return Err(InsertError::WalAppend);
                    }
                }
                #[cfg(not(feature = "wal"))]
                let _ = (chunks_mark, interiors_mark);

                // Release ordering ensures all arena writes (new nodes) are
                // visible before the root pointer becomes visible to
                // readers. Counting and dirty-marking are buffered on the
                // guard and folded in at commit (or at the buffer's fixed
                // watermark).
                self.graph.roots[src as usize].store(new_tree.root, Ordering::Release);
                self.pending_edges += 1;
                self.note_dirty(src);
                self.note_dirty(dst);
                self.maybe_stamp_retired();

                Ok(true)
            }
            InsertResult::Duplicate => Ok(false),
            InsertResult::ArenaFull => Err(InsertError::ArenaFull),
        }
    }

    /// Stamp the retire log at its fixed watermark. Called only after a
    /// root store, so every logged slot is already unreachable.
    #[inline]
    fn maybe_stamp_retired(&mut self) {
        if self.retire_log.wants_flush() {
            // SAFETY: called only after the root stores that unpublished
            // every logged slot.
            unsafe { self.arena.retire(&mut self.retire_log) };
        }
    }

    /// Insert every edge `(src, dst)` for `dst` in an ascending,
    /// deduplicated `dsts` slice, rebuilding `src`'s tree once.
    ///
    /// Ingest streams are heavily source-clustered; inserting D edges one
    /// at a time path-copies the root-to-leaf path D times (O(D log D)
    /// allocations and garbage). This merges the existing neighbors with
    /// `dsts` and builds the new tree in one pass — O(degree + D) arena
    /// bytes, one root publish.
    ///
    /// Returns the number of edges actually new (duplicates are skipped).
    ///
    /// # Panics
    /// Debug-asserts that `dsts` is strictly ascending.
    pub fn insert_edges_sorted(&mut self, src: u32, dsts: &[u32]) -> Result<u64, InsertError> {
        debug_assert!(
            dsts.windows(2).all(|w| w[0] < w[1]),
            "insert_edges_sorted requires strictly ascending dsts"
        );
        if (src as usize) >= self.graph.num_vertices {
            return Err(InsertError::VertexOutOfRange { src, dst: 0 });
        }
        if let Some(&bad) = dsts
            .iter()
            .find(|&&d| (d as usize) >= self.graph.num_vertices)
        {
            return Err(InsertError::VertexOutOfRange { src, dst: bad });
        }
        if dsts.is_empty() {
            return Ok(0);
        }

        // Relaxed: see `insert_edge`.
        let current_root = self.graph.roots[src as usize].load(Ordering::Relaxed);
        let tree = CTree { root: current_root };

        // Merge existing (sorted) neighbors with the new sorted dsts,
        // recording the genuinely-new dsts in their own scratch — the
        // batch's WAL records and dirty marks. A failed batch simply
        // leaves the scratch to be cleared by the next call; nothing is
        // rolled back.
        self.new_dsts.clear();
        let existing = &mut self.rebalance_scratch;
        existing.clear();
        tree.collect_into(&self.graph.arena, existing);
        let merged = &mut self.merge_scratch;
        merged.clear();
        merged.reserve(existing.len() + dsts.len());
        let (mut i, mut j) = (0usize, 0usize);
        while i < existing.len() && j < dsts.len() {
            match existing[i].cmp(&dsts[j]) {
                std::cmp::Ordering::Less => {
                    merged.push(existing[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    merged.push(dsts[j]);
                    self.new_dsts.push(dsts[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    merged.push(existing[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        merged.extend_from_slice(&existing[i..]);
        for &d in &dsts[j..] {
            merged.push(d);
            self.new_dsts.push(d);
        }
        let new_count = self.new_dsts.len() as u64;
        if new_count == 0 {
            return Ok(0);
        }

        // Build the replacement tree first (invisible to readers), then
        // log, then publish — a WAL failure leaves readers on the old
        // root with no record of the unpublished edges.
        let mut built = CTree::from_sorted(&mut self.arena, &self.merge_scratch);
        if built.is_none() {
            // Stamp already-unpublished retirements so the exhausted
            // allocator can reclaim grace-cleared slots, then retry once
            // (see `insert_edge`).
            // SAFETY: everything in the log was unpublished by a prior
            // root store.
            unsafe { self.arena.retire(&mut self.retire_log) };
            built = CTree::from_sorted(&mut self.arena, &self.merge_scratch);
        }
        let Some(new_tree) = built else {
            return Err(InsertError::ArenaFull);
        };

        #[cfg(feature = "wal")]
        if let Some(w) = self.wal_guard.as_mut() {
            for &dst in &self.new_dsts {
                if let Err(e) = w.append_edge(EdgeRecord { src, dst }) {
                    self.wal_failed = true;
                    tracing::error!(error = %e, "WAL append failed");
                    return Err(InsertError::WalAppend);
                }
            }
        }

        self.graph.roots[src as usize].store(new_tree.root, Ordering::Release);
        // The old tree is superseded in full now that the merged rebuild
        // is published; log every one of its nodes for recycling.
        if current_root != crate::ctree::NULL {
            // SAFETY: the old tree is unreachable from the just-published
            // root, and none of its slots were logged before.
            unsafe {
                crate::ctree::retire_subtree(self.arena.arena(), current_root, &mut self.retire_log)
            };
        }
        self.pending_edges += new_count;
        for i in 0..self.new_dsts.len() {
            let d = self.new_dsts[i];
            self.note_dirty(d);
        }
        self.note_dirty(src);
        self.maybe_stamp_retired();

        Ok(new_count)
    }

    /// Record a vertex as dirtied by this guard, flushing at the fixed
    /// watermark so the buffer never grows on the insert path.
    #[inline]
    fn note_dirty(&mut self, v: u32) {
        if self.dirty_buf.len() == DIRTY_BUF_FLUSH_THRESHOLD {
            self.flush_dirty();
        }
        self.dirty_buf.push(v);
    }

    /// Sort, dedup, and fold the buffered dirty vertices into the bitmap
    /// in one sequential pass, keeping the buffer's capacity.
    fn flush_dirty(&mut self) {
        if self.dirty_buf.is_empty() {
            return;
        }
        self.dirty_buf.sort_unstable();
        self.dirty_buf.dedup();
        self.graph.dirty.mark_sorted(&self.dirty_buf);
        self.dirty_buf.clear();
    }

    /// Fold this guard's buffered bookkeeping into the shared state: one
    /// atomic add for the edge count, one sorted sequential pass over the
    /// dirty bitmap.
    fn flush_bookkeeping(&mut self) {
        if self.pending_edges > 0 {
            self.graph
                .num_edges
                .0
                .fetch_add(self.pending_edges, Ordering::Relaxed);
            self.pending_edges = 0;
        }
        self.flush_dirty();
    }

    /// Arena recycling counters. Slots this guard has logged but not yet
    /// stamped count as neither free nor pending.
    pub fn recycle_stats(&self) -> RecycleStats {
        self.arena.recycle_stats()
    }
}

impl Drop for Writer<'_> {
    fn drop(&mut self) {
        // If we're unwinding from a panic mid-insert, the graph's
        // bookkeeping may be partially updated: arena cursor advanced past
        // a node that was never linked into the tree, an edge published
        // but not yet counted or dirty-marked, etc. Published roots always
        // point at fully-built trees, so reads stay consistent — but no
        // further writer can be trusted. Poison the graph, and discard the
        // guard's buffered WAL records so they are not flushed later:
        // the guard never committed, so its records must not become
        // durable. Records the BufWriter already spilled to the OS are
        // out of reach — the discard is best-effort (see the WAL
        // durability contract).
        let panicking = std::thread::panicking();
        if panicking {
            self.graph.poisoned.store(true, Ordering::Release);
            #[cfg(feature = "wal")]
            if let Some(w) = self.wal_guard.as_mut()
                && let Err(e) = w.discard_pending()
            {
                tracing::error!(error = %e, "failed to discard pending WAL records");
            }
            tracing::warn!("DynamicGraph writer panicked — graph poisoned");
            self.graph.writer_locked.store(false, Ordering::Release);
            return;
        }

        // Clean drop path. The order matters: WAL sync FIRST, then fold in
        // this guard's buffered bookkeeping, then advance the epoch — so
        // epoch observers (historical-embedding drains) see the committed
        // dirty set and edge count. If sync fails, we've got in-memory
        // edges that aren't durable — poison instead, discarding the
        // buffered bookkeeping (it describes uncommitted state).
        #[cfg(feature = "wal")]
        let mut durable_failure = self.wal_failed;
        #[cfg(not(feature = "wal"))]
        let durable_failure = false;

        #[cfg(feature = "wal")]
        if let Some(w) = self.wal_guard.as_mut()
            && let Err(e) = w.sync()
        {
            tracing::error!(error = %e, "WAL fsync failed; poisoning graph");
            durable_failure = true;
        }

        if durable_failure {
            // The un-stamped retire log dies with the guard: its slots are
            // never reused, which is exactly right — some of them may
            // belong to trees that are still the published state.
            self.graph.poisoned.store(true, Ordering::Release);
        } else {
            self.flush_bookkeeping();
            // Stamp this guard's remaining retirements (all root stores
            // are done) and fold in any batches whose grace has passed,
            // so a subsequent guard starts with a warm free list.
            // SAFETY: every logged slot was unpublished by its root store.
            unsafe { self.arena.retire(&mut self.retire_log) };
            self.arena.reclaim();
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

/// Error from [`Writer::insert_edge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertError {
    /// The arena has no remaining capacity. Recover with
    /// [`DynamicGraph::compact`] / [`DynamicGraph::compact_with_capacity`].
    ArenaFull,
    /// `src` or `dst` is not a vertex of this graph
    /// (≥ `num_vertices`). The edge was not inserted.
    VertexOutOfRange { src: u32, dst: u32 },
    /// The WAL append for this edge failed. The edge was not published —
    /// readers never see it and it carries no log record — but the guard
    /// is marked failed, so `Writer::drop` poisons the graph. The
    /// underlying I/O error is logged via `tracing`.
    #[cfg(feature = "wal")]
    WalAppend,
}

impl std::fmt::Display for InsertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArenaFull => write!(f, "C-tree arena is full"),
            Self::VertexOutOfRange { src, dst } => {
                write!(f, "edge ({src}, {dst}) references a vertex out of range")
            }
            #[cfg(feature = "wal")]
            Self::WalAppend => write!(f, "WAL append failed; edge not inserted"),
        }
    }
}

impl std::error::Error for InsertError {}
