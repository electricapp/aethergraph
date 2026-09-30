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
            retire_stamp: self.epoch.current().as_u64() + 1,
            scratch: std::mem::take(
                &mut *self
                    .writer_scratch
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            ),
            pending_edges: 0,
            touched_dedup_limit: TOUCHED_LEAVES_DEDUP_START,
            #[cfg(feature = "wal")]
            wal_guard: self
                .wal
                .as_ref()
                .map(|m| m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)),
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

/// First in-place sort+dedup of `touched_leaves`; doubles after each pass
/// so scattered writers pay amortized O(log) sorts, not one per push.
const TOUCHED_LEAVES_DEDUP_START: usize = 1024;

/// Scratch vectors above this many elements are shrunk back when parked,
/// so one huge batch doesn't pin its buffers for the graph's lifetime.
const SCRATCH_KEEP: usize = 1 << 16;

/// Buffers a writer guard reuses across its inserts, parked on the graph
/// between guards so a short guard (one edge from Python) allocates
/// nothing.
#[derive(Default)]
pub(crate) struct WriterScratch {
    /// A scapegoat subtree's elements during a rebalance, or the existing
    /// neighbors during a bulk merge.
    rebalance: Vec<u32>,
    /// The bulk insert path's merged neighbor list.
    merge: Vec<u32>,
    /// The bulk insert path's genuinely-new destinations — the WAL
    /// records and dirty marks a batch produces.
    new_dsts: Vec<u32>,
    /// One source's destinations while [`Writer::insert_edges`] groups a
    /// batch.
    batch_dsts: Vec<u32>,
    /// Vertices touched by this guard's inserts. Sorted, deduplicated, and
    /// flushed to the dirty bitmap in one sequential pass at commit —
    /// marking per edge would cost two random-line atomic RMWs per insert
    /// (each a likely DRAM + TLB miss on a multi-hundred-MB bitmap), and
    /// consumers only drain dirtiness at epoch boundaries anyway.
    dirty: Vec<u32>,
    /// Root-table leaves whose roots this guard stored — the leaves the
    /// commit snapshot must copy. Clustered-deduped on push (skip if same
    /// as last), fully deduped at `touched_dedup_limit` and at commit.
    touched_leaves: Vec<u32>,
    /// Slots superseded by this guard's inserts. Stamped and queued at
    /// the fixed watermark and at commit — always after the root stores
    /// that made the slots unreachable.
    retire_log: RetireLog,
}

impl WriterScratch {
    /// Pre-sized so steady-state inserts never grow a buffer.
    pub(crate) fn new() -> Self {
        Self {
            dirty: Vec::with_capacity(DIRTY_BUF_FLUSH_THRESHOLD),
            retire_log: RetireLog::new(),
            ..Self::default()
        }
    }

    /// Reset for the next guard. The retire log must be empty: a slot
    /// left in it would be retired by a guard that never unpublished it.
    fn park(&mut self) {
        self.retire_log.chunks.clear();
        self.retire_log.interiors.clear();
        self.dirty.clear();
        self.touched_leaves.clear();
        for v in [
            &mut self.rebalance,
            &mut self.merge,
            &mut self.new_dsts,
            &mut self.batch_dsts,
        ] {
            v.clear();
            v.shrink_to(SCRATCH_KEEP);
        }
    }
}

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
    /// Stamp for retirements before commit: above every pre-guard
    /// snapshot's epoch, and at most the epoch the commit's `advance`
    /// returns (other subsystems sharing the clock only move it forward).
    retire_stamp: u64,
    /// Buffers borrowed from the graph for the guard's lifetime.
    scratch: WriterScratch,
    /// Edges inserted by this guard, folded into the shared counter once
    /// at drop — a per-edge atomic RMW on a shared line would buy nothing
    /// from a provably single writer.
    pending_edges: u64,
    touched_dedup_limit: usize,
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
        self.check_wal()?;
        match self.insert_edge_inner(src, dst) {
            Err(InsertError::ArenaFull) => {
                // Both cursors are spent, but slots superseded by earlier
                // (already published) inserts may be waiting in the log.
                // Stamp them so the exhausted allocator can reclaim any
                // grace-cleared batch, then retry once.
                // SAFETY: everything in the log was unpublished by a
                // prior root store.
                unsafe {
                    self.arena
                        .retire(&mut self.scratch.retire_log, self.retire_stamp)
                };
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
        let chunks_mark = self.scratch.retire_log.chunks.len();
        let interiors_mark = self.scratch.retire_log.interiors.len();

        match tree.insert_with_scratch(
            &mut self.arena,
            dst,
            &mut self.scratch.rebalance,
            &mut self.scratch.retire_log,
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
                        self.scratch.retire_log.chunks.truncate(chunks_mark);
                        self.scratch.retire_log.interiors.truncate(interiors_mark);
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
                self.note_root_store(src);
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

    /// Refuse further inserts once this guard's WAL has failed: the log's
    /// state is uncertain, and the guard poisons the graph on drop.
    #[inline]
    fn check_wal(&self) -> Result<(), InsertError> {
        #[cfg(feature = "wal")]
        if self.wal_failed {
            return Err(InsertError::WalAppend);
        }
        Ok(())
    }

    /// Stamp the retire log at its fixed watermark. Called only after a
    /// root store, so every logged slot is already unreachable.
    #[inline]
    fn maybe_stamp_retired(&mut self) {
        if self.scratch.retire_log.wants_flush() {
            // SAFETY: called only after the root stores that unpublished
            // every logged slot.
            unsafe {
                self.arena
                    .retire(&mut self.scratch.retire_log, self.retire_stamp)
            };
        }
    }

    /// Insert a batch of edges given in any order, duplicates allowed.
    ///
    /// Every edge is range-checked before anything is inserted, so an
    /// out-of-range batch changes nothing. The batch is then sorted and
    /// grouped by source, and each source goes through
    /// [`insert_edges_sorted`](Self::insert_edges_sorted) once. An error
    /// after the check (a full arena, a WAL failure) leaves the sources
    /// before it inserted and published.
    ///
    /// Returns the number of edges actually new.
    pub fn insert_edges(&mut self, edges: &mut [(u32, u32)]) -> Result<u64, InsertError> {
        let nv = self.graph.num_vertices;
        if let Some(&(src, dst)) = edges
            .iter()
            .find(|&&(s, d)| s as usize >= nv || d as usize >= nv)
        {
            return Err(InsertError::VertexOutOfRange { src, dst });
        }
        edges.sort_unstable();
        let mut dsts = std::mem::take(&mut self.scratch.batch_dsts);
        let mut inserted = 0;
        let mut result = Ok(());
        for run in edges.chunk_by(|a, b| a.0 == b.0) {
            dsts.clear();
            dsts.extend(run.iter().map(|&(_, d)| d));
            dsts.dedup();
            match self.insert_edges_sorted(run[0].0, SortedDsts(&dsts)) {
                Ok(n) => inserted += n,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        self.scratch.batch_dsts = dsts;
        result.map(|()| inserted)
    }

    /// Insert every edge `(src, dst)` for `dst` in `dsts`, publishing
    /// `src`'s new tree once.
    ///
    /// Ingest streams are heavily source-clustered. A batch that is large
    /// next to the existing degree merges the old neighbors with `dsts` and
    /// builds the new tree in one pass — O(degree + batch) arena bytes. A
    /// small batch into a large tree instead chains path-copying inserts
    /// off to the side and publishes the last — O(batch × depth) — so one
    /// new edge on a million-neighbor hub costs a root-to-leaf path, not a
    /// rebuild of the hub.
    ///
    /// Returns the number of edges actually new (duplicates are skipped).
    /// An empty `dsts` is a no-op, whatever `src` is.
    pub fn insert_edges_sorted(
        &mut self,
        src: u32,
        dsts: SortedDsts<'_>,
    ) -> Result<u64, InsertError> {
        let dsts = dsts.0;
        let Some(&first) = dsts.first() else {
            return Ok(0);
        };
        let nv = self.graph.num_vertices;
        if src as usize >= nv {
            return Err(InsertError::VertexOutOfRange { src, dst: first });
        }
        // Ascending: only the last destination can be the largest.
        if let Some(&last) = dsts.last()
            && last as usize >= nv
        {
            let bad = dsts[dsts.partition_point(|&d| (d as usize) < nv)];
            return Err(InsertError::VertexOutOfRange { src, dst: bad });
        }
        self.check_wal()?;

        // Relaxed: see `insert_edge`.
        let old_root = self.graph.roots[src as usize].load(Ordering::Relaxed);
        let tree = CTree { root: old_root };
        let degree = tree.count(&self.graph.arena);
        if prefer_path_copy(degree, dsts.len()) {
            self.insert_sorted_by_path_copy(src, tree, dsts)
        } else {
            self.insert_sorted_by_rebuild(src, tree, dsts)
        }
    }

    /// Chain path-copying inserts from `tree`, publishing only the last.
    /// The intermediate trees are never reachable by readers; every node
    /// the chain supersedes (old-tree paths included) is logged exactly
    /// once and becomes unreachable at the single root store.
    fn insert_sorted_by_path_copy(
        &mut self,
        src: u32,
        tree: CTree,
        dsts: &[u32],
    ) -> Result<u64, InsertError> {
        let chunks_mark = self.scratch.retire_log.chunks.len();
        let interiors_mark = self.scratch.retire_log.interiors.len();
        let mut retried = false;
        let new_tree = 'chain: loop {
            self.scratch.new_dsts.clear();
            let mut cur = tree;
            for &d in dsts {
                match cur.insert_with_scratch(
                    &mut self.arena,
                    d,
                    &mut self.scratch.rebalance,
                    &mut self.scratch.retire_log,
                ) {
                    InsertResult::Inserted(t) => {
                        cur = t;
                        self.scratch.new_dsts.push(d);
                    }
                    InsertResult::Duplicate => {}
                    InsertResult::ArenaFull => {
                        // `tree` is still published: un-log what the chain
                        // superseded (its fresh nodes leak until compact).
                        self.scratch.retire_log.chunks.truncate(chunks_mark);
                        self.scratch.retire_log.interiors.truncate(interiors_mark);
                        if retried {
                            return Err(InsertError::ArenaFull);
                        }
                        retried = true;
                        // Everything left in the log was unpublished by an
                        // earlier root store; stamp it so the exhausted
                        // allocator can reclaim, then retry once.
                        // SAFETY: see above.
                        unsafe {
                            self.arena
                                .retire(&mut self.scratch.retire_log, self.retire_stamp)
                        };
                        continue 'chain;
                    }
                }
            }
            break cur;
        };
        if self.scratch.new_dsts.is_empty() {
            return Ok(0);
        }
        if let Err(e) = self.append_batch(src) {
            self.scratch.retire_log.chunks.truncate(chunks_mark);
            self.scratch.retire_log.interiors.truncate(interiors_mark);
            return Err(e);
        }
        let inserted = self.publish_batch(src, new_tree);
        self.maybe_stamp_retired();
        Ok(inserted)
    }

    /// Merge the existing neighbors with `dsts` and build the new tree in
    /// one pass; the whole old tree is retired once it is published.
    fn insert_sorted_by_rebuild(
        &mut self,
        src: u32,
        tree: CTree,
        dsts: &[u32],
    ) -> Result<u64, InsertError> {
        // Merge existing (sorted) neighbors with the new sorted dsts,
        // recording the genuinely-new dsts in their own scratch — the
        // batch's WAL records and dirty marks.
        self.scratch.new_dsts.clear();
        let existing = &mut self.scratch.rebalance;
        existing.clear();
        tree.collect_into(&self.graph.arena, existing);
        let merged = &mut self.scratch.merge;
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
                    self.scratch.new_dsts.push(dsts[j]);
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
            self.scratch.new_dsts.push(d);
        }
        if self.scratch.new_dsts.is_empty() {
            return Ok(0);
        }

        // Build the replacement tree first (invisible to readers), then
        // log, then publish — a WAL failure leaves readers on the old
        // root with no record of the unpublished edges.
        let mut built = CTree::from_sorted(&mut self.arena, &self.scratch.merge);
        if built.is_none() {
            // Stamp already-unpublished retirements so the exhausted
            // allocator can reclaim grace-cleared slots, then retry once
            // (see `insert_edge`).
            // SAFETY: everything in the log was unpublished by a prior
            // root store.
            unsafe {
                self.arena
                    .retire(&mut self.scratch.retire_log, self.retire_stamp)
            };
            built = CTree::from_sorted(&mut self.arena, &self.scratch.merge);
        }
        let Some(new_tree) = built else {
            return Err(InsertError::ArenaFull);
        };
        self.append_batch(src)?;
        let inserted = self.publish_batch(src, new_tree);
        // The old tree is superseded in full now that the merged rebuild
        // is published; log every one of its nodes for recycling.
        if tree.root != crate::ctree::NULL {
            // SAFETY: the old tree is unreachable from the just-published
            // root, and none of its slots were logged before.
            unsafe {
                crate::ctree::retire_subtree(
                    self.arena.arena(),
                    tree.root,
                    &mut self.scratch.retire_log,
                );
            };
        }
        self.maybe_stamp_retired();
        Ok(inserted)
    }

    /// Log the batch in `scratch.new_dsts` for `src`, all or nothing: a
    /// failed append stages none of it, so no record of these unpublished
    /// edges can be flushed later.
    fn append_batch(&mut self, src: u32) -> Result<(), InsertError> {
        #[cfg(feature = "wal")]
        if let Some(w) = self.wal_guard.as_mut()
            && let Err(e) = w.append_edges(src, &self.scratch.new_dsts)
        {
            self.wal_failed = true;
            tracing::error!(error = %e, "WAL append failed");
            return Err(InsertError::WalAppend);
        }
        #[cfg(not(feature = "wal"))]
        let _ = src;
        Ok(())
    }

    /// Publish `new_tree` as `src`'s root and account for the batch in
    /// `scratch.new_dsts`. Returns the number of edges it added.
    fn publish_batch(&mut self, src: u32, new_tree: CTree) -> u64 {
        self.graph.roots[src as usize].store(new_tree.root, Ordering::Release);
        self.note_root_store(src);
        let inserted = self.scratch.new_dsts.len() as u64;
        self.pending_edges += inserted;
        for i in 0..self.scratch.new_dsts.len() {
            let d = self.scratch.new_dsts[i];
            self.note_dirty(d);
        }
        self.note_dirty(src);
        inserted
    }

    /// Record that this guard stored a root in `src`'s root-table leaf.
    #[inline]
    fn note_root_store(&mut self, src: u32) {
        let leaf = src >> crate::snapshot::LEAF_BITS;
        if self.scratch.touched_leaves.last() == Some(&leaf) {
            return;
        }
        self.scratch.touched_leaves.push(leaf);
        if self.scratch.touched_leaves.len() >= self.touched_dedup_limit {
            self.scratch.touched_leaves.sort_unstable();
            self.scratch.touched_leaves.dedup();
            self.touched_dedup_limit =
                (self.scratch.touched_leaves.len() * 2).max(TOUCHED_LEAVES_DEDUP_START);
        }
    }

    /// Record a vertex as dirtied by this guard, flushing at the fixed
    /// watermark so the buffer never grows on the insert path.
    #[inline]
    fn note_dirty(&mut self, v: u32) {
        if self.scratch.dirty.len() == DIRTY_BUF_FLUSH_THRESHOLD {
            self.flush_dirty();
        }
        self.scratch.dirty.push(v);
    }

    /// Sort, dedup, and fold the buffered dirty vertices into the bitmap
    /// in one sequential pass, keeping the buffer's capacity.
    fn flush_dirty(&mut self) {
        if self.scratch.dirty.is_empty() {
            return;
        }
        self.scratch.dirty.sort_unstable();
        self.scratch.dirty.dedup();
        self.graph.dirty.mark_sorted(&self.scratch.dirty);
        self.scratch.dirty.clear();
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
            self.release();
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
            // The un-stamped retire log dies with the guard (`release`
            // clears it): its slots are never reused, which is exactly
            // right — some may belong to trees that are still published.
            self.graph.poisoned.store(true, Ordering::Release);
        } else {
            self.flush_bookkeeping();
            // The table is built off-lock; the epoch advances and the
            // snapshot swaps in under one lock, so a reader that observes
            // the new epoch and then acquires sees this commit. Done before
            // releasing the writer lock so the next writer's commit orders
            // after this one's.
            let table = (!self.scratch.touched_leaves.is_empty())
                .then(|| self.graph.build_table(&mut self.scratch.touched_leaves));
            let epoch = self.graph.publish_commit(table);
            // Stamp this guard's remaining retirements (all root stores are
            // done, and the snapshot published above cannot reach them) and
            // fold in any batches whose grace has passed, so a subsequent
            // guard starts with a warm free list.
            // SAFETY: every logged slot was unpublished by its root store.
            unsafe { self.arena.retire(&mut self.scratch.retire_log, epoch) };
            self.arena.reclaim();
            tracing::trace!(epoch, "DynamicGraph writer committed");
        }
        self.release();
    }
}

impl Writer<'_> {
    /// Park the scratch on the graph for the next guard and free the slot.
    fn release(&mut self) {
        self.scratch.park();
        let mut parked = self
            .graph
            .writer_scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::swap(&mut *parked, &mut self.scratch);
        drop(parked);
        self.graph.writer_locked.store(false, Ordering::Release);
    }
}

/// Strictly ascending destinations for
/// [`Writer::insert_edges_sorted`], proven so at construction — the insert
/// path merges and chains on the order without re-checking it.
#[derive(Clone, Copy, Debug)]
pub struct SortedDsts<'a>(&'a [u32]);

impl<'a> SortedDsts<'a> {
    /// `dsts` if it is strictly ascending, else `None`.
    pub fn new(dsts: &'a [u32]) -> Option<Self> {
        dsts.windows(2).all(|w| w[0] < w[1]).then_some(Self(dsts))
    }

    /// Sort and deduplicate `dsts` in place.
    pub fn sort_dedup(dsts: &'a mut Vec<u32>) -> Self {
        dsts.sort_unstable();
        dsts.dedup();
        Self(dsts)
    }

    pub fn as_slice(&self) -> &'a [u32] {
        self.0
    }
}

/// Should `k` new edges into a `degree`-element tree chain path-copying
/// inserts instead of merging and rebuilding? Compares arena slots
/// written: a path copy per edge (≈ depth + 1 slots) against a rebuild of
/// the merged list (≈ 2 slots per leaf), which also reads the whole old
/// tree.
fn prefer_path_copy(degree: usize, k: usize) -> bool {
    let cap = crate::chunk::CHUNK_CAP;
    let depth = degree.div_ceil(cap).max(1).ilog2() as usize + 2;
    k.saturating_mul(depth) < 2 * (degree + k).div_ceil(cap)
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

#[cfg(all(test, feature = "wal"))]
mod wal_tests {
    use crate::DynamicGraph;
    use crate::wal::{BUF_CAPACITY, RECORD_LEN};
    use crate::writer::{InsertError, SortedDsts};

    #[test]
    fn failed_batch_append_publishes_and_persists_nothing() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        // Stage records until the batch below no longer fits beside them,
        // so appending it must first spill — the step the fault fails.
        let staged = (BUF_CAPACITY / RECORD_LEN - 4) as u32;
        {
            let g = DynamicGraph::open_with_wal(&path, 1 << 12, 64 << 20).unwrap();
            let mut w = g.writer().unwrap();
            for i in 0..staged {
                w.insert_edge(i % 1024, 1024 + i / 1024).unwrap();
            }
            w.wal_guard.as_mut().unwrap().inject_write_fault(0);
            let batch: Vec<u32> = (2048..2112).collect();
            let got = w.insert_edges_sorted(3000, SortedDsts::new(&batch).unwrap());
            assert_eq!(got, Err(InsertError::WalAppend));
            assert_eq!(g.degree(3000), 0, "the failed batch was published");
            assert_eq!(
                w.insert_edge(1, 2),
                Err(InsertError::WalAppend),
                "a guard whose WAL failed must refuse further inserts"
            );
            drop(w);
            assert!(g.is_poisoned());
        }
        let mut srcs = Vec::new();
        crate::wal::replay(&path, |r| {
            srcs.push(r.src);
            Ok(())
        })
        .unwrap();
        assert_eq!(
            srcs.len(),
            staged as usize,
            "every published edge is durable"
        );
        assert!(!srcs.contains(&3000), "no record of the failed batch");
    }
}
