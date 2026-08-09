//! Frequency-weighted cache for one tier (GPU or CPU).

use super::slab::RowSlab;
use crate::graph::NodeId;
use ahash::AHashMap;
use rustc_hash::FxHashSet;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, Ordering};

/// One cached row plus its runtime hit counter. The counter is atomic so
/// cache hits bump it under a read lock; a hit that needed the tier's
/// WRITE lock would serialize every concurrent loader on one lock.
struct CacheEntry {
    row: u32,
    hits: AtomicU32,
}

/// Outcome of a tier insert.
pub(super) enum InsertOutcome {
    /// Stored without displacing anyone.
    Stored,
    /// Stored; the victim node was evicted and its row moved out with it,
    /// ready for demotion to the next tier.
    StoredEvicting(NodeId, Vec<f32>),
    /// Not stored — every resident entry is pinned. The caller still owns
    /// the input slice and demotes it to the next tier directly.
    Refused,
}

/// Frequency-weighted cache for a single tier.
///
/// `get()` is O(1), takes `&self`, and is called under a read lock.
/// Eviction pops the lowest-frequency unpinned node from a min-heap. The
/// heap uses lazy deletion -- stale entries are skipped on pop.
pub(super) struct FreqCache {
    capacity: usize,
    cache: AHashMap<NodeId, CacheEntry>,
    /// Row storage for resident entries.
    slab: RowSlab,
    /// Static frequencies from warmup (set once, never mutated after init)
    warmup_freq: AHashMap<NodeId, u32>,
    /// Pinned nodes (never evicted)
    pinned: FxHashSet<NodeId>,
    /// Min-heap: (frequency, node_id). Lazy deletion on pop.
    eviction_heap: BinaryHeap<Reverse<(u32, NodeId)>>,
}

impl FreqCache {
    pub(super) fn new(capacity: usize, dim: usize, warmup_freq: AHashMap<NodeId, u32>) -> Self {
        Self {
            capacity,
            cache: AHashMap::with_capacity(capacity),
            slab: RowSlab::new(dim),
            warmup_freq,
            pinned: FxHashSet::default(),
            eviction_heap: BinaryHeap::with_capacity(capacity),
        }
    }

    fn freq(&self, node: NodeId) -> u32 {
        let hits = self
            .cache
            .get(&node)
            .map(|e| e.hits.load(Ordering::Relaxed))
            .unwrap_or(0);
        self.warmup_freq.get(&node).copied().unwrap_or(0) + hits
    }

    /// Shared-access hit path: one map probe, one relaxed counter bump.
    /// The heap is NOT touched here — a hit's raised frequency only makes
    /// a node less likely to be evicted, and `evict` reconciles stale heap
    /// values lazily.
    pub(super) fn get(&self, node: NodeId) -> Option<&[f32]> {
        let entry = self.cache.get(&node)?;
        entry.hits.fetch_add(1, Ordering::Relaxed);
        Some(self.slab.row(entry.row))
    }

    pub(super) fn insert(&mut self, node: NodeId, features: &[f32]) -> InsertOutcome {
        debug_assert_eq!(features.len(), self.slab.dim);
        if let Some(entry) = self.cache.get_mut(&node) {
            entry.hits.fetch_add(1, Ordering::Relaxed);
            let row = entry.row;
            self.slab.row_mut(row).copy_from_slice(features);
            return InsertOutcome::Stored;
        }

        // Evict if at capacity.
        let victim = if self.cache.len() >= self.capacity {
            match self.evict() {
                Some(victim) => Some(victim),
                // Nothing is evictable (every entry is pinned). Inserting
                // anyway would push len() past capacity, so refuse the new
                // node; the caller demotes it to the next tier instead.
                None => return InsertOutcome::Refused,
            }
        } else {
            None
        };

        let row = self.slab.alloc();
        self.slab.row_mut(row).copy_from_slice(features);
        self.cache.insert(
            node,
            CacheEntry {
                row,
                hits: AtomicU32::new(0),
            },
        );
        let freq = self.freq(node);
        self.eviction_heap.push(Reverse((freq, node)));

        match victim {
            Some((victim, evicted_row)) => InsertOutcome::StoredEvicting(victim, evicted_row),
            None => InsertOutcome::Stored,
        }
    }

    /// Pop the lowest-frequency unpinned node, returning it together with
    /// its row (copied out so the slab slot can be recycled). Lazy
    /// deletion: skip stale entries.
    ///
    /// Because hits raise a node's frequency without re-pushing the heap, a
    /// popped entry's recorded frequency can be stale (lower than the node's
    /// true current frequency). When that happens the entry is re-pushed at its
    /// true frequency and popping continues, so the node actually evicted is the
    /// current minimum rather than a stale one.
    ///
    /// Pinned entries popped along the way are simply dropped from the
    /// heap — they are not evictable, and re-pushing them would make this
    /// loop spin forever once every cached node is pinned (each pop would
    /// be matched by a push, and the caller holds the tier's write lock).
    /// `unpin` re-registers the node in the heap. Returns `None` when
    /// nothing is evictable.
    fn evict(&mut self) -> Option<(NodeId, Vec<f32>)> {
        while let Some(Reverse((heap_freq, node))) = self.eviction_heap.pop() {
            // Skip if no longer in cache (already evicted)
            if !self.cache.contains_key(&node) {
                continue;
            }
            if self.pinned.contains(&node) {
                continue;
            }
            // Lazy staleness check: if the recorded frequency is below the
            // node's current frequency, this pop is stale. Re-push at the true
            // value and keep looking for the real minimum.
            let current_freq = self.freq(node);
            if heap_freq < current_freq {
                self.eviction_heap.push(Reverse((current_freq, node)));
                continue;
            }
            // Evict this node
            let entry = self.cache.remove(&node)?;
            let row = self.slab.row(entry.row).to_vec();
            self.slab.release(entry.row);
            return Some((node, row));
        }
        None
    }

    pub(super) fn pin(&mut self, node: NodeId) {
        self.pinned.insert(node);
    }

    #[allow(dead_code)]
    pub(super) fn unpin(&mut self, node: NodeId) {
        if self.pinned.remove(&node) && self.cache.contains_key(&node) {
            // The node may have been dropped from the eviction heap while
            // pinned; make it evictable again.
            self.eviction_heap.push(Reverse((self.freq(node), node)));
        }
    }

    pub(super) fn is_pinned(&self, node: NodeId) -> bool {
        self.pinned.contains(&node)
    }

    pub(super) fn len(&self) -> usize {
        self.cache.len()
    }
}
