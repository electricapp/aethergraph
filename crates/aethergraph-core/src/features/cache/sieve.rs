//! SIEVE eviction for one cache tier (GPU or CPU).
//!
//! Entries sit in one insertion-ordered queue. Each carries a `visited`
//! bit that a hit sets. Eviction walks a *hand* from the oldest entry
//! toward the newest: a visited entry has its bit cleared and is stepped
//! over, an unvisited one is evicted. The hand keeps its position between
//! evictions and retained entries do not move — that lazy promotion is
//! what separates SIEVE from CLOCK, and it is why a node has to survive a
//! full sweep to stay resident.
//!
//! The property that matters for sampling: one-hit-wonders (nodes touched
//! by a single batch and never again) are inserted at the head and reach
//! the hand still unvisited, so they leave without ever displacing a hub.
//! Under a power-law degree distribution most sampled nodes are exactly
//! that, so the cache stops churning on them.

use super::slab::RowSlab;
use crate::graph::NodeId;
use ahash::AHashMap;
use rustc_hash::FxHashSet;
use std::sync::atomic::{AtomicBool, Ordering};

/// Sentinel for "no entry" in the intrusive queue links.
const NIL: u32 = u32::MAX;

/// One cached row's place in the queue.
///
/// Indexed by the entry's slab row, so the queue needs no allocator of its
/// own — [`RowSlab`] already recycles those indices.
struct Link {
    node: NodeId,
    /// Set on every hit. Atomic because the hit path runs under the
    /// tier's READ lock; a hit needing the write lock would serialize
    /// every concurrent loader on one lock.
    visited: AtomicBool,
    /// Toward the tail (older).
    prev: u32,
    /// Toward the head (newer).
    next: u32,
}

impl Link {
    fn vacant() -> Self {
        Self {
            node: 0,
            visited: AtomicBool::new(false),
            prev: NIL,
            next: NIL,
        }
    }
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

/// SIEVE-evicted cache for a single tier.
///
/// `get()` is O(1), takes `&self`, and is called under a read lock.
/// Eviction is O(1) amortized: the hand advances at most twice per entry
/// across a full sweep, and each step is a bit test, so there is no heap
/// to reconcile and no ordering to maintain on the hit path.
pub(super) struct SieveCache {
    capacity: usize,
    /// Node to slab row, which is also its index into `links`.
    cache: AHashMap<NodeId, u32>,
    /// Row storage for resident entries.
    slab: RowSlab,
    /// Queue links, indexed by slab row.
    links: Vec<Link>,
    /// Newest entry; inserts land here.
    head: u32,
    /// Oldest entry; the hand wraps back to here.
    tail: u32,
    /// Next eviction candidate. Deliberately NOT reset by inserts or
    /// retentions — it resumes where the previous eviction left it.
    hand: u32,
    /// Static frequencies from warmup (set once, never mutated after init).
    warmup_freq: AHashMap<NodeId, u32>,
    /// Pinned nodes (never evicted).
    pinned: FxHashSet<NodeId>,
}

impl SieveCache {
    pub(super) fn new(capacity: usize, dim: usize, warmup_freq: AHashMap<NodeId, u32>) -> Self {
        Self {
            capacity,
            cache: AHashMap::with_capacity(capacity),
            slab: RowSlab::new(dim),
            links: Vec::with_capacity(capacity),
            head: NIL,
            tail: NIL,
            hand: NIL,
            warmup_freq,
            pinned: FxHashSet::default(),
        }
    }

    /// Shared-access hit path: one map probe, one relaxed bit set.
    ///
    /// Marking is a plain store rather than a read-modify-write — the bit
    /// is idempotent, so repeated hits cost no cache-line ownership
    /// beyond the first.
    pub(super) fn get(&self, node: NodeId) -> Option<&[f32]> {
        let &row = self.cache.get(&node)?;
        self.links[row as usize]
            .visited
            .store(true, Ordering::Relaxed);
        Some(self.slab.row(row))
    }

    pub(super) fn insert(&mut self, node: NodeId, features: &[f32]) -> InsertOutcome {
        debug_assert_eq!(features.len(), self.slab.dim);
        if let Some(&row) = self.cache.get(&node) {
            self.links[row as usize]
                .visited
                .store(true, Ordering::Relaxed);
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
        if self.links.len() <= row as usize {
            self.links.resize_with(row as usize + 1, Link::vacant);
        }
        // A node the warmup pass saw enters already visited, so it gets one
        // sweep of grace instead of competing with cold arrivals on its
        // first pass. That is how the warmup signal survives the move off
        // frequency counters.
        let warm = self.warmup_freq.get(&node).is_some_and(|&f| f > 0);
        {
            let link = &mut self.links[row as usize];
            link.node = node;
            link.visited = AtomicBool::new(warm);
        }
        self.push_head(row);
        self.cache.insert(node, row);

        match victim {
            Some((victim, evicted_row)) => InsertOutcome::StoredEvicting(victim, evicted_row),
            None => InsertOutcome::Stored,
        }
    }

    /// Advance the hand until an unvisited, unpinned entry is found and
    /// evict it, returning it with its row (copied out so the slab slot can
    /// be recycled).
    ///
    /// Clearing bits as it goes means two sweeps are enough to leave every
    /// entry unvisited, so the walk is bounded rather than able to spin:
    /// exhausting the budget means every resident entry is pinned, and the
    /// caller demotes instead.
    fn evict(&mut self) -> Option<(NodeId, Vec<f32>)> {
        let mut budget = self.cache.len().saturating_mul(2).saturating_add(2);

        while budget > 0 {
            budget -= 1;

            if self.hand == NIL {
                self.hand = self.tail;
            }
            // Empty queue — nothing to evict.
            if self.hand == NIL {
                return None;
            }

            let cur = self.hand;
            let (node, next, visited) = {
                let link = &self.links[cur as usize];
                (link.node, link.next, link.visited.load(Ordering::Relaxed))
            };

            // Step past entries that must stay, so the hand keeps making
            // progress rather than stalling on a pinned head of queue.
            if self.pinned.contains(&node) {
                self.hand = next;
                continue;
            }
            if visited {
                self.links[cur as usize]
                    .visited
                    .store(false, Ordering::Relaxed);
                self.hand = next;
                continue;
            }

            // Move the hand off the victim before unlinking it.
            self.hand = next;
            self.unlink(cur);
            self.cache.remove(&node);
            let row = self.slab.row(cur).to_vec();
            self.slab.release(cur);
            return Some((node, row));
        }

        None
    }

    /// Remove `r` from the queue, mending its neighbours' links.
    fn unlink(&mut self, r: u32) {
        let (prev, next) = {
            let link = &self.links[r as usize];
            (link.prev, link.next)
        };
        if prev != NIL {
            self.links[prev as usize].next = next;
        } else {
            self.tail = next;
        }
        if next != NIL {
            self.links[next as usize].prev = prev;
        } else {
            self.head = prev;
        }
        let link = &mut self.links[r as usize];
        link.prev = NIL;
        link.next = NIL;
    }

    /// Splice `r` in as the newest entry.
    fn push_head(&mut self, r: u32) {
        let old_head = self.head;
        {
            let link = &mut self.links[r as usize];
            link.prev = old_head;
            link.next = NIL;
        }
        if old_head != NIL {
            self.links[old_head as usize].next = r;
        } else {
            self.tail = r;
        }
        self.head = r;
    }

    pub(super) fn pin(&mut self, node: NodeId) {
        self.pinned.insert(node);
    }

    /// Make `node` evictable again.
    ///
    /// Pinned entries keep their place in the queue rather than being
    /// dropped from it, so unpinning needs no requeue — the hand will
    /// reach the entry on its next pass.
    #[allow(dead_code)]
    pub(super) fn unpin(&mut self, node: NodeId) {
        self.pinned.remove(&node);
    }

    pub(super) fn is_pinned(&self, node: NodeId) -> bool {
        self.pinned.contains(&node)
    }

    pub(super) fn len(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(capacity: usize) -> SieveCache {
        SieveCache::new(capacity, 1, AHashMap::new())
    }

    fn insert(c: &mut SieveCache, node: NodeId) -> Option<NodeId> {
        match c.insert(node, &[node as f32]) {
            InsertOutcome::Stored => None,
            InsertOutcome::StoredEvicting(victim, _) => Some(victim),
            InsertOutcome::Refused => panic!("unexpected refusal"),
        }
    }

    #[test]
    fn evicts_in_insertion_order_when_nothing_was_hit() {
        let mut c = cache(3);
        for n in 0..3 {
            assert_eq!(insert(&mut c, n), None);
        }
        // Oldest first, since no entry was ever visited.
        assert_eq!(insert(&mut c, 3), Some(0));
        assert_eq!(insert(&mut c, 4), Some(1));
    }

    /// The whole point of the policy: a hit buys an entry one sweep, and
    /// the cold neighbour is taken instead.
    #[test]
    fn a_hit_defers_eviction_to_the_next_unvisited_entry() {
        let mut c = cache(3);
        for n in 0..3 {
            insert(&mut c, n);
        }
        assert!(c.get(0).is_some());

        // 0 is visited, so the hand clears its bit and takes 1 instead.
        assert_eq!(insert(&mut c, 3), Some(1));
        assert!(c.get(0).is_some(), "0 should have survived");
    }

    /// Retained entries stay where they are — they are not promoted to the
    /// head as LRU would. The hand therefore leaves a survivor *behind*
    /// it, and keeps working through newer entries; the survivor is not
    /// reconsidered until the hand wraps. This is the lazy-promotion
    /// half of the policy, and it is why a hit is cheap: nothing is
    /// relinked on the hit path.
    #[test]
    fn a_survivor_stays_put_behind_the_hand() {
        let mut c = cache(3);
        for n in 0..3 {
            insert(&mut c, n);
        }
        c.get(0);

        // The hand clears 0's bit, steps over it, and takes 1. From here
        // it is positioned past 0 and chases the newer entries.
        assert_eq!(insert(&mut c, 3), Some(1));
        assert_eq!(insert(&mut c, 4), Some(2));
        assert_eq!(insert(&mut c, 5), Some(3));
        assert!(c.get(0).is_some(), "survivor should sit behind the hand");
    }

    /// When every resident entry has been hit, the sweep clears all the
    /// bits, runs off the head, wraps to the oldest, and evicts there — so
    /// a uniformly-hot cache degrades to FIFO rather than refusing to
    /// evict or spinning.
    #[test]
    fn a_full_sweep_clears_bits_then_wraps_to_the_oldest() {
        let mut c = cache(3);
        for n in 0..3 {
            insert(&mut c, n);
        }
        for n in 0..3 {
            assert!(c.get(n).is_some());
        }

        assert_eq!(insert(&mut c, 3), Some(0));
        assert!(c.get(1).is_some());
        assert!(c.get(2).is_some());
    }

    /// With no hits at all the hand advances one entry per eviction from
    /// the oldest end, so the policy degenerates to FIFO.
    #[test]
    fn degenerates_to_fifo_without_hits() {
        let mut c = cache(3);
        for n in 0..3 {
            insert(&mut c, n);
        }
        for (step, expected) in [(3u32, 0u32), (4, 1), (5, 2), (6, 3)] {
            assert_eq!(insert(&mut c, step), Some(expected));
        }
    }

    #[test]
    fn pinned_entries_are_stepped_over() {
        let mut c = cache(3);
        for n in 0..3 {
            insert(&mut c, n);
        }
        c.pin(0);

        assert_eq!(insert(&mut c, 3), Some(1));
        assert!(c.get(0).is_some(), "pinned node must stay resident");
    }

    #[test]
    fn refuses_when_every_entry_is_pinned() {
        let mut c = cache(2);
        insert(&mut c, 0);
        insert(&mut c, 1);
        c.pin(0);
        c.pin(1);

        assert!(matches!(c.insert(9, &[9.0]), InsertOutcome::Refused));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn unpinning_restores_evictability_without_requeue() {
        let mut c = cache(2);
        insert(&mut c, 0);
        insert(&mut c, 1);
        c.pin(0);
        c.pin(1);
        assert!(matches!(c.insert(9, &[9.0]), InsertOutcome::Refused));

        c.unpin(0);
        assert_eq!(insert(&mut c, 9), Some(0));
    }

    #[test]
    fn warm_nodes_enter_with_a_sweep_of_grace() {
        let mut warmup = AHashMap::new();
        warmup.insert(0u32, 50);
        let mut c = SieveCache::new(3, 1, warmup);
        for n in 0..3 {
            insert(&mut c, n);
        }

        // 0 entered visited, so the first sweep clears it and takes 1.
        assert_eq!(insert(&mut c, 3), Some(1));
        assert!(c.get(0).is_some());
    }

    #[test]
    fn reinsert_updates_row_without_growing_the_queue() {
        let mut c = cache(2);
        insert(&mut c, 0);
        assert!(matches!(c.insert(0, &[42.0]), InsertOutcome::Stored));
        assert_eq!(c.len(), 1);
        assert_eq!(c.get(0), Some(&[42.0][..]));
    }

    /// Hit rate on a skewed key distribution, which is what neighbour
    /// sampling produces: a few hub nodes are requested constantly while a
    /// long tail is touched once and never again.
    ///
    /// Run with `--nocapture` to see the rate; the assertion is a floor
    /// that catches a policy regression rather than a tuned target.
    #[test]
    fn hit_rate_on_a_skewed_workload() {
        const CAPACITY: usize = 1_000;
        const KEYS: u64 = 20_000;
        const REQUESTS: usize = 200_000;

        let mut c = cache(CAPACITY);
        // Deterministic Zipf-like draw: cube a uniform sample so low
        // (hot) keys dominate, without pulling in an RNG dependency.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut hits = 0usize;
        // A clairvoyant cache pinned to the CAPACITY hottest keys would
        // serve exactly the requests that land in that range, so this is
        // the ceiling any policy can reach on this trace. Comparing
        // against it keeps the assertion meaningful when the distribution
        // is tuned, instead of hard-coding a rate that silently stops
        // meaning anything.
        let mut oracle_hits = 0usize;

        for _ in 0..REQUESTS {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            let key = ((u * u * u) * KEYS as f64) as u32 % KEYS as u32;

            if (key as usize) < CAPACITY {
                oracle_hits += 1;
            }
            if c.get(key).is_some() {
                hits += 1;
            } else {
                insert(&mut c, key);
            }
        }

        let rate = hits as f64 / REQUESTS as f64;
        let oracle = oracle_hits as f64 / REQUESTS as f64;
        let efficiency = rate / oracle;
        println!(
            "sieve hit rate: {rate:.4} ({hits}/{REQUESTS}); \
             oracle {oracle:.4}; efficiency {efficiency:.3}"
        );
        assert!(
            efficiency > 0.75,
            "hit rate {rate:.4} is only {efficiency:.3} of the {oracle:.4} ceiling"
        );
    }

    /// Slab rows are recycled, and the queue is indexed by row — so a long
    /// churn must not corrupt the links or leak entries.
    #[test]
    fn sustained_churn_keeps_the_queue_consistent() {
        let mut c = cache(8);
        for n in 0..500u32 {
            insert(&mut c, n);
            if n % 3 == 0 {
                c.get(n);
            }
            assert!(c.len() <= 8, "capacity exceeded at {n}");
        }

        // Every resident node must be reachable by walking the queue from
        // the tail, exactly once.
        let mut seen = 0usize;
        let mut cur = c.tail;
        while cur != NIL {
            seen += 1;
            assert!(seen <= c.len(), "queue contains a cycle or stale link");
            cur = c.links[cur as usize].next;
        }
        assert_eq!(seen, c.len(), "queue and map disagree on residency");
    }
}
