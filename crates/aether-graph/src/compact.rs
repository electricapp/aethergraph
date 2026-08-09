//! Full-graph compaction: parallel rebuild into a fresh arena.

use crate::arena::Arena;
use crate::graph::DynamicGraph;
use std::sync::atomic::AtomicU32;

impl DynamicGraph {
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
    /// The rebuild is parallel: a prepass computes each vertex's exact
    /// slot cost (`ceil(deg/15)` chunks, one fewer interiors — perfectly
    /// balanced trees are deterministic), vertices are partitioned into
    /// contiguous ranges of roughly equal edge count, and each thread
    /// builds its range into a privately reserved, disjoint slot region
    /// of the new arena — no atomics, no allocation races, and the
    /// capacity check happens up front instead of failing mid-rebuild.
    ///
    /// # Panics
    /// Panics if `new_capacity` is 0 or exceeds [`Arena::MAX_CAPACITY`]
    /// (the [`Arena::new`] contract).
    pub fn compact_with_capacity(&mut self, new_capacity: usize) -> Result<(), CompactError> {
        if self.is_poisoned() {
            return Err(CompactError::Poisoned);
        }

        let (offsets, edges) = self.snapshot_csr();
        let nv = self.num_vertices;

        // Exact slot cost of the fully compacted graph, and the per-thread
        // partition. Ranges split on edge count so one hub-heavy stretch
        // doesn't serialize the rebuild.
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, 16);
        let threads = if edges.len() < (1 << 20) { 1 } else { threads };

        let mut bounds = Vec::with_capacity(threads + 1);
        bounds.push(0usize);
        for t in 1..threads {
            let target = (edges.len() as u64) * (t as u64) / (threads as u64);
            let v = offsets.partition_point(|&o| o < target).min(nv);
            bounds.push((*bounds.last().expect("non-empty")).max(v));
        }
        bounds.push(nv);

        // Per-range slot totals from the deterministic cost formula.
        let mut range_chunks = vec![0usize; threads];
        let mut range_interiors = vec![0usize; threads];
        for t in 0..threads {
            let (mut c, mut i) = (0usize, 0usize);
            for v in bounds[t]..bounds[t + 1] {
                let deg = (offsets[v + 1] - offsets[v]) as usize;
                let (dc, di) = crate::ctree::compact_slot_cost(deg);
                c += dc;
                i += di;
            }
            range_chunks[t] = c;
            range_interiors[t] = i;
        }
        let total_chunks: usize = range_chunks.iter().sum();
        let total_interiors: usize = range_interiors.iter().sum();
        if total_chunks * crate::arena::CHUNK_SLOT + total_interiors * crate::arena::INTERIOR_SLOT
            > new_capacity
        {
            return Err(CompactError::ArenaFull);
        }

        let new_arena = Arena::new(new_capacity);
        let mut new_roots: Vec<u32> = vec![crate::ctree::NULL; nv];

        {
            let mut root_slices: Vec<&mut [u32]> = Vec::with_capacity(threads);
            let mut rest: &mut [u32] = &mut new_roots;
            for t in 0..threads {
                let (head, tail) = rest.split_at_mut(bounds[t + 1] - bounds[t]);
                root_slices.push(head);
                rest = tail;
            }

            let arena_ref = &new_arena;
            let offsets_ref = &offsets;
            let edges_ref = &edges;
            let bounds_ref = &bounds;
            std::thread::scope(|s| {
                let mut chunk_base = 0u32;
                let mut interior_base = 0u32;
                for (t, roots_out) in root_slices.into_iter().enumerate() {
                    let cb = chunk_base;
                    let ib = interior_base;
                    let cc = range_chunks[t] as u32;
                    let ic = range_interiors[t] as u32;
                    chunk_base += cc;
                    interior_base += ic;
                    s.spawn(move || {
                        // SAFETY: the ranges handed out here are disjoint
                        // by construction (prefix sums over exact costs)
                        // and committed below after every thread joins.
                        let mut region = unsafe { arena_ref.region(cb, cc, ib, ic) };
                        for (i, v) in (bounds_ref[t]..bounds_ref[t + 1]).enumerate() {
                            let start = offsets_ref[v] as usize;
                            let end = offsets_ref[v + 1] as usize;
                            roots_out[i] = if start == end {
                                crate::ctree::NULL
                            } else {
                                crate::ctree::build_balanced_region(
                                    &mut region,
                                    &edges_ref[start..end],
                                )
                            };
                        }
                    });
                }
            });
        }

        // SAFETY: all region writers joined; their reservations tile the
        // committed extents exactly.
        unsafe { new_arena.commit_regions(total_chunks, total_interiors) };

        // Rebuild succeeded in full — only now replace the live state.
        self.arena = new_arena;
        self.roots = new_roots.into_iter().map(AtomicU32::new).collect();
        Ok(())
    }
}

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
