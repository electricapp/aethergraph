//! Pinned, immutable snapshots of a [`DynamicGraph`].
//!
//! A [`Snapshot`] is a commit artifact: `Writer::drop` path-copies the
//! root-table leaves its guard touched (copy-on-write — every untouched
//! subtree is shared with the previous snapshot) and publishes the result
//! as the graph's latest. [`DynamicGraph::acquire`] clones that latest; the
//! clone's [`PinTicket`] registers its epoch, which holds every arena slot
//! the snapshot can reach out of the recycler until the snapshot drops.
//!
//! Because pinned slots cannot be rewritten, snapshot reads need no
//! [`ReadGuard`](crate::ReadGuard): a sampler pins once per step, then reads
//! with zero per-vertex synchronization while ingest commits underneath it.
//!
//! The latest snapshot is always pinned by the graph itself, so any snapshot
//! `acquire` can hand out is pinned continuously from publication — there is
//! no window in which its slots could be reclaimed.

use crate::arena::{PinRegistry, PinTicket};
use crate::chunk::Chunk;
use crate::ctree::{CTree, NULL};
use crate::graph::DynamicGraph;
use aether_epoch::Epoch;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Roots per leaf (1 KiB). A commit copies each touched leaf whole.
pub(crate) const LEAF_BITS: u32 = 8;
const LEAF: usize = 1 << LEAF_BITS;
/// Children per interior table node (2 KiB of pointers).
const FAN_BITS: u32 = 8;
const FAN: usize = 1 << FAN_BITS;

type Leaf = [u32; LEAF];
/// Covers `LEAF * FAN` vertices.
type Mid = [Arc<Leaf>; FAN];
/// Covers `LEAF * FAN * FAN` vertices; at most `FAN` of them span u32 ids.
type Upper = [Arc<Mid>; FAN];

/// Persistent radix table of C-tree roots:
/// `top[v >> 24][(v >> 16) & 255][(v >> 8) & 255][v & 255]`.
///
/// Nodes are immutable once shared, so a commit rebuilds only the paths to
/// the leaves it touched — O(touched leaves × depth), independent of the
/// vertex count. Slots past `num_vertices` hold `NULL` and are never read.
#[derive(Clone)]
pub(crate) struct RootTable {
    top: Arc<[Arc<Upper>]>,
}

impl RootTable {
    /// All-`NULL` table for `num_vertices` vertices; unused subtrees share
    /// one node per level.
    fn empty(num_vertices: usize) -> Self {
        let leaf: Arc<Leaf> = Arc::new([NULL; LEAF]);
        let mid: Arc<Mid> = Arc::new(std::array::from_fn(|_| Arc::clone(&leaf)));
        let upper: Arc<Upper> = Arc::new(std::array::from_fn(|_| Arc::clone(&mid)));
        let uppers = num_vertices.div_ceil(LEAF * FAN * FAN).max(1);
        Self {
            top: vec![upper; uppers].into(),
        }
    }

    #[inline(always)]
    fn get(&self, v: usize) -> u32 {
        let upper = &self.top[v >> (LEAF_BITS + 2 * FAN_BITS)];
        let mid = &upper[(v >> (LEAF_BITS + FAN_BITS)) & (FAN - 1)];
        let leaf = &mid[(v >> LEAF_BITS) & (FAN - 1)];
        leaf[v & (LEAF - 1)]
    }

    /// A copy of `self` whose leaves in `touched` (ascending, deduplicated
    /// leaf indices) hold the current `roots`; every other node is shared.
    fn with_leaves(&self, touched: &[u32], roots: &[AtomicU32]) -> Self {
        let mut top: Vec<Arc<Upper>> = self.top.to_vec();
        let mut rest = touched;
        while let Some(&first) = rest.first() {
            let u = (first >> (2 * FAN_BITS)) as usize;
            let in_upper = rest.partition_point(|&l| (l >> (2 * FAN_BITS)) as usize == u);
            let (group, tail) = rest.split_at(in_upper);
            rest = tail;
            let mut upper: Upper = (*top[u]).clone();
            let mut group = group;
            while let Some(&first) = group.first() {
                let m = ((first >> FAN_BITS) as usize) & (FAN - 1);
                let in_mid =
                    group.partition_point(|&l| ((l >> FAN_BITS) as usize) & (FAN - 1) == m);
                let (leaves, tail) = group.split_at(in_mid);
                group = tail;
                let mut mid: Mid = (*upper[m]).clone();
                for &l in leaves {
                    mid[l as usize & (FAN - 1)] = Arc::new(copy_leaf(l, roots));
                }
                upper[m] = Arc::new(mid);
            }
            top[u] = Arc::new(upper);
        }
        Self { top: top.into() }
    }

    /// Full table of the current `roots`.
    fn from_roots(roots: &[AtomicU32]) -> Self {
        let leaves: Vec<u32> = (0..roots.len().div_ceil(LEAF) as u32).collect();
        Self::empty(roots.len()).with_leaves(&leaves, roots)
    }
}

/// Leaf `l` of the live roots. Callers run on the writer thread (or under
/// `&mut DynamicGraph`), so Relaxed loads see all prior stores.
fn copy_leaf(l: u32, roots: &[AtomicU32]) -> Leaf {
    let start = (l as usize) << LEAF_BITS;
    let end = (start + LEAF).min(roots.len());
    let mut leaf = [NULL; LEAF];
    for (dst, r) in leaf.iter_mut().zip(&roots[start..end]) {
        *dst = r.load(Ordering::Relaxed);
    }
    leaf
}

/// An immutable view of the graph as of one commit.
///
/// Cheap to clone (table `Arc` + pin re-registration). Reads take the
/// owning graph; using a snapshot against any other graph — including the
/// same graph after a [`compact`](DynamicGraph::compact) — panics.
#[derive(Clone)]
pub struct Snapshot {
    epoch: Epoch,
    num_vertices: usize,
    num_edges: u64,
    table: RootTable,
    /// Holds this epoch's slots out of the recycler; released on drop.
    ticket: PinTicket,
}

impl Snapshot {
    /// Epoch of the commit this snapshot reflects.
    #[inline]
    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// Edge count as of this commit.
    #[inline]
    pub fn num_edges(&self) -> u64 {
        self.num_edges
    }

    /// Root for `vertex`; out-of-range reads as empty.
    #[inline(always)]
    fn root_of(&self, vertex: u32) -> u32 {
        let v = vertex as usize;
        if v >= self.num_vertices {
            return NULL;
        }
        self.table.get(v)
    }

    /// Soundness gate: the table holds slot indices of exactly the arena
    /// whose registry the ticket pins.
    #[inline(always)]
    fn check(&self, graph: &DynamicGraph) {
        assert!(
            Arc::ptr_eq(self.ticket.registry(), graph.arena.pins()),
            "Snapshot used against a different graph (or across a compact)"
        );
    }

    /// Degree of `vertex` at this snapshot's epoch.
    #[inline]
    pub fn degree(&self, graph: &DynamicGraph, vertex: u32) -> usize {
        self.check(graph);
        CTree {
            root: self.root_of(vertex),
        }
        .count(&graph.arena)
    }

    /// Iterate `vertex`'s neighbors in sorted order, one chunk at a time.
    /// Gate-free: the pin keeps every reachable slot unrecycled.
    #[inline]
    pub fn for_each_chunk(&self, graph: &DynamicGraph, vertex: u32, f: impl FnMut(&Chunk)) {
        self.check(graph);
        let tree = CTree {
            root: self.root_of(vertex),
        };
        tree.for_each_chunk(&graph.arena, f);
    }

    /// Collect `vertex`'s neighbors (sorted) into `buf`, clearing it first.
    #[inline]
    pub fn neighbors_into(&self, graph: &DynamicGraph, vertex: u32, buf: &mut Vec<u32>) {
        buf.clear();
        self.check(graph);
        CTree {
            root: self.root_of(vertex),
        }
        .collect_into(&graph.arena, buf);
    }

    /// Does edge (src → dst) exist at this snapshot's epoch?
    #[inline]
    pub fn has_edge(&self, graph: &DynamicGraph, src: u32, dst: u32) -> bool {
        self.check(graph);
        CTree {
            root: self.root_of(src),
        }
        .contains(&graph.arena, dst)
    }

    /// Snapshot to raw CSR arrays — an atomic cut at this commit, unlike
    /// [`DynamicGraph::snapshot_csr`] which reads live per-vertex roots.
    pub fn snapshot_csr(&self, graph: &DynamicGraph) -> (Vec<u64>, Vec<u32>) {
        self.check(graph);
        let mut offsets = Vec::with_capacity(self.num_vertices + 1);
        let mut edges = Vec::with_capacity(self.num_edges as usize);
        offsets.push(0);
        for v in 0..self.num_vertices {
            let tree = CTree {
                root: self.root_of(v as u32),
            };
            tree.for_each_chunk(&graph.arena, |chunk| {
                edges.extend_from_slice(chunk.as_slice());
            });
            offsets.push(edges.len() as u64);
        }
        (offsets, edges)
    }
}

/// Snapshot of an all-`NULL` root table (graph construction).
pub(crate) fn empty_snapshot(num_vertices: usize, pins: &Arc<PinRegistry>, epoch: u64) -> Snapshot {
    Snapshot {
        epoch: Epoch::from(epoch),
        num_vertices,
        num_edges: 0,
        table: RootTable::empty(num_vertices),
        ticket: PinTicket::new(Arc::clone(pins), epoch),
    }
}

impl DynamicGraph {
    /// The latest committed snapshot. Strictly serializable: reflects every
    /// commit up to its [`epoch`](Snapshot::epoch) and nothing of any open
    /// guard. Works on a poisoned graph (the last clean commit stands).
    pub fn acquire(&self) -> Snapshot {
        self.latest.lock().unwrap().clone()
    }

    /// The latest table with the `touched` leaves (leaf indices; sorted and
    /// deduplicated here) re-read from the live roots. Built outside the
    /// `latest` lock: only the writer replaces `latest`, so the table it
    /// starts from is still current when [`publish_commit`] swaps it in.
    ///
    /// [`publish_commit`]: Self::publish_commit
    pub(crate) fn build_table(&self, touched: &mut Vec<u32>) -> RootTable {
        touched.sort_unstable();
        touched.dedup();
        let base = self.latest.lock().unwrap().table.clone();
        base.with_leaves(touched, &self.roots)
    }

    /// Advance the epoch and, with `table`, publish it as that commit's
    /// snapshot — both under the `latest` lock, so an observer of the new
    /// epoch that then acquires sees this commit. The new ticket registers
    /// before the old snapshot drops, so the pinned minimum never regresses;
    /// the old snapshot is released after the lock. Returns the epoch.
    pub(crate) fn publish_commit(&self, table: Option<RootTable>) -> u64 {
        let mut latest = self.latest.lock().unwrap();
        let epoch = self.epoch.advance().as_u64();
        let Some(table) = table else {
            return epoch;
        };
        let next = Snapshot {
            epoch: Epoch::from(epoch),
            num_vertices: self.num_vertices,
            num_edges: self.num_edges.0.load(Ordering::Relaxed),
            table,
            ticket: PinTicket::new(Arc::clone(self.arena.pins()), epoch),
        };
        let prev = std::mem::replace(&mut *latest, next);
        drop(latest);
        drop(prev);
        epoch
    }

    /// Full-copy snapshot of the current roots (post-compact republish).
    pub(crate) fn full_snapshot(&self, epoch: u64) -> Snapshot {
        Snapshot {
            epoch: Epoch::from(epoch),
            num_vertices: self.num_vertices,
            num_edges: self.num_edges.0.load(Ordering::Relaxed),
            table: RootTable::from_roots(&self.roots),
            ticket: PinTicket::new(Arc::clone(self.arena.pins()), epoch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roots(vals: impl IntoIterator<Item = u32>) -> Vec<AtomicU32> {
        vals.into_iter().map(AtomicU32::new).collect()
    }

    #[test]
    fn table_reads_every_vertex_across_levels() {
        // Spans two mids and a partial last leaf.
        let n = LEAF * FAN + 3 * LEAF + 17;
        let live = roots((0..n as u32).map(|v| v ^ 0x5A5A));
        let t = RootTable::from_roots(&live);
        for v in 0..n {
            assert_eq!(t.get(v), v as u32 ^ 0x5A5A, "vertex {v}");
        }
    }

    #[test]
    fn with_leaves_shares_untouched_subtrees() {
        let n = 2 * LEAF * FAN;
        let live = roots((0..n as u32).map(|_| NULL));
        let base = RootTable::from_roots(&live);
        live[5].store(50, Ordering::Relaxed);
        live[LEAF * FAN + 9].store(90, Ordering::Relaxed);
        let touched = [0, (LEAF * FAN + 9) as u32 >> LEAF_BITS];
        let next = base.with_leaves(&touched, &live);
        assert_eq!(next.get(5), 50);
        assert_eq!(next.get(LEAF * FAN + 9), 90);
        assert_eq!(base.get(5), NULL, "the base table is immutable");
        // An untouched leaf is the same allocation in both tables.
        let (b, x) = (&base.top[0][0][1], &next.top[0][0][1]);
        assert!(Arc::ptr_eq(b, x));
        let (b, x) = (&base.top[0][0][0], &next.top[0][0][0]);
        assert!(!Arc::ptr_eq(b, x));
    }

    #[test]
    fn with_leaves_spans_upper_nodes() {
        // Past 2^24 vertices the top level holds more than one upper node.
        let n = LEAF * FAN * FAN + LEAF + 5;
        let live = roots((0..n).map(|_| NULL));
        let base = RootTable::empty(n);
        assert_eq!(base.top.len(), 2);
        live[3].store(30, Ordering::Relaxed);
        live[n - 1].store(99, Ordering::Relaxed);
        let touched = [0, ((n - 1) >> LEAF_BITS) as u32];
        let next = base.with_leaves(&touched, &live);
        assert_eq!(next.get(3), 30);
        assert_eq!(next.get(n - 1), 99);
        assert_eq!(next.get(LEAF * FAN * FAN), NULL);
    }

    #[test]
    fn empty_table_reads_null() {
        let t = RootTable::empty(10);
        for v in 0..LEAF {
            assert_eq!(t.get(v), NULL);
        }
    }
}
