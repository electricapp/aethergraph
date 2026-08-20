//! Pinned, immutable snapshots of a [`DynamicGraph`].
//!
//! A [`Snapshot`] is a commit artifact: `Writer::drop` copies the root-table
//! pages its guard touched (copy-on-write — untouched pages are shared with
//! the previous snapshot) and publishes the result as the graph's latest.
//! [`DynamicGraph::acquire`] clones that latest; the clone's [`PinTicket`]
//! registers its epoch, which holds every arena slot the snapshot can reach
//! out of the recycler until the snapshot drops.
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
use std::sync::atomic::Ordering;

/// Roots per root-table page (4096 → 16 KiB pages).
pub(crate) const PAGE_BITS: u32 = 12;
pub(crate) const PAGE: usize = 1 << PAGE_BITS;

/// An immutable view of the graph as of one commit.
///
/// Cheap to clone (page-table `Arc` + pin re-registration). Reads take the
/// owning graph; using a snapshot against any other graph — including the
/// same graph after a [`compact`](DynamicGraph::compact) — panics.
#[derive(Clone)]
pub struct Snapshot {
    epoch: Epoch,
    num_vertices: usize,
    num_edges: u64,
    /// Two-level root table: `pages[v >> PAGE_BITS][v & (PAGE - 1)]`.
    /// The tail page is exact-sized; the rest hold `PAGE` roots.
    pages: Arc<[Arc<[u32]>]>,
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
        self.pages[v >> PAGE_BITS][v & (PAGE - 1)]
    }

    /// Soundness gate: the pages hold slot indices of exactly the arena
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

/// Snapshot of an all-`NULL` root table (graph construction). Full pages
/// share one allocation; the tail page is exact-sized.
pub(crate) fn empty_snapshot(num_vertices: usize, pins: &Arc<PinRegistry>, epoch: u64) -> Snapshot {
    let n_pages = num_vertices.div_ceil(PAGE);
    let full: Arc<[u32]> = vec![NULL; PAGE].into();
    let mut pages = vec![full; n_pages];
    if let Some(last) = pages.last_mut() {
        let tail = num_vertices - (n_pages - 1) * PAGE;
        if tail != PAGE {
            *last = vec![NULL; tail].into();
        }
    }
    Snapshot {
        epoch: Epoch::from(epoch),
        num_vertices,
        num_edges: 0,
        pages: pages.into(),
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

    /// Copy the live roots of page `p`. Callers run on the writer thread
    /// (or under `&mut self`), so Relaxed loads see all prior stores.
    fn copy_page(&self, p: u32) -> Arc<[u32]> {
        let start = (p as usize) << PAGE_BITS;
        let end = (start + PAGE).min(self.num_vertices);
        self.roots[start..end]
            .iter()
            .map(|r| r.load(Ordering::Relaxed))
            .collect()
    }

    /// Publish the commit-`epoch` snapshot: CoW the touched pages, keep the
    /// rest shared. The new ticket registers before the old snapshot drops,
    /// so the pinned minimum never regresses.
    pub(crate) fn publish_snapshot(&self, epoch: u64, touched: &mut Vec<u32>) {
        touched.sort_unstable();
        touched.dedup();
        let mut latest = self.latest.lock().unwrap();
        let mut pages: Vec<Arc<[u32]>> = latest.pages.to_vec();
        for &p in touched.iter() {
            pages[p as usize] = self.copy_page(p);
        }
        *latest = Snapshot {
            epoch: Epoch::from(epoch),
            num_vertices: self.num_vertices,
            num_edges: self.num_edges.0.load(Ordering::Relaxed),
            pages: pages.into(),
            ticket: PinTicket::new(Arc::clone(self.arena.pins()), epoch),
        };
    }

    /// Full-copy snapshot of the current roots (post-compact republish).
    pub(crate) fn full_snapshot(&self, epoch: u64) -> Snapshot {
        let n_pages = self.num_vertices.div_ceil(PAGE);
        let pages: Vec<Arc<[u32]>> = (0..n_pages as u32).map(|p| self.copy_page(p)).collect();
        Snapshot {
            epoch: Epoch::from(epoch),
            num_vertices: self.num_vertices,
            num_edges: self.num_edges.0.load(Ordering::Relaxed),
            pages: pages.into(),
            ticket: PinTicket::new(Arc::clone(self.arena.pins()), epoch),
        }
    }
}
