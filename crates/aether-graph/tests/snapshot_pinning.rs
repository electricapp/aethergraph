//! Snapshot semantics: commit atomicity, pin-held stability under churn,
//! epoch bookkeeping, and compact interplay.

use aether_graph::{CompactError, DynamicGraph};

fn insert_all(g: &DynamicGraph, edges: &[(u32, u32)]) {
    let mut w = g.writer().unwrap();
    for &(s, d) in edges {
        w.insert_edge(s, d).unwrap();
    }
}

#[test]
fn snapshot_reflects_only_committed_state() {
    let g = DynamicGraph::new(64, 1 << 20);
    let s0 = g.acquire();
    assert_eq!(s0.num_edges(), 0);
    assert_eq!(s0.degree(&g, 0), 0);

    insert_all(&g, &[(0, 1), (0, 2)]);
    // Acquired before the commit: still empty.
    assert_eq!(s0.degree(&g, 0), 0);

    let s1 = g.acquire();
    assert_eq!(s1.num_edges(), 2);
    assert_eq!(s1.degree(&g, 0), 2);
    assert!(s1.has_edge(&g, 0, 1) && s1.has_edge(&g, 0, 2));
    assert!(!s1.has_edge(&g, 0, 3));

    // Mid-guard state is invisible to a fresh acquire.
    let mut w = g.writer().unwrap();
    w.insert_edge(0, 3).unwrap();
    assert_eq!(g.acquire().degree(&g, 0), 2);
    drop(w);
    assert_eq!(g.acquire().degree(&g, 0), 3);
    // Older snapshots are unmoved.
    assert_eq!(s1.degree(&g, 0), 2);
    assert_eq!(s0.degree(&g, 0), 0);
}

#[test]
fn pinned_snapshot_survives_recycling_churn() {
    let g = DynamicGraph::new(16, 4 << 20);
    insert_all(&g, &[(1, 10), (1, 11), (1, 12)]);
    let snap = g.acquire();

    // Heavy churn on the same vertex across many commits recycles the
    // slots of every superseded tree — except those the pin holds.
    for round in 0u32..200 {
        let mut w = g.writer().unwrap();
        for i in 0..64 {
            let dst = (round * 64 + i) % 16;
            let _ = w.insert_edge(1, dst);
        }
    }

    let mut buf = Vec::new();
    snap.neighbors_into(&g, 1, &mut buf);
    assert_eq!(buf, vec![10, 11, 12]);
    assert_eq!(snap.degree(&g, 1), 3);

    // Live graph moved on.
    assert_eq!(g.degree(1), 16);
    drop(snap);
}

#[test]
fn snapshot_epoch_matches_clock() {
    let g = DynamicGraph::new(8, 1 << 20);
    assert_eq!(g.acquire().epoch(), g.current_epoch());
    insert_all(&g, &[(0, 1)]);
    assert_eq!(g.acquire().epoch(), g.current_epoch());
    insert_all(&g, &[(0, 2)]);
    let s = g.acquire();
    assert_eq!(s.epoch(), g.current_epoch());
    assert_eq!(s.epoch().as_u64(), 2);
}

#[test]
fn empty_commit_keeps_previous_snapshot() {
    let g = DynamicGraph::new(8, 1 << 20);
    insert_all(&g, &[(0, 1)]);
    let e1 = g.acquire().epoch();
    // Duplicate-only guard stores no roots: epoch advances, snapshot stays.
    insert_all(&g, &[(0, 1)]);
    assert!(g.current_epoch() > e1);
    let s = g.acquire();
    assert_eq!(s.epoch(), e1);
    assert_eq!(s.degree(&g, 0), 1);
}

#[test]
fn snapshot_csr_is_a_commit_cut() {
    let g = DynamicGraph::new(4, 1 << 20);
    insert_all(&g, &[(0, 1), (0, 2), (2, 3), (3, 0)]);
    let snap = g.acquire();
    insert_all(&g, &[(1, 3), (0, 3)]);

    let (offsets, edges) = snap.snapshot_csr(&g);
    assert_eq!(offsets, vec![0, 2, 2, 3, 4]);
    assert_eq!(edges, vec![1, 2, 3, 0]);
}

#[test]
fn clones_pin_independently() {
    let g = DynamicGraph::new(8, 1 << 20);
    insert_all(&g, &[(0, 1)]);
    let a = g.acquire();
    let b = a.clone();
    drop(a);
    insert_all(&g, &[(0, 2)]);
    assert_eq!(b.degree(&g, 0), 1);
}

#[test]
fn out_of_range_vertex_reads_empty() {
    let g = DynamicGraph::new(4, 1 << 20);
    let s = g.acquire();
    assert_eq!(s.degree(&g, 99), 0);
    assert!(!s.has_edge(&g, 99, 0));
}

#[test]
fn compact_refuses_while_snapshots_held() {
    let mut g = DynamicGraph::new(8, 1 << 20);
    insert_all(&g, &[(0, 1), (0, 2)]);
    let snap = g.acquire();
    assert_eq!(g.compact(), Err(CompactError::Pinned));
    drop(snap);
    g.compact().unwrap();
    // Post-compact snapshots read the rebuilt arena.
    let s = g.acquire();
    assert_eq!(s.degree(&g, 0), 2);
    assert!(s.has_edge(&g, 0, 1));
}

#[test]
#[should_panic(expected = "different graph")]
fn snapshot_rejects_foreign_graph() {
    let g1 = DynamicGraph::new(8, 1 << 20);
    let g2 = DynamicGraph::new(8, 1 << 20);
    let s = g1.acquire();
    let _ = s.degree(&g2, 0);
}

#[test]
fn arena_pressure_relieved_by_dropping_snapshot() {
    // One edge per guard: each commit supersedes the previous guard's
    // tree, so its garbage is "old" and stays pinned by the snapshot
    // until the drop below lets it drain.
    let g = DynamicGraph::new(4096, 96 << 10);
    insert_all(&g, &[(0, 1)]);
    let snap = g.acquire();

    let mut full_at = None;
    for dst in 0u32..100_000 {
        let mut w = g.writer().unwrap();
        if w.insert_edge(1, dst).is_err() {
            full_at = Some(dst);
            break;
        }
    }
    let next = full_at.expect("pinned per-commit churn should exhaust a small arena");

    drop(snap);
    // With the pin gone the old batches drain; the insert's internal
    // reclaim-and-retry path finds the freed slots.
    let mut w = g.writer().unwrap();
    w.insert_edge(1, next).unwrap();
}
