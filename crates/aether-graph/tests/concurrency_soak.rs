//! Concurrency soak test for the lock-free C-tree: one writer and many readers
//! on a real `DynamicGraph`, on real OS threads.
//!
//! This is the real-hardware verification of the root publish/consume that loom
//! cannot model — the published payload is the bump arena (raw memory loom does
//! not instrument), so the only faithful test is to run the actual code under
//! contention and assert readers never observe a torn or partially-published
//! tree. The writer-guard CAS itself is covered separately by `loom_writer_guard`.
//!
//! Invariant: every snapshot a reader takes is in-range, strictly sorted (a
//! C-tree node invariant), and therefore duplicate-free — never garbage from a
//! half-built tree.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use aether_graph::DynamicGraph;

#[test]
fn concurrent_readers_never_observe_a_torn_tree() {
    const NUM_VERTICES: u32 = 256;
    const EDGES_PER_VERTEX: u32 = 64;

    let g = Arc::new(DynamicGraph::new(NUM_VERTICES as usize, 128 << 20));
    let done = Arc::new(AtomicBool::new(false));

    // Readers: continuously snapshot every vertex's neighbors and validate that
    // each snapshot is internally consistent.
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let g = Arc::clone(&g);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let mut buf = Vec::new();
                while !done.load(Ordering::Acquire) {
                    for v in 0..NUM_VERTICES {
                        g.neighbors_into(v, &mut buf);
                        let mut prev: Option<u32> = None;
                        for &n in &buf {
                            assert!(n < NUM_VERTICES, "neighbor {n} out of range");
                            if let Some(p) = prev {
                                assert!(p < n, "snapshot not strictly sorted: {p} >= {n}");
                            }
                            prev = Some(n);
                        }
                    }
                }
            })
        })
        .collect();

    // Writer: grow every vertex to EDGES_PER_VERTEX distinct neighbors, one
    // writer guard per edge (each guard publishes a new root via atomic swap).
    for round in 0..EDGES_PER_VERTEX {
        for v in 0..NUM_VERTICES {
            let dst = (v + 1 + round) % NUM_VERTICES; // distinct per round, != v, in range
            let mut w = g.writer().unwrap();
            w.insert_edge(v, dst).unwrap();
        }
    }

    done.store(true, Ordering::Release);
    for r in readers {
        r.join().expect("reader observed a torn tree");
    }

    // Final state: every vertex holds exactly the edges we inserted.
    let mut buf = Vec::new();
    for v in 0..NUM_VERTICES {
        g.neighbors_into(v, &mut buf);
        assert_eq!(
            buf.len(),
            EDGES_PER_VERTEX as usize,
            "vertex {v} lost edges"
        );
    }
}

/// Gate-free pinned reads must return the exact frozen adjacency while the
/// writer churns and recycles underneath — the pin, not the reader gate,
/// is what keeps the snapshot's slots from being rewritten.
#[test]
fn pinned_snapshot_reads_are_stable_under_concurrent_churn() {
    const NUM_VERTICES: u32 = 128;

    let g = Arc::new(DynamicGraph::new(NUM_VERTICES as usize, 256 << 20));
    {
        let mut w = g.writer().unwrap();
        for v in 0..NUM_VERTICES {
            for k in 1..=8 {
                w.insert_edge(v, (v + k) % NUM_VERTICES).unwrap();
            }
        }
    }
    let snap = g.acquire();

    // Expected adjacency, frozen through the snapshot itself.
    let mut expected: Vec<Vec<u32>> = Vec::with_capacity(NUM_VERTICES as usize);
    let mut buf = Vec::new();
    for v in 0..NUM_VERTICES {
        snap.neighbors_into(&g, v, &mut buf);
        expected.push(buf.clone());
    }

    let done = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let g = Arc::clone(&g);
            let snap = snap.clone();
            let expected = expected.clone();
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let mut buf = Vec::new();
                while !done.load(Ordering::Acquire) {
                    for v in 0..NUM_VERTICES {
                        snap.neighbors_into(&g, v, &mut buf);
                        assert_eq!(buf, expected[v as usize], "snapshot moved under pin");
                    }
                }
            })
        })
        .collect();

    // Writer: many small commits churning every vertex, forcing retire
    // and recycling of everything the pin doesn't hold.
    for round in 0..48u32 {
        for v in 0..NUM_VERTICES {
            let mut w = g.writer().unwrap();
            for k in 9..=24 {
                let _ = w.insert_edge(v, (v + k + round) % NUM_VERTICES);
            }
        }
    }

    done.store(true, Ordering::Release);
    for r in readers {
        r.join().expect("pinned reader observed churn");
    }
}
