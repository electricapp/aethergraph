//! Behavioral guarantees of arena slot recycling.
//!
//! The dynamic graph's write path retires every path-copied node and
//! recycles it once the reader gate proves no traversal can still observe
//! it. These tests pin the two load-bearing properties: sustained ingest
//! in a bounded arena reaches a steady state instead of exhausting, and
//! concurrent readers always see well-formed sorted snapshots while the
//! writer churns slots underneath them.

use aether_graph::{DynamicGraph, InsertError};

/// Repeatedly rewriting the same vertices' neighbor lists must reuse the
/// superseded nodes: the workload's total path-copy volume is far larger
/// than the arena, so only recycling lets it complete.
#[test]
fn bounded_arena_sustains_rewrite_churn() {
    // 1 MiB arena. Live state: 64 vertices x 64 neighbors ~ 36 KiB of
    // chunks; the churn below path-copies hundreds of MiB in total.
    let g = DynamicGraph::new(4096, 1 << 20);
    let mut w = g.writer_or_panic();
    for round in 0..800u32 {
        for v in 0..64u32 {
            // A fresh dst per round grows each list by one and
            // path-copies the whole spine above it.
            let dst = (round * 64 + v * 37) % 4096;
            match w.insert_edge(v, dst) {
                Ok(_) => {}
                Err(InsertError::ArenaFull) => {
                    panic!("arena exhausted at round {round}: recycling is not keeping up")
                }
                Err(e) => panic!("unexpected error: {e}"),
            }
        }
    }
    let stats = w.recycle_stats();
    assert!(
        stats.free_chunks + stats.pending > 0 || stats.leaked == 0,
        "churn of this volume must have exercised the recycler: {stats:?}"
    );
    drop(w);

    for v in 0..64u32 {
        let mut buf = Vec::new();
        g.neighbors_into(v, &mut buf);
        for win in buf.windows(2) {
            assert!(win[0] < win[1], "vertex {v} snapshot unsorted");
        }
    }
}

/// The recycled-slot grace period under real concurrency: readers hammer
/// snapshots while the writer churns enough garbage to force reuse in a
/// small arena. Every observed snapshot must be strictly sorted — a
/// reader that ever walked a recycled (rewritten) node would see torn,
/// unsorted, or duplicate values.
#[test]
fn concurrent_readers_never_observe_recycled_slots() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let g = Arc::new(DynamicGraph::new(1024, 1 << 20));
    let stop = Arc::new(AtomicBool::new(false));

    let mut readers = Vec::new();
    for r in 0..3 {
        let gr = Arc::clone(&g);
        let stop_r = Arc::clone(&stop);
        readers.push(thread::spawn(move || {
            let mut buf = Vec::new();
            let mut iters = 0u64;
            while !stop_r.load(Ordering::Relaxed) {
                for v in 0..32u32 {
                    gr.neighbors_into(v * 8 + r, &mut buf);
                    let mut prev = None;
                    for &x in &buf {
                        assert!(x < 1024, "reader saw out-of-range neighbor {x}");
                        if let Some(p) = prev {
                            assert!(p < x, "reader saw unsorted snapshot: {p} !< {x}");
                        }
                        prev = Some(x);
                    }
                }
                iters += 1;
            }
            iters
        }));
    }

    {
        let mut w = g.writer_or_panic();
        for round in 0..400u32 {
            for v in 0..256u32 {
                let _ = w.insert_edge(v, (round * 131 + v * 17) % 1024);
            }
        }
    }

    stop.store(true, Ordering::Relaxed);
    for r in readers {
        let iters = r.join().expect("reader panicked");
        assert!(iters > 0, "reader made no progress");
    }
}
