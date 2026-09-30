//! Reader-gate stress: more reader threads than gate stripes, with nested
//! guards, while the writer recycles slots in a small arena.
//!
//! Stripes are shared once readers outnumber them, and nested guards share
//! their thread's stripe — the two ways one reader's exit could be counted
//! against another reader's entry. Each hub vertex draws its neighbors from
//! a private id range, so a traversal that walks a recycled slot (rewritten
//! for another hub, or threaded onto a free list) sees ids outside its hub's
//! range, an unsorted chunk, or a missing edge.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use aether_graph::DynamicGraph;

const HUBS: u32 = 48;
const SPAN: u32 = 2048;
const READERS: usize = 40;

fn range_of(v: u32) -> std::ops::Range<u32> {
    let lo = HUBS + v * SPAN;
    lo..lo + SPAN
}

#[test]
fn many_readers_and_nested_guards_never_observe_recycled_slots() {
    let num_vertices = (HUBS + HUBS * SPAN) as usize;
    // Allocation pops the free lists before bumping, so superseded slots
    // are reused as soon as the gate clears them. The headroom absorbs
    // readers descheduled mid-traversal (40 threads outnumber the cores),
    // which hold reclamation back without being a correctness concern.
    let g = Arc::new(DynamicGraph::new(num_vertices, 64 << 20));
    let done = Arc::new(AtomicBool::new(false));

    let readers: Vec<_> = (0..READERS)
        .map(|r| {
            let g = Arc::clone(&g);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let mut v = r as u32 % HUBS;
                while !done.load(Ordering::Acquire) {
                    let range = range_of(v);
                    let mut prev: Option<u32> = None;
                    g.for_each_chunk(v, |chunk| {
                        for &x in chunk.as_slice() {
                            assert!(range.contains(&x), "hub {v} read foreign id {x}");
                            if let Some(p) = prev {
                                assert!(p < x, "hub {v} unsorted: {p} !< {x}");
                            }
                            prev = Some(x);
                        }
                        // Nested guards on this thread's stripe, entered and
                        // left while the outer traversal is still live.
                        if let Some(&first) = chunk.as_slice().first() {
                            assert!(g.has_edge(v, first), "hub {v} lost edge {first}");
                            let _ = g.degree(v);
                        }
                    });
                    v = (v + 7) % HUBS;
                }
            })
        })
        .collect();

    // Scattered insert order so every hub splits and rebalances; small
    // commits so the writer retires and reclaims constantly.
    let mut state = 0x2545_F491_u32;
    let mut recycled = 0usize;
    for _ in 0..8 {
        for step in 0..SPAN / 8 {
            let mut w = g.writer().unwrap();
            for v in 0..HUBS {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let dst = range_of(v).start + (state.wrapping_add(step) % SPAN);
                w.insert_edge(v, dst).unwrap();
            }
            let stats = w.recycle_stats();
            recycled = recycled.max(stats.free_chunks + stats.free_interiors);
        }
    }
    assert!(recycled > 0, "the churn never recycled a slot");

    done.store(true, Ordering::Release);
    for r in readers {
        r.join().expect("reader observed a recycled slot");
    }
}
