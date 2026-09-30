//! Miri smoke tests for the `unsafe` blocks in aether-graph.
//!
//! These tests exercise every `unsafe` call site in `Arena`, `Chunk`, and
//! `CTree` end-to-end. They are deliberately small so miri can run them in
//! seconds; the proptest suite in `tests/proptest_ctree.rs` provides the
//! breadth coverage but is too slow for miri's interpreter. One test runs
//! a real writer/reader race so miri checks the Release/Acquire root
//! publication, not just the single-threaded paths.
//!
//! Run with:
//!     cargo +nightly miri test -p aether-graph --test miri_smoke

use aether_graph::{Arena, CTree, Chunk, DynamicGraph, InsertResult};

#[test]
fn arena_alloc_write_read_roundtrip() {
    let arena = Arena::new(4096);
    let chunk = Chunk::from_sorted(&[7, 8, 9]);
    // SAFETY: sole write handle in a single-threaded test.
    let mut aw = unsafe { arena.writer() };
    let idx = aw.alloc_write_chunk(chunk).unwrap();
    // SAFETY: `idx` was just written as a Chunk and cannot have been retired.
    let read: &Chunk = unsafe { arena.chunk(idx) };
    assert_eq!(read.as_slice(), &[7, 8, 9]);
}

#[test]
fn arena_regions_are_disjoint_and_aligned() {
    let arena = Arena::new(8192);
    // SAFETY: sole write handle in a single-threaded test.
    let mut aw = unsafe { arena.writer() };
    let c = aw.alloc_chunk().unwrap();
    let i = aw.alloc_interior().unwrap();
    // SAFETY: slot `c` was just allocated.
    let pc = unsafe { arena.chunk_ptr(c) };
    // SAFETY: slot `i` was just allocated.
    let pi = unsafe { arena.interior_ptr(i) };
    assert_eq!(pc.addr() % 64, 0);
    assert_eq!(pi.addr() % 16, 0);
    assert!(pc < pi);
}

#[test]
fn chunk_insert_split_merge_roundtrip() {
    let chunk = Chunk::from_sorted(&[1, 3, 5, 7, 9]);
    let inserted = chunk.insert(4).unwrap();
    assert_eq!(inserted.as_slice(), &[1, 3, 4, 5, 7, 9]);
    let (left, right) = inserted.split();
    let merged = Chunk::merge(&left, &right);
    assert_eq!(merged.as_slice(), inserted.as_slice());
}

#[test]
fn ctree_insert_and_contains_exercises_all_unsafe_paths() {
    let arena = Arena::new(1 << 20);
    // SAFETY: sole write handle in a single-threaded test.
    let mut aw = unsafe { arena.writer() };
    let mut tree = CTree::empty();
    // Trigger leaf path (single insert) AND split path (>15 inserts) AND
    // interior insert (>30 inserts) — covers every unsafe block in ctree.rs.
    for v in 0..32u32 {
        match tree.insert(&mut aw, v) {
            InsertResult::Inserted(t) => tree = t,
            other => panic!("unexpected {other:?}"),
        }
    }
    for v in 0..32u32 {
        assert!(tree.contains(&arena, v), "missing {v}");
    }
    let mut buf = Vec::new();
    tree.collect_into(&arena, &mut buf);
    assert_eq!(buf, (0..32u32).collect::<Vec<_>>());
}

#[test]
fn concurrent_writer_and_reader_publication_is_clean() {
    // One writer thread publishing roots (Release stores) racing one
    // reader thread loading them (Acquire loads). Every snapshot a reader
    // observes must be a fully-built, strictly sorted tree. Iteration
    // counts are kept small so `cargo miri test` can interpret the whole
    // interleaving space in reasonable time.
    use std::sync::Arc;
    use std::thread;

    let g = Arc::new(DynamicGraph::new(32, 1 << 16));

    let gw = Arc::clone(&g);
    let writer = thread::spawn(move || {
        let mut w = gw.writer_or_panic();
        for i in 0..100u32 {
            w.insert_edge(i % 4, i / 4).unwrap();
        }
    });

    let gr = Arc::clone(&g);
    let reader = thread::spawn(move || {
        let mut buf = Vec::new();
        for _ in 0..25 {
            for v in 0..4u32 {
                let _ = gr.degree(v);
                gr.neighbors_into(v, &mut buf);
                for w in buf.windows(2) {
                    assert!(w[0] < w[1], "reader saw an unsorted snapshot");
                }
            }
        }
    });

    writer.join().unwrap();
    reader.join().unwrap();

    // After the join, every insert is visible.
    let mut buf = Vec::new();
    for v in 0..4u32 {
        g.neighbors_into(v, &mut buf);
        assert_eq!(buf.len(), 25, "vertex {v} missing edges after join");
    }
}

#[test]
fn odd_capacity_keeps_interior_writes_aligned() {
    // 1001 bytes is not a whole interior slot; the arena rounds it down so
    // the interior region (counted from the top) stays 16-byte aligned.
    let g = DynamicGraph::new(8, 1001);
    let mut w = g.writer_or_panic();
    for dst in 0..8u32 {
        w.insert_edge(0, dst).unwrap();
        w.insert_edge(1, dst).unwrap();
    }
    drop(w);
    let mut buf = Vec::new();
    g.neighbors_into(0, &mut buf);
    assert_eq!(buf, (0..8).collect::<Vec<_>>());
}

#[test]
fn recycling_under_shared_and_nested_guards_is_race_free() {
    // A small arena forces slot reuse while two readers (sharing stripes
    // with the main thread's nested guards) traverse; a slot rewritten
    // under a live traversal is a data race miri reports.
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let g = Arc::new(DynamicGraph::new(8, 16 << 10));
    let done = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..2)
        .map(|_| {
            let g = Arc::clone(&g);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                while !done.load(Ordering::Acquire) {
                    g.for_each_chunk(0, |chunk| {
                        if let Some(&x) = chunk.as_slice().first() {
                            assert!(g.has_edge(0, x));
                        }
                    });
                }
            })
        })
        .collect();
    for round in 0..6u32 {
        let mut w = g.writer_or_panic();
        for i in 0..20u32 {
            let _ = w.insert_edge(0, (round * 20 + i) % 8);
            let _ = w.insert_edge(1 + round % 7, i % 8);
        }
    }
    done.store(true, Ordering::Release);
    for r in readers {
        r.join().unwrap();
    }
}

#[test]
fn batch_strategies_and_snapshot_reads_are_miri_clean() {
    use aether_graph::SortedDsts;

    let g = DynamicGraph::new(512, 1 << 18);
    let wide: Vec<u32> = (0..200).collect();
    {
        let mut w = g.writer_or_panic();
        // Rebuild path.
        w.insert_edges_sorted(3, SortedDsts::new(&wide).unwrap())
            .unwrap();
    }
    let before = g.acquire();
    {
        let mut w = g.writer_or_panic();
        // Path-copy chain on the now-large tree.
        w.insert_edges_sorted(3, SortedDsts::new(&[300, 301]).unwrap())
            .unwrap();
        let mut edges = vec![(9, 1), (8, 2), (9, 0)];
        w.insert_edges(&mut edges).unwrap();
    }
    let after = g.acquire();
    assert_eq!(before.degree(&g, 3), 200);
    assert_eq!(after.degree(&g, 3), 202);
    assert_eq!(after.degree(&g, 9), 2);
    let mut buf = Vec::new();
    after.neighbors_into(&g, 3, &mut buf);
    assert_eq!(buf.len(), 202);
}

#[test]
fn dynamic_graph_writer_path_is_miri_clean() {
    let g = DynamicGraph::new(64, 1 << 16);
    let mut w = g.writer_or_panic();
    for src in 0..8u32 {
        for dst in 0..8u32 {
            if src != dst {
                let _ = w.insert_edge(src, dst);
            }
        }
    }
    drop(w);

    for src in 0..8u32 {
        let mut buf = Vec::new();
        g.neighbors_into(src, &mut buf);
        // Sorted ascending.
        for w in buf.windows(2) {
            assert!(w[0] < w[1]);
        }
    }
}
