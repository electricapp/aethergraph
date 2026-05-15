//! Miri smoke tests for the `unsafe` blocks in aether-graph.
//!
//! These tests exercise every `unsafe` call site in `Arena`, `Chunk`, and
//! `CTree` end-to-end. They are deliberately small so miri can run them in
//! seconds; the proptest suite in `tests/proptest_ctree.rs` provides the
//! breadth coverage but is too slow for miri's interpreter.
//!
//! Run with:
//!     cargo +nightly miri test -p aether-graph --test miri_smoke

use aether_graph::{Arena, CTree, Chunk, DynamicGraph, InsertResult};

#[test]
fn arena_alloc_write_read_roundtrip() {
    let arena = Arena::new(4096);
    // SAFETY: single-threaded test.
    let off = unsafe { arena.alloc_write::<u64>(0x1234_5678_9abc_def0) }.unwrap();
    // SAFETY: single-threaded test; `off` was just returned by `alloc_write`.
    let read: &u64 = unsafe { arena.get(off) };
    assert_eq!(*read, 0x1234_5678_9abc_def0);
}

#[test]
fn arena_alignment_holds_across_allocations() {
    let arena = Arena::new(8192);
    // SAFETY: single-threaded test; alignment is a power of two.
    let _ = unsafe { arena.alloc(1, 1) }.unwrap();
    // SAFETY: single-threaded test; alignment is a power of two.
    let off = unsafe { arena.alloc(64, 64) }.unwrap();
    assert_eq!(off % 64, 0);
    // SAFETY: single-threaded test; alignment is a power of two.
    let off2 = unsafe { arena.alloc(128, 128) }.unwrap();
    assert_eq!(off2 % 128, 0);
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
    let mut tree = CTree::empty();
    // Trigger leaf path (single insert) AND split path (>15 inserts) AND
    // interior insert (>30 inserts) — covers every unsafe block in ctree.rs.
    for v in 0..32u32 {
        // SAFETY: single-threaded test.
        match unsafe { tree.insert(&arena, v) } {
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
