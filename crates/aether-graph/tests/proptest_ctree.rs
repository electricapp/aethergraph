//! Property-based and fuzz tests for Chunk, CTree, and DynamicGraph.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use aether_graph::{Arena, CTree, Chunk, DynamicGraph};
use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Chunk capacity. Mirrors the crate-internal CHUNK_CAP constant.
const CHUNK_CAP: usize = 15;

// ── helpers ──────────────────────────────────────────────────────────────────

fn is_sorted_strict(s: &[u32]) -> bool {
    s.windows(2).all(|w| w[0] < w[1])
}

fn is_sorted_nonstrict(s: &[u32]) -> bool {
    s.windows(2).all(|w| w[0] <= w[1])
}

/// Proptest strategy: a sorted, deduplicated Vec<u32> with length in [0, max_len].
fn sorted_dedup_vec(max_len: usize) -> impl Strategy<Value = Vec<u32>> {
    prop::collection::hash_set(0..100_000u32, 0..=max_len).prop_map(|s| {
        let mut v: Vec<u32> = s.into_iter().collect();
        v.sort();
        v
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Chunk properties
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    // 1. insert into a non-full chunk always produces a sorted result
    #[test]
    fn insert_maintains_sorted(
        vals in sorted_dedup_vec(CHUNK_CAP - 1),
        val in 0..100_000u32,
    ) {
        let chunk = Chunk::from_sorted(&vals);
        if let Some(new) = chunk.insert(val) {
            let s = new.as_slice();
            prop_assert!(is_sorted_strict(s),
                "chunk not sorted after insert({}): {:?}", val, s);
        }
    }

    // 2. after inserting val, chunk.contains(val) is true
    #[test]
    fn insert_then_contains(
        vals in sorted_dedup_vec(CHUNK_CAP - 1),
        val in 0..100_000u32,
    ) {
        let chunk = Chunk::from_sorted(&vals);
        if let Some(new) = chunk.insert(val) {
            prop_assert!(new.contains(val),
                "contains({}) false after insert", val);
        }
        // If insert returned None, val was already present
        if chunk.insert(val).is_none() {
            prop_assert!(chunk.contains(val),
                "insert returned None but contains({}) is false", val);
        }
    }

    // 3. split into left+right, merge back, get original elements
    #[test]
    fn split_preserves_elements(vals in sorted_dedup_vec(CHUNK_CAP)) {
        let chunk = Chunk::from_sorted(&vals);
        if chunk.is_empty() {
            return Ok(());
        }
        let (left, right) = chunk.split();
        let merged = Chunk::merge(&left, &right);
        prop_assert_eq!(merged.as_slice(), chunk.as_slice(),
            "split then merge changed elements");
    }

    // 4. merging two sorted chunks produces a sorted chunk
    #[test]
    fn merge_is_sorted(
        a_vals in sorted_dedup_vec(7),
        b_vals in sorted_dedup_vec(7),
    ) {
        let a = Chunk::from_sorted(&a_vals);
        let b = Chunk::from_sorted(&b_vals);
        if a.len() + b.len() > CHUNK_CAP {
            return Ok(());
        }
        let merged = Chunk::merge(&a, &b);
        prop_assert!(is_sorted_nonstrict(merged.as_slice()),
            "merged chunk not sorted: {:?}", merged.as_slice());
    }

    // 5. inserting a value that exists returns None
    #[test]
    fn no_duplicates_after_insert(vals in sorted_dedup_vec(CHUNK_CAP - 1)) {
        if vals.is_empty() {
            return Ok(());
        }
        let chunk = Chunk::from_sorted(&vals);
        let existing = vals[0];
        prop_assert!(chunk.insert(existing).is_none(),
            "insert of existing value {} should return None", existing);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CTree properties
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    // 1. inserting values in any order produces the same set of neighbors
    #[test]
    fn insert_any_order_same_result(vals in prop::collection::hash_set(0..10_000u32, 1..50)) {
        let arena1 = Arena::new(1 << 20);
        let arena2 = Arena::new(1 << 20);

        let order1: Vec<u32> = vals.iter().copied().collect();
        let mut order2 = order1.clone();
        order2.reverse();

        let mut tree1 = CTree::empty();
        for &v in &order1 {
            if let Some(t) = tree1.insert(&arena1, v) {
                tree1 = t;
            }
        }

        let mut tree2 = CTree::empty();
        for &v in &order2 {
            if let Some(t) = tree2.insert(&arena2, v) {
                tree2 = t;
            }
        }

        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();
        tree1.collect_into(&arena1, &mut buf1);
        tree2.collect_into(&arena2, &mut buf2);

        prop_assert_eq!(&buf1, &buf2,
            "different insertion orders produced different neighbor sets");
    }

    // 2. after inserting N values, all N are found via contains()
    #[test]
    fn insert_then_contains_all(vals in prop::collection::hash_set(0..10_000u32, 0..100)) {
        let arena = Arena::new(2 << 20);
        let mut tree = CTree::empty();
        let vals_vec: Vec<u32> = vals.into_iter().collect();

        for &v in &vals_vec {
            if let Some(t) = tree.insert(&arena, v) {
                tree = t;
            }
        }

        for &v in &vals_vec {
            prop_assert!(tree.contains(&arena, v),
                "value {} not found after insert", v);
        }
    }

    // 3. count always equals number of unique inserts
    #[test]
    fn insert_never_loses_elements(vals in prop::collection::vec(0..5_000u32, 0..200)) {
        let arena = Arena::new(4 << 20);
        let mut tree = CTree::empty();
        let mut unique = BTreeSet::new();

        for &v in &vals {
            if unique.insert(v) {
                let t = tree.insert(&arena, v);
                prop_assert!(t.is_some(),
                    "insert of new value {} returned None (arena full?)", v);
                tree = t.unwrap();
            } else {
                prop_assert!(tree.insert(&arena, v).is_none(),
                    "insert of duplicate {} should return None", v);
            }
        }

        let got = tree.count(&arena);
        let want = unique.len();
        prop_assert_eq!(got, want,
            "count mismatch: tree={}, expected={}", got, want);
    }

    // 4. collecting all elements always produces a sorted list
    #[test]
    fn collect_is_sorted(vals in prop::collection::vec(0..10_000u32, 0..200)) {
        let arena = Arena::new(4 << 20);
        let mut tree = CTree::empty();

        for &v in &vals {
            if let Some(t) = tree.insert(&arena, v) {
                tree = t;
            }
        }

        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        prop_assert!(is_sorted_strict(&buf),
            "collected elements not strictly sorted: first unsorted pair at {:?}",
            buf.windows(2).find(|w| w[0] >= w[1]));
    }

    // 5. functional persistence: after inserting into tree1 to get tree2,
    //    tree1 still has its original elements
    #[test]
    fn functional_persistence(
        phase1 in prop::collection::hash_set(0..10_000u32, 1..30),
        phase2 in prop::collection::hash_set(10_000..20_000u32, 1..30),
    ) {
        let arena = Arena::new(2 << 20);

        // Build tree1
        let mut tree1 = CTree::empty();
        for &v in &phase1 {
            if let Some(t) = tree1.insert(&arena, v) {
                tree1 = t;
            }
        }

        let mut snap1 = Vec::new();
        tree1.collect_into(&arena, &mut snap1);

        // Build tree2 by inserting more into tree1
        let mut tree2 = tree1;
        for &v in &phase2 {
            if let Some(t) = tree2.insert(&arena, v) {
                tree2 = t;
            }
        }

        // tree1 must still have its original elements (functional persistence)
        let mut snap1_after = Vec::new();
        tree1.collect_into(&arena, &mut snap1_after);
        prop_assert_eq!(&snap1, &snap1_after,
            "tree1 was mutated after inserting into tree2");

        // tree2 must contain everything from both phases
        let mut snap2 = Vec::new();
        tree2.collect_into(&arena, &mut snap2);
        let snap2_set: BTreeSet<u32> = snap2.into_iter().collect();
        for &v in &phase1 {
            prop_assert!(snap2_set.contains(&v),
                "tree2 missing phase1 value {}", v);
        }
        for &v in &phase2 {
            prop_assert!(snap2_set.contains(&v),
                "tree2 missing phase2 value {}", v);
        }
    }

    // 6. insert 1000 random u32 values, verify all present and count correct
    #[test]
    fn stress_insert_1000(vals in prop::collection::vec(0..100_000u32, 1000..=1000)) {
        let arena = Arena::new(8 << 20);
        let mut tree = CTree::empty();
        let mut unique = BTreeSet::new();

        for &v in &vals {
            unique.insert(v);
            if let Some(t) = tree.insert(&arena, v) {
                tree = t;
            }
        }

        prop_assert_eq!(tree.count(&arena), unique.len());

        for &v in &unique {
            prop_assert!(tree.contains(&arena, v),
                "missing value {}", v);
        }

        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        prop_assert!(is_sorted_strict(&buf));
        prop_assert_eq!(buf.len(), unique.len());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DynamicGraph properties
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    // 1. for any (src, dst), insert then has_edge returns true
    #[test]
    fn insert_edge_then_has_edge(src in 0..1000u32, dst in 0..1000u32) {
        let g = DynamicGraph::new(1000, 1 << 20);
        let _ = g.insert_edge(src, dst);
        prop_assert!(g.has_edge(src, dst),
            "has_edge({}, {}) false after insert", src, dst);
    }

    // 2. for any sequence of inserts, neighbors() is always sorted
    #[test]
    fn neighbors_always_sorted(
        edges in prop::collection::vec((0..100u32, 0..10_000u32), 0..200)
    ) {
        let g = DynamicGraph::new(10_000, 4 << 20);
        for &(src, dst) in &edges {
            let _ = g.insert_edge(src, dst);
        }

        let mut buf = Vec::new();
        for src in 0..100u32 {
            g.neighbors_into(src, &mut buf);
            prop_assert!(is_sorted_strict(&buf),
                "neighbors of vertex {} not sorted: {:?}",
                src, buf.windows(2).find(|w| w[0] >= w[1]));
        }
    }

    // 3. degree equals length of deduplicated neighbor list
    #[test]
    fn degree_equals_unique_neighbors(
        edges in prop::collection::vec((0..50u32, 0..1000u32), 0..200)
    ) {
        let g = DynamicGraph::new(1000, 4 << 20);
        let mut expected: HashMap<u32, BTreeSet<u32>> = HashMap::new();

        for &(src, dst) in &edges {
            let _ = g.insert_edge(src, dst);
            expected.entry(src).or_default().insert(dst);
        }

        for (src, dsts) in &expected {
            let degree = g.degree(*src);
            let want = dsts.len();
            prop_assert_eq!(degree, want,
                "degree mismatch for vertex {}: got {}, expected {}",
                src, degree, want);
        }
    }

    // 4. inserting same edge twice doesn't change degree
    #[test]
    fn duplicate_edges_idempotent(src in 0..1000u32, dst in 0..1000u32) {
        let g = DynamicGraph::new(1000, 1 << 20);
        let _ = g.insert_edge(src, dst);
        let deg1 = g.degree(src);
        let was_new = g.insert_edge(src, dst).unwrap();
        let deg2 = g.degree(src);
        prop_assert!(!was_new,
            "second insert of ({},{}) claimed it was new", src, dst);
        prop_assert_eq!(deg1, deg2,
            "degree changed after duplicate insert: {} -> {}", deg1, deg2);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DynamicGraph concurrent invariant
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    // 5. spawn reader + writer threads, reader always sees sorted neighbors
    #[test]
    fn concurrent_invariant(
        vals in prop::collection::vec(0..10_000u32, 100..=500)
    ) {
        let g = Arc::new(DynamicGraph::new(10_000, 8 << 20));

        let g_writer = Arc::clone(&g);
        let writer = std::thread::spawn(move || {
            for &dst in &vals {
                let _ = g_writer.insert_edge(0, dst);
            }
        });

        let g_reader = Arc::clone(&g);
        let reader = std::thread::spawn(move || {
            let mut buf = Vec::new();
            for _ in 0..200 {
                g_reader.neighbors_into(0, &mut buf);
                // Must always see a consistent, sorted snapshot
                assert!(is_sorted_strict(&buf),
                    "reader saw unsorted neighbors: first bad pair at {:?}",
                    buf.windows(2).find(|w| w[0] >= w[1]));
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Deterministic stress test: 10K inserts
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn stress_10k_inserts() {
    let g = DynamicGraph::new(10_000, 8 << 20); // 8MB arena
    let mut rng = SmallRng::seed_from_u64(0xDEAD_BEEF);
    let mut expected: HashMap<u32, BTreeSet<u32>> = HashMap::new();

    for _ in 0..10_000 {
        let src = rng.random_range(0..1000u32);
        let dst = rng.random_range(0..10_000u32);
        let _ = g.insert_edge(src, dst);
        expected.entry(src).or_default().insert(dst);
    }

    // Verify every vertex that received edges
    for (src, dsts) in &expected {
        let mut buf = Vec::new();
        g.neighbors_into(*src, &mut buf);
        let actual: BTreeSet<u32> = buf.iter().copied().collect();
        assert_eq!(
            &actual, dsts,
            "mismatch for vertex {}: got {} neighbors, expected {}",
            src, actual.len(), dsts.len()
        );
        // Also verify sorted
        assert!(
            is_sorted_strict(&buf),
            "neighbors of vertex {} not sorted", src
        );
    }

    // Verify degree matches
    for (src, dsts) in &expected {
        assert_eq!(
            g.degree(*src),
            dsts.len(),
            "degree mismatch for vertex {}", src
        );
    }

    // Verify has_edge for every expected edge
    for (src, dsts) in &expected {
        for dst in dsts {
            assert!(
                g.has_edge(*src, *dst),
                "has_edge({}, {}) returned false", src, dst
            );
        }
    }
}
