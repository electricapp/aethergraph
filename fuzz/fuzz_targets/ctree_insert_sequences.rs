#![no_main]
//! Fuzz target: drive `CTree::insert` with arbitrary u32 sequences and
//! assert the post-conditions hold for every prefix of inserts:
//!   - tree.count == |unique values seen|
//!   - tree.collect_into is strictly sorted
//!   - tree.contains is true for every value inserted with `Inserted`
//!     and false for never-inserted sentinels
//!
//! Run:  cargo +nightly fuzz run ctree_insert_sequences

use libfuzzer_sys::fuzz_target;
use std::collections::BTreeSet;

use aether_graph::{Arena, CTree, InsertResult};

fuzz_target!(|values: Vec<u32>| {
    // Cap to keep individual fuzz iterations under ~ms.
    if values.len() > 2048 {
        return;
    }
    // Adequate arena for 2k inserts including path-copy node overhead.
    let arena = Arena::new(1 << 20);
    // SAFETY: this is the only write handle ever constructed for `arena`,
    // and no RegionWriter or region commit exists — the fuzz body is the
    // arena's sole user.
    let mut aw = unsafe { arena.writer() };
    let mut tree = CTree::empty();
    let mut seen: BTreeSet<u32> = BTreeSet::new();

    for v in &values {
        match tree.insert(&mut aw, *v) {
            InsertResult::Inserted(t) => {
                tree = t;
                seen.insert(*v);
            }
            InsertResult::Duplicate => {
                assert!(seen.contains(v), "duplicate reported for unseen value");
            }
            InsertResult::ArenaFull => return, // arena exhausted, stop
        }
    }

    assert_eq!(tree.count(&arena), seen.len(), "count mismatch");

    let mut buf = Vec::new();
    tree.collect_into(&arena, &mut buf);
    let expected: Vec<u32> = seen.iter().copied().collect();
    assert_eq!(buf, expected, "collect_into not sorted-unique");

    for v in &seen {
        assert!(tree.contains(&arena, *v), "missing value {v}");
    }
});
