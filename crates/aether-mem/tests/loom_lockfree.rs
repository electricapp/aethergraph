//! Loom model check for the free-list — the lock-free index allocator.
//!
//! This drives the *real* shipped code: under the `loom` feature, the atomics
//! inside [`aether_mem::FreeList`] are loom's, so `acquire`/`release` here are
//! the exact functions production uses (via `SharedMemoryRing::acquire_index` /
//! `release_index`), not a transcription. There is no model to drift from prod.
//!
//! Deliberately NOT covered here: the `FeatureTable` seqlock. Its reader
//! validates a non-atomic payload against a version counter, which is a data
//! race in the C++11 model loom implements — a faithful model fails loom, and a
//! model rewritten to pass is no longer the shipped algorithm. The production
//! seqlock is correct on real hardware because the writer's first op is a
//! LOCK-prefixed RMW that is globally ordered (see the argument in
//! `aether_stream::feature_table::FeatureTable::read_node`), which loom cannot
//! represent. It is verified by that reasoning plus a real-threads stress test.
//!
//! Run with:
//!     cargo test -p aether-mem --features loom --test loom_lockfree

#![cfg(feature = "loom")]

use aether_mem::FreeList;
use loom::sync::Arc;
use loom::sync::atomic::{AtomicBool, Ordering};
use loom::thread;

/// Concurrent acquirers of the real `FreeList` must never both be handed the
/// same slot index. The `leased` flags are test-only bookkeeping that trips if
/// two owners ever hold one index at once.
#[test]
fn free_list_never_double_leases() {
    loom::model(|| {
        let fl = Arc::new(FreeList::new(2));
        let leased = Arc::new((0..2).map(|_| AtomicBool::new(false)).collect::<Vec<_>>());

        let worker = |fl: Arc<FreeList>, leased: Arc<Vec<AtomicBool>>| {
            move || {
                if let Some(i) = fl.acquire() {
                    let was_leased = leased[i].swap(true, Ordering::AcqRel);
                    assert!(!was_leased, "slot {i} leased to two owners at once");
                    leased[i].store(false, Ordering::Release);
                    fl.release(i);
                }
            }
        };

        let t1 = thread::spawn(worker(Arc::clone(&fl), Arc::clone(&leased)));
        let t2 = thread::spawn(worker(Arc::clone(&fl), Arc::clone(&leased)));
        t1.join().unwrap();
        t2.join().unwrap();
    });
}
