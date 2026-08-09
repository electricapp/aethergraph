//! Loom model check for the free-list — the lock-free index allocator.
//!
//! This drives the *real* shipped code: under `--cfg loom`, the atomics
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
//!     RUSTFLAGS="--cfg loom" cargo test -p aether-mem --test loom_lockfree --release

#![cfg(loom)]

use aether_mem::FreeList;
use loom::cell::UnsafeCell;
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

/// The ABA guard: a victim whose `acquire` stalls between its head load and
/// its CAS must fail that CAS once the stack has mutated underneath it, even
/// if the head index has come back around. The reachable bad history needs
/// three interposed mutations — peer A pops the old head, peer B pops the
/// victim's stale `next` and holds it, peer A pushes the old head back — after
/// which an untagged CAS would succeed and install the stale `next` (B's
/// still-held slot) as the new head, so peer A's second acquire double-leases
/// it and the `leased` flags trip. The u32 tag in the packed head makes the
/// stale CAS fail instead.
///
/// The bad history is reachable with three preemptions (victim stalls mid-
/// acquire; A pauses after its first acquire for B to pop; A pauses after its
/// push for the victim's stale CAS — B's exit and the victim's yield are free
/// switches), so `preemption_bound = 3` covers it while keeping the three-
/// thread search space tractable.
#[test]
fn free_list_aba_tag_blocks_stale_cas() {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);
    builder.check(|| {
        let fl = Arc::new(FreeList::new(3));
        let leased = Arc::new((0..3).map(|_| AtomicBool::new(false)).collect::<Vec<_>>());

        // Victim: a single acquire whose CAS loom can stall arbitrarily long,
        // then a release. Three slots and at most two peer holders mean every
        // acquire in this model must succeed.
        let victim = {
            let fl = Arc::clone(&fl);
            thread::spawn(move || {
                let i = fl.acquire().expect("victim: a slot is free");
                thread::yield_now();
                fl.release(i);
            })
        };

        // Peer A: two acquire+release cycles. The first supplies the pop/push
        // traffic (pop, push back); the second is the acquire that would
        // receive a double-leased slot if the stale CAS were allowed through.
        let peer_a = {
            let fl = Arc::clone(&fl);
            let leased = Arc::clone(&leased);
            thread::spawn(move || {
                for _ in 0..2 {
                    let i = fl.acquire().expect("peer A: a slot is free");
                    assert!(
                        !leased[i].swap(true, Ordering::AcqRel),
                        "slot {i} leased to two owners at once"
                    );
                    leased[i].store(false, Ordering::Release);
                    fl.release(i);
                }
            })
        };

        // Peer B: acquires once and holds across the joins, so its lease
        // overlaps peer A's second acquire in every schedule.
        let peer_b = {
            let fl = Arc::clone(&fl);
            let leased = Arc::clone(&leased);
            thread::spawn(move || {
                let i = fl.acquire().expect("peer B: a slot is free");
                assert!(
                    !leased[i].swap(true, Ordering::AcqRel),
                    "slot {i} leased to two owners at once"
                );
                i
            })
        };

        victim.join().unwrap();
        peer_a.join().unwrap();
        let held = peer_b.join().unwrap();
        leased[held].store(false, Ordering::Release);
        fl.release(held);

        // The stack must still hold each index exactly once — a corrupted
        // head (self-loop or sentinel) would surface here.
        let mut all: Vec<usize> = (0..3)
            .map(|_| fl.acquire().expect("slot missing from free list"))
            .collect();
        all.sort_unstable();
        assert_eq!(all, vec![0, 1, 2]);
        assert!(fl.acquire().is_none());
        for i in all {
            fl.release(i);
        }
    });
}

/// A batched release must splice its private chain onto the stack in one
/// tagged CAS. An acquirer racing the splice contends on the same head —
/// its own CAS carries the ABA tag, so a stale head observed across the
/// splice fails instead of installing a spliced-away index. The acquirer
/// releases via the singleton path (itself a one-element `release_many`),
/// so both entry points hit the shared machinery in one model.
#[test]
fn release_many_splice_races_acquire() {
    loom::model(|| {
        let fl = Arc::new(FreeList::new(4));
        // Lease two slots up front; the releaser splices both back at once.
        let a = fl.acquire().expect("slot free at init");
        let b = fl.acquire().expect("slot free at init");

        let releaser = {
            let fl = Arc::clone(&fl);
            thread::spawn(move || fl.release_many(&[a, b]))
        };
        let acquirer = {
            let fl = Arc::clone(&fl);
            thread::spawn(move || {
                // Two slots sit outside the batch, so this acquire must
                // succeed whichever side of the splice CAS it lands on.
                let i = fl.acquire().expect("a slot is free");
                fl.release(i);
            })
        };
        releaser.join().unwrap();
        acquirer.join().unwrap();

        // The stack must hold each index exactly once afterwards — a torn
        // splice (lost tail, self-loop, duplicated head) would surface here.
        let mut all: Vec<usize> = (0..4)
            .map(|_| fl.acquire().expect("slot missing from free list"))
            .collect();
        all.sort_unstable();
        assert_eq!(all, vec![0, 1, 2, 3]);
        assert!(fl.acquire().is_none());
        for i in all {
            fl.release(i);
        }
    });
}

/// One payload cell per slot. `Sync` because the free-list lease protocol
/// guarantees one holder at a time — exactly the property loom verifies.
struct SlotPayload(UnsafeCell<u64>);

// SAFETY: access is confined to the current lease holder; loom's UnsafeCell
// reports any pair of accesses the free-list handoff fails to order.
unsafe impl Send for SlotPayload {}
// SAFETY: same rationale as Send.
unsafe impl Sync for SlotPayload {}

/// Data handoff through a slot: writes made under one lease must be visible
/// to (and race-free with) the next holder. Each worker increments a plain
/// non-atomic counter in the single slot; the release-CAS → acquire-load edge
/// on the free-list head is the only thing ordering the two increments, and
/// loom's `UnsafeCell` flags the model if that edge is missing.
#[test]
fn slot_payload_handoff_is_race_free() {
    loom::model(|| {
        let fl = Arc::new(FreeList::new(1));
        let payload = Arc::new(SlotPayload(UnsafeCell::new(0u64)));

        let worker = |fl: Arc<FreeList>, payload: Arc<SlotPayload>| {
            move || {
                let Some(i) = fl.acquire() else {
                    // The single slot is currently leased by the other worker.
                    return 0u64;
                };
                // SAFETY: this worker holds the only lease on the slot; the
                // prior holder's release happens-before this acquire.
                payload.0.with_mut(|p| unsafe { *p += 1 });
                fl.release(i);
                1
            }
        };

        let t1 = thread::spawn(worker(Arc::clone(&fl), Arc::clone(&payload)));
        let t2 = thread::spawn(worker(Arc::clone(&fl), Arc::clone(&payload)));
        let increments = t1.join().unwrap() + t2.join().unwrap();

        // Both workers have joined, so their writes are ordered before this
        // read and none may be lost.
        payload.0.with(|p| {
            // SAFETY: no worker is running; the cell is quiescent.
            assert_eq!(unsafe { *p }, increments, "an increment was lost");
        });
    });
}
