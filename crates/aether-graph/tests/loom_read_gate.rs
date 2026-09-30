//! Loom model of the arena's two-phase reader gate.
//!
//! Models the gate in isolation (loom's primitives are not std's): readers
//! enter the bucket named by the phase, load the published root, read the
//! slot it names through an UnsafeCell, and leave the bucket they entered.
//! The writer publishes a fresh slot, then rewrites the superseded one only
//! once the phase has advanced twice past the unpublishing store. Loom
//! flags any read the rewrite races.
//!
//! Every reader shares one stripe — the case an anonymous entry/exit count
//! gets wrong, since a later reader's exit can pay off an earlier reader's
//! entry. Loom treats SeqCst accesses as AcqRel, so the model spells the
//! production code's SeqCst entry RMW + SeqCst root load as a SeqCst fence
//! between them; the two are equivalent for the store-buffer pairing with
//! the writer's fence.
//!
//! Run with:
//!     cargo test -p aether-graph --features loom --test loom_read_gate

#![cfg(feature = "loom")]

use loom::cell::UnsafeCell;
use loom::sync::Arc;
use loom::sync::atomic::{AtomicU32, AtomicU64, Ordering, fence};
use loom::thread;

const LIVE: u32 = 7;
const DEAD: u32 = 0xDEAD;

struct World {
    phase: AtomicU64,
    /// One stripe's live-reader counts, one per phase bucket.
    active: [AtomicU64; 2],
    root: AtomicU32,
    slots: [UnsafeCell<u32>; 2],
}

impl World {
    fn new() -> Self {
        Self {
            phase: AtomicU64::new(0),
            active: [AtomicU64::new(0), AtomicU64::new(0)],
            root: AtomicU32::new(0),
            slots: [UnsafeCell::new(LIVE), UnsafeCell::new(0)],
        }
    }

    fn enter(&self) -> usize {
        let bucket = (self.phase.load(Ordering::Relaxed) & 1) as usize;
        self.active[bucket].fetch_add(1, Ordering::AcqRel);
        fence(Ordering::SeqCst);
        bucket
    }

    fn exit(&self, bucket: usize) {
        self.active[bucket].fetch_sub(1, Ordering::Release);
    }

    fn read_published(&self) {
        let r = self.root.load(Ordering::Acquire) as usize;
        self.slots[r].with(|p| {
            // SAFETY: the gate keeps the slot from being rewritten while
            // this reader is inside it. Loom verifies this claim.
            let v = unsafe { *p };
            assert_eq!(v, LIVE, "reader observed a recycled slot");
        });
    }

    /// Writer-only phase advance, as in `ReadGate::advance`.
    fn advance(&self, phase: &mut u64) {
        fence(Ordering::SeqCst);
        for _ in 0..2 {
            let next = ((*phase + 1) & 1) as usize;
            if self.active[next].load(Ordering::Acquire) != 0 {
                break;
            }
            *phase += 1;
            self.phase.store(*phase, Ordering::Relaxed);
        }
    }

    /// Publish slot 1, retire slot 0, and rewrite it if the gate clears
    /// within a bounded number of attempts.
    fn write(&self) {
        self.slots[1].with_mut(|p| {
            // SAFETY: slot 1 is unpublished until the root store below.
            unsafe { *p = LIVE };
        });
        self.root.store(1, Ordering::Release);
        let mut phase = self.phase.load(Ordering::Relaxed);
        let grace = phase + 2;
        for _ in 0..3 {
            self.advance(&mut phase);
            if phase >= grace {
                self.slots[0].with_mut(|p| {
                    // SAFETY: two advances drained every reader that could
                    // have loaded the old root. Loom verifies this claim.
                    unsafe { *p = DEAD };
                });
                return;
            }
            thread::yield_now();
        }
    }
}

fn model(f: impl Fn() + Sync + Send + 'static) {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);
    builder.check(f);
}

#[test]
fn later_reader_exit_does_not_release_earlier_reader() {
    model(|| {
        let world = Arc::new(World::new());
        let early = {
            let world = Arc::clone(&world);
            thread::spawn(move || {
                let b = world.enter();
                world.read_published();
                world.exit(b);
            })
        };
        let late = {
            let world = Arc::clone(&world);
            thread::spawn(move || {
                let b = world.enter();
                world.exit(b);
            })
        };
        world.write();
        early.join().unwrap();
        late.join().unwrap();
    });
}

#[test]
fn nested_guard_exit_does_not_release_outer_guard() {
    model(|| {
        let world = Arc::new(World::new());
        let reader = {
            let world = Arc::clone(&world);
            thread::spawn(move || {
                let outer = world.enter();
                let r = world.root.load(Ordering::Acquire) as usize;
                let inner = world.enter();
                world.exit(inner);
                world.slots[r].with(|p| {
                    // SAFETY: the outer guard is still live. Loom
                    // verifies this claim.
                    let v = unsafe { *p };
                    assert_eq!(v, LIVE, "outer guard observed a recycled slot");
                });
                world.exit(outer);
            })
        };
        world.write();
        reader.join().unwrap();
    });
}
