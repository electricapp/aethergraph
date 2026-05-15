//! Loom model check for the `DynamicGraph::writer()` CAS-based single-writer
//! guard.
//!
//! This file does NOT use the real `DynamicGraph` because `loom`'s primitives
//! are not the same types as `std::sync::atomic`. It models the writer-guard
//! CAS in isolation and asserts the invariant: at most one writer guard can
//! exist at any moment across all interleavings explored by loom.
//!
//! Run with:
//!     cargo test -p aether-graph --features loom --test loom_writer_guard

#![cfg(feature = "loom")]

use loom::sync::Arc;
use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use loom::thread;

/// A minimal stand-in for `DynamicGraph`'s writer slot.
struct WriterSlot {
    locked: AtomicBool,
    /// Counts how many writers exist simultaneously. Loom asserts this never
    /// exceeds 1.
    live_writers: AtomicUsize,
}

impl WriterSlot {
    fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
            live_writers: AtomicUsize::new(0),
        }
    }

    fn try_acquire(this: &Arc<Self>) -> Option<WriterGuard> {
        this.locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()?;
        let prev = this.live_writers.fetch_add(1, Ordering::AcqRel);
        // The invariant we're checking with loom.
        assert_eq!(prev, 0, "two writers existed concurrently");
        Some(WriterGuard {
            slot: Arc::clone(this),
        })
    }
}

struct WriterGuard {
    slot: Arc<WriterSlot>,
}

impl Drop for WriterGuard {
    fn drop(&mut self) {
        self.slot.live_writers.fetch_sub(1, Ordering::AcqRel);
        self.slot.locked.store(false, Ordering::Release);
    }
}

#[test]
fn writer_is_mutually_exclusive_under_loom() {
    loom::model(|| {
        let slot = Arc::new(WriterSlot::new());
        let a = Arc::clone(&slot);
        let b = Arc::clone(&slot);

        let t1 = thread::spawn(move || {
            if let Some(g) = WriterSlot::try_acquire(&a) {
                drop(g);
            }
        });
        let t2 = thread::spawn(move || {
            if let Some(g) = WriterSlot::try_acquire(&b) {
                drop(g);
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}

#[test]
fn writer_can_reacquire_after_drop_under_loom() {
    loom::model(|| {
        let slot = Arc::new(WriterSlot::new());

        // First acquire releases via Drop at end of block.
        {
            let g = WriterSlot::try_acquire(&slot).expect("first acquire must succeed");
            drop(g);
        }

        // Spawn two threads racing to reacquire — at most one should succeed
        // at a time.
        let a = Arc::clone(&slot);
        let b = Arc::clone(&slot);
        let t1 = thread::spawn(move || {
            let _ = WriterSlot::try_acquire(&a);
        });
        let t2 = thread::spawn(move || {
            let _ = WriterSlot::try_acquire(&b);
        });
        t1.join().unwrap();
        t2.join().unwrap();
    });
}
