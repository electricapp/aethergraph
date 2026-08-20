//! Loom model of the snapshot-pin / reclaim protocol.
//!
//! Models the arena's epoch-pinned reclamation in isolation (loom's
//! primitives are not std's): a reader holds a pin while reading a slot
//! through an UnsafeCell; the writer rewrites the slot only after
//! observing `min_pinned` at-or-past the slot's stamp. Loom explores all
//! interleavings and flags any access race — which is exactly what would
//! happen if the registry's Release store / reclaimer's Acquire load
//! pairing were weakened.
//!
//! Run with:
//!     cargo test -p aether-graph --features loom --test loom_pin_reclaim

#![cfg(feature = "loom")]

use loom::cell::UnsafeCell;
use loom::sync::atomic::{AtomicU64, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;

/// Minimal PinRegistry: epochs behind a mutex, cached min for the writer.
struct Registry {
    pins: Mutex<Vec<u64>>,
    min: AtomicU64,
}

impl Registry {
    fn new() -> Self {
        Self {
            pins: Mutex::new(Vec::new()),
            min: AtomicU64::new(u64::MAX),
        }
    }

    fn register(&self, epoch: u64) {
        let mut pins = self.pins.lock().unwrap();
        pins.push(epoch);
        let min = pins.iter().copied().min().unwrap_or(u64::MAX);
        self.min.store(min, Ordering::Release);
    }

    fn release(&self, epoch: u64) {
        let mut pins = self.pins.lock().unwrap();
        let i = pins.iter().position(|&e| e == epoch).expect("pinned");
        pins.swap_remove(i);
        let min = pins.iter().copied().min().unwrap_or(u64::MAX);
        self.min.store(min, Ordering::Release);
    }

    fn min_pinned(&self) -> u64 {
        self.min.load(Ordering::Acquire)
    }
}

/// A retired "slot": stamped epoch 2, rewritten by the writer once no pin
/// below 2 remains. Only a snapshot at epoch 1 can reach it, and the two
/// invariants the real system provides are modeled faithfully:
/// - the latest snapshot's epoch is always registered ("graph pin"), and
/// - acquire() clones latest under the same mutex publication uses, so a
///   reader only ever re-registers a currently-pinned epoch.
///
/// Loom verified both matter: letting the reader pin a stale epoch (a
/// clone outside the latest mutex) reproduces the rewrite race.
#[test]
fn pinned_reader_never_races_slot_rewrite() {
    loom::model(|| {
        let reg = Arc::new(Registry::new());
        // Publication of V1: graph pin at epoch 1, latest = 1.
        reg.register(1);
        let latest = Arc::new(Mutex::new(1u64));
        let slot = Arc::new(UnsafeCell::new(0u32));

        let reader = {
            let reg = Arc::clone(&reg);
            let latest = Arc::clone(&latest);
            let slot = Arc::clone(&slot);
            thread::spawn(move || {
                // acquire(): clone latest under its mutex — re-registers
                // an epoch that is provably still pinned. The guard must
                // span the register (loom flags the race if it doesn't).
                let pinned = {
                    let guard = latest.lock().unwrap();
                    let e = *guard;
                    reg.register(e);
                    e
                };
                // Only V1 reaches the retired slot; V2 was published
                // after its retirement.
                if pinned == 1 {
                    slot.with(|p| {
                        // SAFETY: pin at 1 blocks the stamp-2 rewrite.
                        let v = unsafe { *p };
                        assert_eq!(v, 0, "reader observed a rewritten slot");
                    });
                }
                reg.release(pinned);
            })
        };

        // Writer commit of V2, under the latest mutex like
        // publish_snapshot: register 2, swap latest, drop the V1 pin.
        {
            let mut l = latest.lock().unwrap();
            reg.register(2);
            *l = 2;
            reg.release(1);
        }
        // Reclaim: rewrite the stamp-2 slot once no pin below 2 remains.
        if reg.min_pinned() >= 2 {
            slot.with_mut(|p| {
                // SAFETY: no pin below the stamp — the slot is
                // unobservable. Loom verifies this claim.
                unsafe { *p = 0xDEAD };
            });
        }

        reader.join().unwrap();
    });
}
