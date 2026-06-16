//! miri smoke test for the ring's unsafe slot-pointer math and the free-list.
//!
//! Run under miri to check the raw-pointer slot addressing, the slot
//! read/write paths, and the lock-free index allocator for undefined behavior
//! (provenance, out-of-bounds, uninitialized reads):
//!     cargo +nightly miri test -p aether-mem --test miri_smoke
//!
//! Single-threaded by design — miri checks UB on the real code, loom checks
//! interleavings. The huge-page mmap path is compiled out under miri (see the
//! `not(miri)` gate in `ring.rs`), so construction takes the std-alloc path.

use aether_mem::{FreeList, SharedMemoryRing};

#[test]
fn ring_slot_roundtrip_is_ub_free() {
    let (ring, _failures) = SharedMemoryRing::new(4, 64, vec![]).expect("ring builds");
    // Acquire, write, read back, release — several cycles to exercise reuse and
    // the unsafe `slot_ptr` / `copy_from_slice` / `as_slice` paths.
    for round in 0..3u8 {
        let mut slot = ring.acquire_slot().expect("a slot is free");
        let payload = [round; 32];
        slot.copy_from_slice(&payload).expect("payload fits in slot");
        assert_eq!(slot.as_slice(), &payload);
        // RingSlot auto-releases on drop at the end of the iteration.
    }
}

#[test]
fn free_list_exhaustion_and_refill_is_ub_free() {
    let fl = FreeList::new(2);
    let a = fl.acquire().expect("slot a");
    let b = fl.acquire().expect("slot b");
    assert_ne!(a, b, "distinct acquisitions get distinct slots");
    assert!(fl.acquire().is_none(), "free list is exhausted");
    fl.release(a);
    let c = fl.acquire().expect("a slot is free again after release");
    assert_eq!(c, a, "LIFO free list hands back the just-released slot");
    fl.release(b);
    fl.release(c);
}
