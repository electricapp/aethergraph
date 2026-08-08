//! SPSC ring wrappers for AF_XDP FILL/RX/TX/COMPLETION rings.
//!
//! These are *not* Treiber stacks — they are kernel-mmap'd single-producer
//! single-consumer rings with shared `AtomicU32` producer/consumer pointers
//! and a power-of-two mask for wraparound.
//!
//! Each side keeps a private mirror of its own index plus a cached view of
//! the peer's (the libbpf `cached_prod`/`cached_cons` pattern): batch
//! operations then touch the shared, kernel-visible atomics once per batch
//! — one acquire refresh when the cache runs out and one release publish
//! per advance — instead of re-loading them per descriptor.

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, Ordering};

/// Offsets within a single ring's mmap'd region (from `XDP_MMAP_OFFSETS`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct XdpRingOffset {
    pub producer: u64,
    pub consumer: u64,
    pub desc: u64,
    pub flags: u64,
}

/// Offsets for all four rings, returned by `getsockopt(XDP_MMAP_OFFSETS)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct XdpMmapOffsets {
    pub rx: XdpRingOffset,
    pub tx: XdpRingOffset,
    pub fr: XdpRingOffset, // fill ring
    pub cr: XdpRingOffset, // completion ring
}

/// RX/TX descriptor (index into UMEM + length).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RxTxDesc {
    /// Offset into the UMEM area.
    pub addr: u64,
    /// Length of the frame data.
    pub len: u32,
    /// Options (reserved, set to 0).
    pub options: u32,
}

/// FILL/COMPLETION descriptor (just a UMEM offset).
pub type UmemDesc = u64;

/// A kernel-mmap'd SPSC ring for AF_XDP.
///
/// Wraps raw pointers to the producer index, consumer index, and descriptor
/// array in the mmap'd region. The ring size must be a power of two;
/// `mask = size - 1` handles wraparound.
///
/// # Safety
/// The caller must ensure the mmap'd region remains valid for the lifetime
/// of this struct. The ring is SPSC — exactly one producer and one consumer
/// thread. The `Cell` mirrors are partitioned by side: `local_cons` /
/// `cached_prod` are touched only by consumer-side methods (`available`,
/// `peek`, `advance_consumer`), `local_prod` / `cached_cons` only by
/// producer-side methods (`free_slots`, `enqueue_at`, `advance_producer`).
/// Calling a side's methods from more than one thread is a data race.
pub struct XdpRing<T: Copy> {
    producer: *const AtomicU32,
    consumer: *const AtomicU32,
    /// Kernel-owned ring flags (carries `XDP_RING_NEED_WAKEUP`). Lives in the
    /// same mmap'd region as the producer/consumer pointers.
    flags: *const AtomicU32,
    ring: *mut T,
    mask: u32,
    /// Consumer thread's private consumer index (published on advance).
    local_cons: Cell<u32>,
    /// Consumer thread's cached view of the kernel's producer index,
    /// refreshed only when it says the ring is empty.
    cached_prod: Cell<u32>,
    /// Producer thread's private producer index (published on advance).
    local_prod: Cell<u32>,
    /// Producer thread's cached view of the kernel's consumer index,
    /// refreshed only when it says the ring is full.
    cached_cons: Cell<u32>,
}

// SAFETY: XdpRing pointers come from mmap and are accessed by one
// producer + one consumer thread; the shared indices are atomics and the
// Cell mirrors are partitioned per side (see the struct's Safety section).
unsafe impl<T: Copy> Send for XdpRing<T> {}
// SAFETY: same as Send — SPSC contract with side-partitioned mirrors.
unsafe impl<T: Copy> Sync for XdpRing<T> {}

impl<T: Copy> XdpRing<T> {
    /// Construct from raw mmap'd pointers.
    ///
    /// # Safety
    /// - `base` must point to a valid mmap'd region for this ring
    /// - `offsets` must contain correct offsets from `getsockopt(XDP_MMAP_OFFSETS)`
    /// - `ring_size` must be a power of two
    pub unsafe fn from_mmap(base: *mut u8, offsets: &XdpRingOffset, ring_size: u32) -> Self {
        debug_assert!(ring_size.is_power_of_two());
        // SAFETY: per the function's safety contract above.
        let producer = unsafe { base.add(offsets.producer as usize) } as *const AtomicU32;
        // SAFETY: per the function's safety contract above.
        let consumer = unsafe { base.add(offsets.consumer as usize) } as *const AtomicU32;
        // SAFETY: per the function's safety contract above.
        let flags = unsafe { base.add(offsets.flags as usize) } as *const AtomicU32;
        // SAFETY: per the function's safety contract above.
        let ring = unsafe { base.add(offsets.desc as usize) } as *mut T;
        // Seed the private mirrors from the ring's current state.
        // SAFETY: the pointers were just derived from the valid mapping.
        let initial_prod = unsafe { (*producer).load(Ordering::Acquire) };
        // SAFETY: same mapping.
        let initial_cons = unsafe { (*consumer).load(Ordering::Acquire) };
        Self {
            producer,
            consumer,
            flags,
            ring,
            mask: ring_size - 1,
            local_cons: Cell::new(initial_cons),
            cached_prod: Cell::new(initial_prod),
            local_prod: Cell::new(initial_prod),
            cached_cons: Cell::new(initial_cons),
        }
    }

    /// Whether the kernel has flagged this ring as needing a syscall wakeup.
    ///
    /// Zero-copy / busy-poll drivers stop draining the FILL ring (and stop
    /// pulling from TX) once they catch up, setting `XDP_RING_NEED_WAKEUP`.
    /// When this returns `true`, userspace must issue `recvfrom`/`sendto` on
    /// the socket fd to resume the driver; otherwise the busy-poll loop spins
    /// forever without the kernel ever moving frames.
    #[inline]
    pub fn needs_wakeup(&self) -> bool {
        // SAFETY: `flags` points into the ring's mmap'd region, valid for the
        // ring's lifetime.
        let flags = unsafe { (*self.flags).load(Ordering::Acquire) };
        flags & super::constants::XDP_RING_NEED_WAKEUP != 0
    }

    /// Number of entries available for consumption (consumer side).
    ///
    /// The kernel's producer index is re-read only when the cached view is
    /// exhausted, so a drain of N descriptors costs one acquire load, not N.
    #[inline]
    pub fn available(&self) -> u32 {
        let cons = self.local_cons.get();
        if self.cached_prod.get() == cons {
            // SAFETY: ring pointers are valid for the ring's lifetime.
            self.cached_prod
                .set(unsafe { (*self.producer).load(Ordering::Acquire) });
        }
        self.cached_prod.get().wrapping_sub(cons)
    }

    /// Peek at the entry at `consumer + offset` without advancing
    /// (consumer side). No shared-index access at all.
    ///
    /// # Safety
    /// Caller must ensure `offset < available()`.
    #[inline]
    pub unsafe fn peek(&self, offset: u32) -> T {
        let idx = (self.local_cons.get().wrapping_add(offset)) & self.mask;
        // SAFETY: caller guarantees `offset < available()`, so `idx` is in-bounds.
        let slot = unsafe { self.ring.add(idx as usize) };
        // SAFETY: `slot` is a valid pointer into the ring.
        unsafe { *slot }
    }

    /// Advance the consumer pointer by `count` (consumer side): one release
    /// store publishes the whole batch to the kernel.
    ///
    /// # Safety
    /// Caller must have consumed `count` entries via `peek`.
    #[inline]
    pub unsafe fn advance_consumer(&self, count: u32) {
        let cons = self.local_cons.get().wrapping_add(count);
        self.local_cons.set(cons);
        // SAFETY: ring pointers are valid for the ring's lifetime.
        unsafe { (*self.consumer).store(cons, Ordering::Release) };
    }

    /// Number of free slots for production (producer side).
    ///
    /// The kernel's consumer index is re-read only when the cached view
    /// says the ring is full.
    #[inline]
    pub fn free_slots(&self) -> u32 {
        let size = self.mask + 1;
        let prod = self.local_prod.get();
        let free = size.wrapping_sub(prod.wrapping_sub(self.cached_cons.get()));
        if free == 0 {
            // SAFETY: ring pointers are valid for the ring's lifetime.
            self.cached_cons
                .set(unsafe { (*self.consumer).load(Ordering::Acquire) });
            return size.wrapping_sub(prod.wrapping_sub(self.cached_cons.get()));
        }
        free
    }

    /// Write an entry at `producer + offset` without advancing (producer
    /// side). No shared-index access at all.
    ///
    /// # Safety
    /// Caller must ensure `offset < free_slots()`.
    #[inline]
    pub unsafe fn enqueue_at(&self, offset: u32, entry: T) {
        let idx = (self.local_prod.get().wrapping_add(offset)) & self.mask;
        // SAFETY: caller guarantees `offset < free_slots()`, so `idx` is in-bounds.
        let slot = unsafe { self.ring.add(idx as usize) };
        // SAFETY: `slot` is a valid mutable pointer into the ring.
        unsafe { *slot = entry };
    }

    /// Advance the producer pointer by `count` (producer side): one release
    /// store publishes the whole batch to the kernel.
    ///
    /// # Safety
    /// Caller must have written `count` entries via `enqueue_at`.
    #[inline]
    pub unsafe fn advance_producer(&self, count: u32) {
        let prod = self.local_prod.get().wrapping_add(count);
        self.local_prod.set(prod);
        // SAFETY: ring pointers are valid for the ring's lifetime.
        unsafe { (*self.producer).store(prod, Ordering::Release) };
    }
}
