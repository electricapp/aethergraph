//! SPSC ring wrappers for AF_XDP FILL/RX/TX/COMPLETION rings.
//!
//! These are *not* Treiber stacks — they are kernel-mmap'd single-producer
//! single-consumer rings with shared `AtomicU32` producer/consumer pointers
//! and a power-of-two mask for wraparound.

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
/// of this struct. The ring is SPSC — exactly one producer and one consumer.
pub struct XdpRing<T: Copy> {
    producer: *const AtomicU32,
    consumer: *const AtomicU32,
    ring: *mut T,
    mask: u32,
}

// SAFETY: XdpRing pointers come from mmap and are accessed by one
// producer + one consumer thread with atomic coordination.
unsafe impl<T: Copy> Send for XdpRing<T> {}
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
        Self {
            producer: base.add(offsets.producer as usize) as *const AtomicU32,
            consumer: base.add(offsets.consumer as usize) as *const AtomicU32,
            ring: base.add(offsets.desc as usize) as *mut T,
            mask: ring_size - 1,
        }
    }

    /// Number of entries available for consumption.
    #[inline]
    pub fn available(&self) -> u32 {
        // SAFETY: pointers are valid for the ring's lifetime
        let prod = unsafe { (*self.producer).load(Ordering::Acquire) };
        let cons = unsafe { (*self.consumer).load(Ordering::Acquire) };
        prod.wrapping_sub(cons)
    }

    /// Peek at the entry at `consumer + offset` without advancing.
    ///
    /// # Safety
    /// Caller must ensure `offset < available()`.
    #[inline]
    pub unsafe fn peek(&self, offset: u32) -> T {
        let cons = (*self.consumer).load(Ordering::Acquire);
        let idx = (cons.wrapping_add(offset)) & self.mask;
        *self.ring.add(idx as usize)
    }

    /// Advance the consumer pointer by `count`.
    ///
    /// # Safety
    /// Caller must have consumed `count` entries via `peek`.
    #[inline]
    pub unsafe fn advance_consumer(&self, count: u32) {
        let cons = (*self.consumer).load(Ordering::Relaxed);
        (*self.consumer).store(cons.wrapping_add(count), Ordering::Release);
    }

    /// Number of free slots for production.
    #[inline]
    pub fn free_slots(&self) -> u32 {
        let prod = unsafe { (*self.producer).load(Ordering::Acquire) };
        let cons = unsafe { (*self.consumer).load(Ordering::Acquire) };
        (self.mask + 1).wrapping_sub(prod.wrapping_sub(cons))
    }

    /// Write an entry at `producer + offset` without advancing.
    ///
    /// # Safety
    /// Caller must ensure `offset < free_slots()`.
    #[inline]
    pub unsafe fn enqueue_at(&self, offset: u32, entry: T) {
        let prod = (*self.producer).load(Ordering::Relaxed);
        let idx = (prod.wrapping_add(offset)) & self.mask;
        *self.ring.add(idx as usize) = entry;
    }

    /// Advance the producer pointer by `count`.
    ///
    /// # Safety
    /// Caller must have written `count` entries via `enqueue_at`.
    #[inline]
    pub unsafe fn advance_producer(&self, count: u32) {
        let prod = (*self.producer).load(Ordering::Relaxed);
        (*self.producer).store(prod.wrapping_add(count), Ordering::Release);
    }
}
