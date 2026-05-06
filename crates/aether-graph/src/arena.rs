//! Bump-allocating arena for C-tree nodes.
//!
//! Pre-allocates a contiguous region at construction. Allocation is a
//! single atomic fetch_add — no locks, no syscalls, no fragmentation.
//! Deallocation is bulk (drop the entire arena).
//!
//! The arena is NOT thread-safe for allocation (single writer). It IS
//! safe for concurrent reads from previously-allocated nodes.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Fixed-capacity bump arena. Pre-allocates all memory upfront.
///
/// Nodes are allocated by bumping `offset`. Freed only when the entire
/// arena is dropped (epoch-based reclamation at the graph level).
pub struct Arena {
    /// Backing storage. 64-byte aligned for chunk cache-line access.
    data: UnsafeCell<Vec<u8>>,
    /// Next free byte offset. Only the writer advances this.
    offset: AtomicUsize,
    /// Total capacity in bytes.
    capacity: usize,
}

// SAFETY: The arena is single-writer (only the graph writer thread allocates).
// Concurrent readers access previously-allocated nodes which are immutable
// after creation (functional data structure — nodes are never modified in place).
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    /// Create an arena with `capacity` bytes, 64-byte aligned.
    pub fn new(capacity: usize) -> Self {
        // Allocate aligned to 64 bytes for cache-line chunks
        let layout = std::alloc::Layout::from_size_align(capacity, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let data = unsafe { Vec::from_raw_parts(ptr, capacity, capacity) };
        Self {
            data: UnsafeCell::new(data),
            offset: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Allocate `size` bytes with `align` alignment. Returns byte offset
    /// into the arena. Returns None if arena is full.
    ///
    /// # Safety
    /// Only the single writer thread should call this.
    #[inline]
    pub fn alloc(&self, size: usize, align: usize) -> Option<u32> {
        let current = self.offset.load(Ordering::Relaxed);
        // Align up
        let aligned = (current + align - 1) & !(align - 1);
        let new_offset = aligned + size;
        if new_offset > self.capacity {
            return None; // arena full
        }
        self.offset.store(new_offset, Ordering::Relaxed);
        Some(aligned as u32)
    }

    /// Allocate and write a value. Returns the offset.
    #[inline]
    pub fn alloc_write<T: Copy>(&self, val: T) -> Option<u32> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        let offset = self.alloc(size, align)?;
        unsafe {
            let ptr = self.ptr_at(offset);
            std::ptr::write(ptr as *mut T, val);
        }
        Some(offset)
    }

    /// Get a pointer to the data at `offset`.
    ///
    /// # Safety
    /// Offset must be within bounds and point to a valid, initialized value.
    #[inline(always)]
    pub unsafe fn ptr_at(&self, offset: u32) -> *const u8 {
        let data = &*self.data.get();
        data.as_ptr().add(offset as usize)
    }

    /// Get a typed reference at `offset`.
    ///
    /// # Safety
    /// Offset must point to a properly aligned, initialized `T`.
    #[inline(always)]
    pub unsafe fn get<T>(&self, offset: u32) -> &T {
        &*(self.ptr_at(offset) as *const T)
    }

    /// Bytes currently allocated.
    pub fn used(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }

    /// Total capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Reset the arena (reclaims all allocations). Writer only.
    ///
    /// # Safety
    /// Caller must ensure no readers hold references to arena data.
    pub unsafe fn reset(&self) {
        self.offset.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;

    #[test]
    fn basic_alloc() {
        let arena = Arena::new(4096);
        assert_eq!(arena.used(), 0);

        let off = arena.alloc(64, 64).unwrap();
        assert_eq!(off, 0);
        assert_eq!(arena.used(), 64);

        let off2 = arena.alloc(64, 64).unwrap();
        assert_eq!(off2, 64);
        assert_eq!(arena.used(), 128);
    }

    #[test]
    fn alloc_write_chunk() {
        let arena = Arena::new(4096);
        let chunk = Chunk::from_sorted(&[10, 20, 30]);

        let off = arena.alloc_write(chunk).unwrap();
        let read: &Chunk = unsafe { arena.get(off) };
        assert_eq!(read.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn arena_full_returns_none() {
        let arena = Arena::new(128);
        let _ = arena.alloc(64, 64).unwrap();
        let _ = arena.alloc(64, 64).unwrap();
        assert!(arena.alloc(64, 64).is_none());
    }

    #[test]
    fn alignment() {
        let arena = Arena::new(4096);
        // Alloc 1 byte, then 64-byte aligned
        let _ = arena.alloc(1, 1).unwrap();
        let off = arena.alloc(64, 64).unwrap();
        assert_eq!(off % 64, 0);
    }

    #[test]
    fn reset() {
        let arena = Arena::new(4096);
        let _ = arena.alloc(64, 64).unwrap();
        assert_eq!(arena.used(), 64);
        unsafe { arena.reset() };
        assert_eq!(arena.used(), 0);
    }
}
