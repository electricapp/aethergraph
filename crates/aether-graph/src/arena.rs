//! Bump-allocating arena for C-tree nodes.
//!
//! Pre-allocates a contiguous region at construction. Allocation is a
//! single atomic fetch_add — no locks, no syscalls, no fragmentation.
//! Deallocation is bulk (drop the entire arena).
//!
//! The arena is NOT thread-safe for allocation (single writer). It IS
//! safe for concurrent reads from previously-allocated nodes.

use std::alloc::Layout;
use std::cell::UnsafeCell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Fixed-capacity bump arena. Pre-allocates all memory upfront.
///
/// Nodes are allocated by bumping `offset`. Freed only when the entire
/// arena is dropped (epoch-based reclamation at the graph level).
///
/// The backing buffer is a raw `NonNull<u8>` paired with the `Layout` it was
/// allocated with. `Vec<u8>` is not used because `Vec` would deallocate with
/// `align_of::<u8>() = 1`, mismatching the 64-byte alignment we need for
/// cache-line chunks and producing UB at drop.
pub struct Arena {
    /// 64-byte aligned backing storage.
    ptr: UnsafeCell<NonNull<u8>>,
    /// Layout the buffer was allocated with — needed for matched dealloc.
    layout: Layout,
    /// Next free byte offset. Only the writer advances this.
    offset: AtomicUsize,
    /// Total capacity in bytes.
    capacity: usize,
}

// SAFETY: The arena is single-writer (only the graph writer thread allocates).
// Concurrent readers access previously-allocated nodes which are immutable
// after creation (functional data structure — nodes are never modified in place).
unsafe impl Send for Arena {}
// SAFETY: see Send impl above — single writer + immutable nodes after allocation
// allow concurrent shared access from any thread.
unsafe impl Sync for Arena {}

impl Arena {
    /// Create an arena with `capacity` bytes, 64-byte aligned.
    ///
    /// # Panics
    /// Panics if `capacity == 0` or `capacity > u32::MAX` (offsets are u32).
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Arena capacity must be > 0");
        // Offsets returned by alloc are u32; arenas larger than u32::MAX would
        // truncate silently and corrupt readers. Cap at u32::MAX up front.
        assert!(
            capacity <= u32::MAX as usize,
            "Arena capacity {capacity} exceeds u32::MAX (offsets are u32)"
        );
        let layout = Layout::from_size_align(capacity, 64)
            .expect("Layout: capacity > 0, alignment is power of two");
        // SAFETY: layout has non-zero size (capacity > 0); alloc_zeroed
        // initializes the buffer so reads of un-allocated bytes are
        // well-defined zero.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(raw) else {
            std::alloc::handle_alloc_error(layout);
        };
        Self {
            ptr: UnsafeCell::new(ptr),
            layout,
            offset: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Allocate `size` bytes with `align` alignment. Returns byte offset
    /// into the arena. Returns None if arena is full.
    ///
    /// # Safety
    /// Single-writer invariant. The bump cursor is updated via a non-atomic
    /// read-modify-write; concurrent callers produce overlapping allocations
    /// and silent corruption. Callers in `aether-graph` go through
    /// `DynamicGraph::writer()`, which holds a runtime-enforced single-writer
    /// guard for the lifetime of allocation.
    #[inline]
    pub unsafe fn alloc(&self, size: usize, align: usize) -> Option<u32> {
        debug_assert!(align.is_power_of_two(), "alignment must be power of two");
        let current = self.offset.load(Ordering::Relaxed);
        // Align up — checked to catch malformed (size, align) inputs.
        let aligned = current.checked_add(align - 1)? & !(align - 1);
        let new_offset = aligned.checked_add(size)?;
        if new_offset > self.capacity {
            return None; // arena full
        }
        self.offset.store(new_offset, Ordering::Relaxed);
        // Capacity ≤ u32::MAX (enforced in `new`), so this cast cannot truncate.
        Some(aligned as u32)
    }

    /// Allocate and write a value. Returns the offset.
    ///
    /// # Safety
    /// Single-writer invariant: only one thread may call this concurrently.
    /// See [`Arena::alloc`].
    #[inline]
    pub unsafe fn alloc_write<T: Copy>(&self, val: T) -> Option<u32> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        // SAFETY: caller upholds the single-writer invariant.
        let offset = unsafe { self.alloc(size, align)? };
        // SAFETY: alloc returned a fresh, in-bounds offset of `size` bytes aligned for T.
        let ptr = unsafe { self.ptr_at(offset) };
        // SAFETY: ptr points to `size_of::<T>()` bytes of uninitialized arena storage
        // aligned for T (alignment guaranteed by alloc above); the writer thread is single.
        unsafe {
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
        // SAFETY: UnsafeCell deref reads the stable base pointer set in `new`;
        // single-writer invariant means no &mut alias exists while readers
        // hold &Arena.
        let base = unsafe { (*self.ptr.get()).as_ptr() };
        // SAFETY: caller asserts offset is within the allocated capacity, so
        // the resulting pointer stays inside the buffer.
        unsafe { base.add(offset as usize) as *const u8 }
    }

    /// Get a typed reference at `offset`.
    ///
    /// # Safety
    /// Offset must point to a properly aligned, initialized `T`.
    #[inline(always)]
    pub unsafe fn get<T>(&self, offset: u32) -> &T {
        // SAFETY: caller asserts offset points to a properly aligned, initialized T;
        // ptr_at returns a pointer into the immutable backing region.
        let ptr = unsafe { self.ptr_at(offset) };
        // SAFETY: caller-asserted invariants make `ptr as *const T` dereferenceable.
        unsafe { &*(ptr as *const T) }
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

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: read of the NonNull<u8> through UnsafeCell — Drop has &mut self
        // so no other reference exists.
        let ptr = unsafe { (*self.ptr.get()).as_ptr() };
        // SAFETY: `ptr` was returned by `alloc_zeroed(self.layout)` in `new`,
        // we own it exclusively here, and the layout matches.
        unsafe {
            std::alloc::dealloc(ptr, self.layout);
        }
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

        // SAFETY: test is single-threaded.
        let off = unsafe { arena.alloc(64, 64) }.unwrap();
        assert_eq!(off, 0);
        assert_eq!(arena.used(), 64);

        // SAFETY: test is single-threaded.
        let off2 = unsafe { arena.alloc(64, 64) }.unwrap();
        assert_eq!(off2, 64);
        assert_eq!(arena.used(), 128);
    }

    #[test]
    fn alloc_write_chunk() {
        let arena = Arena::new(4096);
        let chunk = Chunk::from_sorted(&[10, 20, 30]);

        // SAFETY: test is single-threaded.
        let off = unsafe { arena.alloc_write(chunk) }.unwrap();
        // SAFETY: off was just returned by alloc_write::<Chunk>, so it points to a valid Chunk.
        let read: &Chunk = unsafe { arena.get(off) };
        assert_eq!(read.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn arena_full_returns_none() {
        let arena = Arena::new(128);
        // SAFETY: test is single-threaded.
        let _ = unsafe { arena.alloc(64, 64) }.unwrap();
        // SAFETY: test is single-threaded.
        let _ = unsafe { arena.alloc(64, 64) }.unwrap();
        // SAFETY: test is single-threaded.
        assert!(unsafe { arena.alloc(64, 64) }.is_none());
    }

    #[test]
    fn alignment() {
        let arena = Arena::new(4096);
        // SAFETY: test is single-threaded.
        let _ = unsafe { arena.alloc(1, 1) }.unwrap();
        // SAFETY: test is single-threaded.
        let off = unsafe { arena.alloc(64, 64) }.unwrap();
        assert_eq!(off % 64, 0);
    }

    #[test]
    fn reset() {
        let arena = Arena::new(4096);
        // SAFETY: test is single-threaded.
        let _ = unsafe { arena.alloc(64, 64) }.unwrap();
        assert_eq!(arena.used(), 64);
        // SAFETY: test is single-threaded.
        unsafe { arena.reset() };
        assert_eq!(arena.used(), 0);
    }

    #[test]
    #[should_panic(expected = "Arena capacity must be > 0")]
    fn zero_capacity_panics() {
        let _ = Arena::new(0);
    }
}
