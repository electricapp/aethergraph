//! Bump-allocating arena for C-tree nodes.
//!
//! Pre-allocates a contiguous region at construction. Allocation is a
//! single atomic fetch_add — no locks, no syscalls, no fragmentation.
//! Deallocation is bulk (drop the entire arena).
//!
//! The arena is NOT thread-safe for allocation (single writer). It IS
//! safe for concurrent reads from previously-allocated nodes.

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Pads the writer-side bump cursor onto its own cache line. Every reader
/// node-dereference loads `Arena::ptr`; without this, the writer's 2-5
/// `offset` stores per edge insert would keep invalidating the line that
/// carries `ptr`/`capacity` in every reader's cache — a coherence miss per
/// node visited instead of an L1 hit. 128 bytes covers the adjacent-line
/// prefetcher on x86_64 and Apple Silicon's 128-byte granules.
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), repr(align(128)))]
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    repr(align(64))
)]
struct CachePadded<T>(T);

/// Fixed-capacity bump arena. Pre-allocates all memory upfront.
///
/// Nodes are allocated by bumping `offset`. Individual nodes are never
/// freed; space is reclaimed in bulk by dropping the arena.
///
/// The backing buffer is a raw `NonNull<u8>` paired with the `Layout` it was
/// allocated with. `Vec<u8>` is not used because `Vec` would deallocate with
/// `align_of::<u8>() = 1`, mismatching the 64-byte alignment we need for
/// cache-line chunks and producing UB at drop.
pub struct Arena {
    /// 64-byte aligned backing storage. Set once in `new` and never
    /// reassigned — a plain field (not `UnsafeCell`) so the compiler may
    /// hoist the base-pointer load out of node-walk loops.
    ptr: NonNull<u8>,
    /// Layout the buffer was allocated with — needed for matched dealloc.
    layout: Layout,
    /// Total capacity in bytes.
    capacity: usize,
    /// Next free byte offset. Only the writer advances this; isolated on
    /// its own cache line so writer bumps don't evict the read-mostly
    /// fields above from reader caches.
    offset: CachePadded<AtomicUsize>,
}

// SAFETY: The arena is single-writer (only the graph writer thread allocates).
// Concurrent readers access previously-allocated nodes which are immutable
// after creation (functional data structure — nodes are never modified in place).
unsafe impl Send for Arena {}
// SAFETY: see Send impl above — single writer + immutable nodes after allocation
// allow concurrent shared access from any thread.
unsafe impl Sync for Arena {}

impl Arena {
    /// Largest legal arena capacity: 2 GiB.
    ///
    /// Offsets are u32 and the C-tree steals bit 31 as a leaf/interior
    /// tag, so any offset at or above `1 << 31` would be misread as a
    /// tagged interior node — silent corruption. Capping capacity here
    /// makes the collision impossible.
    pub const MAX_CAPACITY: usize = 1 << 31;

    /// Create an arena with `capacity` bytes, 64-byte aligned.
    ///
    /// # Panics
    /// Panics if `capacity == 0` or `capacity > Self::MAX_CAPACITY`
    /// (offsets are u32 with bit 31 reserved as a node-type tag).
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Arena capacity must be > 0");
        assert!(
            capacity <= Self::MAX_CAPACITY,
            "Arena capacity {capacity} exceeds MAX_CAPACITY {} (offsets are u32 with a tag bit)",
            Self::MAX_CAPACITY
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
            ptr,
            layout,
            capacity,
            offset: CachePadded(AtomicUsize::new(0)),
        }
    }

    /// Allocate `size` bytes with `align` alignment. Returns byte offset
    /// into the arena. Returns None if the arena is full or `size == 0` —
    /// a zero-size allocation at an exactly-full 2 GiB arena would return
    /// offset `1 << 31`, colliding with the C-tree's bit-31 tag.
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
        if size == 0 {
            return None;
        }
        let current = self.offset.0.load(Ordering::Relaxed);
        // Align up — checked to catch malformed (size, align) inputs.
        let aligned = current.checked_add(align - 1)? & !(align - 1);
        let new_offset = aligned.checked_add(size)?;
        if new_offset > self.capacity {
            return None; // arena full
        }
        self.offset.0.store(new_offset, Ordering::Relaxed);
        // Capacity ≤ MAX_CAPACITY = 2^31 (enforced in `new`), so the cast
        // cannot truncate and bit 31 is never set on a returned offset.
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
        // SAFETY: caller asserts offset is within the allocated capacity, so
        // the resulting pointer stays inside the buffer.
        unsafe { self.ptr.as_ptr().add(offset as usize) as *const u8 }
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
        self.offset.0.load(Ordering::Relaxed)
    }

    /// Total capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was returned by `alloc_zeroed(self.layout)` in
        // `new`, we own it exclusively here, and the layout matches.
        unsafe {
            std::alloc::dealloc(self.ptr.as_ptr(), self.layout);
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
    fn zero_size_alloc_returns_none() {
        let arena = Arena::new(4096);
        // SAFETY: test is single-threaded.
        assert!(unsafe { arena.alloc(0, 1) }.is_none());
        assert_eq!(arena.used(), 0);
    }

    #[test]
    #[should_panic(expected = "Arena capacity must be > 0")]
    fn zero_capacity_panics() {
        let _ = Arena::new(0);
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_CAPACITY")]
    fn over_max_capacity_panics() {
        // The assert fires before any allocation happens, so this test
        // does not actually try to reserve >2 GiB.
        let _ = Arena::new(Arena::MAX_CAPACITY + 1);
    }
}
