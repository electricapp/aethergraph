//! Lock-free slab allocator with pre-allocated, page-aligned slots.
//!
//! # Architecture
//! Pre-allocates a contiguous memory region divided into fixed-size slots.
//! Slots are assigned from a lock-free free list (ABA-tagged Treiber stack).
//! A slot is only reused after the completion callback releases it, so
//! out-of-order completion is safe.
//!
//! # Safety
//! - Backpressure prevents data races (slot reuse only after completion)
//! - Static lifetime - allocated at startup, freed at shutdown
//! - slot_count must be power of two (handles usize overflow correctly)
//! - slot_size is page-aligned (4KB) for alignment
//!
//! # Extensibility
//! Post-allocation hooks ([`MemoryHook`]) allow callers to register memory
//! with external systems (e.g. CUDA host pinning, mlock) without coupling
//! the allocator to any specific domain.

use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Hook called after allocation and before deallocation.
///
/// Allows callers to register memory with external systems (CUDA pinning,
/// mlock, etc.) without the ring buffer knowing about those systems.
///
/// Hooks run only at startup/shutdown — never in the hot path.
pub trait MemoryHook: Send + Sync {
    /// Called once after the memory region is allocated and pre-faulted.
    /// Returns `true` if registration succeeded.
    fn on_alloc(&self, ptr: *mut u8, size: usize) -> bool;

    /// Called once before the memory region is freed.
    fn on_dealloc(&self, ptr: *mut u8, size: usize);
}

/// Thread-safe pointer wrapper for shared memory regions.
///
/// Wraps `NonNull<u8>` and implements both `Send` and `Sync`, making it safe to
/// share references to the underlying memory across threads.
///
/// # Safety
/// The caller must ensure the pointed-to memory:
/// - Remains valid for the lifetime of all references
/// - Has appropriate synchronization for concurrent access (e.g., backpressure)
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
struct SyncPtr(NonNull<u8>);

// SAFETY: SyncPtr is designed for memory regions where:
// - The memory is owned/managed by the containing struct (SharedMemoryRing)
// - Concurrent access is controlled externally (backpressure + atomic counter)
// - The memory lifetime exceeds all references (static allocation at startup)
unsafe impl Send for SyncPtr {}
// SAFETY: Same rationale as Send - the pointed memory is managed by SharedMemoryRing
// with external synchronization via atomic counter and backpressure mechanism.
unsafe impl Sync for SyncPtr {}

/// Regular page size (4KB)
pub const PAGE_SIZE: usize = 4096;

/// Huge page size (2MB on x86_64 Linux)
#[cfg(target_os = "linux")]
const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Sentinel for free list end.
const FREE_LIST_EMPTY: u32 = u32::MAX;

#[inline]
const fn pack_free_head(index: u32, tag: u32) -> u64 {
    ((tag as u64) << 32) | (index as u64)
}

#[inline]
const fn unpack_free_head(value: u64) -> (u32, u32) {
    (value as u32, (value >> 32) as u32)
}

fn build_free_list(slot_count: usize) -> (AtomicU64, Box<[AtomicU32]>) {
    let mut next_free: Vec<AtomicU32> = Vec::with_capacity(slot_count);
    for i in 0..slot_count {
        let next = if i + 1 < slot_count {
            (i + 1) as u32
        } else {
            FREE_LIST_EMPTY
        };
        next_free.push(AtomicU32::new(next));
    }
    let free_head = AtomicU64::new(pack_free_head(0, 0));
    (free_head, next_free.into_boxed_slice())
}

/// Ring buffer with pre-allocated slots.
///
/// Each slot can hold one frame of data. Slots are assigned from a free list
/// and only recycled after explicit release.
///
/// # Invariants
/// - `slot_count` is always a power of two (for correct overflow handling)
/// - `slot_size` is always page-aligned (for DMA alignment)
///
/// # Thread Safety
/// `Send + Sync` is derived from `SyncPtr`. The actual thread safety comes from:
/// - The ring owns its memory exclusively
/// - Slot access is controlled by free list + backpressure
/// - Each slot is accessed by exactly one request at a time
pub struct SharedMemoryRing {
    /// Base pointer to the allocated memory region.
    ptr: SyncPtr,
    total_size: usize,
    /// Actual slot size after alignment (>= requested size, page-aligned)
    slot_size: usize,
    /// Always a power of two
    slot_count: usize,
    /// Free-list head with ABA tag
    free_head: AtomicU64,
    /// Next pointers for free list
    next_free: Box<[AtomicU32]>,
    #[cfg(target_os = "linux")]
    uses_huge_pages: bool,
    /// Post-allocation hooks (CUDA pinning, mlock, etc.)
    hooks: Vec<Box<dyn MemoryHook>>,
}

/// Guard providing access to a specific ring buffer slot.
///
/// Does NOT implement Drop - slots are released explicitly by the
/// completion callback via `SharedMemoryRing::release_index`.
#[must_use = "RingSlot provides access to a pre-allocated buffer slot"]
pub struct RingSlot<'a> {
    ring: &'a SharedMemoryRing,
    index: usize,
    len: usize,
}

impl SharedMemoryRing {
    /// Create a new ring buffer with the specified slot count and size.
    ///
    /// # Arguments
    /// * `slot_count` - Must be a power of two (for correct overflow handling)
    /// * `slot_size` - Will be rounded up to page boundary (4KB) for alignment
    /// * `hooks` - Post-allocation hooks (e.g. CUDA pinning, mlock)
    ///
    /// Total memory = slot_count × aligned_slot_size
    ///
    /// # Panics
    /// Panics if `slot_count` is not a power of two.
    ///
    /// # Returns
    /// None on allocation failure.
    pub fn new(
        slot_count: usize,
        slot_size: usize,
        hooks: Vec<Box<dyn MemoryHook>>,
    ) -> Option<Self> {
        assert!(slot_count > 0, "slot_count must be > 0");
        assert!(slot_size > 0, "slot_size must be > 0");
        assert!(
            slot_count <= u32::MAX as usize,
            "slot_count must fit in u32"
        );

        assert!(
            slot_count.is_power_of_two(),
            "slot_count must be power of two for correct overflow handling (got {})",
            slot_count
        );

        // Align slot_size to page boundary (4KB).
        let aligned_slot_size = (slot_size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

        let total_size = slot_count.checked_mul(aligned_slot_size)?;

        #[cfg(feature = "tracing")]
        tracing::info!(
            slot_count,
            requested_slot_size = slot_size,
            aligned_slot_size,
            total_mb = total_size / (1024 * 1024),
            "Allocating shared memory ring buffer"
        );

        // Try huge pages on Linux, falling back to regular pages.
        #[cfg(target_os = "linux")]
        let hooks = match Self::try_alloc_huge(slot_count, aligned_slot_size, total_size, hooks) {
            Ok(ring) => return Some(ring),
            Err(hooks) => {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    total_mb = total_size / (1024 * 1024),
                    "Huge page allocation failed, falling back to 4KB pages (expect higher TLB pressure)"
                );
                hooks
            }
        };

        Self::try_alloc_regular(slot_count, aligned_slot_size, total_size, hooks)
    }

    /// Try to allocate using huge pages (Linux only).
    /// Returns `Err(hooks)` on failure so they can be reused by the fallback path.
    #[cfg(target_os = "linux")]
    fn try_alloc_huge(
        slot_count: usize,
        slot_size: usize,
        total_size: usize,
        hooks: Vec<Box<dyn MemoryHook>>,
    ) -> Result<Self, Vec<Box<dyn MemoryHook>>> {
        use libc::{
            MAP_ANONYMOUS, MAP_FAILED, MAP_HUGETLB, MAP_POPULATE, MAP_PRIVATE, PROT_READ,
            PROT_WRITE, mmap,
        };

        // Round up to huge page boundary
        let alloc_size = (total_size + HUGE_PAGE_SIZE - 1) & !(HUGE_PAGE_SIZE - 1);

        // SAFETY: mmap with MAP_ANONYMOUS creates a new private mapping.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                alloc_size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
                -1,
                0,
            )
        };

        if ptr == MAP_FAILED || ptr.is_null() {
            return Err(hooks);
        }

        let Some(non_null) = NonNull::new(ptr as *mut u8) else {
            return Err(hooks);
        };
        let ptr = SyncPtr(non_null);

        // Pre-fault all huge pages to guarantee physical backing.
        let base_addr = ptr.0.as_ptr() as usize;
        let num_pages = alloc_size / HUGE_PAGE_SIZE;

        #[cfg(feature = "parallel-prefault")]
        {
            use rayon::prelude::*;
            (0..num_pages).into_par_iter().for_each(|page_idx| {
                let addr = base_addr + page_idx * HUGE_PAGE_SIZE;
                // SAFETY: addr is within bounds, each thread writes to distinct page
                unsafe { std::ptr::write_volatile(addr as *mut u8, 0u8) };
            });
        }

        #[cfg(not(feature = "parallel-prefault"))]
        {
            for page_idx in 0..num_pages {
                let addr = base_addr + page_idx * HUGE_PAGE_SIZE;
                // SAFETY: addr is within bounds, sequential access
                unsafe { std::ptr::write_volatile(addr as *mut u8, 0u8) };
            }
        }

        // Run post-allocation hooks
        for hook in &hooks {
            hook.on_alloc(ptr.0.as_ptr(), alloc_size);
        }

        #[cfg(feature = "tracing")]
        tracing::info!(alloc_size, "Ring buffer allocated with huge pages");

        let (free_head, next_free) = build_free_list(slot_count);
        Ok(Self {
            ptr,
            total_size: alloc_size,
            slot_size,
            slot_count,
            free_head,
            next_free,
            uses_huge_pages: true,
            hooks,
        })
    }

    /// Allocate using regular pages.
    fn try_alloc_regular(
        slot_count: usize,
        slot_size: usize,
        total_size: usize,
        hooks: Vec<Box<dyn MemoryHook>>,
    ) -> Option<Self> {
        use std::alloc::{Layout, alloc};

        let layout = Layout::from_size_align(total_size, PAGE_SIZE).ok()?;

        // SAFETY: Layout is valid
        let ptr = unsafe { alloc(layout) };
        let ptr = SyncPtr(NonNull::new(ptr)?);

        // Pre-fault all pages.
        let base_addr = ptr.0.as_ptr() as usize;
        let num_pages = total_size / PAGE_SIZE;

        #[cfg(feature = "parallel-prefault")]
        {
            use rayon::prelude::*;
            (0..num_pages).into_par_iter().for_each(|page_idx| {
                let addr = base_addr + page_idx * PAGE_SIZE;
                // SAFETY: addr is within bounds, each thread writes to distinct page
                unsafe { std::ptr::write_volatile(addr as *mut u8, 0u8) };
            });
        }

        #[cfg(not(feature = "parallel-prefault"))]
        {
            for page_idx in 0..num_pages {
                let addr = base_addr + page_idx * PAGE_SIZE;
                // SAFETY: addr is within bounds, sequential access
                unsafe { std::ptr::write_volatile(addr as *mut u8, 0u8) };
            }
        }

        // Run post-allocation hooks
        for hook in &hooks {
            hook.on_alloc(ptr.0.as_ptr(), total_size);
        }

        #[cfg(feature = "tracing")]
        tracing::info!(total_size, "Ring buffer allocated with regular pages");

        let (free_head, next_free) = build_free_list(slot_count);
        Some(Self {
            ptr,
            total_size,
            slot_size,
            slot_count,
            free_head,
            next_free,
            #[cfg(target_os = "linux")]
            uses_huge_pages: false,
            hooks,
        })
    }

    /// Acquire a slot from the free list.
    ///
    /// Lock-free via ABA-tagged Treiber stack. A slot is only reused
    /// after it has been explicitly released.
    #[inline]
    pub fn acquire_slot(&self) -> Option<RingSlot<'_>> {
        let index = self.acquire_index()?;
        Some(RingSlot {
            ring: self,
            index,
            len: 0,
        })
    }

    /// Acquire a slot index from the free list.
    #[inline]
    pub fn acquire_index(&self) -> Option<usize> {
        loop {
            let state = self.free_head.load(Ordering::Acquire);
            let (head, tag) = unpack_free_head(state);
            if head == FREE_LIST_EMPTY {
                return None;
            }

            let next_cell = self.next_free.get(head as usize)?;
            let next = next_cell.load(Ordering::Relaxed);
            let new_state = pack_free_head(next, tag.wrapping_add(1));

            if self
                .free_head
                .compare_exchange_weak(state, new_state, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Some(head as usize);
            }
            std::hint::spin_loop();
        }
    }

    /// Release a slot index back to the free list.
    #[inline]
    pub fn release_index(&self, index: usize) {
        debug_assert!(index < self.slot_count);
        let Some(slot) = self.next_free.get(index) else {
            return;
        };

        loop {
            let state = self.free_head.load(Ordering::Acquire);
            let (head, tag) = unpack_free_head(state);
            slot.store(head, Ordering::Relaxed);
            let new_state = pack_free_head(index as u32, tag.wrapping_add(1));

            if self
                .free_head
                .compare_exchange_weak(state, new_state, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
            std::hint::spin_loop();
        }
    }

    /// Get pointer to a specific slot.
    #[inline]
    fn slot_ptr(&self, index: usize) -> *mut u8 {
        debug_assert!(index < self.slot_count);
        // SAFETY: index is bounds-checked, offset is within allocation.
        unsafe { self.ptr.0.as_ptr().add(index * self.slot_size) }
    }

    /// Get raw pointer to a specific slot for FFI use.
    ///
    /// # Safety Contract
    /// Caller must ensure:
    /// - `index < slot_count`
    /// - Memory is not accessed after ring buffer is dropped
    #[inline]
    pub fn slot_ptr_for_ffi(&self, index: usize) -> *mut u8 {
        self.slot_ptr(index)
    }

    /// Base address of the entire allocation.
    #[inline]
    pub fn base_addr(&self) -> *mut u8 {
        self.ptr.0.as_ptr()
    }

    /// Total allocated size.
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Size of each slot (page-aligned, may be larger than requested).
    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    /// Number of slots (always a power of two).
    pub fn slot_count(&self) -> usize {
        self.slot_count
    }
}

#[allow(clippy::expect_used)]
impl Drop for SharedMemoryRing {
    fn drop(&mut self) {
        // Run pre-deallocation hooks (reverse order for LIFO cleanup)
        for hook in self.hooks.iter().rev() {
            hook.on_dealloc(self.ptr.0.as_ptr(), self.total_size);
        }

        #[cfg(target_os = "linux")]
        if self.uses_huge_pages {
            // SAFETY: ptr was returned by mmap, total_size matches
            unsafe {
                libc::munmap(self.ptr.0.as_ptr() as *mut libc::c_void, self.total_size);
            }
            return;
        }

        let layout = std::alloc::Layout::from_size_align(self.total_size, PAGE_SIZE)
            .expect("Invalid layout");
        // SAFETY: ptr was allocated with this exact layout in try_alloc_regular.
        unsafe {
            std::alloc::dealloc(self.ptr.0.as_ptr(), layout);
        }
    }
}

impl<'a> RingSlot<'a> {
    /// Copy data into this slot.
    ///
    /// # Panics
    /// Panics if data exceeds slot capacity.
    #[inline]
    pub fn copy_from_slice(&mut self, data: &[u8]) {
        assert!(
            data.len() <= self.ring.slot_size,
            "Data {} exceeds slot capacity {}",
            data.len(),
            self.ring.slot_size
        );

        self.write_with(data.len(), |dst| {
            // SAFETY: dst points to slot_size bytes (asserted above), data.len() <= slot_size,
            // and src/dst regions don't overlap (dst is in the ring's exclusive slot).
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            }
        });
    }

    /// Write `len` bytes using a caller-provided writer.
    ///
    /// # Panics
    /// Panics if length exceeds slot capacity.
    #[inline]
    pub fn write_with<F>(&mut self, len: usize, writer: F)
    where
        F: FnOnce(*mut u8),
    {
        assert!(
            len <= self.ring.slot_size,
            "Length {} exceeds slot capacity {}",
            len,
            self.ring.slot_size
        );

        let dst = self.ring.slot_ptr(self.index);
        writer(dst);
        self.len = len;
    }

    /// Get the slot contents as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: len was set by copy_from_slice, exclusive access
        unsafe { std::slice::from_raw_parts(self.ring.slot_ptr(self.index) as *const u8, self.len) }
    }

    /// Get raw pointer for FFI.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ring.slot_ptr(self.index)
    }

    /// Current data length in this slot.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if slot is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Slot index.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Byte offset from ring start.
    #[inline]
    pub fn offset(&self) -> usize {
        self.index * self.ring.slot_size
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let ring = SharedMemoryRing::new(4, 1024, vec![]).expect("alloc failed");
        assert_eq!(ring.slot_count(), 4);
        assert_eq!(ring.slot_size(), PAGE_SIZE);
    }

    #[test]
    fn test_slot_acquisition() {
        let ring = SharedMemoryRing::new(4, 1024, vec![]).expect("alloc failed");

        let slot0 = ring.acquire_slot().unwrap();
        let slot1 = ring.acquire_slot().unwrap();
        let slot2 = ring.acquire_slot().unwrap();
        let slot3 = ring.acquire_slot().unwrap();

        let mut indices = vec![slot0.index(), slot1.index(), slot2.index(), slot3.index()];
        indices.sort_unstable();
        assert_eq!(indices, vec![0, 1, 2, 3]);

        ring.release_index(slot0.index());
        ring.release_index(slot1.index());
        ring.release_index(slot2.index());
        ring.release_index(slot3.index());
    }

    #[test]
    fn test_slot_copy() {
        let ring = SharedMemoryRing::new(4, 1024, vec![]).expect("alloc failed");
        let mut slot = ring.acquire_slot().unwrap();

        let data = b"hello world";
        slot.copy_from_slice(data);

        assert_eq!(slot.len(), data.len());
        assert_eq!(slot.as_slice(), data);
        assert_eq!(slot.offset(), 0);
        ring.release_index(slot.index());
    }

    #[test]
    fn test_slot_offset() {
        let ring = SharedMemoryRing::new(4, 1024, vec![]).expect("alloc failed");
        let aligned_size = ring.slot_size();

        let slot0 = ring.acquire_slot().unwrap();
        assert_eq!(slot0.offset(), 0);

        let slot1 = ring.acquire_slot().unwrap();
        assert_eq!(slot1.offset(), aligned_size);

        let slot2 = ring.acquire_slot().unwrap();
        assert_eq!(slot2.offset(), aligned_size * 2);

        ring.release_index(slot0.index());
        ring.release_index(slot1.index());
        ring.release_index(slot2.index());
    }

    #[test]
    #[should_panic(expected = "exceeds slot capacity")]
    fn test_slot_overflow() {
        let ring = SharedMemoryRing::new(4, 64, vec![]).expect("alloc failed");
        let mut slot = ring.acquire_slot().unwrap();
        let data = vec![0u8; PAGE_SIZE + 1];
        slot.copy_from_slice(&data);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn test_non_power_of_two_panics() {
        let _ = SharedMemoryRing::new(100, 1024, vec![]);
    }

    #[test]
    fn test_power_of_two_accepted() {
        for &count in &[1, 2, 4, 8, 16, 32, 64, 128, 256] {
            let ring = SharedMemoryRing::new(count, 1024, vec![]).expect("alloc failed");
            assert_eq!(ring.slot_count(), count);
        }
    }

    #[test]
    fn slot_pointers_page_aligned() {
        let ring = SharedMemoryRing::new(4, 1024, vec![]).expect("alloc failed");
        let mut slots = Vec::new();
        for _ in 0..4 {
            let slot = ring.acquire_slot().unwrap();
            assert_eq!(slot.as_ptr() as usize % PAGE_SIZE, 0);
            slots.push(slot.index());
        }
        for idx in slots {
            ring.release_index(idx);
        }
    }

    #[test]
    fn test_free_list_exhaustion_and_release() {
        let ring = SharedMemoryRing::new(2, 1024, vec![]).expect("alloc failed");
        let slot0 = ring.acquire_slot().unwrap();
        let slot1 = ring.acquire_slot().unwrap();
        assert!(ring.acquire_index().is_none());
        ring.release_index(slot0.index());
        let slot2 = ring.acquire_slot().unwrap();
        assert_eq!(slot2.index(), slot0.index());
        ring.release_index(slot1.index());
        ring.release_index(slot2.index());
    }

    #[test]
    fn acquire_slot_returns_none_when_exhausted() {
        let ring = SharedMemoryRing::new(1, 1024, vec![]).expect("alloc failed");
        let slot = ring.acquire_slot().unwrap();
        assert!(ring.acquire_slot().is_none());
        ring.release_index(slot.index());
        assert!(ring.acquire_slot().is_some());
    }

    #[test]
    fn base_addr_is_valid() {
        let ring = SharedMemoryRing::new(4, 1024, vec![]).expect("alloc failed");
        let base = ring.base_addr();
        assert!(!base.is_null());
        assert_eq!(base as usize % PAGE_SIZE, 0);
    }

    #[test]
    fn test_hook_lifecycle() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        struct TestHook {
            alloc_called: Arc<AtomicBool>,
            dealloc_called: Arc<AtomicBool>,
        }

        impl MemoryHook for TestHook {
            fn on_alloc(&self, ptr: *mut u8, size: usize) -> bool {
                assert!(!ptr.is_null());
                assert!(size > 0);
                self.alloc_called.store(true, Ordering::SeqCst);
                true
            }

            fn on_dealloc(&self, ptr: *mut u8, size: usize) {
                assert!(!ptr.is_null());
                assert!(size > 0);
                self.dealloc_called.store(true, Ordering::SeqCst);
            }
        }

        let alloc_called = Arc::new(AtomicBool::new(false));
        let dealloc_called = Arc::new(AtomicBool::new(false));

        let hook = TestHook {
            alloc_called: alloc_called.clone(),
            dealloc_called: dealloc_called.clone(),
        };

        {
            let _ring = SharedMemoryRing::new(4, 1024, vec![Box::new(hook)]).expect("alloc failed");
            assert!(alloc_called.load(Ordering::SeqCst));
            assert!(!dealloc_called.load(Ordering::SeqCst));
        }

        assert!(dealloc_called.load(Ordering::SeqCst));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn slot_roundtrip(data in proptest::collection::vec(any::<u8>(), 0..1024)) {
            let ring = SharedMemoryRing::new(4, 1024, vec![]).unwrap();
            let mut slot = ring.acquire_slot().unwrap();
            slot.copy_from_slice(&data);
            prop_assert_eq!(&slot.as_slice()[..data.len()], &data[..]);
            ring.release_index(slot.index());
        }

        #[test]
        fn acquire_release_roundtrip(iterations in 1usize..1000) {
            let ring = SharedMemoryRing::new(4, 64, vec![]).unwrap();
            for _ in 0..iterations {
                let slot = ring.acquire_slot().unwrap();
                prop_assert!(slot.index() < 4);
                ring.release_index(slot.index());
            }
        }
    }
}
