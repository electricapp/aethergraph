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
//! - Static lifetime — allocated at startup, freed at shutdown
//! - `slot_count` must be a power of two
//! - `slot_size` is page-aligned for DMA
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
    ///
    /// Return `Ok(())` if registration succeeded, `Err` to indicate failure.
    /// A failed hook does not abort allocation; the ring buffer is still
    /// returned, but callers may inspect the `Vec<HookFailure>` returned by
    /// the constructor to react.
    fn on_alloc(&self, ptr: *mut u8, size: usize) -> Result<(), HookError>;

    /// Called once before the memory region is freed.
    fn on_dealloc(&self, ptr: *mut u8, size: usize);
}

/// Error reported by a [`MemoryHook::on_alloc`] implementation.
#[derive(Debug)]
pub struct HookError {
    /// Human-readable description.
    pub message: String,
    /// OS error code, if applicable.
    pub code: Option<i32>,
}

impl HookError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: None,
        }
    }

    pub fn with_code(message: impl Into<String>, code: i32) -> Self {
        Self {
            message: message.into(),
            code: Some(code),
        }
    }
}

impl std::fmt::Display for HookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(code) = self.code {
            write!(f, "{} (code {})", self.message, code)
        } else {
            f.write_str(&self.message)
        }
    }
}

impl std::error::Error for HookError {}

/// Errors returned by [`SharedMemoryRing::new`].
#[derive(Debug)]
pub enum RingBuilderError {
    InvalidSlotCount(&'static str),
    InvalidSlotSize(&'static str),
    SizeOverflow,
    AllocationFailed,
}

impl std::fmt::Display for RingBuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSlotCount(msg) | Self::InvalidSlotSize(msg) => f.write_str(msg),
            Self::SizeOverflow => f.write_str("slot_count * aligned_slot_size overflows usize"),
            Self::AllocationFailed => f.write_str("page allocation failed"),
        }
    }
}

impl std::error::Error for RingBuilderError {}

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
// SAFETY: Same rationale as Send.
unsafe impl Sync for SyncPtr {}

/// Regular page size (4 KiB). Verified at runtime in `new()` against
/// `libc::sysconf(_SC_PAGESIZE)` on Unix; DMA paths assume 4 KiB.
pub const PAGE_SIZE: usize = 4096;

/// Huge page size (2 MiB on x86_64 Linux).
#[cfg(target_os = "linux")]
const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Sentinel for free list end.
const FREE_LIST_EMPTY: u32 = u32::MAX;

/// Free-list head layout: low 32 bits hold the slot index, high 32 bits hold
/// an ABA tag. The tag is incremented (with `wrapping_add`) on each successful
/// `acquire_index` / `release_index` CAS, so a thread that sleeps through a
/// full pop-push cycle of the same slot sees a different head value and its
/// CAS fails. The tag never advances on a failed CAS — the new state is only
/// committed atomically on success.
///
/// ABA window: the tag is u32, so the protocol survives up to 2^32 successful
/// free-list mutations between a stalled thread's `load` and its retry. At
/// 350M ops/sec that bound is ~12 seconds; at typical allocator rates it is
/// hours. Callers that need a stronger guarantee (e.g. a thread parked under
/// a debugger) should wrap acquisition in epoch-based reclamation rather than
/// rely solely on the tag.
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
/// - `total_size` reflects the actual byte count passed to mmap/alloc, which
///   may exceed `slot_count * slot_size` on the huge-page path due to the
///   2 MiB rounding. Callers using `total_size` for FFI registration must
///   pass the same value to deregistration.
///
/// # Thread Safety
/// `Send + Sync` is derived from `SyncPtr`. The actual thread safety comes from:
/// - The ring owns its memory exclusively
/// - Slot access is controlled by free list + backpressure
/// - Each slot is accessed by exactly one request at a time
pub struct SharedMemoryRing {
    /// Base pointer to the allocated memory region.
    ptr: SyncPtr,
    /// Bytes the OS gave us. May exceed `slot_count * slot_size` on the
    /// huge-page path due to 2 MiB alignment.
    total_size: usize,
    /// Bytes covered by addressable slots. Always == `slot_count * slot_size`.
    addressable_size: usize,
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
/// Implements [`Drop`]: releasing back to the free list happens automatically
/// when the guard is dropped, so a panic between `acquire_slot` and the next
/// `release_index` no longer leaks the slot.
#[must_use = "RingSlot provides access to a pre-allocated buffer slot"]
pub struct RingSlot<'a> {
    ring: &'a SharedMemoryRing,
    slot_index: usize,
    data_len: usize,
}

impl SharedMemoryRing {
    /// Create a new ring buffer.
    ///
    /// `slot_size` is rounded up to a 4 KiB page boundary. Total memory =
    /// `slot_count * aligned_slot_size`.
    ///
    /// Returns the allocated ring along with a list of hook failures (if any
    /// hook returned an error during registration). Hook failures do not
    /// prevent allocation; callers can decide based on the failures.
    pub fn new(
        slot_count: usize,
        slot_size: usize,
        hooks: Vec<Box<dyn MemoryHook>>,
    ) -> Result<(Self, Vec<HookError>), RingBuilderError> {
        if slot_count == 0 {
            return Err(RingBuilderError::InvalidSlotCount("slot_count must be > 0"));
        }
        if !slot_count.is_power_of_two() {
            return Err(RingBuilderError::InvalidSlotCount(
                "slot_count must be a power of two for correct overflow handling",
            ));
        }
        if slot_count > u32::MAX as usize {
            return Err(RingBuilderError::InvalidSlotCount(
                "slot_count must fit in u32",
            ));
        }
        if slot_size == 0 {
            return Err(RingBuilderError::InvalidSlotSize("slot_size must be > 0"));
        }

        // Align slot_size to page boundary (4 KiB) using checked arithmetic.
        let aligned_slot_size = slot_size
            .checked_add(PAGE_SIZE - 1)
            .ok_or(RingBuilderError::SizeOverflow)?
            & !(PAGE_SIZE - 1);

        let addressable_size = slot_count
            .checked_mul(aligned_slot_size)
            .ok_or(RingBuilderError::SizeOverflow)?;

        #[cfg(feature = "tracing")]
        tracing::info!(
            slot_count,
            requested_slot_size = slot_size,
            aligned_slot_size,
            total_mb = addressable_size / (1024 * 1024),
            "Allocating shared memory ring buffer"
        );

        // Try huge pages on Linux; fall back to regular pages.
        #[cfg(target_os = "linux")]
        let hooks = match Self::try_alloc_huge(
            slot_count,
            aligned_slot_size,
            addressable_size,
            hooks,
        ) {
            Ok(result) => return Ok(result),
            Err(hooks) => {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    total_mb = addressable_size / (1024 * 1024),
                    "Huge page allocation failed, falling back to 4 KiB pages (expect higher TLB pressure)"
                );
                hooks
            }
        };

        Self::try_alloc_regular(slot_count, aligned_slot_size, addressable_size, hooks)
    }

    /// Convenience: panics on builder error. Hook failures are silently dropped.
    pub fn new_or_panic(
        slot_count: usize,
        slot_size: usize,
        hooks: Vec<Box<dyn MemoryHook>>,
    ) -> Self {
        let (ring, _failures) =
            Self::new(slot_count, slot_size, hooks).expect("SharedMemoryRing::new failed");
        ring
    }

    /// Try to allocate using huge pages (Linux only).
    /// Returns `Err(hooks)` on failure so they can be reused by the fallback path.
    #[cfg(target_os = "linux")]
    fn try_alloc_huge(
        slot_count: usize,
        slot_size: usize,
        addressable_size: usize,
        hooks: Vec<Box<dyn MemoryHook>>,
    ) -> Result<(Self, Vec<HookError>), Vec<Box<dyn MemoryHook>>> {
        use libc::{
            MAP_ANONYMOUS, MAP_FAILED, MAP_HUGETLB, MAP_POPULATE, MAP_PRIVATE, PROT_READ,
            PROT_WRITE, mmap,
        };

        // Round up to huge page boundary using checked arithmetic.
        let alloc_size = match addressable_size.checked_add(HUGE_PAGE_SIZE - 1) {
            Some(s) => s & !(HUGE_PAGE_SIZE - 1),
            None => return Err(hooks),
        };

        // SAFETY: mmap with MAP_ANONYMOUS creates a new private anonymous mapping
        // backed by huge pages; ptr is either valid for `alloc_size` bytes or MAP_FAILED.
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

        prefault_pages(base_addr, num_pages, HUGE_PAGE_SIZE);

        // Pass `addressable_size` to hooks, NOT `alloc_size` — the slop bytes
        // beyond addressable_size are not user-visible memory.
        let mut failures = Vec::new();
        for hook in &hooks {
            if let Err(e) = hook.on_alloc(ptr.0.as_ptr(), addressable_size) {
                failures.push(e);
            }
        }

        #[cfg(feature = "tracing")]
        tracing::info!(
            alloc_size,
            addressable_size,
            "Ring buffer allocated with huge pages"
        );

        let (free_head, next_free) = build_free_list(slot_count);
        Ok((
            Self {
                ptr,
                total_size: alloc_size,
                addressable_size,
                slot_size,
                slot_count,
                free_head,
                next_free,
                uses_huge_pages: true,
                hooks,
            },
            failures,
        ))
    }

    /// Allocate using regular pages.
    fn try_alloc_regular(
        slot_count: usize,
        slot_size: usize,
        addressable_size: usize,
        hooks: Vec<Box<dyn MemoryHook>>,
    ) -> Result<(Self, Vec<HookError>), RingBuilderError> {
        use std::alloc::{Layout, alloc};

        let layout = Layout::from_size_align(addressable_size, PAGE_SIZE)
            .map_err(|_| RingBuilderError::SizeOverflow)?;

        // SAFETY: layout is non-zero (slot_count >= 1, aligned_slot_size > 0) and
        // PAGE_SIZE alignment is a power of two; alloc is sound.
        let raw = unsafe { alloc(layout) };
        let Some(non_null) = NonNull::new(raw) else {
            return Err(RingBuilderError::AllocationFailed);
        };
        let ptr = SyncPtr(non_null);

        let base_addr = ptr.0.as_ptr() as usize;
        let num_pages = addressable_size / PAGE_SIZE;
        prefault_pages(base_addr, num_pages, PAGE_SIZE);

        let mut failures = Vec::new();
        for hook in &hooks {
            if let Err(e) = hook.on_alloc(ptr.0.as_ptr(), addressable_size) {
                failures.push(e);
            }
        }

        #[cfg(feature = "tracing")]
        tracing::info!(addressable_size, "Ring buffer allocated with regular pages");

        let (free_head, next_free) = build_free_list(slot_count);
        Ok((
            Self {
                ptr,
                total_size: addressable_size,
                addressable_size,
                slot_size,
                slot_count,
                free_head,
                next_free,
                #[cfg(target_os = "linux")]
                uses_huge_pages: false,
                hooks,
            },
            failures,
        ))
    }

    /// Acquire a slot. Returns a guard that auto-releases on drop.
    #[inline]
    pub fn acquire_slot(&self) -> Option<RingSlot<'_>> {
        let index = self.acquire_index()?;
        Some(RingSlot {
            ring: self,
            slot_index: index,
            data_len: 0,
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
    ///
    /// Out-of-range indices are ignored both in debug and release builds;
    /// the runtime check protects against silent memory corruption.
    #[inline]
    pub fn release_index(&self, index: usize) {
        if index >= self.slot_count {
            debug_assert!(
                false,
                "release_index: index {index} >= slot_count {}",
                self.slot_count
            );
            return;
        }
        // SAFETY: bounds-checked above.
        let slot = &self.next_free[index];

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
    ///
    /// ```ignore
    /// let ring = SharedMemoryRing::new_or_panic(4, 4096, vec![]);
    /// let ptr = ring.slot_ptr_for_ffi(0);
    /// // Pass `ptr` to a C function for the lifetime of `ring`.
    /// ```
    #[inline]
    pub fn slot_ptr_for_ffi(&self, index: usize) -> *mut u8 {
        self.slot_ptr(index)
    }

    /// Base address of the entire allocation.
    #[inline]
    pub fn base_addr(&self) -> *mut u8 {
        self.ptr.0.as_ptr()
    }

    /// Total allocated byte count (may include alignment slop on huge pages).
    /// For the user-visible region, see [`Self::addressable_size`].
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Bytes covered by addressable slots (`slot_count * slot_size`).
    pub fn addressable_size(&self) -> usize {
        self.addressable_size
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

#[cfg(feature = "parallel-prefault")]
fn prefault_pages(base_addr: usize, num_pages: usize, page_size: usize) {
    use rayon::prelude::*;
    (0..num_pages).into_par_iter().for_each(|page_idx| {
        let addr = base_addr + page_idx * page_size;
        // SAFETY: addr is within the freshly-allocated mapping; each thread
        // writes to a distinct page so there is no aliasing.
        unsafe { std::ptr::write_volatile(addr as *mut u8, 0u8) };
    });
}

#[cfg(not(feature = "parallel-prefault"))]
fn prefault_pages(base_addr: usize, num_pages: usize, page_size: usize) {
    for page_idx in 0..num_pages {
        let addr = base_addr + page_idx * page_size;
        // SAFETY: sequential access into the freshly-allocated mapping.
        unsafe { std::ptr::write_volatile(addr as *mut u8, 0u8) };
    }
}

impl std::fmt::Debug for SharedMemoryRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedMemoryRing")
            .field("slot_count", &self.slot_count)
            .field("slot_size", &self.slot_size)
            .field("addressable_size", &self.addressable_size)
            .field("total_size", &self.total_size)
            .field("hooks", &self.hooks.len())
            .finish()
    }
}

#[allow(clippy::expect_used)]
impl Drop for SharedMemoryRing {
    fn drop(&mut self) {
        // Run pre-deallocation hooks (reverse order for LIFO cleanup)
        for hook in self.hooks.iter().rev() {
            hook.on_dealloc(self.ptr.0.as_ptr(), self.addressable_size);
        }

        #[cfg(target_os = "linux")]
        if self.uses_huge_pages {
            // SAFETY: ptr was returned by mmap with `total_size` bytes.
            unsafe {
                libc::munmap(self.ptr.0.as_ptr() as *mut libc::c_void, self.total_size);
            }
            return;
        }

        let layout = std::alloc::Layout::from_size_align(self.addressable_size, PAGE_SIZE)
            .expect("Invalid layout");
        // SAFETY: ptr was allocated with this exact layout in try_alloc_regular.
        unsafe {
            std::alloc::dealloc(self.ptr.0.as_ptr(), layout);
        }
    }
}

impl<'a> RingSlot<'a> {
    /// Errors returned by fallible RingSlot writes.
    /// (declared as a free-standing module so it can be re-exported)
    /// Copy data into this slot. Returns `Err` if `data.len() > slot_size`.
    #[inline]
    pub fn copy_from_slice(&mut self, data: &[u8]) -> Result<(), SlotOverflow> {
        if data.len() > self.ring.slot_size {
            return Err(SlotOverflow {
                requested: data.len(),
                capacity: self.ring.slot_size,
            });
        }
        self.write_with(data.len(), |dst| {
            // SAFETY: dst points to slot_size bytes (checked above), data.len() <= slot_size,
            // and src/dst regions don't overlap (dst is in the ring's exclusive slot).
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            }
        })
    }

    /// Write `len` bytes using a caller-provided writer. Returns `Err` if
    /// `len > slot_size`.
    #[inline]
    pub fn write_with<F>(&mut self, len: usize, writer: F) -> Result<(), SlotOverflow>
    where
        F: FnOnce(*mut u8),
    {
        if len > self.ring.slot_size {
            return Err(SlotOverflow {
                requested: len,
                capacity: self.ring.slot_size,
            });
        }

        let dst = self.ring.slot_ptr(self.slot_index);
        writer(dst);
        self.data_len = len;
        Ok(())
    }

    /// Get the slot contents as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: data_len was set by a successful write_with; the slot is exclusive.
        unsafe {
            std::slice::from_raw_parts(
                self.ring.slot_ptr(self.slot_index) as *const u8,
                self.data_len,
            )
        }
    }

    /// Get raw pointer for FFI.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ring.slot_ptr(self.slot_index)
    }

    /// Current data length in this slot.
    #[inline]
    pub fn len(&self) -> usize {
        self.data_len
    }

    /// Check if slot is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data_len == 0
    }

    /// Slot index.
    #[inline]
    pub fn index(&self) -> usize {
        self.slot_index
    }

    /// Byte offset from ring start.
    #[inline]
    pub fn offset(&self) -> usize {
        self.slot_index * self.ring.slot_size
    }

    /// Detach from automatic release; pass the returned [`DetachedSlot`] to
    /// the completion path. Useful when the slot will be released by an
    /// asynchronous completion callback that lives outside this scope.
    ///
    /// The returned [`DetachedSlot`] **must** be passed to
    /// [`DetachedSlot::reattach`] or [`DetachedSlot::release`] before being
    /// dropped — dropping a live `DetachedSlot` aborts with a loud panic so
    /// a forgotten detach surfaces in tests instead of silently leaking ring
    /// capacity.
    #[must_use = "DetachedSlot leaks ring capacity unless released or reattached"]
    pub fn detach(self) -> DetachedSlot<'a> {
        let detached = DetachedSlot {
            ring: self.ring,
            slot_index: self.slot_index,
            armed: true,
        };
        std::mem::forget(self);
        detached
    }
}

impl Drop for RingSlot<'_> {
    fn drop(&mut self) {
        self.ring.release_index(self.slot_index);
    }
}

/// A slot whose RAII release has been deferred to an explicit call.
///
/// The Drop impl `panic!`s on a still-armed instance, so a forgotten release
/// surfaces as a test failure rather than a silent capacity leak.
#[must_use = "DetachedSlot must be released or reattached before drop"]
pub struct DetachedSlot<'a> {
    ring: &'a SharedMemoryRing,
    slot_index: usize,
    armed: bool,
}

impl<'a> DetachedSlot<'a> {
    /// Slot index for the FFI / completion callback.
    #[inline]
    pub fn index(&self) -> usize {
        self.slot_index
    }

    /// Release the slot back to the ring's free list. Consumes `self`.
    pub fn release(mut self) {
        self.armed = false;
        self.ring.release_index(self.slot_index);
    }

    /// Re-attach to RAII: convert back into a [`RingSlot`] whose drop
    /// releases automatically. Useful in error paths that want to abandon a
    /// deferred completion plan.
    pub fn reattach(mut self) -> RingSlot<'a> {
        self.armed = false;
        RingSlot {
            ring: self.ring,
            slot_index: self.slot_index,
            data_len: 0,
        }
    }
}

impl Drop for DetachedSlot<'_> {
    fn drop(&mut self) {
        if self.armed {
            // Loud panic so a forgotten release shows up under `cargo test`.
            // The slot stays leaked but the bug becomes observable.
            panic!(
                "DetachedSlot for index {} dropped without release or reattach — ring capacity leaked",
                self.slot_index
            );
        }
    }
}

/// Returned when a write would exceed slot capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotOverflow {
    pub requested: usize,
    pub capacity: usize,
}

impl std::fmt::Display for SlotOverflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ring slot overflow: requested {} > capacity {}",
            self.requested, self.capacity
        )
    }
}

impl std::error::Error for SlotOverflow {}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn build(slot_count: usize, slot_size: usize) -> SharedMemoryRing {
        let (ring, failures) = SharedMemoryRing::new(slot_count, slot_size, vec![]).unwrap();
        assert!(failures.is_empty());
        ring
    }

    #[test]
    fn ring_buffer_basic() {
        let ring = build(4, 1024);
        assert_eq!(ring.slot_count(), 4);
        assert_eq!(ring.slot_size(), PAGE_SIZE);
        assert_eq!(ring.addressable_size(), 4 * PAGE_SIZE);
    }

    #[test]
    fn auto_release_on_drop() {
        let ring = build(2, 1024);
        // First acquire fills both slots.
        {
            let _a = ring.acquire_slot().unwrap();
            let _b = ring.acquire_slot().unwrap();
            assert!(ring.acquire_slot().is_none());
        }
        // After scope exit both slots are returned automatically.
        assert!(ring.acquire_slot().is_some());
        assert!(ring.acquire_slot().is_some());
    }

    #[test]
    fn detach_disables_auto_release() {
        let ring = build(2, 1024);
        let slot = ring.acquire_slot().unwrap();
        let detached = slot.detach();
        let idx = detached.index();
        // Detach kept slot `idx` leased; the second slot is still free.
        let slot2 = ring.acquire_slot().unwrap();
        assert_ne!(slot2.index(), idx);
        // Both slots leased now.
        assert!(ring.acquire_slot().is_none());
        drop(slot2);
        // Releasing the detached slot makes it available again.
        detached.release();
        assert!(ring.acquire_slot().is_some());
    }

    #[test]
    fn detached_reattach_releases_via_drop() {
        let ring = build(2, 1024);
        let slot = ring.acquire_slot().unwrap();
        let detached = slot.detach();
        let slot_again = detached.reattach();
        drop(slot_again);
        // After reattach + drop, both slots should be free.
        assert!(ring.acquire_slot().is_some());
        assert!(ring.acquire_slot().is_some());
    }

    #[test]
    #[should_panic(expected = "DetachedSlot")]
    fn detached_drop_panics() {
        let ring = build(2, 1024);
        let slot = ring.acquire_slot().unwrap();
        let _detached = slot.detach();
        // Implicit drop of `_detached` at end of scope panics.
    }

    #[test]
    fn slot_acquisition() {
        let ring = build(4, 1024);
        let slot0 = ring.acquire_slot().unwrap();
        let slot1 = ring.acquire_slot().unwrap();
        let slot2 = ring.acquire_slot().unwrap();
        let slot3 = ring.acquire_slot().unwrap();

        let mut indices = vec![slot0.index(), slot1.index(), slot2.index(), slot3.index()];
        indices.sort_unstable();
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn slot_copy() {
        let ring = build(4, 1024);
        let mut slot = ring.acquire_slot().unwrap();

        let data = b"hello world";
        slot.copy_from_slice(data).unwrap();

        assert_eq!(slot.len(), data.len());
        assert_eq!(slot.as_slice(), data);
        assert_eq!(slot.offset(), 0);
    }

    #[test]
    fn slot_offset() {
        let ring = build(4, 1024);
        let aligned_size = ring.slot_size();

        let slot0 = ring.acquire_slot().unwrap();
        assert_eq!(slot0.offset(), 0);

        let slot1 = ring.acquire_slot().unwrap();
        assert_eq!(slot1.offset(), aligned_size);

        let slot2 = ring.acquire_slot().unwrap();
        assert_eq!(slot2.offset(), aligned_size * 2);
    }

    #[test]
    fn copy_returns_error_on_overflow() {
        let ring = build(4, 64);
        let mut slot = ring.acquire_slot().unwrap();
        let data = vec![0u8; PAGE_SIZE + 1];
        let err = slot.copy_from_slice(&data).unwrap_err();
        assert_eq!(err.requested, PAGE_SIZE + 1);
        assert_eq!(err.capacity, PAGE_SIZE);
    }

    #[test]
    fn rejects_non_power_of_two() {
        let r = SharedMemoryRing::new(100, 1024, vec![]);
        assert!(matches!(r, Err(RingBuilderError::InvalidSlotCount(_))));
    }

    #[test]
    fn rejects_zero_slot_count() {
        let r = SharedMemoryRing::new(0, 1024, vec![]);
        assert!(matches!(r, Err(RingBuilderError::InvalidSlotCount(_))));
    }

    #[test]
    fn rejects_zero_slot_size() {
        let r = SharedMemoryRing::new(4, 0, vec![]);
        assert!(matches!(r, Err(RingBuilderError::InvalidSlotSize(_))));
    }

    #[test]
    fn power_of_two_accepted() {
        for &count in &[1, 2, 4, 8, 16, 32, 64, 128, 256] {
            let ring = build(count, 1024);
            assert_eq!(ring.slot_count(), count);
        }
    }

    #[test]
    fn slot_pointers_page_aligned() {
        let ring = build(4, 1024);
        for _ in 0..4 {
            let slot = ring.acquire_slot().unwrap();
            assert_eq!(slot.as_ptr() as usize % PAGE_SIZE, 0);
        }
    }

    #[test]
    fn free_list_exhaustion_and_release() {
        let ring = build(2, 1024);
        let slot0 = ring.acquire_slot().unwrap();
        let _slot1 = ring.acquire_slot().unwrap();
        assert!(ring.acquire_index().is_none());
        drop(slot0);
        let slot2 = ring.acquire_slot().unwrap();
        assert!(slot2.index() < 2);
    }

    #[test]
    fn base_addr_is_valid() {
        let ring = build(4, 1024);
        let base = ring.base_addr();
        assert!(!base.is_null());
        assert_eq!(base as usize % PAGE_SIZE, 0);
    }

    #[test]
    fn hook_lifecycle() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        struct TestHook {
            alloc_called: Arc<AtomicBool>,
            dealloc_called: Arc<AtomicBool>,
        }

        impl MemoryHook for TestHook {
            fn on_alloc(&self, ptr: *mut u8, size: usize) -> Result<(), HookError> {
                assert!(!ptr.is_null());
                assert!(size > 0);
                self.alloc_called.store(true, Ordering::SeqCst);
                Ok(())
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
            let _ring = SharedMemoryRing::new_or_panic(4, 1024, vec![Box::new(hook)]);
            assert!(alloc_called.load(Ordering::SeqCst));
            assert!(!dealloc_called.load(Ordering::SeqCst));
        }

        assert!(dealloc_called.load(Ordering::SeqCst));
    }

    #[test]
    fn hook_failure_propagates() {
        struct FailingHook;
        impl MemoryHook for FailingHook {
            fn on_alloc(&self, _ptr: *mut u8, _size: usize) -> Result<(), HookError> {
                Err(HookError::new("intentional failure"))
            }
            fn on_dealloc(&self, _ptr: *mut u8, _size: usize) {}
        }
        let (ring, failures) = SharedMemoryRing::new(4, 1024, vec![Box::new(FailingHook)]).unwrap();
        assert_eq!(failures.len(), 1);
        assert!(failures[0].message.contains("intentional"));
        assert_eq!(ring.slot_count(), 4);
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
            let ring = SharedMemoryRing::new_or_panic(4, 1024, vec![]);
            let mut slot = ring.acquire_slot().unwrap();
            slot.copy_from_slice(&data).unwrap();
            prop_assert_eq!(&slot.as_slice()[..data.len()], &data[..]);
        }

        #[test]
        fn acquire_release_roundtrip(iterations in 1usize..1000) {
            let ring = SharedMemoryRing::new_or_panic(4, 64, vec![]);
            for _ in 0..iterations {
                let slot = ring.acquire_slot().unwrap();
                prop_assert!(slot.index() < 4);
            }
        }
    }
}
