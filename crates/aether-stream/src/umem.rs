//! UMEM frame pool — wraps `SharedMemoryRing` as the AF_XDP frame allocator.
//!
//! Each UMEM frame corresponds to one ring buffer slot. The Treiber stack
//! free list provides lock-free frame acquire/release for the ingestion loop.

use aether_mem::hooks::MlockHook;
use aether_mem::{MemoryHook, SharedMemoryRing};

/// AF_XDP UMEM frame pool backed by `SharedMemoryRing`.
///
/// Frame size must be 2048 or 4096 (kernel requirement).
/// Frames are page-aligned and mlock'd to prevent swapping.
pub struct Umem {
    ring: SharedMemoryRing,
    frame_size: u32,
}

impl Umem {
    /// Create a new UMEM pool.
    ///
    /// # Arguments
    /// * `frame_count` - Number of frames (must be power of two)
    /// * `frame_size` - 2048 or 4096
    /// * `extra_hooks` - Additional hooks (e.g. RdmaRegHook for RDMA one-sided reads)
    ///
    /// # Panics
    /// Panics if `frame_size` is not 2048 or 4096.
    pub fn new(
        frame_count: usize,
        frame_size: u32,
        extra_hooks: Vec<Box<dyn MemoryHook>>,
    ) -> Option<Self> {
        // Kernel allows 2048 or 4096, but our SharedMemoryRing rounds slot_size
        // up to a 4 KB page (see ring.rs:171). Requesting 2048 yielded a 4 KB
        // stride while the kernel was told chunk_size=2048 — packet at
        // chunk N landed at base+N*2048, our frame_ptr(N) returned base+N*4096,
        // so half the frames pointed to the wrong UMEM region. Until we add a
        // sub-page allocator path, only 4096 is correct.
        assert_eq!(
            frame_size, 4096,
            "Umem currently supports only 4096-byte frames (got {}). \
             SharedMemoryRing rounds slot_size to PAGE_SIZE; a 2048 frame_size \
             would silently mismatch the kernel-registered chunk stride.",
            frame_size
        );

        let mut hooks: Vec<Box<dyn MemoryHook>> = vec![Box::new(MlockHook)];
        hooks.extend(extra_hooks);

        let (ring, _hook_failures) =
            SharedMemoryRing::new(frame_count, frame_size as usize, hooks).ok()?;

        Some(Self { ring, frame_size })
    }

    /// Base address of the UMEM region (for `XDP_UMEM_REG`).
    #[inline]
    pub fn base_addr(&self) -> u64 {
        self.ring.base_addr() as u64
    }

    /// Total size of the UMEM region.
    #[inline]
    pub fn total_size(&self) -> usize {
        self.ring.total_size()
    }

    /// Frame size.
    #[inline]
    pub fn frame_size(&self) -> u32 {
        self.frame_size
    }

    /// Number of frames.
    #[inline]
    pub fn frame_count(&self) -> usize {
        self.ring.slot_count()
    }

    /// Acquire a free frame index. Returns `None` if pool is exhausted.
    #[inline]
    pub fn acquire_frame(&self) -> Option<usize> {
        self.ring.acquire_index()
    }

    /// Release a frame back to the pool.
    #[inline]
    pub fn release_frame(&self, index: usize) {
        self.ring.release_index(index);
    }

    /// Compute the UMEM offset for a frame index (for ring descriptors).
    #[inline]
    pub fn frame_addr(&self, index: usize) -> u64 {
        (index as u64) * (self.ring.slot_size() as u64)
    }

    /// Get raw pointer to a frame's data.
    #[inline]
    pub fn frame_ptr(&self, index: usize) -> *mut u8 {
        self.ring.slot_ptr_for_ffi(index)
    }

    /// Access the underlying ring (for advanced usage).
    pub fn ring(&self) -> &SharedMemoryRing {
        &self.ring
    }
}
