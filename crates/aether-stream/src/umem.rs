//! UMEM frame pool — wraps `SharedMemoryRing` as the AF_XDP frame allocator.
//!
//! Each UMEM frame corresponds to one ring buffer slot. The Treiber stack
//! free list provides lock-free frame acquire/release for the ingestion loop.

use aether_mem::hooks::MlockHook;
use aether_mem::{MemoryHook, SharedMemoryRing};

/// AF_XDP UMEM frame pool backed by `SharedMemoryRing`.
///
/// Frame size must be 4096 (see [`Umem::new`]).
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
    /// * `frame_size` - must be 4096
    /// * `extra_hooks` - Additional hooks (e.g. RdmaRegHook for RDMA one-sided reads)
    ///
    /// # Panics
    /// Panics if `frame_size` is not 4096.
    ///
    /// Returns `None` (with an error log) if the ring's page-rounded slot
    /// stride diverges from `frame_size` — on kernels with a page size above
    /// 4 KiB the ring lays frames out at page stride while the kernel is
    /// told `chunk_size = frame_size`, which would corrupt the frame lease
    /// protocol.
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

        let mut hooks: Vec<Box<dyn MemoryHook>> = vec![Box::new(MlockHook::new())];
        hooks.extend(extra_hooks);

        let (ring, hook_failures) =
            SharedMemoryRing::new(frame_count, frame_size as usize, hooks).ok()?;
        for failure in &hook_failures {
            tracing::warn!(error = %failure, "UMEM memory hook failed");
        }

        // The frame lease protocol addresses frames at `ring.slot_size()`
        // stride while the kernel is registered with `chunk_size =
        // frame_size`. SharedMemoryRing rounds slots up to the runtime page
        // size, so on 16K/64K-page kernels the two diverge — every
        // frame_ptr/frame_addr past index 0 would point at the wrong chunk.
        if ring.slot_size() != frame_size as usize {
            tracing::error!(
                ring_slot_size = ring.slot_size(),
                frame_size,
                "UMEM frame stride mismatch: runtime page size rounds ring \
                 slots past the kernel-registered chunk_size; refusing to \
                 build a corrupt frame pool"
            );
            return None;
        }

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
