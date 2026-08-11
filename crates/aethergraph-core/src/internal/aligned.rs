//! Page-aligned buffers for O_DIRECT I/O.
//!
//! O_DIRECT requires landing buffers aligned to the filesystem's block size
//! (typically 512 or 4096 bytes). [`AlignedBuffer`] owns one such
//! allocation; [`AlignedBufferPool`] carves a contiguous aligned region
//! into fixed per-read slots for batch operations.

#![cfg(target_os = "linux")]

use anyhow::Result;
use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;

/// Default alignment for O_DIRECT buffers (4KB, typical page size).
/// NVMe devices may accept 512B alignment, but 4KB is safe for all.
pub const DIRECT_IO_ALIGNMENT: usize = 4096;

/// A page-aligned buffer for O_DIRECT I/O.
///
/// O_DIRECT requires buffers to be aligned to the filesystem's block size
/// (typically 512 or 4096 bytes). This struct ensures proper alignment
/// and handles deallocation correctly.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

impl AlignedBuffer {
    /// Allocate a new aligned buffer, returning an error on failure.
    ///
    /// # Arguments
    /// * `len` - Desired buffer size (will be rounded up to alignment)
    /// * `alignment` - Required alignment (default: 4096)
    pub fn try_new(len: usize, alignment: usize) -> Result<Self> {
        // Handle edge case: zero-length buffers
        if len == 0 {
            anyhow::bail!("cannot allocate zero-length buffer");
        }

        // Round up to alignment (checked for overflow)
        let aligned_len = len
            .checked_add(alignment - 1)
            .ok_or_else(|| anyhow::anyhow!("buffer size overflow: {} + {}", len, alignment - 1))?
            & !(alignment - 1);

        let layout = Layout::from_size_align(aligned_len, alignment)
            .map_err(|e| anyhow::anyhow!("invalid layout: {}", e))?;

        // SAFETY: layout is valid and non-zero
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| anyhow::anyhow!("allocation failed: {} bytes", aligned_len))?;

        Ok(Self {
            ptr,
            len: aligned_len,
            layout,
        })
    }

    /// Allocate a new aligned buffer.
    ///
    /// # Arguments
    /// * `len` - Desired buffer size (will be rounded up to alignment)
    /// * `alignment` - Required alignment (default: 4096)
    ///
    /// # Panics
    /// Panics if allocation fails (out of memory).
    #[cfg(test)]
    #[allow(dead_code)]
    #[inline]
    pub fn new(len: usize, alignment: usize) -> Self {
        Self::try_new(len, alignment).expect("AlignedBuffer allocation failed")
    }

    /// Try to allocate with default 4KB alignment.
    pub fn try_new_default(len: usize) -> Result<Self> {
        Self::try_new(len, DIRECT_IO_ALIGNMENT)
    }

    /// Allocate with default 4KB alignment.
    ///
    /// # Panics
    /// Panics if allocation fails.
    #[cfg(test)]
    #[inline]
    pub fn new_default(len: usize) -> Self {
        Self::try_new_default(len).expect("AlignedBuffer allocation failed")
    }

    /// Get a mutable slice of the buffer.
    #[cfg(test)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid for len bytes, properly aligned
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get an immutable slice of the buffer.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for len bytes, properly aligned
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get raw pointer for io_uring operations.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get buffer length.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty.
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated with this layout
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

// SAFETY: AlignedBuffer owns its memory and doesn't share it.
unsafe impl Send for AlignedBuffer {}
// SAFETY: see Send impl above.
unsafe impl Sync for AlignedBuffer {}

/// A collection of aligned buffers for batch operations.
///
/// Pre-allocates a contiguous aligned region and provides slices into it.
pub struct AlignedBufferPool {
    buffer: AlignedBuffer,
    slot_size: usize,
    num_slots: usize,
}

impl AlignedBufferPool {
    /// Create a pool with `num_slots` buffers of `slot_size` bytes each, returning an error on failure.
    pub fn try_new(num_slots: usize, slot_size: usize) -> Result<Self> {
        if num_slots == 0 || slot_size == 0 {
            anyhow::bail!("AlignedBufferPool requires non-zero num_slots and slot_size");
        }

        // Round slot size up to alignment for proper per-slot alignment
        let aligned_slot_size =
            slot_size
                .checked_add(DIRECT_IO_ALIGNMENT - 1)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "slot size overflow: {} + {}",
                        slot_size,
                        DIRECT_IO_ALIGNMENT - 1
                    )
                })?
                & !(DIRECT_IO_ALIGNMENT - 1);

        let total_size = num_slots.checked_mul(aligned_slot_size).ok_or_else(|| {
            anyhow::anyhow!(
                "AlignedBufferPool size overflow: {} slots * {} bytes",
                num_slots,
                aligned_slot_size
            )
        })?;

        Ok(Self {
            buffer: AlignedBuffer::try_new_default(total_size)?,
            slot_size: aligned_slot_size,
            num_slots,
        })
    }

    /// Create a pool with `num_slots` buffers of `slot_size` bytes each.
    ///
    /// # Panics
    /// Panics if total size overflows usize or allocation fails.
    #[cfg(test)]
    #[inline]
    pub fn new(num_slots: usize, slot_size: usize) -> Self {
        Self::try_new(num_slots, slot_size).expect("AlignedBufferPool allocation failed")
    }

    /// Get pointer to slot `index`.
    ///
    /// # Panics
    /// Panics if index >= num_slots or if offset calculation overflows.
    pub fn slot_ptr(&mut self, index: usize) -> *mut u8 {
        assert!(index < self.num_slots, "slot index out of bounds");
        let offset = index
            .checked_mul(self.slot_size)
            .expect("slot offset overflow");
        // SAFETY: offset is within buffer bounds (verified by construction)
        unsafe { self.buffer.as_mut_ptr().add(offset) }
    }

    /// Get slice for slot `index`.
    ///
    /// # Arguments
    /// * `index` - Slot index
    /// * `len` - Actual data length (must be <= slot_size)
    ///
    /// # Panics
    /// Panics if index >= num_slots, len > slot_size, or offset calculation overflows.
    pub fn slot_slice(&self, index: usize, len: usize) -> &[u8] {
        assert!(index < self.num_slots, "slot index out of bounds");
        assert!(len <= self.slot_size, "len exceeds slot size");
        let offset = index
            .checked_mul(self.slot_size)
            .expect("slot offset overflow");
        let end = offset.checked_add(len).expect("slot end overflow");
        &self.buffer.as_slice()[offset..end]
    }

    /// Base pointer of the contiguous region holding every slot.
    pub fn region_ptr(&mut self) -> *mut u8 {
        self.buffer.as_mut_ptr()
    }

    /// Byte length of the contiguous region holding every slot.
    pub fn region_len(&self) -> usize {
        self.buffer.len
    }

    /// Get the slot size (may be larger than requested due to alignment).
    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    /// Get number of slots.
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer() {
        let mut buf = AlignedBuffer::new_default(1000);

        // Should be rounded up to 4096
        assert!(buf.len() >= 1000);
        assert_eq!(buf.len() % DIRECT_IO_ALIGNMENT, 0);

        // Should be aligned
        assert_eq!(buf.as_mut_ptr() as usize % DIRECT_IO_ALIGNMENT, 0);

        // Should be writable
        let slice = buf.as_mut_slice();
        slice[0] = 42;
        assert_eq!(buf.as_slice()[0], 42);
    }

    #[test]
    fn test_buffer_pool() {
        let mut pool = AlignedBufferPool::new(4, 100);

        assert_eq!(pool.num_slots(), 4);
        assert!(pool.slot_size() >= 100);

        // Each slot should be aligned
        for i in 0..4 {
            let ptr = pool.slot_ptr(i);
            assert_eq!(ptr as usize % DIRECT_IO_ALIGNMENT, 0);
        }
    }
}
