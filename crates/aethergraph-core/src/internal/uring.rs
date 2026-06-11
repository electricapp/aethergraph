//! io_uring utilities for proper O_DIRECT and SQPOLL usage.
//!
//! This module provides the low-level primitives needed for correct io_uring usage:
//! - O_DIRECT file opening (required for IOPOLL)
//! - Page-aligned buffer allocation (required for O_DIRECT)
//! - SQPOLL-aware submission (check NEED_WAKEUP before syscall)

#![cfg(target_os = "linux")]

use anyhow::{Context, Result};
use std::alloc::{Layout, alloc, dealloc};
use std::fs::File;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ptr::NonNull;
use tracing::{debug, trace, warn};

/// Default alignment for O_DIRECT buffers (4KB, typical page size).
/// NVMe devices may accept 512B alignment, but 4KB is safe for all.
pub const DIRECT_IO_ALIGNMENT: usize = 4096;

/// Default io_uring SQ/CQ depth.
///
/// Sized to comfortably hold the largest expected single-batch submission
/// (matches the typical RDMA QP send-WR cap). Smaller values (e.g. 256) cause
/// `batch_read` to thrash the slow-path overflow loop on every batch >SQ-depth,
/// triggering kernel-thread wakeups per push — measured 15× regression on
/// 1024-read batches before this was raised. Override per-deployment if your
/// workload knows it never exceeds a smaller burst.
pub const DEFAULT_RING_ENTRIES: u32 = 4096;

/// Minimum alignment for O_DIRECT file offsets (512 bytes for most NVMe/SSD).
/// File offsets must be aligned to this value for O_DIRECT reads to succeed.
pub const DIRECT_IO_OFFSET_ALIGNMENT: usize = 512;

/// Check if a feature file layout is compatible with O_DIRECT.
///
/// O_DIRECT requires both buffer and file offset alignment. For feature files,
/// this means:
/// - `features_start_offset` must be aligned to 512 bytes
/// - `feature_size` (feature_dim * 4) must be aligned to 512 bytes
///
/// If either is not aligned, O_DIRECT reads will fail with EINVAL.
pub fn is_layout_direct_io_compatible(features_start_offset: u64, feature_size: usize) -> bool {
    let offset_aligned =
        (features_start_offset as usize).is_multiple_of(DIRECT_IO_OFFSET_ALIGNMENT);
    let size_aligned = feature_size.is_multiple_of(DIRECT_IO_OFFSET_ALIGNMENT);
    offset_aligned && size_aligned
}

/// Open a file with O_DIRECT for use with io_uring IOPOLL.
///
/// O_DIRECT bypasses the page cache, allowing IOPOLL to poll the NVMe
/// device directly for completions instead of waiting for interrupts.
///
/// # Requirements
/// - All reads must use aligned buffers (see `AlignedBuffer`)
/// - Read offsets should ideally be aligned (though many filesystems relax this)
pub fn open_direct(path: impl AsRef<Path>) -> Result<File> {
    let path = path.as_ref();

    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .with_context(|| format!("failed to open {} with O_DIRECT", path.display()))
}

/// Try to open with O_DIRECT, fall back to regular open if unsupported.
///
/// Some filesystems (tmpfs, some network FS) don't support O_DIRECT.
/// This function tries O_DIRECT first and falls back gracefully.
///
/// Returns (File, bool) where the bool indicates if O_DIRECT succeeded.
pub fn open_direct_or_fallback(path: impl AsRef<Path>) -> Result<(File, bool)> {
    let path = path.as_ref();

    match open_direct(path) {
        Ok(file) => {
            debug!("Opened {} with O_DIRECT", path.display());
            Ok((file, true))
        }
        Err(e) => {
            // Check if it's EINVAL (O_DIRECT not supported)
            let is_unsupported = e
                .downcast_ref::<std::io::Error>()
                .map(|io_err| io_err.raw_os_error() == Some(libc::EINVAL))
                .unwrap_or(false);

            if is_unsupported {
                warn!(
                    "O_DIRECT not supported for {}, falling back to buffered I/O",
                    path.display()
                );
                let file = File::open(path)
                    .with_context(|| format!("failed to open {}", path.display()))?;
                Ok((file, false))
            } else {
                Err(e)
            }
        }
    }
}

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

    /// Get the slot size (may be larger than requested due to alignment).
    #[cfg(test)]
    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    /// Get number of slots.
    #[cfg(test)]
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }
}

/// SQPOLL-aware io_uring wrapper.
///
/// Handles the complexity of SQPOLL mode correctly:
/// - Only calls io_uring_enter() when the kernel thread needs wakeup
/// - Uses registered file descriptors properly
/// - Manages IOPOLL completion polling
pub struct UringHandle {
    ring: io_uring::IoUring,
    /// Whether SQPOLL is enabled
    sqpoll_enabled: bool,
    /// Whether IOPOLL is enabled
    iopoll_enabled: bool,
    /// Registered file descriptor index (if any)
    registered_fd: Option<u32>,
}

impl UringHandle {
    /// Create a new io_uring handle with SQPOLL + IOPOLL.
    ///
    /// IOPOLL requires O_DIRECT files with aligned offsets. Use `new_sqpoll_only`
    /// if your file offsets aren't aligned.
    ///
    /// # Arguments
    /// * `entries` - Number of SQ/CQ entries (power of 2, typically 256)
    /// * `sqpoll_idle_ms` - Kernel thread idle timeout in ms (typically 1000)
    pub fn new(entries: u32, sqpoll_idle_ms: u32) -> Result<Self> {
        // Try full SQPOLL + IOPOLL first
        match io_uring::IoUring::builder()
            .setup_sqpoll(sqpoll_idle_ms)
            .setup_iopoll()
            .build(entries)
        {
            Ok(ring) => {
                debug!("io_uring initialized with SQPOLL + IOPOLL");
                Ok(Self {
                    ring,
                    sqpoll_enabled: true,
                    iopoll_enabled: true,
                    registered_fd: None,
                })
            }
            Err(e) => {
                warn!("SQPOLL/IOPOLL failed ({}), trying SQPOLL-only", e);
                // Full fallback chain: SQPOLL+IOPOLL -> SQPOLL-only -> standard io_uring
                Self::new_sqpoll_only(entries, sqpoll_idle_ms)
            }
        }
    }

    /// Create io_uring with SQPOLL only (no IOPOLL).
    ///
    /// Use this when O_DIRECT is not available or file offsets aren't aligned.
    /// SQPOLL still reduces syscalls by having kernel poll the submission queue.
    ///
    /// # Arguments
    /// * `entries` - Number of SQ/CQ entries (power of 2, typically 256)
    /// * `sqpoll_idle_ms` - Kernel thread idle timeout in ms (typically 1000)
    pub fn new_sqpoll_only(entries: u32, sqpoll_idle_ms: u32) -> Result<Self> {
        // Try SQPOLL without IOPOLL
        match io_uring::IoUring::builder()
            .setup_sqpoll(sqpoll_idle_ms)
            .build(entries)
        {
            Ok(ring) => {
                debug!("io_uring initialized with SQPOLL (no IOPOLL)");
                Ok(Self {
                    ring,
                    sqpoll_enabled: true,
                    iopoll_enabled: false,
                    registered_fd: None,
                })
            }
            Err(e) => {
                warn!("SQPOLL failed ({}), trying standard io_uring", e);
                // Fallback to standard io_uring
                let ring = io_uring::IoUring::new(entries).context("failed to create io_uring")?;
                Ok(Self {
                    ring,
                    sqpoll_enabled: false,
                    iopoll_enabled: false,
                    registered_fd: None,
                })
            }
        }
    }

    /// Register a file descriptor for fast access.
    ///
    /// Returns the registered index to use with `types::Fixed`.
    pub fn register_fd(&mut self, file: &File) -> Result<u32> {
        let fd = file.as_raw_fd();
        self.ring
            .submitter()
            .register_files(&[fd])
            .context("failed to register file descriptor")?;

        let index = 0u32; // First registered file
        self.registered_fd = Some(index);
        debug!("Registered fd {} at index {}", fd, index);
        Ok(index)
    }

    /// Check if we have a registered file descriptor.
    pub fn registered_fd_index(&self) -> Option<u32> {
        self.registered_fd
    }

    /// Check if SQPOLL is enabled.
    pub fn is_sqpoll(&self) -> bool {
        self.sqpoll_enabled
    }

    /// Check if IOPOLL is enabled.
    pub fn is_iopoll(&self) -> bool {
        self.iopoll_enabled
    }

    /// Submit entries, respecting SQPOLL mode.
    ///
    /// With SQPOLL: Only calls io_uring_enter if kernel thread needs wakeup.
    /// Without SQPOLL: Always submits.
    ///
    /// This function syncs the SQ internally before checking NEED_WAKEUP to avoid
    /// race conditions where the flag changes between sync and check.
    pub fn submit(&mut self) -> Result<usize> {
        if self.sqpoll_enabled {
            // IMPORTANT: Sync SQ immediately before checking NEED_WAKEUP.
            // This ensures we read the latest flag from shared memory.
            // Without this sync, the flag could be stale, causing missed wakeups.
            self.sq_sync();

            // With SQPOLL, check if kernel thread needs wakeup
            // Must read from the actual SQ shared memory, not from params
            if self.submission().need_wakeup() {
                trace!("SQPOLL kernel thread sleeping, waking up");
                self.ring
                    .submitter()
                    .submit()
                    .context("io_uring submit failed")
            } else {
                // Kernel thread is polling, no syscall needed!
                trace!("SQPOLL active, no syscall needed");
                Ok(0)
            }
        } else {
            self.ring
                .submitter()
                .submit()
                .context("io_uring submit failed")
        }
    }

    /// Submit and wait for at least `want` completions.
    pub fn submit_and_wait(&self, want: usize) -> Result<usize> {
        self.ring
            .submitter()
            .submit_and_wait(want)
            .context("io_uring submit_and_wait failed")
    }

    /// Get access to submission queue (shared proxy; owned wrapper in io-uring 0.7).
    pub fn submission(&mut self) -> io_uring::squeue::SubmissionQueue<'_> {
        // SAFETY: `&mut self` ensures no other reference to the shared SQ exists.
        unsafe { self.ring.submission_shared() }
    }

    /// Get access to completion queue (shared proxy; owned wrapper in io-uring 0.7).
    pub fn completion(&mut self) -> io_uring::cqueue::CompletionQueue<'_> {
        // SAFETY: `&mut self` ensures no other reference to the shared CQ exists.
        unsafe { self.ring.completion_shared() }
    }

    /// Split into (submitter, sq, cq) for concurrent access.
    pub fn split(
        &mut self,
    ) -> (
        io_uring::Submitter<'_>,
        io_uring::squeue::SubmissionQueue<'_>,
        io_uring::cqueue::CompletionQueue<'_>,
    ) {
        self.ring.split()
    }

    /// Sync submission queue.
    pub fn sq_sync(&mut self) {
        self.ring.submission().sync();
    }

    /// Sync completion queue.
    pub fn cq_sync(&mut self) {
        self.ring.completion().sync();
    }
}

impl Drop for UringHandle {
    fn drop(&mut self) {
        // Unregister file descriptor if registered
        if self.registered_fd.is_some() {
            if let Err(e) = self.ring.submitter().unregister_files() {
                // Just log, don't panic in Drop
                warn!("Failed to unregister files: {}", e);
            } else {
                trace!("Unregistered file descriptors");
            }
        }
    }
}

/// Build one `UringHandle` for the feature-read hot path. Picks the best tier
/// the kernel + layout permit:
///
/// 1. `direct_io` + SQPOLL + IOPOLL — zero-syscall NVMe polling
/// 2. SQPOLL only — reduced syscalls, falls here when O_DIRECT is unavailable
///    or IOPOLL construction fails
/// 3. `None` — even SQPOLL refused (kernel too old / missing perms)
///
/// Emits one `debug!` line describing which tier was taken. Caller is
/// responsible for `register_fd` if FD-register is wanted.
pub fn create_feature_uring(direct_io: bool) -> Option<UringHandle> {
    let handle = if direct_io {
        match UringHandle::new(DEFAULT_RING_ENTRIES, 1000) {
            Ok(h) => h,
            Err(e) => {
                warn!("Failed to create io_uring with IOPOLL: {}", e);
                match UringHandle::new_sqpoll_only(DEFAULT_RING_ENTRIES, 1000) {
                    Ok(h) => h,
                    Err(e2) => {
                        warn!("SQPOLL fallback also failed: {}", e2);
                        return None;
                    }
                }
            }
        }
    } else {
        match UringHandle::new_sqpoll_only(DEFAULT_RING_ENTRIES, 1000) {
            Ok(h) => h,
            Err(e) => {
                warn!("Failed to create io_uring: {}", e);
                return None;
            }
        }
    };
    if handle.is_sqpoll() && handle.is_iopoll() && direct_io {
        debug!("io_uring: SQPOLL + IOPOLL + O_DIRECT (near-zero-syscall)");
    } else if handle.is_sqpoll() {
        debug!("io_uring: SQPOLL (reduced syscalls, no IOPOLL)");
    } else {
        debug!("io_uring: standard (batched I/O)");
    }
    Some(handle)
}

/// Perform a batch of reads using io_uring with proper SQPOLL handling.
///
/// Caller-supplied buffers: each tuple is `(file_offset, dest_buffer_ptr, length)`.
/// This is the canonical batch-read path — `prefetch::batch_read_uring_aligned`,
/// `async_store`, and `async_graph` all funnel through here so SQPOLL/short-read
/// fixes apply in one place.
///
/// Every SQE handed to the kernel is reaped before this function returns,
/// success or failure: the kernel may DMA into a read's buffer until its
/// CQE arrives, so returning early on the first failed completion would
/// let the caller free buffers the kernel is still writing. Errors
/// (failed reads, short reads) are recorded and the first one is
/// returned only after all submitted entries have completed. This also
/// keeps the ring clean for the next batch — a leftover CQE would be
/// misattributed to the next batch's `user_data` index space.
///
/// # Safety
/// Each `dest_buffer_ptr` must point to writable memory of at least `length` bytes
/// and remain valid for the duration of this call.
pub fn batch_read(
    handle: &mut UringHandle,
    fd: i32,
    reads: &[(u64, *mut u8, usize)], // (offset, dest_ptr, length)
) -> Result<()> {
    use io_uring::{opcode, types};

    if reads.is_empty() {
        return Ok(());
    }

    // Validate read lengths don't exceed u32::MAX (io_uring limit)
    for &(_, _, len) in reads {
        anyhow::ensure!(
            len <= u32::MAX as usize,
            "read length {} exceeds io_uring limit of {} bytes",
            len,
            u32::MAX
        );
    }

    // Defensive: discard completions left over from a previous batch on
    // this ring. Their user_data indexes a reads slice that no longer
    // exists, so they must not be counted against this batch.
    let mut stale = 0usize;
    handle.cq_sync();
    for _cqe in handle.completion() {
        stale += 1;
    }
    if stale > 0 {
        warn!("io_uring: discarded {stale} stale completions from a previous batch");
    }

    // io-uring 0.7 removed types::Target; Fd and Fixed are distinct types, so the
    // choice has to be inlined at the opcode call site.
    let registered_idx = handle.registered_fd_index();
    let is_sqpoll = handle.is_sqpoll();

    // Push all reads, counting how many SQEs the kernel will see. Each
    // of them produces exactly one CQE that the drain loop below must
    // reap before we return.
    let mut first_err: Option<anyhow::Error> = None;
    let mut submitted = 0usize;
    {
        let (submitter, mut sq, _cq) = handle.split();

        'push: for (i, &(offset, buf_ptr, len)) in reads.iter().enumerate() {
            let entry = match registered_idx {
                Some(idx) => opcode::Read::new(types::Fixed(idx), buf_ptr, len as u32),
                None => opcode::Read::new(types::Fd(fd), buf_ptr, len as u32),
            }
            .offset(offset)
            .build()
            .user_data(i as u64);

            // Push to SQ, submitting if full. A submit failure stops the
            // push phase — entries already pushed are (or will be) in
            // flight and still need draining.
            // SAFETY: the buffers referenced by the SQEs live until the
            // drain loop below has reaped every submitted entry.
            unsafe {
                while sq.push(&entry).is_err() {
                    sq.sync();
                    let res = if is_sqpoll {
                        // Check NEED_WAKEUP from SQ shared memory
                        if sq.need_wakeup() {
                            submitter.submit().map(|_| ())
                        } else {
                            Ok(())
                        }
                    } else {
                        submitter.submit().map(|_| ())
                    };
                    if let Err(e) = res {
                        first_err =
                            Some(anyhow::Error::from(e).context("io_uring submit failed"));
                        break 'push;
                    }
                }
            }
            submitted += 1;
        }

        sq.sync();
    }

    // Kick off the submission (respecting SQPOLL). On failure the pushed
    // entries are still visible to the kernel — SQPOLL consumes them
    // asynchronously and the drain loop's submit_and_wait retries the
    // non-SQPOLL case — so draining remains mandatory.
    if let Err(e) = handle.submit()
        && first_err.is_none()
    {
        first_err = Some(e);
    }

    // Drain exactly `submitted` completions, recording (not returning)
    // errors so the kernel is provably done with every buffer first.
    let mut completed = 0usize;
    let mut wait_failures = 0u32;
    while completed < submitted {
        handle.cq_sync();

        for cqe in handle.completion() {
            completed += 1;
            let result = cqe.result();
            let idx = cqe.user_data() as usize;
            let Some(&(offset, _, expected_len)) = reads.get(idx) else {
                if first_err.is_none() {
                    first_err = Some(anyhow::anyhow!(
                        "io_uring completion with unknown user_data {idx}"
                    ));
                }
                continue;
            };
            if result < 0 {
                if first_err.is_none() {
                    first_err = Some(anyhow::anyhow!(
                        "io_uring read at offset {} failed with error {}",
                        offset,
                        -result
                    ));
                }
                continue;
            }
            // Short reads on local files mean truncation/corruption.
            if ((result as usize) < expected_len) && first_err.is_none() {
                first_err = Some(anyhow::anyhow!(
                    "io_uring short read at offset {}: got {} bytes, expected {} \
                     (file may be truncated or corrupted)",
                    offset,
                    result,
                    expected_len
                ));
            }
        }

        if completed < submitted {
            match handle.submit_and_wait(1) {
                Ok(_) => wait_failures = 0,
                Err(e) => {
                    // Returning here would free buffers the kernel may
                    // still write into. Transient errnos (EINTR/EAGAIN/
                    // EBUSY) clear on retry; if the wait fails persistently
                    // we cannot prove quiescence, and aborting is the only
                    // way to keep the kernel from corrupting freed memory.
                    wait_failures += 1;
                    if wait_failures >= 1000 {
                        tracing::error!(
                            error = %e,
                            completed,
                            submitted,
                            "io_uring wait failed repeatedly with completions \
                             outstanding; aborting to avoid use-after-free"
                        );
                        std::process::abort();
                    }
                    std::thread::yield_now();
                }
            }
        }
    }

    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

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

    #[test]
    fn test_open_fallback() {
        // Create a temp file (on tmpfs, O_DIRECT may not work)
        let mut temp = NamedTempFile::new().unwrap();
        temp.write_all(b"test data").unwrap();

        // Should succeed with fallback
        let (file, direct) = open_direct_or_fallback(temp.path()).unwrap();

        // If O_DIRECT worked, great; if not, we still have a file
        drop(file);
        println!("O_DIRECT supported: {}", direct);
    }

    #[test]
    fn test_uring_handle_creation() {
        // Test that UringHandle::new() succeeds with some configuration
        // Even if SQPOLL/IOPOLL aren't supported, it should fall back gracefully
        let result = UringHandle::new(32, 1000);

        match result {
            Ok(handle) => {
                // Verify the handle reports its capabilities
                println!("SQPOLL enabled: {}", handle.is_sqpoll());
                println!("IOPOLL enabled: {}", handle.is_iopoll());
                // Handle should work regardless of which mode it's in
            }
            Err(e) => {
                // io_uring may not be available at all (e.g., old kernel, container)
                println!("io_uring not available: {}", e);
            }
        }
    }

    #[test]
    fn test_uring_handle_sqpoll_only() {
        // Test SQPOLL-only mode (no IOPOLL)
        let result = UringHandle::new_sqpoll_only(32, 1000);

        match result {
            Ok(handle) => {
                println!("SQPOLL enabled: {}", handle.is_sqpoll());
                assert!(
                    !handle.is_iopoll(),
                    "IOPOLL should be disabled in sqpoll_only mode"
                );
            }
            Err(e) => {
                // io_uring may not be available
                println!("io_uring not available: {}", e);
            }
        }
    }

    #[test]
    fn error_batch_drains_fully_and_ring_stays_usable() {
        // A batch where one read fails (short read past EOF) must still
        // reap every completion before returning, and the same ring must
        // serve the next batch with no stale-completion crosstalk.
        let mut temp = NamedTempFile::new().unwrap();
        let data: Vec<u8> = (0..8192u32).map(|i| (i % 251) as u8).collect();
        temp.write_all(&data).unwrap();
        temp.flush().unwrap();

        let Some(mut handle) = create_feature_uring(false) else {
            println!("io_uring not available; skipping");
            return;
        };
        let file = File::open(temp.path()).unwrap();
        let fd = file.as_raw_fd();

        let mut bufs: Vec<Vec<u8>> = (0..4).map(|_| vec![0u8; 4096]).collect();
        let ptrs: Vec<*mut u8> = bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
        let reads: Vec<(u64, *mut u8, usize)> = vec![
            (0, ptrs[0], 4096),
            (4096, ptrs[1], 4096),
            (1 << 20, ptrs[2], 4096), // past EOF → short read
            (4096, ptrs[3], 4096),
        ];
        let err = batch_read(&mut handle, fd, &reads);
        assert!(err.is_err(), "short read past EOF must surface as Err");

        // Fresh batch on the same handle: must succeed with correct data.
        let mut buf_a = vec![0u8; 4096];
        let mut buf_b = vec![0u8; 4096];
        let reads2: Vec<(u64, *mut u8, usize)> = vec![
            (0, buf_a.as_mut_ptr(), 4096),
            (4096, buf_b.as_mut_ptr(), 4096),
        ];
        batch_read(&mut handle, fd, &reads2).unwrap();
        assert_eq!(&buf_a[..], &data[..4096]);
        assert_eq!(&buf_b[..], &data[4096..8192]);
    }

    #[test]
    fn test_uring_fallback_chain() {
        // Verify the fallback chain works: SQPOLL+IOPOLL -> SQPOLL -> basic io_uring
        // We can't force failures, but we can verify the code paths exist

        // Try full mode
        if let Ok(handle) = UringHandle::new(32, 1000) {
            if handle.is_sqpoll() && handle.is_iopoll() {
                println!("Full SQPOLL+IOPOLL mode available");
            } else if handle.is_sqpoll() {
                println!("SQPOLL only (IOPOLL failed, fell back)");
            } else {
                println!("Basic io_uring (SQPOLL failed, fell back)");
            }
        }

        // Try SQPOLL-only mode
        if let Ok(handle) = UringHandle::new_sqpoll_only(32, 1000) {
            if handle.is_sqpoll() {
                println!("SQPOLL-only mode available");
            } else {
                println!("Basic io_uring (SQPOLL failed, fell back)");
            }
        }
    }
}
