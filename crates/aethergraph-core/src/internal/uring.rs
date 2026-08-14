//! io_uring utilities for proper O_DIRECT and SQPOLL usage.
//!
//! This module provides the low-level primitives needed for correct io_uring usage:
//! - O_DIRECT file opening (required for IOPOLL)
//! - SQPOLL-aware submission (check NEED_WAKEUP before syscall)
//!
//! The aligned landing buffers O_DIRECT requires live in
//! [`crate::internal::aligned`].

#![cfg(target_os = "linux")]

use crate::internal::aligned::AlignedBufferPool;
use anyhow::{Context, Result};
use std::fs::File;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use tracing::{debug, trace, warn};

/// Default io_uring SQ/CQ depth.
///
/// Sized to comfortably hold the largest expected single-batch submission
/// (matches the typical RDMA QP send-WR cap). Smaller values (e.g. 256) make
/// `batch_read` thrash the slow-path overflow loop on every batch >SQ-depth,
/// triggering kernel-thread wakeups per push — measured 15× slower on
/// 1024-read batches. Override per-deployment if your workload knows it
/// never exceeds a smaller burst.
pub const DEFAULT_RING_ENTRIES: u32 = 4096;

/// Minimum alignment for O_DIRECT file offsets (512 bytes for most NVMe/SSD).
/// File offsets must be aligned to this value for O_DIRECT reads to succeed.
pub const DIRECT_IO_OFFSET_ALIGNMENT: usize = 512;

/// The file's real O_DIRECT offset alignment, from `statx(STATX_DIOALIGN)`.
///
/// [`DIRECT_IO_OFFSET_ALIGNMENT`] is the historical 512-byte assumption. It
/// is not universal: a 4Kn drive, or a filesystem layered over one, requires
/// 4096, and a layout that satisfies 512 but not 4096 passes the static
/// check and then fails every read with `EINVAL`. `STATX_DIOALIGN` (Linux
/// 6.1) reports what the backing device actually needs, so the compatibility
/// decision can be made against the truth rather than a guess.
///
/// Returns `None` when the kernel or filesystem does not report it — the
/// caller then keeps the conservative default.
pub fn direct_io_offset_alignment(file: &File) -> Option<usize> {
    // STATX_DIOALIGN, from <linux/stat.h>.
    const STATX_DIOALIGN: libc::c_uint = 0x0000_2000;
    // SAFETY: an all-zero statx is a valid out-parameter the kernel fills.
    let mut st: libc::statx = unsafe { std::mem::zeroed() };
    // SAFETY: `file` provides a live fd; with AT_EMPTY_PATH and an empty
    // path the fd itself is the target, and `st` is a valid out-pointer.
    let rc = unsafe {
        libc::statx(
            file.as_raw_fd(),
            c"".as_ptr(),
            libc::AT_EMPTY_PATH,
            STATX_DIOALIGN,
            &mut st,
        )
    };
    if rc != 0 {
        return None;
    }
    // The mask reports what the kernel actually filled in; a kernel that
    // does not know the flag succeeds and simply leaves it clear.
    if st.stx_mask & STATX_DIOALIGN == 0 {
        return None;
    }
    let align = st.stx_dio_offset_align as usize;
    // Zero means the file does not support O_DIRECT at all.
    (align > 0).then_some(align)
}

/// Check if a feature file layout is compatible with O_DIRECT.
///
/// O_DIRECT requires both buffer and file offset alignment. For feature
/// files this means `features_start_offset` and `feature_size` must both be
/// multiples of the device's offset alignment; if either is not, reads fail
/// with `EINVAL`.
///
/// `alignment` comes from [`direct_io_offset_alignment`] where the kernel
/// reports it, and [`DIRECT_IO_OFFSET_ALIGNMENT`] otherwise.
pub fn is_layout_direct_io_compatible_with(
    features_start_offset: u64,
    feature_size: usize,
    alignment: usize,
) -> bool {
    if alignment == 0 {
        return false;
    }
    (features_start_offset as usize).is_multiple_of(alignment)
        && feature_size.is_multiple_of(alignment)
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

/// One ring plus its reusable landing buffers.
///
/// The landing buffers live with the ring under one lock, so per-batch
/// buffer setup amortizes to steady state: a batch lands in the previous
/// batch's allocation, and buffers only grow when a larger batch arrives.
pub struct UringLane {
    pub handle: UringHandle,
    /// Reusable O_DIRECT landing slots.
    pool: Option<AlignedBufferPool>,
    /// Reusable buffered-I/O landing bytes.
    scratch: Vec<u8>,
}

impl UringLane {
    pub fn new(handle: UringHandle) -> Self {
        Self {
            handle,
            pool: None,
            scratch: Vec::new(),
        }
    }

    /// The aligned pool, rebuilt only when the requested geometry outgrows
    /// the cached one.
    ///
    /// A (re)built pool's region is registered with the ring as a fixed
    /// buffer, so reads landing in it use `ReadFixed` — the kernel skips
    /// the per-op page pin/unpin cycle. Registration failure (typically
    /// the locked-memory rlimit) is non-fatal: reads fall back to plain
    /// `Read` on the same buffers.
    pub fn direct_pool(
        &mut self,
        num_slots: usize,
        slot_size: usize,
    ) -> Result<&mut AlignedBufferPool> {
        let rebuild = match &self.pool {
            Some(p) => p.num_slots() < num_slots || p.slot_size() < slot_size,
            None => true,
        };
        if rebuild {
            // The live registration references the allocation replaced
            // below; the kernel must let go of it before it frees.
            self.handle.unregister_buffer_region();
            let mut pool = AlignedBufferPool::try_new(num_slots, slot_size)?;
            // SAFETY: the region lives in `self.pool` beside the handle;
            // this method unregisters before every rebuild, and on drop
            // the handle (declared first) closes the ring before the pool
            // field frees.
            let registered = unsafe {
                self.handle
                    .register_buffer_region(pool.region_ptr(), pool.region_len())
            };
            if let Err(e) = registered {
                warn!("io_uring buffer registration failed ({e}); reads stay unregistered");
            }
            self.pool = Some(pool);
        }
        Ok(self.pool.as_mut().expect("pool populated above"))
    }

    /// The buffered-I/O scratch, grown (never shrunk) to `len` bytes.
    pub fn scratch(&mut self, len: usize) -> &mut [u8] {
        if self.scratch.len() < len {
            self.scratch.resize(len, 0);
        }
        &mut self.scratch[..len]
    }
}

/// Optional CPU to pin SQPOLL kernel threads to, read once from
/// `AETHERGRAPH_SQPOLL_CPU`. Unpinned SQPOLL threads migrate freely and can
/// land on the cores running samplers; deployments that pin their compute
/// threads should park the poller on a housekeeping core.
fn sqpoll_cpu() -> Option<u32> {
    static CPU: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CPU.get_or_init(|| {
        std::env::var("AETHERGRAPH_SQPOLL_CPU")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}

/// Build a ring with no kernel poller, taking the modern setup flags the
/// running kernel supports.
///
/// Two flags apply to a batch submitter like [`batch_read`] and neither
/// constrains which thread submits:
///
/// - `COOP_TASKRUN` (5.19) lets the kernel run completion task-work when
///   the submitting task next enters the kernel, instead of driving it with
///   an IPI. A batch that submits and then waits is already about to enter,
///   so the interrupt buys nothing and costs both cores a round trip.
/// - `SUBMIT_ALL` (5.18) keeps submitting the rest of a batch after one SQE
///   is rejected, rather than stopping at the first. `batch_read` already
///   records the first error and drains everything it submitted, so
///   partial-submit is the behaviour it wants.
///
/// Deliberately absent: `SINGLE_ISSUER`/`DEFER_TASKRUN`, the bigger win of
/// the two. Both require every submission to come from the task that
/// created the ring, and the feature-store lanes are built once and then
/// shared as `Arc<Mutex<UringLane>>` across worker threads — whichever
/// worker takes the lock submits. Enabling them would fail submission with
/// `EEXIST` off the creating thread. Making lanes thread-owned is the
/// prerequisite, and that is a restructuring rather than a flag.
///
/// Each flag is dropped individually if the kernel rejects it, so an older
/// kernel loses only the flags it lacks.
fn build_plain_ring(entries: u32) -> Result<io_uring::IoUring> {
    let attempts: [&[&str]; 3] = [&["coop_taskrun", "submit_all"], &["submit_all"], &[]];
    let mut last_err = None;
    for flags in attempts {
        let mut builder = io_uring::IoUring::builder();
        if flags.contains(&"coop_taskrun") {
            builder.setup_coop_taskrun();
        }
        if flags.contains(&"submit_all") {
            builder.setup_submit_all();
        }
        match builder.build(entries) {
            Ok(ring) => {
                debug!(?flags, "io_uring initialized without a kernel poller");
                return Ok(ring);
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(anyhow::Error::from(last_err.unwrap_or_else(|| {
        std::io::Error::other("io_uring setup failed")
    })))
    .context("failed to create io_uring")
}

/// SQPOLL-aware io_uring wrapper.
///
/// Handles the complexity of SQPOLL mode correctly:
/// - Only calls io_uring_enter() when the kernel thread needs wakeup
/// - Uses registered file descriptors properly
/// - Manages IOPOLL completion polling
pub struct UringHandle {
    ring: io_uring::IoUring,
    /// SQ/CQ depth the ring was built with.
    entries: u32,
    /// Whether SQPOLL is enabled
    sqpoll_enabled: bool,
    /// Whether IOPOLL is enabled
    iopoll_enabled: bool,
    /// Registered file descriptor index (if any)
    registered_fd: Option<u32>,
    /// Registered fixed-buffer region as (base address, length), if any.
    registered_buf: Option<(usize, usize)>,
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
        let mut builder = io_uring::IoUring::builder();
        builder.setup_sqpoll(sqpoll_idle_ms).setup_iopoll();
        if let Some(cpu) = sqpoll_cpu() {
            builder.setup_sqpoll_cpu(cpu);
        }
        match builder.build(entries) {
            Ok(ring) => {
                debug!("io_uring initialized with SQPOLL + IOPOLL");
                Ok(Self {
                    ring,
                    entries,
                    sqpoll_enabled: true,
                    iopoll_enabled: true,
                    registered_fd: None,
                    registered_buf: None,
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
        let mut builder = io_uring::IoUring::builder();
        builder.setup_sqpoll(sqpoll_idle_ms);
        if let Some(cpu) = sqpoll_cpu() {
            builder.setup_sqpoll_cpu(cpu);
        }
        match builder.build(entries) {
            Ok(ring) => {
                debug!("io_uring initialized with SQPOLL (no IOPOLL)");
                Ok(Self {
                    ring,
                    entries,
                    sqpoll_enabled: true,
                    iopoll_enabled: false,
                    registered_fd: None,
                    registered_buf: None,
                })
            }
            Err(e) => {
                warn!("SQPOLL failed ({}), trying standard io_uring", e);
                // Fallback to standard io_uring, with the cheap modern
                // setup flags where the kernel has them.
                let ring = build_plain_ring(entries)?;
                Ok(Self {
                    ring,
                    entries,
                    sqpoll_enabled: false,
                    iopoll_enabled: false,
                    registered_fd: None,
                    registered_buf: None,
                })
            }
        }
    }

    /// SQ/CQ depth this ring was built with.
    pub fn entries(&self) -> u32 {
        self.entries
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

    /// Register one contiguous memory region as fixed buffer index 0.
    ///
    /// Reads targeting memory inside the region can then use `ReadFixed`,
    /// which skips the kernel's per-op get_user_pages pin/unpin cycle.
    /// Any prior registration is replaced.
    ///
    /// # Safety
    /// The region must stay allocated until it is unregistered (via
    /// [`Self::unregister_buffer_region`] or a replacing registration) or
    /// the ring is closed, whichever comes first.
    pub unsafe fn register_buffer_region(&mut self, ptr: *mut u8, len: usize) -> Result<()> {
        self.unregister_buffer_region();
        let iov = libc::iovec {
            iov_base: ptr as *mut libc::c_void,
            iov_len: len,
        };
        // SAFETY: caller guarantees the region outlives the registration.
        unsafe { self.ring.submitter().register_buffers(&[iov]) }
            .context("failed to register fixed buffer region")?;
        self.registered_buf = Some((ptr as usize, len));
        debug!("Registered {len}-byte fixed buffer region");
        Ok(())
    }

    /// Drop the fixed-buffer registration, if any. Idempotent.
    pub fn unregister_buffer_region(&mut self) {
        if self.registered_buf.take().is_some()
            && let Err(e) = self.ring.submitter().unregister_buffers()
        {
            warn!("Failed to unregister fixed buffers: {e}");
        }
    }

    /// Registered fixed-buffer region as (base address, length), if any.
    pub fn registered_buf_range(&self) -> Option<(usize, usize)> {
        self.registered_buf
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
        self.unregister_buffer_region();
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
/// This is the canonical batch-read path — the feature-store gather
/// (`features::gather::uring_gather_rows`) and `async_graph` funnel through
/// here so SQPOLL/short-read fixes apply in one place.
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

    // A negative fd would be handed straight to the kernel as a read target;
    // failures then surface only as CQE errors, and a persistently failing
    // wait can drive the drain loop to `process::abort()`. Reject it up front.
    // (When a registered fd is in use the raw `fd` is ignored — see below — so
    // this only matters for the non-registered path, but checking always is
    // cheap and keeps the contract simple.)
    if handle.registered_fd_index().is_none() {
        anyhow::ensure!(fd >= 0, "batch_read called with invalid fd {fd}");
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

    // In io-uring 0.7, Fd and Fixed are distinct types with no unifying
    // Target, so the choice is inlined at the opcode call site.
    //
    // When a registered fd is present every SQE targets that registered index
    // (`types::Fixed`); the raw `fd` argument is deliberately not used in that
    // mode. Callers must pass the same file they registered — there is no
    // per-call fd override once a fd is registered on the handle.
    let registered_idx = handle.registered_fd_index();
    // Reads landing entirely inside the registered fixed-buffer region use
    // `ReadFixed` (no per-op page pinning); anything else takes plain `Read`
    // on the same ring.
    let registered_buf = handle.registered_buf_range();
    let in_registered_buf = |ptr: *mut u8, len: usize| {
        registered_buf.is_some_and(|(base, region_len)| {
            let p = ptr as usize;
            p >= base && len <= region_len && p - base <= region_len - len
        })
    };
    let is_sqpoll = handle.is_sqpoll();
    // During the push phase, hand accumulated SQEs to the kernel and reap
    // available completions at this cadence — the device starts working
    // while later SQEs are still being built, the queue depth stays high
    // instead of sawtoothing submit-everything → drain-everything, and a
    // batch larger than the CQ can never overflow it.
    let pipeline_stride = (handle.entries() as usize / 4).max(1);

    // Records one CQE against the batch. Every submitted SQE produces
    // exactly one CQE, all of which must be reaped before returning.
    let process_cqe = |cqe: io_uring::cqueue::Entry, first_err: &mut Option<anyhow::Error>| {
        let result = cqe.result();
        let idx = cqe.user_data() as usize;
        let Some(&(offset, _, expected_len)) = reads.get(idx) else {
            if first_err.is_none() {
                *first_err = Some(anyhow::anyhow!(
                    "io_uring completion with unknown user_data {idx}"
                ));
            }
            return;
        };
        if result < 0 {
            if first_err.is_none() {
                *first_err = Some(anyhow::anyhow!(
                    "io_uring read at offset {} failed with error {}",
                    offset,
                    -result
                ));
            }
            return;
        }
        // Short reads on local files mean truncation/corruption.
        if ((result as usize) < expected_len) && first_err.is_none() {
            *first_err = Some(anyhow::anyhow!(
                "io_uring short read at offset {}: got {} bytes, expected {} \
                 (file may be truncated or corrupted)",
                offset,
                result,
                expected_len
            ));
        }
    };

    // Push all reads, counting how many SQEs the kernel will see. Each
    // of them produces exactly one CQE that must be reaped — inline during
    // the push phase or by the drain loop below — before we return.
    let mut first_err: Option<anyhow::Error> = None;
    let mut submitted = 0usize;
    let mut completed = 0usize;
    {
        let (submitter, mut sq, mut cq) = handle.split();

        'push: for (i, &(offset, buf_ptr, len)) in reads.iter().enumerate() {
            let entry = match (registered_idx, in_registered_buf(buf_ptr, len)) {
                (Some(idx), true) => {
                    opcode::ReadFixed::new(types::Fixed(idx), buf_ptr, len as u32, 0)
                        .offset(offset)
                        .build()
                }
                (Some(idx), false) => opcode::Read::new(types::Fixed(idx), buf_ptr, len as u32)
                    .offset(offset)
                    .build(),
                (None, true) => opcode::ReadFixed::new(types::Fd(fd), buf_ptr, len as u32, 0)
                    .offset(offset)
                    .build(),
                (None, false) => opcode::Read::new(types::Fd(fd), buf_ptr, len as u32)
                    .offset(offset)
                    .build(),
            }
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
                        first_err = Some(anyhow::Error::from(e).context("io_uring submit failed"));
                        break 'push;
                    }
                }
            }
            submitted += 1;

            // Pipeline: kick submission and reap whatever has already
            // finished, without blocking.
            if submitted.is_multiple_of(pipeline_stride) {
                sq.sync();
                let res = if is_sqpoll {
                    if sq.need_wakeup() {
                        submitter.submit().map(|_| ())
                    } else {
                        Ok(())
                    }
                } else {
                    submitter.submit().map(|_| ())
                };
                if let Err(e) = res {
                    first_err = Some(anyhow::Error::from(e).context("io_uring submit failed"));
                    break 'push;
                }
                cq.sync();
                for cqe in &mut cq {
                    completed += 1;
                    process_cqe(cqe, &mut first_err);
                }
            }
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

    // Drain the remaining completions, recording (not returning) errors so
    // the kernel is provably done with every buffer first.
    let mut wait_failures = 0u32;
    while completed < submitted {
        handle.cq_sync();

        for cqe in handle.completion() {
            completed += 1;
            process_cqe(cqe, &mut first_err);
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

    /// A layout that clears 512 but not 4096 is the case the static
    /// assumption got wrong: it would be accepted on a 4Kn device and then
    /// fail every read with EINVAL. The decision must follow the alignment
    /// it is given.
    #[test]
    fn layout_compatibility_follows_the_reported_alignment() {
        // 512-aligned but not 4096-aligned.
        let offset = 512u64;
        let size = 1536usize;
        assert!(is_layout_direct_io_compatible_with(offset, size, 512));
        assert!(
            !is_layout_direct_io_compatible_with(offset, size, 4096),
            "a 4Kn device must reject a merely 512-aligned layout"
        );

        // Aligned for both.
        assert!(is_layout_direct_io_compatible_with(8192, 4096, 512));
        assert!(is_layout_direct_io_compatible_with(8192, 4096, 4096));

        // A zero alignment means the file cannot do O_DIRECT at all, and
        // must never divide.
        assert!(!is_layout_direct_io_compatible_with(0, 0, 0));
    }

    /// The query either reports a real power-of-two alignment or declines.
    /// A bogus value would silently disable O_DIRECT or, worse, wave
    /// through a layout the device rejects.
    #[test]
    fn reported_alignment_is_sane_or_absent() {
        let temp = NamedTempFile::new().unwrap();
        let file = File::open(temp.path()).unwrap();
        if let Some(align) = direct_io_offset_alignment(&file) {
            assert!(align.is_power_of_two(), "alignment {align} is not 2^n");
            assert!(align >= 512, "alignment {align} is below a sector");
        }
    }

    /// The setup flags degrade one at a time, so a kernel missing any of
    /// them still yields a usable ring rather than failing construction.
    /// Whatever it settles on must be able to submit and reap — the flags
    /// are an optimization, and a ring that carries them but cannot
    /// complete an operation is worse than one without.
    #[test]
    fn plain_ring_builds_and_completes_regardless_of_flag_support() {
        let mut ring = match build_plain_ring(8) {
            Ok(r) => r,
            // A sandbox with io_uring disabled entirely is not this test's
            // subject.
            Err(e) => {
                println!("io_uring unavailable here ({e}); skipping");
                return;
            }
        };

        // A no-op SQE is enough to prove submission and completion work
        // under whichever flag combination survived.
        let entry = io_uring::opcode::Nop::new().build().user_data(42);
        {
            let mut sq = ring.submission();
            // SAFETY: Nop references no buffers, so there is nothing that
            // has to outlive the call.
            unsafe { sq.push(&entry) }.expect("8-entry ring accepts one SQE");
        }
        ring.submit_and_wait(1).expect("submit_and_wait");
        let cqe = ring.completion().next().expect("one completion");
        assert_eq!(cqe.user_data(), 42);
        assert!(cqe.result() >= 0, "nop failed: {}", cqe.result());
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
    fn registered_buffer_reads_and_pool_rebuild() {
        // Reads landing in the registered pool region take the ReadFixed
        // path; growing the pool must re-register the new region; reads
        // outside the region in the same batch take the plain path. All
        // must return correct data on one lane.
        let mut temp = NamedTempFile::new().unwrap();
        let data: Vec<u8> = (0..65536u32).map(|i| (i % 251) as u8).collect();
        temp.write_all(&data).unwrap();
        temp.flush().unwrap();

        let Some(handle) = create_feature_uring(false) else {
            println!("io_uring not available; skipping");
            return;
        };
        let mut lane = UringLane::new(handle);
        let file = File::open(temp.path()).unwrap();
        let fd = file.as_raw_fd();

        // Two geometries: the second forces a pool rebuild + re-register.
        for (num_slots, slot_size) in [(4usize, 4096usize), (16, 8192)] {
            let pool = lane.direct_pool(num_slots, slot_size).unwrap();
            let mut reads: Vec<(u64, *mut u8, usize)> = Vec::new();
            for i in 0..4 {
                reads.push(((i * 4096) as u64, pool.slot_ptr(i), 4096));
            }
            // One destination outside the registered region.
            let mut outside = vec![0u8; 4096];
            reads.push(((4 * 4096) as u64, outside.as_mut_ptr(), 4096));

            batch_read(&mut lane.handle, fd, &reads).unwrap();

            let pool = lane.direct_pool(num_slots, slot_size).unwrap();
            for i in 0..4 {
                assert_eq!(
                    pool.slot_slice(i, 4096),
                    &data[i * 4096..(i + 1) * 4096],
                    "slot {i} at geometry {num_slots}x{slot_size}"
                );
            }
            assert_eq!(&outside[..], &data[4 * 4096..5 * 4096]);
        }
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
