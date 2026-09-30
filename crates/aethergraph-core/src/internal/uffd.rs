//! Userspace demand paging for out-of-core feature stores via
//! `userfaultfd`.
//!
//! `madvise(MADV_WILLNEED)` is a *hint* — the kernel may ignore it and
//! picks its own eviction victims. This is the opposite: we register an
//! anonymous region in fault-missing mode and become its pager. A fault
//! traps to our thread, which reads the page from the backing store and
//! installs it with `UFFDIO_COPY`; eviction is `MADV_DONTNEED` under a
//! policy the kernel can't express — degree-weighted retention, so a hot
//! high-degree node's pages outlive a cold leaf's.
//!
//! This lets a dataset many times larger than RAM present as one flat
//! mapping the sampler indexes directly, with residency bounded by a
//! caller-set budget.
//!
//! Requires `vm.unprivileged_userfaultfd=1` or `CAP_SYS_PTRACE`;
//! [`PagedRegion::new`] returns an error otherwise and the caller falls
//! back to a plain mmap.

#![cfg(all(target_os = "linux", feature = "uffd"))]

use anyhow::{Context, Result, bail};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// --- userfaultfd ABI (linux/userfaultfd.h) --------------------------------

const UFFD_API: u64 = 0xAA;
const UFFDIO_REGISTER_MODE_MISSING: u64 = 1 << 0;
const UFFD_EVENT_PAGEFAULT: u8 = 0x12;
/// Restrict faults to user-mode accesses — required on hardened kernels
/// for the unprivileged path.
const UFFD_USER_MODE_ONLY: i32 = 1;

const fn ioc(dir: libc::c_ulong, nr: u32, size: usize) -> libc::c_ulong {
    const TYPE: libc::c_ulong = 0xAA; // UFFDIO
    (dir << 30) | ((size as libc::c_ulong) << 16) | (TYPE << 8) | nr as libc::c_ulong
}
const fn iowr(nr: u32, size: usize) -> libc::c_ulong {
    ioc(3, nr, size) // READ|WRITE
}
const fn ior(nr: u32, size: usize) -> libc::c_ulong {
    ioc(2, nr, size)
}

#[repr(C)]
struct UffdioApi {
    api: u64,
    features: u64,
    ioctls: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct UffdioRange {
    start: u64,
    len: u64,
}
#[repr(C)]
struct UffdioRegister {
    range: UffdioRange,
    mode: u64,
    ioctls: u64,
}
#[repr(C)]
struct UffdioCopy {
    dst: u64,
    src: u64,
    len: u64,
    mode: u64,
    copy: i64,
}
#[repr(C)]
struct UffdioZeropage {
    range: UffdioRange,
    mode: u64,
    zeropage: i64,
}
/// `uffd_msg` — 32 bytes. Only the pagefault arm is read; the union is
/// modeled as its largest member (three u64s).
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct UffdMsg {
    event: u8,
    _reserved1: u8,
    _reserved2: u16,
    _reserved3: u32,
    arg0: u64, // pagefault.flags
    arg1: u64, // pagefault.address
    arg2: u64, // pagefault.ptid / padding
}
const _: () = assert!(core::mem::size_of::<UffdMsg>() == 32);

fn uffdio_api() -> libc::c_ulong {
    iowr(0x3F, core::mem::size_of::<UffdioApi>())
}
fn uffdio_register() -> libc::c_ulong {
    iowr(0x00, core::mem::size_of::<UffdioRegister>())
}
fn uffdio_unregister() -> libc::c_ulong {
    ior(0x01, core::mem::size_of::<UffdioRange>())
}
fn uffdio_copy() -> libc::c_ulong {
    iowr(0x03, core::mem::size_of::<UffdioCopy>())
}
fn uffdio_zeropage() -> libc::c_ulong {
    iowr(0x04, core::mem::size_of::<UffdioZeropage>())
}

/// Errnos that clear on retry: interrupted, momentarily out of memory, or
/// racing a concurrent change to the address space.
fn is_transient(errno: Option<i32>) -> bool {
    matches!(
        errno,
        Some(libc::EINTR | libc::EAGAIN | libc::ENOMEM | libc::EBUSY)
    )
}

/// Run `op` until it succeeds or fails with a non-transient error, backing
/// off between attempts. A transient error that persists past the budget
/// (about a second) is returned as permanent.
fn retry_transient<T>(mut op: impl FnMut() -> std::io::Result<T>) -> std::io::Result<T> {
    let mut backoff = Duration::from_micros(50);
    for _ in 0..100 {
        match op() {
            Err(e) if is_transient(e.raw_os_error()) => {
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_millis(10));
            }
            other => return other,
        }
    }
    op()
}

// --- backing store -------------------------------------------------------

/// Source of page contents for a [`PagedRegion`]. Implementors fill the
/// page at a given byte offset; the pager calls this on the fault thread,
/// so it should read from a file or compute deterministically without
/// touching the paged region itself.
pub trait PageSource: Send + Sync {
    /// Fill `page` (exactly one page) with the region's contents starting
    /// at byte `offset` (a page-aligned offset into the region).
    fn fill(&self, offset: u64, page: &mut [u8]) -> Result<()>;
}

/// A [`PageSource`] backed by a file: page N comes from file offset N.
pub struct FileSource {
    file: std::fs::File,
}

impl FileSource {
    pub fn new(file: std::fs::File) -> Self {
        Self { file }
    }
}

impl PageSource for FileSource {
    fn fill(&self, offset: u64, page: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        // A short read is not EOF by itself; keep reading until the page is
        // full or the file ends. Past EOF the tail is zero — the region may
        // be larger than the backing file (sparse feature stores).
        let mut filled = 0;
        while filled < page.len() {
            match self
                .file
                .read_at(&mut page[filled..], offset + filled as u64)
            {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e).context("backing read"),
            }
        }
        page[filled..].fill(0);
        Ok(())
    }
}

// --- the paged region ----------------------------------------------------

/// An anonymous mapping whose pages are demand-loaded from a
/// [`PageSource`] and evicted under a residency budget.
///
/// The region is one flat address range: read any offset and the pager
/// materializes it. Residency is capped at `budget_pages`; installing a
/// page over budget first evicts the lowest-weight resident page that was
/// not among the most recent installs.
///
/// Two conditions make reads return something other than the source's
/// contents, and [`Self::check`] reports both:
///
/// - The pager hit an error it could not retry past. It resolves that
///   fault — and, if it cannot keep serving, every later one — with zero
///   pages rather than leave readers blocked forever, and the failure
///   sticks.
/// - The region was inherited across `fork`. It is mapped `MADV_DONTFORK`,
///   so a child has no mapping there at all and faults instead of reading
///   silent zeros.
///
/// Copying readers should call [`Self::check`] after copying; a borrowed
/// view is only as good as the check made after its last access.
pub struct PagedRegion {
    base: *mut u8,
    len: usize,
    page_size: usize,
    uffd: RawFd,
    owner_pid: u32,
    shutdown: Arc<AtomicBool>,
    health: Arc<PagerHealth>,
    faults: Arc<AtomicU64>,
    evictions: Arc<AtomicU64>,
    pager: Option<std::thread::JoinHandle<()>>,
}

/// Sticky pager failure, set once and never cleared.
#[derive(Default)]
struct PagerHealth {
    failed: AtomicBool,
    reason: Mutex<Option<String>>,
}

impl PagerHealth {
    fn fail(&self, reason: String) {
        let mut slot = self.reason.lock().unwrap_or_else(|p| p.into_inner());
        slot.get_or_insert(reason);
        // Released after the reason is stored, and before the fault it
        // describes is resolved, so a reader that saw the zero page sees
        // the failure too.
        self.failed.store(true, Ordering::Release);
    }
}

// SAFETY: the mapping and uffd are owned; the pager thread is the only
// other accessor and is joined on drop.
unsafe impl Send for PagedRegion {}
// SAFETY: see the Send impl above — the raw pointer and fd are only ever
// read through `&self` slices; the pager thread owns all mutation.
unsafe impl Sync for PagedRegion {}

impl PagedRegion {
    /// Map `len` bytes (rounded up to a page) demand-loaded from `source`,
    /// holding at most `budget_pages` resident. `weights` optionally gives
    /// a per-page retention weight (e.g. node degree); higher weight is
    /// evicted later. Missing/short `weights` default to weight 0.
    ///
    /// `budget_pages` must be at least 2: one access can straddle a page
    /// boundary, and both pages have to be resident at once for it to
    /// complete. Half the budget (at most 64 pages) protects the most
    /// recent installs, which is what lets several threads' straddling
    /// accesses complete concurrently.
    pub fn new(
        len: usize,
        budget_pages: usize,
        source: Arc<dyn PageSource>,
        weights: Arc<PageWeights>,
    ) -> Result<Self> {
        let page_size = page_size();
        let len = len.next_multiple_of(page_size);
        if len == 0 {
            bail!("PagedRegion length must be > 0");
        }
        if budget_pages < 2 {
            bail!(
                "budget_pages must be at least 2 (got {budget_pages}): an access \
                 straddling two pages needs both resident"
            );
        }

        // Create the userfaultfd. O_CLOEXEC | O_NONBLOCK; USER_MODE_ONLY
        // for the unprivileged path, retried without it on EINVAL (older
        // kernels don't know the flag).
        let uffd = create_uffd()?;

        // Handshake the API version and confirm the kernel offers it.
        let mut api = UffdioApi {
            api: UFFD_API,
            features: 0,
            ioctls: 0,
        };
        // SAFETY: `uffd` is a fresh userfaultfd; `api` is a valid in/out arg.
        if unsafe { libc::ioctl(uffd, uffdio_api(), &mut api) } != 0 {
            let e = std::io::Error::last_os_error();
            close_uffd(uffd);
            bail!("UFFDIO_API failed: {e}");
        }

        // Anonymous mapping to be paged.
        // SAFETY: standard anonymous mmap; ptr checked below.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            let e = std::io::Error::last_os_error();
            close_uffd(uffd);
            bail!("mmap failed: {e}");
        }
        let base = base as *mut u8;

        // A forked child inherits neither the pager thread nor (without
        // EVENT_FORK) the registration, so its faults on this range would
        // zero-fill silently. Leaving the range out of the child makes an
        // access there fault loudly instead.
        // SAFETY: `base`/`len` is exactly the mapping just created.
        if unsafe { libc::madvise(base as *mut libc::c_void, len, libc::MADV_DONTFORK) } != 0 {
            let e = std::io::Error::last_os_error();
            unmap_region(base, len);
            close_uffd(uffd);
            bail!("MADV_DONTFORK failed: {e}");
        }

        // Register the whole range in missing-fault mode. The kernel
        // writes the supported ioctls back into `reg`.
        let mut reg = UffdioRegister {
            range: UffdioRange {
                start: base as u64,
                len: len as u64,
            },
            mode: UFFDIO_REGISTER_MODE_MISSING,
            ioctls: 0,
        };
        // SAFETY: the range is exactly the mapping just created; `reg` is a
        // valid in/out argument.
        if unsafe { libc::ioctl(uffd, uffdio_register(), &mut reg) } != 0 {
            let e = std::io::Error::last_os_error();
            unmap_region(base, len);
            close_uffd(uffd);
            bail!("UFFDIO_REGISTER failed: {e}");
        }

        let shutdown = Arc::new(AtomicBool::new(false));
        let health = Arc::new(PagerHealth::default());
        let faults = Arc::new(AtomicU64::new(0));
        let evictions = Arc::new(AtomicU64::new(0));

        let spawned = {
            let shutdown = Arc::clone(&shutdown);
            let health = Arc::clone(&health);
            let faults = Arc::clone(&faults);
            let evictions = Arc::clone(&evictions);
            let base_addr = base as usize;
            std::thread::Builder::new()
                .name("aether-uffd-pager".into())
                .spawn(move || {
                    let mut pager = Pager {
                        uffd,
                        base: base_addr,
                        len,
                        page_size,
                        budget_pages,
                        source,
                        weights,
                        resident: HashMap::new(),
                        evict_queue: BinaryHeap::new(),
                        recent: VecDeque::new(),
                        protect: (budget_pages / 2).clamp(1, RECENT_INSTALLS),
                        health,
                        faults,
                        evictions,
                    };
                    pager.run(&shutdown);
                })
        };
        let pager = match spawned {
            Ok(p) => p,
            Err(e) => {
                unmap_region(base, len);
                close_uffd(uffd);
                return Err(e).context("spawn pager thread");
            }
        };

        Ok(Self {
            base,
            len,
            page_size,
            uffd,
            owner_pid: std::process::id(),
            shutdown,
            health,
            faults,
            evictions,
            pager: Some(pager),
        })
    }

    /// Whether reads through this region return the source's contents:
    /// an error if the pager failed permanently (it then serves zero pages)
    /// or if this is a forked copy of the region.
    pub fn check(&self) -> Result<()> {
        if std::process::id() != self.owner_pid {
            bail!(
                "paged feature region used in a forked child; its pager lives in \
                 the parent — reopen the store in the child"
            );
        }
        if self.health.failed.load(Ordering::Acquire) {
            let reason = self
                .health
                .reason
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone()
                .unwrap_or_default();
            bail!("paged feature region's pager failed; reads may be zero-filled: {reason}");
        }
        Ok(())
    }

    /// The region as a read-only slice. Touching any byte demand-loads its
    /// page through the pager.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `base..base+len` is the registered mapping; reads fault
        // in through the pager, which installs a full page before the
        // access completes.
        unsafe { std::slice::from_raw_parts(self.base, self.len) }
    }

    /// Number of page faults serviced so far.
    pub fn fault_count(&self) -> u64 {
        self.faults.load(Ordering::Relaxed)
    }

    /// Number of evictions performed so far.
    pub fn eviction_count(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }

    /// Page size the region was built with.
    pub fn page_size(&self) -> usize {
        self.page_size
    }
}

impl Drop for PagedRegion {
    fn drop(&mut self) {
        if std::process::id() != self.owner_pid {
            // A forked copy: the pager thread and the mapping exist only in
            // the parent, and joining a thread the child never had would
            // block forever. Only the fd is the child's to close.
            if let Some(pager) = self.pager.take() {
                std::mem::forget(pager);
            }
            close_uffd(self.uffd);
            return;
        }
        self.shutdown.store(true, Ordering::SeqCst);
        // Nudge the pager off its poll and join it before unmapping, so no
        // fault handling races the munmap.
        if let Some(pager) = self.pager.take() {
            let _ = pager.join();
        }
        // The pager is joined; `base`/`len` and `uffd` are ours to release.
        unmap_region(self.base, self.len);
        close_uffd(self.uffd);
    }
}

/// Per-page retention weights. Higher weight evicts later; the canonical
/// fill is node degree so hot high-degree nodes stay resident.
#[derive(Default)]
pub struct PageWeights {
    weights: Vec<u32>,
}

impl PageWeights {
    /// Weights indexed by page number. Pages beyond the vector weigh 0.
    pub fn from_vec(weights: Vec<u32>) -> Self {
        Self { weights }
    }

    /// Build page weights from per-node degrees for a feature store whose
    /// payload starts at `payload_offset` with `row_bytes` per node.
    ///
    /// A page's weight is the highest degree among the rows it holds: one
    /// hub node is enough to keep its page resident, which is the point —
    /// pages are evicted by their most valuable occupant, not an average
    /// that a hub's cold neighbors would dilute.
    ///
    /// Rows straddling a page boundary contribute to both pages.
    pub fn from_node_degrees(
        degrees: &[u32],
        payload_offset: u64,
        row_bytes: usize,
        page_size: usize,
    ) -> Self {
        if row_bytes == 0 || page_size == 0 {
            return Self::default();
        }
        let end_byte = payload_offset + (degrees.len() as u64) * (row_bytes as u64);
        let num_pages = usize::try_from(end_byte.div_ceil(page_size as u64)).unwrap_or(usize::MAX);
        let mut weights = vec![0u32; num_pages];

        for (node, &degree) in degrees.iter().enumerate() {
            let start = payload_offset + (node as u64) * (row_bytes as u64);
            let last = start + row_bytes as u64 - 1;
            let first_page = (start / page_size as u64) as usize;
            let last_page = (last / page_size as u64) as usize;
            for w in &mut weights[first_page..=last_page.min(num_pages - 1)] {
                *w = (*w).max(degree);
            }
        }
        Self { weights }
    }

    fn weight(&self, page: usize) -> u32 {
        self.weights.get(page).copied().unwrap_or(0)
    }
}

/// Most recent installs shielded from eviction. The pager cannot see
/// accesses, only faults, so a page it just installed may not have been
/// read yet: evicting it could undo the install before the faulting access
/// retries. An access straddling two pages needs both, so each concurrently
/// faulting thread needs two of these.
const RECENT_INSTALLS: usize = 64;

/// The pager thread's private state.
struct Pager {
    uffd: RawFd,
    base: usize,
    len: usize,
    page_size: usize,
    budget_pages: usize,
    source: Arc<dyn PageSource>,
    weights: Arc<PageWeights>,
    /// Resident page number → its retention weight.
    resident: HashMap<usize, u32>,
    /// Eviction candidates ordered by weight, lowest first. Entries are
    /// never removed on eviction — a pop whose page is no longer resident
    /// is stale and discarded — so a page installed, evicted, and
    /// installed again can hold several entries. They all carry the same
    /// weight, which is a pure function of the page, so whichever survives
    /// is the right one. The queue is rebuilt from `resident` when the
    /// stale entries outgrow the live ones.
    evict_queue: BinaryHeap<Reverse<(u32, usize)>>,
    /// The last `protect` pages installed, oldest first; never evicted.
    recent: VecDeque<usize>,
    /// Half the budget, between 1 and [`RECENT_INSTALLS`]: the newest
    /// install always survives, and the other half of the budget is still
    /// retained by weight.
    protect: usize,
    health: Arc<PagerHealth>,
    faults: Arc<AtomicU64>,
    evictions: Arc<AtomicU64>,
}

impl Pager {
    fn run(&mut self, shutdown: &AtomicBool) {
        let mut pollfd = libc::pollfd {
            fd: self.uffd,
            events: libc::POLLIN,
            revents: 0,
        };
        // Reusable page-sized staging buffer for UFFDIO_COPY sources.
        let mut staging = vec![0u8; self.page_size];

        while !shutdown.load(Ordering::SeqCst) {
            // Short timeout so shutdown is observed promptly.
            // SAFETY: single valid pollfd.
            let n = unsafe { libc::poll(&mut pollfd, 1, 100) };
            if n <= 0 {
                continue;
            }
            let mut msg = UffdMsg::default();
            // SAFETY: read one uffd_msg from the userfaultfd.
            let r = unsafe {
                libc::read(
                    self.uffd,
                    &mut msg as *mut UffdMsg as *mut libc::c_void,
                    core::mem::size_of::<UffdMsg>(),
                )
            };
            if r != core::mem::size_of::<UffdMsg>() as isize {
                continue;
            }
            if msg.event != UFFD_EVENT_PAGEFAULT {
                continue;
            }
            let Err(e) = self.service_fault(msg.arg1, &mut staging) else {
                continue;
            };
            // The faulting thread stays blocked until its page is
            // installed. Record the failure first, so a reader that gets
            // past the fault also sees why its data is zeros, then give it
            // a zero page.
            tracing::error!(error = %e, "uffd pager could not load a page");
            self.health.fail(format!("{e:#}"));
            if let Err(e) = self.install_zero_page(msg.arg1) {
                // Nothing can be installed. Unregistering wakes every
                // blocked reader, and later faults zero-fill without the
                // pager — the failure recorded above reports it.
                tracing::error!(error = %e, "uffd pager releasing the region");
                self.unregister_all();
                break;
            }
        }
    }

    fn page_of(&self, fault_addr: u64) -> (usize, usize) {
        let page_start = (fault_addr as usize) & !(self.page_size - 1);
        (page_start, (page_start - self.base) / self.page_size)
    }

    fn service_fault(&mut self, fault_addr: u64, staging: &mut [u8]) -> Result<()> {
        let (page_start, page_no) = self.page_of(fault_addr);
        let offset = (page_no * self.page_size) as u64;

        // Evict down to budget-1 before installing the newcomer. A pass
        // that cannot free anything ends the loop: every candidate is
        // protected, and retrying would spin the pager thread forever with
        // every faulting reader blocked behind it.
        while self.resident.len() >= self.budget_pages {
            if !self.evict_one(page_no)? {
                break;
            }
        }

        let mut attempts = 0;
        loop {
            match self.source.fill(offset, staging) {
                Ok(()) => break,
                Err(e) => {
                    let transient = e
                        .chain()
                        .filter_map(|c| c.downcast_ref::<std::io::Error>())
                        .any(|io| is_transient(io.raw_os_error()));
                    attempts += 1;
                    if !transient || attempts >= 100 {
                        return Err(e);
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
        // Count the fault and record residency *before* UFFDIO_COPY: the
        // copy is what unblocks the faulting thread, so a reader that sees
        // its access complete also sees this fault reflected in the
        // counters (the copy ioctl and the thread's resume bracket the
        // store with kernel barriers).
        self.record_install(page_no);
        self.faults.fetch_add(1, Ordering::Relaxed);
        let uffd = self.uffd;
        let result = retry_transient(|| {
            let mut copy = UffdioCopy {
                dst: page_start as u64,
                src: staging.as_ptr() as u64,
                len: self.page_size as u64,
                mode: 0,
                copy: 0,
            };
            // SAFETY: `dst` is the faulting page inside the registered
            // range; `src` is a full page of staging owned by this thread;
            // `copy` is a valid in/out argument.
            if unsafe { libc::ioctl(uffd, uffdio_copy(), &mut copy) } == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        });
        match result {
            // EEXIST: another thread's access already installed it — the
            // fault is resolved either way.
            Err(e) if e.raw_os_error() != Some(libc::EEXIST) => bail!("UFFDIO_COPY failed: {e}"),
            _ => Ok(()),
        }
    }

    /// Record `page_no` as resident and among the protected recent
    /// installs.
    fn record_install(&mut self, page_no: usize) {
        let weight = self.weights.weight(page_no);
        if self.resident.insert(page_no, weight).is_none() {
            self.evict_queue.push(Reverse((weight, page_no)));
        }
        self.recent.retain(|&p| p != page_no);
        self.recent.push_back(page_no);
        while self.recent.len() > self.protect {
            self.recent.pop_front();
        }
    }

    /// Resolve the fault at `fault_addr` with a zero page, for when its
    /// contents cannot be loaded.
    fn install_zero_page(&mut self, fault_addr: u64) -> Result<()> {
        let (page_start, page_no) = self.page_of(fault_addr);
        self.record_install(page_no);
        let uffd = self.uffd;
        let result = retry_transient(|| {
            let mut zero = UffdioZeropage {
                range: UffdioRange {
                    start: page_start as u64,
                    len: self.page_size as u64,
                },
                mode: 0,
                zeropage: 0,
            };
            // SAFETY: the range is one page inside the registered mapping;
            // `zero` is a valid in/out argument.
            if unsafe { libc::ioctl(uffd, uffdio_zeropage(), &mut zero) } == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        });
        match result {
            Err(e) if e.raw_os_error() != Some(libc::EEXIST) => {
                bail!("UFFDIO_ZEROPAGE failed: {e}")
            }
            _ => Ok(()),
        }
    }

    /// Drop the whole range's registration, waking every blocked fault.
    fn unregister_all(&self) {
        let range = UffdioRange {
            start: self.base as u64,
            len: self.len as u64,
        };
        // SAFETY: the range is exactly the registered mapping.
        if unsafe { libc::ioctl(self.uffd, uffdio_unregister(), &range) } != 0 {
            tracing::error!(
                error = %std::io::Error::last_os_error(),
                "UFFDIO_UNREGISTER failed; readers of unloaded pages stay blocked"
            );
        }
    }

    /// Evict the lowest-weight resident page that is neither `keep` (the
    /// page about to be installed) nor a protected recent install.
    /// `MADV_DONTNEED` drops it and re-arms its fault.
    ///
    /// Returns whether a page was actually evicted. `false` means every
    /// candidate left is protected, and the caller must stop asking.
    ///
    /// Pulling the victim from a weight-ordered queue keeps this
    /// O(log n): scanning `resident` for the minimum would make every
    /// fault cost a pass over the whole residency budget, which for a
    /// multi-gigabyte region is the dominant cost of servicing a fault.
    fn evict_one(&mut self, keep: usize) -> Result<bool> {
        let mut set_aside = Vec::new();
        let victim = loop {
            let Some(Reverse((weight, page))) = self.evict_queue.pop() else {
                break None;
            };
            if self.resident.get(&page) != Some(&weight) {
                // Stale: the page was evicted since this entry was pushed.
                continue;
            }
            if page == keep || self.recent.contains(&page) {
                set_aside.push(Reverse((weight, page)));
                continue;
            }
            break Some(page);
        };
        for entry in set_aside {
            self.evict_queue.push(entry);
        }
        let Some(victim) = victim else {
            return Ok(false);
        };
        let addr = self.base + victim * self.page_size;
        retry_transient(|| {
            // SAFETY: `addr` is a page inside the registered mapping.
            let ret = unsafe {
                libc::madvise(
                    addr as *mut libc::c_void,
                    self.page_size,
                    libc::MADV_DONTNEED,
                )
            };
            if ret == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        })
        .context("MADV_DONTNEED failed")?;
        self.resident.remove(&victim);
        self.evictions.fetch_add(1, Ordering::Relaxed);

        // Reclaim the queue when stale entries outnumber live ones, so
        // repeated install/evict churn on the same pages cannot grow it
        // without bound.
        if self.evict_queue.len() > 2 * self.resident.len().max(1) {
            self.evict_queue = self
                .resident
                .iter()
                .map(|(&page, &weight)| Reverse((weight, page)))
                .collect();
        }
        Ok(true)
    }
}

/// The system page size — the granularity of both faulting and eviction,
/// so callers sizing a residency budget or building [`PageWeights`] need
/// the same number the region uses.
pub fn page_size() -> usize {
    // SAFETY: sysconf with a constant name is always valid.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 { v as usize } else { 4096 }
}

/// Close a userfaultfd owned by the caller (teardown paths).
fn close_uffd(fd: RawFd) {
    // SAFETY: `fd` is a userfaultfd the caller owns and is discarding.
    unsafe { libc::close(fd) };
}

/// Unmap a region the caller created (teardown paths).
fn unmap_region(base: *mut u8, len: usize) {
    // SAFETY: `base`/`len` describe a mapping the caller created and owns.
    unsafe { libc::munmap(base as *mut libc::c_void, len) };
}

fn create_uffd() -> Result<RawFd> {
    let flags = libc::O_CLOEXEC | libc::O_NONBLOCK;
    // Try USER_MODE_ONLY first (hardened kernels require it unprivileged);
    // fall back without it on EINVAL.
    for extra in [UFFD_USER_MODE_ONLY, 0] {
        // SAFETY: userfaultfd(2) via raw syscall; flags are valid.
        let fd = unsafe { libc::syscall(libc::SYS_userfaultfd, flags | extra) };
        if fd >= 0 {
            return Ok(fd as RawFd);
        }
        let e = std::io::Error::last_os_error();
        if e.raw_os_error() == Some(libc::EINVAL) && extra != 0 {
            continue; // retry without USER_MODE_ONLY
        }
        bail!(
            "userfaultfd() failed: {e} (need vm.unprivileged_userfaultfd=1 \
             or CAP_SYS_PTRACE)"
        );
    }
    unreachable!("loop returns or bails")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn uffd_available() -> bool {
        match create_uffd() {
            Ok(fd) => {
                // SAFETY: fd is a fresh userfaultfd we own.
                unsafe { libc::close(fd) };
                true
            }
            Err(_) => false,
        }
    }

    #[test]
    fn demand_pages_from_backing_file() {
        if !uffd_available() {
            eprintln!("userfaultfd unavailable (set vm.unprivileged_userfaultfd=1); skipping");
            return;
        }
        let ps = page_size();
        let npages = 64usize;

        // Backing file: page p filled with byte (p % 251).
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        for p in 0..npages {
            tmp.write_all(&vec![(p % 251) as u8; ps]).unwrap();
        }
        tmp.flush().unwrap();
        let file = tmp.reopen().unwrap();

        let source = Arc::new(FileSource::new(file));
        let weights = Arc::new(PageWeights::default());
        let region = PagedRegion::new(npages * ps, 8, source, weights).unwrap();
        let data = region.as_slice();

        // Touch every page out of order; each first byte must match.
        let order = [0, 63, 7, 31, 8, 62, 1, 40, 16, 55, 3, 50];
        for &p in &order {
            assert_eq!(data[p * ps], (p % 251) as u8, "page {p} first byte");
            // A mid-page byte too, to confirm the whole page installed.
            assert_eq!(data[p * ps + ps / 2], (p % 251) as u8, "page {p} mid");
        }

        // Residency stayed within budget → evictions must have happened
        // (we touched 12 distinct pages with an 8-page budget).
        assert!(region.fault_count() >= order.len() as u64);
        assert!(
            region.eviction_count() > 0,
            "touching 12 pages with budget 8 must evict"
        );
    }

    fn page_file(npages: usize) -> std::fs::File {
        let ps = page_size();
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        for p in 0..npages {
            tmp.write_all(&vec![(p % 251) as u8; ps]).unwrap();
        }
        tmp.flush().unwrap();
        tmp.reopen().unwrap()
    }

    /// Run `f` on its own thread, failing (instead of hanging the suite)
    /// if it does not finish: a livelocked pager blocks its reader forever.
    fn within_deadline<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(Duration::from_secs(20))
            .expect("read did not complete: the pager is stuck")
    }

    /// One 8-byte load spanning `data[boundary - 4..boundary + 4]`, so it
    /// touches both pages in a single instruction.
    fn load_across(data: &[u8], boundary: usize) -> [u8; 8] {
        let span = &data[boundary - 4..boundary + 4];
        // SAFETY: `span` is 8 in-bounds bytes; the read is unaligned-safe.
        unsafe { (span.as_ptr() as *const u64).read_unaligned() }.to_ne_bytes()
    }

    /// One access can straddle a page boundary and needs both pages at
    /// once, so a single-page budget can never serve it.
    #[test]
    fn budget_below_two_pages_is_refused() {
        if !uffd_available() {
            eprintln!("userfaultfd unavailable; skipping");
            return;
        }
        let source = Arc::new(FileSource::new(page_file(4)));
        let err = PagedRegion::new(4 * page_size(), 1, source, Arc::default())
            .err()
            .expect("a one-page budget must be refused");
        assert!(err.to_string().contains("at least 2"), "{err}");
    }

    /// A load straddling two pages faults them one at a time and completes
    /// only once both are resident. With uniform weights the two are also
    /// the lowest-numbered — the first eviction candidates — so without
    /// protection each install evicts the other and the access retries
    /// forever. A tight two-page budget is the sharpest case.
    #[test]
    fn straddling_access_completes_under_a_tight_budget() {
        if !uffd_available() {
            eprintln!("userfaultfd unavailable; skipping");
            return;
        }
        let ps = page_size();
        let npages = 8usize;
        let source = Arc::new(FileSource::new(page_file(npages)));
        let region = Arc::new(PagedRegion::new(npages * ps, 2, source, Arc::default()).unwrap());

        let r = Arc::clone(&region);
        within_deadline(move || {
            let data = r.as_slice();
            // Fill the budget with high-numbered pages first.
            assert_eq!(data[5 * ps], 5);
            assert_eq!(data[6 * ps], 6);
            for boundary in 1..npages {
                let bytes = load_across(data, boundary * ps);
                assert_eq!(bytes[..4], [((boundary - 1) % 251) as u8; 4]);
                assert_eq!(bytes[4..], [(boundary % 251) as u8; 4]);
            }
        });
        region.check().unwrap();
    }

    /// Many threads straddling different boundaries at once, with room for
    /// every thread's pair.
    #[test]
    fn concurrent_straddling_accesses_all_complete() {
        if !uffd_available() {
            eprintln!("userfaultfd unavailable; skipping");
            return;
        }
        let ps = page_size();
        let npages = 64usize;
        let source = Arc::new(FileSource::new(page_file(npages)));
        let region = Arc::new(PagedRegion::new(npages * ps, 16, source, Arc::default()).unwrap());

        let r = Arc::clone(&region);
        within_deadline(move || {
            std::thread::scope(|s| {
                for t in 0..4usize {
                    let r = &r;
                    s.spawn(move || {
                        let data = r.as_slice();
                        for round in 0..64 {
                            let boundary = 1 + (round * 7 + t * 13) % (npages - 1);
                            let bytes = load_across(data, boundary * ps);
                            assert_eq!(bytes[4], (boundary % 251) as u8);
                        }
                    });
                }
            });
        });
        assert!(
            region.eviction_count() > 0,
            "64 pages under a 16-page budget"
        );
        region.check().unwrap();
    }

    /// A source that fails transiently a few times, then permanently for
    /// one page.
    struct FlakySource {
        inner: FileSource,
        transient_left: std::sync::atomic::AtomicU32,
        broken_page: u64,
    }

    impl PageSource for FlakySource {
        fn fill(&self, offset: u64, page: &mut [u8]) -> Result<()> {
            if offset == self.broken_page * page.len() as u64 {
                return Err(std::io::Error::from_raw_os_error(libc::EIO))
                    .context("simulated bad sector");
            }
            if self
                .transient_left
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
                .is_ok()
            {
                return Err(std::io::Error::from_raw_os_error(libc::EAGAIN).into());
            }
            self.inner.fill(offset, page)
        }
    }

    /// Transient source errors are retried invisibly. A permanent one must
    /// neither hang the reader nor pass silently: the reader gets zeros,
    /// the region reports the failure from then on, and other pages keep
    /// loading.
    #[test]
    fn source_failures_are_retried_or_reported_never_hung() {
        if !uffd_available() {
            eprintln!("userfaultfd unavailable; skipping");
            return;
        }
        let ps = page_size();
        let npages = 8usize;
        let source = Arc::new(FlakySource {
            inner: FileSource::new(page_file(npages)),
            transient_left: std::sync::atomic::AtomicU32::new(3),
            broken_page: 3,
        });
        let region = Arc::new(PagedRegion::new(npages * ps, 4, source, Arc::default()).unwrap());

        let r = Arc::clone(&region);
        let broken_first_byte = within_deadline(move || {
            let data = r.as_slice();
            assert_eq!(data[ps], 1, "a page behind transient errors loads");
            r.check().unwrap();
            data[3 * ps]
        });
        assert_eq!(broken_first_byte, 0, "an unloadable page reads as zeros");
        let err = region.check().unwrap_err();
        assert!(err.to_string().contains("simulated bad sector"), "{err:#}");

        let r = Arc::clone(&region);
        let later = within_deadline(move || r.as_slice()[5 * ps]);
        assert_eq!(later, 5, "the pager keeps serving other pages");
        assert!(region.check().is_err(), "the failure is sticky");
    }

    /// A forked child has no pager: the region must not be mapped there,
    /// so a child access faults rather than zero-filling silently.
    #[test]
    fn forked_child_cannot_read_the_region_silently() {
        if !uffd_available() {
            eprintln!("userfaultfd unavailable; skipping");
            return;
        }
        let ps = page_size();
        let source = Arc::new(FileSource::new(page_file(4)));
        let region = PagedRegion::new(4 * ps, 4, source, Arc::default()).unwrap();
        assert_eq!(region.as_slice()[ps], 1, "resident in the parent");
        let addr = region.as_slice().as_ptr() as usize + ps;

        // SAFETY: the child only touches memory and exits, without
        // allocating or taking locks another thread may have held at fork.
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork failed");
        if pid == 0 {
            // SAFETY: deliberately reads an address that must be unmapped
            // in the child; a SIGSEGV here is the expected outcome.
            let byte = unsafe { std::ptr::read_volatile(addr as *const u8) };
            // SAFETY: `_exit` is async-signal-safe.
            unsafe { libc::_exit(i32::from(byte) + 1) };
        }
        let mut status = 0;
        // SAFETY: `pid` is our child; `status` is a valid out-pointer.
        assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
        assert!(
            libc::WIFSIGNALED(status) && libc::WTERMSIG(status) == libc::SIGSEGV,
            "child read the region instead of faulting (status {status:#x})"
        );
        region.check().unwrap();
    }

    #[test]
    fn degree_weighted_retention_keeps_hot_pages() {
        if !uffd_available() {
            eprintln!("userfaultfd unavailable; skipping");
            return;
        }
        let ps = page_size();
        let npages = 16usize;

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        for p in 0..npages {
            tmp.write_all(&vec![p as u8; ps]).unwrap();
        }
        tmp.flush().unwrap();
        let file = tmp.reopen().unwrap();

        // Page 0 has a huge weight; everything else weight 0. It should
        // survive eviction pressure once resident.
        let mut w = vec![0u32; npages];
        w[0] = u32::MAX;
        let source = Arc::new(FileSource::new(file));
        let weights = Arc::new(PageWeights::from_vec(w));
        let region = PagedRegion::new(npages * ps, 4, source, weights).unwrap();
        let data = region.as_slice();

        // Make page 0 resident, then churn the others past the budget.
        assert_eq!(data[0], 0);
        for p in 1..npages {
            let _ = data[p * ps];
        }
        // Re-touching page 0 should not fault again if it was retained.
        let before = region.fault_count();
        assert_eq!(data[0], 0);
        let after = region.fault_count();
        assert_eq!(
            after, before,
            "high-weight page 0 must not have been evicted"
        );
    }

    #[test]
    fn degree_weights_follow_hub_nodes() {
        let ps = 4096usize;
        let row_bytes = 1024usize; // 4 rows per page
        // Node 5 is a hub; it shares page 1 with three cold nodes.
        let mut degrees = vec![1u32; 12];
        degrees[5] = 9_000;

        let w = PageWeights::from_node_degrees(&degrees, 0, row_bytes, ps);

        assert_eq!(w.weight(0), 1, "page of four cold rows stays low");
        assert_eq!(w.weight(1), 9_000, "the hub sets its whole page's weight");
        assert_eq!(w.weight(2), 1);
        assert_eq!(w.weight(99), 0, "pages past the payload weigh 0");
    }

    #[test]
    fn degree_weights_span_rows_crossing_pages() {
        let ps = 4096usize;
        let row_bytes = 3000usize; // rows straddle page boundaries
        let degrees = [1u32, 500, 1];

        // Row 1 covers bytes 3000..6000, so it lands on pages 0 and 1.
        let w = PageWeights::from_node_degrees(&degrees, 0, row_bytes, ps);
        assert_eq!(w.weight(0), 500);
        assert_eq!(w.weight(1), 500);
    }
}
