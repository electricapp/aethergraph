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
use std::collections::{BinaryHeap, HashMap};
use std::os::unix::io::RawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// --- userfaultfd ABI (linux/userfaultfd.h) --------------------------------

const UFFD_API: u64 = 0xAA;
const UFFDIO_REGISTER_MODE_MISSING: u64 = 1 << 0;
const UFFD_EVENT_PAGEFAULT: u8 = 0x12;
/// Restrict faults to user-mode accesses — required on hardened kernels
/// for the unprivileged path.
const UFFD_USER_MODE_ONLY: i32 = 1;

const fn iowr(nr: u32, size: usize) -> libc::c_ulong {
    const DIR_RW: libc::c_ulong = 3; // (READ|WRITE)
    const TYPE: libc::c_ulong = 0xAA; // UFFDIO
    (DIR_RW << 30) | ((size as libc::c_ulong) << 16) | (TYPE << 8) | nr as libc::c_ulong
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
fn uffdio_copy() -> libc::c_ulong {
    iowr(0x03, core::mem::size_of::<UffdioCopy>())
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
        // Short reads past EOF leave the tail zero — the region may be
        // larger than the backing file (sparse feature stores).
        let n = self.file.read_at(page, offset).context("backing read")?;
        for b in &mut page[n..] {
            *b = 0;
        }
        Ok(())
    }
}

// --- the paged region ----------------------------------------------------

/// An anonymous mapping whose pages are demand-loaded from a
/// [`PageSource`] and evicted under a residency budget.
///
/// The region is one flat address range: read any offset and the pager
/// materializes it. Residency is capped at `budget_pages`; installing a
/// page over budget first evicts the lowest-weight resident page.
pub struct PagedRegion {
    base: *mut u8,
    len: usize,
    page_size: usize,
    uffd: RawFd,
    shutdown: Arc<AtomicBool>,
    faults: Arc<AtomicU64>,
    evictions: Arc<AtomicU64>,
    pager: Option<std::thread::JoinHandle<()>>,
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
        if budget_pages == 0 {
            bail!("budget_pages must be > 0");
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

        // Register the whole range in missing-fault mode.
        let reg = UffdioRegister {
            range: UffdioRange {
                start: base as u64,
                len: len as u64,
            },
            mode: UFFDIO_REGISTER_MODE_MISSING,
            ioctls: 0,
        };
        // SAFETY: the range is exactly the mapping just created.
        if unsafe { libc::ioctl(uffd, uffdio_register(), &reg) } != 0 {
            let e = std::io::Error::last_os_error();
            unmap_region(base, len);
            close_uffd(uffd);
            bail!("UFFDIO_REGISTER failed: {e}");
        }

        let shutdown = Arc::new(AtomicBool::new(false));
        let faults = Arc::new(AtomicU64::new(0));
        let evictions = Arc::new(AtomicU64::new(0));

        let pager = {
            let shutdown = Arc::clone(&shutdown);
            let faults = Arc::clone(&faults);
            let evictions = Arc::clone(&evictions);
            let base_addr = base as usize;
            std::thread::Builder::new()
                .name("aether-uffd-pager".into())
                .spawn(move || {
                    let mut pager = Pager {
                        uffd,
                        base: base_addr,
                        page_size,
                        budget_pages,
                        source,
                        weights,
                        resident: HashMap::new(),
                        evict_queue: BinaryHeap::new(),
                        faults,
                        evictions,
                    };
                    pager.run(&shutdown);
                })
                .context("spawn pager thread")?
        };

        Ok(Self {
            base,
            len,
            page_size,
            uffd,
            shutdown,
            faults,
            evictions,
            pager: Some(pager),
        })
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

/// The pager thread's private state.
struct Pager {
    uffd: RawFd,
    base: usize,
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
            if let Err(e) = self.service_fault(msg.arg1, &mut staging) {
                tracing::error!(error = %e, "uffd pager fault handling failed");
                // A failed install leaves the faulting thread blocked; the
                // region is unusable, so stop rather than spin.
                break;
            }
        }
    }

    fn service_fault(&mut self, fault_addr: u64, staging: &mut [u8]) -> Result<()> {
        let page_start = (fault_addr as usize) & !(self.page_size - 1);
        let offset = (page_start - self.base) as u64;
        let page_no = offset as usize / self.page_size;

        // Evict down to budget-1 before installing the newcomer. A pass
        // that cannot free anything ends the loop: the only page left to
        // drop is the one being installed, and retrying would spin the
        // pager thread forever with every faulting reader blocked behind
        // it.
        while self.resident.len() >= self.budget_pages {
            if !self.evict_one(page_no)? {
                break;
            }
        }

        self.source.fill(offset, staging)?;
        // Count the fault and record residency *before* UFFDIO_COPY: the
        // copy is what unblocks the faulting thread, so a reader that sees
        // its access complete also sees this fault reflected in the
        // counters (the copy ioctl and the thread's resume bracket the
        // store with kernel barriers).
        let weight = self.weights.weight(page_no);
        if self.resident.insert(page_no, weight).is_none() {
            self.evict_queue.push(Reverse((weight, page_no)));
        }
        self.faults.fetch_add(1, Ordering::Relaxed);
        let copy = UffdioCopy {
            dst: page_start as u64,
            src: staging.as_ptr() as u64,
            len: self.page_size as u64,
            mode: 0,
            copy: 0,
        };
        // SAFETY: `dst` is the faulting page inside the registered range;
        // `src` is a full page of staging owned by this thread.
        if unsafe { libc::ioctl(self.uffd, uffdio_copy(), &copy) } != 0 {
            let e = std::io::Error::last_os_error();
            // EEXIST means another thread's access already installed it —
            // benign, the fault is resolved either way.
            if e.raw_os_error() != Some(libc::EEXIST) {
                bail!("UFFDIO_COPY failed: {e}");
            }
        }
        Ok(())
    }

    /// Evict the lowest-weight resident page (never the one about to be
    /// installed). `MADV_DONTNEED` drops it and re-arms its fault.
    ///
    /// Returns whether a page was actually evicted. `false` means the only
    /// candidate left is `keep`, and the caller must stop asking.
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
            if page == keep {
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
        // SAFETY: `addr` is a page inside the registered mapping.
        let ret = unsafe {
            libc::madvise(
                addr as *mut libc::c_void,
                self.page_size,
                libc::MADV_DONTNEED,
            )
        };
        if ret != 0 {
            bail!("MADV_DONTNEED failed: {}", std::io::Error::last_os_error());
        }
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

    /// A one-page budget is the degenerate case for eviction: every fault
    /// has to drop the only resident page, and several threads racing on
    /// the same page can fault it again while it is already installed. The
    /// eviction pass has nothing it may drop then, so it must give up
    /// rather than retry — a pager thread spinning here blocks every
    /// reader behind it, and the test would hang rather than fail.
    #[test]
    fn single_page_budget_serves_concurrent_faults() {
        if !uffd_available() {
            eprintln!("userfaultfd unavailable; skipping");
            return;
        }
        let ps = page_size();
        let npages = 8usize;

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        for p in 0..npages {
            tmp.write_all(&vec![(p % 251) as u8; ps]).unwrap();
        }
        tmp.flush().unwrap();
        let file = tmp.reopen().unwrap();

        let source = Arc::new(FileSource::new(file));
        let weights = Arc::new(PageWeights::default());
        let region = Arc::new(PagedRegion::new(npages * ps, 1, source, weights).unwrap());

        std::thread::scope(|s| {
            for _ in 0..4 {
                let region = Arc::clone(&region);
                s.spawn(move || {
                    let data = region.as_slice();
                    for round in 0..16 {
                        // Every thread hammers the same page each round, so
                        // the second and later faults land on a page that is
                        // already resident.
                        let p = round % npages;
                        assert_eq!(data[p * ps], (p % 251) as u8, "page {p} first byte");
                    }
                });
            }
        });

        assert!(region.fault_count() > 0, "faults must have been served");
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
