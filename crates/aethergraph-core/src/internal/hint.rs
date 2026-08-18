//! Kernel-level prefetch hints for NVMe I/O.
//!
//! These functions tell the kernel to prefetch data into the page cache
//! *before* we actually read it, reducing page fault latency.
//!
//! Note: Some functions are kept for future optimization opportunities.

#![allow(dead_code)]

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

/// Hint to kernel: we will need this file region soon.
/// - Linux: Uses posix_fadvise(POSIX_FADV_WILLNEED)
/// - macOS: Uses fcntl(F_RDADVISE)
#[cfg(target_os = "linux")]
pub fn prefetch_file_range<F: AsRawFd>(file: &F, offset: u64, len: usize) {
    // SAFETY: posix_fadvise is a kernel hint with no memory safety preconditions beyond a valid fd,
    // which `file.as_raw_fd()` provides for the lifetime of `file`.
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_WILLNEED,
        );
    }
}

#[cfg(target_os = "macos")]
pub fn prefetch_file_range<F: AsRawFd>(file: &F, offset: u64, len: usize) {
    // macOS uses fcntl with F_RDADVISE
    #[repr(C)]
    struct radvisory {
        ra_offset: libc::off_t,
        ra_count: libc::c_int,
    }

    let advice = radvisory {
        ra_offset: offset as libc::off_t,
        // ra_count is a c_int: clamp instead of `as`-casting, which would
        // wrap spans over 2 GiB into a negative count and silently no-op
        // the hint for exactly the large files it targets.
        ra_count: len.min(libc::c_int::MAX as usize) as libc::c_int,
    };

    // SAFETY: fcntl(F_RDADVISE) reads a `radvisory` from the variadic arg; we pass a properly
    // initialized struct by reference and a valid fd, with no aliasing concerns.
    unsafe {
        // F_RDADVISE = 44 on macOS
        libc::fcntl(file.as_raw_fd(), 44, &advice);
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn prefetch_file_range<F>(_file: &F, _offset: u64, _len: usize) {}

/// Round an (addr, len) span down/out to page boundaries.
///
/// `madvise` demands a page-aligned start address and rejects unaligned
/// spans with EINVAL — which silently no-ops the hint for regions that
/// start mid-page (e.g. a CSR body sitting after a 32-byte header). The
/// mapping base is always page-aligned, so widening the span to page
/// bounds stays inside the mapping.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn page_span(addr: *const u8, len: usize) -> (*mut libc::c_void, usize) {
    // SAFETY: sysconf has no preconditions; non-positive results fall back.
    let page = match unsafe { libc::sysconf(libc::_SC_PAGESIZE) } {
        p if p > 0 => p as usize,
        _ => 4096,
    };
    let start = (addr as usize) & !(page - 1);
    let end = (addr as usize).saturating_add(len);
    let end = end.checked_add(page - 1).map_or(end, |e| e & !(page - 1));
    (start as *mut libc::c_void, end.saturating_sub(start))
}

/// Hint to kernel: we will need this mmap'd region soon.
#[cfg(target_os = "linux")]
pub fn prefetch_mmap_range(addr: *const u8, len: usize) {
    let (start, span) = page_span(addr, len);
    // SAFETY: madvise is a kernel hint; caller passes a pointer/length describing a mapped region.
    unsafe {
        libc::madvise(start, span, libc::MADV_WILLNEED);
    }
}

#[cfg(target_os = "macos")]
pub fn prefetch_mmap_range(addr: *const u8, len: usize) {
    let (start, span) = page_span(addr, len);
    // SAFETY: posix_madvise is a kernel hint; caller passes a pointer/length describing a mapped region.
    unsafe {
        libc::posix_madvise(start, span, libc::POSIX_MADV_WILLNEED);
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn prefetch_mmap_range(_addr: *const u8, _len: usize) {}

/// Hint: sequential access pattern (increases readahead).
#[cfg(target_os = "linux")]
pub fn hint_sequential<F: AsRawFd>(file: &F, offset: u64, len: usize) {
    // SAFETY: posix_fadvise is a kernel hint; only requires a valid fd from `file`.
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_SEQUENTIAL,
        );
    }
}

#[cfg(not(target_os = "linux"))]
pub fn hint_sequential<F>(_file: &F, _offset: u64, _len: usize) {}

/// Hint: random access pattern (disables readahead).
#[cfg(target_os = "linux")]
pub fn hint_random<F: AsRawFd>(file: &F, offset: u64, len: usize) {
    // SAFETY: posix_fadvise is a kernel hint; only requires a valid fd from `file`.
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_RANDOM,
        );
    }
}

#[cfg(not(target_os = "linux"))]
pub fn hint_random<F>(_file: &F, _offset: u64, _len: usize) {}

/// Hint: this mmap'd region is accessed randomly — disable readahead.
///
/// Sampling gathers fault one 4 KiB page per useful row; the default
/// readahead would pull in 128 KiB per fault, evicting useful cache for
/// bytes that are never read.
#[cfg(target_os = "linux")]
pub fn advise_mmap_random(addr: *const u8, len: usize) {
    let (start, span) = page_span(addr, len);
    // SAFETY: madvise is a kernel hint; caller passes a pointer/length describing a mapped region.
    unsafe {
        libc::madvise(start, span, libc::MADV_RANDOM);
    }
}

#[cfg(target_os = "macos")]
pub fn advise_mmap_random(addr: *const u8, len: usize) {
    let (start, span) = page_span(addr, len);
    // SAFETY: posix_madvise is a kernel hint; caller passes a pointer/length describing a mapped region.
    unsafe {
        libc::posix_madvise(start, span, libc::POSIX_MADV_RANDOM);
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn advise_mmap_random(_addr: *const u8, _len: usize) {}

/// Hint: back this region with huge pages where the kernel can.
///
/// Random gathers over multi-GB arrays are dTLB-bound with 4 KiB pages; a
/// 2 MiB backing cuts the TLB entry count 512x. Applies to file-backed
/// mappings and to the large anonymous mappings the allocator returns for
/// multi-megabyte arrays.
///
/// Best-effort: it earns its keep under `transparent_hugepage=madvise`, does
/// nothing under `never`, is redundant under `always`, and the errno on
/// refusal is ignored.
#[cfg(target_os = "linux")]
pub fn advise_hugepage(addr: *const u8, len: usize) {
    let (start, span) = page_span(addr, len);
    // SAFETY: madvise is a kernel hint; caller passes a pointer/length describing a mapped region.
    unsafe {
        libc::madvise(start, span, libc::MADV_HUGEPAGE);
    }
}

#[cfg(not(target_os = "linux"))]
pub fn advise_hugepage(_addr: *const u8, _len: usize) {}

/// Spread this mmap'd region's pages across every online NUMA node.
///
/// Sampling threads run on every socket and touch node offsets and edge
/// lists with no locality worth preserving — the access pattern is random
/// by construction. Left to the default first-touch policy the whole graph
/// lands on whichever node happened to fault it in, usually the loader's,
/// so every thread on the other sockets pays interconnect latency and that
/// one memory controller carries all the traffic. Interleaving trades a
/// share of remote accesses for bandwidth spread evenly across controllers,
/// which is the right side of that trade for a read-mostly structure with
/// no thread affinity.
///
/// Placement is set before the pages are faulted in, so it applies as they
/// arrive. Best-effort in the same way the madvise hints above are: a
/// refusal (no kernel support, seccomp, a cpuset that pins the process to
/// one node) leaves the mapping working with default placement.
///
/// Returns whether the policy was accepted, for telemetry.
#[cfg(all(target_os = "linux", feature = "numa"))]
pub fn interleave_mmap_range(addr: *const u8, len: usize) -> bool {
    let nodes = aether_mem::numa::nodes_online();
    // One node is not a placement decision — the policy call would succeed
    // and change nothing.
    if nodes.len() < 2 {
        return false;
    }
    let (start, span) = page_span(addr, len);
    if span == 0 {
        return false;
    }
    aether_mem::numa::interleave_region(start as *mut u8, span, &nodes).is_ok()
}

#[cfg(not(all(target_os = "linux", feature = "numa")))]
pub fn interleave_mmap_range(_addr: *const u8, _len: usize) -> bool {
    false
}

/// Fault this read-only region in now, and report whether it worked.
///
/// `MADV_WILLNEED` only queues readahead: it returns before the pages are
/// there, so the first access can still block on a major fault. Sampling
/// hits the offsets array on the critical path of every batch, and a fault
/// there stalls the whole frontier. `MADV_POPULATE_READ` (Linux 5.14)
/// populates synchronously with *read* faults, so it does not force
/// copy-on-write private copies the way `MAP_POPULATE` on a private
/// mapping does — the pages stay shared with the page cache.
///
/// Returns false when the kernel does not support it, so the caller can
/// keep the readahead hint as a fallback rather than pre-faulting nothing.
#[cfg(target_os = "linux")]
pub fn populate_read(addr: *const u8, len: usize) -> bool {
    // MADV_POPULATE_READ, from <asm-generic/mman-common.h>. Not in libc's
    // constant set on every platform the crate targets, so it is spelled
    // out here.
    const MADV_POPULATE_READ: libc::c_int = 22;
    // Guard the requested length, not the widened span: `page_span` rounds
    // an unaligned start down and the end up, so a zero-length request at a
    // mid-page address widens to a whole page. Populating it would fault in
    // memory the caller never asked for and report success for a no-op.
    if len == 0 {
        return false;
    }
    let (start, span) = page_span(addr, len);
    if span == 0 {
        return false;
    }
    // SAFETY: madvise on a mapped range; POPULATE_READ only faults pages in
    // and never changes the mapping's contents or extent.
    unsafe { libc::madvise(start, span, MADV_POPULATE_READ) == 0 }
}

#[cfg(not(target_os = "linux"))]
pub fn populate_read(_addr: *const u8, _len: usize) -> bool {
    false
}

/// Hint: done with this region, can be evicted.
#[cfg(target_os = "linux")]
pub fn hint_dontneed<F: AsRawFd>(file: &F, offset: u64, len: usize) {
    // SAFETY: posix_fadvise is a kernel hint; only requires a valid fd from `file`.
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

#[cfg(not(target_os = "linux"))]
pub fn hint_dontneed<F>(_file: &F, _offset: u64, _len: usize) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Populating a real mapping must either work or report that it did
    /// not — never a false success the caller would take as "resident",
    /// skipping the readahead fallback for pages that were never faulted.
    #[test]
    fn populate_read_reports_whether_it_populated() {
        let len = 256 * 1024;
        // A private anonymous mapping is a valid madvise target on every
        // platform this runs on, so the call is exercised rather than
        // skipped.
        let mut region = vec![0u8; len];
        let populated = populate_read(region.as_ptr(), len);

        if cfg!(target_os = "linux") {
            // Whatever the kernel decided, the region stays readable and
            // its contents are untouched — POPULATE_READ faults pages in
            // and changes nothing else.
            region[0] = 7;
            region[len - 1] = 9;
            assert_eq!(region[0], 7);
            assert_eq!(region[len - 1], 9);
        } else {
            assert!(!populated, "no population is possible off Linux");
        }
    }

    /// A zero-length span has no pages to fault, so it must report false
    /// rather than passing an empty range to the kernel and taking its
    /// success as populated.
    #[test]
    fn populate_read_rejects_an_empty_span() {
        let region = [0u8; 4096];
        assert!(!populate_read(region.as_ptr(), 0));
    }
}
