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
/// 2 MiB backing cuts the TLB entry count 512x. Applies to both file-backed
/// mappings and the large anonymous mappings the allocator hands out for
/// multi-megabyte arrays.
///
/// Best-effort in three ways: file-backed THP requires kernel support, the
/// hint does nothing at all when `transparent_hugepage/enabled` is `never`
/// and is redundant when it is `always` (it earns its keep under the
/// `madvise` setting), and the errno on refusal is ignored.
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
