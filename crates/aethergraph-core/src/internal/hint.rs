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
        ra_count: len as libc::c_int,
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

/// Hint to kernel: we will need this mmap'd region soon.
#[cfg(target_os = "linux")]
pub fn prefetch_mmap_range(addr: *const u8, len: usize) {
    // SAFETY: madvise is a kernel hint; caller passes a pointer/length describing a mapped region.
    unsafe {
        libc::madvise(addr as *mut libc::c_void, len, libc::MADV_WILLNEED);
    }
}

#[cfg(target_os = "macos")]
pub fn prefetch_mmap_range(addr: *const u8, len: usize) {
    // SAFETY: posix_madvise is a kernel hint; caller passes a pointer/length describing a mapped region.
    unsafe {
        libc::posix_madvise(addr as *mut libc::c_void, len, libc::POSIX_MADV_WILLNEED);
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
