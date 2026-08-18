//! GDRCopy: sub-microsecond CPU stores directly into VRAM.
//!
//! A `cudaMemcpy` for a few bytes pays a full launch/round-trip; for the
//! stack's small-write case — updating a seqlock version stamp or a head
//! pointer on the GPU-resident feature table — that latency dominates.
//! GDRCopy BAR1-maps a pinned GPU buffer into the CPU's address space so
//! the CPU stores to it with ordinary writes, landing in VRAM at sub-µs
//! latency with no CUDA API call on the hot path.
//!
//! This binds NVIDIA's userspace `libgdrapi` (backed by the `gdrdrv`
//! kernel module — the same vendor-shipped class as `nvidia-peermem`).
//! Compile-gated behind a `gdrcopy` feature; the library is linked only
//! then, and the live path is validated on hardware, not in CI.

use std::io;
use std::os::raw::{c_int, c_void};

use cudarc::driver::sys;

// --- libgdrapi FFI (gdrapi.h) --------------------------------------------

/// Opaque `gdr_t` handle.
#[repr(C)]
struct Gdr {
    _opaque: [u8; 0],
}

/// `gdr_mh_t` — an opaque mapping handle carrying a single word.
#[repr(C)]
#[derive(Clone, Copy)]
struct GdrMh {
    h: std::os::raw::c_ulong,
}

/// `gdr_info_t` — details of a mapping, including the page offset the
/// caller must add to the mapped VA before storing.
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct GdrInfo {
    va: u64,
    mapped_size: u64,
    page_size: u32,
    tm_cycles: u64,
    cookie: u32,
    mapped: c_int,
    wc_mapping: c_int,
}

unsafe extern "C" {
    fn gdr_open() -> *mut Gdr;
    fn gdr_close(g: *mut Gdr) -> c_int;
    fn gdr_pin_buffer(
        g: *mut Gdr,
        addr: std::os::raw::c_ulong,
        size: usize,
        p2p_token: u64,
        va_space: u32,
        handle: *mut GdrMh,
    ) -> c_int;
    fn gdr_unpin_buffer(g: *mut Gdr, handle: GdrMh) -> c_int;
    fn gdr_map(g: *mut Gdr, handle: GdrMh, va: *mut *mut c_void, size: usize) -> c_int;
    fn gdr_unmap(g: *mut Gdr, handle: GdrMh, va: *mut c_void, size: usize) -> c_int;
    fn gdr_get_info(g: *mut Gdr, handle: GdrMh, info: *mut GdrInfo) -> c_int;
    fn gdr_copy_to_mapping(
        handle: GdrMh,
        map_d_ptr: *mut c_void,
        h_ptr: *const c_void,
        size: usize,
    ) -> c_int;
    fn gdr_copy_from_mapping(
        handle: GdrMh,
        h_ptr: *mut c_void,
        map_d_ptr: *const c_void,
        size: usize,
    ) -> c_int;
}

// --- safe wrapper ---------------------------------------------------------

/// A pinned, BAR1-mapped GPU buffer the CPU can store into directly.
///
/// Construct over a device pointer inside a CUDA allocation; the mapped
/// region is accessed through [`Self::copy_to`] / [`Self::copy_from`],
/// which write through the BAR1 mapping without a CUDA API call.
pub struct GdrMapping {
    g: *mut Gdr,
    handle: GdrMh,
    /// Base of the BAR1 mapping, before the page-offset adjustment.
    map_base: *mut c_void,
    /// Store target: `map_base` plus the buffer's page offset.
    store_ptr: *mut c_void,
    size: usize,
}

// SAFETY: the gdr handle and mapping are owned for the struct's lifetime;
// BAR1 stores are as thread-safe as any shared buffer (the seqlock
// protocol coordinates concurrent access).
unsafe impl Send for GdrMapping {}
// SAFETY: see the Send impl above.
unsafe impl Sync for GdrMapping {}

impl GdrMapping {
    /// Pin and BAR1-map `size` bytes at device pointer `dev_ptr` (which
    /// must be GPU-page aligned, as `cuMemAlloc` returns). The CPU can
    /// then store into the region via [`Self::copy_to`].
    pub fn new(dev_ptr: sys::CUdeviceptr, size: usize) -> io::Result<Self> {
        // SAFETY: gdr_open takes no arguments and returns a handle or null.
        let g = unsafe { gdr_open() };
        if g.is_null() {
            return Err(io::Error::other(
                "gdr_open failed (is the gdrdrv module loaded?)",
            ));
        }

        let mut handle = GdrMh { h: 0 };
        // SAFETY: `g` is a live gdr handle; `dev_ptr`/`size` describe a
        // pinned CUDA allocation; `handle` is a valid out-pointer.
        let rc = unsafe { gdr_pin_buffer(g, dev_ptr, size, 0, 0, &mut handle) };
        if rc != 0 {
            // SAFETY: `g` is live and closed exactly once on this path.
            unsafe { gdr_close(g) };
            return Err(io::Error::other(format!("gdr_pin_buffer failed: {rc}")));
        }

        let mut map_base: *mut c_void = std::ptr::null_mut();
        // SAFETY: `handle` pins a `size`-byte region; `map_base` receives
        // the BAR1 VA.
        let rc = unsafe { gdr_map(g, handle, &mut map_base, size) };
        if rc != 0 {
            // SAFETY: `handle` is pinned on `g`; unpinned once here.
            unsafe { gdr_unpin_buffer(g, handle) };
            // SAFETY: `g` is live; closed once here.
            unsafe { gdr_close(g) };
            return Err(io::Error::other(format!("gdr_map failed: {rc}")));
        }

        // The store address is the mapping base plus the buffer's offset
        // within its GPU page, reported by gdr_get_info.
        let mut info = GdrInfo::default();
        // SAFETY: `handle`/`g` are live; `info` is a valid out-pointer.
        let rc = unsafe { gdr_get_info(g, handle, &mut info) };
        if rc != 0 {
            // SAFETY: `map_base`/`size` was mapped for `handle`; unmapped once.
            unsafe { gdr_unmap(g, handle, map_base, size) };
            // SAFETY: `handle` is pinned on `g`; unpinned once here.
            unsafe { gdr_unpin_buffer(g, handle) };
            // SAFETY: `g` is live; closed once here.
            unsafe { gdr_close(g) };
            return Err(io::Error::other(format!("gdr_get_info failed: {rc}")));
        }
        // `info.va` is the page-aligned base of the pinned region, so it
        // should never exceed `dev_ptr` — but it comes back from a C
        // library, and a wrapped subtraction here would put `store_ptr`
        // arbitrarily far outside the BAR1 mapping and every later copy
        // with it. Check rather than assume.
        let page_off = match dev_ptr.checked_sub(info.va) {
            Some(off) if (off as usize) < size => off as usize,
            _ => {
                // SAFETY: `map_base`/`size` was mapped for `handle`; unmapped once.
                unsafe { gdr_unmap(g, handle, map_base, size) };
                // SAFETY: `handle` is pinned on `g`; unpinned once here.
                unsafe { gdr_unpin_buffer(g, handle) };
                // SAFETY: `g` is live; closed once here.
                unsafe { gdr_close(g) };
                return Err(io::Error::other(format!(
                    "gdr_get_info reported va {:#x} for buffer {dev_ptr:#x} of {size} bytes",
                    info.va
                )));
            }
        };
        // SAFETY: `page_off < size`, so the offset is inside the mapping.
        let store_ptr = unsafe { (map_base as *mut u8).add(page_off) as *mut c_void };

        Ok(Self {
            g,
            handle,
            map_base,
            store_ptr,
            size,
        })
    }

    /// Store `src` into the mapped VRAM at byte `offset`, via the BAR1
    /// mapping — no CUDA call, sub-µs for small writes.
    pub fn copy_to(&self, offset: usize, src: &[u8]) -> io::Result<()> {
        // Checked: `offset + src.len()` wrapping would pass a plain
        // comparison and then store through `store_ptr + offset` far
        // outside the BAR1 mapping. This is a safe function, so the
        // arithmetic is part of what makes it one.
        if !end_within(offset, src.len(), self.size) {
            return Err(io::Error::other("gdr copy_to out of range"));
        }
        // SAFETY: `offset + src.len() <= size`, so `store_ptr + offset`
        // stays inside the mapping.
        let dst = unsafe { (self.store_ptr as *mut u8).add(offset) as *mut c_void };
        // SAFETY: `dst` addresses `src.len()` mapped bytes; the call issues
        // ordered stores through the BAR1 mapping.
        let rc = unsafe {
            gdr_copy_to_mapping(self.handle, dst, src.as_ptr() as *const c_void, src.len())
        };
        if rc != 0 {
            return Err(io::Error::other(format!(
                "gdr_copy_to_mapping failed: {rc}"
            )));
        }
        Ok(())
    }

    /// Read `dst.len()` bytes from the mapped VRAM at byte `offset` back
    /// into `dst`.
    pub fn copy_from(&self, offset: usize, dst: &mut [u8]) -> io::Result<()> {
        if !end_within(offset, dst.len(), self.size) {
            return Err(io::Error::other("gdr copy_from out of range"));
        }
        // SAFETY: `offset + dst.len() <= size`, so `store_ptr + offset`
        // stays inside the mapping.
        let src = unsafe { (self.store_ptr as *const u8).add(offset) as *const c_void };
        // SAFETY: `src` addresses `dst.len()` mapped bytes; the call reads
        // them back through the BAR1 mapping.
        let rc = unsafe {
            gdr_copy_from_mapping(self.handle, dst.as_mut_ptr() as *mut c_void, src, dst.len())
        };
        if rc != 0 {
            return Err(io::Error::other(format!(
                "gdr_copy_from_mapping failed: {rc}"
            )));
        }
        Ok(())
    }
}

/// Whether `[offset, offset + len)` lies inside `size`, without the
/// wrapping a plain `offset + len` comparison admits.
fn end_within(offset: usize, len: usize, size: usize) -> bool {
    offset.checked_add(len).is_some_and(|end| end <= size)
}

impl Drop for GdrMapping {
    fn drop(&mut self) {
        // SAFETY: `map_base`/`size` is this mapping's BAR1 region; unmapped once.
        unsafe { gdr_unmap(self.g, self.handle, self.map_base, self.size) };
        // SAFETY: `handle` is pinned on `self.g`; unpinned once.
        unsafe { gdr_unpin_buffer(self.g, self.handle) };
        // SAFETY: `self.g` is a live gdr handle; closed once.
        unsafe { gdr_close(self.g) };
    }
}
