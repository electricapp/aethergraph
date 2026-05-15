//! RDMA memory registration hook.
//!
//! Registers the ring buffer memory with the InfiniBand HCA via `ibv_reg_mr`,
//! enabling one-sided RDMA reads from GPU nodes without waking the CPU.
//!
//! # Safety
//! Requires `libibverbs` at link time. Only available behind `rdma` feature.

use crate::{HookError, MemoryHook};
use std::sync::Mutex;

// ibverbs FFI — minimal subset for memory registration.
// `#[link(name = "ibverbs")]` is required so rust-lld actually resolves these
// symbols at link time (omitting it works only when no test/binary references
// the externs — a latent footgun for any consumer that does).
#[link(name = "ibverbs")]
unsafe extern "C" {
    fn ibv_reg_mr(
        pd: *mut IbvPd,
        addr: *mut libc::c_void,
        length: usize,
        access: i32,
    ) -> *mut IbvMr;
    fn ibv_dereg_mr(mr: *mut IbvMr) -> i32;
}

/// Opaque ibverbs protection domain handle.
#[repr(C)]
pub struct IbvPd {
    _opaque: [u8; 0],
}

/// ibverbs memory region — non-opaque layout.
///
/// This struct layout has been stable since libibverbs 1.0. The fields
/// through `rkey` are ABI-stable across all MLNX and rdma-core versions.
#[repr(C)]
pub struct IbvMr {
    pub context: *mut libc::c_void,
    pub pd: *mut IbvPd,
    pub addr: *mut libc::c_void,
    pub length: usize,
    pub handle: u32,
    pub lkey: u32,
    pub rkey: u32,
}

// SAFETY: ibv_mr handles are thread-safe after registration
unsafe impl Send for MrPtr {}
unsafe impl Sync for MrPtr {}

struct MrPtr(*mut IbvMr);

// ibverbs access bits — defined here because the C header values are stable
// since libibverbs 1.0 and we link the symbols directly.
const IBV_ACCESS_LOCAL_WRITE: i32 = 0x01;
const IBV_ACCESS_REMOTE_READ: i32 = 0x02;
/// Local writes (so the HCA can DMA *into* this region) plus remote reads
/// (so peers can RDMA-read it).
const IBV_ACCESS_FLAGS: i32 = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;

/// Registers memory with an InfiniBand HCA for RDMA access.
///
/// After registration, GPU nodes can perform one-sided RDMA reads
/// using the `rkey` from the registered memory region.
///
/// # Usage
/// ```ignore
/// let pd = /* obtain protection domain from ibverbs */;
/// let hook = RdmaRegHook::new(pd);
/// let ring = SharedMemoryRing::new(256, 4096, vec![Box::new(hook)]);
/// // After construction, hook.rkey() returns the remote key
/// ```
pub struct RdmaRegHook {
    pd: *mut IbvPd,
    mr: Mutex<Option<MrPtr>>,
}

// SAFETY: Protection domain handles are thread-safe in ibverbs
unsafe impl Send for RdmaRegHook {}
unsafe impl Sync for RdmaRegHook {}

impl RdmaRegHook {
    /// Create a new RDMA registration hook.
    ///
    /// # Safety
    /// `pd` must be a valid ibverbs protection domain that outlives this hook.
    pub unsafe fn new(pd: *mut IbvPd) -> Self {
        Self {
            pd,
            mr: Mutex::new(None),
        }
    }

    /// Get the remote key for this memory region (available after allocation).
    ///
    /// Returns `None` if registration hasn't happened or failed. A poisoned
    /// mutex (a prior panic while locked) is recovered transparently — the
    /// stored `MrPtr` is plain data and remains valid.
    pub fn rkey(&self) -> Option<u32> {
        let guard = self.mr.lock().unwrap_or_else(|e| e.into_inner());
        let mr = guard.as_ref()?;
        // SAFETY: mr.0 is a valid ibv_mr pointer from ibv_reg_mr
        Some(unsafe { (*mr.0).rkey })
    }

    /// Get the local key for this memory region (available after allocation).
    ///
    /// Returns `None` if registration hasn't happened or failed. Poisoning
    /// is recovered transparently for the same reason as `rkey`.
    pub fn lkey(&self) -> Option<u32> {
        let guard = self.mr.lock().unwrap_or_else(|e| e.into_inner());
        let mr = guard.as_ref()?;
        // SAFETY: mr.0 is a valid ibv_mr pointer from ibv_reg_mr
        Some(unsafe { (*mr.0).lkey })
    }
}

impl MemoryHook for RdmaRegHook {
    fn on_alloc(&self, ptr: *mut u8, size: usize) -> Result<(), HookError> {
        // SAFETY: pd is valid (caller invariant); ptr/size from ring allocation.
        let mr = unsafe { ibv_reg_mr(self.pd, ptr as *mut libc::c_void, size, IBV_ACCESS_FLAGS) };

        if mr.is_null() {
            #[cfg(feature = "tracing")]
            tracing::warn!(size, "ibv_reg_mr failed — RDMA reads will not work");
            return Err(HookError::new("ibv_reg_mr returned null"));
        }

        #[cfg(feature = "tracing")]
        tracing::info!(size, "RDMA memory region registered");

        // Recover from poison: dropping the freshly-registered MR here without
        // storing it would leak the ibverbs registration. If a previous MR is
        // still stored (poison left state behind), tear it down before
        // overwriting so the old registration is not also leaked.
        let mut guard = self.mr.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(prev) = guard.take() {
            // SAFETY: prev was registered via ibv_reg_mr in a prior on_alloc.
            unsafe { ibv_dereg_mr(prev.0) };
        }
        *guard = Some(MrPtr(mr));
        Ok(())
    }

    fn on_dealloc(&self, _ptr: *mut u8, _size: usize) {
        // Recover from poison so a previously-registered MR is still
        // deregistered. Failure to dereg leaks an ibverbs MR and pins the
        // underlying pages.
        let mut guard = self.mr.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(mr) = guard.take() {
            // SAFETY: mr was registered via ibv_reg_mr in on_alloc.
            unsafe {
                ibv_dereg_mr(mr.0);
            }
        }
    }
}
