//! CUDA IPC for a multi-process device-side feature cache.
//!
//! One process builds the growable VRAM cache ([`super::vmm::GrowableVram`],
//! whose physical handles are POSIX-fd shareable); this module exports a
//! chunk's handle to an OS file descriptor with
//! `cuMemExportToShareableHandle`, which is passed to peer trainer
//! processes over the same `SCM_RIGHTS` channel as the host-side memfd
//! cache. A peer imports it with `cuMemImportFromShareableHandle` and maps
//! it into its own reserved range — N processes share one device-side
//! cache with no per-process copy.
//!
//! Pairs with the host memfd cache into one "attach" story: a single token
//! carries both the host-tier memfd and the device-tier IPC fd.
//! Compile-gated behind `cuda`; validated on hardware.

use std::io;
use std::os::unix::io::RawFd;
use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};

/// Export `handle` (a physical allocation from `cuMemCreate` requesting a
/// POSIX-fd handle type) to an OS file descriptor for passing to a peer.
///
/// The returned fd is owned by the caller and should be sent over a
/// `SCM_RIGHTS` control message, then closed once sent.
pub fn export_handle_to_fd(handle: sys::CUmemGenericAllocationHandle) -> io::Result<RawFd> {
    let mut fd: RawFd = -1;
    // SAFETY: `&mut fd` receives the exported descriptor; the handle type
    // matches the POSIX-fd type the allocation was created with.
    let res = unsafe {
        sys::cuMemExportToShareableHandle(
            &mut fd as *mut RawFd as *mut std::ffi::c_void,
            handle,
            sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0,
        )
    };
    cuda_ok(res, "cuMemExportToShareableHandle")?;
    if fd < 0 {
        return Err(io::Error::other("export produced an invalid fd"));
    }
    Ok(fd)
}

/// A device-side cache imported from a peer's exported fd and mapped into
/// this process's own reserved virtual range.
pub struct ImportedVram {
    ctx: Arc<CudaContext>,
    device: sys::CUdevice,
    base: sys::CUdeviceptr,
    size: usize,
    handle: sys::CUmemGenericAllocationHandle,
}

// SAFETY: the imported handle and reservation are owned for the struct's
// lifetime; the CUDA context is thread-safe after creation.
unsafe impl Send for ImportedVram {}
// SAFETY: see the Send impl above.
unsafe impl Sync for ImportedVram {}

impl ImportedVram {
    /// Import the physical allocation referenced by `fd` (received over
    /// `SCM_RIGHTS` from the owner) and map `size` bytes of it into a fresh
    /// reserved range, read-write from `device`. Takes ownership of `fd`.
    ///
    /// # Safety
    /// `fd` must be a CUDA shareable handle exported by a trusted peer via
    /// [`export_handle_to_fd`], and `size` must match the exported
    /// allocation — a mismatch maps past the physical block.
    pub unsafe fn import(
        ctx: &Arc<CudaContext>,
        device: sys::CUdevice,
        fd: RawFd,
        size: usize,
    ) -> io::Result<Self> {
        let mut handle: sys::CUmemGenericAllocationHandle = 0;
        // SAFETY: `fd` is a valid CUDA POSIX-fd shareable handle per the
        // caller's contract; `handle` is a valid out-pointer.
        let res = unsafe {
            sys::cuMemImportFromShareableHandle(
                &mut handle,
                fd as *mut std::ffi::c_void,
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            )
        };
        cuda_ok(res, "cuMemImportFromShareableHandle")?;

        // Reserve a range and map the imported handle into it.
        let mut base: sys::CUdeviceptr = 0;
        // SAFETY: reserve `size` bytes of VA for the mapping.
        let res = unsafe { sys::cuMemAddressReserve(&mut base, size, 0, 0, 0) };
        if let Err(e) = cuda_ok(res, "cuMemAddressReserve") {
            // SAFETY: `handle` was just imported and not yet mapped.
            unsafe { sys::cuMemRelease(handle) };
            return Err(e);
        }
        // SAFETY: `[base, base+size)` is the fresh reservation; `handle`
        // is the imported physical allocation of `size`.
        let res = unsafe { sys::cuMemMap(base, size, 0, handle, 0) };
        if let Err(e) = cuda_ok(res, "cuMemMap") {
            // SAFETY: `base`/`size` is the reservation just made, never
            // mapped on this path; freed once.
            unsafe { sys::cuMemAddressFree(base, size) };
            // SAFETY: `handle` was imported above and not mapped; released
            // once.
            unsafe { sys::cuMemRelease(handle) };
            return Err(e);
        }

        let desc = sys::CUmemAccessDesc {
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: device,
            },
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        // SAFETY: the range was just mapped; one valid access descriptor.
        let res = unsafe { sys::cuMemSetAccess(base, size, &desc, 1) };
        cuda_ok(res, "cuMemSetAccess")?;

        Ok(Self {
            ctx: ctx.clone(),
            device,
            base,
            size,
            handle,
        })
    }

    /// The device pointer of the imported, mapped cache.
    pub fn device_ptr(&self) -> sys::CUdeviceptr {
        self.base
    }

    /// Size of the mapped region in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The device this mapping grants access from.
    pub fn device(&self) -> sys::CUdevice {
        self.device
    }
}

impl Drop for ImportedVram {
    fn drop(&mut self) {
        let _ctx = &self.ctx;
        // SAFETY: `base`/`size` is the mapped range this struct owns;
        // unmapped once.
        unsafe { sys::cuMemUnmap(self.base, self.size) };
        // SAFETY: `base`/`size` is this struct's reservation; freed once.
        unsafe { sys::cuMemAddressFree(self.base, self.size) };
        // SAFETY: `handle` is the imported allocation; released once.
        unsafe { sys::cuMemRelease(self.handle) };
    }
}

fn cuda_ok(res: sys::CUresult, what: &str) -> io::Result<()> {
    if res == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(io::Error::other(format!("{what} failed: {res:?}")))
    }
}
