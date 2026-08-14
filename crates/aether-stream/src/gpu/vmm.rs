//! Growable VRAM cache via the CUDA virtual-memory-management API.
//!
//! A device-side feature cache that grows without the realloc-and-copy a
//! plain `cuMemAlloc` would force: reserve a large virtual address range
//! once with `cuMemAddressReserve`, then commit physical VRAM into it
//! incrementally with `cuMemCreate` + `cuMemMap` as the cache fills. The
//! virtual base never moves, so device pointers handed out earlier stay
//! valid as the cache grows behind them.
//!
//! The physical handles are created with a POSIX-fd shareable type, so
//! the same allocation can later be exported for CUDA IPC (see
//! [`super::ipc`]). Compile-gated behind `cuda`; validated on hardware.

use std::io;
use std::sync::Arc;

use cudarc::driver::{CudaContext, sys};

/// A growable device allocation over a reserved virtual range.
///
/// Reserve `max_bytes` of address space up front; [`Self::grow_to`]
/// commits physical memory up to a high-water mark. Nothing is committed
/// at construction — the reservation is free until backed.
pub struct GrowableVram {
    ctx: Arc<CudaContext>,
    device: sys::CUdevice,
    base: sys::CUdeviceptr,
    reserved: usize,
    committed: usize,
    granularity: usize,
    /// Physical handles, one per committed chunk, kept for unmap/release.
    chunks: Vec<(sys::CUmemGenericAllocationHandle, sys::CUdeviceptr, usize)>,
}

// SAFETY: the reservation and handles are owned for the struct's lifetime;
// the CUDA context is thread-safe after creation.
unsafe impl Send for GrowableVram {}
// SAFETY: see the Send impl above.
unsafe impl Sync for GrowableVram {}

impl GrowableVram {
    /// Reserve `max_bytes` (rounded up to allocation granularity) of device
    /// virtual address space on `device`. No physical VRAM is used yet.
    pub fn reserve(
        ctx: &Arc<CudaContext>,
        device: sys::CUdevice,
        max_bytes: usize,
    ) -> io::Result<Self> {
        let prop = alloc_prop(device);
        let mut granularity: usize = 0;
        // SAFETY: `prop` is a fully initialized allocation property; the
        // granularity out-pointer is valid.
        let res = unsafe {
            sys::cuMemGetAllocationGranularity(
                &mut granularity,
                &prop,
                sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        };
        cuda_ok(res, "cuMemGetAllocationGranularity")?;
        if granularity == 0 {
            return Err(io::Error::other("zero allocation granularity"));
        }

        let reserved = round_up(max_bytes.max(granularity), granularity);
        let mut base: sys::CUdeviceptr = 0;
        // SAFETY: reserve `reserved` bytes of VA; addr=0 lets the driver
        // choose the base, alignment=0 uses the default.
        let res = unsafe { sys::cuMemAddressReserve(&mut base, reserved, 0, 0, 0) };
        cuda_ok(res, "cuMemAddressReserve")?;

        Ok(Self {
            ctx: ctx.clone(),
            device,
            base,
            reserved,
            committed: 0,
            granularity,
            chunks: Vec::new(),
        })
    }

    /// The stable base device pointer. Valid for the reserved range for
    /// the cache's whole life, even as it grows.
    pub fn device_ptr(&self) -> sys::CUdeviceptr {
        self.base
    }

    /// The physical allocation handle backing the first committed chunk,
    /// or `None` before the first [`Self::grow_to`].
    ///
    /// Chunks are created with the POSIX-fd handle type requested, so this
    /// can be handed to
    /// [`export_handle_to_fd`](super::ipc::export_handle_to_fd) for a peer
    /// process to map. Only the first chunk is exportable as a unit: a
    /// grown pool is several physical allocations behind one contiguous
    /// VA range, so a peer importing "the pool" would need each handle and
    /// its own reservation. Reserve enough up front, and grow once, when
    /// the pool is meant to be shared.
    pub fn first_chunk(&self) -> Option<(sys::CUmemGenericAllocationHandle, usize)> {
        self.chunks.first().map(|&(handle, _, len)| (handle, len))
    }

    /// Bytes currently backed by physical VRAM.
    pub fn committed(&self) -> usize {
        self.committed
    }

    /// Bytes of virtual address reserved.
    pub fn reserved(&self) -> usize {
        self.reserved
    }

    /// Commit physical VRAM so at least `target` bytes are backed. Rounds
    /// up to granularity; a no-op if already committed that far. The new
    /// pages are mapped contiguously after the existing ones and made
    /// read-write from `device`.
    pub fn grow_to(&mut self, target: usize) -> io::Result<()> {
        if target <= self.committed {
            return Ok(());
        }
        let target = round_up(target, self.granularity);
        if target > self.reserved {
            return Err(io::Error::other(format!(
                "grow_to {target} exceeds reserved {}",
                self.reserved
            )));
        }
        let add = target - self.committed;

        let prop = alloc_prop(self.device);
        let mut handle: sys::CUmemGenericAllocationHandle = 0;
        // SAFETY: `prop` is initialized; `handle` is a valid out-pointer.
        let res = unsafe { sys::cuMemCreate(&mut handle, add, &prop, 0) };
        cuda_ok(res, "cuMemCreate")?;

        let map_at = self.base + self.committed as u64;
        // SAFETY: `[map_at, map_at+add)` is inside the reservation and not
        // yet mapped; `handle` is a fresh physical allocation of `add`.
        let res = unsafe { sys::cuMemMap(map_at, add, 0, handle, 0) };
        if let Err(e) = cuda_ok(res, "cuMemMap") {
            // SAFETY: `handle` was just created and never mapped.
            unsafe { sys::cuMemRelease(handle) };
            return Err(e);
        }

        // Grant this device read-write access to the freshly mapped range.
        let desc = sys::CUmemAccessDesc {
            location: mem_location(self.device),
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        // SAFETY: the range was just mapped; `desc` is a single valid
        // access descriptor.
        let res = unsafe { sys::cuMemSetAccess(map_at, add, &desc, 1) };
        if let Err(e) = cuda_ok(res, "cuMemSetAccess") {
            // The chunk is mapped but not yet recorded, so `Drop` would
            // not reach it: unwind it here or the physical memory leaks
            // and the next grow_to maps over a range still held, since
            // `committed` has not advanced either.
            // SAFETY: `[map_at, add)` was mapped immediately above and is
            // unmapped exactly once here.
            unsafe { sys::cuMemUnmap(map_at, add) };
            // SAFETY: `handle` backs that chunk and is released once.
            unsafe { sys::cuMemRelease(handle) };
            return Err(e);
        }

        self.chunks.push((handle, map_at, add));
        self.committed = target;
        Ok(())
    }
}

impl Drop for GrowableVram {
    fn drop(&mut self) {
        let _ctx = &self.ctx;
        for &(handle, at, size) in &self.chunks {
            // SAFETY: `at`/`size` is a chunk this struct mapped and still
            // owns; unmapped exactly once here.
            unsafe { sys::cuMemUnmap(at, size) };
            // SAFETY: `handle` is the physical allocation backing that
            // chunk, created here and released exactly once.
            unsafe { sys::cuMemRelease(handle) };
        }
        if self.reserved > 0 {
            // SAFETY: `base`/`reserved` is the reservation from `reserve`.
            unsafe { sys::cuMemAddressFree(self.base, self.reserved) };
        }
    }
}

/// Physical-allocation properties: pinned device memory on `device`,
/// requesting a POSIX-fd shareable handle so the block can be exported for
/// IPC later.
fn alloc_prop(device: sys::CUdevice) -> sys::CUmemAllocationProp {
    // SAFETY: `CUmemAllocationProp` is a plain C struct of integers, an
    // enum, a location, and a nullable pointer; all-zero is a valid
    // initial state that the fields below then fill in.
    let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes =
        sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.location = mem_location(device);
    prop
}

fn mem_location(device: sys::CUdevice) -> sys::CUmemLocation {
    sys::CUmemLocation {
        type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
        id: device,
    }
}

fn round_up(value: usize, granularity: usize) -> usize {
    value.div_ceil(granularity) * granularity
}

fn cuda_ok(res: sys::CUresult, what: &str) -> io::Result<()> {
    if res == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(io::Error::other(format!("{what} failed: {res:?}")))
    }
}
