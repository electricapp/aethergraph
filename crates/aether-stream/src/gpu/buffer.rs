//! VRAM staging buffer for GPUDirect RDMA feature gather.
//!
//! Allocates a contiguous VRAM region and registers it with the
//! InfiniBand NIC for direct PCIe DMA — via nvidia-peermem where the
//! platform supports it, falling back to dma-buf registration
//! (`ibv_reg_dmabuf_mr`). When an RDMA READ completes, data lands
//! directly in VRAM — the host CPU's memory bus is never touched.

use crate::feature_table::FeatureSchema;
use crate::rdma::context::{RdmaContext, RegisteredMr};
use crate::rdma::ffi;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, sys};
use std::io;
use std::sync::Arc;

/// Register a CUDA VA range with the HCA.
///
/// Tries the peer-memory path (`ibv_reg_mr` on the device VA via
/// nvidia-peermem) first — the fast path on bare metal. Where that fails (hypervisors
/// block the p2p page path; the registration EFAULTs even with
/// `nvidia_peermem` loaded), falls back to exporting the range as a
/// dma-buf via `cuMemGetHandleForAddressRange` and registering with
/// `ibv_reg_dmabuf_mr`, which goes through the kernel DMA API and works
/// under IOMMU-mediated virtualization. The MR's `iova` is the device VA,
/// so work-request address arithmetic is identical on both paths.
///
/// Requires driver ≥ 515 (open kernel modules preferred) and
/// rdma-core ≥ v34 for the fallback.
///
/// # Safety
/// `[device_ptr, device_ptr + len)` must be a live CUDA allocation that
/// stays allocated until the returned `RegisteredMr` is dropped and all
/// in-flight work requests referencing it have completed.
pub unsafe fn reg_mr_cuda(
    rdma_ctx: &RdmaContext,
    device_ptr: u64,
    len: usize,
    access: i32,
) -> io::Result<RegisteredMr> {
    // SAFETY: the caller's contract — `[device_ptr, device_ptr + len)` is a
    // live CUDA allocation that outlives the returned MR and every WR that
    // references it.
    let peermem_err = match unsafe { rdma_ctx.reg_mr(device_ptr as *mut u8, len, access) } {
        Ok(mr) => return Ok(mr),
        Err(e) => e,
    };

    // dma-buf export needs a host-page-aligned range; cudaMalloc regions
    // are page-aligned in practice, but align defensively and register the
    // exact sub-range via offset/iova.
    let page = 4096u64;
    let base = device_ptr & !(page - 1);
    let end = (device_ptr + len as u64).div_ceil(page) * page;
    let range_len = (end - base) as usize;

    let mut fd: i32 = -1;
    // SAFETY: `fd` outlives the call; `base/range_len` describe a live CUDA
    // allocation per the caller's contract.
    let res = unsafe {
        sys::cuMemGetHandleForAddressRange(
            &mut fd as *mut i32 as *mut core::ffi::c_void,
            base,
            range_len,
            sys::CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
            0,
        )
    };
    if res != sys::CUresult::CUDA_SUCCESS || fd < 0 {
        return Err(io::Error::other(format!(
            "CUDA MR registration failed on both paths: \
             peermem ibv_reg_mr: {peermem_err}; \
             cuMemGetHandleForAddressRange: {res:?}. \
             peermem typically needs bare metal + `nvidia-peermem` loaded; \
             dma-buf needs driver ≥ 515 and rdma-core ≥ v34"
        )));
    }

    let mr = rdma_ctx.reg_mr_dmabuf(fd, device_ptr - base, len, device_ptr, access);
    // The MR holds its own reference to the dma-buf; drop ours either way.
    // SAFETY: `fd` is a live fd we own, returned by the export above.
    unsafe { libc::close(fd) };
    mr.map_err(|dmabuf_err| {
        io::Error::other(format!(
            "CUDA MR registration failed on both paths: \
             peermem ibv_reg_mr: {peermem_err}; \
             dmabuf ibv_reg_dmabuf_mr: {dmabuf_err}. \
             peermem typically needs bare metal + `nvidia-peermem` loaded; \
             dma-buf needs driver ≥ 515 and rdma-core ≥ v34 (VM/IOMMU fallback)"
        ))
    })
}

/// VRAM staging buffer for batched RDMA READs of feature table slots.
///
/// Each slot in the staging buffer mirrors the server-side slot layout:
/// `[head_version: u64] [features: f32 × feature_dim] [tail_version: u64]`
///
/// The buffer holds TWO staging regions of `max_batch_size` slots each —
/// one per snapshot of the two-snapshot seqlock protocol (see the RDMA
/// reader contract in `feature_table.rs`). Snapshot 1 lands at
/// [`slot_addr`](Self::slot_addr), snapshot 2 at
/// [`slot_addr2`](Self::slot_addr2). After both READ rounds complete, the
/// GPU kernel cross-validates the snapshots and compacts valid features
/// into the output tensor.
pub struct GpuGatherBuffer {
    /// MR registered against the NIC; declared first so it drops BEFORE
    /// `_allocation` — `ibv_dereg_mr` must run while the VRAM is still valid.
    mr: RegisteredMr,
    /// Held so the context outlives `device_ptr`, never read.
    _ctx: Arc<CudaContext>,
    /// Raw CUDA device pointer (for RDMA WR local_addr).
    device_ptr: u64,
    /// Full slot size matching server layout (head + features + padding + tail).
    slot_size: usize,
    /// Maximum number of slots (batch size).
    max_batch_size: usize,
    /// feature_dim * sizeof(f32).
    feature_bytes: usize,
    /// Keep the CudaSlice alive so the allocation isn't freed.
    _allocation: CudaSlice<u8>,
}

// SAFETY: VRAM allocation + RegisteredMr are valid for the buffer's lifetime;
// CUDA device pointers and ibverbs MRs are safe to share across threads.
unsafe impl Send for GpuGatherBuffer {}
// SAFETY: see Send impl above.
unsafe impl Sync for GpuGatherBuffer {}

impl GpuGatherBuffer {
    /// Allocate VRAM and register with the InfiniBand NIC.
    ///
    /// Registration goes through [`reg_mr_cuda`]: nvidia-peermem where
    /// available, dma-buf otherwise. Errors carry both paths' failures.
    pub fn new(
        rdma_ctx: &RdmaContext,
        cuda_ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        max_batch_size: usize,
        schema: &FeatureSchema,
    ) -> io::Result<Self> {
        let slot_size = schema.slot_size;
        let feature_bytes = schema.feature_dim * std::mem::size_of::<f32>();
        // Two snapshot regions of `max_batch_size` slots each.
        let total_bytes = slot_size
            .checked_mul(max_batch_size)
            .and_then(|b| b.checked_mul(2))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "staging buffer size overflow: slot_size {slot_size} * max_batch_size {max_batch_size} * 2"
                    ),
                )
            })?;

        // Allocate VRAM
        let allocation: CudaSlice<u8> = stream
            .alloc_zeros(total_bytes)
            .map_err(|e| io::Error::other(format!("cuMemAlloc failed: {e}")))?;

        // Get raw device pointer. Scope the borrow guard so it drops
        // before we move `allocation` into Self.
        let device_ptr = {
            let (ptr, _guard) = allocation.device_ptr(stream);
            ptr
        };

        // IBV_ACCESS_LOCAL_WRITE is required for RDMA READ target buffers.
        let access = ffi::IBV_ACCESS_LOCAL_WRITE;
        // SAFETY: `[device_ptr, device_ptr + total_bytes)` is the live CUDA
        // allocation held in `_allocation`, which this struct keeps alive for
        // its whole lifetime; `mr` is declared before `_allocation` so the MR
        // deregisters first on drop.
        let mr = unsafe { reg_mr_cuda(rdma_ctx, device_ptr, total_bytes, access)? };

        Ok(Self {
            _ctx: cuda_ctx.clone(),
            device_ptr,
            mr,
            slot_size,
            max_batch_size,
            feature_bytes,
            _allocation: allocation,
        })
    }

    /// Local key for RDMA work requests.
    pub fn lkey(&self) -> u32 {
        self.mr.lkey()
    }

    /// Address of slot `i` in the snapshot-1 region (for RDMA WR `local_addr`).
    #[inline]
    pub fn slot_addr(&self, i: usize) -> u64 {
        debug_assert!(i < self.max_batch_size);
        self.device_ptr + (i * self.slot_size) as u64
    }

    /// Address of slot `i` in the snapshot-2 region (for RDMA WR `local_addr`).
    #[inline]
    pub fn slot_addr2(&self, i: usize) -> u64 {
        debug_assert!(i < self.max_batch_size);
        self.device_ptr + ((self.max_batch_size + i) * self.slot_size) as u64
    }

    /// Raw device pointer to the snapshot-1 staging region (for CUDA kernel launch).
    pub fn staging_ptr(&self) -> u64 {
        self.device_ptr
    }

    /// Raw device pointer to the snapshot-2 staging region (for CUDA kernel launch).
    pub fn staging2_ptr(&self) -> u64 {
        self.device_ptr + (self.max_batch_size * self.slot_size) as u64
    }

    /// Slot size in bytes.
    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    /// Feature payload size in bytes.
    pub fn feature_bytes(&self) -> usize {
        self.feature_bytes
    }

    /// Maximum batch size.
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Grow staging VRAM + re-register once so `needed` slots fit.
    ///
    /// Registration stays off the per-gather hot path except when capacity
    /// actually increases (amortized). Existing gathers keep working at the
    /// old size until this returns.
    pub fn ensure_capacity(
        &mut self,
        rdma_ctx: &RdmaContext,
        stream: &Arc<CudaStream>,
        needed: usize,
        schema: &FeatureSchema,
    ) -> io::Result<()> {
        if needed <= self.max_batch_size {
            return Ok(());
        }
        let new_max = needed.next_power_of_two().max(needed);
        let slot_size = schema.slot_size;
        let feature_bytes = schema.feature_dim * std::mem::size_of::<f32>();
        let total_bytes = slot_size
            .checked_mul(new_max)
            .and_then(|b| b.checked_mul(2))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "staging grow overflow: slot_size {slot_size} * max_batch_size {new_max} * 2"
                    ),
                )
            })?;

        let allocation: CudaSlice<u8> = stream
            .alloc_zeros(total_bytes)
            .map_err(|e| io::Error::other(format!("cuMemAlloc failed during staging grow: {e}")))?;
        let device_ptr = {
            let (ptr, _guard) = allocation.device_ptr(stream);
            ptr
        };
        let access = ffi::IBV_ACCESS_LOCAL_WRITE;
        // SAFETY: new allocation outlives the MR via field order below.
        let mr = unsafe { reg_mr_cuda(rdma_ctx, device_ptr, total_bytes, access)? };

        self.mr = mr;
        self.device_ptr = device_ptr;
        self.slot_size = slot_size;
        self.max_batch_size = new_max;
        self.feature_bytes = feature_bytes;
        self._allocation = allocation;
        Ok(())
    }
}

// `mr` (RegisteredMr) drops first via field-declaration order, then
// `_allocation` frees the VRAM. No manual Drop impl needed.
