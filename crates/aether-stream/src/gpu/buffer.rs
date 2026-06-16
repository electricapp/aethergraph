//! VRAM staging buffer for GPUDirect RDMA feature gather.
//!
//! Allocates a contiguous VRAM region and registers it with the
//! InfiniBand NIC via nvidia-peermem for direct PCIe DMA.
//! When an RDMA READ completes, data lands directly in VRAM —
//! the host CPU's memory bus is never touched.

use crate::feature_table::FeatureSchema;
use crate::rdma::context::{RdmaContext, RegisteredMr};
use crate::rdma::ffi;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use std::io;
use std::sync::Arc;

/// VRAM staging buffer for batched RDMA READs of feature table slots.
///
/// Each slot in the staging buffer mirrors the server-side slot layout:
/// `[head_version: u64] [features: f32 × feature_dim] [tail_version: u64]`
///
/// After RDMA READs complete, the GPU kernel validates head == tail
/// and compacts valid features into the output tensor.
pub struct GpuGatherBuffer {
    /// MR registered against the NIC; declared first so it drops BEFORE
    /// `_allocation` — `ibv_dereg_mr` must run while the VRAM is still valid.
    mr: RegisteredMr,
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
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
    /// Requires the `nvidia-peermem` kernel module to be loaded.
    /// Returns `io::Error` if `ibv_reg_mr` fails on the GPU pointer
    /// (usually means nvidia-peermem is not loaded).
    pub fn new(
        rdma_ctx: &RdmaContext,
        cuda_ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        max_batch_size: usize,
        schema: &FeatureSchema,
    ) -> io::Result<Self> {
        let slot_size = schema.slot_size;
        let feature_bytes = schema.feature_dim * std::mem::size_of::<f32>();
        let total_bytes = slot_size.checked_mul(max_batch_size).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "staging buffer size overflow: slot_size {slot_size} * max_batch_size {max_batch_size}"
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
            ptr as u64
        };

        // Register GPU memory with the NIC via nvidia-peermem.
        // IBV_ACCESS_LOCAL_WRITE is required for RDMA READ target buffers.
        let access = ffi::IBV_ACCESS_LOCAL_WRITE;
        let mr = rdma_ctx.reg_mr(device_ptr as *mut u8, total_bytes, access)?;

        Ok(Self {
            ctx: cuda_ctx.clone(),
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

    /// Address of slot `i` in VRAM (for RDMA WR `local_addr`).
    #[inline]
    pub fn slot_addr(&self, i: usize) -> u64 {
        debug_assert!(i < self.max_batch_size);
        self.device_ptr + (i * self.slot_size) as u64
    }

    /// Raw device pointer to the staging region (for CUDA kernel launch).
    pub fn staging_ptr(&self) -> u64 {
        self.device_ptr
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
}

// `mr` (RegisteredMr) drops first via field-declaration order, then
// `_allocation` frees the VRAM. No manual Drop impl needed.
