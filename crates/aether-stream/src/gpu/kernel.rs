//! CUDA seqlock validation and feature compaction kernel.
//!
//! After both snapshot READ rounds land in the VRAM staging regions, this
//! kernel runs on the GPU to:
//! 1. Cross-validate the two snapshots of each slot: versions must all
//!    match (even, nonzero) and the payload bytes must be identical
//! 2. Compact valid features (from snapshot 1) into a contiguous output tensor
//! 3. Mark inconsistent rows for CPU-initiated retry of both snapshots
//!
//! The CUDA source lives in `validate_and_compact.cu` alongside this file.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

const KERNEL_SRC: &str = include_str!("validate_and_compact.cu");
const KERNEL_NAME: &str = "validate_and_compact";

/// GPU-side seqlock validator and feature compactor.
///
/// Compiled once via nvrtc, reused across gather calls.
pub struct SeqlockValidator {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    output: CudaSlice<f32>,
    retry_mask: CudaSlice<i32>,
    retry_count: CudaSlice<i32>,
    /// Pre-allocated single-element zero buffer for resetting retry_count
    /// without hitting cuMemAlloc/cuMemFree on every validate() call.
    zero_scratch: CudaSlice<i32>,
    feature_dim: usize,
    max_batch_size: usize,
}

impl SeqlockValidator {
    /// Compile the validation kernel and allocate output buffers.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        max_batch_size: usize,
        feature_dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Compile CUDA source to PTX via nvrtc
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SRC)?;
        let module = ctx.load_module(ptx)?;
        let func = module.load_function(KERNEL_NAME)?;

        // Allocate output buffers
        let output = stream.alloc_zeros::<f32>(max_batch_size * feature_dim)?;
        let retry_mask = stream.alloc_zeros::<i32>(max_batch_size)?;
        let retry_count = stream.alloc_zeros::<i32>(1)?;
        let zero_scratch = stream.alloc_zeros::<i32>(1)?;

        Ok(Self {
            stream: stream.clone(),
            func,
            output,
            retry_mask,
            retry_count,
            zero_scratch,
            feature_dim,
            max_batch_size,
        })
    }

    /// Launch validation kernel against two raw VRAM staging pointers.
    /// Returns the number of slots the kernel flagged for retry.
    ///
    /// `staging1_ptr` and `staging2_ptr` each point to `batch_size`
    /// consecutive slots of `slot_size` bytes — the two sequential snapshots
    /// of the same remote rows. Caller owns the buffers' lifetimes. This
    /// intentionally does not take a `GpuGatherBuffer` so that CUDA-only
    /// tests can exercise the kernel without an RDMA-registered MR.
    pub fn validate(
        &mut self,
        staging1_ptr: u64,
        staging2_ptr: u64,
        slot_size: usize,
        batch_size: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if batch_size > self.max_batch_size {
            return Err(format!(
                "batch_size {batch_size} exceeds kernel max_batch_size {}",
                self.max_batch_size
            )
            .into());
        }

        // Zero the retry counter from pre-allocated scratch (no cuMemAlloc)
        self.stream
            .memcpy_dtod(&self.zero_scratch, &mut self.retry_count)?;

        // Launch kernel
        let threads_per_block = 256u32;
        let blocks = ((batch_size as u32) + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let feature_dim = self.feature_dim as i32;
        let batch_size_i32 = batch_size as i32;
        let slot_size_i32 = slot_size as i32;

        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(&staging1_ptr)
                .arg(&staging2_ptr)
                .arg(&mut self.output)
                .arg(&mut self.retry_mask)
                .arg(&mut self.retry_count)
                .arg(&feature_dim)
                .arg(&batch_size_i32)
                .arg(&slot_size_i32)
                .launch(cfg)?;
        }

        // Synchronize the stream before reading back to host so the kernel and
        // all preceding async ops have completed.
        //
        // Cost note: this full device sync plus the D2H `retry_count` copy below
        // runs on EVERY `validate` call — once per gather and again on each
        // torn-read retry iteration in `RdmaFeatureClient::gather`. On a clean
        // (untorn) gather that is one host/device round-trip per batch. A
        // lower-latency design would poll a host-pinned mapped counter or fuse
        // validation into the post path to avoid the per-retry sync; that
        // restructuring is deferred (it needs on-hardware verification).
        self.stream.synchronize()?;

        // Read retry count back to host
        let mut count = [0i32];
        self.stream.memcpy_dtoh(&self.retry_count, &mut count)?;
        if count[0] < 0 {
            return Err(format!("kernel returned negative retry count: {}", count[0]).into());
        }
        Ok(count[0] as usize)
    }

    /// Get the output CudaSlice directly (caller manages stream ordering).
    pub fn output(&self) -> &CudaSlice<f32> {
        &self.output
    }

    /// Stream used by this validator (for callers that need stream-ordered access).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Get indices of nodes that need retry (torn reads).
    ///
    /// Copies retry_mask from device to host and returns indices where mask == 1.
    pub fn retry_indices(
        &self,
        batch_size: usize,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        // `retry_mask` is allocated at max_batch_size; only the first
        // `batch_size` entries belong to this gather. Copy exactly those —
        // the D2H copy requires dst.len() >= src.len().
        let mut mask = vec![0i32; batch_size];
        self.stream
            .memcpy_dtoh(&self.retry_mask.slice(0..batch_size), &mut mask)?;
        Ok(mask
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0)
            .map(|(i, _)| i)
            .collect())
    }
}
