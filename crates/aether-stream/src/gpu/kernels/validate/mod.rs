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
//!
//! # K5.0 — captured launch graphs + mapped retry_count
//!
//! Steady-state validates with a fixed `(staging1, staging2, slot_size,
//! batch_size)` signature replay a captured [`CudaGraph`] (kernel launch)
//! instead of rebuilding the launch sequence every batch. `retry_count` is a
//! host-mapped (`CU_MEMHOSTALLOC_DEVICEMAP`) counter when the driver allows
//! it: the host zeros and reads it directly after stream sync, so there is
//! no D2H memcpy for the counter. RDMA READs stay outside the graph.
//!
//! cudarc's [`CudaStream::alloc_zeros`] uses stream-ordered `cuMemAllocAsync`.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    result, sys,
};
use std::sync::Arc;

const KERNEL_SRC: &str = concat!(
    include_str!("../common.cuh"),
    "\n",
    include_str!("validate_and_compact.cu")
);
const KERNEL_NAME: &str = "validate_and_compact";

/// Cached CUDA graph for one validate signature (K5.0).
struct CapturedValidate {
    staging1: u64,
    staging2: u64,
    slot_size: usize,
    batch_size: usize,
    graph: CudaGraph,
}

/// Device-visible retry counter. Prefers host-mapped memory so the host can
/// read the result after synchronize without a D2H copy.
enum RetryCount {
    Mapped {
        host: *mut i32,
        device: u64,
    },
    Device {
        slice: CudaSlice<i32>,
        zero_scratch: CudaSlice<i32>,
    },
}

// SAFETY: the pointer is exclusively owned by SeqlockValidator and only
// touched on the CUDA context's thread after stream synchronization.
unsafe impl Send for RetryCount {}
unsafe impl Sync for RetryCount {}

impl Drop for RetryCount {
    fn drop(&mut self) {
        if let Self::Mapped { host, .. } = self {
            // SAFETY: host came from cuMemAllocHost / malloc_host.
            let _ = unsafe { result::free_host(*host as *mut _) };
        }
    }
}

impl RetryCount {
    fn try_mapped(_ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
        // SAFETY: one i32 of unset host memory; we zero it before first use.
        let host = unsafe {
            result::malloc_host(
                std::mem::size_of::<i32>(),
                sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE,
            )?
        } as *mut i32;
        let mut device = 0u64;
        // SAFETY: host is a live DEVICEMAP allocation; flags must be 0.
        let status = unsafe {
            sys::cuMemHostGetDevicePointer_v2(
                &mut device as *mut u64 as *mut sys::CUdeviceptr,
                host as *mut _,
                0,
            )
        };
        if let Err(e) = status.result() {
            let _ = unsafe { result::free_host(host as *mut _) };
            return Err(e.into());
        }
        unsafe {
            *host = 0;
        }
        Ok(Self::Mapped { host, device })
    }

    fn device_fallback(stream: &Arc<CudaStream>) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::Device {
            slice: stream.alloc_zeros::<i32>(1)?,
            zero_scratch: stream.alloc_zeros::<i32>(1)?,
        })
    }

    fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match Self::try_mapped(ctx) {
            Ok(mapped) => Ok(mapped),
            Err(e) => {
                tracing::debug!(error = %e, "mapped retry_count unavailable; device fallback");
                Self::device_fallback(stream)
            }
        }
    }

    fn is_mapped(&self) -> bool {
        matches!(self, Self::Mapped { .. })
    }

    /// Zero before enqueue. Mapped path writes the host view; device path
    /// uses a stream-ordered DtoD from a permanent zero scratch.
    fn clear(&mut self, stream: &CudaStream) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Self::Mapped { host, .. } => {
                // SAFETY: exclusive host mapping; kernel is not running yet
                // (caller clears before launch / outside the captured graph).
                unsafe {
                    *host = 0;
                }
                Ok(())
            }
            Self::Device {
                slice,
                zero_scratch,
            } => {
                stream.memcpy_dtod(zero_scratch, slice)?;
                Ok(())
            }
        }
    }

    fn read_after_sync(&self) -> Result<i32, Box<dyn std::error::Error>> {
        match self {
            Self::Mapped { host, .. } => {
                // SAFETY: caller synchronized the stream; kernel writes are visible.
                Ok(unsafe { *host })
            }
            Self::Device { .. } => {
                Err("device retry_count requires D2H via SeqlockValidator::finish_validate".into())
            }
        }
    }
}

/// GPU-side seqlock validator and feature compactor.
pub struct SeqlockValidator {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    output: CudaSlice<f32>,
    retry_mask: CudaSlice<i32>,
    retry_count: RetryCount,
    /// Device fallback zero scratch is inside RetryCount::Device; this field
    /// is only kept so ensure_capacity paths stay simple when remapping.
    feature_dim: usize,
    max_batch_size: usize,
    captured: Option<CapturedValidate>,
    use_graph: bool,
}

impl SeqlockValidator {
    /// Compile the validation kernel and allocate output buffers.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        max_batch_size: usize,
        feature_dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SRC)?;
        let module = ctx.load_module(ptx)?;
        let func = module.load_function(KERNEL_NAME)?;

        Ok(Self {
            stream: stream.clone(),
            func,
            output: stream.alloc_zeros::<f32>(max_batch_size * feature_dim)?,
            retry_mask: stream.alloc_zeros::<i32>(max_batch_size)?,
            retry_count: RetryCount::new(ctx, stream)?,
            feature_dim,
            max_batch_size,
            captured: None,
            use_graph: true,
        })
    }

    /// Enable or disable K5.0 CUDA-graph replay (default: enabled).
    pub fn set_use_graph(&mut self, enabled: bool) {
        self.use_graph = enabled;
        if !enabled {
            self.captured = None;
        }
    }

    /// Whether the next matching validate will replay a captured graph.
    pub fn has_captured_graph(&self) -> bool {
        self.captured.is_some()
    }

    /// Whether retry_count is host-mapped (no D2H on the hot path).
    pub fn has_mapped_retry_count(&self) -> bool {
        self.retry_count.is_mapped()
    }

    /// Grow host-visible validation buffers so `needed` rows fit.
    pub fn ensure_capacity(&mut self, needed: usize) -> Result<(), Box<dyn std::error::Error>> {
        if needed <= self.max_batch_size {
            return Ok(());
        }
        let new_max = needed.next_power_of_two().max(needed);
        self.output = self.stream.alloc_zeros::<f32>(new_max * self.feature_dim)?;
        self.retry_mask = self.stream.alloc_zeros::<i32>(new_max)?;
        self.max_batch_size = new_max;
        self.captured = None;
        Ok(())
    }

    /// Launch validation against two VRAM staging pointers; returns torn rows.
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
        if batch_size == 0 {
            return Ok(0);
        }

        // Zero the counter on the host (mapped) or via DtoD (fallback) before
        // any capture/replay so the graph body is just the kernel launch.
        self.retry_count.clear(&self.stream)?;

        if self.use_graph {
            let replay = self.captured.as_ref().is_some_and(|cap| {
                cap.staging1 == staging1_ptr
                    && cap.staging2 == staging2_ptr
                    && cap.slot_size == slot_size
                    && cap.batch_size == batch_size
            });
            if replay {
                self.captured
                    .as_ref()
                    .expect("replay implies captured")
                    .graph
                    .launch()?;
                return self.finish_validate();
            }
            match self.capture_and_launch(staging1_ptr, staging2_ptr, slot_size, batch_size) {
                Ok(()) => return self.finish_validate(),
                Err(e) => {
                    tracing::debug!(error = %e, "CUDA graph capture failed; eager validate");
                    self.captured = None;
                    // Counter was already cleared; re-clear in case capture
                    // partially ran the kernel (it should not have).
                    self.retry_count.clear(&self.stream)?;
                }
            }
        }

        self.enqueue_kernel(staging1_ptr, staging2_ptr, slot_size, batch_size)?;
        self.finish_validate()
    }

    fn enqueue_kernel(
        &mut self,
        staging1_ptr: u64,
        staging2_ptr: u64,
        slot_size: usize,
        batch_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let threads_per_block = 256u32;
        let blocks = (batch_size as u32).div_ceil(threads_per_block);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let feature_dim = self.feature_dim as i32;
        let batch_size_i32 = batch_size as i32;
        let slot_size_i32 = slot_size as i32;

        // SAFETY: args match validate_and_compact; buffers sized for batch.
        unsafe {
            let mut launch = self.stream.launch_builder(&self.func);
            launch
                .arg(&staging1_ptr)
                .arg(&staging2_ptr)
                .arg(&mut self.output)
                .arg(&mut self.retry_mask);
            match &mut self.retry_count {
                RetryCount::Mapped { device, .. } => {
                    launch.arg(device);
                }
                RetryCount::Device { slice, .. } => {
                    launch.arg(slice);
                }
            }
            launch
                .arg(&feature_dim)
                .arg(&batch_size_i32)
                .arg(&slot_size_i32)
                .launch(cfg)?;
        }
        Ok(())
    }

    fn capture_and_launch(
        &mut self,
        staging1_ptr: u64,
        staging2_ptr: u64,
        slot_size: usize,
        batch_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)?;
        let enqueue_result = self.enqueue_kernel(staging1_ptr, staging2_ptr, slot_size, batch_size);
        let graph = match self
            .stream
            .end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD)
        {
            Ok(Some(g)) => g,
            Ok(None) => {
                enqueue_result?;
                return Err("CUDA stream capture produced a null graph".into());
            }
            Err(e) => {
                let _ = enqueue_result;
                return Err(e.into());
            }
        };
        enqueue_result?;
        graph.upload()?;
        graph.launch()?;
        self.captured = Some(CapturedValidate {
            staging1: staging1_ptr,
            staging2: staging2_ptr,
            slot_size,
            batch_size,
            graph,
        });
        Ok(())
    }

    fn finish_validate(&mut self) -> Result<usize, Box<dyn std::error::Error>> {
        self.stream.synchronize()?;
        let count = match &self.retry_count {
            RetryCount::Mapped { .. } => self.retry_count.read_after_sync()?,
            RetryCount::Device { slice, .. } => {
                let mut count = [0i32];
                self.stream.memcpy_dtoh(slice, &mut count)?;
                count[0]
            }
        };
        if count < 0 {
            return Err(format!("kernel returned negative retry count: {count}").into());
        }
        Ok(count as usize)
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
    pub fn retry_indices(
        &self,
        batch_size: usize,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
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
