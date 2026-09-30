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
//! One warp per row, so both snapshot reads and the compacted write are
//! coalesced; the grid is sized in warps and grid-strides. Rows flagged in
//! [`SeqlockValidator::retry_indices`] hold undefined bytes in
//! [`SeqlockValidator::output`]: the compare aborts on the first mismatching
//! chunk rather than pay a second pass to keep them pristine.
//!
//! # K5.0 — captured launch graphs + mapped retry_count
//!
//! Steady-state validates with a fixed `(staging1, staging2, batch_size)`
//! signature replay a captured [`CudaGraph`] (kernel launch) instead of
//! rebuilding the launch sequence every batch. Capture needs a real stream —
//! the legacy default stream cannot be captured — and runs in thread-local
//! mode, so CUDA calls other threads make meanwhile (PyTorch's allocator
//! among them) are unaffected. The kernel's buffers are passed as raw device
//! pointers, so the captured launch carries no event bookkeeping; the stream
//! synchronize that ends every validate orders all later access. A failed
//! capture disables graphs for the validator's life. `retry_count` is a
//! host-mapped (`CU_MEMHOSTALLOC_DEVICEMAP`) counter when the driver allows
//! it: the host zeros and reads it directly after stream sync, so there is
//! no D2H memcpy for the counter. RDMA READs stay outside the graph.
//!
//! cudarc's [`CudaStream::alloc_zeros`] uses stream-ordered `cuMemAllocAsync`.

use crate::rdma::layout::SlotGeometry;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg, result, sys,
};
use std::sync::Arc;

pub(super) const KERNEL_SRC: &str = concat!(
    include_str!("../common.cuh"),
    "\n",
    include_str!("validate_and_compact.cu")
);
const KERNEL_NAME: &str = "validate_and_compact";

/// Ceiling on the validate grid; 4096 blocks of 8 warps oversubscribes any
/// current part, and the kernel's grid-stride loop takes the rest.
const MAX_VALIDATE_BLOCKS: u32 = 4096;

/// The two snapshot staging regions a validate reads: `rows` slots each at
/// the geometry's stride. Construction is the proof the kernel's reads stay
/// inside them and its vector loads are aligned; `'a` ties the regions to
/// the allocation backing them.
#[derive(Debug, Clone, Copy)]
pub struct StagingRegions<'a> {
    snap1: u64,
    snap2: u64,
    rows: usize,
    geometry: SlotGeometry,
    _alloc: std::marker::PhantomData<&'a ()>,
}

impl StagingRegions<'_> {
    /// Describe two staging regions of `rows` slots at `geometry`.
    ///
    /// Both bases must be 16-byte aligned — with the geometry's 16-aligned
    /// stride that keeps every payload on a vector boundary.
    ///
    /// # Safety
    /// `[snap1, snap1 + rows * stride)` and `[snap2, snap2 + rows * stride)`
    /// must be device memory on the validator's context that stays allocated
    /// for the lifetime the returned regions carry.
    pub unsafe fn new(
        snap1: u64,
        snap2: u64,
        rows: usize,
        geometry: SlotGeometry,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        const ALIGN: u64 = 16;
        for (name, ptr) in [("staging1", snap1), ("staging2", snap2)] {
            if !ptr.is_multiple_of(ALIGN) {
                return Err(format!("{name} pointer {ptr:#x} is not {ALIGN}-byte aligned").into());
            }
        }
        if i32::try_from(rows).is_err() {
            return Err(format!("{rows} staging rows exceed the kernel's i32 index").into());
        }
        Ok(Self {
            snap1,
            snap2,
            rows,
            geometry,
            _alloc: std::marker::PhantomData,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn geometry(&self) -> &SlotGeometry {
        &self.geometry
    }
}

/// Cached CUDA graph for one validate signature (K5.0).
struct CapturedValidate {
    staging1: u64,
    staging2: u64,
    batch_size: usize,
    graph: CudaGraph,
}

// SAFETY: the graph and its exec handle are owned for the struct's lifetime
// and only launched through `&mut SeqlockValidator` on the owning context,
// which CUDA makes thread-safe after creation.
unsafe impl Send for CapturedValidate {}
// SAFETY: see the Send impl above.
unsafe impl Sync for CapturedValidate {}

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
        /// `slice`'s device address, resolved once outside any capture.
        device: u64,
    },
}

// SAFETY: the pointer is exclusively owned by SeqlockValidator and only
// touched on the CUDA context's thread after stream synchronization.
unsafe impl Send for RetryCount {}
// SAFETY: as above — no shared mutation without stream synchronization.
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
            // SAFETY: host is the live allocation from malloc_host above, and
            // nothing else has taken ownership of it on this path.
            let _ = unsafe { result::free_host(host as *mut _) };
            return Err(e.into());
        }
        // SAFETY: host is a live, suitably aligned i32 allocation not yet
        // shared with the device.
        unsafe {
            *host = 0;
        }
        Ok(Self::Mapped { host, device })
    }

    fn device_fallback(stream: &Arc<CudaStream>) -> Result<Self, Box<dyn std::error::Error>> {
        let slice = stream.alloc_zeros::<i32>(1)?;
        let device = raw_ptr(&slice, stream);
        Ok(Self::Device {
            slice,
            zero_scratch: stream.alloc_zeros::<i32>(1)?,
            device,
        })
    }

    /// Device address the kernel increments.
    fn device_ptr(&self) -> u64 {
        match self {
            Self::Mapped { device, .. } | Self::Device { device, .. } => *device,
        }
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
    fn clear(&mut self, stream: &Arc<CudaStream>) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Self::Mapped { host, .. } => {
                // SAFETY: exclusive host mapping; kernel is not running yet
                // (caller clears before launch / outside the captured graph).
                unsafe {
                    **host = 0;
                }
                Ok(())
            }
            Self::Device {
                slice,
                zero_scratch,
                ..
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
                Ok(unsafe { **host })
            }
            Self::Device { .. } => {
                Err("device retry_count requires D2H via SeqlockValidator::finish_validate".into())
            }
        }
    }
}

/// Device address of `slice` on `stream`, for raw kernel arguments.
fn raw_ptr<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> u64 {
    let (ptr, _record) = slice.device_ptr(stream);
    ptr
}

/// Output rows and per-row retry flags, sized for `rows`, with the raw
/// addresses the kernel is launched against.
struct ValidateBuffers {
    output: CudaSlice<f32>,
    retry_mask: CudaSlice<i32>,
    output_ptr: u64,
    retry_mask_ptr: u64,
    rows: usize,
}

impl ValidateBuffers {
    fn alloc(
        stream: &Arc<CudaStream>,
        rows: usize,
        feature_dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let len = rows
            .checked_mul(feature_dim)
            .ok_or("validator output size overflows usize")?;
        let output = stream.alloc_zeros::<f32>(len.max(1))?;
        let retry_mask = stream.alloc_zeros::<i32>(rows.max(1))?;
        let output_ptr = raw_ptr(&output, stream);
        let retry_mask_ptr = raw_ptr(&retry_mask, stream);
        Ok(Self {
            output,
            retry_mask,
            output_ptr,
            retry_mask_ptr,
            rows,
        })
    }
}

/// GPU-side seqlock validator and feature compactor.
pub struct SeqlockValidator {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    buffers: ValidateBuffers,
    retry_count: RetryCount,
    geometry: SlotGeometry,
    captured: Option<CapturedValidate>,
    use_graph: bool,
}

impl SeqlockValidator {
    /// Compile the validation kernel and allocate output buffers for
    /// `max_batch_size` rows of `geometry`'s slots.
    ///
    /// Graph replay is on when `stream` can be captured — not the legacy
    /// default stream.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        max_batch_size: usize,
        geometry: &SlotGeometry,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = super::compile_for_device(ctx, KERNEL_SRC)?;
        let module = ctx.load_module(ptx)?;
        let func = module.load_function(KERNEL_NAME)?;

        Ok(Self {
            stream: stream.clone(),
            func,
            buffers: ValidateBuffers::alloc(stream, max_batch_size, geometry.feature_dim())?,
            retry_count: RetryCount::new(ctx, stream)?,
            geometry: *geometry,
            captured: None,
            use_graph: !stream.cu_stream().is_null(),
        })
    }

    /// Enable or disable K5.0 CUDA-graph replay (default: enabled on a
    /// capturable stream). Enabling on the legacy default stream is ignored.
    pub fn set_use_graph(&mut self, enabled: bool) {
        self.use_graph = enabled && !self.stream.cu_stream().is_null();
        if !self.use_graph {
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
        if needed <= self.buffers.rows {
            return Ok(());
        }
        let new_max = needed.next_power_of_two().max(needed);
        self.buffers = ValidateBuffers::alloc(&self.stream, new_max, self.geometry.feature_dim())?;
        self.captured = None;
        Ok(())
    }

    /// Validate the first `batch_size` rows of `staging`; returns torn rows.
    pub fn validate(
        &mut self,
        staging: &StagingRegions<'_>,
        batch_size: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if staging.geometry != self.geometry {
            return Err(format!(
                "staging geometry {:?} differs from the validator's {:?}",
                staging.geometry, self.geometry
            )
            .into());
        }
        let bound = self.buffers.rows.min(staging.rows);
        if batch_size > bound {
            return Err(format!(
                "batch_size {batch_size} exceeds {} validator rows / {} staging rows",
                self.buffers.rows, staging.rows
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
                cap.staging1 == staging.snap1
                    && cap.staging2 == staging.snap2
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
            match self.capture_and_launch(staging, batch_size) {
                Ok(()) => return self.finish_validate(),
                Err(e) => {
                    // A stack that refuses capture refuses it every time;
                    // don't pay the attempt on every batch.
                    tracing::debug!(error = %e, "CUDA graph capture failed; eager from now on");
                    self.use_graph = false;
                    self.captured = None;
                    // Counter was already cleared; re-clear in case capture
                    // partially ran the kernel (it should not have).
                    self.retry_count.clear(&self.stream)?;
                }
            }
        }

        self.enqueue_kernel(staging, batch_size)?;
        self.finish_validate()
    }

    fn enqueue_kernel(
        &self,
        staging: &StagingRegions<'_>,
        batch_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Sized in warps, capped, and grid-strided past the cap.
        let threads_per_block = 256u32;
        let warps_per_block = threads_per_block / 32;
        let blocks = u32::try_from(batch_size)
            .unwrap_or(u32::MAX)
            .div_ceil(warps_per_block)
            .clamp(1, MAX_VALIDATE_BLOCKS);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        // All in i32 range: `SlotGeometry` bounds the stride (and so the
        // dim) by i32::MAX, `StagingRegions` the row count.
        let feature_dim = self.geometry.feature_dim() as i32;
        let batch_size_i32 = batch_size as i32;
        let slot_size_i32 = self.geometry.stride() as i32;
        let retry_count = self.retry_count.device_ptr();

        // SAFETY: args match validate_and_compact. The staging reads stay in
        // the regions `StagingRegions` vouches for (batch_size <= rows at the
        // geometry's stride, tail inside the stride); the output and retry
        // buffers hold `buffers.rows >= batch_size` rows.
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(&staging.snap1)
                .arg(&staging.snap2)
                .arg(&self.buffers.output_ptr)
                .arg(&self.buffers.retry_mask_ptr)
                .arg(&retry_count)
                .arg(&feature_dim)
                .arg(&batch_size_i32)
                .arg(&slot_size_i32)
                .launch(cfg)?;
        }
        Ok(())
    }

    fn capture_and_launch(
        &mut self,
        staging: &StagingRegions<'_>,
        batch_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let enqueue_result = self.enqueue_kernel(staging, batch_size);
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
            staging1: staging.snap1,
            staging2: staging.snap2,
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
    ///
    /// Rows listed by [`Self::retry_indices`] hold undefined bytes here.
    pub fn output(&self) -> &CudaSlice<f32> {
        &self.buffers.output
    }

    /// Stream used by this validator (for callers that need stream-ordered access).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Slot layout this validator reads.
    pub fn geometry(&self) -> &SlotGeometry {
        &self.geometry
    }

    /// Get indices of nodes that need retry (torn reads).
    pub fn retry_indices(
        &self,
        batch_size: usize,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        if batch_size > self.buffers.rows {
            return Err(format!(
                "batch_size {batch_size} exceeds {} validator rows",
                self.buffers.rows
            )
            .into());
        }
        let mut mask = vec![0i32; batch_size];
        self.stream
            .memcpy_dtoh(&self.buffers.retry_mask.slice(0..batch_size), &mut mask)?;
        Ok(mask
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0)
            .map(|(i, _)| i)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Refused before any pointer is used, so no device is needed.
    #[test]
    fn staging_regions_refuse_what_the_kernel_cannot_read() {
        let g = SlotGeometry::packed(128).unwrap();
        // SAFETY: construction is refused before either pointer is used.
        assert!(unsafe { StagingRegions::new(0x1008, 0x2000, 4, g) }.is_err());
        // SAFETY: as above.
        assert!(unsafe { StagingRegions::new(0x1000, 0x2004, 4, g) }.is_err());
        // SAFETY: as above.
        assert!(unsafe { StagingRegions::new(0x1000, 0x2000, i32::MAX as usize + 1, g) }.is_err());
        // SAFETY: never launched.
        let ok = unsafe { StagingRegions::new(0x1000, 0x2000, 4, g) }.unwrap();
        assert_eq!(ok.rows(), 4);
        assert_eq!(ok.geometry(), &g);
    }
}
