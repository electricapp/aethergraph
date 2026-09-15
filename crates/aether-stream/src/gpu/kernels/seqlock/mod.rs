//! PTX-acquire seqlock snapshot reader (K5.3).
//!
//! Unlike [`super::validate::SeqlockValidator`], this reads one device-resident
//! FeatureTable image. Callers that need protection from independently DMAed
//! snapshots should continue to use the two-snapshot validator.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

pub(super) const KERNEL_SRC: &str = concat!(
    include_str!("../common.cuh"),
    "\n",
    include_str!("seqlock_reader.cu")
);
const KERNEL_NAME: &str = "seqlock_snapshot_rows";

/// Portable acceptance oracle; implemented in aethergraph-core so it is
/// available on hosts where this Linux/GPU module is not compiled.
pub use aethergraph_core::cpu_seqlock_accept;

/// Compiled K5.3 snapshot reader and its output buffers.
pub struct SeqlockSnapshotReader {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    output: CudaSlice<f32>,
    valid_mask: CudaSlice<i32>,
    feature_dim: usize,
    max_rows: usize,
}

impl SeqlockSnapshotReader {
    /// Compiles the reader with NVRTC and creates stream-ordered output buffers.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        max_rows: usize,
        feature_dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let module = ctx.load_module(super::compile_for_device(ctx, KERNEL_SRC)?)?;
        let func = module.load_function(KERNEL_NAME)?;
        Ok(Self {
            stream: stream.clone(),
            func,
            output: stream.alloc_zeros(max_rows * feature_dim)?,
            valid_mask: stream.alloc_zeros(max_rows)?,
            feature_dim,
            max_rows,
        })
    }

    /// Launch a stable-read check for `row_count` slots of `slot_size` bytes.
    pub fn snapshot(
        &mut self,
        slots_ptr: u64,
        slot_size: usize,
        row_count: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if row_count > self.max_rows {
            return Err(format!("row_count {row_count} exceeds {}", self.max_rows).into());
        }
        if row_count == 0 {
            return Ok(());
        }
        let feature_dim = i32::try_from(self.feature_dim)?;
        let row_count_i32 = i32::try_from(row_count)?;
        let slot_size_i32 = i32::try_from(slot_size)?;
        // One warp per row; the kernel grid-strides past the cap.
        let threads = 256u32;
        let blocks = (row_count as u32).div_ceil(threads / 32).clamp(1, 4096);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: CUDA arguments match the kernel signature and output
        // allocations cover every checked row.
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(&slots_ptr)
                .arg(&mut self.output)
                .arg(&mut self.valid_mask)
                .arg(&feature_dim)
                .arg(&row_count_i32)
                .arg(&slot_size_i32)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Feature output; valid rows are selected by [`Self::valid_mask`].
    pub fn output(&self) -> &CudaSlice<f32> {
        &self.output
    }

    /// One `1`/`0` value per snapshot row.
    pub fn valid_mask(&self) -> &CudaSlice<i32> {
        &self.valid_mask
    }
}
