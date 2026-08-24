//! K5.4 dense aggregation after gather (not sparse gather).
//!
//! Bandwidth-roofed GEMV: B staged in shared memory, A via `ld.global.cs.v4`.
//! Full Hopper TMA/WGMMA and Blackwell tcgen05 need host tensor-map builders
//! (`require_sm` 90+ / 100+) — see `TODO(HARDWARE)`.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

const KERNEL_SRC: &str = concat!(
    include_str!("../common.cuh"),
    "\n",
    include_str!("tma_wgmma.cu")
);
const KERNEL_NAME: &str = "dense_tile_accumulate";
/// Soft cap so dynamic smem stays within the default 48 KiB budget.
const MAX_B_SMEM_COLS: u32 = 12_288;

/// Intended stages of a TMA-to-tensor-core feature tile pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorTileStage {
    /// Transfer a feature tile into shared memory with TMA.
    Transfer,
    /// Consume a Hopper WGMMA tile.
    Wgmma,
    /// Consume a Blackwell tcgen05 tile.
    Tcgen05,
}

/// Shape metadata for a dense aggregation tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorTileShape {
    pub rows: u32,
    pub cols: u32,
    pub element_bytes: u8,
}

/// Dense tile accumulator used as the K5.4 oracle-friendly baseline.
pub struct TmaAggregator {
    stream: Arc<CudaStream>,
    func: CudaFunction,
}

impl TmaAggregator {
    /// Compile the dense accumulate kernel.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let module = ctx.load_module(cudarc::nvrtc::compile_ptx(KERNEL_SRC)?)?;
        Ok(Self {
            stream: stream.clone(),
            func: module.load_function(KERNEL_NAME)?,
        })
    }

    /// `out[row] += sum_k a[row, k] * b[k]` for a dense tile (aggregation after gather).
    pub fn accumulate(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: u32,
        cols: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        if cols > MAX_B_SMEM_COLS {
            return Err(format!(
                "cols {cols} exceeds smem tile cap {MAX_B_SMEM_COLS}; tile B on host or raise cap"
            )
            .into());
        }
        let rows_i = rows as i32;
        let cols_i = cols as i32;
        let threads = 256u32;
        // SAFETY: matches dense_tile_accumulate; buffers sized by caller.
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(a)
                .arg(b)
                .arg(out)
                .arg(&rows_i)
                .arg(&cols_i)
                .launch(LaunchConfig {
                    grid_dim: (rows.div_ceil(threads), 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: cols * 4,
                })?;
        }
        Ok(())
    }
}

// TODO(HARDWARE): wire cuTensorMapEncodeTiled + WGMMA/tcgen05 on SM90/SM100;
// this path is the bandwidth-roofed GEMV stand-in.
