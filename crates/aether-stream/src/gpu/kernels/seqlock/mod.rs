//! PTX-acquire seqlock snapshot reader (K5.3).
//!
//! Unlike [`super::validate::SeqlockValidator`], this reads one device-resident
//! FeatureTable image. Callers that need protection from independently DMAed
//! snapshots should continue to use the two-snapshot validator.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

pub(super) const KERNEL_SRC: &str = concat!(
    include_str!("../common.cuh"),
    "\n",
    include_str!("seqlock_reader.cu")
);
const KERNEL_NAME: &str = "seqlock_snapshot_rows";

/// Alignment the kernel's `.v4` payload loads need from every slot base.
const SLOT_ALIGN: usize = 16;

/// Portable acceptance oracle; implemented in aethergraph-core so it is
/// available on hosts where this Linux/GPU module is not compiled.
pub use aethergraph_core::cpu_seqlock_accept;

/// Compiled K5.3 snapshot reader and its output buffers.
///
/// The slot geometry (`feature_dim`, `slot_size`) is checked once at
/// construction; [`Self::snapshot`] then only has to bound the buffer.
pub struct SeqlockSnapshotReader {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    output: CudaSlice<f32>,
    valid_mask: CudaSlice<i32>,
    feature_dim: i32,
    slot_size: usize,
    slot_stride: i64,
    max_rows: usize,
}

impl SeqlockSnapshotReader {
    /// Compile the reader for slots of `feature_dim` features laid out
    /// `slot_size` bytes apart, with output room for `max_rows` rows.
    ///
    /// `slot_size` must hold the head, payload, and tail
    /// ([`aethergraph_core::feature_slot_size`]) and be a multiple of 16, so
    /// every slot's payload keeps the alignment the vector loads need.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        max_rows: usize,
        feature_dim: usize,
        slot_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if feature_dim == 0 {
            return Err("feature_dim must be nonzero".into());
        }
        let min_slot = aethergraph_core::feature_slot_size(feature_dim);
        if slot_size < min_slot || !slot_size.is_multiple_of(SLOT_ALIGN) {
            return Err(format!(
                "slot_size {slot_size} must be >= {min_slot} and a multiple of {SLOT_ALIGN}"
            )
            .into());
        }
        let feature_dim_i32 = i32::try_from(feature_dim)?;
        let slot_stride = i64::try_from(slot_size)?;
        let output_len = max_rows
            .checked_mul(feature_dim)
            .ok_or("max_rows * feature_dim overflows")?;
        let module = ctx.load_module(super::compile_for_device(ctx, KERNEL_SRC)?)?;
        let func = module.load_function(KERNEL_NAME)?;
        Ok(Self {
            stream: stream.clone(),
            func,
            output: stream.alloc_zeros(output_len.max(1))?,
            valid_mask: stream.alloc_zeros(max_rows.max(1))?,
            feature_dim: feature_dim_i32,
            slot_size,
            slot_stride,
            max_rows,
        })
    }

    /// Launch a stable-read check over the first `row_count` slots of
    /// `slots`.
    pub fn snapshot<S: DevicePtr<u8>>(
        &mut self,
        slots: &S,
        row_count: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if row_count > self.max_rows {
            return Err(format!("row_count {row_count} exceeds {}", self.max_rows).into());
        }
        if row_count == 0 {
            return Ok(());
        }
        let span = row_count
            .checked_mul(self.slot_size)
            .ok_or("row_count * slot_size overflows")?;
        if span > slots.len() {
            return Err(format!(
                "{row_count} slots of {} bytes exceed the {}-byte buffer",
                self.slot_size,
                slots.len()
            )
            .into());
        }
        let row_count_i32 = i32::try_from(row_count)?;
        // One warp per row; the kernel grid-strides past the cap.
        let threads = 256u32;
        let blocks = u32::try_from(row_count.div_ceil((threads / 32) as usize))
            .unwrap_or(u32::MAX)
            .clamp(1, 4096);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let (slots_ptr, _record) = slots.device_ptr(&self.stream);
        if !slots_ptr.is_multiple_of(SLOT_ALIGN as u64) {
            return Err(
                format!("slot base {slots_ptr:#x} is not {SLOT_ALIGN}-byte aligned").into(),
            );
        }
        // SAFETY: arguments match the kernel signature; `slots` covers
        // `row_count` slots of the checked geometry and the outputs hold
        // `max_rows >= row_count` rows. `_record` orders the read after
        // earlier writes to `slots` and records it for later ones.
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(&slots_ptr)
                .arg(&mut self.output)
                .arg(&mut self.valid_mask)
                .arg(&self.feature_dim)
                .arg(&row_count_i32)
                .arg(&self.slot_stride)
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
