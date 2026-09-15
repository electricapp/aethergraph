//! K5.5 feature-payload decode — block-scaled int8 rows expanded in VRAM.
//!
//! [`super::decompress`] decodes topology; this decodes the payload, the term
//! that scales with batch size and sets the gather's streaming asymptote.
//!
//! Output is bit-identical to [`BlockScaledI8::decode`] — both are one IEEE
//! multiply — which is what lets the oracle test assert equality.

use aethergraph_core::BlockScaledI8;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

pub(super) const KERNEL_SRC: &str = concat!(
    include_str!("../common.cuh"),
    "\n",
    include_str!("feature_dequant.cu")
);
const KERNEL_NAME: &str = "dequant_block_i8_rows";

/// Ceiling on the dequant grid; the kernel grid-strides past it.
const MAX_DEQUANT_BLOCKS: u32 = 4096;

/// One encoded feature block resident in VRAM.
pub struct QuantizedRowsDevice {
    /// `rows * feature_dim` codes, row-major, as raw bytes.
    pub codes: CudaSlice<u8>,
    /// `rows * blocks_per_row` scales, row-major.
    pub scales: CudaSlice<f32>,
    pub rows: usize,
    pub feature_dim: usize,
}

impl QuantizedRowsDevice {
    /// Upload an encoded block onto `stream`.
    pub fn upload(
        stream: &Arc<CudaStream>,
        encoded: &BlockScaledI8,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // i8 and u8 have the same layout; the kernel re-signs each byte on
        // read, so the transfer is a plain byte copy.
        // SAFETY: `i8` and `u8` have identical size and alignment, the
        // pointer comes from a live slice of that length, and the borrow
        // lives no longer than `encoded`.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(encoded.codes().as_ptr().cast::<u8>(), encoded.codes().len())
        };
        let mut codes = stream.alloc_zeros::<u8>(bytes.len().max(1))?;
        if !bytes.is_empty() {
            stream.memcpy_htod(bytes, &mut codes)?;
        }
        let mut scales = stream.alloc_zeros::<f32>(encoded.scales().len().max(1))?;
        if !encoded.scales().is_empty() {
            stream.memcpy_htod(encoded.scales(), &mut scales)?;
        }
        Ok(Self {
            codes,
            scales,
            rows: encoded.rows(),
            feature_dim: encoded.dim(),
        })
    }
}

/// Compiled block-scaled int8 row decoder and its output buffer.
pub struct FeatureDequantizer {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    output: CudaSlice<f32>,
    capacity: usize,
}

impl FeatureDequantizer {
    /// Compile the decoder and allocate room for `max_elements` f32.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        max_elements: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let module = ctx.load_module(super::compile_for_device(ctx, KERNEL_SRC)?)?;
        Ok(Self {
            stream: stream.clone(),
            func: module.load_function(KERNEL_NAME)?,
            output: stream.alloc_zeros::<f32>(max_elements.max(1))?,
            capacity: max_elements,
        })
    }

    /// Enqueue a decode of every row in `src`.
    pub fn decode(&mut self, src: &QuantizedRowsDevice) -> Result<(), Box<dyn std::error::Error>> {
        let elements = src.rows * src.feature_dim;
        if elements > self.capacity {
            return Err(format!("{elements} elements exceeds capacity {}", self.capacity).into());
        }
        if elements == 0 {
            return Ok(());
        }
        let feature_dim = i32::try_from(src.feature_dim)?;
        let rows = i32::try_from(src.rows)?;
        let threads = 256u32;
        let blocks = (src.rows as u32)
            .div_ceil(threads / 32)
            .clamp(1, MAX_DEQUANT_BLOCKS);
        // SAFETY: argument list matches dequant_block_i8_rows, and `output`
        // holds `rows * feature_dim` f32 (checked above).
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(&src.codes)
                .arg(&src.scales)
                .arg(&mut self.output)
                .arg(&feature_dim)
                .arg(&rows)
                .launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Decoded device output, `rows * feature_dim` f32.
    pub fn output(&self) -> &CudaSlice<f32> {
        &self.output
    }
}
