//! KERNELS.md Tier A device kernels (NVRTC `include_str!`) plus Tier B GPU
//! helpers (IBGDA post, coherent advise).
//!
//! Layout:
//! - [`validate`] — K5.0 graphs + K5.6 `ld.cs` compaction
//! - [`seqlock`] — K5.3 PTX acquire reader
//! - [`sampler`] — K5.2 warp reservoir
//! - [`decompress`] — K5.5 StreamVByte / Elias-Fano (topology)
//! - [`quant`] — K5.5 block-scaled int8 feature rows (payload)
//! - [`persistent`] — K5.1 work-ring drain
//! - [`tma`] — K5.4 dense aggregation
//! - [`ibgda`] — K2.1 GPU WQE poster
//! - [`coherent`] — K3.3 placement hints
//! - [`harness`] — skip-friendly CUDA helpers for tests/benches

use cudarc::driver::CudaContext;
use cudarc::nvrtc::{CompileOptions, Ptx};
use std::sync::Arc;

pub mod coherent;
pub mod decompress;
pub mod harness;
pub mod ibgda;
pub mod persistent;
pub mod quant;
pub mod sampler;
pub mod seqlock;
pub mod tma;
pub mod validate;

pub use coherent::apply_coherent_placement;
pub use decompress::{
    EliasFanoDecoder, EliasFanoDeviceParts, StreamVByteDecoder, cpu_streamvbyte_delta_decode,
};
pub use ibgda::IbgdaPoster;
pub use persistent::{PersistentWork, PersistentWorkKind, PersistentWorker};
pub use quant::{FeatureDequantizer, QuantizedRowsDevice};
pub use sampler::{SampleRequest, WarpSampler, philox};
pub use seqlock::{SeqlockSnapshotReader, cpu_seqlock_accept};
pub use tma::{TensorTileShape, TensorTileStage, TmaAggregator};
pub use validate::SeqlockValidator;

/// Lowest virtual architecture every unit here compiles for; set by
/// `__nanosleep` in `persistent.cu` and `ld.acquire.sys` in
/// `seqlock_reader.cu`. Device-free tests compile against it.
pub const NVRTC_ARCH_FLOOR: &str = "compute_70";

/// Compile a unit for the architecture of the device behind `ctx`.
///
/// `cudarc::nvrtc::compile_ptx` sends no `--gpu-architecture`, so NVRTC
/// targets its own default — a floor below any supported GPU, which rejects
/// newer instructions and caps the rest. Naming the device's `compute_XY` is
/// what makes a runtime compile worth its cost.
pub fn compile_for_device(
    ctx: &Arc<CudaContext>,
    src: &str,
) -> Result<Ptx, Box<dyn std::error::Error>> {
    use cudarc::driver::sys::CUdevice_attribute as Attr;
    let major = ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    Ok(cudarc::nvrtc::compile_ptx_with_opts(
        src,
        CompileOptions {
            options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
            ..Default::default()
        },
    )?)
}

/// Every NVRTC translation unit in this module, as `(name, source)`.
///
/// `cargo` never reads a `.cu` file, so `nvrtc_compiles_every_unit` and
/// `ptxas_assembles_every_unit` gate them instead. Both need the toolkit but
/// no device.
#[must_use]
pub fn nvrtc_units() -> &'static [(&'static str, &'static str)] {
    &[
        ("validate", validate::KERNEL_SRC),
        ("seqlock", seqlock::KERNEL_SRC),
        ("sampler", sampler::KERNEL_SRC),
        ("decompress", decompress::KERNEL_SRC),
        ("quant", quant::KERNEL_SRC),
        ("persistent", persistent::KERNEL_SRC),
        ("tma", tma::KERNEL_SRC),
        ("ibgda", ibgda::KERNEL_SRC),
    ]
}
