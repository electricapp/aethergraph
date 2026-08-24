//! KERNELS.md Tier A device kernels (NVRTC `include_str!`) plus Tier B GPU
//! helpers (IBGDA post, coherent advise).
//!
//! Layout:
//! - [`validate`] — K5.0 graphs + K5.6 `ld.cs` compaction
//! - [`seqlock`] — K5.3 PTX acquire reader
//! - [`sampler`] — K5.2 warp reservoir
//! - [`decompress`] — K5.5 StreamVByte / Elias-Fano
//! - [`persistent`] — K5.1 work-ring drain
//! - [`tma`] — K5.4 dense aggregation
//! - [`ibgda`] — K2.1 GPU WQE poster
//! - [`coherent`] — K3.3 placement hints
//! - [`harness`] — skip-friendly CUDA helpers for tests/benches

pub mod coherent;
pub mod decompress;
pub mod harness;
pub mod ibgda;
pub mod persistent;
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
pub use sampler::{WarpSampler, philox};
pub use seqlock::{SeqlockSnapshotReader, cpu_seqlock_accept};
pub use tma::{TensorTileShape, TensorTileStage, TmaAggregator};
pub use validate::SeqlockValidator;
