//! GPU-side infrastructure for GPUDirect RDMA feature gathering.
//!
//! Device kernels live under [`kernels`] (KERNELS.md Tier A). Orchestration
//! primitives that drive the GPU without writing kernels: [`uvm`], [`vmm`],
//! [`ipc`], [`gdrcopy`]. [`pool`] composes those into the gather allocation.

pub mod buffer;
#[cfg(feature = "gdrcopy")]
pub mod gdrcopy;
pub mod ipc;
pub mod kernels;
pub mod pool;
pub mod uvm;
pub mod vmm;

pub use kernels::SeqlockValidator;
pub use pool::FeaturePool;

/// Compatibility re-exports for call sites that still use the pre-reorg paths.
pub mod kernel {
    pub use super::kernels::validate::SeqlockValidator;
}
pub mod seqlock_reader {
    pub use super::kernels::seqlock::*;
}
pub mod sampler {
    pub use super::kernels::sampler::*;
}
pub mod decompress {
    pub use super::kernels::decompress::*;
}
pub mod persistent {
    pub use super::kernels::persistent::*;
}
pub mod tma_mma {
    pub use super::kernels::tma::*;
}
