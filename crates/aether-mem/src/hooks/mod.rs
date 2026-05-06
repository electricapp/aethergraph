//! Pluggable memory hooks for registering allocations with external systems.

mod mlock;

#[cfg(feature = "rdma")]
pub mod rdma;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use mlock::MlockHook;
