//! RDMA one-sided read support.
//!
//! Provides ibverbs FFI bindings, device context management, QP lifecycle,
//! a TCP control plane for advertisement + QP exchange, and a high-level
//! GPUDirect RDMA feature gather client.

#[cfg(feature = "rdma")]
pub mod ffi;

#[cfg(feature = "rdma")]
pub mod context;

#[cfg(feature = "rdma")]
pub mod qp;

#[cfg(feature = "rdma")]
pub mod control;

#[cfg(feature = "rdma")]
pub mod sharded;

#[cfg(feature = "efa")]
pub mod efa_ffi;

#[cfg(feature = "efa")]
pub mod srd;

#[cfg(feature = "gpudirect")]
pub mod client;

#[cfg(feature = "gpudirect")]
pub mod gather;
