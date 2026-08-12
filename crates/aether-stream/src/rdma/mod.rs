//! RDMA one-sided read support.
//!
//! Provides ibverbs FFI bindings, device context management, QP lifecycle,
//! a TCP control plane for advertisement + QP exchange, and a high-level
//! GPUDirect RDMA feature gather client.
//!
//! # Verbs implemented ahead of their consumer
//!
//! The gather path this crate serves today is **pull-mode and read-only**:
//! a trainer issues RDMA READs against a server's registered feature
//! table. Several verbs here are implemented and covered by
//! `tests/softroce_e2e.rs` but have no product caller, because the thing
//! that would call them is a feature-server *write* path that does not
//! exist in this tree:
//!
//! - [`qp::RdmaQp::post_fetch_add`] / [`qp::RdmaQp::post_compare_swap`] —
//!   remote cursor reservation and one-sided seqlock write claims.
//! - [`qp::RdmaQp::post_writes`] (WRITE_WITH_IMM) plus its receive half,
//!   [`qp::RdmaQp::post_recv_sentinels`] — push-mode publication.
//! - [`srq`] and [`qp::RdmaQp::create_with_cqs_srq`] — one receive queue
//!   shared across many client QPs, which pays off server-side.
//! - [`context::RdmaContext::reg_mr_implicit_odp`] and
//!   [`context::RdmaContext::odp_caps`] — registration without pinning.
//!
//! They are kept rather than deleted because each is a load-bearing piece
//! of a design already sketched (see the roadmap), the FFI and safety work
//! is the hard part and is done, and the e2e tests pin the ABI against
//! SoftRoCE. Wire them when the server side is designed — not by inventing
//! a caller to make them look used. Each carries a `TODO(deferred)` at its
//! definition naming what it waits on.

#[cfg(feature = "rdma")]
pub mod ffi;

#[cfg(feature = "rdma")]
pub mod context;

#[cfg(feature = "rdma")]
pub mod qp;

#[cfg(feature = "rdma")]
pub mod control;

#[cfg(feature = "rdma")]
pub mod event;

#[cfg(feature = "rdma")]
pub mod sharded;

#[cfg(feature = "rdma")]
pub mod srq;

#[cfg(feature = "efa")]
pub mod efa_ffi;

#[cfg(feature = "efa")]
pub mod srd;

#[cfg(feature = "gpudirect")]
pub mod client;

#[cfg(feature = "gpudirect")]
pub mod gather;
