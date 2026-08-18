//! AetherStream — Kernel-bypass streaming engine for aethergraph.
//!
//! Provides AF_XDP ingestion, a seqlock-protected feature table in HugePage RAM,
//! and an RDMA control plane for one-sided reads from GPU nodes.
//!
//! This entire crate is Linux-only. On other platforms it compiles to an empty
//! crate with no dependencies.

#![cfg(target_os = "linux")]

pub mod feature_table;
pub mod ingest;
pub mod span;
pub mod umem;
pub mod xdp;

#[cfg(feature = "rdma")]
pub mod rdma;

#[cfg(feature = "gpudirect")]
pub mod gpu;
