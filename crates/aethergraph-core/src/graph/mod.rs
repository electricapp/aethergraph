//! Graph storage and representation.
//!
//! This module provides the core `Graph` type - a CSR (Compressed Sparse Row)
//! representation optimized for GNN neighborhood sampling.

mod async_graph;
mod csr;
pub mod hetero;
mod reorder;

pub use async_graph::AsyncCsrGraph;
pub use csr::{EdgeOffset, Graph, GraphStats, GraphValidationMode, NodeId};
pub use hetero::{EdgeTypeId, EdgeTypeMeta, HeteroGraph, NodeTypeId, NodeTypeMeta};
pub use reorder::partition_aligned_batches;
