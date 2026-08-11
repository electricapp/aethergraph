//! AetherGraph Core - High-performance graph storage for GNN training
//!
//! This library provides zero-copy, memory-mapped graph storage optimized for
//! neighborhood sampling in Graph Neural Networks at billion-scale.
//!
//! # Public API
//!
//! The public API is intentionally minimal:
//!
//! - [`Graph`] - Load and query graphs
//! - [`NeighborLoader`] - Sample k-hop neighborhoods with prefetching
//!
//! Everything else is internal implementation detail.

pub mod features;
pub mod graph;
pub(crate) mod internal;
pub mod loader;
pub mod metrics;

/// Re-exported epoch primitives. Lives in `aether-epoch` so both
/// `aether-graph` (dynamic writer) and this crate (feature store, samplers)
/// can share the same monotonic version clock without a circular dep.
pub use aether_epoch::{Epoch, EpochClock};

// Primary public exports
pub use graph::{
    AsyncCsrGraph, CsrView, EdgeOffset, EdgeTypeId, EdgeTypeMeta, Graph, GraphStats,
    GraphValidationMode, HeteroGraph, NodeId, NodeTypeId, NodeTypeMeta, partition_aligned_batches,
};
pub use loader::{
    HeteroNeighborLoader, HeteroNeighborSampler, HeteroSampledSubgraph, HeteroSamplingConfig,
    LocalEdgeIndex, NeighborLoader, NeighborSampler, ParallelBatchSampler, PrefetchError,
    PrefetchStats, SampledSubgraph, SamplingConfig, SubgraphType, SyncFeatureStore,
    TemporalSamplingError, TemporalStrategy,
};

// Feature store exports (needed for file-backed features)
pub use features::{
    AsyncFeatureStore, FeatureData, FeatureDtype, FeatureLoadTelemetry, FeatureStore,
    NodeFeatureSource,
};
#[cfg(feature = "gds")]
pub use features::{GdsFeatureStore, GdsReadResult, gds_driver_close, gds_driver_open};
pub use features::{
    create_features, save_feature_data, save_features, save_features_f16, save_features_ndarray,
};

// Graph (de)serialization entry points — public API surface; the `internal`
// module path is an implementation detail.
pub use internal::mmap::{
    load_graph, load_graph_mmap, load_graph_owned, load_graph_with_validation, save_graph,
};
pub use internal::mmap_hetero::{load_hetero_graph, save_hetero_graph};
#[cfg(feature = "parquet")]
pub use internal::parquet_import::{from_parquet, from_parquet_files};
// Vectorized half-precision → f32 conversions with runtime SIMD dispatch
// (AVX-512 / F16C / NEON for f16, AVX2 for bf16, scalar fallback).
pub use internal::simd::{bf16_le_to_f32, f16_le_to_f32};
pub use internal::telemetry::{SamplingTelemetry, TelemetrySummary};
