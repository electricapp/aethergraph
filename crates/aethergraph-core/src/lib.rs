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

// Primary public exports
pub use graph::{
    partition_aligned_batches, AsyncCsrGraph, EdgeOffset, EdgeTypeId, EdgeTypeMeta, Graph,
    GraphStats, GraphValidationMode, HeteroGraph, NodeId, NodeTypeId, NodeTypeMeta,
};
pub use loader::{
    HeteroNeighborSampler, HeteroSampledSubgraph, HeteroSamplingConfig, NeighborLoader,
    NeighborSampler, ParallelBatchSampler, PrefetchStats, SampledSubgraph, SamplingConfig,
    SubgraphType, SyncFeatureStore, TemporalStrategy,
};

// Feature store exports (needed for file-backed features)
pub use features::{
    create_features, save_feature_data, save_features, save_features_f16, save_features_ndarray,
};
pub use features::{
    AsyncFeatureStore, FeatureData, FeatureDtype, FeatureLoadTelemetry, FeatureStore,
    NodeFeatureSource,
};
#[cfg(feature = "gds")]
pub use features::{gds_driver_close, gds_driver_open, GdsFeatureStore, GdsReadResult};

// Internal utilities (re-exported for backward compatibility, will be removed)
pub use internal::mmap::{
    load_graph, load_graph_mmap, load_graph_owned, load_graph_with_validation, save_graph,
};
pub use internal::mmap_hetero::{load_hetero_graph, save_hetero_graph};
#[cfg(feature = "parquet")]
pub use internal::parquet_import::{from_parquet, from_parquet_files};
pub use internal::telemetry::{SamplingTelemetry, TelemetrySummary};
