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
pub use features::{FeatureHeader, parse_feature_header};
#[cfg(feature = "gds")]
pub use features::{GdsFeatureStore, GdsReadResult, gds_driver_close, gds_driver_open};
#[cfg(all(target_os = "linux", feature = "shm"))]
pub use features::{ShareHandle, SharedFeatureStore};
pub use features::{
    create_features, save_feature_data, save_features, save_features_bf16, save_features_f16,
    save_features_ndarray,
};

// Graph (de)serialization entry points — public API surface; the `internal`
// module path is an implementation detail.
pub use internal::compressed_graph::{load_graph_compressed, save_graph_compressed};
pub use internal::mmap::{
    load_graph, load_graph_mmap, load_graph_owned, load_graph_with_validation, save_graph,
};
pub use internal::mmap_hetero::{load_hetero_graph, save_hetero_graph};
#[cfg(feature = "parquet")]
pub use internal::parquet_import::{from_parquet, from_parquet_files};
#[cfg(all(target_os = "linux", feature = "perf"))]
pub use internal::perf::{Counter, CounterReadings, CounterSet};
/// Counter-based RNG shared by CPU sampling oracles and GPU kernels.
pub use internal::philox::{philox_u32, philox4x32_10, reservoir_sample};
/// Portable predicate shared by host and device seqlock readers.
pub use internal::seqlock::cpu_seqlock_accept;
#[cfg(all(target_os = "linux", feature = "shm"))]
pub use internal::shm::{SharedRegion, recv_fd, send_fd, socket_pair};
// Vectorized bf16 → f32 conversion with runtime SIMD dispatch (AVX2, scalar
// fallback). The f16 path is reached through `FeatureDtype::row_decoder`,
// which resolves its dispatch once per batch rather than per call.
/// KERNELS.md Tier B / K4 device-path types (builders + loaders).
pub use internal::device::{
    BamController, CoherentAllocation, CoherentPlacementHint, CxlNumaBinding, DamonConfig,
    DamonSysfs, DevxGpuEthPlan, FdpPlacementId, FlexIoHost, IbgdaQueue, Mlx5RdmaReadWqe, NvmeRwSqe,
    P2pdmaPath, P2pdmaPolicy, SchedExtLoader, SchedExtPolicy, ZoneAppendWal, validate_p2pdma_path,
};
pub use internal::simd::bf16_le_to_f32;
pub use internal::succinct::{EliasFano, StreamVByte};
pub use internal::telemetry::{SamplingTelemetry, TelemetrySummary};
#[cfg(all(target_os = "linux", feature = "uffd"))]
pub use internal::uffd::{
    FileSource, PageSource, PageWeights, PagedRegion, page_size as uffd_page_size,
};
