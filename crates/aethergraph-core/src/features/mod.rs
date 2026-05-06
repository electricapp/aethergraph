//! Feature storage for node embeddings.
//!
//! Provides memory-mapped and async feature stores for billion-scale
//! node feature access.

mod async_store;
mod cache;
#[cfg(feature = "gds")]
mod gds;
pub(crate) mod header;
mod store;

use crate::graph::NodeId;

pub use async_store::AsyncFeatureStore;
pub use cache::{CacheStats, FeatureCache, FeatureCacheConfig, count_node_frequencies};
#[cfg(feature = "gds")]
pub use gds::{GdsFeatureStore, GdsReadResult, gds_driver_close, gds_driver_open};
pub use header::FeatureDtype;
pub use store::{
    FeatureData, FeatureLoadTelemetry, FeatureStore, create_features, save_feature_data,
    save_features, save_features_f16, save_features_ndarray,
};

/// Trait for reading node features from any backing store.
///
/// Implemented by `FeatureStore` (mmap'd files) and by `FeatureTable`
/// (seqlock in HugePage RAM) in `aether-stream`.
pub trait NodeFeatureSource: Send + Sync {
    /// Number of f32 features per node.
    fn feature_dim(&self) -> usize;

    /// Read features for `node` into `out`. Returns `true` if the node
    /// exists and was read successfully.
    fn read_node(&self, node: NodeId, out: &mut [f32]) -> bool;
}
