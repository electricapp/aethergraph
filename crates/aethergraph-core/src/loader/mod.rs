//! Neighborhood sampling and data loading for GNN training.
//!
//! This module provides `NeighborLoader` - the main interface for sampling
//! k-hop neighborhoods from a graph with prefetching and optional feature loading.

pub mod hetero_sampler;
mod prefetch;
mod sampler;

pub use hetero_sampler::{HeteroNeighborSampler, HeteroSampledSubgraph, HeteroSamplingConfig};
pub use prefetch::{
    NeighborLoader, PrefetchResult, PrefetchStats, PrefetchWork, SubmitError, SyncFeatureStore,
};
pub use sampler::{
    NeighborSampler, ParallelBatchSampler, SampledSubgraph, SamplingConfig, SubgraphType,
    TemporalSamplingError, TemporalStrategy,
};
