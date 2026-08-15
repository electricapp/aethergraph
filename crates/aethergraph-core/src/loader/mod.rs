//! Neighborhood sampling and data loading for GNN training.
//!
//! This module provides `NeighborLoader` - the main interface for sampling
//! k-hop neighborhoods from a graph with prefetching and optional feature loading.

pub mod hetero_sampler;
mod prefetch;
mod sampler;

/// Capacity to reserve for a per-call output buffer that last held `produced`
/// elements, never below `floor`.
///
/// Both samplers hand their filled output buffers to the caller and swap empty
/// ones back in, so every call starts those arrays from scratch. The count the
/// previous call produced is the best available estimate for the next one, but
/// sampling is randomized: consecutive batches of the same shape drift by a few
/// percent. Reserving an exact fit would leave roughly half of all calls one
/// element short, and falling one element short costs a full-array
/// reallocation and copy. An eighth of headroom absorbs that drift; the excess
/// rides along in the returned subgraph, which is consumed and dropped before
/// the next batch.
///
/// A `floor` of zero keeps `produced == 0` at zero, which `Vec` serves without
/// allocating — that is what lets a wide heterogeneous schema pay nothing for
/// the types a batch never visits.
#[inline]
pub(crate) fn planned_capacity(produced: usize, floor: usize) -> usize {
    (produced + produced / 8).max(floor)
}

pub use hetero_sampler::{HeteroNeighborSampler, HeteroSampledSubgraph, HeteroSamplingConfig};
pub use prefetch::{
    HeteroNeighborLoader, HeteroPrefetchResult, NeighborLoader, PrefetchError, PrefetchResult,
    PrefetchStats, PrefetchWork, SubgraphWithFeatures, SubmitError, SyncFeatureStore,
};
pub use sampler::{
    LocalEdgeIndex, NeighborSampler, ParallelBatchSampler, SampledSubgraph, SamplingConfig,
    SubgraphType, TemporalSamplingError, TemporalStrategy,
};
