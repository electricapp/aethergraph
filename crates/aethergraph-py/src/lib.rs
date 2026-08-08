// AetherGraph Python Bindings
//
// High-performance Python bindings for AetherGraph using PyO3.
// Enables zero-copy graph sampling and feature caching for GNN training.

#![allow(non_local_definitions)] // PyO3 macros generate non-local definitions

// The batch loop allocates steadily (per-batch subgraph buffers, i64
// widenings, feature vectors); mimalloc's sharded thread-local heaps beat
// the system allocator on exactly that churn, and as the Rust-side
// `#[global_allocator]` it never touches Python's own allocation paths.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod arrow_utils;
mod async_feature_store;
#[cfg(feature = "gpudirect")]
mod dlpack;
mod dynamic_graph;
mod error;
mod feature_cache;
mod feature_store;
mod feature_telemetry;
mod graph;
mod hetero;
mod metrics;
mod prefetch;
mod sampler;

use pyo3::prelude::*;

use dynamic_graph::PyDynamicGraph;
use error::{ArrowConversionError, CacheError, GraphLoadError, SamplingError};
use feature_cache::{PyFeatureCache, PyFeatureCacheConfig};
use graph::PyCsrGraph;
use hetero::{
    PyHeteroCsrGraph, PyHeteroNeighborSampler, PyHeteroSampledSubgraph, PyHeteroSamplingConfig,
};
use metrics::PyMetricsSnapshot;
use prefetch::{PyNeighborLoader, PyPrefetchStats};
use sampler::{
    PyNeighborSampler, PyParallelBatchSampler, PySampledSubgraph, PySamplingConfig,
    PySamplingTelemetry,
};

/// AetherGraph: High-performance graph sampling for GNN training.
///
/// This module provides Python bindings to AetherGraph's Rust-based graph engine,
/// enabling billion-scale graph neural network training with:
/// - Zero-copy data access via Apache Arrow
/// - Memory-mapped graph storage
/// - SIMD-accelerated neighborhood sampling
/// - GPU/CPU/NVMe tiered feature caching
///
/// Example:
///     >>> import aethergraph_core as ag
///     >>> graph = ag.CsrGraph.load("graph.bin")
///     >>> config = ag.SamplingConfig(num_neighbors=[25, 10], replace=False)
///     >>> sampler = ag.NeighborSampler(graph, config)
///     >>> subgraph = sampler.sample([0, 1, 2])
///     >>> print(f"Sampled {subgraph.num_nodes()} nodes")
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    // Register classes
    m.add_class::<PyCsrGraph>()?;
    m.add_class::<PyDynamicGraph>()?;
    m.add_class::<PySamplingConfig>()?;
    m.add_class::<PySampledSubgraph>()?;
    m.add_class::<PySamplingTelemetry>()?;
    m.add_class::<PyNeighborSampler>()?;
    m.add_class::<PyParallelBatchSampler>()?;
    m.add_class::<PyFeatureCacheConfig>()?;
    m.add_class::<PyFeatureCache>()?;
    m.add_class::<PyNeighborLoader>()?;
    m.add_class::<PyPrefetchStats>()?;
    m.add_class::<PyMetricsSnapshot>()?;

    // Heterogeneous graph classes
    m.add_class::<PyHeteroCsrGraph>()?;
    m.add_class::<PyHeteroSamplingConfig>()?;
    m.add_class::<PyHeteroSampledSubgraph>()?;
    m.add_class::<PyHeteroNeighborSampler>()?;

    // Register feature store
    feature_store::register_feature_store(py, m)?;
    async_feature_store::register_async_feature_store(py, m)?;

    // Register custom exceptions
    m.add("GraphLoadError", py.get_type::<GraphLoadError>())?;
    m.add("SamplingError", py.get_type::<SamplingError>())?;
    m.add("CacheError", py.get_type::<CacheError>())?;
    m.add(
        "ArrowConversionError",
        py.get_type::<ArrowConversionError>(),
    )?;

    // Module metadata — baked in at compile time from Cargo.toml. Updating
    // `version` or `authors` in Cargo.toml requires a rebuild to take effect.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", env!("CARGO_PKG_AUTHORS"))?;

    // Compile-time Cargo feature flags exposed as data. Python checks these
    // directly instead of probing for method existence (`hasattr`), keeping
    // capability detection out of the structural-typing world.
    m.add("HAS_GPUDIRECT", cfg!(feature = "gpudirect"))?;

    #[cfg(feature = "gpudirect")]
    m.add_function(wrap_pyfunction!(
        dlpack::dlpack_capsule_from_cuda_ptr_py,
        m
    )?)?;

    Ok(())
}
