//! PyO3 bindings for the prefetching sampler.

use aethergraph_core::Graph;
use aethergraph_core::{NeighborLoader, PrefetchError};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::sampling_error;
use crate::graph::PyCsrGraph;
use crate::sampler::{PreparedSubgraph, PySampledSubgraph, PySamplingConfig};

/// Return type of [`PyNeighborLoader::next_with_features`].
///
/// `Some((subgraph, Some(features)))` — both subgraph and features available.
/// `Some((subgraph, None))` — loader has no feature column attached.
/// `None` — the loader was shut down / drained and there are no more batches.
pub type NextWithFeaturesResult<'py> =
    Option<(PySampledSubgraph, Option<Bound<'py, PyArray2<f32>>>)>;

/// Return type of [`PyNeighborLoader::next_batch`]: the batch's `batch_idx`
/// ahead of the [`NextWithFeaturesResult`] pair.
pub type NextBatchResult<'py> =
    Option<(usize, PySampledSubgraph, Option<Bound<'py, PyArray2<f32>>>)>;

/// Maps a core [`PrefetchError`] onto the closest Python exception type.
///
/// - `Timeout` → `TimeoutError` carrying the waited duration; the worker may
///   just be slow, so the caller may retry the call.
/// - `WorkerExited` → `RuntimeError` including the captured panic/error
///   message when one is available.
/// - `FeatureLoad` → `RuntimeError` naming the batch whose features failed.
pub(crate) fn prefetch_error_to_py(err: PrefetchError) -> PyErr {
    match err {
        PrefetchError::Timeout { waited } => pyo3::exceptions::PyTimeoutError::new_err(format!(
            "prefetch timed out after {waited:?}; the worker may be slow or deadlocked \
             — the call may be retried"
        )),
        PrefetchError::WorkerExited {
            message: Some(message),
        } => {
            pyo3::exceptions::PyRuntimeError::new_err(format!("prefetch worker exited: {message}"))
        }
        PrefetchError::WorkerExited { message: None } => {
            pyo3::exceptions::PyRuntimeError::new_err("prefetch worker exited unexpectedly")
        }
        PrefetchError::FeatureLoad { batch_idx, source } => {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "feature load failed for batch {batch_idx}: {source:#}"
            ))
        }
    }
}

/// Python wrapper for PrefetchStats.
///
/// Snapshot of the loader's atomic counters at the moment `.stats()` was
/// called. The Rust [`PrefetchStats`] keeps the live atomic state; we copy
/// it out so callers see a stable view.
#[pyclass(name = "PrefetchStats", frozen)]
pub struct PyPrefetchStats {
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) total: u64,
    pub(crate) sample_time_ns: u64,
    pub(crate) feature_load_time_ns: u64,
}

impl PyPrefetchStats {
    pub(crate) fn snapshot(stats: &aethergraph_core::PrefetchStats) -> Self {
        use std::sync::atomic::Ordering::Relaxed;
        Self {
            hits: stats.hits.load(Relaxed),
            misses: stats.misses.load(Relaxed),
            total: stats.total.load(Relaxed),
            sample_time_ns: stats.sample_time_ns.load(Relaxed),
            feature_load_time_ns: stats.feature_load_time_ns.load(Relaxed),
        }
    }
}

#[pymethods]
impl PyPrefetchStats {
    /// Number of batches delivered without waiting.
    #[getter]
    fn hits(&self) -> u64 {
        self.hits
    }

    /// Number of batches the consumer had to wait for.
    #[getter]
    fn misses(&self) -> u64 {
        self.misses
    }

    /// Total batches delivered.
    #[getter]
    fn total(&self) -> u64 {
        self.total
    }

    /// Hit rate (0.0 to 1.0).
    #[getter]
    fn hit_rate(&self) -> f64 {
        if self.total == 0 {
            1.0
        } else {
            self.hits as f64 / self.total as f64
        }
    }

    /// Cumulative nanoseconds the prefetch worker spent sampling.
    #[getter]
    fn sample_time_ns(&self) -> u64 {
        self.sample_time_ns
    }

    /// Cumulative nanoseconds the prefetch worker spent loading features.
    #[getter]
    fn feature_load_time_ns(&self) -> u64 {
        self.feature_load_time_ns
    }

    fn __repr__(&self) -> String {
        format!(
            "PrefetchStats(hits={}, misses={}, hit_rate={:.1}%)",
            self.hits,
            self.misses,
            self.hit_rate() * 100.0
        )
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("hits", self.hits)?;
        dict.set_item("misses", self.misses)?;
        dict.set_item("total", self.total)?;
        dict.set_item("hit_rate", self.hit_rate())?;
        dict.set_item("sample_time_ns", self.sample_time_ns)?;
        dict.set_item("feature_load_time_ns", self.feature_load_time_ns)?;
        Ok(dict.into())
    }
}

/// Converts a delivered subgraph for Python. Touches no Python state, so
/// callers run it with the GIL released and only wrap the result after.
fn prepare(subgraph: aethergraph_core::SampledSubgraph) -> PyResult<PreparedSubgraph> {
    PreparedSubgraph::new(subgraph).map_err(sampling_error)
}

/// Reshape a flat feature vector into `(num_nodes, feature_dim)` without a copy.
fn features_array(
    py: Python<'_>,
    features: Vec<f32>,
    num_nodes: usize,
    feature_dim: usize,
) -> PyResult<Bound<'_, PyArray2<f32>>> {
    PyArray1::from_vec(py, features)
        .reshape([num_nodes, feature_dim])
        .map_err(|e| sampling_error(format!("Failed to reshape features: {e}")))
}

/// Prefetching neighbor sampler for pipelined GNN training.
///
/// Spawns a dedicated thread that samples batches ahead of time,
/// ensuring the training loop never waits for the sampler.
///
/// On Linux with NVMe storage, uses io_uring with SQPOLL for
/// zero-syscall I/O. Can also load features alongside sampling.
///
/// Every method is safe to call from any thread at any time: `shutdown()`
/// (or leaving a `with` block) from one thread wakes others blocked in
/// `submit()` or `next()`.
///
/// Args:
///     graph: Graph to sample from
///     config: SamplingConfig with num_neighbors, replace, seed parameters
///     prefetch_depth: Number of batches to keep ready (default: 2)
///
/// Example (sampling only):
///     >>> prefetcher = NeighborLoader(graph, config, prefetch_depth=3)
///     >>> for i, batch in enumerate(batches):
///     ...     prefetcher.submit(i, batch)
///     >>> for _ in range(len(batches)):
///     ...     subgraph = prefetcher.next()  # Already ready!
///     ...     train(subgraph)
///
/// Example (sampling + features):
///     >>> prefetcher = NeighborLoader.with_features(graph, config, "features.bin", prefetch_depth=3)
///     >>> for i, batch in enumerate(batches):
///     ...     prefetcher.submit(i, batch)
///     >>> for _ in range(len(batches)):
///     ...     subgraph, features = prefetcher.next_with_features()  # Both ready!
///     ...     train(subgraph, features)
#[pyclass(name = "NeighborLoader", frozen)]
pub struct PyNeighborLoader {
    inner: NeighborLoader,
    // Keep graph alive for the lifetime of the sampler
    _graph: Arc<Graph>,
    // Feature dimension (if loading features)
    feature_dim: Option<usize>,
    // RDMA feature gather (gpudirect path); gathers need exclusive access.
    #[cfg(all(target_os = "linux", feature = "gpudirect"))]
    rdma_gather: Option<parking_lot::Mutex<aether_stream::rdma::gather::RdmaFeatureGather>>,
}

impl PyNeighborLoader {
    fn wrap(inner: NeighborLoader, graph: Arc<Graph>) -> Self {
        Self {
            feature_dim: inner.feature_dim(),
            inner,
            _graph: graph,
            #[cfg(all(target_os = "linux", feature = "gpudirect"))]
            rdma_gather: None,
        }
    }
}

#[pymethods]
impl PyNeighborLoader {
    /// Create a new prefetching sampler (sampling only, no features).
    ///
    /// Args:
    ///     graph: Graph to sample from
    ///     config: SamplingConfig for sampling parameters. With
    ///         `disjoint=True` every result carries its `batch` vector.
    ///     prefetch_depth: Number of batches to prefetch ahead (default: 2)
    ///     sampler_threads: Sampler worker threads pulling from the shared
    ///         work queue (default: 1). With a config seed, results come
    ///         back in submission order; without one, as workers finish.
    #[new]
    #[pyo3(signature = (graph, config, prefetch_depth=2, sampler_threads=1))]
    fn new(
        graph: &PyCsrGraph,
        config: &PySamplingConfig,
        prefetch_depth: usize,
        sampler_threads: usize,
    ) -> PyResult<Self> {
        // Share the same graph backing across Python and prefetch threads.
        let graph_arc = graph.inner_arc();

        let inner = NeighborLoader::new(
            graph_arc.clone(),
            config.inner().clone(),
            prefetch_depth,
            sampler_threads,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create NeighborLoader: {e}"
            ))
        })?;

        Ok(Self::wrap(inner, graph_arc))
    }

    /// Create a prefetching sampler that also loads features.
    ///
    /// After sampling the subgraph, the pipeline also loads features for
    /// all nodes. On Linux, uses io_uring for parallel reads.
    ///
    /// Args:
    ///     graph: Graph to sample from
    ///     config: SamplingConfig for sampling parameters
    ///     feature_path: Path to feature file (AETHFEAT format)
    ///     prefetch_depth: Number of batches to prefetch ahead (default: 2)
    ///     sampler_threads: Sampler worker threads feeding the feature
    ///         loader (default: 1)
    ///
    /// Returns:
    ///     NeighborLoader configured to load features
    #[staticmethod]
    #[pyo3(signature = (graph, config, feature_path, prefetch_depth=2, sampler_threads=1))]
    fn with_features(
        graph: &PyCsrGraph,
        config: &PySamplingConfig,
        feature_path: PathBuf,
        prefetch_depth: usize,
        sampler_threads: usize,
    ) -> PyResult<Self> {
        let graph_arc = graph.inner_arc();

        let inner = NeighborLoader::with_features(
            graph_arc.clone(),
            config.inner().clone(),
            &feature_path,
            prefetch_depth,
            sampler_threads,
        )
        .map_err(|e| sampling_error(format!("Failed to create prefetcher with features: {e}")))?;

        Ok(Self::wrap(inner, graph_arc))
    }

    /// Create a prefetching sampler for NVMe-backed graphs (Linux only).
    ///
    /// Uses io_uring with SQPOLL for graph topology reads: the header and
    /// offsets are read and validated here, and each batch reads only the
    /// neighbor entries it samples. Ideal for graphs that don't fit in RAM.
    ///
    /// The io_uring path supports plain uniform sampling only: configs that
    /// set weighted, temporal_strategy, disjoint, deterministic, max_degree,
    /// or a non-default subgraph_type are rejected.
    ///
    /// Args:
    ///     graph_path: Path to binary graph file (AETHGRAPH format)
    ///     config: SamplingConfig for sampling parameters
    ///     prefetch_depth: Number of batches to prefetch ahead (default: 2)
    ///
    /// Returns:
    ///     NeighborLoader that reads graph topology via io_uring
    ///
    /// Raises:
    ///     OSError: If not on Linux, io_uring is unavailable, the graph file
    ///         is invalid, or the config sets an option the io_uring path
    ///         does not support
    ///
    /// Example:
    ///     >>> # Linux only - for TB-scale graphs
    ///     >>> prefetcher = NeighborLoader.new_nvme("large_graph.bin", config, prefetch_depth=3)
    ///     >>> for i, batch in enumerate(batches):
    ///     ...     prefetcher.submit(i, batch)
    ///     >>> for _ in range(len(batches)):
    ///     ...     subgraph = prefetcher.next()  # Graph edges read via io_uring!
    #[staticmethod]
    #[pyo3(signature = (graph_path, config, prefetch_depth=2))]
    #[cfg(target_os = "linux")]
    fn new_nvme(
        py: Python<'_>,
        graph_path: PathBuf,
        config: &PySamplingConfig,
        prefetch_depth: usize,
    ) -> PyResult<Self> {
        let config = config.inner().clone();
        // Reading the offsets array can take a while on a large graph.
        let inner = py
            .detach(|| NeighborLoader::new_nvme(&graph_path, config, prefetch_depth))
            .map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(format!("Failed to open NVMe graph: {e}"))
            })?;

        // Placeholder: the topology is file-backed.
        Ok(Self::wrap(inner, Arc::new(Graph::empty())))
    }

    /// Create a prefetching sampler for NVMe-backed graphs (Linux only).
    ///
    /// This method is only available on Linux with io_uring support.
    /// On other platforms, raises OSError.
    #[staticmethod]
    #[pyo3(signature = (graph_path, config, prefetch_depth=2))]
    #[cfg(not(target_os = "linux"))]
    #[allow(unused_variables)]
    fn new_nvme(
        graph_path: PathBuf,
        config: &PySamplingConfig,
        prefetch_depth: usize,
    ) -> PyResult<Self> {
        Err(pyo3::exceptions::PyOSError::new_err(
            "NVMe graph sampling requires Linux with io_uring support. \
             On macOS/Windows, use in-memory graphs with NeighborLoader() instead.",
        ))
    }

    /// Submit a batch to be sampled.
    ///
    /// `batch_idx` is caller bookkeeping: it comes back with the result from
    /// `next_batch()`, and any value is accepted, repeats and gaps included.
    /// With a config seed, results come back in submission order — so one
    /// loader serves any number of epochs — and without one, in completion
    /// order across the worker pool.
    ///
    /// Blocks (with the GIL released) when the pipeline is full, until the
    /// consumer drains a result — so interleave `submit()` with `next()`, or
    /// submit from a separate thread. A `shutdown()` from another thread
    /// unblocks it.
    ///
    /// Args:
    ///     batch_idx: Caller-chosen index for this batch, echoed back on the
    ///         result for bookkeeping.
    ///     seeds: Seed node IDs (numpy uint32, numpy int64, or `list[int]`),
    ///         each below `num_nodes`.
    ///
    /// Raises:
    ///     SamplingError: A seed is out of range, or the loader was shut
    ///         down or a worker failed.
    fn submit(&self, py: Python<'_>, batch_idx: usize, seeds: &Bound<'_, PyAny>) -> PyResult<()> {
        let seeds = crate::error::extract_seed_batch(seeds, self.inner.num_nodes())?;

        // The bounded work channel blocks when the pipeline is full; release
        // the GIL so the consumer thread can drain. No Python object is
        // touched inside.
        py.detach(|| self.inner.submit(batch_idx, seeds))
            .map_err(|e| sampling_error(format!("Submit failed: {e}")))
    }

    /// Submit all batches for an epoch, indexed from 0.
    ///
    /// Each submission can block when the pipeline is full (see `submit()`),
    /// so call this from a separate thread unless a consumer is draining
    /// results concurrently.
    ///
    /// Args:
    ///     batches: List of seed arrays, one per batch
    fn submit_epoch(&self, py: Python<'_>, batches: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        for (idx, seeds) in batches.iter().enumerate() {
            self.submit(py, idx, seeds)?;
        }
        Ok(())
    }

    /// Get the next sampled subgraph (blocking).
    ///
    /// Returns:
    ///     SampledSubgraph: The next prefetched subgraph
    ///     None: Only after `shutdown()` — the pipeline was drained and no
    ///         more batches will arrive
    ///
    /// Raises:
    ///     TimeoutError: No result arrived within the wait window; the worker
    ///         may just be slow, so the call may be retried.
    ///     RuntimeError: A prefetch worker failed.
    fn next(&self, py: Python<'_>) -> PyResult<Option<PySampledSubgraph>> {
        // The blocking recv can stall while the worker samples, and the
        // widening is O(N + E); both run with the GIL released. No Python
        // object is touched inside.
        py.detach(|| {
            self.inner
                .next()
                .map_err(prefetch_error_to_py)?
                .map(prepare)
                .transpose()
        })?
        .map(|prepared| PySampledSubgraph::from_prepared(py, prepared))
        .transpose()
    }

    /// Try to get the next subgraph without blocking.
    ///
    /// Returns:
    ///     SampledSubgraph: If a batch is immediately available
    ///     None: If no batch is ready yet, or after `shutdown()`
    ///
    /// Raises:
    ///     RuntimeError: A prefetch worker failed.
    fn try_next(&self, py: Python<'_>) -> PyResult<Option<PySampledSubgraph>> {
        py.detach(|| {
            self.inner
                .try_next()
                .map_err(prefetch_error_to_py)?
                .map(prepare)
                .transpose()
        })?
        .map(|prepared| PySampledSubgraph::from_prepared(py, prepared))
        .transpose()
    }

    /// Get next subgraph with features (blocking).
    ///
    /// Use this when the prefetcher was created with `with_features()`.
    /// Returns both the subgraph and features as numpy arrays.
    ///
    /// Returns:
    ///     tuple: (SampledSubgraph, features) where features is a 2D numpy array
    ///            of shape (num_nodes, feature_dim), or None if no features loaded
    ///     None: Only after `shutdown()` — the pipeline was drained and no
    ///         more batches will arrive
    ///
    /// Raises:
    ///     TimeoutError: No result arrived within the wait window; the call
    ///         may be retried.
    ///     RuntimeError: A prefetch worker failed, or loading this batch's
    ///         features failed.
    fn next_with_features<'py>(&self, py: Python<'py>) -> PyResult<NextWithFeaturesResult<'py>> {
        Ok(self
            .next_batch(py)?
            .map(|(_, subgraph, features)| (subgraph, features)))
    }

    /// Get the next batch with its `batch_idx` and features (blocking).
    ///
    /// Returns:
    ///     tuple: (batch_idx, SampledSubgraph, features) — `batch_idx` as
    ///         passed to `submit()`, features as in `next_with_features()`
    ///     None: Only after `shutdown()`
    ///
    /// Raises:
    ///     TimeoutError: No result arrived within the wait window; the call
    ///         may be retried.
    ///     RuntimeError: A prefetch worker failed, or loading this batch's
    ///         features failed.
    fn next_batch<'py>(&self, py: Python<'py>) -> PyResult<NextBatchResult<'py>> {
        // The blocking recv and the O(N + E) widening run with the GIL
        // released; the numpy outputs only wrap the buffers afterward.
        let Some((batch_idx, prepared, features, num_nodes)) = py.detach(|| {
            let Some(batch) = self.inner.next_batch().map_err(prefetch_error_to_py)? else {
                return Ok(None);
            };
            let num_nodes = batch.subgraph.nodes.len();
            let prepared = prepare(batch.subgraph)?;
            PyResult::Ok(Some((batch.batch_idx, prepared, batch.features, num_nodes)))
        })?
        else {
            return Ok(None);
        };
        let subgraph = PySampledSubgraph::from_prepared(py, prepared)?;
        let features = match (features, self.feature_dim) {
            (Some(features), Some(dim)) => Some(features_array(py, features, num_nodes, dim)?),
            _ => None,
        };
        Ok(Some((batch_idx, subgraph, features)))
    }

    /// Create a prefetcher that gathers features via GPUDirect RDMA.
    ///
    /// Features land directly in VRAM — never touch CPU.
    /// Use `next_batch_gpu()` to get DLPack capsules that
    /// `torch.from_dlpack()` converts to CUDA tensors (zero-copy).
    ///
    /// Args:
    ///     graph: Graph to sample from
    ///     config: SamplingConfig for sampling parameters
    ///     server_addr: RDMA control plane address ("host:port")
    ///     gpu_id: CUDA device ordinal (default: 0)
    ///     max_batch_nodes: Upper bound on nodes per subgraph (default: 65536)
    ///     prefetch_depth: Number of batches to prefetch ahead (default: 2)
    ///     gid_index: Local GID-table index for RoCEv2 (default: 1, the
    ///         typical IPv4-mapped GID on Linux; verify with `show_gids`)
    ///     sampler_threads: Sampler threads feeding the pipeline (default: 1)
    #[staticmethod]
    #[pyo3(signature = (graph, config, server_addr, gpu_id=0, max_batch_nodes=65536, prefetch_depth=2, gid_index=1, sampler_threads=1))]
    #[cfg(all(target_os = "linux", feature = "gpudirect"))]
    #[allow(clippy::too_many_arguments)]
    fn with_rdma_features(
        graph: &PyCsrGraph,
        config: &PySamplingConfig,
        server_addr: &str,
        gpu_id: usize,
        max_batch_nodes: usize,
        prefetch_depth: usize,
        gid_index: u8,
        sampler_threads: usize,
    ) -> PyResult<Self> {
        let graph_arc = graph.inner_arc();

        let inner = NeighborLoader::new(
            graph_arc.clone(),
            config.inner().clone(),
            prefetch_depth,
            sampler_threads,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create NeighborLoader: {e}"
            ))
        })?;

        let rdma = aether_stream::rdma::gather::RdmaFeatureGather::connect(
            server_addr,
            gpu_id,
            max_batch_nodes,
            gid_index,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to connect RDMA feature gather: {e}"
            ))
        })?;

        let feature_dim = Some(rdma.feature_dim());

        Ok(Self {
            inner,
            _graph: graph_arc,
            feature_dim,
            rdma_gather: Some(parking_lot::Mutex::new(rdma)),
        })
    }

    /// Get the next batch with GPU features (RDMA path, blocking).
    ///
    /// Samples a subgraph, then gathers features via RDMA directly into VRAM.
    /// Returns (batch_idx, SampledSubgraph, PyCapsule) where the capsule is a
    /// DLPack managed tensor owning its VRAM — the buffer stays valid for the
    /// consuming tensor's whole lifetime, independent of later gathers or the
    /// loader itself. Use `torch.from_dlpack(capsule)` in Python.
    ///
    /// Returns None only after `shutdown()` — the pipeline was drained and no
    /// more batches will arrive.
    ///
    /// Raises:
    ///     TimeoutError: No result arrived within the wait window; the call
    ///         may be retried.
    ///     RuntimeError: A prefetch worker failed, or the RDMA gather failed.
    #[cfg(all(target_os = "linux", feature = "gpudirect"))]
    fn next_batch_gpu(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<(usize, PySampledSubgraph, Py<PyAny>)>> {
        let rdma = self.rdma_gather.as_ref().ok_or_else(|| {
            sampling_error("Not an RDMA-enabled loader. Use with_rdma_features().")
        })?;

        // Both the blocking recv and the RDMA gather can stall; run them with
        // the GIL released. No Python object is touched inside the closure —
        // it works purely on Rust data and hands back the subgraph plus the
        // owned GPU feature buffer.
        type GpuBatch = (
            usize,
            PreparedSubgraph,
            aether_stream::rdma::gather::OwnedGpuFeatures,
        );
        let result = py.detach(|| -> PyResult<Option<GpuBatch>> {
            let Some((batch_idx, subgraph)) = self
                .inner
                .next_batch()
                .map_err(prefetch_error_to_py)?
                .map(|b| (b.batch_idx, b.subgraph))
            else {
                return Ok(None);
            };

            // Gather features via RDMA into VRAM (~20μs). `subgraph.nodes` is
            // already `Vec<u32>`; we hand a `&[u32]` straight to RDMA.
            let gpu_features = rdma
                .lock()
                .gather(&subgraph.nodes)
                .map_err(|e| sampling_error(format!("RDMA gather failed: {e}")))?;

            Ok(Some((batch_idx, prepare(subgraph)?, gpu_features)))
        })?;

        let Some((batch_idx, prepared, gpu_features)) = result else {
            return Ok(None);
        };

        let py_subgraph = PySampledSubgraph::from_prepared(py, prepared)?;

        // DLPack capsule for zero-copy transfer to PyTorch. The capsule takes
        // ownership of the gathered buffer, so the tensor's VRAM lives until
        // its consumer drops it.
        let capsule = crate::dlpack::create_dlpack_capsule(py, gpu_features)?;

        Ok(Some((batch_idx, py_subgraph, capsule)))
    }

    /// Get feature dimension (if loading features).
    #[getter]
    fn feature_dim(&self) -> Option<usize> {
        self.feature_dim
    }

    /// Check if this prefetcher loads features.
    #[getter]
    fn has_features(&self) -> bool {
        self.feature_dim.is_some()
    }

    /// Node count of the sampled graph; seeds must lie below it.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Get current prefetch statistics.
    ///
    /// Returns:
    ///     PrefetchStats: Statistics about hit rate, misses, etc.
    fn stats(&self) -> PyPrefetchStats {
        PyPrefetchStats::snapshot(self.inner.stats())
    }

    /// Get the prefetch depth.
    #[getter]
    fn prefetch_depth(&self) -> usize {
        self.inner.prefetch_depth()
    }

    /// Shut the pipeline down and join its threads.
    ///
    /// Safe to call from any thread, and more than once: threads blocked in
    /// `submit()` or `next()` return promptly. Called automatically when the
    /// object is garbage collected.
    fn shutdown(&self, py: Python<'_>) {
        py.detach(|| self.inner.shutdown());
    }

    fn __repr__(&self) -> String {
        // `__repr__` is called on every `print()` and inside debuggers, so we
        // skip live stats here to keep it allocation-cheap. Call `.stats()`
        // explicitly to inspect hit rates.
        format!(
            "NeighborLoader(prefetch_depth={}, has_features={})",
            self.inner.prefetch_depth(),
            self.feature_dim.is_some()
        )
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) {
        self.shutdown(py);
    }
}
