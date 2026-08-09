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
use crate::sampler::{PySampledSubgraph, PySamplingConfig};

/// Return type of [`PyNeighborLoader::next_with_features`].
///
/// `Some((subgraph, Some(features)))` — both subgraph and features available.
/// `Some((subgraph, None))` — loader has no feature column attached.
/// `None` — the loader was shut down / drained and there are no more batches.
pub type NextWithFeaturesResult<'py> =
    Option<(PySampledSubgraph, Option<Bound<'py, PyArray2<f32>>>)>;

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
#[pyclass(name = "PrefetchStats")]
pub struct PyPrefetchStats {
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) total: u64,
    pub(crate) sample_time_ns: u64,
    pub(crate) feature_load_time_ns: u64,
}

#[pymethods]
impl PyPrefetchStats {
    /// Number of batches that were immediately available.
    #[getter]
    fn hits(&self) -> u64 {
        self.hits
    }

    /// Number of times the consumer had to wait.
    #[getter]
    fn misses(&self) -> u64 {
        self.misses
    }

    /// Total batches processed.
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

/// Prefetching neighbor sampler for pipelined GNN training.
///
/// Spawns a dedicated thread that samples batches ahead of time,
/// ensuring the training loop never waits for the sampler.
///
/// On Linux with NVMe storage, uses io_uring with SQPOLL for
/// zero-syscall I/O. Can also load features alongside sampling.
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
#[pyclass(name = "NeighborLoader")]
pub struct PyNeighborLoader {
    inner: Option<NeighborLoader>,
    // Keep graph alive for the lifetime of the sampler
    _graph: Arc<Graph>,
    // Feature dimension (if loading features)
    feature_dim: Option<usize>,
    // RDMA feature gather (gpudirect path)
    #[cfg(all(target_os = "linux", feature = "gpudirect"))]
    rdma_gather: Option<aether_stream::rdma::gather::RdmaFeatureGather>,
}

#[pymethods]
impl PyNeighborLoader {
    /// Create a new prefetching sampler (sampling only, no features).
    ///
    /// Args:
    ///     graph: Graph to sample from
    ///     config: SamplingConfig for sampling parameters
    ///     prefetch_depth: Number of batches to prefetch ahead (default: 2)
    ///     sampler_threads: Sampler worker threads pulling from the shared
    ///         work queue (default: 1). Results arrive unordered across the
    ///         pool; each carries its identity, so consumers are unaffected.
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
                "Failed to create NeighborLoader: {}",
                e
            ))
        })?;

        Ok(Self {
            inner: Some(inner),
            _graph: graph_arc,
            feature_dim: None,
            #[cfg(all(target_os = "linux", feature = "gpudirect"))]
            rdma_gather: None,
        })
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
        .map_err(|e| sampling_error(format!("Failed to create prefetcher with features: {}", e)))?;

        let feature_dim = inner.feature_dim();

        Ok(Self {
            inner: Some(inner),
            _graph: graph_arc,
            feature_dim,
            #[cfg(all(target_os = "linux", feature = "gpudirect"))]
            rdma_gather: None,
        })
    }

    /// Create a prefetching sampler for NVMe-backed graphs (Linux only).
    ///
    /// Uses io_uring with SQPOLL for zero-syscall graph topology reads.
    /// Ideal for graphs that don't fit in RAM - topology stays on NVMe.
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
    ///     OSError: If not on Linux, io_uring is unavailable, or the config
    ///         sets an option the io_uring path does not support
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
        graph_path: PathBuf,
        config: &PySamplingConfig,
        prefetch_depth: usize,
    ) -> PyResult<Self> {
        let inner = NeighborLoader::new_nvme(&graph_path, config.inner().clone(), prefetch_depth)
            .map_err(|e| {
            pyo3::exceptions::PyOSError::new_err(format!("Failed to open NVMe graph: {}", e))
        })?;

        Ok(Self {
            inner: Some(inner),
            _graph: Arc::new(Graph::empty()), // Placeholder - graph is file-backed
            feature_dim: None,
            #[cfg(all(target_os = "linux", feature = "gpudirect"))]
            rdma_gather: None,
        })
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
    /// Results come back from `next()` in completion order, which is FIFO —
    /// the single sampler thread processes submissions in the order they were
    /// sent, so it matches submission order. `batch_idx` is caller bookkeeping
    /// echoed back on the result; nothing is reordered by it and duplicate or
    /// gapped indices are harmless.
    ///
    /// Blocks (with the GIL released) when the pipeline is full, until the
    /// consumer drains a result — so interleave `submit()` with `next()`, or
    /// submit from a separate thread.
    ///
    /// Args:
    ///     batch_idx: Caller-chosen index for this batch, echoed back on the
    ///         result for bookkeeping.
    ///     seeds: Seed node IDs (numpy uint32, numpy int64, or `list[int]`).
    fn submit(&self, py: Python<'_>, batch_idx: usize, seeds: &Bound<'_, PyAny>) -> PyResult<()> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| sampling_error("Prefetcher has been shut down"))?;

        let seeds_vec = crate::error::extract_seeds(seeds)?;

        // The bounded work channel blocks when the pipeline is full; release
        // the GIL so the consumer thread can drain. No Python object is
        // touched inside.
        py.detach(|| inner.submit(batch_idx, seeds_vec))
            .map_err(|e| sampling_error(format!("Submit failed: {}", e)))
    }

    /// Submit all batches for an epoch.
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
    ///     RuntimeError: The prefetch worker exited unexpectedly.
    fn next(&self, py: Python<'_>) -> PyResult<Option<PySampledSubgraph>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| sampling_error("Prefetcher has been shut down"))?;

        // The blocking recv can stall while the worker samples; release the GIL
        // so other Python threads run. No Python object is touched inside.
        match py.detach(|| inner.next()).map_err(prefetch_error_to_py)? {
            Some(subgraph) => {
                let py_subgraph = PySampledSubgraph::from_subgraph(py, subgraph)?;
                Ok(Some(py_subgraph))
            }
            None => Ok(None),
        }
    }

    /// Try to get the next subgraph without blocking.
    ///
    /// Returns:
    ///     SampledSubgraph: If a batch is immediately available
    ///     None: If no batch is ready yet, or after `shutdown()`
    ///
    /// Raises:
    ///     RuntimeError: The prefetch worker exited unexpectedly.
    fn try_next(&self, py: Python<'_>) -> PyResult<Option<PySampledSubgraph>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| sampling_error("Prefetcher has been shut down"))?;

        // Non-blocking, but the channel recv still does a little work; release
        // the GIL across it for consistency with `next()`.
        match py
            .detach(|| inner.try_next())
            .map_err(prefetch_error_to_py)?
        {
            Some(subgraph) => {
                let py_subgraph = PySampledSubgraph::from_subgraph(py, subgraph)?;
                Ok(Some(py_subgraph))
            }
            None => Ok(None),
        }
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
    ///     RuntimeError: The prefetch worker exited unexpectedly, or loading
    ///         this batch's features failed.
    fn next_with_features<'py>(&self, py: Python<'py>) -> PyResult<NextWithFeaturesResult<'py>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| sampling_error("Prefetcher has been shut down"))?;

        // The blocking recv can stall while the worker samples and loads
        // features; release the GIL across it. The numpy/py outputs are built
        // afterward, back under the GIL.
        match py
            .detach(|| inner.next_with_features())
            .map_err(prefetch_error_to_py)?
        {
            Some((subgraph, features_opt)) => {
                let num_nodes = subgraph.nodes.len();
                let py_subgraph = PySampledSubgraph::from_subgraph(py, subgraph)?;

                let py_features = if let Some(features) = features_opt {
                    if let Some(dim) = self.feature_dim {
                        // Reshape flat features to (num_nodes, feature_dim)
                        let arr = PyArray1::from_vec(py, features);
                        Some(arr.reshape([num_nodes, dim]).map_err(|e| {
                            sampling_error(format!("Failed to reshape features: {}", e))
                        })?)
                    } else {
                        None
                    }
                } else {
                    None
                };

                Ok(Some((py_subgraph, py_features)))
            }
            None => Ok(None),
        }
    }

    /// Create a prefetcher that gathers features via GPUDirect RDMA.
    ///
    /// Features land directly in VRAM — never touch CPU.
    /// Use `next_with_gpu_features()` to get DLPack capsules that
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
            inner: Some(inner),
            _graph: graph_arc,
            feature_dim,
            rdma_gather: Some(rdma),
        })
    }

    /// Get next subgraph with GPU features (RDMA path).
    ///
    /// Samples a subgraph, then gathers features via RDMA directly into VRAM.
    /// Returns (SampledSubgraph, PyCapsule) where the capsule is a DLPack
    /// managed tensor owning its VRAM — the buffer stays valid for the
    /// consuming tensor's whole lifetime, independent of later gathers or the
    /// loader itself. Use `torch.from_dlpack(capsule)` in Python.
    ///
    /// Returns None only after `shutdown()` — the pipeline was drained and no
    /// more batches will arrive.
    ///
    /// Raises:
    ///     TimeoutError: No result arrived within the wait window; the call
    ///         may be retried.
    ///     RuntimeError: The prefetch worker exited unexpectedly, or the RDMA
    ///         gather failed.
    #[cfg(all(target_os = "linux", feature = "gpudirect"))]
    fn next_with_gpu_features(
        &mut self,
        py: Python<'_>,
    ) -> PyResult<Option<(PySampledSubgraph, Py<PyAny>)>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| sampling_error("Prefetcher has been shut down"))?;

        let rdma = self.rdma_gather.as_mut().ok_or_else(|| {
            sampling_error("Not an RDMA-enabled loader. Use with_rdma_features().")
        })?;

        // Both the blocking recv and the RDMA gather can stall; run them with
        // the GIL released. No Python object is touched inside the closure —
        // it works purely on Rust data and hands back the subgraph plus the
        // owned GPU feature buffer.
        type GpuBatch = (
            aethergraph_core::SampledSubgraph,
            aether_stream::rdma::gather::OwnedGpuFeatures,
        );
        let result = py.detach(|| -> PyResult<Option<GpuBatch>> {
            let subgraph = match inner.next().map_err(prefetch_error_to_py)? {
                Some(sg) => sg,
                None => return Ok(None),
            };

            // Gather features via RDMA into VRAM (~20μs). `subgraph.nodes` is
            // already `Vec<u32>`; we hand a `&[u32]` straight to RDMA.
            let gpu_features = rdma
                .gather(&subgraph.nodes)
                .map_err(|e| sampling_error(format!("RDMA gather failed: {e}")))?;

            Ok(Some((subgraph, gpu_features)))
        })?;

        let Some((subgraph, gpu_features)) = result else {
            return Ok(None);
        };

        let py_subgraph = PySampledSubgraph::from_subgraph(py, subgraph)?;

        // DLPack capsule for zero-copy transfer to PyTorch. The capsule takes
        // ownership of the gathered buffer, so the tensor's VRAM lives until
        // its consumer drops it.
        let capsule = crate::dlpack::create_dlpack_capsule(py, gpu_features)?;

        Ok(Some((py_subgraph, capsule)))
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

    /// Get current prefetch statistics.
    ///
    /// Returns:
    ///     PrefetchStats: Statistics about hit rate, misses, etc.
    fn stats(&self) -> PyResult<PyPrefetchStats> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| sampling_error("Prefetcher has been shut down"))?;

        let stats = inner.stats();
        Ok(PyPrefetchStats {
            hits: stats.hits.load(std::sync::atomic::Ordering::Relaxed),
            misses: stats.misses.load(std::sync::atomic::Ordering::Relaxed),
            total: stats.total.load(std::sync::atomic::Ordering::Relaxed),
            sample_time_ns: stats
                .sample_time_ns
                .load(std::sync::atomic::Ordering::Relaxed),
            feature_load_time_ns: stats
                .feature_load_time_ns
                .load(std::sync::atomic::Ordering::Relaxed),
        })
    }

    /// Get the prefetch depth.
    #[getter]
    fn prefetch_depth(&self) -> PyResult<usize> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| sampling_error("Prefetcher has been shut down"))?;
        Ok(inner.prefetch_depth())
    }

    /// Shutdown the prefetch thread.
    ///
    /// Called automatically when the object is garbage collected.
    fn shutdown(&mut self) {
        if let Some(mut inner) = self.inner.take() {
            inner.shutdown();
        }
    }

    fn __repr__(&self) -> String {
        // `__repr__` is called on every `print()` and inside debuggers, so we
        // skip live stats here to keep it allocation-cheap. Call `.stats()`
        // explicitly to inspect hit rates.
        match &self.inner {
            Some(inner) => format!(
                "NeighborLoader(prefetch_depth={}, has_features={})",
                inner.prefetch_depth(),
                self.feature_dim.is_some()
            ),
            None => "NeighborLoader(shutdown)".to_string(),
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) {
        self.shutdown();
    }
}

impl Drop for PyNeighborLoader {
    fn drop(&mut self) {
        self.shutdown();
    }
}
