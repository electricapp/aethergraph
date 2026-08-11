//! Python bindings for FeatureStore with zero-copy numpy integration.

use aethergraph_core::{
    FeatureData as CoreFeatureData, FeatureStore as CoreFeatureStore, NodeId,
    create_features as core_create_features, save_feature_data as core_save_feature_data,
    save_features as core_save_features,
};
use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

/// Python wrapper for FeatureStore (memory-mapped feature access).
///
/// # Zero-Copy Integration
/// - Returns numpy arrays that reference mmap'd memory (no copy)
/// - Perfect for TB-scale features that don't fit in RAM
/// - Automatic OS paging from NVMe as needed
///
/// # Example
/// ```python
/// from aethergraph import FeatureStore, save_features
/// import numpy as np
///
/// # Create features
/// features = np.random.randn(100000, 768).astype(np.float32)
/// save_features("features.bin", features)
///
/// # Load with zero-copy mmap
/// store = FeatureStore.load("features.bin")
/// print(f"{store.num_nodes} nodes, {store.feature_dim} dims")
///
/// # Get features for sampled nodes (batch optimized)
/// sampled_nodes = [0, 100, 1000, 5000]
/// batch_features = store.get_batch(sampled_nodes)  # shape: [4, 768]
/// ```
#[pyclass(name = "FeatureStore")]
pub struct PyFeatureStore {
    pub(crate) inner: CoreFeatureStore,
}

#[pymethods]
impl PyFeatureStore {
    /// Load features from file (memory-mapped, zero-copy).
    ///
    /// # Arguments
    /// - path: Path to feature file created with save_features()
    /// - telemetry: Enable telemetry tracking (default: False)
    ///
    /// # Performance
    /// - O(1) load time (just mmaps the file)
    /// - No data copied into memory
    /// - ~100ns per node access if in cache, ~10μs if on NVMe
    ///
    /// # Example
    /// ```python
    /// store = FeatureStore.load("features.bin", telemetry=True)
    /// # ... do feature loading ...
    /// stats = store.telemetry()
    /// print(f"Throughput: {stats.throughput_gb_per_sec():.2f} GB/s")
    /// ```
    #[staticmethod]
    #[pyo3(signature = (path, telemetry=false))]
    fn load(path: std::path::PathBuf, telemetry: bool) -> PyResult<Self> {
        let mut inner = CoreFeatureStore::load(&path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load features: {}", e))
        })?;

        if telemetry {
            inner = inner.with_telemetry();
        }

        Ok(Self { inner })
    }

    /// Load features demand-paged from disk with a bounded resident set.
    ///
    /// Unlike `load`, which maps the file and lets the kernel decide what
    /// stays in RAM, this registers the region with `userfaultfd` and
    /// pages it in on demand, holding at most `budget_pages` pages. Pass
    /// `degrees` (one entry per node) to make eviction degree-weighted, so
    /// a hub node's pages outlive a leaf's.
    ///
    /// # Arguments
    /// - path: Path to feature file created with save_features()
    /// - budget_pages: Maximum resident pages (page size is typically 4096)
    /// - degrees: Optional per-node degrees driving retention
    /// - telemetry: Enable telemetry tracking (default: False)
    ///
    /// Requires Linux with `vm.unprivileged_userfaultfd=1` or
    /// CAP_SYS_PTRACE; raises OSError elsewhere, where `load` is the
    /// fallback.
    #[cfg(all(target_os = "linux", feature = "uffd"))]
    #[staticmethod]
    #[pyo3(signature = (path, budget_pages, degrees=None, telemetry=false))]
    fn load_paged(
        path: std::path::PathBuf,
        budget_pages: usize,
        degrees: Option<numpy::PyReadonlyArray1<'_, u32>>,
        telemetry: bool,
    ) -> PyResult<Self> {
        use aethergraph_core::{PageWeights, uffd_page_size};

        let weights = match &degrees {
            Some(d) => {
                let file = std::fs::File::open(&path).map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!("Failed to open features: {e}"))
                })?;
                let header = aethergraph_core::parse_feature_header(&file).map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!("Failed to read header: {e}"))
                })?;
                PageWeights::from_node_degrees(
                    d.as_slice()?,
                    header.features_start_offset,
                    header.feature_size,
                    uffd_page_size(),
                )
            }
            None => PageWeights::default(),
        };

        let mut inner =
            CoreFeatureStore::load_paged(&path, budget_pages, std::sync::Arc::new(weights))
                .map_err(|e| {
                    pyo3::exceptions::PyOSError::new_err(format!("Failed to page features: {e}"))
                })?;

        if telemetry {
            inner = inner.with_telemetry();
        }

        Ok(Self { inner })
    }

    /// Pager counters as `(faults, evictions)`, or None for a store opened
    /// with `load` rather than `load_paged`.
    #[cfg(all(target_os = "linux", feature = "uffd"))]
    fn pager_stats(&self) -> Option<(u64, u64)> {
        self.inner.pager_stats()
    }

    /// Get telemetry statistics (if enabled).
    ///
    /// Returns None if telemetry not enabled.
    fn telemetry(&self) -> Option<crate::feature_telemetry::PyFeatureLoadTelemetry> {
        self.inner
            .telemetry()
            .map(|t| crate::feature_telemetry::PyFeatureLoadTelemetry { inner: t })
    }

    /// Get number of nodes
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Get feature dimension
    #[getter]
    fn feature_dim(&self) -> usize {
        self.inner.feature_dim()
    }

    /// Get features for a single node (zero-copy).
    ///
    /// # Returns
    /// Read-only numpy array of shape [feature_dim] (float32), viewing the
    /// mmap'd file. Writing to it raises ValueError.
    ///
    /// # Example
    /// ```python
    /// features = store.get(node_id)  # shape: [768]
    /// ```
    fn get<'py>(slf: Bound<'py, Self>, node: NodeId) -> PyResult<Bound<'py, PyArray1<f32>>> {
        // Lift the (ptr, len) out of the borrow so we can move `slf` into the
        // numpy container after the borrow ends.
        let (ptr, len) = {
            let store = slf.borrow();
            let features = store
                .inner
                .get(node)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
            (features.as_ptr(), features.len())
        };

        // SAFETY: `ptr`/`len` describe a slice into the mmap owned by `inner`.
        // The mmap is stable for the FeatureStore's lifetime and is never
        // reallocated. Passing `slf` as the numpy container increfs the
        // PyFeatureStore so the mmap outlives the returned array.
        let view = unsafe { ArrayView1::from_shape_ptr(len, ptr) };
        // SAFETY: same backing mmap; `slf.into_any()` is the lifetime anchor.
        let arr = unsafe { PyArray1::borrow_from_array(&view, slf.into_any()) };
        // The backing mmap is read-only; clear the WRITEABLE flag so Python
        // writes raise ValueError instead of faulting on the mapped pages.
        arr.readwrite().make_nonwriteable();
        Ok(arr)
    }

    /// Get features for multiple nodes in batch (optimized).
    ///
    /// # Arguments
    /// - nodes: List or numpy array of node IDs. Accepts `int64` (PyTorch
    ///   default index dtype) — values are bounds-checked and converted to
    ///   `u32` by the core. Negative values or values exceeding `u32::MAX`
    ///   raise `ValueError` rather than wrapping silently.
    ///
    /// # Returns
    /// numpy array of shape [len(nodes), feature_dim] (float32)
    ///
    /// # Why i64 here when get() takes u32?
    /// PyTorch's `Tensor` index dtype is `int64`. Batched ID arrays come from
    /// PyG/PyTorch which already produce `int64`. Accepting `int64` here saves
    /// a Python-side `.astype(np.uint32)`. The single-node `get()` is hand
    /// indexed and `u32` is the more natural Pythonic choice there.
    ///
    /// # Performance
    /// - Vectorized batch loading (better than individual gets)
    /// - ~1-10ms for 1000 nodes depending on cache hits
    /// - Accepts numpy arrays directly (zero-copy on the input side)
    ///
    /// # Example
    /// ```python
    /// sampled_nodes = np.array([0, 10, 100, 1000], dtype=np.int64)
    /// batch = store.get_batch(sampled_nodes)  # shape: [4, feature_dim]
    /// ```
    fn get_batch(&self, py: Python, nodes: numpy::PyReadonlyArray1<i64>) -> PyResult<Py<PyAny>> {
        let nodes_slice = nodes.as_slice()?;

        // The core `get_batch<T>` is generic over `T: TryInto<NodeId>` and
        // returns a clear error per-element. So negatives, out-of-range
        // values, and out-of-graph IDs all surface as Python ValueError.
        //
        // A cold mmap page fault during the gather can block; release the GIL
        // across the gather so other Python threads run. The store and the
        // input slice are plain memory (no Python objects), and the numpy
        // array is built afterward, back under the GIL.
        let store = &self.inner;
        let features = py
            .detach(|| store.get_batch(nodes_slice))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        let num_nodes = nodes_slice.len();
        let feature_dim = self.inner.feature_dim();

        let array = PyArray1::from_vec(py, features);
        let reshaped = array.reshape([num_nodes, feature_dim])?;
        Ok(reshaped.unbind().into_any())
    }

    /// Get all features as a 2D array (zero-copy when possible).
    ///
    /// # Returns
    /// Read-only numpy array of shape [num_nodes, feature_dim] (float32),
    /// viewing the mmap'd file. Writing to it raises ValueError.
    ///
    /// # Warning
    /// Only use this if features fit in RAM! For large graphs, use get_batch() instead.
    ///
    /// # Example
    /// ```python
    /// all_features = store.features()  # shape: [num_nodes, feature_dim]
    /// ```
    fn features<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (ptr, num_nodes, feature_dim) = {
            let store = slf.borrow();
            let features = store
                .inner
                .features()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let num_nodes = store.inner.num_nodes();
            let feature_dim = store.inner.feature_dim();
            debug_assert_eq!(features.len(), num_nodes * feature_dim);
            (features.as_ptr(), num_nodes, feature_dim)
        };

        // SAFETY: `ptr` points to `num_nodes * feature_dim` contiguous f32s in
        // the mmap inside `inner`. The mmap is stable and never reallocated.
        // Passing `slf` as the container keeps the PyFeatureStore — and thus
        // the mmap — alive for as long as Python holds the array.
        let view = unsafe { ArrayView2::from_shape_ptr((num_nodes, feature_dim), ptr) };
        // SAFETY: same backing mmap; `slf.into_any()` is the lifetime anchor.
        let arr = unsafe { PyArray2::borrow_from_array(&view, slf.into_any()) };
        // The backing mmap is read-only; clear the WRITEABLE flag so Python
        // writes raise ValueError instead of faulting on the mapped pages.
        arr.readwrite().make_nonwriteable();
        Ok(arr)
    }

    fn __repr__(&self) -> String {
        format!(
            "FeatureStore(num_nodes={}, feature_dim={})",
            self.inner.num_nodes(),
            self.inner.feature_dim()
        )
    }
}

/// Save features to file (memory-mapped format).
///
/// # Arguments
/// - path: Output file path
/// - features: numpy array of shape [num_nodes, feature_dim] (float32)
///
/// # File Format
/// Simple binary format with 32-byte header + raw f32 data.
/// Compatible with both FeatureStore (sync) and AsyncFeatureStore (async).
///
/// # Performance
/// - ~1-2 GB/s write throughput
/// - Single-file format (no fragmentation)
///
/// # Example
/// ```python
/// import numpy as np
/// from aethergraph import save_features
///
/// features = np.random.randn(100000, 768).astype(np.float32)
/// save_features("features.bin", features)
/// ```
#[pyfunction]
#[pyo3(signature = (path, features))]
fn save_features(path: std::path::PathBuf, features: numpy::PyReadonlyArray2<f32>) -> PyResult<()> {
    let features_array = features.as_array();
    let shape = features_array.shape();
    let num_nodes = shape[0];
    let feature_dim = shape[1];

    // C-contiguous input (the overwhelmingly common case) materializes with
    // one bulk memcpy via `as_slice`; only genuinely non-contiguous views
    // fall back to element iteration, which yields C order regardless of
    // the source layout.
    let features_vec: Vec<f32> = match features_array.as_slice() {
        Some(s) => s.to_vec(),
        None => features_array.iter().copied().collect(),
    };

    core_save_features(&path, features_vec, num_nodes, feature_dim).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to save features: {}", e))
    })?;

    Ok(())
}

/// Mutable feature data builder for incremental construction.
///
/// # Example
/// ```python
/// from aethergraph import FeatureData
/// import numpy as np
///
/// # Create empty feature data
/// data = FeatureData(num_nodes=1000, feature_dim=128)
///
/// # Set features for each node
/// for node_id in range(1000):
///     features = np.random.randn(128).astype(np.float32)
///     data.set(node_id, features)
///
/// # Save to file
/// data.save("features.bin")
/// ```
#[pyclass(name = "FeatureData")]
pub struct PyFeatureData {
    inner: CoreFeatureData,
}

#[pymethods]
impl PyFeatureData {
    /// Create new empty feature data.
    ///
    /// # Arguments
    /// - num_nodes: Number of nodes
    /// - feature_dim: Feature dimension per node
    #[new]
    fn new(num_nodes: usize, feature_dim: usize) -> Self {
        Self {
            inner: core_create_features(num_nodes, feature_dim),
        }
    }

    /// Get number of nodes
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes as usize
    }

    /// Get feature dimension
    #[getter]
    fn feature_dim(&self) -> usize {
        self.inner.feature_dim as usize
    }

    /// Get features for a node.
    fn get(&self, py: Python, node: NodeId) -> PyResult<Py<PyAny>> {
        let features = self.inner.get(node).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("node {} out of bounds", node))
        })?;

        Ok(PyArray1::from_slice(py, features).unbind().into_any())
    }

    /// Set features for a node.
    ///
    /// # Arguments
    /// - node: Node ID
    /// - features: numpy array of shape [feature_dim] (float32)
    fn set(&mut self, node: NodeId, features: numpy::PyReadonlyArray1<f32>) -> PyResult<()> {
        let features_array = features.as_array();

        if features_array.len() != self.inner.feature_dim as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "feature dimension mismatch: expected {}, got {}",
                self.inner.feature_dim,
                features_array.len()
            )));
        }

        let slot = self.inner.get_mut(node).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("node {} out of bounds", node))
        })?;

        slot.copy_from_slice(features_array.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("features array must be C-contiguous")
        })?);
        Ok(())
    }

    /// Save to file.
    fn save(&self, path: std::path::PathBuf) -> PyResult<()> {
        core_save_feature_data(&path, &self.inner).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to save features: {}", e))
        })?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "FeatureData(num_nodes={}, feature_dim={})",
            self.inner.num_nodes, self.inner.feature_dim
        )
    }
}

/// Register feature store module with Python
pub fn register_feature_store(_py: Python, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_class::<PyFeatureStore>()?;
    parent_module.add_class::<PyFeatureData>()?;
    parent_module.add_class::<crate::feature_telemetry::PyFeatureLoadTelemetry>()?;
    parent_module.add_function(wrap_pyfunction!(save_features, parent_module)?)?;
    Ok(())
}
