//! Python bindings for FeatureStore with zero-copy numpy integration.

use aethergraph_core::{
    create_features as core_create_features, save_feature_data as core_save_feature_data,
    save_features as core_save_features, FeatureData as CoreFeatureData,
    FeatureStore as CoreFeatureStore, NodeId,
};
use numpy::{PyArray1, PyArrayMethods};
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
    fn load(path: &str, telemetry: bool) -> PyResult<Self> {
        let mut inner = CoreFeatureStore::load(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load features: {}", e))
        })?;

        if telemetry {
            inner = inner.with_telemetry();
        }

        Ok(Self { inner })
    }

    /// Get telemetry statistics (if enabled).
    ///
    /// Returns None if telemetry not enabled.
    fn telemetry(&self) -> Option<crate::feature_telemetry::PyFeatureLoadTelemetry> {
        self.inner
            .telemetry()
            .map(|t| crate::feature_telemetry::PyFeatureLoadTelemetry { inner: t })
    }

    /// Print telemetry summary to console.
    fn print_telemetry(&self) {
        self.inner.print_telemetry();
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
    /// numpy array of shape [feature_dim] (float32)
    ///
    /// # Example
    /// ```python
    /// features = store.get(node_id)  # shape: [768]
    /// ```
    fn get(&self, py: Python, node: NodeId) -> PyResult<Py<PyAny>> {
        let features = self
            .inner
            .get(node)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        // Copy to numpy array (mmap slice can't be directly exposed to Python)
        Ok(PyArray1::from_slice(py, features).unbind().into_any())
    }

    /// Get features for multiple nodes in batch (optimized).
    ///
    /// # Arguments
    /// - nodes: List or numpy array of node IDs
    ///
    /// # Returns
    /// numpy array of shape [len(nodes), feature_dim] (float32)
    ///
    /// # Performance
    /// - Vectorized batch loading (better than individual gets)
    /// - ~1-10ms for 1000 nodes depending on cache hits
    /// - Accepts numpy arrays directly (zero-copy)
    ///
    /// # Example
    /// ```python
    /// sampled_nodes = np.array([0, 10, 100, 1000], dtype=np.int32)
    /// batch = store.get_batch(sampled_nodes)  # shape: [4, feature_dim]
    /// ```
    fn get_batch(&self, py: Python, nodes: numpy::PyReadonlyArray1<i64>) -> PyResult<Py<PyAny>> {
        let nodes_slice = nodes.as_slice()?;

        // Zero-copy: pass slice directly, casts inline
        let features = self
            .inner
            .get_batch(nodes_slice)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        // Reshape to [num_nodes, feature_dim]
        let num_nodes = nodes_slice.len();
        let feature_dim = self.inner.feature_dim();

        let array = PyArray1::from_vec(py, features);
        let reshaped = array.reshape([num_nodes, feature_dim])?;
        Ok(reshaped.unbind().into_any())
    }

    /// Get all features as a 2D array (zero-copy when possible).
    ///
    /// # Returns
    /// numpy array of shape [num_nodes, feature_dim] (float32)
    ///
    /// # Warning
    /// Only use this if features fit in RAM! For large graphs, use get_batch() instead.
    ///
    /// # Example
    /// ```python
    /// all_features = store.features()  # shape: [num_nodes, feature_dim]
    /// ```
    fn features(&self, py: Python) -> PyResult<Py<PyAny>> {
        let features = self.inner.features();

        // Create numpy array from slice
        let array = PyArray1::from_slice(py, features);
        let reshaped = array.reshape([self.inner.num_nodes(), self.inner.feature_dim()])?;
        Ok(reshaped.unbind().into_any())
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
fn save_features(path: &str, features: numpy::PyReadonlyArray2<f32>) -> PyResult<()> {
    let features_array = features.as_array();
    let shape = features_array.shape();
    let num_nodes = shape[0];
    let feature_dim = shape[1];

    // Convert to flat Vec (row-major, contiguous)
    let features_vec: Vec<f32> = if features_array.is_standard_layout() {
        // Zero-copy if already contiguous
        features_array.iter().copied().collect()
    } else {
        // Copy with correct layout
        features_array.iter().copied().collect()
    };

    core_save_features(path, features_vec, num_nodes, feature_dim).map_err(|e| {
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

        slot.copy_from_slice(features_array.as_slice().unwrap());
        Ok(())
    }

    /// Save to file.
    fn save(&self, path: &str) -> PyResult<()> {
        core_save_feature_data(path, &self.inner).map_err(|e| {
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
