//! Python bindings for AsyncFeatureStore with io_uring support.

use aethergraph_core::{AsyncFeatureStore as CoreAsyncFeatureStore, NodeId};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use std::sync::Arc;

/// Python wrapper for AsyncFeatureStore (io_uring-accelerated feature loading).
///
/// # Performance Tiers
/// 1. Linux + io_uring + SQPOLL + IOPOLL: Zero-syscall async I/O (best)
/// 2. Linux + io_uring (standard): Async I/O with syscalls (good)
/// 3. Tokio spawn_blocking: Multi-threaded sync fallback (works everywhere)
///
/// # Example
/// ```python
/// import asyncio
/// from aethergraph import AsyncFeatureStore, save_async_features
/// import numpy as np
///
/// # Create features
/// features = np.random.randn(100000, 768).astype(np.float32)
/// save_async_features("features_async.bin", features)
///
/// async def main():
///     # Load with io_uring on Linux
///     store = await AsyncFeatureStore.load("features_async.bin")
///     print(f"{store.num_nodes} nodes, {store.feature_dim} dims")
///
///     # Get features for sampled nodes (parallel async)
///     sampled_nodes = [0, 100, 1000, 5000]
///     batch_features = await store.get_batch(sampled_nodes)  # shape: [4, 768]
///
/// asyncio.run(main())
/// ```
#[pyclass(name = "AsyncFeatureStore")]
pub struct PyAsyncFeatureStore {
    inner: Arc<CoreAsyncFeatureStore>,
}

#[pymethods]
impl PyAsyncFeatureStore {
    /// Load features from file for async access (io_uring on Linux).
    ///
    /// # Arguments
    /// - path: Path to async feature file created with save_async_features()
    /// - telemetry: If true, instrument the store with a load-telemetry collector
    ///
    /// # Performance
    /// - Linux with SQPOLL: Zero-syscall parallel reads, 5-15x faster
    /// - Linux standard: Async io_uring with syscalls
    /// - Other platforms: Async with tokio spawn_blocking
    ///
    /// # Example
    /// ```python
    /// store = await AsyncFeatureStore.load("features_async.bin", telemetry=True)
    /// ```
    #[staticmethod]
    #[pyo3(signature = (path, telemetry=false))]
    fn load<'py>(py: Python<'py>, path: String, telemetry: bool) -> PyResult<Bound<'py, PyAny>> {
        future_into_py(py, async move {
            let mut inner = CoreAsyncFeatureStore::load(&path).await.map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to load async features: {e}"))
            })?;
            if telemetry {
                inner = inner.with_telemetry();
            }
            Ok(Self {
                inner: Arc::new(inner),
            })
        })
    }

    /// Get telemetry statistics (if enabled).
    fn telemetry(&self) -> Option<crate::feature_telemetry::PyFeatureLoadTelemetry> {
        self.inner
            .telemetry()
            .map(|t| crate::feature_telemetry::PyFeatureLoadTelemetry { inner: t })
    }

    /// Get number of nodes (Python attribute, not a method).
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Get feature dimension (Python attribute, not a method).
    #[getter]
    fn feature_dim(&self) -> usize {
        self.inner.feature_dim()
    }

    /// Get features for a single node (async).
    ///
    /// # Arguments
    /// - node: Node ID
    ///
    /// # Returns
    /// numpy array of shape [feature_dim]
    ///
    /// # Performance
    /// - io_uring: ~5-10μs per node (Linux with SQPOLL)
    /// - fallback: ~10-100μs per node
    fn get<'py>(&self, py: Python<'py>, node: NodeId) -> PyResult<Bound<'py, PyAny>> {
        let store = Arc::clone(&self.inner);

        future_into_py(py, async move {
            let features = store.get(node).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get features: {e}"))
            })?;

            Python::attach(|py| Ok(PyArray1::from_vec(py, features).unbind().into_any()))
        })
    }

    /// Get features for multiple nodes in batch (async, parallel).
    ///
    /// This is the key performance API - issues many parallel async reads.
    ///
    /// # Arguments
    /// - nodes: List of node IDs
    ///
    /// # Returns
    /// numpy array of shape [len(nodes), feature_dim]
    ///
    /// # Performance
    /// - io_uring with SQPOLL: ~100μs for 1000 nodes (zero syscalls!)
    /// - io_uring standard: ~500μs for 1000 nodes
    /// - tokio fallback: ~10ms for 1000 nodes
    ///
    /// # Example
    /// ```python
    /// features = await store.get_batch([0, 1, 2, 100, 500])
    /// assert features.shape == (5, 768)
    /// ```
    fn get_batch<'py>(&self, py: Python<'py>, nodes: Vec<NodeId>) -> PyResult<Bound<'py, PyAny>> {
        let store = Arc::clone(&self.inner);
        let feature_dim = self.inner.feature_dim();

        future_into_py(py, async move {
            let features_flat = store.get_batch(&nodes).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to get batch features: {e}"
                ))
            })?;

            // Convert flat Vec<f32> to 2D numpy array
            Python::attach(|py| {
                let arr = PyArray1::from_vec(py, features_flat);
                let shape = (nodes.len(), feature_dim);
                let arr_2d = arr.reshape(shape).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Failed to reshape array: {e}"))
                })?;
                Ok(arr_2d.unbind().into_any())
            })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "AsyncFeatureStore(num_nodes={}, feature_dim={})",
            self.inner.num_nodes(),
            self.inner.feature_dim()
        )
    }
}

/// Register AsyncFeatureStore with Python module.
pub fn register_async_feature_store(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAsyncFeatureStore>()?;
    Ok(())
}
