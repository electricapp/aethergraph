//! Python bindings for the cross-process shared feature store.

use aethergraph_core::{ShareHandle, SharedFeatureStore};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

/// A feature store held once in shared memory and mapped by many
/// processes.
///
/// The owner publishes a feature file into a sealed `memfd`, then serves
/// the descriptor on a Unix socket; each worker attaches and maps the same
/// physical pages read-only. N workers therefore cost one copy of the
/// feature matrix, and attaching is a mmap rather than a read.
///
/// Linux only.
///
/// # Example
/// ```python
/// # In the owner process:
/// store = SharedFeatureStore.publish("features.bin")
/// store.serve("/tmp/aether-features.sock")
///
/// # In each worker process:
/// store = SharedFeatureStore.attach("/tmp/aether-features.sock")
/// rows = store.get_batch(np.array([1, 2, 3], dtype=np.int64))
/// ```
#[pyclass(name = "SharedFeatureStore")]
pub struct PySharedFeatureStore {
    inner: SharedFeatureStore,
    /// Live only in the owner process, and only after `serve`. Dropping it
    /// stops the listener, so it is held here for the store's lifetime.
    handle: Option<ShareHandle>,
}

#[pymethods]
impl PySharedFeatureStore {
    /// Copy a feature file's payload into a fresh sealed shared region.
    ///
    /// # Arguments
    /// - path: Path to a feature file created with save_features()
    #[staticmethod]
    fn publish(py: Python<'_>, path: std::path::PathBuf) -> PyResult<Self> {
        // The copy is O(payload) and touches no Python state.
        let inner = py
            .detach(|| SharedFeatureStore::publish(&path))
            .map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(format!("Failed to publish features: {e}"))
            })?;
        Ok(Self {
            inner,
            handle: None,
        })
    }

    /// Attach to a store being served at `socket_path`, mapping its pages
    /// read-only into this process.
    #[staticmethod]
    fn attach(py: Python<'_>, socket_path: std::path::PathBuf) -> PyResult<Self> {
        let inner = py
            .detach(|| SharedFeatureStore::attach(&socket_path))
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("Failed to attach: {e}")))?;
        Ok(Self {
            inner,
            handle: None,
        })
    }

    /// Start serving this store to workers on `socket_path`.
    ///
    /// Serving continues until `stop_serving()` or the store is dropped.
    /// Calling it twice replaces the previous listener.
    fn serve(&mut self, py: Python<'_>, socket_path: std::path::PathBuf) -> PyResult<()> {
        // Drop any previous listener first, so its socket file is removed
        // before a new bind — otherwise the old thread outlives the path.
        self.handle = None;
        let handle = py
            .detach(|| self.inner.serve(&socket_path))
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("Failed to serve: {e}")))?;
        self.handle = Some(handle);
        Ok(())
    }

    /// Stop serving. Already-attached workers keep their mappings.
    fn stop_serving(&mut self) {
        self.handle = None;
    }

    /// Whether this store is currently serving workers.
    #[getter]
    fn is_serving(&self) -> bool {
        self.handle.is_some()
    }

    /// Number of nodes.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Feature dimension.
    #[getter]
    fn feature_dim(&self) -> usize {
        self.inner.feature_dim()
    }

    /// Bytes of shared memory backing the store — one copy per host,
    /// however many processes attach.
    #[getter]
    fn shared_bytes(&self) -> usize {
        self.inner.shared_bytes()
    }

    /// Gather features for `nodes`.
    ///
    /// # Returns
    /// Array of shape [len(nodes), feature_dim] (float32), copied out of
    /// the shared pages.
    fn get_batch<'py>(
        &self,
        py: Python<'py>,
        nodes: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let ids = crate::error::copy_node_ids_i64(nodes)?;

        let flat = py
            .detach(|| self.inner.get_batch(&ids))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

        let dim = self.inner.feature_dim();
        let arr = numpy::PyArray1::from_vec(py, flat);
        arr.reshape([ids.len(), dim])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "SharedFeatureStore(num_nodes={}, feature_dim={}, shared_bytes={}, serving={})",
            self.inner.num_nodes(),
            self.inner.feature_dim(),
            self.inner.shared_bytes(),
            self.handle.is_some()
        )
    }
}
