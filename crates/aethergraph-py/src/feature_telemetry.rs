//! Python bindings for FeatureLoadTelemetry

use aethergraph_core::FeatureLoadTelemetry as CoreFeatureLoadTelemetry;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python wrapper for feature loading telemetry.
///
/// Tracks performance metrics for feature loading operations.
#[pyclass(name = "FeatureLoadTelemetry", from_py_object)]
#[derive(Clone)]
pub struct PyFeatureLoadTelemetry {
    pub(crate) inner: CoreFeatureLoadTelemetry,
}

#[pymethods]
impl PyFeatureLoadTelemetry {
    /// Number of single-node get() calls
    #[getter]
    fn single_gets(&self) -> u64 {
        self.inner.single_gets
    }

    /// Number of batch get_batch() calls
    #[getter]
    fn batch_gets(&self) -> u64 {
        self.inner.batch_gets
    }

    /// Total nodes loaded
    #[getter]
    fn total_nodes_loaded(&self) -> u64 {
        self.inner.total_nodes_loaded
    }

    /// Total features loaded (nodes × feature_dim)
    #[getter]
    fn total_features_loaded(&self) -> u64 {
        self.inner.total_features_loaded
    }

    /// Total bytes loaded
    #[getter]
    fn total_bytes_loaded(&self) -> u64 {
        self.inner.total_bytes_loaded
    }

    /// Total time in seconds
    fn total_time_secs(&self) -> f64 {
        self.inner.total_time_secs()
    }

    /// Throughput in features/second
    fn throughput_features_per_sec(&self) -> f64 {
        self.inner.throughput_features_per_sec()
    }

    /// Throughput in GB/second
    fn throughput_gb_per_sec(&self) -> f64 {
        self.inner.throughput_gb_per_sec()
    }

    /// Average batch size (nodes per batch)
    fn avg_batch_size(&self) -> f64 {
        self.inner.avg_batch_size()
    }

    /// Get summary as dictionary
    fn summary(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("single_gets", self.inner.single_gets)?;
        dict.set_item("batch_gets", self.inner.batch_gets)?;
        dict.set_item("total_nodes_loaded", self.inner.total_nodes_loaded)?;
        dict.set_item("total_features_loaded", self.inner.total_features_loaded)?;
        dict.set_item("total_bytes_loaded", self.inner.total_bytes_loaded)?;
        dict.set_item("total_time_secs", self.inner.total_time_secs())?;
        dict.set_item(
            "throughput_features_per_sec",
            self.inner.throughput_features_per_sec(),
        )?;
        dict.set_item("throughput_gb_per_sec", self.inner.throughput_gb_per_sec())?;
        dict.set_item("avg_batch_size", self.inner.avg_batch_size())?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "FeatureLoadTelemetry(nodes={}, features={}, time={:.4}s, throughput={:.2} GB/s)",
            self.inner.total_nodes_loaded,
            self.inner.total_features_loaded,
            self.inner.total_time_secs(),
            self.inner.throughput_gb_per_sec()
        )
    }
}
