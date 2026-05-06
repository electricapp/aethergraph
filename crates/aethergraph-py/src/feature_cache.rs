use aethergraph_core::features::{FeatureCache, FeatureCacheConfig};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_async_runtimes::tokio::future_into_py;
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::cache_error;

/// Python wrapper for FeatureCacheConfig.
#[pyclass(name = "FeatureCacheConfig", from_py_object)]
#[derive(Clone)]
pub struct PyFeatureCacheConfig {
    inner: FeatureCacheConfig,
}

#[pymethods]
impl PyFeatureCacheConfig {
    /// Create a new feature cache configuration.
    ///
    /// Args:
    ///     gpu_capacity: Maximum number of features in GPU cache
    ///     cpu_capacity: Maximum number of features in CPU cache
    ///     feature_dim: Feature vector dimension
    ///     nvme_path: Path to NVMe storage for cold features
    ///
    /// Returns:
    ///     FeatureCacheConfig: Configuration object
    #[new]
    #[pyo3(signature = (gpu_capacity=10_000, cpu_capacity=1_000_000, feature_dim=128, nvme_path=None))]
    fn new(
        gpu_capacity: usize,
        cpu_capacity: usize,
        feature_dim: usize,
        nvme_path: Option<&str>,
    ) -> Self {
        Self {
            inner: FeatureCacheConfig {
                gpu_capacity,
                cpu_capacity,
                feature_dim,
                nvme_path: nvme_path.map(PathBuf::from),
                // Advanced warm-start knobs; not exposed through Python yet.
                // Defaults match FeatureCacheConfig::default() — no frequency
                // warmup, 0.8 GPU pin ratio if one is provided later.
                warmup_frequencies: None,
                pin_ratio: 0.8,
            },
        }
    }

    #[getter]
    fn gpu_capacity(&self) -> usize {
        self.inner.gpu_capacity
    }

    #[getter]
    fn cpu_capacity(&self) -> usize {
        self.inner.cpu_capacity
    }

    #[getter]
    fn feature_dim(&self) -> usize {
        self.inner.feature_dim
    }

    #[getter]
    fn nvme_path(&self) -> Option<String> {
        self.inner
            .nvme_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string())
    }

    fn __repr__(&self) -> String {
        let nvme_str = match &self.inner.nvme_path {
            Some(p) => p.display().to_string(),
            None => "None".to_string(),
        };
        format!(
            "FeatureCacheConfig(gpu_capacity={}, cpu_capacity={}, feature_dim={}, nvme_path={})",
            self.inner.gpu_capacity, self.inner.cpu_capacity, self.inner.feature_dim, nvme_str
        )
    }
}

/// Python wrapper for FeatureCache.
#[pyclass(name = "FeatureCache")]
pub struct PyFeatureCache {
    inner: Arc<FeatureCache>,
}

#[pymethods]
impl PyFeatureCache {
    /// Create a new feature cache (async constructor).
    ///
    /// Args:
    ///     config: FeatureCacheConfig with capacity and dimension settings
    ///
    /// Returns:
    ///     Coroutine that resolves to FeatureCache instance
    ///
    /// Example:
    ///     config = FeatureCacheConfig()
    ///     cache = await FeatureCache.create(config)
    #[staticmethod]
    fn create<'py>(py: Python<'py>, config: PyFeatureCacheConfig) -> PyResult<Bound<'py, PyAny>> {
        future_into_py(py, async move {
            let cache = FeatureCache::new(config.inner)
                .await
                .map_err(|e| cache_error(format!("Failed to create feature cache: {}", e)))?;

            Ok(PyFeatureCache {
                inner: Arc::new(cache),
            })
        })
    }

    /// Get features for a single node (async).
    ///
    /// Args:
    ///     node: Node ID
    ///
    /// Returns:
    ///     numpy.ndarray: Feature vector (dtype=float32)
    fn get<'py>(&self, py: Python<'py>, node: u32) -> PyResult<Bound<'py, PyAny>> {
        let cache = Arc::clone(&self.inner);

        future_into_py(py, async move {
            let features = cache.get(node).await.map_err(|e| {
                cache_error(format!("Failed to get features for node {}: {}", node, e))
            })?;

            Ok(Python::attach(|py| {
                PyArray1::from_vec(py, features).unbind().into_any()
            }))
        })
    }

    /// Get features for multiple nodes in batch (async).
    ///
    /// This is more efficient than calling get() multiple times.
    ///
    /// Args:
    ///     nodes: List of node IDs
    ///
    /// Returns:
    ///     numpy.ndarray: Feature matrix with shape [len(nodes), feature_dim] (dtype=float32)
    fn get_batch<'py>(&self, py: Python<'py>, nodes: Vec<u32>) -> PyResult<Bound<'py, PyAny>> {
        let cache = Arc::clone(&self.inner);

        future_into_py(py, async move {
            let features_vec = cache
                .get_batch(&nodes)
                .await
                .map_err(|e| cache_error(format!("Failed to get batch features: {}", e)))?;

            Python::attach(|py| {
                // Convert Vec<Vec<f32>> to 2D numpy array
                let num_nodes = features_vec.len();
                let feature_dim = if num_nodes > 0 {
                    features_vec[0].len()
                } else {
                    0
                };

                // Flatten into 1D array
                let flat: Vec<f32> = features_vec.into_iter().flatten().collect();

                // Create 2D array
                let array = PyArray1::from_vec(py, flat);
                let array_2d = array
                    .reshape([num_nodes, feature_dim])
                    .map_err(|e| cache_error(format!("Failed to reshape array: {}", e)))?;

                Ok(array_2d.unbind().into_any())
            })
        })
    }

    /// Insert features for a node into the cache (async).
    ///
    /// Args:
    ///     node: Node ID
    ///     features: Feature vector as numpy array or list
    fn insert<'py>(
        &self,
        py: Python<'py>,
        node: u32,
        features: Vec<f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cache = Arc::clone(&self.inner);

        future_into_py(py, async move {
            cache
                .insert(node, features)
                .await
                .map_err(|e| cache_error(format!("Failed to insert features: {}", e)))?;

            Ok(())
        })
    }

    /// Get cache statistics.
    ///
    /// Returns:
    ///     dict: Dictionary with keys:
    ///         - gpu_hits: Number of GPU cache hits
    ///         - cpu_hits: Number of CPU cache hits
    ///         - nvme_hits: Number of NVMe cache hits
    ///         - misses: Number of cache misses
    ///         - evictions: Number of evictions
    ///         - gpu_hit_rate: GPU hit rate percentage
    ///         - cpu_hit_rate: CPU hit rate percentage
    ///         - total_requests: Total number of requests
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let stats = self.inner.stats();

        let total_requests = stats.gpu_hits + stats.cpu_hits + stats.nvme_hits + stats.misses;

        let gpu_hit_rate = if total_requests > 0 {
            (stats.gpu_hits as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };

        let cpu_hit_rate = if total_requests > 0 {
            (stats.cpu_hits as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };

        let dict = PyDict::new(py);
        dict.set_item("gpu_hits", stats.gpu_hits)?;
        dict.set_item("cpu_hits", stats.cpu_hits)?;
        dict.set_item("nvme_hits", stats.nvme_hits)?;
        dict.set_item("misses", stats.misses)?;
        dict.set_item("evictions", stats.evictions)?;
        dict.set_item("gpu_hit_rate", gpu_hit_rate)?;
        dict.set_item("cpu_hit_rate", cpu_hit_rate)?;
        dict.set_item("total_requests", total_requests)?;

        Ok(dict.into())
    }

    /// Print cache statistics to console.
    fn print_stats(&self) {
        self.inner.print_stats();
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "FeatureCache(gpu_hits={}, cpu_hits={}, nvme_hits={}, misses={})",
            stats.gpu_hits, stats.cpu_hits, stats.nvme_hits, stats.misses
        )
    }
}

impl PyFeatureCache {
    /// Get a reference to the inner FeatureCache (for internal use).
    pub fn inner(&self) -> &FeatureCache {
        &self.inner
    }
}
