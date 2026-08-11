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
    ///     nvme_path: Path to NVMe storage for cold features. Required —
    ///         the cache always spills cold features to this directory.
    ///     cold_store_path: Optional feature-store file to compress into a
    ///         resident backing tier. With it, a node in no other tier
    ///         decompresses out of its block instead of raising, so the
    ///         cache becomes a complete feature source.
    ///     cold_level: zstd level for that tier (1-22, default 12).
    ///
    /// Returns:
    ///     FeatureCacheConfig: Configuration object
    ///
    /// Raises:
    ///     ValueError: If nvme_path is missing, feature_dim is 0, both
    ///         capacities are 0, or cold_level is out of range.
    #[new]
    #[pyo3(signature = (gpu_capacity=10_000, cpu_capacity=1_000_000, feature_dim=128, nvme_path=None, cold_store_path=None, cold_level=12))]
    fn new(
        gpu_capacity: usize,
        cpu_capacity: usize,
        feature_dim: usize,
        nvme_path: Option<&str>,
        cold_store_path: Option<&str>,
        cold_level: i32,
    ) -> PyResult<Self> {
        if feature_dim == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature_dim must be > 0",
            ));
        }
        if gpu_capacity == 0 && cpu_capacity == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "at least one of gpu_capacity / cpu_capacity must be > 0",
            ));
        }
        let Some(nvme_path) = nvme_path else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "nvme_path is required: the cache spills cold features to it, \
                 e.g. FeatureCacheConfig(nvme_path=\"/tmp/feature_cache\")",
            ));
        };
        if !(1..=22).contains(&cold_level) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cold_level must be in 1..=22",
            ));
        }
        Ok(Self {
            inner: FeatureCacheConfig {
                gpu_capacity,
                cpu_capacity,
                feature_dim,
                nvme_path: Some(PathBuf::from(nvme_path)),
                warmup_frequencies: None,
                pin_ratio: 0.8,
                cold_store_path: cold_store_path.map(PathBuf::from),
                cold_level,
            },
        })
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

    #[getter]
    fn cold_store_path(&self) -> Option<String> {
        self.inner
            .cold_store_path
            .as_ref()
            .map(|p| p.to_string_lossy().to_string())
    }

    #[getter]
    fn cold_level(&self) -> i32 {
        self.inner.cold_level
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
    ///     config = FeatureCacheConfig(nvme_path="/tmp/feature_cache")
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
    ///
    /// GIL: the returned coroutine releases the GIL while it awaits the
    /// underlying I/O, so other Python threads run concurrently. The GIL is
    /// reacquired only to materialize the numpy result.
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
    ///
    /// # Async/GIL contract
    /// The coroutine returned by `future_into_py` runs the inner `async move`
    /// block on the `pyo3-async-runtimes` tokio runtime. The GIL is released
    /// across `cache.get_batch(...).await`. The `Python::attach(|py| ...)`
    /// block at the end reacquires the GIL on the worker thread purely to
    /// materialize the numpy result — `pyo3-async-runtimes` guarantees this
    /// is sound for any task it drives. Do not hand-spawn the inner future
    /// onto a different runtime, or `Python::attach` may panic.
    fn get_batch<'py>(&self, py: Python<'py>, nodes: Vec<u32>) -> PyResult<Bound<'py, PyAny>> {
        let cache = Arc::clone(&self.inner);

        future_into_py(py, async move {
            let features_vec = cache
                .get_batch(&nodes)
                .await
                .map_err(|e| cache_error(format!("Failed to get batch features: {}", e)))?;

            Python::attach(|py| {
                let num_nodes = features_vec.len();
                let feature_dim = if num_nodes > 0 {
                    features_vec[0].len()
                } else {
                    0
                };

                // Pre-allocated single buffer + `extend_from_slice` per row:
                // one allocation total, no per-element iterator state.
                let mut flat: Vec<f32> = Vec::with_capacity(num_nodes * feature_dim);
                for row in features_vec {
                    if row.len() != feature_dim {
                        return Err(cache_error(format!(
                            "feature row length mismatch: row has {}, expected {}",
                            row.len(),
                            feature_dim
                        )));
                    }
                    flat.extend_from_slice(&row);
                }

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
    ///         - cold_hits: Number of compressed backing-tier hits
    ///         - misses: Number of cache misses
    ///         - evictions: Number of evictions
    ///         - gpu_hit_rate: GPU hit rate percentage
    ///         - cpu_hit_rate: CPU hit rate percentage
    ///         - total_requests: Total number of requests
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let stats = self.inner.stats();

        let total_requests =
            stats.gpu_hits + stats.cpu_hits + stats.nvme_hits + stats.cold_hits + stats.misses;

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
        dict.set_item("cold_hits", stats.cold_hits)?;
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

    /// `cache[node]` — awaitable shorthand for `cache.get(node)`.
    ///
    /// Returns the coroutine; users still need to `await` it. Matches the
    /// `__getitem__` shape PyG users expect from feature stores.
    fn __getitem__<'py>(&self, py: Python<'py>, node: u32) -> PyResult<Bound<'py, PyAny>> {
        self.get(py, node)
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
