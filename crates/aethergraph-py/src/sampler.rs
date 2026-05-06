use aethergraph_core::{
    NeighborSampler, ParallelBatchSampler, SampledSubgraph, SamplingConfig, SamplingTelemetry,
    SubgraphType, TemporalStrategy,
};
use arrow_array::{RecordBatch, UInt32Array};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

use crate::arrow_utils::subgraph_to_arrow;
use crate::error::sampling_error;
use crate::graph::PyCsrGraph;

/// Lightweight telemetry for sampling operations.
///
/// Thread-safe, lock-free metrics collection with zero overhead.
/// Call `summary()` to get current metrics on-demand.
#[pyclass(name = "SamplingTelemetry")]
#[derive(Clone)]
pub struct PySamplingTelemetry {
    inner: Arc<SamplingTelemetry>,
}

#[pymethods]
impl PySamplingTelemetry {
    /// Create a new telemetry collector.
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(SamplingTelemetry::new()),
        }
    }

    /// Get a summary of current metrics.
    ///
    /// Returns:
    ///     dict: Dictionary with keys:
    ///         - total_samples: Total sampling operations
    ///         - hub_nodes_capped: Number of hub nodes encountered
    ///         - total_nodes_sampled: Total nodes across all operations
    ///         - total_edges_sampled: Total edges across all operations
    ///         - avg_latency_us: Average latency in microseconds
    fn summary(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let summary = self.inner.summary();
        let dict = PyDict::new(py);
        dict.set_item("total_samples", summary.total_samples)?;
        dict.set_item("hub_nodes_capped", summary.hub_nodes_capped)?;
        dict.set_item("total_nodes_sampled", summary.total_nodes_sampled)?;
        dict.set_item("total_edges_sampled", summary.total_edges_sampled)?;
        dict.set_item("avg_latency_us", summary.avg_latency_us)?;
        Ok(dict.into())
    }

    /// Reset all counters to zero.
    fn reset(&self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        let summary = self.inner.summary();
        format!(
            "SamplingTelemetry(samples={}, hub_nodes={}, avg_latency={}µs)",
            summary.total_samples, summary.hub_nodes_capped, summary.avg_latency_us
        )
    }

    fn __str__(&self) -> String {
        self.inner.summary().to_string()
    }
}

/// Python wrapper for SamplingConfig.
#[pyclass(name = "SamplingConfig")]
#[derive(Clone)]
pub struct PySamplingConfig {
    inner: SamplingConfig,
}

#[pymethods]
impl PySamplingConfig {
    /// Create a new sampling configuration.
    ///
    /// Args:
    ///     num_neighbors: List of neighbor counts to sample per hop (e.g., [25, 10] for 2-hop sampling)
    ///     replace: Whether to sample with replacement (default: False)
    ///     seed: Random seed for reproducibility (default: None for random)
    ///     max_degree: Maximum degree cap for hub nodes (default: 10000)
    ///     cumulative: Whether to use cumulative sampling (PyG-style, default: True)
    ///         - True: Sample from all nodes seen so far at each hop (larger subgraphs)
    ///         - False: Sample only from new frontier at each hop (smaller subgraphs)
    ///     weighted: Whether to use edge weights for sampling (default: False)
    ///         When True, neighbors are sampled proportionally to their edge weights.
    ///     subgraph_type: Type of subgraph to extract (default: "directional")
    ///         - "directional": Return edges as sampled (default)
    ///         - "induced": Only edges where both endpoints are in sampled nodes
    ///         - "bidirectional": Add reverse edges for undirected graphs
    ///     track_edge_ids: Whether to track global edge IDs for e_id (default: True)
    ///         Set to False for ~10-15% speedup if you don't need edge features.
    ///     temporal_strategy: Temporal sampling strategy (default: None = disabled)
    ///         - "uniform": Sample uniformly from edges with timestamp < node time
    ///         - "last": Take the k most recent edges with timestamp < node time
    ///     disjoint: Whether to produce disjoint subgraphs per seed (default: False)
    ///         Each seed gets an isolated subgraph with no node dedup across seeds.
    ///     telemetry: Optional SamplingTelemetry for metrics collection (default: None)
    ///
    /// Returns:
    ///     SamplingConfig: Configuration object
    #[new]
    #[pyo3(signature = (num_neighbors, replace=false, seed=None, max_degree=None, cumulative=true, weighted=false, subgraph_type="directional", track_edge_ids=true, temporal_strategy=None, disjoint=false, telemetry=None))]
    #[allow(clippy::too_many_arguments)] // Python API is explicit and mirrors documented kwargs.
    fn new(
        num_neighbors: Vec<usize>,
        replace: bool,
        seed: Option<u64>,
        max_degree: Option<usize>,
        cumulative: bool,
        weighted: bool,
        subgraph_type: &str,
        track_edge_ids: bool,
        temporal_strategy: Option<&str>,
        disjoint: bool,
        telemetry: Option<PySamplingTelemetry>,
    ) -> PyResult<Self> {
        let subgraph_type = match subgraph_type {
            "directional" => SubgraphType::Directional,
            "induced" => SubgraphType::Induced,
            "bidirectional" => SubgraphType::Bidirectional,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid subgraph_type '{}'. Must be 'directional', 'induced', or 'bidirectional'",
                subgraph_type
            )))
            }
        };

        let temporal = match temporal_strategy {
            None => None,
            Some("uniform") => Some(TemporalStrategy::Uniform),
            Some("last") => Some(TemporalStrategy::Last),
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid temporal_strategy '{}'. Must be 'uniform' or 'last'",
                    other
                )))
            }
        };

        Ok(Self {
            inner: SamplingConfig {
                fanout: num_neighbors,
                replace,
                seed,
                max_degree,
                cumulative,
                weighted,
                subgraph_type,
                track_edge_ids,
                temporal_strategy: temporal,
                disjoint,
                telemetry: telemetry.map(|t| t.inner.clone()),
            },
        })
    }

    /// Get the num_neighbors configuration.
    #[getter]
    fn num_neighbors(&self) -> Vec<usize> {
        self.inner.fanout.clone()
    }

    /// Get the replace flag.
    #[getter]
    fn replace(&self) -> bool {
        self.inner.replace
    }

    /// Get the random seed.
    #[getter]
    fn seed(&self) -> Option<u64> {
        self.inner.seed
    }

    /// Get the max degree cap.
    #[getter]
    fn max_degree(&self) -> Option<usize> {
        self.inner.max_degree
    }

    /// Get the cumulative sampling flag.
    #[getter]
    fn cumulative(&self) -> bool {
        self.inner.cumulative
    }

    /// Get the weighted sampling flag.
    #[getter]
    fn weighted(&self) -> bool {
        self.inner.weighted
    }

    /// Get the subgraph type.
    #[getter]
    fn subgraph_type(&self) -> &'static str {
        match self.inner.subgraph_type {
            SubgraphType::Directional => "directional",
            SubgraphType::Induced => "induced",
            SubgraphType::Bidirectional => "bidirectional",
        }
    }

    /// Get the track_edge_ids flag.
    #[getter]
    fn track_edge_ids(&self) -> bool {
        self.inner.track_edge_ids
    }

    /// Get the temporal strategy.
    #[getter]
    fn temporal_strategy(&self) -> Option<&'static str> {
        self.inner.temporal_strategy.map(|s| match s {
            TemporalStrategy::Uniform => "uniform",
            TemporalStrategy::Last => "last",
        })
    }

    /// Get the disjoint flag.
    #[getter]
    fn disjoint(&self) -> bool {
        self.inner.disjoint
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplingConfig(num_neighbors={:?}, replace={}, seed={:?}, max_degree={:?}, cumulative={}, weighted={}, subgraph_type={:?}, track_edge_ids={}, temporal_strategy={:?}, disjoint={})",
            self.inner.fanout, self.inner.replace, self.inner.seed, self.inner.max_degree, self.inner.cumulative, self.inner.weighted, self.subgraph_type(), self.inner.track_edge_ids, self.temporal_strategy(), self.inner.disjoint
        )
    }
}

impl PySamplingConfig {
    /// Get a reference to the inner SamplingConfig (for use by other Rust modules).
    pub fn inner(&self) -> &SamplingConfig {
        &self.inner
    }
}

/// Python wrapper for SampledSubgraph.
///
/// ZERO-COPY: Arrays are int64 so PyTorch can use them directly without conversion.
#[pyclass(name = "SampledSubgraph")]
pub struct PySampledSubgraph {
    // Pre-computed numpy arrays as int64 (PyTorch's native index type)
    nodes: Py<PyArray1<i64>>,
    seeds: Py<PyArray1<i64>>,
    edge_index: Py<PyArray2<i64>>,
    // Pre-computed local edge index (remapped to [0, num_nodes))
    edge_index_local: Py<PyArray2<i64>>,
    // Global edge IDs (position in CSR edges array)
    edge_ids: Py<PyArray1<i64>>,
    // Local seed indices (position in sorted nodes array)
    seed_indices: Py<PyArray1<i64>>,
    // Batch vector (disjoint mode only): maps each node to its seed index
    batch_vec: Option<Py<PyArray1<i64>>>,
    num_nodes: usize,
    num_edges: usize,
    num_seeds: usize,
    // Per-hop sampling stats (for PyG compatibility)
    num_sampled_nodes: Vec<usize>,
    num_sampled_edges: Vec<usize>,
}

impl PySampledSubgraph {
    /// Create from SampledSubgraph, converting to int64 for PyTorch compatibility.
    /// Pre-computes local edge indices using binary search.
    pub fn from_subgraph(py: Python<'_>, mut subgraph: SampledSubgraph) -> PyResult<Self> {
        let num_nodes = subgraph.nodes.len();
        let num_edges = subgraph.edge_src.len();
        let num_seeds = subgraph.seeds.len();
        let num_sampled_nodes = subgraph.num_sampled_nodes.clone();
        let num_sampled_edges = subgraph.num_sampled_edges.clone();

        let is_disjoint = subgraph.batch.is_some();

        // Compute local edge indices in Rust (zero-copy move for disjoint precomputed)
        let (src_local, dst_local) = subgraph
            .edge_index_local()
            .map_err(|e| sampling_error(format!("Failed to compute local edge indices: {}", e)))?;

        // In disjoint mode, local_index is empty so seed_indices_local won't work.
        // Each seed is the first node of its per-seed subgraph — find their offsets.
        let seed_indices_local = if is_disjoint {
            // Seeds are at the start of each seed's block. batch vector tells us where.
            let batch_ref = subgraph.batch.as_ref().unwrap();
            let mut indices = Vec::with_capacity(subgraph.seeds.len());
            let mut current_seed = 0u32;
            for (i, &b) in batch_ref.iter().enumerate() {
                if b == current_seed {
                    indices.push(i as u32);
                    current_seed += 1;
                    if current_seed as usize >= subgraph.seeds.len() {
                        break;
                    }
                }
            }
            indices
        } else {
            subgraph
                .seed_indices_local()
                .map_err(|e| sampling_error(format!("Failed to compute seed indices: {}", e)))?
        };

        // Convert u32 to i64 - let LLVM auto-vectorize (it's very good at this)
        let nodes_i64: Vec<i64> = subgraph.nodes.into_iter().map(|n| n as i64).collect();
        let seeds_i64: Vec<i64> = subgraph.seeds.into_iter().map(|s| s as i64).collect();
        let seed_indices_i64: Vec<i64> = seed_indices_local.into_iter().map(|i| i as i64).collect();
        let edge_ids_i64: Vec<i64> = subgraph.edge_ids.into_iter().map(|e| e as i64).collect();

        // Global edge_index (original node IDs)
        let mut edge_data: Vec<i64> = Vec::with_capacity(num_edges * 2);
        edge_data.extend(subgraph.edge_src.into_iter().map(|e| e as i64));
        edge_data.extend(subgraph.edge_dst.into_iter().map(|e| e as i64));

        // Local edge_index (remapped to [0, num_nodes))
        let mut edge_data_local: Vec<i64> = Vec::with_capacity(num_edges * 2);
        edge_data_local.extend(src_local.into_iter().map(|e| e as i64));
        edge_data_local.extend(dst_local.into_iter().map(|e| e as i64));

        let nodes = PyArray1::from_vec(py, nodes_i64).unbind();
        let seeds = PyArray1::from_vec(py, seeds_i64).unbind();
        let seed_indices = PyArray1::from_vec(py, seed_indices_i64).unbind();
        let edge_ids = PyArray1::from_vec(py, edge_ids_i64).unbind();

        // Reshape to 2xN - this is just a view, no copy
        let edge_array = PyArray1::from_vec(py, edge_data);
        let edge_index = edge_array
            .reshape([2, num_edges])
            .map_err(|e| sampling_error(format!("Failed to reshape edge index: {}", e)))?
            .unbind();

        let edge_array_local = PyArray1::from_vec(py, edge_data_local);
        let edge_index_local = edge_array_local
            .reshape([2, num_edges])
            .map_err(|e| sampling_error(format!("Failed to reshape local edge index: {}", e)))?
            .unbind();

        // Build batch vector if present (disjoint mode)
        let batch_vec = subgraph.batch.map(|b| {
            let batch_i64: Vec<i64> = b.into_iter().map(|x| x as i64).collect();
            PyArray1::from_vec(py, batch_i64).unbind()
        });

        Ok(Self {
            nodes,
            seeds,
            edge_index,
            edge_index_local,
            edge_ids,
            seed_indices,
            batch_vec,
            num_nodes,
            num_edges,
            num_seeds,
            num_sampled_nodes,
            num_sampled_edges,
        })
    }
}

#[pymethods]
impl PySampledSubgraph {
    /// Returns the number of nodes in the subgraph.
    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of edges in the subgraph.
    fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Returns the number of seed nodes.
    fn num_seeds(&self) -> usize {
        self.num_seeds
    }

    /// Returns all node IDs in the subgraph as a numpy array (zero-copy).
    ///
    /// Returns:
    ///     numpy.ndarray: Array of node IDs (dtype=uint32)
    fn nodes(&self, py: Python<'_>) -> Py<PyAny> {
        self.nodes.clone_ref(py).into_any()
    }

    /// Returns all seed node IDs as a numpy array (zero-copy).
    ///
    /// Returns:
    ///     numpy.ndarray: Array of seed node IDs (dtype=uint32)
    fn seeds(&self, py: Python<'_>) -> Py<PyAny> {
        self.seeds.clone_ref(py).into_any()
    }

    /// Returns edge index in PyTorch Geometric COO format as a 2xN numpy array (zero-copy).
    /// Uses global node IDs.
    ///
    /// Returns:
    ///     numpy.ndarray: Edge index array with shape [2, num_edges] (dtype=int64)
    fn edge_index(&self, py: Python<'_>) -> Py<PyAny> {
        self.edge_index.clone_ref(py).into_any()
    }

    /// Returns edge index with local node IDs remapped to [0, num_nodes).
    /// Use this for PyG models to avoid OOM on large graphs.
    ///
    /// Returns:
    ///     numpy.ndarray: Edge index array with shape [2, num_edges] (dtype=int64)
    fn edge_index_local(&self, py: Python<'_>) -> Py<PyAny> {
        self.edge_index_local.clone_ref(py).into_any()
    }

    /// Returns global edge IDs (positions in the CSR edges array).
    /// Useful for looking up edge features or weights.
    ///
    /// Returns:
    ///     numpy.ndarray: Array of edge IDs (dtype=int64)
    fn edge_ids(&self, py: Python<'_>) -> Py<PyAny> {
        self.edge_ids.clone_ref(py).into_any()
    }

    /// Returns local indices of seed nodes in the sorted nodes array.
    ///
    /// Returns:
    ///     numpy.ndarray: Array of seed indices (dtype=int64)
    fn seed_indices(&self, py: Python<'_>) -> Py<PyAny> {
        self.seed_indices.clone_ref(py).into_any()
    }

    /// Returns the batch vector mapping each node to its seed index (disjoint mode only).
    ///
    /// Returns:
    ///     numpy.ndarray or None: Batch assignments (dtype=int64), or None if not disjoint.
    fn batch(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.batch_vec.as_ref().map(|b| b.clone_ref(py).into_any())
    }

    /// Returns the number of nodes sampled at each hop (for PyG compatibility).
    ///
    /// Returns:
    ///     List[int]: Number of new nodes sampled at each hop
    fn num_sampled_nodes_per_hop(&self) -> Vec<usize> {
        self.num_sampled_nodes.clone()
    }

    /// Returns the number of edges sampled at each hop (for PyG compatibility).
    ///
    /// Returns:
    ///     List[int]: Number of edges sampled at each hop
    fn num_sampled_edges_per_hop(&self) -> Vec<usize> {
        self.num_sampled_edges.clone()
    }

    /// Converts the subgraph to an Arrow RecordBatch for zero-copy data transfer.
    ///
    /// This is useful for integration with Ray Data and other Arrow-based systems.
    ///
    /// Returns:
    ///     pyarrow.RecordBatch: Arrow record batch with columns:
    ///         - edge_src: Source node IDs
    ///         - edge_dst: Destination node IDs
    ///         - nodes: All node IDs in subgraph
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Extract SOA edge arrays from edge_index (stored as i64, convert back to u32)
        let edge_index_bound = self.edge_index.bind(py);
        let edge_index_readonly = edge_index_bound.readonly();
        let flat_data = edge_index_readonly.as_slice()?;
        let num_edges = flat_data.len() / 2;

        let edge_src: Vec<u32> = flat_data[..num_edges].iter().map(|&x| x as u32).collect();
        let edge_dst: Vec<u32> = flat_data[num_edges..].iter().map(|&x| x as u32).collect();

        let nodes_bound = self.nodes.bind(py);
        let nodes_readonly = nodes_bound.readonly();
        let nodes_slice: Vec<u32> = nodes_readonly
            .as_slice()?
            .iter()
            .map(|&x| x as u32)
            .collect();

        let seeds_bound = self.seeds.bind(py);
        let seeds_readonly = seeds_bound.readonly();
        let seeds_slice: Vec<u32> = seeds_readonly
            .as_slice()?
            .iter()
            .map(|&x| x as u32)
            .collect();

        // Get edge_ids for temp subgraph
        let edge_ids_bound = self.edge_ids.bind(py);
        let edge_ids_readonly = edge_ids_bound.readonly();
        let edge_ids_slice: Vec<u64> = edge_ids_readonly
            .as_slice()?
            .iter()
            .map(|&x| x as u64)
            .collect();

        let temp_subgraph = SampledSubgraph::from_parts(
            nodes_slice,
            edge_src,
            edge_dst,
            edge_ids_slice,
            seeds_slice,
            self.num_sampled_nodes.clone(),
            self.num_sampled_edges.clone(),
        );

        let record_batch = subgraph_to_arrow(&temp_subgraph)?;

        // Convert to PyArrow object via Python
        // We need to use PyArrow's Python API since arrow-rs doesn't have direct PyO3 integration yet
        let pyarrow = py.import("pyarrow")?;

        // Create schema
        let schema = pyarrow.getattr("schema")?;
        let fields = vec![
            schema.call_method1("field", ("edge_src", pyarrow.getattr("uint32")?))?,
            schema.call_method1("field", ("edge_dst", pyarrow.getattr("uint32")?))?,
            schema.call_method1("field", ("nodes", pyarrow.getattr("uint32")?))?,
        ];
        let py_schema = schema.call1((fields,))?;

        // Create arrays
        let arrays = vec![
            self.arrow_array_to_pyarrow(py, &record_batch, 0)?,
            self.arrow_array_to_pyarrow(py, &record_batch, 1)?,
            self.arrow_array_to_pyarrow(py, &record_batch, 2)?,
        ];

        // Create RecordBatch
        let rb_class = pyarrow.getattr("RecordBatch")?;
        let py_record_batch = rb_class.call_method1("from_arrays", (arrays, py_schema))?;

        Ok(py_record_batch.unbind())
    }

    /// Returns a dictionary representation of the subgraph.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("num_nodes", self.num_nodes)?;
        dict.set_item("num_edges", self.num_edges)?;
        dict.set_item("nodes", self.nodes(py))?;
        dict.set_item("seeds", self.seeds(py))?;
        dict.set_item("edge_index", self.edge_index(py))?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "SampledSubgraph(num_nodes={}, num_edges={}, num_seeds={})",
            self.num_nodes, self.num_edges, self.num_seeds
        )
    }

    fn __str__(&self) -> String {
        format!(
            "Subgraph with {} nodes, {} edges (from {} seeds)",
            self.num_nodes, self.num_edges, self.num_seeds
        )
    }
}

impl PySampledSubgraph {
    /// Helper to convert Arrow array to PyArrow array
    fn arrow_array_to_pyarrow(
        &self,
        py: Python<'_>,
        batch: &RecordBatch,
        col_idx: usize,
    ) -> PyResult<Py<PyAny>> {
        let pyarrow = py.import("pyarrow")?;
        let array_class = pyarrow.getattr("array")?;

        let column = batch.column(col_idx);
        let uint32_array = column
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| sampling_error("Expected UInt32Array"))?;

        // Convert to Python list
        let values: Vec<u32> = uint32_array.values().to_vec();
        let py_list = PyList::new(py, &values)?;

        // Create PyArrow array from Python list
        let py_array = array_class.call1((py_list, pyarrow.getattr("uint32")?))?;

        Ok(py_array.unbind())
    }
}

/// Python wrapper for NeighborSampler.
#[pyclass(name = "NeighborSampler")]
pub struct PyNeighborSampler {
    graph: Py<PyCsrGraph>,
    config: PySamplingConfig,
}

#[pymethods]
impl PyNeighborSampler {
    /// Create a new neighbor sampler.
    ///
    /// Args:
    ///     graph: CsrGraph to sample from
    ///     config: SamplingConfig with num_neighbors, replace, and seed parameters
    ///
    /// Returns:
    ///     NeighborSampler: Sampler instance
    #[new]
    fn new(graph: Py<PyCsrGraph>, config: PySamplingConfig) -> Self {
        Self { graph, config }
    }

    /// Sample k-hop neighborhoods for a batch of seed nodes.
    ///
    /// Automatically routes to temporal/disjoint paths based on config.
    ///
    /// Args:
    ///     seeds: Seed node IDs as numpy array (int64) or list
    ///     input_times: Per-seed timestamps (float64 array), required if temporal_strategy is set
    ///
    /// Returns:
    ///     SampledSubgraph: Sampled subgraph containing nodes and edges
    #[pyo3(signature = (seeds, input_times=None))]
    fn sample(
        &mut self,
        py: Python<'_>,
        seeds: &Bound<'_, PyAny>,
        input_times: Option<numpy::PyReadonlyArray1<f64>>,
    ) -> PyResult<PySampledSubgraph> {
        let graph = self.graph.borrow(py);
        let mut sampler = NeighborSampler::new(graph.inner(), self.config.inner.clone());

        let seeds_vec: Vec<u32> = if let Ok(arr) = seeds.extract::<numpy::PyReadonlyArray1<i64>>() {
            arr.as_slice()?
                .iter()
                .map(|&x| {
                    u32::try_from(x).map_err(|_| {
                        sampling_error(format!("seed node {} out of range [0, {}]", x, u32::MAX))
                    })
                })
                .collect::<PyResult<Vec<u32>>>()?
        } else if let Ok(arr) = seeds.extract::<numpy::PyReadonlyArray1<u32>>() {
            arr.as_slice()?.to_vec()
        } else {
            seeds.extract::<Vec<u32>>()?
        };

        let subgraph = if self.config.inner.disjoint {
            let times = input_times
                .as_ref()
                .map(|t| t.as_slice())
                .transpose()?;
            sampler.sample_neighbors_disjoint(&seeds_vec, times)
        } else if self.config.inner.temporal_strategy.is_some() {
            let times = input_times
                .as_ref()
                .ok_or_else(|| sampling_error("temporal_strategy requires input_times"))?;
            sampler.sample_neighbors_temporal(&seeds_vec, times.as_slice()?)
        } else {
            sampler.sample_neighbors(&seeds_vec)
        };

        PySampledSubgraph::from_subgraph(py, subgraph)
    }

    fn __repr__(&self) -> String {
        format!(
            "NeighborSampler(num_neighbors={:?}, replace={})",
            self.config.inner.fanout, self.config.inner.replace
        )
    }
}

/// Python wrapper for ParallelBatchSampler.
#[pyclass(name = "ParallelBatchSampler")]
pub struct PyParallelBatchSampler {
    graph: Py<PyCsrGraph>,
    config: PySamplingConfig,
}

#[pymethods]
impl PyParallelBatchSampler {
    /// Create a new parallel batch sampler.
    ///
    /// Args:
    ///     graph: CsrGraph to sample from
    ///     config: SamplingConfig with num_neighbors, replace, and seed parameters
    ///
    /// Returns:
    ///     ParallelBatchSampler: Parallel sampler instance
    #[new]
    fn new(graph: Py<PyCsrGraph>, config: PySamplingConfig) -> Self {
        Self { graph, config }
    }

    /// Sample neighborhoods for multiple batches in parallel.
    ///
    /// Args:
    ///     batches: List of seed node ID lists, one per batch
    ///
    /// Returns:
    ///     List[SampledSubgraph]: List of sampled subgraphs, one per batch
    fn sample_batches(
        &self,
        py: Python<'_>,
        batches: Vec<Vec<u32>>,
    ) -> PyResult<Vec<PySampledSubgraph>> {
        let graph = self.graph.borrow(py);
        let sampler = ParallelBatchSampler::new(graph.inner(), self.config.inner.clone());
        let subgraphs = sampler.sample_batches(&batches);

        subgraphs
            .into_iter()
            .map(|sg| PySampledSubgraph::from_subgraph(py, sg))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ParallelBatchSampler(num_neighbors={:?}, replace={})",
            self.config.inner.fanout, self.config.inner.replace
        )
    }
}
