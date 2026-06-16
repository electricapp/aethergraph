use aethergraph_core::{
    NeighborSampler, ParallelBatchSampler, SampledSubgraph, SamplingConfig, SamplingTelemetry,
    SubgraphType, TemporalStrategy,
};
use arrow_array::{RecordBatch, UInt32Array};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

use crate::error::sampling_error;
use crate::graph::PyCsrGraph;

/// Lightweight telemetry for sampling operations.
///
/// Thread-safe, lock-free metrics collection with zero overhead.
/// Call `summary()` to get current metrics on-demand.
#[pyclass(name = "SamplingTelemetry", from_py_object)]
#[derive(Clone)]
pub struct PySamplingTelemetry {
    pub(crate) inner: Arc<SamplingTelemetry>,
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
#[pyclass(name = "SamplingConfig", from_py_object)]
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
    #[pyo3(signature = (num_neighbors, replace=false, seed=None, max_degree=None, cumulative=true, weighted=false, subgraph_type="directional", track_edge_ids=true, temporal_strategy=None, disjoint=false, deterministic=false, telemetry=None))]
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
        deterministic: bool,
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
                )));
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
                )));
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
                deterministic,
                telemetry: telemetry.map(|t| t.inner.clone()),
            },
        })
    }

    /// Bit-deterministic mode. When True, parallel sampling paths
    /// serialize so the same seed produces byte-identical output across
    /// runs / machines / core counts. Default False.
    #[getter]
    fn deterministic(&self) -> bool {
        self.inner.deterministic
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
            self.inner.fanout,
            self.inner.replace,
            self.inner.seed,
            self.inner.max_degree,
            self.inner.cumulative,
            self.inner.weighted,
            self.subgraph_type(),
            self.inner.track_edge_ids,
            self.temporal_strategy(),
            self.inner.disjoint
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
/// Two storage layers:
/// - `*_i64` numpy arrays — what Python sees on `.nodes()`, `.edges()` etc.
///   PyTorch's `Tensor` indexing uses `int64`, so we widen once at conversion
///   time and never narrow again on the Python-facing path.
/// - `original_*` `Arc<[u32]>` — the canonical Rust-side data, kept so that
///   `to_arrow()` and other consumers don't have to round-trip i64→u32 (which
///   in addition to being wasteful would lose the bounds check that node IDs
///   actually fit in u32).
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
    // Canonical u32 buffers — Arc so to_arrow() etc. can hand them to Arrow
    // (which takes ownership of a Vec) without an additional clone.
    original_nodes: Arc<[u32]>,
    original_seeds: Arc<[u32]>,
    original_edge_src: Arc<[u32]>,
    original_edge_dst: Arc<[u32]>,
    original_edge_ids: Arc<[u64]>,
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

        // In disjoint mode `local_index` is intentionally empty (each seed has
        // its own remap) so `seed_indices_local()` cannot be used. The seeds
        // appear in the global `nodes` array in the same order they were
        // submitted, with each seed at the start of its per-seed block. We
        // reconstruct the seed positions by walking `batch_vec` and recording
        // the index of the first occurrence of seed 0, 1, 2, ...
        //
        // Invariant assumed: `batch_vec` is monotonic non-decreasing — once we
        // see seed `k`, we never see seed `k-1` again. The core sampler
        // guarantees this. We `debug_assert!` it here for catch-on-test if it
        // ever drifts; in release we trust the producer.
        let seed_indices_local = if is_disjoint {
            let batch_ref = subgraph
                .batch
                .as_ref()
                .expect("disjoint mode without batch vector — core sampler bug");
            debug_assert!(
                batch_ref.windows(2).all(|w| w[0] <= w[1]),
                "batch vector must be monotonic non-decreasing"
            );
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
            if indices.len() != subgraph.seeds.len() {
                return Err(sampling_error(format!(
                    "disjoint mode: found {} seed offsets in batch vector, expected {}",
                    indices.len(),
                    subgraph.seeds.len()
                )));
            }
            indices
        } else {
            subgraph
                .seed_indices_local()
                .map_err(|e| sampling_error(format!("Failed to compute seed indices: {}", e)))?
        };

        // Cache canonical u32 buffers before widening into the Python-facing
        // i64 arrays so `to_arrow()` and similar consumers can hand them
        // straight to Arrow without an i64→u32 narrowing pass.
        let original_nodes: Arc<[u32]> = subgraph.nodes.clone().into();
        let original_seeds: Arc<[u32]> = subgraph.seeds.clone().into();
        let original_edge_src: Arc<[u32]> = subgraph.edge_src.clone().into();
        let original_edge_dst: Arc<[u32]> = subgraph.edge_dst.clone().into();
        let original_edge_ids: Arc<[u64]> = subgraph.edge_ids.clone().into();

        // u32 → i64 widening, one pass per buffer. The cast is monotonic and
        // free of branches; whether LLVM emits a vectorized form
        // (vpmovzxdq on x86_64 AVX2 / uxtl on aarch64 NEON) is target- and
        // codegen-flag-dependent and has NOT been audited here.
        // TODO(perf): pin this in CI with a `cargo asm` snapshot job that
        // fails the build if the widening lowers to a scalar loop on
        // either target.
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

        // Reshape to 2xN — `numpy` ndarray reshape is a view, not a copy.
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
            original_nodes,
            original_seeds,
            original_edge_src,
            original_edge_dst,
            original_edge_ids,
        })
    }
}

#[pymethods]
impl PySampledSubgraph {
    /// Number of unique nodes in the subgraph.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Number of edges in the subgraph.
    #[getter]
    fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Number of seed nodes used to produce this subgraph.
    #[getter]
    fn num_seeds(&self) -> usize {
        self.num_seeds
    }

    /// All node IDs in the subgraph as a numpy `int64` array (zero-copy view).
    /// `int64` matches PyTorch's `Tensor` index dtype.
    #[getter]
    fn nodes(&self, py: Python<'_>) -> Py<PyAny> {
        self.nodes.clone_ref(py).into_any()
    }

    /// Seed node IDs as a numpy `int64` array (zero-copy view).
    #[getter]
    fn seeds(&self, py: Python<'_>) -> Py<PyAny> {
        self.seeds.clone_ref(py).into_any()
    }

    /// Edge index in PyTorch Geometric COO format — shape `[2, num_edges]`,
    /// dtype `int64`. Global node IDs.
    #[getter]
    fn edge_index(&self, py: Python<'_>) -> Py<PyAny> {
        self.edge_index.clone_ref(py).into_any()
    }

    /// Edge index with local IDs remapped to `[0, num_nodes)` — shape
    /// `[2, num_edges]`, dtype `int64`. Use this for PyG models to avoid OOM
    /// on large graphs.
    #[getter]
    fn edge_index_local(&self, py: Python<'_>) -> Py<PyAny> {
        self.edge_index_local.clone_ref(py).into_any()
    }

    /// Global edge IDs (positions in the CSR edges array) — dtype `int64`.
    /// Use for looking up edge features or weights.
    #[getter]
    fn edge_ids(&self, py: Python<'_>) -> Py<PyAny> {
        self.edge_ids.clone_ref(py).into_any()
    }

    /// Local indices of seed nodes in the sorted `nodes` array — dtype `int64`.
    #[getter]
    fn seed_indices(&self, py: Python<'_>) -> Py<PyAny> {
        self.seed_indices.clone_ref(py).into_any()
    }

    /// Batch vector mapping each node to its seed index (disjoint mode only),
    /// or `None` when sampling was non-disjoint. dtype `int64`.
    #[getter]
    fn batch(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.batch_vec.as_ref().map(|b| b.clone_ref(py).into_any())
    }

    /// Number of new nodes sampled at each hop (PyG-compatible).
    #[getter]
    fn num_sampled_nodes_per_hop(&self) -> Vec<usize> {
        self.num_sampled_nodes.clone()
    }

    /// Number of edges sampled at each hop (PyG-compatible).
    #[getter]
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
    /// Convert this subgraph to a dict of PyArrow `RecordBatch`es:
    /// ```python
    /// {"edges": RecordBatch(edge_src, edge_dst, edge_id),
    ///  "nodes": RecordBatch(nodes),
    ///  "seeds": RecordBatch(seeds)}
    /// ```
    ///
    /// Three batches rather than one because Arrow `RecordBatch` requires
    /// equal-length columns, and the edge / node / seed arrays have different
    /// lengths.
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let temp_subgraph = SampledSubgraph::from_parts(
            self.original_nodes.to_vec(),
            self.original_edge_src.to_vec(),
            self.original_edge_dst.to_vec(),
            self.original_edge_ids.to_vec(),
            self.original_seeds.to_vec(),
            self.num_sampled_nodes.clone(),
            self.num_sampled_edges.clone(),
        );

        let batches = crate::arrow_utils::subgraph_into_arrow(temp_subgraph)?;

        let pyarrow = py.import("pyarrow")?;
        let rb_class = pyarrow.getattr("RecordBatch")?;
        let schema_fn = pyarrow.getattr("schema")?;
        let uint32_dt = pyarrow.getattr("uint32")?;

        let edges_py = build_py_record_batch(
            py,
            &pyarrow,
            &rb_class,
            &schema_fn,
            &uint32_dt,
            &batches.edges,
            &["edge_src", "edge_dst", "edge_id"],
        )?;
        let nodes_py = build_py_record_batch(
            py,
            &pyarrow,
            &rb_class,
            &schema_fn,
            &uint32_dt,
            &batches.nodes,
            &["nodes"],
        )?;
        let seeds_py = build_py_record_batch(
            py,
            &pyarrow,
            &rb_class,
            &schema_fn,
            &uint32_dt,
            &batches.seeds,
            &["seeds"],
        )?;

        let dict = PyDict::new(py);
        dict.set_item("edges", edges_py)?;
        dict.set_item("nodes", nodes_py)?;
        dict.set_item("seeds", seeds_py)?;
        Ok(dict.into())
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
        self.__repr__()
    }

    /// `len(subgraph)` returns the number of nodes (matches PyG convention).
    fn __len__(&self) -> usize {
        self.num_nodes
    }
}

/// Convert a Rust Arrow `RecordBatch` into a `pyarrow.RecordBatch`.
///
/// arrow-rs and pyarrow share a binary buffer layout but no stable PyO3
/// bridge, so we round-trip column buffers through Python lists. This is a
/// per-call O(n) conversion (one Python list element per value) and is meant
/// for epoch / batch boundaries, not a per-batch hot path — Python-side Arrow
/// consumers (Ray Data, etc.) consume these infrequently. A zero-copy Arrow C
/// Data Interface path would remove the element-by-element conversion.
fn build_py_record_batch(
    py: Python<'_>,
    pyarrow: &Bound<'_, pyo3::types::PyModule>,
    rb_class: &Bound<'_, PyAny>,
    schema_fn: &Bound<'_, PyAny>,
    uint32_dt: &Bound<'_, PyAny>,
    batch: &RecordBatch,
    field_names: &[&str],
) -> PyResult<Py<PyAny>> {
    debug_assert_eq!(field_names.len(), batch.num_columns());
    let array_class = pyarrow.getattr("array")?;

    let mut arrays = Vec::with_capacity(batch.num_columns());
    let mut fields = Vec::with_capacity(batch.num_columns());
    for (i, name) in field_names.iter().enumerate() {
        let column = batch.column(i);
        let uint32_array = column
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| {
                sampling_error(format!(
                    "expected UInt32Array at column {i} ({name}), got {:?}",
                    column.data_type()
                ))
            })?;
        let values: Vec<u32> = uint32_array.values().to_vec();
        let py_list = PyList::new(py, &values)?;
        let py_array = array_class.call1((py_list, uint32_dt.clone()))?;
        arrays.push(py_array);
        fields.push(schema_fn.call_method1("field", (*name, uint32_dt.clone()))?);
    }

    let py_schema = schema_fn.call1((fields,))?;
    let py_record_batch = rb_class.call_method1("from_arrays", (arrays, py_schema))?;
    Ok(py_record_batch.unbind())
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
        // Pull everything out of Python objects up front: the seeds, the owned
        // graph handle, and the per-seed timestamps. Once these are plain Rust
        // data we can run the sampler with the GIL released so background
        // Python threads keep running.
        let seeds_vec: Vec<u32> = crate::error::extract_seeds(seeds)?;
        let graph_arc = self.graph.borrow(py).inner_arc();
        let times_vec: Option<Vec<f64>> = input_times
            .as_ref()
            .map(|t| t.as_slice().map(|s| s.to_vec()))
            .transpose()?;
        let config = self.config.inner.clone();
        let disjoint = config.disjoint;
        let temporal = config.temporal_strategy.is_some();

        // Heavy sampling runs without the GIL. No Python object is touched
        // inside this closure — it works purely on owned Rust data and returns
        // a plain `SampledSubgraph`.
        let subgraph = py.detach(move || -> PyResult<SampledSubgraph> {
            let mut sampler = NeighborSampler::new(&graph_arc, config);
            let subgraph = if disjoint {
                sampler.sample_neighbors_disjoint(&seeds_vec, times_vec.as_deref())
            } else if temporal {
                let times = times_vec
                    .as_deref()
                    .ok_or_else(|| sampling_error("temporal_strategy requires input_times"))?;
                sampler
                    .sample_neighbors_temporal(&seeds_vec, times)
                    .map_err(|e| sampling_error(e.to_string()))?
            } else {
                sampler.sample_neighbors(&seeds_vec)
            };
            Ok(subgraph)
        })?;

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
        // Capture an owned graph handle and config so the parallel sampling
        // sweep can run with the GIL released.
        let graph_arc = self.graph.borrow(py).inner_arc();
        let config = self.config.inner.clone();

        // The rayon-parallel batch sweep touches no Python state; run it
        // detached and convert the plain `SampledSubgraph`s afterward.
        let subgraphs = py.detach(move || {
            let sampler = ParallelBatchSampler::new(&graph_arc, config);
            sampler.sample_batches(&batches)
        });

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
