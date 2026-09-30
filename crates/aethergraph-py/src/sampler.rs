use aethergraph_core::{
    Graph, NeighborSampler, ParallelBatchSampler, SampledSubgraph, SamplingConfig,
    SamplingTelemetry, Seeds, SubgraphType, TemporalStrategy,
};
use arrow_array::{RecordBatch, UInt32Array, UInt64Array};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;
use std::sync::OnceLock;

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
    /// This constructor is the one runtime check for every field; the
    /// Python `aethergraph.SamplingConfig` dataclass builds one to validate.
    ///
    /// Args:
    ///     num_neighbors: List of neighbor counts to sample per hop (e.g., [25, 10] for 2-hop sampling)
    ///     replace: Whether to sample with replacement (default: False)
    ///     seed: Random seed for reproducibility (default: None for random)
    ///     max_degree: Degree above which a node counts as a hub (default:
    ///         None). Uniform draws never cap; weighted and uniform-temporal
    ///         sampling draw from `max_degree` random positions of a hub's row.
    ///     cumulative: Whether every hop re-expands all nodes seen so far
    ///         (default: False, PyG semantics: each hop expands only the
    ///         previous hop's new nodes). No edge is emitted twice either way.
    ///     weighted: Whether to use edge weights for sampling (default: False)
    ///         When True, neighbors are sampled proportionally to their edge weights.
    ///     subgraph_type: Type of subgraph to extract (default: "directional")
    ///         - "directional": Edges exactly as sampled
    ///         - "induced": Every graph edge between sampled nodes
    ///         - "bidirectional": Sampled edges plus reverses, each ordered pair once
    ///     track_edge_ids: Whether to track global edge IDs for e_id (default: True)
    ///         Set to False for ~10-15% speedup if you don't need edge features.
    ///     temporal_strategy: Temporal sampling strategy (default: None = disabled)
    ///         - "uniform": Sample uniformly from edges with timestamp < node time
    ///         - "last": Take the k most recent edges with timestamp < node time
    ///     disjoint: Whether to produce disjoint subgraphs per seed (default: False)
    ///         Each seed gets an isolated subgraph with no node dedup across seeds.
    ///     deterministic: Has no effect; a fixed `seed` already gives
    ///         bit-identical output at any thread count.
    ///     telemetry: Optional SamplingTelemetry for metrics collection (default: None)
    ///
    /// Returns:
    ///     SamplingConfig: Configuration object
    #[new]
    #[pyo3(signature = (num_neighbors, replace=false, seed=None, max_degree=None, cumulative=false, weighted=false, subgraph_type="directional", track_edge_ids=true, temporal_strategy=None, disjoint=false, deterministic=false, telemetry=None))]
    #[allow(clippy::too_many_arguments)] // Python API is explicit and mirrors documented kwargs.
    fn new(
        num_neighbors: Vec<i64>,
        replace: bool,
        seed: Option<u64>,
        max_degree: Option<i64>,
        cumulative: bool,
        weighted: bool,
        subgraph_type: &str,
        track_edge_ids: bool,
        temporal_strategy: Option<&str>,
        disjoint: bool,
        deterministic: bool,
        telemetry: Option<PySamplingTelemetry>,
    ) -> PyResult<Self> {
        if num_neighbors.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_neighbors must be a non-empty list",
            ));
        }
        let fanout = num_neighbors
            .iter()
            .map(|&n| usize::try_from(n))
            .collect::<Result<Vec<usize>, _>>()
            .map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "num_neighbors values must be non-negative, got {num_neighbors:?}"
                ))
            })?;
        let max_degree = match max_degree {
            None => None,
            Some(d) if d > 0 => Some(d as usize),
            Some(d) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "max_degree must be > 0 if specified, got {d}"
                )));
            }
        };
        let subgraph_type = match subgraph_type {
            "directional" => SubgraphType::Directional,
            "induced" => SubgraphType::Induced,
            "bidirectional" => SubgraphType::Bidirectional,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid subgraph_type '{subgraph_type}'. Must be 'directional', 'induced', or 'bidirectional'"
                )));
            }
        };

        let temporal = match temporal_strategy {
            None => None,
            Some("uniform") => Some(TemporalStrategy::Uniform),
            Some("last") => Some(TemporalStrategy::Last),
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid temporal_strategy '{other}'. Must be 'uniform' or 'last'"
                )));
            }
        };

        Ok(Self {
            inner: SamplingConfig {
                fanout,
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
                telemetry: telemetry.map(|t| t.inner),
            },
        })
    }

    /// The `deterministic` flag as given. It has no effect: a fixed `seed`
    /// already produces byte-identical output across runs, machines, and
    /// thread counts.
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

/// A sampled subgraph with every O(N + E) conversion already done: IDs
/// widened to `int64` (PyTorch's index dtype) and the local edge index laid
/// out in message-passing order.
///
/// Building one touches no Python state, so callers build it with the GIL
/// released; [`PySampledSubgraph::from_prepared`] then only wraps the
/// vectors, which `PyArray1::from_vec` does without copying.
pub struct PreparedSubgraph {
    nodes: Vec<i64>,
    seeds: Vec<i64>,
    seed_indices: Vec<i64>,
    edge_ids: Vec<i64>,
    /// `[2, num_edges]` row-major: sampled neighbors, then the nodes that
    /// sampled them.
    edge_index_local: Vec<i64>,
    batch: Option<Vec<i64>>,
    num_edges: usize,
    num_sampled_nodes: Vec<usize>,
    num_sampled_edges: Vec<usize>,
    original_nodes: Vec<u32>,
    original_seeds: Vec<u32>,
    original_edge_src: Vec<u32>,
    original_edge_dst: Vec<u32>,
    original_edge_ids: Vec<u64>,
}

impl PreparedSubgraph {
    /// Widen and reorient `subgraph`. The core records edges in stored
    /// direction (`src` expanded, `dst` drawn from its row); Python sees
    /// PyG's source-to-target order, so the rows swap here.
    ///
    /// # Errors
    /// Only for `SampledSubgraph::from_parts` input whose edge endpoints or
    /// seeds are missing from its node list.
    pub fn new(subgraph: SampledSubgraph) -> Result<Self, String> {
        let num_edges = subgraph.edge_src.len();

        // u32 → i64 widening, one pass per buffer. The cast is monotonic and
        // free of branches, so it vectorizes: on aarch64 the release build
        // emits ushll.2d/ushll2.2d, widening eight lanes per unrolled
        // iteration.
        let mut edge_index_local: Vec<i64> = Vec::with_capacity(num_edges * 2);
        let seed_indices: Vec<i64> = {
            let (src_local, dst_local) = subgraph
                .edge_index_local()
                .map_err(|e| format!("failed to compute local edge indices: {e}"))?;
            edge_index_local.extend(dst_local.iter().map(|&e| i64::from(e)));
            edge_index_local.extend(src_local.iter().map(|&e| i64::from(e)));
            subgraph
                .seed_indices_local()
                .map_err(|e| format!("failed to compute seed indices: {e}"))?
                .iter()
                .map(|&e| i64::from(e))
                .collect()
        };

        Ok(Self {
            nodes: subgraph.nodes.iter().map(|&n| i64::from(n)).collect(),
            seeds: subgraph.seeds.iter().map(|&s| i64::from(s)).collect(),
            seed_indices,
            edge_ids: subgraph.edge_ids.iter().map(|&e| e as i64).collect(),
            edge_index_local,
            batch: subgraph
                .batch
                .map(|b| b.into_iter().map(i64::from).collect()),
            num_edges,
            num_sampled_nodes: subgraph.num_sampled_nodes,
            num_sampled_edges: subgraph.num_sampled_edges,
            // Move (not copy) the canonical buffers out of the core subgraph.
            original_nodes: subgraph.nodes,
            original_seeds: subgraph.seeds,
            original_edge_src: subgraph.edge_src,
            original_edge_dst: subgraph.edge_dst,
            original_edge_ids: subgraph.edge_ids,
        })
    }
}

/// Python wrapper for SampledSubgraph.
///
/// Edge arrays follow PyG's source-to-target convention: row 0 of
/// `edge_index` is the sampled neighbor (message source) and row 1 the node
/// whose neighborhood was expanded, so messages flow toward the seeds.
/// `edge_ids` name the stored edge from row 1 to row 0.
///
/// Two storage layers:
/// - `*_i64` numpy arrays — what Python sees on `.nodes()`, `.edges()` etc.
///   PyTorch's `Tensor` indexing uses `int64`, so we widen once at conversion
///   time and never narrow again on the Python-facing path.
/// - `original_*` `Vec` buffers — the canonical Rust-side data, moved (not
///   copied) out of the sampled subgraph so that `to_arrow()` and other
///   consumers don't have to round-trip i64→u32 (which in addition to being
///   wasteful would lose the bounds check that node IDs actually fit in u32).
///   They are copied on demand only when such a consumer asks.
#[pyclass(name = "SampledSubgraph")]
pub struct PySampledSubgraph {
    // Pre-computed numpy arrays as int64 (PyTorch's native index type)
    nodes: Py<PyArray1<i64>>,
    seeds: Py<PyArray1<i64>>,
    // Global-ID edge index, built lazily on first access: the standard
    // training path consumes only `edge_index_local`, so the eager 2E x 8B
    // widening pass and numpy allocation would be pure per-batch waste.
    edge_index: OnceLock<Py<PyArray2<i64>>>,
    // Pre-computed local edge index (remapped to [0, num_nodes))
    edge_index_local: Py<PyArray2<i64>>,
    // Global edge IDs (position in CSR edges array)
    edge_ids: Py<PyArray1<i64>>,
    // Local seed indices (position in the discovery-order nodes array)
    seed_indices: Py<PyArray1<i64>>,
    // Batch vector (disjoint mode only): maps each node to its seed index
    batch_vec: Option<Py<PyArray1<i64>>>,
    num_nodes: usize,
    num_edges: usize,
    num_seeds: usize,
    // Per-hop sampling stats (for PyG compatibility)
    num_sampled_nodes: Vec<usize>,
    num_sampled_edges: Vec<usize>,
    // Canonical u32/u64 buffers, moved out of the core subgraph at
    // construction; `to_arrow()` and the lazy `edge_index` build read from
    // them on demand.
    original_nodes: Vec<u32>,
    original_seeds: Vec<u32>,
    original_edge_src: Vec<u32>,
    original_edge_dst: Vec<u32>,
    original_edge_ids: Vec<u64>,
}

impl PySampledSubgraph {
    /// Convert a core subgraph with the GIL held. Callers that can release
    /// the GIL build a [`PreparedSubgraph`] inside `py.detach` and call
    /// [`Self::from_prepared`] instead.
    pub fn from_subgraph(py: Python<'_>, subgraph: SampledSubgraph) -> PyResult<Self> {
        let prepared = PreparedSubgraph::new(subgraph).map_err(sampling_error)?;
        Self::from_prepared(py, prepared)
    }

    /// Wrap already-converted buffers as numpy arrays — O(1) per array, no
    /// element is touched.
    pub fn from_prepared(py: Python<'_>, prepared: PreparedSubgraph) -> PyResult<Self> {
        let num_edges = prepared.num_edges;
        let edge_index_local = PyArray1::from_vec(py, prepared.edge_index_local)
            .reshape([2, num_edges])
            .map_err(|e| sampling_error(format!("Failed to reshape local edge index: {e}")))?
            .unbind();
        Ok(Self {
            num_nodes: prepared.nodes.len(),
            num_seeds: prepared.seeds.len(),
            num_edges,
            nodes: PyArray1::from_vec(py, prepared.nodes).unbind(),
            seeds: PyArray1::from_vec(py, prepared.seeds).unbind(),
            edge_index: OnceLock::new(),
            edge_index_local,
            edge_ids: PyArray1::from_vec(py, prepared.edge_ids).unbind(),
            seed_indices: PyArray1::from_vec(py, prepared.seed_indices).unbind(),
            batch_vec: prepared.batch.map(|b| PyArray1::from_vec(py, b).unbind()),
            num_sampled_nodes: prepared.num_sampled_nodes,
            num_sampled_edges: prepared.num_sampled_edges,
            original_nodes: prepared.original_nodes,
            original_seeds: prepared.original_seeds,
            original_edge_src: prepared.original_edge_src,
            original_edge_dst: prepared.original_edge_dst,
            original_edge_ids: prepared.original_edge_ids,
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

    /// All node IDs in the subgraph as a numpy `int64` array. `int64`
    /// matches PyTorch's `Tensor` index dtype. Returns the same cached
    /// Python-owned array object on every access; it never aliases Rust
    /// memory — treat it as immutable.
    #[getter]
    fn nodes(&self, py: Python<'_>) -> Py<PyAny> {
        self.nodes.clone_ref(py).into_any()
    }

    /// Seed node IDs as a numpy `int64` array. Returns the same cached
    /// Python-owned array object on every access; it never aliases Rust
    /// memory — treat it as immutable.
    #[getter]
    fn seeds(&self, py: Python<'_>) -> Py<PyAny> {
        self.seeds.clone_ref(py).into_any()
    }

    /// Edge index in PyTorch Geometric COO format — shape `[2, num_edges]`,
    /// dtype `int64`. Global node IDs; row 0 is the sampled neighbor, row 1
    /// the node that sampled it. Built on first access and cached — the
    /// standard training path only touches `edge_index_local`, so the
    /// widening pass would otherwise be paid on every batch for nothing.
    #[getter]
    fn edge_index(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(cached) = self.edge_index.get() {
            return Ok(cached.clone_ref(py).into_any());
        }
        let mut edge_data: Vec<i64> = Vec::with_capacity(self.num_edges * 2);
        edge_data.extend(self.original_edge_dst.iter().map(|&e| i64::from(e)));
        edge_data.extend(self.original_edge_src.iter().map(|&e| i64::from(e)));
        let arr = PyArray1::from_vec(py, edge_data)
            .reshape([2, self.num_edges])
            .map_err(|e| sampling_error(format!("Failed to reshape edge index: {e}")))?
            .unbind();
        let _ = self.edge_index.set(arr);
        Ok(self
            .edge_index
            .get()
            .expect("edge_index cache initialized above")
            .clone_ref(py)
            .into_any())
    }

    /// Edge index with local IDs remapped to `[0, num_nodes)` — shape
    /// `[2, num_edges]`, dtype `int64`, same orientation as `edge_index`.
    /// Use this for PyG models to avoid OOM on large graphs.
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

    /// Local indices of seed nodes in the `nodes` array — dtype `int64`.
    /// Returns the same cached Python-owned array object on every access;
    /// treat it as immutable.
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

    /// Nodes each stage added, in PyG's `num_sampled_nodes` layout:
    /// `[seed nodes, hop 1, ..., hop k]`.
    #[getter]
    fn num_sampled_nodes_per_hop(&self) -> Vec<usize> {
        self.num_sampled_nodes.clone()
    }

    /// Edges each hop sampled, one entry per hop (PyG's
    /// `num_sampled_edges`). Induced and bidirectional subgraphs rewrite the
    /// edge arrays afterwards; these counts describe the sampling pass.
    #[getter]
    fn num_sampled_edges_per_hop(&self) -> Vec<usize> {
        self.num_sampled_edges.clone()
    }

    /// Convert this subgraph to a dict of PyArrow `RecordBatch`es:
    /// ```python
    /// {"edges": RecordBatch(edge_src, edge_dst, edge_id),
    ///  "nodes": RecordBatch(nodes),
    ///  "seeds": RecordBatch(seeds)}
    /// ```
    ///
    /// `edge_src`/`edge_dst` follow `edge_index` (source = sampled
    /// neighbor); `edge_id` is `uint64`, node columns `uint32`.
    ///
    /// Three batches rather than one because Arrow `RecordBatch` requires
    /// equal-length columns, and the edge / node / seed arrays have different
    /// lengths.
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let temp_subgraph = SampledSubgraph::from_parts(
            self.original_nodes.clone(),
            self.original_edge_dst.clone(),
            self.original_edge_src.clone(),
            self.original_edge_ids.clone(),
            self.original_seeds.clone(),
            self.num_sampled_nodes.clone(),
            self.num_sampled_edges.clone(),
        );

        let batches = crate::arrow_utils::subgraph_into_arrow(temp_subgraph)?;

        let pyarrow = py.import("pyarrow")?;
        let edges_py = build_py_record_batch(&pyarrow, &batches.edges)?;
        let nodes_py = build_py_record_batch(&pyarrow, &batches.nodes)?;
        let seeds_py = build_py_record_batch(&pyarrow, &batches.seeds)?;

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
        dict.set_item("edge_index", self.edge_index(py)?)?;
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

/// Convert a Rust Arrow `RecordBatch` into a `pyarrow.RecordBatch`, keeping
/// its column names and its `UInt32`/`UInt64` column types.
///
/// arrow-rs and pyarrow share a binary buffer layout but no stable PyO3
/// bridge, so we round-trip column buffers through Python lists. This is a
/// per-call O(n) conversion (one Python list element per value) and is meant
/// for epoch / batch boundaries, not a per-batch hot path — Python-side Arrow
/// consumers (Ray Data, etc.) consume these infrequently. A zero-copy Arrow C
/// Data Interface path would remove the element-by-element conversion.
fn build_py_record_batch(
    pyarrow: &Bound<'_, pyo3::types::PyModule>,
    batch: &RecordBatch,
) -> PyResult<Py<PyAny>> {
    let py = pyarrow.py();
    let array_class = pyarrow.getattr("array")?;
    let schema_fn = pyarrow.getattr("schema")?;
    let field_fn = pyarrow.getattr("field")?;

    let schema = batch.schema();
    let mut arrays = Vec::with_capacity(batch.num_columns());
    let mut fields = Vec::with_capacity(batch.num_columns());
    for (field, column) in schema.fields().iter().zip(batch.columns()) {
        let name = field.name();
        let any = column.as_any();
        // `pyarrow.uint32` is a factory; the DataType is its return value.
        let (values, dtype) = if let Some(a) = any.downcast_ref::<UInt32Array>() {
            (
                PyList::new(py, a.values().iter())?,
                pyarrow.getattr("uint32")?.call0()?,
            )
        } else if let Some(a) = any.downcast_ref::<UInt64Array>() {
            (
                PyList::new(py, a.values().iter())?,
                pyarrow.getattr("uint64")?.call0()?,
            )
        } else {
            return Err(sampling_error(format!(
                "unsupported Arrow column {name}: {:?}",
                column.data_type()
            )));
        };
        arrays.push(array_class.call1((values, dtype.clone()))?);
        fields.push(field_fn.call1((name, dtype))?);
    }

    let kwargs = PyDict::new(py);
    kwargs.set_item("schema", schema_fn.call1((fields,))?)?;
    let py_record_batch =
        pyarrow
            .getattr("RecordBatch")?
            .call_method("from_arrays", (arrays,), Some(&kwargs))?;
    Ok(py_record_batch.unbind())
}

/// Persistent core sampler bound to an owned graph handle.
///
/// Mirrors the hetero binding's `OwnedSampler` (see `crate::hetero` for the
/// full soundness discussion — the same four conditions hold here): the core
/// sampler borrows the graph, and rebuilding it per `sample()` call would
/// reallocate — and, in direct-dedup mode, zero — the num_nodes-sized
/// scratch on every batch.
struct OwnedNeighborSampler {
    sampler: NeighborSampler<'static>,
    // SAFETY-LOAD-BEARING: must drop AFTER `sampler`. Marker leading
    // underscore signals "don't reorder me" to readers.
    _arc: Arc<Graph>,
}

// Compile-time field-order guard: the borrow holder must drop before its
// backing storage. See the identical guard on the hetero `OwnedSampler`.
const _: () = {
    let s = std::mem::offset_of!(OwnedNeighborSampler, sampler);
    let a = std::mem::offset_of!(OwnedNeighborSampler, _arc);
    assert!(
        s < a,
        "OwnedNeighborSampler field order violates the drop-before-arc invariant"
    );
};

/// Erase the lifetime of a borrow of `arc`'s graph.
///
/// # Safety
/// The caller must keep `arc` (or a clone) alive, and drop every value built
/// from the returned reference before it.
unsafe fn erase_graph_lifetime(arc: &Arc<Graph>) -> &'static Graph {
    // SAFETY: `Arc::as_ptr` is stable for the allocation's lifetime, which
    // the caller extends past every use of the reference.
    unsafe { &*Arc::as_ptr(arc) }
}

/// Map a config/graph mismatch to `ValueError`.
fn config_error(e: aethergraph_core::SamplerConfigError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Borrow the graph handle's `Arc`, failing instead of panicking while
/// another thread mutably borrows the graph object.
fn graph_arc(py: Python<'_>, graph: &Py<PyCsrGraph>) -> PyResult<Arc<Graph>> {
    Ok(graph
        .try_borrow(py)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("graph is busy: {e}")))?
        .inner_arc())
}

impl OwnedNeighborSampler {
    fn try_new(arc: Arc<Graph>, config: SamplingConfig) -> PyResult<Self> {
        // SAFETY: the Arc clone moved into `_arc` keeps the graph alive for
        // `Self`'s lifetime, and `sampler` drops first (field order, guarded
        // above). The Arc is private and never cloned out.
        let graph_ref = unsafe { erase_graph_lifetime(&arc) };
        let sampler = NeighborSampler::try_new(graph_ref, config).map_err(config_error)?;
        Ok(Self { sampler, _arc: arc })
    }
}

/// Python wrapper for NeighborSampler.
#[pyclass(name = "NeighborSampler")]
pub struct PyNeighborSampler {
    inner: OwnedNeighborSampler,
    config: PySamplingConfig,
}

#[pymethods]
impl PyNeighborSampler {
    /// Create a new neighbor sampler.
    ///
    /// The core sampler (with its pre-allocated dedup and scratch buffers)
    /// is built once here and reused across every `sample()` call.
    ///
    /// Args:
    ///     graph: CsrGraph to sample from
    ///     config: SamplingConfig with num_neighbors, replace, and seed parameters
    ///
    /// Returns:
    ///     NeighborSampler: Sampler instance
    ///
    /// Raises:
    ///     ValueError: `weighted` or `temporal_strategy` is set on a graph
    ///         without edge weights or timestamps.
    #[new]
    fn new(py: Python<'_>, graph: Py<PyCsrGraph>, config: PySamplingConfig) -> PyResult<Self> {
        let inner = OwnedNeighborSampler::try_new(graph_arc(py, &graph)?, config.inner.clone())?;
        Ok(Self { inner, config })
    }

    /// Sample k-hop neighborhoods for a batch of seed nodes.
    ///
    /// Routes to the disjoint path when the config asks for it. The
    /// persistent sampler's RNG advances across calls, so consecutive
    /// batches draw different samples (a fresh sampler with the same seed
    /// reproduces the same sequence from the start).
    ///
    /// Args:
    ///     seeds: Seed node IDs as numpy array (uint32 or int64) or list
    ///     input_times: Per-seed time bounds (float64, one per seed) under
    ///         temporal sampling; None leaves every seed unbounded.
    ///
    /// Returns:
    ///     SampledSubgraph: Sampled subgraph containing nodes and edges
    ///
    /// Raises:
    ///     SamplingError: A seed is outside the graph, or `input_times` is
    ///         given without a temporal strategy or with the wrong length.
    #[pyo3(signature = (seeds, input_times=None))]
    fn sample(
        &mut self,
        py: Python<'_>,
        seeds: &Bound<'_, PyAny>,
        input_times: Option<numpy::PyReadonlyArray1<f64>>,
    ) -> PyResult<PySampledSubgraph> {
        // Pull everything out of Python objects up front: the seeds and the
        // per-seed timestamps. Once these are plain Rust data we can run the
        // sampler with the GIL released so background Python threads keep
        // running.
        let seeds = crate::error::extract_seed_batch(seeds, self.inner._arc.num_nodes())?;
        let times_vec: Option<Vec<f64>> = input_times
            .as_ref()
            .map(|t| t.as_slice().map(<[f64]>::to_vec))
            .transpose()?;

        // Sampling and the int64 conversion run without the GIL. No Python
        // object is touched inside this closure.
        let sampler = &mut self.inner.sampler;
        let prepared = py.detach(move || {
            let subgraph = sampler
                .sample(&seeds, times_vec.as_deref())
                .map_err(|e| e.to_string())?;
            PreparedSubgraph::new(subgraph)
        });
        PySampledSubgraph::from_prepared(py, prepared.map_err(sampling_error)?)
    }

    fn __repr__(&self) -> String {
        format!(
            "NeighborSampler(num_neighbors={:?}, replace={})",
            self.config.inner.fanout, self.config.inner.replace
        )
    }
}

/// Core parallel sampler bound to an owned graph handle; same construction
/// and drop-order contract as [`OwnedNeighborSampler`].
struct OwnedParallelBatchSampler {
    sampler: ParallelBatchSampler<'static>,
    // SAFETY-LOAD-BEARING: must drop AFTER `sampler`.
    _arc: Arc<Graph>,
}

const _: () = {
    let s = std::mem::offset_of!(OwnedParallelBatchSampler, sampler);
    let a = std::mem::offset_of!(OwnedParallelBatchSampler, _arc);
    assert!(
        s < a,
        "OwnedParallelBatchSampler field order violates the drop-before-arc invariant"
    );
};

/// Python wrapper for ParallelBatchSampler.
///
/// The core sampler — its per-thread sampler pool and its position in the
/// batch stream — lives as long as this object, so repeated calls reuse the
/// node-sized dedup tables and keep drawing fresh samples.
#[pyclass(name = "ParallelBatchSampler")]
pub struct PyParallelBatchSampler {
    inner: OwnedParallelBatchSampler,
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
    ///
    /// Raises:
    ///     ValueError: `weighted` or `temporal_strategy` is set on a graph
    ///         without edge weights or timestamps.
    #[new]
    fn new(py: Python<'_>, graph: Py<PyCsrGraph>, config: PySamplingConfig) -> PyResult<Self> {
        let arc = graph_arc(py, &graph)?;
        // SAFETY: `arc` moves into `_arc` below, and `sampler` drops first
        // (field order, guarded above).
        let graph_ref = unsafe { erase_graph_lifetime(&arc) };
        let sampler =
            ParallelBatchSampler::try_new(graph_ref, config.inner.clone()).map_err(config_error)?;
        Ok(Self {
            inner: OwnedParallelBatchSampler { sampler, _arc: arc },
            config,
        })
    }

    /// Sample neighborhoods for multiple batches in parallel.
    ///
    /// Each batch draws from its own RNG stream, derived from the config's
    /// seed and the batch's position among every batch this sampler has
    /// handled: a fixed seed reproduces the same sequence of calls exactly,
    /// at any thread count.
    ///
    /// Args:
    ///     batches: One seed collection per batch. Numpy ``uint32`` /
    ///         ``int64`` arrays cross the FFI boundary as bulk slice copies;
    ///         Python ``list[int]`` also works but pays per-element
    ///         unboxing.
    ///
    /// Returns:
    ///     List[SampledSubgraph]: List of sampled subgraphs, one per batch
    ///
    /// Raises:
    ///     SamplingError: A seed is outside the graph.
    fn sample_batches(
        &self,
        py: Python<'_>,
        batches: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<PySampledSubgraph>> {
        // Parse every batch while the GIL is held (numpy arrays land as
        // single slice copies); sampling and conversion then run detached.
        let num_nodes = self.inner._arc.num_nodes();
        let batches: Vec<Seeds> = batches
            .iter()
            .map(|b| crate::error::extract_seed_batch(b, num_nodes))
            .collect::<PyResult<_>>()?;

        let sampler = &self.inner.sampler;
        let prepared = py.detach(move || -> Result<Vec<PreparedSubgraph>, String> {
            sampler
                .sample_batches(&batches)
                .map_err(|e| e.to_string())?
                .into_iter()
                .map(PreparedSubgraph::new)
                .collect()
        });

        prepared
            .map_err(sampling_error)?
            .into_iter()
            .map(|p| PySampledSubgraph::from_prepared(py, p))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ParallelBatchSampler(num_neighbors={:?}, replace={})",
            self.config.inner.fanout, self.config.inner.replace
        )
    }
}
