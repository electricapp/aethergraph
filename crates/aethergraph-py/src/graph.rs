use aethergraph_core::Graph;
use aethergraph_core::{
    GraphValidationMode, load_graph, load_graph_mmap, load_graph_owned, load_graph_with_validation,
    save_graph, save_graph_compressed,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::path::Path;
use std::sync::Arc;

use crate::error::graph_load_error;

const FULL_VALIDATION_THRESHOLD_BYTES: u64 = 512 * 1024 * 1024;

/// Python wrapper for Graph.
///
/// This exposes the high-performance Rust graph structure to Python
/// with zero-copy access to neighbor data.
#[pyclass(name = "CsrGraph")]
pub struct PyCsrGraph {
    pub(crate) inner: Arc<Graph>,
}

#[pymethods]
impl PyCsrGraph {
    /// Load a graph from disk.
    ///
    /// Single entry point; pick mmap vs owned via the `storage` argument.
    ///
    /// Args:
    ///     path: Path to graph file.
    ///     storage: One of:
    ///         - `"auto"` (default) — mmap-backed if the file is larger than
    ///           the system's preferred page-cache threshold, owned otherwise.
    ///         - `"mmap"` — explicit mmap-backed; topology stays on disk and
    ///           is paged in on access.
    ///         - `"owned"` — read the whole file into RAM at load time.
    ///     validation: One of:
    ///         - `"auto"` (default) — `"offsets_only"` for files >512 MiB,
    ///           `"full"` for smaller. Mmap path uses `"offsets_only"`.
    ///         - `"header_only"` — verify magic + version only.
    ///         - `"offsets_only"` — also check that CSR offsets are monotonic.
    ///         - `"full"` — additionally check every neighbor ID is in range.
    ///
    /// Returns:
    ///     CsrGraph: Loaded graph instance.
    ///
    /// Raises:
    ///     GraphLoadError: If `path` cannot be read, the file is corrupt, or
    ///         validation fails.
    #[staticmethod]
    #[pyo3(signature = (path, *, storage = "auto", validation = "auto"))]
    fn load(
        py: Python<'_>,
        path: std::path::PathBuf,
        storage: &str,
        validation: &str,
    ) -> PyResult<Self> {
        let storage_norm = storage.to_ascii_lowercase();
        let validation_norm = validation.to_ascii_lowercase();

        // Reading + validating the file is O(E) I/O and touches no Python
        // state; release the GIL so other Python threads keep running.
        // `load_graph*` functions take `impl AsRef<Path>` so `&path` works
        // directly — no string round-trip needed.
        let graph = py.detach(|| -> PyResult<_> {
            let result = match storage_norm.as_str() {
                "auto" => {
                    if validation_norm == "auto" {
                        load_graph(&path)
                    } else {
                        let mode = parse_validation_mode(&validation_norm)?;
                        load_graph_with_validation(&path, mode)
                    }
                }
                "mmap" => {
                    let mode = if validation_norm == "auto" {
                        GraphValidationMode::OffsetsOnly
                    } else {
                        parse_validation_mode(&validation_norm)?
                    };
                    load_graph_mmap(&path, mode)
                }
                "owned" => {
                    let mode = if validation_norm == "auto" {
                        auto_validation_mode(&path)?
                    } else {
                        parse_validation_mode(&validation_norm)?
                    };
                    load_graph_owned(&path, mode)
                }
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid storage mode '{other}'. Must be one of: auto, mmap, owned"
                    )));
                }
            };
            result.map_err(|e| {
                graph_load_error(format!("Failed to load graph ({storage_norm}): {e}"))
            })
        })?;

        Ok(Self {
            inner: Arc::new(graph),
        })
    }

    /// Create a graph from edge arrays.
    ///
    /// Args:
    ///     num_nodes: Number of nodes in the graph
    ///     src: Source node array (uint32)
    ///     dst: Destination node array (uint32)
    ///
    /// Returns:
    ///     CsrGraph: New graph instance
    #[staticmethod]
    #[pyo3(signature = (num_nodes, src, dst, weights=None))]
    fn from_edges(
        py: Python<'_>,
        num_nodes: usize,
        src: PyReadonlyArray1<u32>,
        dst: PyReadonlyArray1<u32>,
        weights: Option<PyReadonlyArray1<f32>>,
    ) -> PyResult<Self> {
        let src_slice = src.as_slice()?;
        let dst_slice = dst.as_slice()?;
        if src_slice.len() != dst_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "src and dst must have the same length, got {} and {}",
                src_slice.len(),
                dst_slice.len()
            )));
        }
        let w = weights.as_ref().map(|w| w.as_slice()).transpose()?;
        // The O(E) CSR build reads the numpy-backed src/dst/weight slices
        // directly (kept alive by the readonly guards above) — no
        // interleaved (src, dst) tuple copy of the edge list is
        // materialized. Release the GIL across the build.
        let graph = py
            .detach(|| Graph::from_src_dst(num_nodes, src_slice, dst_slice, w))
            .map_err(|e| graph_load_error(format!("Failed to create graph: {}", e)))?;
        Ok(Self {
            inner: Arc::new(graph),
        })
    }

    /// Compute Rabbit Order permutation for improved cache locality.
    ///
    /// Returns:
    ///     numpy.ndarray: uint32 array where perm[new_id] = old_id.
    fn reorder_rabbit<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let perm = py.detach(|| self.inner.reorder_rabbit());
        PyArray1::from_vec(py, perm)
    }

    /// Apply a node permutation to create a reordered graph.
    ///
    /// Args:
    ///     perm: uint32 array where perm[new_id] = old_id.
    ///
    /// Returns:
    ///     CsrGraph: Reordered graph with improved cache locality.
    fn permute(&self, py: Python<'_>, perm: PyReadonlyArray1<u32>) -> PyResult<Self> {
        let perm_slice = perm.as_slice()?;
        // The O(V + E) rebuild reads only the shared graph and the
        // numpy-backed permutation slice (kept alive by the readonly guard);
        // release the GIL across it.
        let new_graph = py
            .detach(|| self.inner.permute(perm_slice))
            .map_err(|e| graph_load_error(format!("Permutation failed: {}", e)))?;
        Ok(Self {
            inner: Arc::new(new_graph),
        })
    }

    /// Save graph to a binary file.
    ///
    /// Args:
    ///     path: Path to save the binary graph file
    ///     compressed: Write the succinct-coded format (Elias-Fano offsets,
    ///         StreamVByte edges) instead of flat arrays. Typically 2-4x
    ///         smaller at rest. `load` reads either format automatically,
    ///         but a compressed file always loads into owned storage, so
    ///         `storage="mmap"` rejects it.
    #[pyo3(signature = (path, *, compressed = false))]
    fn save(&self, py: Python<'_>, path: std::path::PathBuf, compressed: bool) -> PyResult<()> {
        // The O(E) file write touches no Python state; release the GIL.
        py.detach(|| {
            if compressed {
                save_graph_compressed(self.inner.as_ref(), &path)
            } else {
                save_graph(self.inner.as_ref(), &path)
            }
        })
        .map_err(|e| graph_load_error(format!("Failed to save graph: {}", e)))
    }

    /// Number of nodes in the graph.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Number of edges in the graph.
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Returns the degree (number of outgoing edges) for a node.
    ///
    /// Args:
    ///     node: Node ID (u32)
    ///
    /// Returns:
    ///     int: Number of outgoing edges from this node
    fn degree(&self, node: u32) -> usize {
        self.inner.degree(node)
    }

    /// Returns the degree of every node as a numpy array.
    ///
    /// One FFI call and a single pass over the CSR offsets — use this instead
    /// of a per-node `degree()` loop when the whole distribution is needed.
    ///
    /// Returns:
    ///     numpy.ndarray: Per-node degrees, length num_nodes (dtype=uint32)
    fn degrees<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        // O(V) scan over the (possibly mmap'd) offsets; release the GIL.
        let degrees = py.detach(|| self.inner.degrees());
        PyArray1::from_vec(py, degrees)
    }

    /// Returns the neighbor IDs for a given node as a numpy array.
    ///
    /// Args:
    ///     node: Node ID (u32)
    ///
    /// Returns:
    ///     numpy.ndarray: Array of neighbor node IDs (dtype=uint32)
    fn neighbors<'py>(&self, py: Python<'py>, node: u32) -> Bound<'py, PyArray1<u32>> {
        let neighbors = self.inner.neighbors(node);
        PyArray1::from_slice(py, neighbors)
    }

    /// Returns neighbors for multiple nodes in a batch.
    fn batch_neighbors<'py>(
        &self,
        py: Python<'py>,
        nodes: Vec<u32>,
    ) -> Vec<Bound<'py, PyArray1<u32>>> {
        nodes
            .iter()
            .map(|&node| PyArray1::from_slice(py, self.inner.neighbors(node)))
            .collect()
    }

    /// Returns edge weights for a node's neighbors, if available.
    fn neighbor_weights<'py>(
        &self,
        py: Python<'py>,
        node: u32,
    ) -> Option<Bound<'py, PyArray1<f32>>> {
        self.inner
            .neighbor_weights(node)
            .map(|weights| PyArray1::from_slice(py, weights))
    }

    /// Whether the graph has edge weights.
    #[getter]
    fn has_weights(&self) -> bool {
        self.inner.weights().is_some()
    }

    /// Set edge timestamps (parallel to the edges array).
    ///
    /// Required for temporal sampling. Each timestamp corresponds to an edge
    /// in the CSR edges array.
    ///
    /// Args:
    ///     timestamps: numpy array of f64 timestamps (length must equal num_edges)
    fn set_timestamps(&mut self, timestamps: PyReadonlyArray1<f64>) -> PyResult<()> {
        let ts = timestamps.as_slice()?.to_vec();
        // `Arc::get_mut` mutates the single shared allocation in place. If a
        // sampler or loader already captured an `inner_arc()` clone, there is
        // more than one owner and we refuse rather than silently deep-copying
        // (which would leave those consumers reading the timestamp-less graph).
        Arc::get_mut(&mut self.inner)
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "cannot set timestamps: the graph is already shared with a sampler/loader; \
                     set timestamps before constructing them",
                )
            })?
            .set_timestamps(ts)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Whether the graph has edge timestamps.
    #[getter]
    fn has_timestamps(&self) -> bool {
        self.inner.has_timestamps()
    }

    /// Returns basic graph statistics.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let stats = self.inner.stats();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("num_nodes", stats.num_nodes)?;
        dict.set_item("num_edges", stats.num_edges)?;
        dict.set_item("max_degree", stats.max_degree)?;
        dict.set_item("avg_degree", stats.avg_degree)?;
        dict.set_item("has_weights", stats.has_weights)?;

        Ok(dict.into())
    }

    /// `len(graph)` returns the number of nodes.
    fn __len__(&self) -> usize {
        self.inner.num_nodes()
    }

    /// `graph[node]` returns the sorted neighbor array (uint32).
    ///
    /// The returned numpy array is allocated fresh (a copy of the CSR slice),
    /// so it remains valid even after the graph is dropped or replaced. This
    /// matches the behaviour of `neighbors(node)`.
    fn __getitem__<'py>(&self, py: Python<'py>, node: u32) -> Bound<'py, PyArray1<u32>> {
        self.neighbors(py, node)
    }

    /// String representation. `__str__` aliases `__repr__` to keep Python
    /// `print()` and the REPL consistent.
    fn __repr__(&self) -> String {
        format!(
            "CsrGraph(num_nodes={}, num_edges={}, weighted={})",
            self.inner.num_nodes(),
            self.inner.num_edges(),
            self.inner.weights().is_some()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl PyCsrGraph {
    /// Get a reference to the inner Graph (for use by other Rust modules).
    pub fn inner(&self) -> &Graph {
        self.inner.as_ref()
    }

    /// Get a shared Arc to the graph (for zero-copy sharing across loaders).
    pub fn inner_arc(&self) -> Arc<Graph> {
        Arc::clone(&self.inner)
    }
}

fn parse_validation_mode(value: &str) -> PyResult<GraphValidationMode> {
    match value {
        "header_only" => Ok(GraphValidationMode::HeaderOnly),
        "offsets_only" => Ok(GraphValidationMode::OffsetsOnly),
        "full" => Ok(GraphValidationMode::Full),
        "auto" => Err(pyo3::exceptions::PyValueError::new_err(
            "'auto' validation is only resolved by CsrGraph.load(); call sites here expect a concrete mode",
        )),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid validation mode '{value}'. Must be one of: auto, header_only, offsets_only, full"
        ))),
    }
}

fn auto_validation_mode(path: &Path) -> PyResult<GraphValidationMode> {
    let metadata = std::fs::metadata(path).map_err(|e| {
        graph_load_error(format!(
            "Failed to stat graph file for auto validation mode selection: {}",
            e
        ))
    })?;

    if metadata.len() > FULL_VALIDATION_THRESHOLD_BYTES {
        Ok(GraphValidationMode::OffsetsOnly)
    } else {
        Ok(GraphValidationMode::Full)
    }
}
