use aethergraph_core::Graph;
use aethergraph_core::{
    GraphValidationMode, load_graph, load_graph_mmap, load_graph_owned, load_graph_with_validation,
    save_graph,
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
    /// Load a graph with mmap-backed storage.
    ///
    /// Args:
    ///     path: Path to graph file
    ///     validation: "auto" | "header_only" | "offsets_only" | "full"
    ///
    /// Returns:
    ///     CsrGraph: Loaded graph instance
    #[staticmethod]
    #[pyo3(signature = (path, validation = "auto"))]
    fn load(path: &str, validation: &str) -> PyResult<Self> {
        let graph = if validation.eq_ignore_ascii_case("auto") {
            load_graph(path)
        } else {
            let mode = parse_validation_mode(validation)?;
            load_graph_with_validation(path, mode)
        }
        .map_err(|e| graph_load_error(format!("Failed to load graph: {}", e)))?;

        Ok(Self {
            inner: Arc::new(graph),
        })
    }

    /// Explicit mmap-backed load path.
    ///
    /// Args:
    ///     path: Path to graph file
    ///     validation: "header_only" | "offsets_only" | "full"
    #[staticmethod]
    #[pyo3(signature = (path, validation = "offsets_only"))]
    fn load_mmap(path: &str, validation: &str) -> PyResult<Self> {
        let mode = parse_validation_mode(validation)?;
        let graph = load_graph_mmap(path, mode)
            .map_err(|e| graph_load_error(format!("Failed to mmap-load graph: {}", e)))?;

        Ok(Self {
            inner: Arc::new(graph),
        })
    }

    /// Explicit owned in-memory load path.
    ///
    /// Args:
    ///     path: Path to graph file
    ///     validation: "auto" | "header_only" | "offsets_only" | "full"
    #[staticmethod]
    #[pyo3(signature = (path, validation = "full"))]
    fn load_owned(path: &str, validation: &str) -> PyResult<Self> {
        let mode = if validation.eq_ignore_ascii_case("auto") {
            auto_validation_mode(path)?
        } else {
            parse_validation_mode(validation)?
        };

        let graph = load_graph_owned(path, mode)
            .map_err(|e| graph_load_error(format!("Failed to owned-load graph: {}", e)))?;

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
        let edges: Vec<(u32, u32)> = src_slice
            .iter()
            .zip(dst_slice.iter())
            .map(|(&s, &d)| (s, d))
            .collect();
        let w = weights.as_ref().map(|w| w.as_slice()).transpose()?;
        let graph = Graph::from_edges(num_nodes, &edges, w)
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
    fn permute(&self, perm: PyReadonlyArray1<u32>) -> PyResult<Self> {
        let perm_slice = perm.as_slice()?;
        let new_graph = self
            .inner
            .permute(perm_slice)
            .map_err(|e| graph_load_error(format!("Permutation failed: {}", e)))?;
        Ok(Self {
            inner: Arc::new(new_graph),
        })
    }

    /// Save graph to a binary file.
    ///
    /// Args:
    ///     path: Path to save the binary graph file
    fn save(&self, path: &str) -> PyResult<()> {
        save_graph(self.inner.as_ref(), path)
            .map_err(|e| graph_load_error(format!("Failed to save graph: {}", e)))
    }

    /// Returns the number of nodes in the graph.
    ///
    /// Returns:
    ///     int: Number of nodes
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the number of edges in the graph.
    ///
    /// Returns:
    ///     int: Number of edges
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

    /// Returns whether the graph has edge weights.
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
        Arc::make_mut(&mut self.inner).set_timestamps(ts);
        Ok(())
    }

    /// Returns whether the graph has edge timestamps.
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

    /// Returns a string representation of the graph.
    fn __repr__(&self) -> String {
        format!(
            "CsrGraph(num_nodes={}, num_edges={}, weighted={})",
            self.inner.num_nodes(),
            self.inner.num_edges(),
            self.inner.weights().is_some()
        )
    }

    /// Returns a short string representation of the graph.
    fn __str__(&self) -> String {
        format!(
            "Graph with {} nodes and {} edges",
            self.inner.num_nodes(),
            self.inner.num_edges()
        )
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
    match value.to_ascii_lowercase().as_str() {
        "header_only" => Ok(GraphValidationMode::HeaderOnly),
        "offsets_only" => Ok(GraphValidationMode::OffsetsOnly),
        "full" => Ok(GraphValidationMode::Full),
        "auto" => Err(pyo3::exceptions::PyValueError::new_err(
            "'auto' is only supported by CsrGraph.load() and CsrGraph.load_owned()",
        )),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid validation mode '{}'. Must be one of: auto, header_only, offsets_only, full",
            value
        ))),
    }
}

fn auto_validation_mode(path: &str) -> PyResult<GraphValidationMode> {
    let metadata = std::fs::metadata(Path::new(path)).map_err(|e| {
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
