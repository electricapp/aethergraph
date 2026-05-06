use aethergraph_core::Graph;
use numpy::{PyArray1, PyReadonlyArray1};
use parking_lot::Mutex;
use pyo3::prelude::*;
use std::sync::Arc;

use crate::graph::PyCsrGraph;

/// Python wrapper for the lock-free DynamicGraph.
///
/// Supports concurrent edge inserts and neighbor reads for live
/// GNN training on evolving graphs.
#[pyclass(name = "DynamicGraph")]
pub struct PyDynamicGraph {
    inner: Arc<aether_graph::DynamicGraph>,
    /// Reusable buffer for neighbor collection (avoids per-call allocation).
    buf: Mutex<Vec<u32>>,
}

#[pymethods]
impl PyDynamicGraph {
    /// Create an empty dynamic graph.
    ///
    /// Args:
    ///     num_vertices: Number of vertices (fixed at construction).
    ///     arena_mb: Arena capacity in megabytes (default 256).
    #[new]
    #[pyo3(signature = (num_vertices, arena_mb = 256))]
    fn new(num_vertices: usize, arena_mb: usize) -> Self {
        let arena_bytes = arena_mb * 1024 * 1024;
        Self {
            inner: Arc::new(aether_graph::DynamicGraph::new(num_vertices, arena_bytes)),
            buf: Mutex::new(Vec::new()),
        }
    }

    /// Build a DynamicGraph from numpy edge arrays.
    ///
    /// Args:
    ///     num_vertices: Number of vertices.
    ///     src: Source vertex array (uint32).
    ///     dst: Destination vertex array (uint32).
    ///     arena_mb: Arena capacity in megabytes (default 256).
    ///
    /// Returns:
    ///     DynamicGraph with all edges inserted.
    #[staticmethod]
    #[pyo3(signature = (num_vertices, src, dst, arena_mb = 256))]
    fn from_edges(
        num_vertices: usize,
        src: PyReadonlyArray1<u32>,
        dst: PyReadonlyArray1<u32>,
        arena_mb: usize,
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

        let arena_bytes = arena_mb * 1024 * 1024;
        let edges: Vec<(u32, u32)> = src_slice
            .iter()
            .zip(dst_slice.iter())
            .map(|(&s, &d)| (s, d))
            .collect();
        let graph = aether_graph::DynamicGraph::from_edges(num_vertices, &edges, arena_bytes);

        Ok(Self {
            inner: Arc::new(graph),
            buf: Mutex::new(Vec::new()),
        })
    }

    /// Insert a directed edge from src to dst.
    ///
    /// Returns True if the edge was new, False if it already existed.
    /// Raises RuntimeError if the arena is full.
    fn insert_edge(&self, py: Python<'_>, src: u32, dst: u32) -> PyResult<bool> {
        let inner = Arc::clone(&self.inner);
        py.detach(move || {
            inner
                .insert_edge(src, dst)
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("C-tree arena is full"))
        })
    }

    /// Batch-insert edges from numpy arrays.
    ///
    /// Args:
    ///     src: Source vertex array (uint32).
    ///     dst: Destination vertex array (uint32).
    ///
    /// Returns:
    ///     Number of new edges inserted (duplicates are skipped).
    ///
    /// Raises:
    ///     RuntimeError: If the arena is full.
    fn insert_edges(
        &self,
        py: Python<'_>,
        src: PyReadonlyArray1<u32>,
        dst: PyReadonlyArray1<u32>,
    ) -> PyResult<usize> {
        let src_slice = src.as_slice()?;
        let dst_slice = dst.as_slice()?;
        if src_slice.len() != dst_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "src and dst must have the same length, got {} and {}",
                src_slice.len(),
                dst_slice.len()
            )));
        }

        // Copy into owned vecs so we can release the GIL
        let src_vec: Vec<u32> = src_slice.to_vec();
        let dst_vec: Vec<u32> = dst_slice.to_vec();
        let inner = Arc::clone(&self.inner);

        py.detach(move || {
            let mut count = 0usize;
            for (&s, &d) in src_vec.iter().zip(dst_vec.iter()) {
                match inner.insert_edge(s, d) {
                    Ok(true) => count += 1,
                    Ok(false) => {} // duplicate
                    Err(_) => {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "C-tree arena is full",
                        ));
                    }
                }
            }
            Ok(count)
        })
    }

    /// Degree (number of outgoing edges) for a vertex.
    fn degree(&self, vertex: u32) -> usize {
        self.inner.degree(vertex)
    }

    /// Check whether edge (src -> dst) exists.
    fn has_edge(&self, src: u32, dst: u32) -> bool {
        self.inner.has_edge(src, dst)
    }

    /// Get sorted neighbor array for a vertex as int64 (PyTorch compatible).
    fn neighbors<'py>(&self, py: Python<'py>, vertex: u32) -> Bound<'py, PyArray1<i64>> {
        let mut buf = self.buf.lock();
        self.inner.neighbors_into(vertex, &mut buf);
        // Convert u32 -> i64 for PyTorch compatibility
        let i64_vec: Vec<i64> = buf.iter().map(|&v| v as i64).collect();
        PyArray1::from_vec(py, i64_vec)
    }

    /// Number of vertices (fixed at construction).
    #[getter]
    fn num_vertices(&self) -> usize {
        self.inner.num_vertices()
    }

    /// Total number of edges.
    #[getter]
    fn num_edges(&self) -> u64 {
        self.inner.num_edges()
    }

    /// Arena bytes currently used.
    #[getter]
    fn arena_used(&self) -> usize {
        self.inner.arena_used()
    }

    /// Arena total capacity in bytes.
    #[getter]
    fn arena_capacity(&self) -> usize {
        self.inner.arena_capacity()
    }

    /// Create a frozen CSR snapshot for use with NeighborSampler/NeighborLoader.
    ///
    /// Collects all edges from the C-tree neighbor lists into a static CSR
    /// graph. O(V + E) time — call once per epoch, not per batch.
    ///
    /// Returns:
    ///     CsrGraph: A static graph snapshot usable with the existing sampler.
    fn snapshot(&self) -> PyCsrGraph {
        let (offsets, edges) = self.inner.snapshot_csr();
        let num_vertices = self.inner.num_vertices();
        let graph = Graph::from_csr_arrays(num_vertices, offsets, edges, None);
        PyCsrGraph {
            inner: Arc::new(graph),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DynamicGraph(num_vertices={}, num_edges={}, arena={}/{}MB)",
            self.inner.num_vertices(),
            self.inner.num_edges(),
            self.inner.arena_used() / (1024 * 1024),
            self.inner.arena_capacity() / (1024 * 1024),
        )
    }

    fn __str__(&self) -> String {
        format!(
            "DynamicGraph with {} vertices and {} edges",
            self.inner.num_vertices(),
            self.inner.num_edges()
        )
    }
}
