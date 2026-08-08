use aethergraph_core::Graph;
use numpy::{PyArray1, PyReadonlyArray1};
use parking_lot::Mutex;
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;

use crate::graph::PyCsrGraph;

/// Python wrapper for the lock-free DynamicGraph.
///
/// Supports concurrent edge inserts and neighbor reads for live
/// GNN training on evolving graphs.
///
/// At the Rust level, only one writer may exist at a time; Python callers
/// are serialized via the internal `write_lock` so multiple threads can
/// call `insert_edge*` without observing the underlying single-writer
/// contention error.
#[pyclass(name = "DynamicGraph")]
pub struct PyDynamicGraph {
    inner: Arc<aether_graph::DynamicGraph>,
    /// Serializes Python-side writes when the GIL is released.
    write_lock: Mutex<()>,
    /// Reusable buffer for neighbor collection (avoids per-call allocation).
    buf: Mutex<Vec<u32>>,
}

impl PyDynamicGraph {
    fn map_insert_err(err: aether_graph::InsertError) -> PyErr {
        match err {
            aether_graph::InsertError::ArenaFull => {
                pyo3::exceptions::PyRuntimeError::new_err("C-tree arena is full")
            }
            aether_graph::InsertError::VertexOutOfRange { src, dst } => {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "edge ({src}, {dst}) references a vertex out of range for this graph"
                ))
            }
            aether_graph::InsertError::WalAppend => pyo3::exceptions::PyOSError::new_err(
                "WAL append failed; the edge was not published and the graph is \
                 poisoned — rebuild from a checkpoint (see logs for the I/O error)",
            ),
        }
    }

    fn map_writer_err(err: aether_graph::WriterError) -> PyErr {
        match err {
            aether_graph::WriterError::Busy => pyo3::exceptions::PyRuntimeError::new_err(
                "DynamicGraph: writer slot already held by another path (likely an \
                 internal helper that took an Arc<DynamicGraph> clone). Drop the \
                 other writer before calling insert_edge*.",
            ),
            aether_graph::WriterError::Poisoned => pyo3::exceptions::PyRuntimeError::new_err(
                "DynamicGraph: poisoned by a previous panicking writer. The graph's \
                 internal state may be inconsistent — rebuild from a checkpoint.",
            ),
        }
    }

    fn map_wal_err(err: aether_graph::WalError) -> PyErr {
        use aether_graph::WalError;
        match err {
            WalError::Io(e) => pyo3::exceptions::PyOSError::new_err(format!("WAL io error: {e}")),
            WalError::BadMagic { found } => pyo3::exceptions::PyValueError::new_err(format!(
                "WAL bad magic header: {found:?} — file is not an AetherGraph WAL"
            )),
            WalError::UnknownVersion(v) => pyo3::exceptions::PyValueError::new_err(format!(
                "WAL version {v} not supported by this build"
            )),
            WalError::Locked => pyo3::exceptions::PyOSError::new_err(
                "WAL file is exclusively locked by another writer (this process \
                 or another); close the other DynamicGraph before reopening",
            ),
            WalError::RecordOutOfRange {
                src,
                dst,
                num_vertices,
            } => pyo3::exceptions::PyValueError::new_err(format!(
                "WAL record ({src}, {dst}) exceeds num_vertices {num_vertices}; \
                 reopen with the num_vertices the log was written against"
            )),
            WalError::ReplayArenaFull => pyo3::exceptions::PyRuntimeError::new_err(
                "arena filled during WAL replay; reopen with a larger arena_mb",
            ),
        }
    }

    /// Bounds-check a vertex id against `num_vertices` so out-of-range reads
    /// raise `ValueError` instead of panicking inside the Rust core.
    fn check_vertex(&self, vertex: u32) -> PyResult<()> {
        let num_vertices = self.inner.num_vertices();
        if (vertex as usize) < num_vertices {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "vertex {vertex} out of range for graph with {num_vertices} vertices"
            )))
        }
    }
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
            write_lock: Mutex::new(()),
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
            write_lock: Mutex::new(()),
            buf: Mutex::new(Vec::new()),
        })
    }

    /// Open a DynamicGraph backed by an append-only write-ahead log.
    ///
    /// If the WAL at ``path`` already contains records from a previous run,
    /// they are replayed before this returns; if the file ends in a torn
    /// record (mid-write crash), the trailing bytes are truncated. Every
    /// subsequent ``insert_edge*`` call appends to the log and fsyncs at
    /// writer-guard close.
    ///
    /// Args:
    ///     path: WAL file path. Created if it does not exist.
    ///     num_vertices: Number of vertices (fixed at construction).
    ///     arena_mb: Arena capacity in megabytes (default 256).
    ///
    /// Raises:
    ///     OSError: I/O failure opening, reading, or writing the WAL.
    ///     ValueError: File exists but is not a valid AetherGraph WAL
    ///         (bad magic or unsupported version).
    ///     RuntimeError: WAL is corrupt past the recoverable prefix.
    #[staticmethod]
    #[pyo3(signature = (path, num_vertices, arena_mb = 256))]
    fn open_with_wal(path: PathBuf, num_vertices: usize, arena_mb: usize) -> PyResult<Self> {
        let arena_bytes = arena_mb * 1024 * 1024;
        let graph = aether_graph::DynamicGraph::open_with_wal(&path, num_vertices, arena_bytes)
            .map_err(Self::map_wal_err)?;
        Ok(Self {
            inner: Arc::new(graph),
            write_lock: Mutex::new(()),
            buf: Mutex::new(Vec::new()),
        })
    }

    /// Insert a directed edge from src to dst.
    ///
    /// Returns True if the edge was new, False if it already existed.
    /// Raises RuntimeError if the arena is full or the underlying writer slot
    /// is held by another path.
    fn insert_edge(&self, py: Python<'_>, src: u32, dst: u32) -> PyResult<bool> {
        let inner = Arc::clone(&self.inner);
        // The closure runs without the GIL but still under `write_lock`,
        // so concurrent Python callers serialize at the wrapper layer. The
        // underlying `inner.writer()` can still fail if anyone else holds an
        // `Arc<DynamicGraph>` clone with an outstanding writer guard — we
        // surface that as a clean Python error instead of a Rust panic.
        py.detach(move || {
            let _guard = self.write_lock.lock();
            let mut writer = inner.writer().map_err(Self::map_writer_err)?;
            writer.insert_edge(src, dst).map_err(Self::map_insert_err)
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

        // Pack (src, dst) into one u64 per edge: a single sort groups the
        // batch by source so each source's destinations go through the bulk
        // merge path (one tree walk per source) instead of one path-copying
        // insert per edge.
        let mut packed: Vec<u64> = src_slice
            .iter()
            .zip(dst_slice)
            .map(|(&s, &d)| ((s as u64) << 32) | d as u64)
            .collect();
        let inner = Arc::clone(&self.inner);

        py.detach(move || {
            let _guard = self.write_lock.lock();
            let mut writer = inner.writer().map_err(Self::map_writer_err)?;
            packed.sort_unstable();
            packed.dedup();
            let mut count = 0u64;
            let mut dsts: Vec<u32> = Vec::new();
            for run in packed.chunk_by(|a, b| (a >> 32) == (b >> 32)) {
                let s = (run[0] >> 32) as u32;
                dsts.clear();
                dsts.extend(run.iter().map(|&p| p as u32));
                count += writer
                    .insert_edges_sorted(s, &dsts)
                    .map_err(Self::map_insert_err)?;
            }
            Ok(count as usize)
        })
    }

    /// Degree (number of outgoing edges) for a vertex.
    ///
    /// Raises ValueError if the vertex is out of range.
    fn degree(&self, vertex: u32) -> PyResult<usize> {
        self.check_vertex(vertex)?;
        Ok(self.inner.degree(vertex))
    }

    /// Check whether edge (src -> dst) exists.
    ///
    /// Raises ValueError if src or dst is out of range.
    fn has_edge(&self, src: u32, dst: u32) -> PyResult<bool> {
        self.check_vertex(src)?;
        self.check_vertex(dst)?;
        Ok(self.inner.has_edge(src, dst))
    }

    /// Get sorted neighbor array for a vertex as int64 (PyTorch compatible).
    ///
    /// Raises ValueError if the vertex is out of range.
    fn neighbors<'py>(&self, py: Python<'py>, vertex: u32) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.check_vertex(vertex)?;
        let mut buf = self.buf.lock();
        self.inner.neighbors_into(vertex, &mut buf);
        let i64_vec: Vec<i64> = buf.iter().map(|&v| v as i64).collect();
        Ok(PyArray1::from_vec(py, i64_vec))
    }

    /// Get sorted neighbor array for a vertex as uint32 (zero-copy convertible).
    ///
    /// Raises ValueError if the vertex is out of range.
    fn neighbors_u32<'py>(
        &self,
        py: Python<'py>,
        vertex: u32,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        self.check_vertex(vertex)?;
        let mut buf = self.buf.lock();
        self.inner.neighbors_into(vertex, &mut buf);
        Ok(PyArray1::from_slice(py, &buf))
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

    /// Current epoch — the monotonic version counter advanced once per
    /// successful writer-guard drop (panic-poisoned drops are skipped).
    /// Pin this before a multi-source read to coordinate consistency with
    /// other subsystems sharing the same `EpochClock`.
    #[getter]
    fn current_epoch(&self) -> u64 {
        self.inner.current_epoch().as_u64()
    }

    /// Create a frozen CSR snapshot for use with NeighborSampler/NeighborLoader.
    ///
    /// Collects all edges from the C-tree neighbor lists into a static CSR
    /// graph. O(V + E) time — call once per epoch, not per batch.
    fn snapshot(&self, py: Python<'_>) -> PyCsrGraph {
        let inner = Arc::clone(&self.inner);
        // The O(V + E) collection touches no Python state; release the GIL so
        // other Python threads (e.g. concurrent inserters) keep running.
        let graph = py.detach(move || {
            let (offsets, edges) = inner.snapshot_csr();
            Graph::from_csr_arrays(inner.num_vertices(), offsets, edges, None)
        });
        PyCsrGraph {
            inner: Arc::new(graph),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.num_vertices()
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
        self.__repr__()
    }
}
