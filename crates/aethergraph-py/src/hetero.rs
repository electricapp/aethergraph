//! PyO3 bindings for heterogeneous graph support.
//!
//! Thin wrappers around the core `HeteroGraph`, `HeteroNeighborSampler`, and
//! `HeteroSampledSubgraph`. The sampling hot path runs entirely in the core
//! crate; this layer only converts results to numpy arrays at the boundary.
//!
//! Zero-copy path: `Vec<NodeId>` → `PyArray1::from_vec` transfers buffer
//! ownership to numpy without memcpy. The only per-element work is the
//! u32→i64 widening required by PyTorch.

use aethergraph_core::Graph;
use aethergraph_core::graph::NodeId;
use aethergraph_core::graph::hetero::{EdgeTypeId, HeteroGraph, NodeTypeId};
use aethergraph_core::loader::HeteroNeighborLoader;
use aethergraph_core::loader::hetero_sampler::{
    HeteroNeighborSampler, HeteroSampledSubgraph, HeteroSamplingConfig,
};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::sampling_error;

// ---------------------------------------------------------------------------
// HeteroCsrGraph
// ---------------------------------------------------------------------------

/// (src_type, edge_type, dst_type, src_ids, dst_ids) — one COO edge bundle.
type EdgeArrayTuple<'py> = (
    String,
    String,
    String,
    PyReadonlyArray1<'py, u32>,
    PyReadonlyArray1<'py, u32>,
);

#[pyclass(name = "HeteroCsrGraph")]
pub struct PyHeteroCsrGraph {
    pub(crate) inner: Arc<HeteroGraph>,
}

/// One edge type's parsed COO arrays, awaiting its CSR build.
struct EdgeBundle {
    src_type: String,
    rel: String,
    dst_type: String,
    num_src: usize,
    num_dst: usize,
    src: Vec<NodeId>,
    dst: Vec<NodeId>,
}

#[pymethods]
impl PyHeteroCsrGraph {
    /// Build from per-edge-type COO arrays.
    ///
    /// Raises:
    ///     ValueError: Mismatched array lengths, an unknown node type, an
    ///         endpoint outside its type's node count, a repeated node or
    ///         edge type, or more than 255 of either.
    #[staticmethod]
    fn from_edge_arrays(
        py: Python<'_>,
        node_types: &Bound<'_, PyDict>,
        edge_types: Vec<EdgeArrayTuple<'_>>,
    ) -> PyResult<Self> {
        let mut nt_vec: Vec<(String, usize)> = Vec::with_capacity(node_types.len());
        for (key, value) in node_types.iter() {
            let name: String = key.extract()?;
            let count: usize = value.extract()?;
            nt_vec.push((name, count));
        }

        let nt_counts: HashMap<String, usize> = nt_vec.iter().cloned().collect();
        let mut bundles: Vec<EdgeBundle> = Vec::with_capacity(edge_types.len());

        for (src_type, rel, dst_type, src_arr, dst_arr) in edge_types {
            let src_vec = crate::error::copy_array1(src_arr)?;
            let dst_vec = crate::error::copy_array1(dst_arr)?;

            if src_vec.len() != dst_vec.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "src and dst arrays must have the same length for ({src_type}, {rel}, {dst_type})"
                )));
            }

            let num_src = *nt_counts.get(&src_type).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown source type '{src_type}' in ({src_type}, {rel}, {dst_type})"
                ))
            })?;
            let num_dst = *nt_counts.get(&dst_type).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown dest type '{dst_type}' in ({src_type}, {rel}, {dst_type})"
                ))
            })?;

            // Endpoint ranges are checked by the bipartite builder below.
            bundles.push(EdgeBundle {
                src_type,
                rel,
                dst_type,
                num_src,
                num_dst,
                src: src_vec,
                dst: dst_vec,
            });
        }

        // The builds touch only the owned arrays above; run them without
        // the GIL.
        let built = py.detach(|| {
            bundles
                .into_iter()
                .map(|b| {
                    let graph =
                        Graph::from_bipartite_src_dst(b.num_src, b.num_dst, &b.src, &b.dst, None)
                            .map_err(|e| {
                            format!(
                                "failed to build CSR for ({}, {}, {}): {e}",
                                b.src_type, b.rel, b.dst_type
                            )
                        })?;
                    Ok((b.src_type, b.rel, b.dst_type, graph))
                })
                .collect::<Result<Vec<_>, String>>()
                .and_then(|et_vec| {
                    HeteroGraph::try_from_parts(nt_vec, et_vec).map_err(|e| e.to_string())
                })
        });

        Ok(Self {
            inner: Arc::new(built.map_err(pyo3::exceptions::PyValueError::new_err)?),
        })
    }

    fn node_types(&self) -> Vec<String> {
        self.inner
            .node_type_names()
            .into_iter()
            .map(std::borrow::ToOwned::to_owned)
            .collect()
    }

    fn edge_types(&self) -> Vec<(String, String, String)> {
        self.inner
            .edge_type_names()
            .into_iter()
            .map(|(s, r, d)| (s.to_owned(), r.to_owned(), d.to_owned()))
            .collect()
    }

    fn num_nodes(&self, node_type: &str) -> PyResult<usize> {
        let id = self.inner.node_type_id(node_type).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node type '{node_type}'"))
        })?;
        Ok(self.inner.num_nodes(id))
    }

    fn num_edges(&self, src_type: &str, rel: &str, dst_type: &str) -> PyResult<usize> {
        let id = self
            .inner
            .edge_type_id(src_type, rel, dst_type)
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "unknown edge type ('{src_type}', '{rel}', '{dst_type}')"
                ))
            })?;
        Ok(self.inner.num_edges(id))
    }

    fn total_nodes(&self) -> usize {
        self.inner.total_nodes()
    }

    fn total_edges(&self) -> usize {
        self.inner.total_edges()
    }

    fn __repr__(&self) -> String {
        format!(
            "HeteroCsrGraph(node_types={}, edge_types={}, total_nodes={}, total_edges={})",
            self.inner.node_type_count(),
            self.inner.edge_type_count(),
            self.inner.total_nodes(),
            self.inner.total_edges(),
        )
    }
}

impl PyHeteroCsrGraph {
    pub fn inner_arc(&self) -> Arc<HeteroGraph> {
        Arc::clone(&self.inner)
    }
}

// ---------------------------------------------------------------------------
// HeteroSamplingConfig
// ---------------------------------------------------------------------------

#[pyclass(name = "HeteroSamplingConfig", from_py_object)]
#[derive(Clone)]
pub struct PyHeteroSamplingConfig {
    pub(crate) num_neighbors: HashMap<(String, String, String), Vec<usize>>,
    pub(crate) replace: bool,
    pub(crate) seed: Option<u64>,
    pub(crate) max_degree: Option<usize>,
}

#[pymethods]
impl PyHeteroSamplingConfig {
    /// # Arguments
    /// * `num_neighbors` — dict mapping `(src_type, rel, dst_type)` to a list of
    ///   neighbor counts (one per hop): how many `src_type` nodes with an edge
    ///   into each expanded `dst_type` node to draw, as in PyG. Edge types left
    ///   out draw nothing.
    /// * `replace` — sample with replacement (default `False`).
    /// * `seed` — optional RNG seed for reproducible sampling. `None` uses a
    ///   non-deterministic seed.
    /// * `max_degree` — accepted for parity with `SamplingConfig`; every
    ///   heterogeneous draw is uniform over the whole in-neighborhood in
    ///   O(fanout), so no cap applies.
    #[new]
    #[pyo3(signature = (num_neighbors, replace=false, seed=None, max_degree=None))]
    fn new(
        num_neighbors: HashMap<(String, String, String), Vec<usize>>,
        replace: bool,
        seed: Option<u64>,
        max_degree: Option<usize>,
    ) -> PyResult<Self> {
        if num_neighbors.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_neighbors must not be empty",
            ));
        }
        let mut hop_counts = num_neighbors.values().map(std::vec::Vec::len);
        let first_hops = hop_counts.next().unwrap_or(0);
        if first_hops == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_neighbors hop lists must be non-empty",
            ));
        }
        if hop_counts.any(|h| h != first_hops) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "all edge types must have the same number of hops",
            ));
        }
        if matches!(max_degree, Some(0)) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_degree must be > 0 if specified",
            ));
        }
        Ok(Self {
            num_neighbors,
            replace,
            seed,
            max_degree,
        })
    }

    #[getter]
    fn get_num_neighbors(&self) -> HashMap<(String, String, String), Vec<usize>> {
        self.num_neighbors.clone()
    }

    #[getter]
    fn get_replace(&self) -> bool {
        self.replace
    }

    #[getter]
    fn get_seed(&self) -> Option<u64> {
        self.seed
    }

    #[getter]
    fn get_max_degree(&self) -> Option<usize> {
        self.max_degree
    }

    #[getter]
    fn get_num_hops(&self) -> usize {
        self.num_hops()
    }

    fn __repr__(&self) -> String {
        format!(
            "HeteroSamplingConfig(edge_types={}, num_hops={}, replace={})",
            self.num_neighbors.len(),
            self.num_hops(),
            self.replace,
        )
    }
}

impl PyHeteroSamplingConfig {
    pub fn num_hops(&self) -> usize {
        self.num_neighbors
            .values()
            .next()
            .map(std::vec::Vec::len)
            .unwrap_or(0)
    }

    /// Convert Python config to core config, resolving string type names
    /// to integer IDs using the graph's metadata.
    fn to_core_config(&self, graph: &HeteroGraph) -> PyResult<HeteroSamplingConfig> {
        let num_hops = self.num_hops();
        let num_edge_types = graph.edge_type_count();
        let mut fanout = vec![vec![0usize; num_hops]; num_edge_types];

        for ((src, rel, dst), hops) in &self.num_neighbors {
            let eid = graph.edge_type_id(src, rel, dst).ok_or_else(|| {
                sampling_error(format!("unknown edge type ('{src}', '{rel}', '{dst}')"))
            })?;
            fanout[eid as usize] = hops.clone();
        }

        Ok(HeteroSamplingConfig {
            fanout,
            replace: self.replace,
            seed: self.seed,
            max_degree: self.max_degree,
            num_hops,
        })
    }
}

// ---------------------------------------------------------------------------
// HeteroSampledSubgraph
// ---------------------------------------------------------------------------

/// Wraps a core `HeteroSampledSubgraph` and lazily creates numpy arrays
/// only when Python accesses them.
#[pyclass(name = "HeteroSampledSubgraph")]
pub struct PyHeteroSampledSubgraph {
    inner: HeteroSampledSubgraph,
    graph: Arc<HeteroGraph>,
}

#[pymethods]
impl PyHeteroSampledSubgraph {
    /// List of node-type names present in the subgraph.
    #[getter]
    fn node_types(&self) -> Vec<String> {
        let mut types = Vec::new();
        for nt_id in 0..self.graph.node_type_count() as NodeTypeId {
            if !self.inner.nodes[nt_id as usize].is_empty() {
                types.push(self.graph.node_type_meta(nt_id).name.clone());
            }
        }
        types
    }

    /// List of `(src_type, relation, dst_type)` edge types in the subgraph.
    #[getter]
    fn edge_types(&self) -> Vec<(String, String, String)> {
        let mut types = Vec::new();
        for et_id in 0..self.graph.edge_type_count() as EdgeTypeId {
            if !self.inner.edge_src_local[et_id as usize].is_empty() {
                let meta = self.graph.edge_type_meta(et_id);
                let src = &self.graph.node_type_meta(meta.src_type).name;
                let dst = &self.graph.node_type_meta(meta.dst_type).name;
                types.push((src.clone(), meta.relation.clone(), dst.clone()));
            }
        }
        types
    }

    /// Returns sampled node IDs for a node type as `int64` numpy array.
    ///
    /// PyTorch indexing requires `int64`, so we widen from the u32 storage
    /// in the core sampler. The widening is a single LLVM-vectorized pass.
    fn nodes<'py>(&self, py: Python<'py>, node_type: &str) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let nt_id = self.graph.node_type_id(node_type).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node type '{node_type}'"))
        })?;
        let nodes = &self.inner.nodes[nt_id as usize];
        let arr: Vec<i64> = nodes.iter().map(|&n| i64::from(n)).collect();
        Ok(PyArray1::from_vec(py, arr))
    }

    /// Returns sampled node IDs as `uint32` numpy array (one bulk slice
    /// copy, no widening). Use this when feeding back into another aether
    /// API that expects u32 — saves a `.astype(np.uint32)` on the Python side.
    fn nodes_u32<'py>(
        &self,
        py: Python<'py>,
        node_type: &str,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let nt_id = self.graph.node_type_id(node_type).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node type '{node_type}'"))
        })?;
        Ok(PyArray1::from_slice(py, &self.inner.nodes[nt_id as usize]))
    }

    /// Returns local edge index as (2, E) i64 numpy array: row 0 indexes
    /// `nodes(src)`, row 1 `nodes(dst)`, in the edge type's stored direction.
    /// The destination is always the node that was expanded, so messages
    /// along these edges flow toward the seeds (PyG's convention).
    /// Local indices are pre-computed during sampling — no binary search.
    fn edge_index_local<'py>(
        &self,
        py: Python<'py>,
        src: &str,
        rel: &str,
        dst: &str,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let et_id = self.graph.edge_type_id(src, rel, dst).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "unknown edge type ('{src}', '{rel}', '{dst}')"
            ))
        })?;

        let et = et_id as usize;
        let src_local = &self.inner.edge_src_local[et];
        let dst_local = &self.inner.edge_dst_local[et];

        let num_edges = src_local.len();
        let mut data: Vec<i64> = Vec::with_capacity(num_edges * 2);
        data.extend(src_local.iter().map(|&x| i64::from(x)));
        data.extend(dst_local.iter().map(|&x| i64::from(x)));

        let flat = PyArray1::from_vec(py, data);
        flat.reshape([2, num_edges])
            .map_err(|e| sampling_error(format!("reshape failed: {e}")))
    }

    /// Name of the node type the sample was rooted at.
    #[getter]
    fn seed_type(&self) -> &str {
        &self.graph.node_type_meta(self.inner.seed_type).name
    }

    /// Seed node IDs as a numpy `int64` array (PyTorch index dtype).
    #[getter]
    fn seeds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let arr: Vec<i64> = self.inner.seeds.iter().map(|&s| i64::from(s)).collect();
        PyArray1::from_vec(py, arr)
    }

    /// Local indices of each seed into `nodes(seed_type)`, one per input
    /// seed (duplicates preserved). Same contract as homogeneous
    /// `SampledSubgraph.seed_indices`.
    #[getter]
    fn seed_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let arr: Vec<i64> = self
            .inner
            .seed_indices
            .iter()
            .map(|&s| i64::from(s))
            .collect();
        PyArray1::from_vec(py, arr)
    }

    fn __repr__(&self) -> String {
        format!(
            "HeteroSampledSubgraph(node_types={}, edge_types={}, seed_type='{}')",
            // Getters are still methods at the Rust ABI level — the `#[getter]`
            // attribute only changes Python-side dispatch.
            self.node_types().len(),
            self.edge_types().len(),
            self.seed_type(),
        )
    }
}

// ---------------------------------------------------------------------------
// HeteroNeighborSampler
// ---------------------------------------------------------------------------

/// Self-owning sampler bundle.
///
/// `sampler` borrows from the `HeteroGraph` reached through `arc`. The borrow
/// is erased to `'static` so the struct can be stored in a `#[pyclass]`. The
/// Arc is kept alongside so the `HeteroGraph` cannot be deallocated while the
/// sampler is alive.
///
/// # Why this is sound
/// 1. `Arc::as_ptr(&arc)` returns a stable pointer to the heap allocation
///    that lasts as long as any Arc clone exists. The Arc inside this struct
///    is one such clone.
/// 2. Field declaration order is `sampler` then `arc`. Rust drops fields in
///    declaration order, so `sampler` (which holds the borrow) drops before
///    `arc` (which holds the allocation). The borrow can never observe a
///    freed `HeteroGraph`.
/// 3. The Arc is **private**: no method exposes it or clones it out, so
///    external code cannot create an additional Arc that would extend the
///    lifetime of the allocation past `self`'s drop in a way the type system
///    cannot see.
/// 4. The sampler is never moved out of this struct after construction —
///    `sample()` takes `&mut self.sampler` only.
///
/// The two `Arc<HeteroGraph>` clones (this one and the one in
/// [`PyHeteroNeighborSampler::graph`]) are independent: dropping the latter
/// does not invalidate the former.
struct OwnedSampler {
    sampler: HeteroNeighborSampler<'static>,
    // SAFETY-LOAD-BEARING: must drop AFTER `sampler`. Marker leading underscore
    // signals "don't reorder me" to readers.
    _arc: Arc<HeteroGraph>,
}

// Compile-time field-order guard. Rust drops `#[repr(Rust)]` struct fields in
// declaration order, so the byte offset of the field that must drop first
// (`sampler`) MUST be strictly less than the offset of the field that holds
// its backing storage (`_arc`). If someone reorders these fields, this const
// fires at compile time.
const _: () = {
    let s = std::mem::offset_of!(OwnedSampler, sampler);
    let a = std::mem::offset_of!(OwnedSampler, _arc);
    assert!(
        s < a,
        "OwnedSampler field order violates the drop-before-arc invariant — \
         see the SAFETY comment on the struct"
    );
};

impl OwnedSampler {
    fn new(arc: Arc<HeteroGraph>, config: HeteroSamplingConfig) -> Self {
        // SAFETY: `Arc::as_ptr(&arc)` returns a pointer to the `HeteroGraph`
        // inside the Arc's allocation. The Arc clone we move into `_arc`
        // keeps that allocation alive for the lifetime of `Self`. The
        // sampler is dropped before `_arc` (struct field declaration order),
        // so the erased `'static` borrow never observes a deallocated graph.
        let graph_ref: &'static HeteroGraph = unsafe { &*Arc::as_ptr(&arc) };
        let sampler = HeteroNeighborSampler::new(graph_ref, config);
        Self { sampler, _arc: arc }
    }

    fn sampler_mut(&mut self) -> &mut HeteroNeighborSampler<'static> {
        &mut self.sampler
    }
}

#[cfg(test)]
mod owned_sampler_drop_order {
    use super::*;

    /// Mirror of the `const _` assertion higher up, run as a normal unit
    /// test so the failure shows up in the standard `cargo test` output
    /// (not just in `cargo build`). Catches drop-order drift even if the
    /// `const _` is accidentally deleted.
    #[test]
    fn sampler_offset_is_less_than_arc_offset() {
        let sampler = std::mem::offset_of!(OwnedSampler, sampler);
        let arc = std::mem::offset_of!(OwnedSampler, _arc);
        assert!(
            sampler < arc,
            "OwnedSampler field order: sampler offset {sampler}, _arc offset {arc} \
             — sampler must drop before _arc"
        );
    }
}

#[pyclass(name = "HeteroNeighborSampler")]
pub struct PyHeteroNeighborSampler {
    /// Owns its own Arc clone (private, never aliased outwards).
    inner: OwnedSampler,
    /// Separate Arc clone for Python-facing methods (type ID lookups etc.).
    /// Independent from `inner._arc` — both refer to the same allocation,
    /// so the graph is alive as long as either is.
    graph: Arc<HeteroGraph>,
}

#[pymethods]
impl PyHeteroNeighborSampler {
    #[new]
    fn new(graph: &PyHeteroCsrGraph, config: PyHeteroSamplingConfig) -> PyResult<Self> {
        let graph_arc = graph.inner_arc();
        let core_config = config.to_core_config(&graph_arc)?;
        Ok(Self {
            inner: OwnedSampler::new(Arc::clone(&graph_arc), core_config),
            graph: graph_arc,
        })
    }

    fn sample(
        &mut self,
        py: Python<'_>,
        seed_type: &str,
        seeds: &Bound<'_, PyAny>,
    ) -> PyResult<PyHeteroSampledSubgraph> {
        let seed_type_id = self
            .graph
            .node_type_id(seed_type)
            .ok_or_else(|| sampling_error(format!("unknown seed type '{seed_type}'")))?;

        let seeds = crate::error::extract_seed_batch(seeds, self.graph.num_nodes(seed_type_id))?;

        // Run the core sampler with the GIL released. The closure touches no
        // Python state — it walks the owned `HeteroGraph` (kept alive by the
        // sampler's Arc) and the owned `seeds`, returning an owned
        // `HeteroSampledSubgraph`.
        let sampler = self.inner.sampler_mut();
        let sub = py
            .detach(move || sampler.sample(seed_type_id, &seeds))
            .map_err(|e| sampling_error(e.to_string()))?;

        Ok(PyHeteroSampledSubgraph {
            inner: sub,
            graph: Arc::clone(&self.graph),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "HeteroNeighborSampler(edge_types={})",
            self.graph.edge_type_count(),
        )
    }
}

// ---------------------------------------------------------------------------
// HeteroNeighborLoader
// ---------------------------------------------------------------------------

/// Prefetching heterogeneous neighbor loader for pipelined GNN training.
///
/// Spawns `sampler_threads` Rust worker threads over an MPMC work queue —
/// the same pipeline as the homogeneous `NeighborLoader`. With a config
/// seed, results come back in submission order; without one, in completion
/// order across the pool. The seed node type is fixed at construction;
/// every submitted batch is rooted at it.
///
/// Every method is safe to call from any thread at any time: `shutdown()`
/// from one thread wakes others blocked in `submit()` or `next()`.
///
/// Args:
///     graph: HeteroCsrGraph to sample from
///     config: HeteroSamplingConfig with per-edge-type fanout parameters
///     seed_type: Node type name every submitted seed batch is rooted at
///     prefetch_depth: Number of batches to keep ready (default: 2)
///     sampler_threads: Sampler worker threads pulling from the shared
///         work queue (default: 1)
///
/// Example:
///     >>> loader = HeteroNeighborLoader(graph, config, "user", prefetch_depth=3)
///     >>> for i, batch in enumerate(batches):
///     ...     loader.submit(i, batch)
///     >>> for _ in range(len(batches)):
///     ...     subgraph = loader.next()  # Already ready!
///     ...     train(subgraph)
#[pyclass(name = "HeteroNeighborLoader", frozen)]
pub struct PyHeteroNeighborLoader {
    inner: HeteroNeighborLoader,
    /// Graph handle for wrapping results (type-name lookups in
    /// `PyHeteroSampledSubgraph`); the loader's workers hold their own Arcs.
    graph: Arc<HeteroGraph>,
}

impl PyHeteroNeighborLoader {
    fn wrap(&self, subgraph: HeteroSampledSubgraph) -> PyHeteroSampledSubgraph {
        PyHeteroSampledSubgraph {
            inner: subgraph,
            graph: Arc::clone(&self.graph),
        }
    }
}

#[pymethods]
impl PyHeteroNeighborLoader {
    #[new]
    #[pyo3(signature = (graph, config, seed_type, prefetch_depth=2, sampler_threads=1))]
    fn new(
        graph: &PyHeteroCsrGraph,
        config: PyHeteroSamplingConfig,
        seed_type: &str,
        prefetch_depth: usize,
        sampler_threads: usize,
    ) -> PyResult<Self> {
        let graph_arc = graph.inner_arc();
        let seed_type_id = graph_arc
            .node_type_id(seed_type)
            .ok_or_else(|| sampling_error(format!("unknown seed type '{seed_type}'")))?;
        let core_config = config.to_core_config(&graph_arc)?;

        let inner = HeteroNeighborLoader::new(
            Arc::clone(&graph_arc),
            core_config,
            seed_type_id,
            prefetch_depth,
            sampler_threads,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create HeteroNeighborLoader: {e}"
            ))
        })?;

        Ok(Self {
            inner,
            graph: graph_arc,
        })
    }

    /// Submit a batch to be sampled.
    ///
    /// `batch_idx` is caller bookkeeping: it comes back with the result from
    /// `next_batch()`, and any value is accepted, repeats and gaps included.
    ///
    /// Blocks (with the GIL released) when the pipeline is full, until the
    /// consumer drains a result — so interleave `submit()` with `next()`, or
    /// submit from a separate thread. A `shutdown()` from another thread
    /// unblocks it.
    ///
    /// Args:
    ///     batch_idx: Caller-chosen index for this batch.
    ///     seeds: Seed node IDs of the loader's seed type (numpy uint32,
    ///         numpy int64, or `list[int]`), each below that type's count.
    ///
    /// Raises:
    ///     SamplingError: A seed is out of range, or the loader was shut
    ///         down or a worker failed.
    fn submit(&self, py: Python<'_>, batch_idx: usize, seeds: &Bound<'_, PyAny>) -> PyResult<()> {
        let seeds = crate::error::extract_seed_batch(seeds, self.inner.seed_nodes())?;

        // The bounded work channel blocks when the pipeline is full; release
        // the GIL so the consumer thread can drain. No Python object is
        // touched inside.
        py.detach(|| self.inner.submit(batch_idx, seeds))
            .map_err(|e| sampling_error(format!("Submit failed: {e}")))
    }

    /// Get the next sampled subgraph (blocking).
    ///
    /// Returns:
    ///     HeteroSampledSubgraph: The next prefetched subgraph
    ///     None: Only after `shutdown()` — the pipeline was drained and no
    ///         more batches will arrive
    ///
    /// Raises:
    ///     TimeoutError: No result arrived within the wait window; the
    ///         workers may just be slow, so the call may be retried.
    ///     RuntimeError: A sampler worker failed.
    fn next(&self, py: Python<'_>) -> PyResult<Option<PyHeteroSampledSubgraph>> {
        Ok(self.next_batch(py)?.map(|(_, subgraph)| subgraph))
    }

    /// Get the next batch with its `batch_idx` (blocking).
    ///
    /// Returns:
    ///     tuple: (batch_idx, HeteroSampledSubgraph) — `batch_idx` as passed
    ///         to `submit()`
    ///     None: Only after `shutdown()`
    ///
    /// Raises:
    ///     TimeoutError: No result arrived within the wait window; the call
    ///         may be retried.
    ///     RuntimeError: A sampler worker failed.
    fn next_batch(&self, py: Python<'_>) -> PyResult<Option<(usize, PyHeteroSampledSubgraph)>> {
        // The blocking recv can stall while the workers sample; release the
        // GIL so other Python threads run. No Python object is touched inside.
        Ok(py
            .detach(|| self.inner.next_batch())
            .map_err(crate::prefetch::prefetch_error_to_py)?
            .map(|r| (r.batch_idx, self.wrap(r.subgraph))))
    }

    /// Get current prefetch statistics.
    ///
    /// Returns:
    ///     PrefetchStats: Statistics about hit rate, misses, etc.
    ///     (`feature_load_time_ns` stays 0 — this loader samples only).
    fn stats(&self) -> crate::prefetch::PyPrefetchStats {
        crate::prefetch::PyPrefetchStats::snapshot(self.inner.stats())
    }

    /// Get the prefetch depth.
    #[getter]
    fn prefetch_depth(&self) -> usize {
        self.inner.prefetch_depth()
    }

    /// Shut the sampler pool down and join its threads.
    ///
    /// Safe to call from any thread, and more than once. Called
    /// automatically when the object is garbage collected.
    fn shutdown(&self, py: Python<'_>) {
        py.detach(|| self.inner.shutdown());
    }

    fn __repr__(&self) -> String {
        format!(
            "HeteroNeighborLoader(prefetch_depth={})",
            self.inner.prefetch_depth()
        )
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) {
        self.shutdown(py);
    }
}
