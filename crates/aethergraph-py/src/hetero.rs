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

#[pymethods]
impl PyHeteroCsrGraph {
    #[staticmethod]
    fn from_edge_arrays(
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
        let mut et_vec: Vec<(String, String, String, Graph)> = Vec::with_capacity(edge_types.len());

        for (src_type, rel, dst_type, src_arr, dst_arr) in edge_types {
            let src_slice = src_arr.as_slice()?;
            let dst_slice = dst_arr.as_slice()?;

            if src_slice.len() != dst_slice.len() {
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

            // Validate that every endpoint falls within the declared per-type
            // node count. Without this, an out-of-range edge would silently
            // become an unreachable phantom node in the CSR and reads of that
            // node would return empty neighbor lists — confusing failure.
            if let Some(&bad) = src_slice.iter().find(|&&s| (s as usize) >= num_src) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "src node {bad} >= num_{src_type} ({num_src}) in ({src_type}, {rel}, {dst_type})"
                )));
            }
            if let Some(&bad) = dst_slice.iter().find(|&&d| (d as usize) >= num_dst) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "dst node {bad} >= num_{dst_type} ({num_dst}) in ({src_type}, {rel}, {dst_type})"
                )));
            }

            let csr_num_nodes = num_src.max(num_dst);
            let edges: Vec<(NodeId, NodeId)> = src_slice
                .iter()
                .zip(dst_slice.iter())
                .map(|(&s, &d)| (s, d))
                .collect();

            let graph = Graph::from_edges(csr_num_nodes, &edges, None).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "failed to build CSR for ({src_type}, {rel}, {dst_type}): {e}"
                ))
            })?;

            et_vec.push((src_type, rel, dst_type, graph));
        }

        Ok(Self {
            inner: Arc::new(HeteroGraph::from_parts(nt_vec, et_vec)),
        })
    }

    fn node_types(&self) -> Vec<String> {
        self.inner
            .node_type_names()
            .into_iter()
            .map(|s| s.to_owned())
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
    ///   neighbor counts (one per hop).
    /// * `replace` — sample with replacement (default `False`).
    /// * `seed` — optional RNG seed for reproducible sampling. `None` uses a
    ///   non-deterministic seed.
    /// * `max_degree` — hard cap on per-node neighbor count. `None` disables
    ///   the cap and lets `num_neighbors` decide alone; values > `max_degree`
    ///   are clamped during sampling.
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
        let mut hop_counts = num_neighbors.values().map(|v| v.len());
        let first_hops = hop_counts.next().unwrap_or(0);
        if hop_counts.any(|h| h != first_hops) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "all edge types must have the same number of hops",
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
            .map(|v| v.len())
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
        let arr: Vec<i64> = nodes.iter().map(|&n| n as i64).collect();
        Ok(PyArray1::from_vec(py, arr))
    }

    /// Returns sampled node IDs as `uint32` numpy array (zero-copy slice copy,
    /// no widening). Use this when feeding back into another aether API that
    /// expects u32 — saves a `.astype(np.uint32)` on the Python side.
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

    /// Returns local edge index as (2, E) i64 numpy array.
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
        data.extend(src_local.iter().map(|&x| x as i64));
        data.extend(dst_local.iter().map(|&x| x as i64));

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
        let arr: Vec<i64> = self.inner.seeds.iter().map(|&s| s as i64).collect();
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

        let seeds_vec: Vec<u32> = crate::error::extract_seeds(seeds)?;

        // Run the core sampler with the GIL released. The closure touches no
        // Python state — it walks the owned `HeteroGraph` (kept alive by the
        // sampler's Arc) and the plain `seeds_vec`, returning an owned
        // `HeteroSampledSubgraph`.
        let sampler = self.inner.sampler_mut();
        let sub = py.detach(move || sampler.sample_neighbors(seed_type_id, &seeds_vec));

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
