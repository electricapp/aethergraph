use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use std::fmt;

/// Custom error type for AetherGraph Python bindings
#[derive(Debug)]
pub struct AetherGraphError(pub anyhow::Error);

impl fmt::Display for AetherGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl std::error::Error for AetherGraphError {}

impl From<anyhow::Error> for AetherGraphError {
    fn from(err: anyhow::Error) -> Self {
        Self(err)
    }
}

impl From<AetherGraphError> for PyErr {
    fn from(err: AetherGraphError) -> Self {
        PyException::new_err(format!("{}", err.0))
    }
}

// Define custom Python exception types
pyo3::create_exception!(
    aethergraph,
    GraphLoadError,
    PyException,
    "Error loading graph from file"
);
pyo3::create_exception!(
    aethergraph,
    SamplingError,
    PyException,
    "Error during graph sampling"
);
pyo3::create_exception!(
    aethergraph,
    CacheError,
    PyException,
    "Error accessing feature cache"
);
pyo3::create_exception!(
    aethergraph,
    ArrowConversionError,
    PyException,
    "Error converting to Arrow format"
);

/// Helper function to convert anyhow errors to specific Python exceptions
pub fn graph_load_error(msg: impl Into<String>) -> PyErr {
    GraphLoadError::new_err(msg.into())
}

pub fn sampling_error(msg: impl Into<String>) -> PyErr {
    SamplingError::new_err(msg.into())
}

pub fn cache_error(msg: impl Into<String>) -> PyErr {
    CacheError::new_err(msg.into())
}

pub fn arrow_conversion_error(msg: impl Into<String>) -> PyErr {
    ArrowConversionError::new_err(msg.into())
}

/// Convenience macro for converting anyhow errors to PyErr
#[macro_export]
macro_rules! py_err {
    ($err:expr) => {
        $err.map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{:?}", e)))
    };
}

/// Copy a contiguous NumPy 1-d array into an owned `Vec`.
///
/// Callers that release the GIL / detach under `gil_used = false` must not
/// borrow the ndarray buffer across the detach — another free-threaded
/// Python thread can resize or free it. Own the bytes first; the copy is the
/// price of soundness on the free-threaded build (and is cheap relative to
/// the O(E)/gather work that follows).
pub fn copy_array1<T>(arr: numpy::PyReadonlyArray1<'_, T>) -> PyResult<Vec<T>>
where
    T: Copy + numpy::Element,
{
    Ok(arr.as_slice()?.to_vec())
}

/// Parse an int64 NumPy ID array into owned [`NodeId`]s in one pass.
///
/// Owns the result before any `py.detach`, and avoids a temporary `Vec<i64>`
/// plus a second conversion inside the core gather.
pub fn copy_node_ids_i64(
    arr: numpy::PyReadonlyArray1<'_, i64>,
) -> PyResult<Vec<aethergraph_core::NodeId>> {
    use aethergraph_core::NodeId;
    arr.as_slice()?
        .iter()
        .map(|&x| {
            NodeId::try_from(x).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "node id {x} out of u32 range [0, {}]",
                    u32::MAX
                ))
            })
        })
        .collect()
}

/// Coerce a Python seed argument into `Vec<u32>`.
///
/// Accepts (in order): `numpy.ndarray[uint32]` (copied), `numpy.ndarray[int64]`
/// (range-checked), or any `Sequence[int]` extractable as `Vec<u32>`. Returns a
/// [`SamplingError`] on out-of-range i64 values so the caller never silently
/// truncates negatives or values > u32::MAX.
pub fn extract_seeds(seeds: &pyo3::Bound<'_, pyo3::PyAny>) -> PyResult<Vec<u32>> {
    use numpy::PyReadonlyArray1;
    if let Ok(arr) = seeds.extract::<PyReadonlyArray1<u32>>() {
        return copy_array1(arr);
    }
    if let Ok(arr) = seeds.extract::<PyReadonlyArray1<i64>>() {
        return arr
            .as_slice()?
            .iter()
            .map(|&x| {
                u32::try_from(x).map_err(|_| {
                    sampling_error(format!("seed node {x} out of u32 range [0, {}]", u32::MAX))
                })
            })
            .collect();
    }
    seeds.extract::<Vec<u32>>()
}

// Integration-level tests for `extract_seeds` live in
// `python/tests/test_extract_seeds.py` (or equivalent) — they need a live
// Python interpreter to construct numpy / list inputs, which the cdylib
// build profile cannot provide from a `cargo test` binary without an
// `auto-initialize`-enabled pyo3 build. The pure-Rust path (i64 → u32
// range check) is covered transitively through every NeighborSampler test
// in `aethergraph_core::loader::sampler::tests`.
