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
        AetherGraphError(err)
    }
}

impl From<AetherGraphError> for PyErr {
    fn from(err: AetherGraphError) -> PyErr {
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
