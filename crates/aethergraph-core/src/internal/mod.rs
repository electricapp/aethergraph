//! Internal utilities (not part of public API).
//!
//! These modules provide low-level functionality used by the rest of the library.

pub mod hint;
pub mod mmap;
pub mod mmap_hetero;
#[cfg(feature = "parquet")]
pub mod parquet_import;
pub mod telemetry;

#[cfg(target_os = "linux")]
pub mod uring;
