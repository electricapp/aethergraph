//! Internal utilities (not part of public API).
//!
//! These modules provide low-level functionality used by the rest of the library.

#[cfg(target_os = "linux")]
pub mod aligned;
pub mod genstamp;
pub mod hint;
pub mod mmap;
pub mod mmap_hetero;
#[cfg(feature = "parquet")]
pub mod parquet_import;
#[cfg(all(target_os = "linux", feature = "perf"))]
pub mod perf;
pub mod prefetch;
pub mod simd;
pub mod telemetry;

#[cfg(target_os = "linux")]
pub mod uring;
