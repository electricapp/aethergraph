//! Internal utilities (not part of public API).
//!
//! These modules provide low-level functionality used by the rest of the library.

#[cfg(target_os = "linux")]
pub mod aligned;
pub mod compressed_graph;
/// Pure-logic device protocol layouts and hardware-gated integration plans
/// from `KERNELS.md`.
pub mod device;
pub mod genstamp;
pub mod hint;
pub mod mmap;
pub mod mmap_hetero;
pub mod numa;
#[cfg(all(target_os = "linux", feature = "nvme-passthru"))]
pub mod nvme;
#[cfg(feature = "parquet")]
pub mod parquet_import;
#[cfg(all(target_os = "linux", feature = "perf"))]
pub mod perf;
pub mod philox;
pub mod prefetch;
pub mod probe;
pub mod seqlock;
#[cfg(all(target_os = "linux", feature = "shm"))]
pub mod shm;
pub mod simd;
pub mod succinct;
pub mod telemetry;

#[cfg(all(target_os = "linux", feature = "uffd"))]
pub mod uffd;
#[cfg(target_os = "linux")]
pub mod uring;
