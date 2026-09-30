//! Feature storage for node embeddings.
//!
//! Provides memory-mapped and async feature stores for billion-scale
//! node feature access.

mod async_store;
mod cache;
#[cfg(feature = "zstd-tier")]
mod cold_tier;
#[cfg(target_os = "linux")]
pub(crate) mod gather;
#[cfg(all(target_os = "linux", feature = "gds"))]
mod gds;
pub(crate) mod header;
#[cfg(all(target_os = "linux", feature = "shm"))]
mod shared_store;
mod store;

use crate::graph::NodeId;

pub use async_store::AsyncFeatureStore;
pub use cache::{CacheStats, FeatureCache, FeatureCacheConfig, count_node_frequencies};
#[cfg(feature = "zstd-tier")]
pub use cold_tier::{ColdStore, ColdTier, ROWS_PER_BLOCK};
#[cfg(all(target_os = "linux", feature = "gds"))]
pub use gds::{GdsFeatureStore, GdsReadResult, gds_driver_close, gds_driver_open};
pub use header::{FeatureDtype, FeatureHeader, parse_feature_header};
#[cfg(all(target_os = "linux", feature = "shm"))]
pub use shared_store::{ShareHandle, SharedFeatureStore};
pub(crate) use store::PaddedAtomicU64;
pub use store::{
    FeatureData, FeatureLoadTelemetry, FeatureStore, create_features, save_feature_data,
    save_features, save_features_bf16, save_features_f16, save_features_ndarray,
};

/// Turn `O_DIRECT` on or off for an open descriptor, in place.
///
/// Reopening by path would race a replacement of the file; flipping the
/// flag keeps I/O on the inode whose header was already validated. Turning
/// it on fails with `EINVAL` where the filesystem cannot do direct I/O.
#[cfg(target_os = "linux")]
pub(crate) fn set_direct_io(file: &std::fs::File, on: bool) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let fd = file.as_raw_fd();
    // SAFETY: `fd` is live for the call; F_GETFL takes no argument.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let flags = if on {
        flags | libc::O_DIRECT
    } else {
        flags & !libc::O_DIRECT
    };
    // SAFETY: as above; F_SETFL takes the int flag word.
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Trait for reading node features from any backing store.
///
/// Implemented by `FeatureStore` (mmap'd files) and by `FeatureTable`
/// (seqlock in HugePage RAM) in `aether-stream`.
pub trait NodeFeatureSource: Send + Sync {
    /// Number of f32 features per node.
    fn feature_dim(&self) -> usize;

    /// Read features for `node` into `out`. Returns `true` if the node
    /// exists and was read successfully.
    fn read_node(&self, node: NodeId, out: &mut [f32]) -> bool;
}
