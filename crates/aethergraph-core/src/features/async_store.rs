//! Async feature store with io_uring for parallel feature reads from NVMe.
//!
//! **The Problem**: TB-scale node features don't fit in RAM. Loading features for
//! sampled nodes with sync mmap blocks threads waiting for page faults (~10-100μs each).
//!
//! **The Solution**:
//! - Store features in binary format with explicit payload offset
//! - Use io_uring (Linux) to issue many parallel async reads
//! - Read feature vectors directly with aligned buffers (O_DIRECT)
//! - Parallel reads hide NVMe latency, keep GPU fed
//!
//! **Performance Tiers**:
//! 1. Linux + io_uring + SQPOLL + IOPOLL + O_DIRECT: Near-zero-syscall async I/O (best)
//!    - SQPOLL: Kernel polls SQ (reduced syscalls)
//!    - IOPOLL: Poll NVMe for completions (requires O_DIRECT)
//!    - O_DIRECT: Bypass page cache for true async
//! 2. Linux + io_uring + SQPOLL: Reduced syscalls without O_DIRECT (good)
//! 3. Tokio spawn_blocking: Multi-threaded sync fallback (works everywhere)
//!
//! **O_DIRECT Limitation**: O_DIRECT requires both file offsets and buffer addresses
//! to be aligned to the filesystem block size (typically 512 bytes). New files store
//! an aligned payload offset, but legacy files may still use offset 32. For those
//! legacy files, we fall back to SQPOLL-only mode (tier 2).

use super::header::{FeatureDtype, parse_feature_header};
use super::store::{FeatureLoadTelemetry, TelemetryCells};
use crate::graph::NodeId;
use anyhow::{Context, Result};
use half::f16;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tracing::{debug, trace};

#[cfg(target_os = "linux")]
use tracing::warn;

#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;

/// Async feature store backed by NVMe with io_uring.
///
/// Features are stored in a binary format with a fixed 32-byte metadata header and
/// an explicit payload offset (legacy files default to offset 32).
/// On Linux, uses io_uring with O_DIRECT for efficient parallel async reads.
///
/// **io_uring Performance (Linux)**:
/// - SQPOLL: Kernel polls submission queue (check NEED_WAKEUP before syscall)
/// - IOPOLL: Poll NVMe for completions (requires O_DIRECT)
/// - O_DIRECT: Bypass page cache, requires aligned buffers
/// - Registered FDs: Pre-register file descriptor for faster access
pub struct AsyncFeatureStore {
    /// File descriptor for the feature file (may be O_DIRECT on Linux)
    file: Arc<File>,

    /// Path to feature file
    #[allow(dead_code)]
    path: PathBuf,

    /// Number of nodes
    num_nodes: usize,

    /// Feature dimension per node
    feature_dim: usize,

    /// Byte offset where feature data starts in file (after header)
    features_start_offset: u64,

    /// Element data type (F32 or F16).
    dtype: FeatureDtype,

    /// Pool of io_uring lanes (ring + reusable landing buffers) for
    /// concurrent batch reads (Linux only)
    #[cfg(target_os = "linux")]
    uring_pool: Option<Arc<crate::internal::uring::RingPool<crate::internal::uring::UringLane>>>,

    /// Whether O_DIRECT is enabled (required for IOPOLL)
    #[cfg(target_os = "linux")]
    direct_io: bool,

    /// Optional NVMe passthrough backend. `Some` only when the feature is
    /// built, the store lives on an `/dev/ng*`-backed namespace, and its
    /// extents are stably mapped; consulted before the io_uring gather and
    /// silently skipped otherwise.
    #[cfg(all(target_os = "linux", feature = "nvme-passthru"))]
    nvme: Option<Arc<parking_lot::Mutex<super::gather::NvmeGather>>>,

    /// Optional telemetry collector
    telemetry: Option<Arc<TelemetryCells>>,
}

impl AsyncFeatureStore {
    /// Load feature store from disk for async access.
    ///
    /// On Linux, attempts to use io_uring with SQPOLL for reduced syscalls.
    /// O_DIRECT + IOPOLL is only used if the feature layout is aligned (512-byte).
    pub async fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        debug!("Loading async feature store from {}", path.display());

        // Open without O_DIRECT first so we can read + validate the header.
        let header_file = {
            let path_owned = path.to_path_buf();
            tokio::task::spawn_blocking(move || File::open(&path_owned))
                .await
                .context("spawn_blocking failed")??
        };
        let header = parse_feature_header(&header_file)?;

        // On Linux, check if layout is O_DIRECT compatible before trying O_DIRECT
        #[cfg(target_os = "linux")]
        let (std_file, direct_io) = {
            use crate::internal::uring::{
                DIRECT_IO_OFFSET_ALIGNMENT, direct_io_offset_alignment,
                is_layout_direct_io_compatible_with,
            };

            // The device's real requirement, not the 512-byte floor — see
            // `direct_io_offset_alignment`.
            let alignment =
                direct_io_offset_alignment(&header_file).unwrap_or(DIRECT_IO_OFFSET_ALIGNMENT);
            let layout_compatible = is_layout_direct_io_compatible_with(
                header.features_start_offset,
                header.feature_size,
                alignment,
            );

            if layout_compatible {
                // Layout is aligned, try O_DIRECT
                drop(header_file);
                let path_owned = path.to_path_buf();
                let (f, direct) = tokio::task::spawn_blocking(move || {
                    use crate::internal::uring::open_direct_or_fallback;
                    open_direct_or_fallback(&path_owned)
                })
                .await
                .context("spawn_blocking failed")??;
                if direct {
                    debug!(
                        "Feature layout is O_DIRECT compatible (offset={}, size={})",
                        header.features_start_offset, header.feature_size
                    );
                }
                (f, direct)
            } else {
                // Layout not aligned, O_DIRECT won't work. Use buffered I/O with SQPOLL only.
                debug!(
                    "Feature layout not O_DIRECT compatible: offset={}, size={}",
                    header.features_start_offset, header.feature_size
                );
                (header_file, false)
            }
        };

        #[cfg(not(target_os = "linux"))]
        let std_file = header_file;

        #[cfg(target_os = "linux")]
        debug!(
            "Feature metadata: nodes={}, dims={}, O_DIRECT={}",
            header.num_nodes, header.feature_dim, direct_io
        );

        #[cfg(not(target_os = "linux"))]
        debug!(
            "Feature metadata: nodes={}, dims={}",
            header.num_nodes, header.feature_dim
        );

        let file_arc = Arc::new(std_file);

        // Initialize io_uring pool (Linux only)
        #[cfg(target_os = "linux")]
        let uring_pool = Self::setup_uring(&file_arc, direct_io);

        // Try the NVMe passthrough backend (feature-gated, runtime-probed).
        // Only meaningful for the O_DIRECT layout: the device reads share
        // the same LBA-alignment the direct path already guarantees.
        #[cfg(all(target_os = "linux", feature = "nvme-passthru"))]
        let nvme = if direct_io {
            match super::gather::NvmeGather::build(&file_arc) {
                Ok(Some(g)) => {
                    debug!("NVMe passthrough gather enabled");
                    Some(Arc::new(parking_lot::Mutex::new(g)))
                }
                Ok(None) => None,
                Err(e) => {
                    debug!("NVMe passthrough unavailable: {e}");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            file: file_arc,
            path: path.to_path_buf(),
            num_nodes: header.num_nodes,
            feature_dim: header.feature_dim,
            features_start_offset: header.features_start_offset,
            dtype: header.dtype,
            #[cfg(target_os = "linux")]
            uring_pool,
            #[cfg(target_os = "linux")]
            direct_io,
            #[cfg(all(target_os = "linux", feature = "nvme-passthru"))]
            nvme,
            telemetry: None,
        })
    }

    /// Build one uring lane per CPU (clamped to 1..=4) so concurrent
    /// `get_batch` calls can interleave across lanes; each ring is
    /// pre-registered with `file` so the hot path only uses the fixed-fd
    /// form, and each lane carries its own reusable landing buffers.
    ///
    /// Each lane's ring is created on the thread that will own it — see
    /// [`RingPool`] — which is what lets it ask for `SINGLE_ISSUER` and
    /// `DEFER_TASKRUN`, and what removes the blocking-pool hop the shared
    /// version needed.
    #[cfg(target_os = "linux")]
    fn setup_uring(
        file: &Arc<File>,
        direct_io: bool,
    ) -> Option<Arc<crate::internal::uring::RingPool<crate::internal::uring::UringLane>>> {
        use crate::internal::uring::{RingPool, UringLane, create_owned_feature_uring};

        let pool_size = std::thread::available_parallelism()
            .map(|n| n.get().clamp(1, 4))
            .unwrap_or(1);
        let file = Arc::clone(file);
        let pool = RingPool::new(pool_size, "aethergraph-feat-ring", move |idx| {
            let mut handle = create_owned_feature_uring(direct_io)?;
            if let Err(e) = handle.register_fd(&file) {
                warn!("Failed to register FD on handle {}: {}", idx, e);
            }
            Some(UringLane::new(handle))
        })?;
        debug!("Initialized io_uring pool with {} lane(s)", pool.lanes());
        Some(Arc::new(pool))
    }

    /// Enable telemetry tracking
    pub fn with_telemetry(mut self) -> Self {
        self.telemetry = Some(Arc::new(TelemetryCells::default()));
        self
    }

    /// Get telemetry statistics
    pub fn telemetry(&self) -> Option<FeatureLoadTelemetry> {
        self.telemetry.as_ref().map(|t| t.snapshot())
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get feature dimension
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Get features for a single node (async).
    ///
    /// For single-node reads, a plain pread is used (io_uring overhead
    /// isn't worth it), wrapped in `spawn_blocking` so a cold NVMe read
    /// doesn't stall a tokio worker thread. For batch reads, use
    /// `get_batch()` which leverages io_uring.
    pub async fn get(&self, node: NodeId) -> Result<Vec<f32>> {
        let start = self.telemetry.is_some().then(Instant::now);

        anyhow::ensure!(
            (node as usize) < self.num_nodes,
            "node {} out of bounds",
            node
        );

        let feature_size = self.feature_dim * self.dtype.element_size();
        let offset = self.features_start_offset + (node as u64 * feature_size as u64);
        let file = Arc::clone(&self.file);
        let dtype = self.dtype;

        let feature_dim = self.feature_dim;
        let features: Vec<f32> = tokio::task::spawn_blocking(move || {
            let features: Vec<f32> = match dtype {
                FeatureDtype::F32 => {
                    // Read straight into an aligned f32 buffer: on little-endian
                    // targets the bytes are already native order, so this is a
                    // bulk copy with no per-element decode.
                    let mut features = vec![0f32; feature_dim];
                    file.read_exact_at(bytemuck::cast_slice_mut(&mut features), offset)
                        .context("sync read failed")?;
                    features
                }
                FeatureDtype::F16 => {
                    let mut buffer = vec![0u8; feature_size];
                    file.read_exact_at(&mut buffer, offset)
                        .context("sync read failed")?;
                    buffer
                        .chunks_exact(2)
                        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                        .collect()
                }
            };
            Ok::<_, anyhow::Error>(features)
        })
        .await
        .context("spawn_blocking failed")??;

        // Track telemetry
        if let (Some(stats), Some(start)) = (self.telemetry.as_deref(), start) {
            stats.single_gets.fetch_add(1, Ordering::Relaxed);
            stats.total_nodes_loaded.fetch_add(1, Ordering::Relaxed);
            stats
                .total_features_loaded
                .fetch_add(self.feature_dim as u64, Ordering::Relaxed);
            stats
                .total_bytes_loaded
                .fetch_add(feature_size as u64, Ordering::Relaxed);
            stats
                .get_time_ns
                .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        Ok(features)
    }

    /// Get features for multiple nodes in batch (async, parallel).
    ///
    /// # Performance
    /// - Linux io_uring + O_DIRECT: Parallel reads with aligned buffers, ~100μs for 1000 nodes
    /// - Tokio fallback: Parallel spawn_blocking reads
    pub async fn get_batch(&self, nodes: &[NodeId]) -> Result<Vec<f32>> {
        let start = self.telemetry.is_some().then(Instant::now);

        trace!("Batch loading {} node features (async)", nodes.len());

        let feature_size = self.feature_dim * self.dtype.element_size();
        let all_features: Vec<f32>;

        #[cfg(target_os = "linux")]
        {
            // The pool round-robins across its own lanes, so there is no
            // index to carry here.
            if let Some(ref uring_pool) = self.uring_pool {
                all_features = self
                    .batch_read_uring_blocking(nodes, feature_size, uring_pool)
                    .await?;
            } else {
                self.prefetch_batch_range(nodes, feature_size);
                all_features = self.batch_read_tokio(nodes, feature_size).await?;
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            self.prefetch_batch_range(nodes, feature_size);
            all_features = self.batch_read_tokio(nodes, feature_size).await?;
        }

        // Track telemetry
        if let (Some(stats), Some(start)) = (self.telemetry.as_deref(), start) {
            stats.batch_gets.fetch_add(1, Ordering::Relaxed);
            stats
                .total_nodes_loaded
                .fetch_add(nodes.len() as u64, Ordering::Relaxed);
            stats
                .total_features_loaded
                .fetch_add((nodes.len() * self.feature_dim) as u64, Ordering::Relaxed);
            stats.total_bytes_loaded.fetch_add(
                all_features.len() as u64 * std::mem::size_of::<f32>() as u64,
                Ordering::Relaxed,
            );
            stats
                .batch_time_ns
                .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        Ok(all_features)
    }

    /// Issue one prefetch hint spanning the min/max node range in this batch,
    /// but only when the batch is dense in that range.
    ///
    /// A WILLNEED hint forces readahead of the entire span. For a random
    /// batch over a large file (the normal GNN sampling case) the
    /// min..max span approximates the whole file, so an unconditional
    /// hint would fault in gigabytes of unwanted pages and evict the
    /// page cache — strictly worse than no hint.
    fn prefetch_batch_range(&self, nodes: &[NodeId], feature_size: usize) {
        const MAX_SPAN_FACTOR: usize = 4;
        if let (Some(&min_node), Some(&max_node)) = (nodes.iter().min(), nodes.iter().max()) {
            // Defensive: min <= max guaranteed by min()/max(), but check anyway
            if min_node <= max_node
                && let (Some(min_offset), Some(range_len), Some(batch_bytes)) = (
                    self.features_start_offset
                        .checked_add((min_node as u64).saturating_mul(feature_size as u64)),
                    (max_node as usize)
                        .checked_sub(min_node as usize)
                        .and_then(|d| d.checked_add(1))
                        .and_then(|n| n.checked_mul(feature_size)),
                    nodes.len().checked_mul(feature_size),
                )
                && range_len <= batch_bytes.saturating_mul(MAX_SPAN_FACTOR)
            {
                crate::internal::hint::prefetch_file_range(&*self.file, min_offset, range_len);
            }
        }
    }

    /// Batch reads via io_uring, run in spawn_blocking to not block tokio runtime.
    ///
    /// The gather itself is [`super::gather::uring_gather_rows`], shared
    /// with `SyncFeatureStore`: the lane's persistent buffers land the
    /// reads (no per-batch aligned allocation), one pipelined `batch_read`
    /// call submits the whole batch, and rows decode straight into the
    /// output vector. Concurrent batches interleave across the lanes of
    /// the pool rather than time-slicing one ring.
    #[cfg(target_os = "linux")]
    async fn batch_read_uring_blocking(
        &self,
        nodes: &[NodeId],
        feature_size: usize,
        pool: &Arc<crate::internal::uring::RingPool<crate::internal::uring::UringLane>>,
    ) -> Result<Vec<f32>> {
        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Validate nodes - O(n) to find max, O(1) to check bounds
        // More efficient than O(n) individual checks with potential error allocation
        if let Some(&max_node) = nodes.iter().max() {
            anyhow::ensure!(
                (max_node as usize) < self.num_nodes,
                "node {} out of bounds (num_nodes={})",
                max_node,
                self.num_nodes
            );
        }

        // Clone what the lane thread needs to own.
        let file = Arc::clone(&self.file);
        let nodes = nodes.to_vec();
        let features_start_offset = self.features_start_offset;
        let direct_io = self.direct_io;
        let dtype = self.dtype;
        let feature_dim = self.feature_dim;
        #[cfg(feature = "nvme-passthru")]
        let nvme = self.nvme.clone();

        // Runs on the thread that owns the ring, so nothing here occupies a
        // tokio blocking-pool slot and the ring keeps its single submitter.
        let features = pool
            .submit(move |lane| {
                // NVMe passthrough first when available: it lands rows in the
                // lane's own aligned pool, so a fall-through to the io_uring
                // gather reuses the same buffers. `Ok(None)` means a row wasn't
                // LBA-resolvable — take the io_uring path for the whole batch.
                #[cfg(feature = "nvme-passthru")]
                if direct_io && let Some(nvme) = &nvme {
                    let slots = lane.direct_pool(nodes.len(), feature_size)?;
                    let mut backend = nvme.lock();
                    if let Some(features) = backend.gather(
                        slots,
                        &nodes,
                        features_start_offset,
                        feature_size,
                        dtype,
                        feature_dim,
                    )? {
                        return Ok(features);
                    }
                }

                super::gather::uring_gather_rows(
                    lane,
                    file.as_raw_fd(),
                    &nodes,
                    features_start_offset,
                    feature_size,
                    direct_io,
                    dtype,
                    feature_dim,
                )
            })
            .await
            .context("io_uring lane thread stopped before answering")??;

        Ok(features)
    }

    /// Tokio fallback implementation - async with spawn_blocking
    ///
    /// Splits the batch into one contiguous chunk per available core and
    /// runs a pread loop inside each `spawn_blocking` task. One task per
    /// node would put a blocking-pool dispatch (~µs) and a Vec allocation
    /// in front of every read — for a 1000-node batch that overhead
    /// dominates the preads whenever pages are cached.
    async fn batch_read_tokio(&self, nodes: &[NodeId], feature_size: usize) -> Result<Vec<f32>> {
        use tokio::task;

        trace!("Using tokio fallback for batch feature read (not io_uring)");

        if nodes.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(&max_node) = nodes.iter().max() {
            anyhow::ensure!(
                (max_node as usize) < self.num_nodes,
                "node {} out of bounds (num_nodes={})",
                max_node,
                self.num_nodes
            );
        }

        let dtype = self.dtype;
        let feature_dim = self.feature_dim;
        let parallelism = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(nodes.len());
        let chunk_len = nodes.len().div_ceil(parallelism);

        let tasks: Vec<_> = nodes
            .chunks(chunk_len)
            .map(|chunk| {
                let chunk: Vec<NodeId> = chunk.to_vec();
                let file = Arc::clone(&self.file);
                let offset = self.features_start_offset;

                task::spawn_blocking(move || {
                    let mut features = Vec::with_capacity(chunk.len() * feature_dim);
                    match dtype {
                        FeatureDtype::F32 => {
                            // Read each row straight into an aligned f32 buffer
                            // (bulk copy on little-endian) and append it.
                            let mut row = vec![0f32; feature_dim];
                            for &node in &chunk {
                                let byte_offset = offset + (node as u64 * feature_size as u64);
                                file.read_exact_at(bytemuck::cast_slice_mut(&mut row), byte_offset)
                                    .context("failed to read features")?;
                                features.extend_from_slice(&row);
                            }
                        }
                        FeatureDtype::F16 => {
                            let mut buffer = vec![0u8; feature_size];
                            for &node in &chunk {
                                let byte_offset = offset + (node as u64 * feature_size as u64);
                                file.read_exact_at(&mut buffer, byte_offset)
                                    .context("failed to read features")?;
                                features.extend(buffer.chunks_exact(2).map(|chunk| {
                                    f16::from_le_bytes([chunk[0], chunk[1]]).to_f32()
                                }));
                            }
                        }
                    }
                    Ok::<_, anyhow::Error>(features)
                })
            })
            .collect();

        // Collect results in order
        let mut all_features = Vec::with_capacity(nodes.len() * self.feature_dim);
        for task in tasks {
            let features = task.await.context("blocking task failed")??;
            all_features.extend(features);
        }

        Ok(all_features)
    }

    /// Batch sync reads (sequential) - used internally for synchronous contexts
    #[allow(dead_code)]
    fn batch_read_sync(&self, nodes: &[NodeId], feature_size: usize) -> Result<Vec<f32>> {
        // Validate all nodes upfront - O(n) to find max, O(1) to check
        if let Some(&max_node) = nodes.iter().max() {
            anyhow::ensure!(
                (max_node as usize) < self.num_nodes,
                "node {} out of bounds (num_nodes={})",
                max_node,
                self.num_nodes
            );
        }

        let mut all_features = Vec::with_capacity(nodes.len() * self.feature_dim);

        for &node in nodes {
            let offset = self.features_start_offset + (node as u64 * feature_size as u64);
            let mut buffer = vec![0u8; feature_size];

            self.file
                .read_exact_at(&mut buffer, offset)
                .context("sync read failed")?;

            match self.dtype {
                FeatureDtype::F32 => {
                    for chunk in buffer.chunks_exact(4) {
                        all_features
                            .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                }
                FeatureDtype::F16 => {
                    for chunk in buffer.chunks_exact(2) {
                        all_features.push(f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
                    }
                }
            }
        }

        Ok(all_features)
    }
}
