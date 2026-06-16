//! Prefetching sampler for pipelined GNN training.
//!
//! This module provides a prefetching layer that samples batches ahead of time
//! in a dedicated thread, ensuring the training loop never waits for the sampler.
//!
//! # Architecture
//!
//! ```text
//!                        Dedicated Prefetch Thread
//!                        ┌─────────────────────────────────────────────────────┐
//!                        │                                                     │
//! ┌──────────┐  seeds    │  ┌─────────┐    ┌────────────┐    ┌────────────┐   │  results   ┌──────────┐
//! │ Python   │ ─────────►│  │ Work    │ ─► │ Sample     │ ─► │ Load       │   │ ─────────► │ Python   │
//! │ submit() │           │  │ Queue   │    │ Subgraph   │    │ Features   │   │            │ next()   │
//! └──────────┘           │  └─────────┘    └────────────┘    └────────────┘   │            └──────────┘
//!                        │                      │                  │          │
//!                        │                  io_uring           io_uring       │
//!                        │                 (NVMe graph)      (NVMe features)  │
//!                        └────────────────────────────────────────────────────┘
//!                              No tokio. No async. Just threads + io_uring.
//! ```
//!
//! # io_uring Performance Tiers
//!
//! 1. **SQPOLL + IOPOLL + O_DIRECT**: True zero-syscall I/O (best)
//!    - Kernel thread polls submission queue
//!    - IOPOLL polls NVMe for completions (requires O_DIRECT)
//!    - Only wake kernel thread if it went to sleep (NEED_WAKEUP flag)
//!
//! 2. **SQPOLL without IOPOLL**: Reduced syscalls
//!    - Used when O_DIRECT not available (tmpfs, network FS)
//!    - Still benefits from kernel SQ polling
//!
//! 3. **Standard io_uring**: Async batched I/O
//!    - Falls back when SQPOLL unavailable (permissions, old kernel)
//!    - Still faster than sequential sync I/O due to batching

use super::sampler::{NeighborSampler, SampledSubgraph, SamplingConfig};
use crate::features::header::{FeatureDtype, parse_feature_header};
use crate::graph::{Graph, NodeId};
use crate::internal::hint;
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use half::f16;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use tracing::{debug, trace, warn};

use std::os::unix::fs::FileExt;
#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;

#[cfg(target_os = "linux")]
const GRAPH_MAGIC: u32 = 0x4145_5448;
#[cfg(target_os = "linux")]
const GRAPH_VERSION: u32 = 1;
/// Graph file header is 32 bytes: magic(u32) + version(u32) + num_nodes(u64)
/// + num_edges(u64) + reserved(8 bytes).
#[cfg(target_os = "linux")]
const GRAPH_HEADER_SIZE: u64 = 32;
#[cfg(target_os = "linux")]
const MAX_GRAPH_NODES: u64 = 10_000_000_000;
#[cfg(target_os = "linux")]
const MAX_GRAPH_EDGES: u64 = 100_000_000_000;

/// Upper bound on a single readahead hint span. A random batch over a large
/// feature file has a min..max span approximating the whole file, so the hint
/// is clamped here to avoid faulting in (and evicting) gigabytes of pages.
const PREFETCH_SPAN_CAP_BYTES: u64 = 8 * 1024 * 1024;

/// Work item for the prefetch thread.
#[derive(Debug)]
pub struct PrefetchWork {
    /// Batch index (for ordering)
    pub batch_idx: usize,
    /// Seed nodes for this batch
    pub seeds: Vec<NodeId>,
}

/// Completed prefetch result.
#[derive(Debug)]
pub struct PrefetchResult {
    /// Batch index
    pub batch_idx: usize,
    /// Sampled subgraph
    pub subgraph: SampledSubgraph,
    /// Optional features for all nodes in the subgraph (flattened: num_nodes * feature_dim)
    pub features: Option<Vec<f32>>,
    /// Feature dimension (if features are loaded)
    pub feature_dim: Option<usize>,
}

/// Sync feature store for use in prefetch thread.
///
/// Unlike AsyncFeatureStore, this is designed for synchronous access
/// from the prefetch thread, optionally using io_uring for parallel reads.
///
/// On Linux, uses O_DIRECT with aligned buffers for true async NVMe I/O.
pub struct SyncFeatureStore {
    /// File handle (may be O_DIRECT on Linux)
    file: Arc<File>,
    /// Path to feature file
    #[allow(dead_code)]
    path: PathBuf,
    /// Number of nodes
    num_nodes: usize,
    /// Feature dimension per node
    feature_dim: usize,
    /// Byte offset where feature data starts
    features_start_offset: u64,
    /// Element data type (F32 or F16).
    dtype: FeatureDtype,
    /// io_uring handle with SQPOLL-aware submission (Linux only)
    #[cfg(target_os = "linux")]
    uring: Option<crate::internal::uring::UringHandle>,
    /// Whether O_DIRECT is enabled (required for IOPOLL)
    #[cfg(target_os = "linux")]
    direct_io: bool,
}

impl SyncFeatureStore {
    /// Load feature store from disk.
    ///
    /// On Linux, attempts to open with O_DIRECT for use with io_uring IOPOLL.
    /// Falls back gracefully if:
    /// - O_DIRECT is not supported (tmpfs, network FS)
    /// - Feature layout isn't O_DIRECT compatible (unaligned offsets)
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        debug!("Loading sync feature store from {}", path.display());

        // Open without O_DIRECT first so we can read + validate the header.
        let header_file = File::open(path)?;
        let header = parse_feature_header(&header_file)?;

        // On Linux, check if layout is O_DIRECT compatible before trying O_DIRECT
        #[cfg(target_os = "linux")]
        let (file, direct_io) = {
            use crate::internal::uring::{is_layout_direct_io_compatible, open_direct_or_fallback};

            let layout_compatible =
                is_layout_direct_io_compatible(header.features_start_offset, header.feature_size);

            if layout_compatible {
                // Layout is aligned, try O_DIRECT
                drop(header_file);
                let (f, direct) = open_direct_or_fallback(path)?;
                if direct {
                    debug!(
                        "Feature layout is O_DIRECT compatible (offset={}, size={})",
                        header.features_start_offset, header.feature_size
                    );
                }
                (f, direct)
            } else {
                // Layout not aligned, O_DIRECT would fail with EINVAL
                warn!(
                    "Feature layout not O_DIRECT compatible: offset={} (aligned={}), size={} (aligned={})",
                    header.features_start_offset,
                    header.features_start_offset % 512 == 0,
                    header.feature_size,
                    header.feature_size % 512 == 0
                );
                (header_file, false)
            }
        };

        #[cfg(not(target_os = "linux"))]
        let file = header_file;

        debug!(
            "Feature store: {} nodes, {} dims, data_offset={}, O_DIRECT={}",
            header.num_nodes,
            header.feature_dim,
            header.features_start_offset,
            cfg!(target_os = "linux") && {
                #[cfg(target_os = "linux")]
                {
                    direct_io
                }
                #[cfg(not(target_os = "linux"))]
                {
                    false
                }
            }
        );

        // Setup io_uring on Linux
        #[cfg(target_os = "linux")]
        let uring = Self::setup_uring(&file, direct_io);

        Ok(Self {
            file: Arc::new(file),
            path: path.to_path_buf(),
            num_nodes: header.num_nodes,
            feature_dim: header.feature_dim,
            features_start_offset: header.features_start_offset,
            dtype: header.dtype,
            #[cfg(target_os = "linux")]
            uring,
            #[cfg(target_os = "linux")]
            direct_io,
        })
    }

    #[cfg(target_os = "linux")]
    fn setup_uring(file: &File, direct_io: bool) -> Option<crate::internal::uring::UringHandle> {
        let mut handle = crate::internal::uring::create_feature_uring(direct_io)?;
        if let Err(e) = handle.register_fd(file) {
            warn!("Failed to register FD: {}", e);
        }
        Some(handle)
    }

    /// Get feature dimension
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get file handle (for prefetch hints)
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Get features start offset (for prefetch hints)
    pub fn features_start_offset(&self) -> u64 {
        self.features_start_offset
    }

    /// Issue prefetch hint for a set of nodes (for lookahead).
    ///
    /// The min..max span is computed in `u64` and clamped to a few MB so a
    /// single low+high node pair can't hint the whole file (which would fault
    /// in gigabytes and evict the page cache).
    pub fn prefetch_nodes(&self, nodes: &[NodeId]) {
        let feature_size = self.feature_dim * self.dtype.element_size();
        if let (Some(&min), Some(&max)) = (nodes.iter().min(), nodes.iter().max()) {
            let offset = self.features_start_offset + (min as u64 * feature_size as u64);
            let span_rows = (max as u64 - min as u64) + 1;
            let len = span_rows
                .saturating_mul(feature_size as u64)
                .min(PREFETCH_SPAN_CAP_BYTES);
            hint::prefetch_file_range(&*self.file, offset, len as usize);
        }
    }

    /// Load features for multiple nodes (sync, uses io_uring on Linux).
    ///
    /// With O_DIRECT + io_uring IOPOLL, this achieves near-zero-syscall I/O.
    pub fn get_batch(&mut self, nodes: &[NodeId]) -> anyhow::Result<Vec<f32>> {
        let feature_size = self.feature_dim * self.dtype.element_size();

        #[cfg(target_os = "linux")]
        {
            if self.uring.is_some() {
                // Take io_uring handle out temporarily to avoid borrow conflict
                let mut handle = self.uring.take().unwrap();
                let result = Self::batch_read_uring_aligned(
                    &self.file,
                    nodes,
                    feature_size,
                    self.num_nodes,
                    self.features_start_offset,
                    &mut handle,
                    self.direct_io,
                    self.dtype,
                );
                self.uring = Some(handle);
                return result;
            }
        }

        // Sync fallback: issue single prefetch hint for the range, computed in
        // u64 and clamped so a low+high node pair can't hint the whole file.
        if let (Some(&min_node), Some(&max_node)) = (nodes.iter().min(), nodes.iter().max()) {
            let min_offset = self.features_start_offset + (min_node as u64 * feature_size as u64);
            let span_rows = (max_node as u64 - min_node as u64) + 1;
            let range_len = span_rows
                .saturating_mul(feature_size as u64)
                .min(PREFETCH_SPAN_CAP_BYTES);
            hint::prefetch_file_range(&*self.file, min_offset, range_len as usize);
        }

        self.batch_read_sync(nodes, feature_size)
    }

    /// Batch read using io_uring with proper SQPOLL and aligned buffers.
    #[cfg(target_os = "linux")]
    #[allow(clippy::too_many_arguments)]
    fn batch_read_uring_aligned(
        file: &File,
        nodes: &[NodeId],
        feature_size: usize,
        num_nodes: usize,
        features_start_offset: u64,
        handle: &mut crate::internal::uring::UringHandle,
        direct_io: bool,
        dtype: FeatureDtype,
    ) -> anyhow::Result<Vec<f32>> {
        use crate::internal::uring::{AlignedBufferPool, batch_read};

        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Validate all nodes first
        for &node in nodes {
            if node as usize >= num_nodes {
                anyhow::bail!("node {} out of bounds (max {})", node, num_nodes);
            }
        }

        let fd = file.as_raw_fd();

        // Allocate aligned buffers for O_DIRECT, or regular buffers otherwise
        let mut aligned_pool: Option<AlignedBufferPool> = None;
        let mut regular_buffer: Vec<u8> = Vec::new();

        if direct_io {
            // O_DIRECT requires aligned buffers
            aligned_pool = Some(AlignedBufferPool::try_new(nodes.len(), feature_size)?)
        } else {
            // Regular I/O can use any buffer
            let total_size = nodes.len().checked_mul(feature_size).ok_or_else(|| {
                anyhow::anyhow!(
                    "buffer size overflow: {} nodes * {} bytes",
                    nodes.len(),
                    feature_size
                )
            })?;
            regular_buffer = vec![0u8; total_size];
        }

        // Build list of reads: (file_offset, buffer_ptr, length)
        let mut reads: Vec<(u64, *mut u8, usize)> = Vec::with_capacity(nodes.len());
        for (i, &node) in nodes.iter().enumerate() {
            let file_offset = features_start_offset + (node as u64 * feature_size as u64);
            let buf_ptr = if direct_io {
                aligned_pool.as_mut().unwrap().slot_ptr(i)
            } else {
                // SAFETY: regular_buffer is `nodes.len() * feature_size` bytes; `i < nodes.len()`.
                unsafe { regular_buffer.as_mut_ptr().add(i * feature_size) }
            };
            reads.push((file_offset, buf_ptr, feature_size));
        }

        // SAFETY: each buf_ptr is valid for `feature_size` bytes (allocated above);
        // they outlive this call.
        batch_read(handle, fd, &reads)?;

        // Convert buffer data to f32
        let decode = |bytes: &[u8]| -> Vec<f32> {
            match dtype {
                FeatureDtype::F32 => bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                FeatureDtype::F16 => bytes
                    .chunks_exact(2)
                    .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
            }
        };

        let features: Vec<f32> = if direct_io {
            let pool = aligned_pool.as_ref().unwrap();
            (0..nodes.len())
                .flat_map(|i| decode(pool.slot_slice(i, feature_size)))
                .collect()
        } else {
            decode(&regular_buffer)
        };

        Ok(features)
    }

    /// Sync batch read fallback.
    fn batch_read_sync(&self, nodes: &[NodeId], feature_size: usize) -> anyhow::Result<Vec<f32>> {
        let mut all_features = Vec::with_capacity(nodes.len() * self.feature_dim);

        for &node in nodes {
            if node as usize >= self.num_nodes {
                anyhow::bail!("node {} out of bounds", node);
            }

            let offset = self.features_start_offset + (node as u64 * feature_size as u64);
            let mut buffer = vec![0u8; feature_size];

            self.file.read_exact_at(&mut buffer, offset)?;

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

/// Lock-free statistics.
#[derive(Debug, Default)]
pub struct PrefetchStats {
    /// Batches immediately available (no wait)
    pub hits: AtomicU64,
    /// Consumer had to wait
    pub misses: AtomicU64,
    /// Total batches processed
    pub total: AtomicU64,
    /// Cumulative nanoseconds spent sampling
    pub sample_time_ns: AtomicU64,
    /// Cumulative nanoseconds spent loading features
    pub feature_load_time_ns: AtomicU64,
}

impl PrefetchStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            1.0
        } else {
            self.hits.load(Ordering::Relaxed) as f64 / total as f64
        }
    }

    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.total.store(0, Ordering::Relaxed);
        self.sample_time_ns.store(0, Ordering::Relaxed);
        self.feature_load_time_ns.store(0, Ordering::Relaxed);
    }
}

/// Message passed from sampler thread to feature loader thread.
struct SampledWork {
    work: PrefetchWork,
    subgraph: SampledSubgraph,
}

/// Sampler-only worker loop for the two-thread pipeline.
fn worker_loop_sampler(
    graph: Arc<Graph>,
    config: SamplingConfig,
    work_rx: Receiver<PrefetchWork>,
    sample_tx: Sender<SampledWork>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<PrefetchStats>,
) {
    debug!("Sampler thread started");
    let mut sampler = NeighborSampler::new(&graph, config);

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let work = match work_rx.recv() {
            Ok(w) => w,
            Err(_) => break,
        };

        trace!(
            batch_idx = work.batch_idx,
            seeds = work.seeds.len(),
            "sampling"
        );
        let t0 = std::time::Instant::now();
        let subgraph = sampler.sample_neighbors(&work.seeds);
        let elapsed_ns = t0.elapsed().as_nanos() as u64;
        stats
            .sample_time_ns
            .fetch_add(elapsed_ns, Ordering::Relaxed);

        let msg = SampledWork { work, subgraph };

        if sample_tx.send(msg).is_err() {
            break;
        }
    }

    debug!("Sampler thread stopped");
}

/// Feature-loader worker loop for the two-thread pipeline.
fn worker_loop_feature_loader(
    sample_rx: Receiver<SampledWork>,
    result_tx: Sender<PrefetchResult>,
    shutdown: Arc<AtomicBool>,
    mut feature_store: SyncFeatureStore,
    stats: Arc<PrefetchStats>,
) {
    debug!("Feature loader thread started");
    let mut pending: Option<SampledWork> = None;

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        // Get current item: use pending if available, otherwise recv
        let sampled = if let Some(s) = pending.take() {
            trace!(batch_idx = s.work.batch_idx, "using pending sampled work");
            s
        } else {
            match sample_rx.recv() {
                Ok(s) => s,
                Err(_) => break,
            }
        };

        // Lookahead: try to grab next item and issue prefetch hints
        if let Ok(next_sampled) = sample_rx.try_recv() {
            feature_store.prefetch_nodes(&next_sampled.subgraph.nodes);
            trace!(
                batch_idx = next_sampled.work.batch_idx,
                nodes = next_sampled.subgraph.nodes.len(),
                "prefetch hints issued for next batch"
            );
            pending = Some(next_sampled);
        }

        // Load features for current batch
        let t0 = std::time::Instant::now();
        let (features, feature_dim) = match feature_store.get_batch(&sampled.subgraph.nodes) {
            Ok(feats) => {
                trace!(
                    batch_idx = sampled.work.batch_idx,
                    nodes = sampled.subgraph.nodes.len(),
                    "features loaded"
                );
                (Some(feats), Some(feature_store.feature_dim()))
            }
            Err(e) => {
                warn!(batch_idx = sampled.work.batch_idx, error = %e, "feature load failed");
                (None, None)
            }
        };
        let elapsed_ns = t0.elapsed().as_nanos() as u64;
        stats
            .feature_load_time_ns
            .fetch_add(elapsed_ns, Ordering::Relaxed);

        let result = PrefetchResult {
            batch_idx: sampled.work.batch_idx,
            subgraph: sampled.subgraph,
            features,
            feature_dim,
        };

        if result_tx.send(result).is_err() {
            break;
        }
    }

    debug!("Feature loader thread stopped");
}

/// Prefetching neighbor sampler.
///
/// Spawns a dedicated thread that samples batches ahead of time.
/// On Linux with io_uring support, uses zero-syscall I/O for NVMe-backed graphs.
/// Optionally also loads features for sampled nodes.
pub struct NeighborLoader {
    /// Send work to prefetch thread (Option so we can take it on shutdown)
    work_tx: Option<Sender<PrefetchWork>>,
    /// Receive results from prefetch thread
    result_rx: Receiver<PrefetchResult>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Thread handles (1 for no-features mode, 2 for with-features mode)
    handles: Vec<JoinHandle<()>>,
    /// Statistics
    stats: Arc<PrefetchStats>,
    /// Prefetch depth
    prefetch_depth: usize,
    /// Feature dimension (if features are being loaded)
    feature_dim: Option<usize>,
}

impl NeighborLoader {
    /// Create a new prefetching sampler for in-memory graphs (no feature loading).
    ///
    /// # Arguments
    /// * `graph` - Arc to CSR graph (shared with prefetch thread)
    /// * `config` - Sampling configuration
    /// * `prefetch_depth` - How many batches to keep ready (default: 2-3)
    ///
    /// # Errors
    /// Returns an error if the prefetch thread cannot be spawned.
    #[tracing::instrument(
        skip(graph, config),
        fields(num_nodes = graph.num_nodes(), prefetch_depth)
    )]
    pub fn new(
        graph: Arc<Graph>,
        config: SamplingConfig,
        prefetch_depth: usize,
    ) -> std::io::Result<Self> {
        if prefetch_depth == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "prefetch_depth must be >= 1",
            ));
        }

        // Both channels bounded:
        //   - work channel  : producer blocks when the worker falls behind
        //                     by more than `prefetch_depth * 8` submitted
        //                     batches. Prevents unbounded RAM growth when a
        //                     producer stages many epochs at once but the
        //                     worker is slow / stuck.
        //   - result channel: worker blocks when the consumer falls behind
        //                     by `prefetch_depth` produced subgraphs. This is
        //                     the canonical pipeline-stall backpressure.
        let work_capacity = prefetch_depth.saturating_mul(8).max(prefetch_depth);
        let (work_tx, work_rx) = bounded::<PrefetchWork>(work_capacity);
        let (result_tx, result_rx) = bounded::<PrefetchResult>(prefetch_depth);

        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(PrefetchStats::default());

        let handle = {
            let shutdown = shutdown.clone();
            thread::Builder::new()
                .name("aethergraph-prefetch".into())
                .spawn(move || {
                    Self::worker_loop_inmemory(graph, config, work_rx, result_tx, shutdown, None);
                })?
        };

        debug!(
            prefetch_depth,
            "NeighborLoader started (in-memory mode, no features)"
        );

        Ok(Self {
            work_tx: Some(work_tx),
            result_rx,
            shutdown,
            handles: vec![handle],
            stats,
            prefetch_depth,
            feature_dim: None,
        })
    }

    /// Create a prefetching sampler that also loads features.
    ///
    /// After sampling the subgraph, the worker thread also loads features
    /// for all nodes in the subgraph. On Linux, uses io_uring for parallel reads.
    ///
    /// # Arguments
    /// * `graph` - Arc to CSR graph
    /// * `config` - Sampling configuration
    /// * `feature_path` - Path to feature file (AETHFEAT format)
    /// * `prefetch_depth` - How many batches to keep ready
    ///
    /// # Errors
    /// Returns an error if the feature file cannot be loaded or the prefetch thread cannot be spawned.
    pub fn with_features(
        graph: Arc<Graph>,
        config: SamplingConfig,
        feature_path: impl AsRef<Path>,
        prefetch_depth: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(prefetch_depth > 0, "prefetch_depth must be >= 1");

        // Load feature store to get metadata
        let feature_store = SyncFeatureStore::load(feature_path.as_ref())?;
        let feature_dim = feature_store.feature_dim();

        // Same bounding strategy as the in-memory path: producers see
        // backpressure rather than silently growing the work queue.
        let work_capacity = prefetch_depth.saturating_mul(8).max(prefetch_depth);
        let (work_tx, work_rx) = bounded::<PrefetchWork>(work_capacity);
        let (sample_tx, sample_rx) = bounded::<SampledWork>(2);
        let (result_tx, result_rx) = bounded::<PrefetchResult>(prefetch_depth);

        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(PrefetchStats::default());

        let sampler_handle = {
            let shutdown = shutdown.clone();
            let stats = stats.clone();
            thread::Builder::new()
                .name("aethergraph-sampler".into())
                .spawn(move || {
                    worker_loop_sampler(graph, config, work_rx, sample_tx, shutdown, stats);
                })
                .map_err(|e| anyhow::anyhow!("failed to spawn sampler thread: {}", e))?
        };

        let loader_handle = {
            let shutdown = shutdown.clone();
            let stats = stats.clone();
            thread::Builder::new()
                .name("aethergraph-feat-loader".into())
                .spawn(move || {
                    worker_loop_feature_loader(
                        sample_rx,
                        result_tx,
                        shutdown,
                        feature_store,
                        stats,
                    );
                })
                .map_err(|e| anyhow::anyhow!("failed to spawn feature loader thread: {}", e))?
        };

        debug!(
            prefetch_depth,
            feature_dim, "NeighborLoader started (2-thread pipeline: sampler + feature loader)"
        );

        Ok(Self {
            work_tx: Some(work_tx),
            result_rx,
            shutdown,
            handles: vec![sampler_handle, loader_handle],
            stats,
            prefetch_depth,
            feature_dim: Some(feature_dim),
        })
    }

    /// Create a prefetching sampler for NVMe-backed graphs (Linux only).
    ///
    /// Uses io_uring with SQPOLL for zero-syscall I/O.
    ///
    /// # Errors
    /// Returns an error if the prefetch thread cannot be spawned.
    #[cfg(target_os = "linux")]
    pub fn new_nvme(
        graph_path: &std::path::Path,
        config: SamplingConfig,
        prefetch_depth: usize,
    ) -> std::io::Result<Self> {
        Self::new_nvme_inner(graph_path, config, prefetch_depth, None)
    }

    /// Create a prefetching sampler for NVMe-backed graphs with feature loading.
    ///
    /// Combines io_uring graph sampling with mmap-backed feature gathering.
    ///
    /// # Errors
    /// Returns an error if the feature file cannot be loaded or the prefetch thread cannot be spawned.
    #[cfg(target_os = "linux")]
    pub fn with_features_nvme(
        graph_path: &std::path::Path,
        config: SamplingConfig,
        feature_path: impl AsRef<Path>,
        prefetch_depth: usize,
    ) -> anyhow::Result<Self> {
        let feature_store = SyncFeatureStore::load(feature_path.as_ref())?;
        Self::new_nvme_inner(graph_path, config, prefetch_depth, Some(feature_store))
            .map_err(|e| anyhow::anyhow!(e))
    }

    #[cfg(target_os = "linux")]
    fn new_nvme_inner(
        graph_path: &std::path::Path,
        config: SamplingConfig,
        prefetch_depth: usize,
        feature_store: Option<SyncFeatureStore>,
    ) -> std::io::Result<Self> {
        if prefetch_depth == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "prefetch_depth must be >= 1",
            ));
        }

        // Same bounding strategy as the other constructors: bounded work
        // channel applies producer-side backpressure so a runaway submit
        // loop can't grow RAM without limit.
        let work_capacity = prefetch_depth.saturating_mul(8).max(prefetch_depth);
        let (work_tx, work_rx) = bounded::<PrefetchWork>(work_capacity);
        let (result_tx, result_rx) = bounded::<PrefetchResult>(prefetch_depth);

        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(PrefetchStats::default());

        let feature_dim = feature_store.as_ref().map(|s| s.feature_dim());
        let path = graph_path.to_path_buf();
        let handle = {
            let shutdown = shutdown.clone();
            thread::Builder::new()
                .name("aethergraph-prefetch-nvme".into())
                .spawn(move || {
                    if let Err(e) = Self::worker_loop_nvme(
                        &path,
                        config,
                        work_rx,
                        result_tx,
                        shutdown,
                        feature_store,
                    ) {
                        warn!("NVMe prefetch worker error: {}", e);
                    }
                })?
        };

        debug!(
            prefetch_depth,
            ?feature_dim,
            "NeighborLoader started (NVMe io_uring mode)"
        );

        Ok(Self {
            work_tx: Some(work_tx),
            result_rx,
            shutdown,
            handles: vec![handle],
            stats,
            prefetch_depth,
            feature_dim,
        })
    }

    /// Submit a batch to be sampled.
    pub fn submit(&self, batch_idx: usize, seeds: Vec<NodeId>) -> Result<(), SubmitError> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(SubmitError::Shutdown);
        }
        match &self.work_tx {
            Some(tx) => tx
                .send(PrefetchWork { batch_idx, seeds })
                .map_err(|_| SubmitError::ChannelClosed),
            None => Err(SubmitError::Shutdown),
        }
    }

    /// Submit all batches for an epoch.
    pub fn submit_epoch(&self, batches: Vec<Vec<NodeId>>) -> Result<(), SubmitError> {
        for (idx, seeds) in batches.into_iter().enumerate() {
            self.submit(idx, seeds)?;
        }
        Ok(())
    }

    /// Get next sampled subgraph (blocking).
    ///
    /// Returns `None` both when the worker has exited (channel disconnected —
    /// the normal end-of-epoch state, logged at debug) and when the 30s wait
    /// elapses (logged at warn, indicating a deadlock or very slow I/O). The
    /// two cases are distinguishable in the logs.
    pub fn next(&self) -> Option<SampledSubgraph> {
        self.stats.total.fetch_add(1, Ordering::Relaxed);

        // Try non-blocking first to track hit rate
        match self.result_rx.try_recv() {
            Ok(result) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                trace!(batch_idx = result.batch_idx, "prefetch hit");
                Some(result.subgraph)
            }
            Err(TryRecvError::Empty) => {
                // Cache miss - need to wait (with timeout to detect issues)
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                trace!("prefetch miss - blocking");
                match self
                    .result_rx
                    .recv_timeout(std::time::Duration::from_secs(30))
                {
                    Ok(r) => Some(r.subgraph),
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        warn!(
                            "Prefetch timeout after 30s - worker may have deadlocked or I/O is very slow"
                        );
                        None
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                        // Distinct from a timeout: the worker has exited (epoch
                        // drained or shut down), which is the normal end state.
                        debug!("prefetch channel disconnected - worker exited");
                        None
                    }
                }
            }
            Err(TryRecvError::Disconnected) => {
                debug!("prefetch channel disconnected - worker exited");
                None
            }
        }
    }

    /// Try to get next without blocking.
    pub fn try_next(&self) -> Option<SampledSubgraph> {
        match self.result_rx.try_recv() {
            Ok(result) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                self.stats.total.fetch_add(1, Ordering::Relaxed);
                Some(result.subgraph)
            }
            _ => None,
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &PrefetchStats {
        &self.stats
    }

    /// Get prefetch depth.
    pub fn prefetch_depth(&self) -> usize {
        self.prefetch_depth
    }

    /// Get feature dimension (if features are being loaded).
    pub fn feature_dim(&self) -> Option<usize> {
        self.feature_dim
    }

    /// Get next sampled subgraph with features (blocking).
    ///
    /// Returns (subgraph, optional features). Features are `Some` if the
    /// prefetcher was created with `with_features()`.
    ///
    /// Returns `None` both when the worker has exited (channel disconnected —
    /// the normal end-of-epoch state, logged at debug) and when the 30s wait
    /// elapses (logged at warn, indicating a deadlock or very slow I/O). The
    /// two cases are distinguishable in the logs.
    pub fn next_with_features(&self) -> Option<(SampledSubgraph, Option<Vec<f32>>)> {
        self.stats.total.fetch_add(1, Ordering::Relaxed);

        match self.result_rx.try_recv() {
            Ok(result) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                trace!(batch_idx = result.batch_idx, "prefetch hit");
                Some((result.subgraph, result.features))
            }
            Err(TryRecvError::Empty) => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                trace!("prefetch miss - blocking");
                match self
                    .result_rx
                    .recv_timeout(std::time::Duration::from_secs(30))
                {
                    Ok(r) => Some((r.subgraph, r.features)),
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        warn!(
                            "Prefetch timeout after 30s - worker may have deadlocked or I/O is very slow"
                        );
                        None
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                        debug!("prefetch channel disconnected - worker exited");
                        None
                    }
                }
            }
            Err(TryRecvError::Disconnected) => {
                debug!("prefetch channel disconnected - worker exited");
                None
            }
        }
    }

    /// Shutdown the prefetch thread(s).
    #[tracing::instrument(skip(self), fields(num_handles = self.handles.len()))]
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Drop sender to unblock workers (take it so channel actually closes)
        drop(self.work_tx.take());
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }

    /// Worker loop for in-memory graphs (optionally with feature loading).
    ///
    /// Uses lookahead prefetching: while loading batch N's features, we sample
    /// batch N+1 and issue kernel prefetch hints for its feature offsets.
    fn worker_loop_inmemory(
        graph: Arc<Graph>,
        config: SamplingConfig,
        work_rx: Receiver<PrefetchWork>,
        result_tx: Sender<PrefetchResult>,
        shutdown: Arc<AtomicBool>,
        feature_store: Option<SyncFeatureStore>,
    ) {
        let has_features = feature_store.is_some();
        debug!(
            has_features,
            "In-memory prefetch worker started (with lookahead)"
        );

        let mut sampler = NeighborSampler::new(&graph, config);
        let mut feature_store = feature_store;

        // Lookahead state: pre-sampled next batch
        let mut pending: Option<(PrefetchWork, SampledSubgraph)> = None;

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Get current work item: use pending if available, otherwise recv
            let (work, subgraph) = if let Some((w, sg)) = pending.take() {
                trace!(batch_idx = w.batch_idx, "using pre-sampled batch");
                (w, sg)
            } else {
                match work_rx.recv() {
                    Ok(work) => {
                        trace!(
                            batch_idx = work.batch_idx,
                            seeds = work.seeds.len(),
                            "sampling"
                        );
                        let subgraph = sampler.sample_neighbors(&work.seeds);
                        (work, subgraph)
                    }
                    Err(_) => break,
                }
            };

            // Lookahead: try to get and pre-process next batch
            // Issue prefetch hints for N+1 while we load N's features
            if let Ok(next_work) = work_rx.try_recv() {
                trace!(batch_idx = next_work.batch_idx, "lookahead sampling");
                let next_subgraph = sampler.sample_neighbors(&next_work.seeds);

                // Issue kernel prefetch hints for next batch's features
                if let Some(ref store) = feature_store {
                    store.prefetch_nodes(&next_subgraph.nodes);
                    trace!(
                        batch_idx = next_work.batch_idx,
                        nodes = next_subgraph.nodes.len(),
                        "prefetch hints issued"
                    );
                }

                pending = Some((next_work, next_subgraph));
            }

            // Load current batch features (kernel may be prefetching next batch in parallel)
            let (features, feature_dim) = if let Some(ref mut store) = feature_store {
                match store.get_batch(&subgraph.nodes) {
                    Ok(feats) => {
                        trace!(
                            batch_idx = work.batch_idx,
                            nodes = subgraph.nodes.len(),
                            "features loaded"
                        );
                        (Some(feats), Some(store.feature_dim()))
                    }
                    Err(e) => {
                        warn!(batch_idx = work.batch_idx, error = %e, "feature load failed");
                        (None, None)
                    }
                }
            } else {
                (None, None)
            };

            let result = PrefetchResult {
                batch_idx: work.batch_idx,
                subgraph,
                features,
                feature_dim,
            };

            if result_tx.send(result).is_err() {
                break; // Consumer gone
            }
        }

        debug!("In-memory prefetch worker stopped");
    }

    /// Worker loop for NVMe-backed graphs using io_uring.
    ///
    /// Uses SQPOLL for reduced syscalls. Note: Graph adjacency reads are
    /// variable-sized, so we don't use O_DIRECT/IOPOLL here (would require
    /// complex buffer alignment for each variable-length read).
    #[cfg(target_os = "linux")]
    fn worker_loop_nvme(
        path: &std::path::Path,
        config: SamplingConfig,
        work_rx: Receiver<PrefetchWork>,
        result_tx: Sender<PrefetchResult>,
        shutdown: Arc<AtomicBool>,
        mut feature_store: Option<SyncFeatureStore>,
    ) -> anyhow::Result<()> {
        use crate::internal::uring::UringHandle;
        use anyhow::Context;

        debug!("NVMe prefetch worker starting with io_uring");

        // Open graph file (no O_DIRECT - variable-sized reads are complex to align)
        let file = File::open(path).context("failed to open graph file")?;
        let fd = file.as_raw_fd();

        // Read + validate graph header
        let mut header = [0u8; GRAPH_HEADER_SIZE as usize];
        file.read_exact_at(&mut header, 0)
            .context("failed to read header")?;
        let magic = u32::from_le_bytes(header[0..4].try_into()?);
        let version = u32::from_le_bytes(header[4..8].try_into()?);
        anyhow::ensure!(
            magic == GRAPH_MAGIC,
            "invalid graph magic: expected {:#x}, got {:#x}",
            GRAPH_MAGIC,
            magic
        );
        anyhow::ensure!(
            version == GRAPH_VERSION,
            "unsupported graph version: expected {}, got {}",
            GRAPH_VERSION,
            version
        );

        let num_nodes_u64 = u64::from_le_bytes(header[8..16].try_into()?);
        let num_edges_u64 = u64::from_le_bytes(header[16..24].try_into()?);
        anyhow::ensure!(
            num_nodes_u64 <= MAX_GRAPH_NODES,
            "num_nodes {} exceeds maximum {}",
            num_nodes_u64,
            MAX_GRAPH_NODES
        );
        anyhow::ensure!(
            num_edges_u64 <= MAX_GRAPH_EDGES,
            "num_edges {} exceeds maximum {}",
            num_edges_u64,
            MAX_GRAPH_EDGES
        );
        let num_nodes = usize::try_from(num_nodes_u64)
            .map_err(|_| anyhow::anyhow!("num_nodes does not fit in usize"))?;
        let num_edges = usize::try_from(num_edges_u64)
            .map_err(|_| anyhow::anyhow!("num_edges does not fit in usize"))?;

        // Read offsets array into memory (small)
        let offsets_size = num_nodes
            .checked_add(1)
            .and_then(|n| n.checked_mul(std::mem::size_of::<u64>()))
            .ok_or_else(|| anyhow::anyhow!("offsets array size overflow"))?;
        let offsets_start = GRAPH_HEADER_SIZE;
        let edges_start = offsets_start
            .checked_add(offsets_size as u64)
            .ok_or_else(|| anyhow::anyhow!("edges_start overflow"))?;
        let min_edges_bytes = (num_edges as u64)
            .checked_mul(std::mem::size_of::<NodeId>() as u64)
            .ok_or_else(|| anyhow::anyhow!("edge byte size overflow"))?;
        let min_file_size = edges_start
            .checked_add(min_edges_bytes)
            .ok_or_else(|| anyhow::anyhow!("minimum graph file size overflow"))?;
        let file_size = file.metadata().context("failed to stat graph file")?.len();
        anyhow::ensure!(
            file_size >= min_file_size,
            "graph file truncated: expected at least {} bytes, got {}",
            min_file_size,
            file_size
        );

        let mut offsets_bytes = vec![0u8; offsets_size];
        file.read_exact_at(&mut offsets_bytes, offsets_start)
            .context("failed to read offsets")?;

        let offsets: Vec<u64> = offsets_bytes
            .chunks_exact(8)
            .map(|c| {
                let arr: [u8; 8] = c
                    .try_into()
                    .expect("chunks_exact(8) guarantees 8-byte chunks");
                u64::from_le_bytes(arr)
            })
            .collect();
        anyhow::ensure!(offsets.len() == num_nodes + 1, "invalid offsets length");
        anyhow::ensure!(offsets[0] == 0, "invalid offsets: offsets[0] must be 0");
        for (i, window) in offsets.windows(2).enumerate() {
            anyhow::ensure!(
                window[0] <= window[1],
                "invalid offsets: offsets[{}]={} > offsets[{}]={}",
                i,
                window[0],
                i + 1,
                window[1]
            );
            anyhow::ensure!(
                window[1] <= num_edges as u64,
                "invalid offsets: offsets[{}]={} exceeds num_edges {}",
                i + 1,
                window[1],
                num_edges
            );
        }
        anyhow::ensure!(
            offsets[num_nodes] == num_edges as u64,
            "invalid offsets tail: offsets[last]={} != num_edges {}",
            offsets[num_nodes],
            num_edges
        );

        debug!(num_nodes, "Loaded offsets array for NVMe graph");

        // Setup io_uring with SQPOLL (reduced syscalls via kernel SQ polling)
        // Note: We don't use IOPOLL since we're not using O_DIRECT
        let mut handle = UringHandle::new(crate::internal::uring::DEFAULT_RING_ENTRIES, 1000)?;

        // Register file descriptor for faster access
        if let Err(e) = handle.register_fd(&file) {
            warn!("Failed to register graph fd: {}", e);
        }

        if handle.is_sqpoll() {
            debug!("io_uring: SQPOLL enabled (reduced syscalls)");
        } else {
            debug!("io_uring: standard mode (batched I/O)");
        }

        // Main sampling loop
        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            let work = match work_rx.recv() {
                Ok(w) => w,
                Err(_) => break,
            };

            trace!(
                batch_idx = work.batch_idx,
                seeds = work.seeds.len(),
                "NVMe sampling"
            );

            // Sample using io_uring batch reads
            let subgraph = sample_with_uring(
                &mut handle,
                fd,
                &offsets,
                edges_start,
                num_nodes,
                &work.seeds,
                &config,
            )?;

            let (features, feature_dim) = if let Some(ref mut store) = feature_store {
                match store.get_batch(&subgraph.nodes) {
                    Ok(feats) => (Some(feats), Some(store.feature_dim())),
                    Err(e) => {
                        warn!(batch_idx = work.batch_idx, error = %e, "NVMe feature load failed");
                        (None, None)
                    }
                }
            } else {
                (None, None)
            };

            let result = PrefetchResult {
                batch_idx: work.batch_idx,
                subgraph,
                features,
                feature_dim,
            };

            if result_tx.send(result).is_err() {
                break;
            }
        }

        debug!("NVMe prefetch worker stopped");
        Ok(())
    }
}

impl Drop for NeighborLoader {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// io_uring-based k-hop sampling for NVMe graphs.
///
/// This path honors `fanout`, `replace`, `cumulative`, and `seed`. It does NOT
/// yet honor the following `SamplingConfig` fields, which are silently ignored:
/// `weighted`, `temporal_strategy`, `disjoint`, `deterministic`,
/// `track_edge_ids` (edge ids are always tracked), `max_degree`, and
/// `subgraph_type` (edges are always returned directional). Callers needing
/// those must use the in-memory [`NeighborSampler`] path.
#[cfg(target_os = "linux")]
fn sample_with_uring(
    handle: &mut crate::internal::uring::UringHandle,
    fd: i32,
    offsets: &[u64],
    edges_start: u64,
    num_nodes: usize,
    seeds: &[NodeId],
    config: &SamplingConfig,
) -> anyhow::Result<SampledSubgraph> {
    use rustc_hash::FxHashSet;

    let num_hops = config.fanout.len();
    let mut all_nodes: FxHashSet<NodeId> = seeds.iter().copied().collect();
    let mut frontier: Vec<NodeId> = seeds.to_vec();
    let mut edge_src = Vec::new();
    let mut edge_dst = Vec::new();
    let mut edge_ids = Vec::new();
    let mut num_sampled_nodes = Vec::with_capacity(num_hops);
    let mut num_sampled_edges = Vec::with_capacity(num_hops);

    // Match the in-memory sampler: fall back to system entropy when no seed is
    // given, instead of a fixed constant.
    let mut rng = WyRand::new(config.seed.unwrap_or_else(rand::random::<u64>));

    for hop in 0..num_hops {
        let fanout = config.fanout[hop];
        if frontier.is_empty() {
            num_sampled_nodes.push(0);
            num_sampled_edges.push(0);
            continue;
        }

        let edges_before = edge_src.len();

        // Batch read all frontier neighbors via io_uring
        let neighbors_batch =
            batch_read_neighbors_uring(handle, fd, offsets, edges_start, num_nodes, &frontier)?;

        let mut next_frontier = Vec::new();

        for (&node, neighbors) in frontier.iter().zip(neighbors_batch.iter()) {
            if neighbors.is_empty() {
                continue;
            }

            let node_idx = node as usize;
            // Frontier nodes are valid IDs, so this index is always in range.
            // Skip rather than mint a bogus edge id 0 if it somehow is not.
            debug_assert!(node_idx < offsets.len());
            if node_idx >= offsets.len() {
                continue;
            }
            let edge_offset = offsets[node_idx];

            // Sample fanout neighbors (returns indices into neighbors array)
            let sampled_indices =
                sample_neighbor_indices(&mut rng, neighbors.len(), fanout, config.replace);

            for idx in sampled_indices {
                let neighbor = neighbors[idx];
                edge_src.push(node);
                edge_dst.push(neighbor);
                edge_ids.push(edge_offset + idx as u64);
                if all_nodes.insert(neighbor) {
                    next_frontier.push(neighbor);
                }
            }
        }

        num_sampled_nodes.push(next_frontier.len());
        num_sampled_edges.push(edge_src.len() - edges_before);

        frontier = if config.cumulative {
            frontier.into_iter().chain(next_frontier).collect()
        } else {
            next_frontier
        };
    }

    let nodes: Vec<NodeId> = all_nodes.into_iter().collect();

    Ok(SampledSubgraph::from_parts(
        nodes,
        edge_src,
        edge_dst,
        edge_ids,
        seeds.to_vec(),
        num_sampled_nodes,
        num_sampled_edges,
    ))
}

/// Batch read neighbors for multiple nodes using io_uring.
///
/// Uses SQPOLL-aware submission and registered file descriptors.
#[cfg(target_os = "linux")]
fn batch_read_neighbors_uring(
    handle: &mut crate::internal::uring::UringHandle,
    fd: i32,
    offsets: &[u64],
    edges_start: u64,
    num_nodes: usize,
    nodes: &[NodeId],
) -> anyhow::Result<Vec<Vec<NodeId>>> {
    use crate::internal::uring::batch_read;

    // `buffers[i]` is empty for invalid/zero-degree nodes; `reads` carries
    // (file_offset, ptr, len) for the non-empty subset.
    let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(nodes.len());
    let mut reads: Vec<(u64, *mut u8, usize)> = Vec::with_capacity(nodes.len());

    for &node in nodes {
        let idx = node as usize;
        if idx >= num_nodes {
            buffers.push(Vec::new());
            continue;
        }
        let start = offsets[idx] as usize;
        let end = offsets[idx + 1] as usize;
        let num_neighbors = end - start;
        if num_neighbors == 0 {
            buffers.push(Vec::new());
            continue;
        }
        let byte_offset = edges_start + (start * 4) as u64;
        let byte_size = num_neighbors * 4;
        let mut buf = vec![0u8; byte_size];
        let ptr = buf.as_mut_ptr();
        buffers.push(buf);
        reads.push((byte_offset, ptr, byte_size));
    }

    if !reads.is_empty() {
        // SAFETY: every ptr in `reads` points into `buffers[i]` which lives until
        // this function returns; batch_read writes synchronously and returns
        // before we touch the buffers again.
        batch_read(handle, fd, &reads)?;
    }

    Ok(buffers
        .into_iter()
        .map(|buf| {
            buf.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
        .collect())
}

/// Fast PRNG (same as in sampler.rs)
#[cfg(target_os = "linux")]
#[derive(Clone)]
struct WyRand {
    state: u64,
}

#[cfg(target_os = "linux")]
impl WyRand {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0xa0761d6478bd642f);
        let t = (self.state as u128).wrapping_mul((self.state ^ 0xe7037ed1a0b428db) as u128);
        ((t >> 64) ^ t) as u32
    }
}

/// Sample k neighbor indices using Lemire's method.
/// Returns indices into the neighbors array, not the actual neighbor values.
#[cfg(target_os = "linux")]
fn sample_neighbor_indices(
    rng: &mut WyRand,
    num_neighbors: usize,
    k: usize,
    replace: bool,
) -> Vec<usize> {
    if num_neighbors == 0 {
        return Vec::new();
    }

    if num_neighbors <= k {
        return (0..num_neighbors).collect();
    }

    let n = num_neighbors as u64;
    let mut result = Vec::with_capacity(k);

    if replace {
        for _ in 0..k {
            let idx = ((rng.next_u32() as u64 * n) >> 32) as usize;
            result.push(idx);
        }
    } else {
        // Floyd's algorithm for sampling without replacement
        use rustc_hash::FxHashSet;
        let mut seen = FxHashSet::default();

        for i in (num_neighbors - k)..num_neighbors {
            let j = ((rng.next_u32() as u64 * (i as u64 + 1)) >> 32) as usize;
            if seen.contains(&j) {
                result.push(i);
                seen.insert(i);
            } else {
                result.push(j);
                seen.insert(j);
            }
        }
    }

    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    Shutdown,
    ChannelClosed,
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubmitError::Shutdown => write!(f, "prefetcher shut down"),
            SubmitError::ChannelClosed => write!(f, "channel closed"),
        }
    }
}

impl std::error::Error for SubmitError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_graph() -> Arc<Graph> {
        let edges = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
        ];
        Arc::new(Graph::from_edges(5, &edges, None).unwrap())
    }

    #[test]
    fn test_prefetch_inmemory() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2],
            replace: false,
            seed: Some(42),
            ..Default::default()
        };

        let prefetcher = NeighborLoader::new(graph, config, 2).unwrap();

        prefetcher.submit(0, vec![0]).unwrap();
        prefetcher.submit(1, vec![1]).unwrap();
        prefetcher.submit(2, vec![2]).unwrap();

        let sg1 = prefetcher.next().unwrap();
        assert_eq!(sg1.num_seeds(), 1);

        let sg2 = prefetcher.next().unwrap();
        assert_eq!(sg2.num_seeds(), 1);

        let sg3 = prefetcher.next().unwrap();
        assert_eq!(sg3.num_seeds(), 1);
    }

    #[test]
    fn test_prefetch_hit_rate() {
        let graph = create_test_graph();
        let config = SamplingConfig::default();

        let prefetcher = NeighborLoader::new(graph, config, 3).unwrap();

        // Submit all upfront
        for i in 0..10 {
            prefetcher.submit(i, vec![i as u32 % 5]).unwrap();
        }

        // Let worker prefetch
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Consume all
        for _ in 0..10 {
            prefetcher.next().unwrap();
        }

        let hit_rate = prefetcher.stats().hit_rate();
        println!("Hit rate: {:.1}%", hit_rate * 100.0);
        // With prefetch_depth=3 and 10 batches, we expect some hits
        // The exact rate depends on timing, so just verify we got some hits
        assert!(
            hit_rate >= 0.3,
            "Expected hit_rate >= 0.3, got {}",
            hit_rate
        );
    }

    #[test]
    fn test_sync_feature_store_legacy_header_offset() {
        let temp_file = NamedTempFile::new().unwrap();

        // Craft legacy AETHFEAT file where bytes 24..32 are zero and payload starts at 32.
        let features = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(temp_file.path())
            .unwrap();
        file.write_all(b"AETHFEAT").unwrap();
        file.write_all(&(2u64).to_le_bytes()).unwrap();
        file.write_all(&(3u64).to_le_bytes()).unwrap();
        file.write_all(&(0u64).to_le_bytes()).unwrap(); // legacy offset field
        let feature_bytes: &[u8] = bytemuck::cast_slice(&features);
        file.write_all(feature_bytes).unwrap();
        file.sync_all().unwrap();

        let mut store = SyncFeatureStore::load(temp_file.path()).unwrap();
        let batch = store.get_batch(&[0, 1]).unwrap();

        assert_eq!(batch, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
