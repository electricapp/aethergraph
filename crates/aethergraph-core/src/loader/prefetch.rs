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

use super::hetero_sampler::{HeteroNeighborSampler, HeteroSampledSubgraph, HeteroSamplingConfig};
use super::sampler::{NeighborSampler, SampledSubgraph, SamplingConfig};
use crate::features::header::{FeatureDtype, parse_feature_header};
use crate::graph::hetero::{HeteroGraph, NodeTypeId};
use crate::graph::{Graph, NodeId};
#[cfg(target_os = "linux")]
use crate::internal::genstamp::WyRand;
use crate::internal::hint;
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;
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

/// How long a blocking consumer call waits for the worker before reporting
/// [`PrefetchError::Timeout`].
const RECV_TIMEOUT: Duration = Duration::from_secs(30);

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
    /// Features for all nodes in the subgraph (flattened: num_nodes * feature_dim).
    /// `None` when the loader has no feature column attached; `Some(Err(_))`
    /// when a feature column is attached but loading it failed.
    pub features: Option<anyhow::Result<Vec<f32>>>,
    /// Feature dimension (if a feature column is attached)
    pub feature_dim: Option<usize>,
}

/// Completed hetero prefetch result.
#[derive(Debug)]
pub struct HeteroPrefetchResult {
    /// Batch index
    pub batch_idx: usize,
    /// Sampled heterogeneous subgraph
    pub subgraph: HeteroSampledSubgraph,
}

/// A sampled subgraph paired with its features. The feature slot is `Some`
/// when the loader has a feature column attached and `None` when it does not.
pub type SubgraphWithFeatures = (SampledSubgraph, Option<Vec<f32>>);

/// Error returned by [`NeighborLoader::next`], [`NeighborLoader::try_next`],
/// and [`NeighborLoader::next_with_features`].
#[derive(Debug)]
pub enum PrefetchError {
    /// No result arrived within the wait window. The worker may just be slow
    /// (or deadlocked); the caller may call again.
    Timeout {
        /// How long the call waited before giving up.
        waited: Duration,
    },
    /// The worker exited without `shutdown()` being requested (panic or
    /// internal error). `message` carries the captured panic payload or
    /// worker error when one is available.
    WorkerExited { message: Option<String> },
    /// A feature column is attached but loading features for this batch
    /// failed. The batch's subgraph is dropped along with the error.
    FeatureLoad {
        batch_idx: usize,
        source: anyhow::Error,
    },
}

impl std::fmt::Display for PrefetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrefetchError::Timeout { waited } => {
                write!(f, "prefetch timed out after {waited:?}")
            }
            PrefetchError::WorkerExited {
                message: Some(message),
            } => {
                write!(f, "prefetch worker exited: {message}")
            }
            PrefetchError::WorkerExited { message: None } => {
                write!(f, "prefetch worker exited unexpectedly")
            }
            PrefetchError::FeatureLoad { batch_idx, source } => {
                write!(f, "feature load failed for batch {batch_idx}: {source}")
            }
        }
    }
}

impl std::error::Error for PrefetchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PrefetchError::FeatureLoad { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

/// Runs a worker body, recording a panic payload or error into `fault` so the
/// consumer can report why the result channel disconnected.
fn run_worker(fault: &OnceLock<String>, body: impl FnOnce() -> anyhow::Result<()>) {
    let message = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(Ok(())) => return,
        Ok(Err(e)) => format!("{e:#}"),
        Err(payload) => panic_message(payload.as_ref()),
    };
    warn!("prefetch worker fault: {}", message);
    // First fault wins. `set` returning Err means another worker already
    // recorded one, which is the same outcome the old `get_or_insert` had —
    // and unlike a mutex there is no poisoning to unwrap past, since a
    // panic here cannot leave the slot half-written.
    let _ = fault.set(message);
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        format!("panic: {s}")
    } else if let Some(s) = payload.downcast_ref::<String>() {
        format!("panic: {s}")
    } else {
        "panic with non-string payload".to_string()
    }
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
    /// io_uring lane (ring + reusable landing buffers) with SQPOLL-aware
    /// submission (Linux only)
    #[cfg(target_os = "linux")]
    uring: Option<crate::internal::uring::UringLane>,
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
            use crate::internal::uring::{
                DIRECT_IO_OFFSET_ALIGNMENT, direct_io_offset_alignment,
                is_layout_direct_io_compatible_with, open_direct_or_fallback,
            };

            // Ask the file what its device actually requires. The 512-byte
            // default is only a floor: on a 4Kn device a layout that clears
            // 512 but not 4096 would pass the check and then fail every
            // read with EINVAL.
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
                let (f, direct) = open_direct_or_fallback(path)?;
                if direct {
                    debug!(
                        "Feature layout is O_DIRECT compatible (offset={}, size={}, alignment={})",
                        header.features_start_offset, header.feature_size, alignment
                    );
                }
                (f, direct)
            } else {
                // Layout not aligned, O_DIRECT would fail with EINVAL
                warn!(
                    "Feature layout not O_DIRECT compatible at {}-byte alignment: offset={} (aligned={}), size={} (aligned={})",
                    alignment,
                    header.features_start_offset,
                    (header.features_start_offset as usize).is_multiple_of(alignment),
                    header.feature_size,
                    header.feature_size.is_multiple_of(alignment)
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
    fn setup_uring(file: &File, direct_io: bool) -> Option<crate::internal::uring::UringLane> {
        let mut handle = crate::internal::uring::create_feature_uring(direct_io)?;
        if let Err(e) = handle.register_fd(file) {
            warn!("Failed to register FD: {}", e);
        }
        Some(crate::internal::uring::UringLane::new(handle))
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
                // Take the io_uring lane out temporarily to avoid borrow conflict
                let mut lane = self.uring.take().unwrap();
                let result = Self::batch_read_uring_aligned(
                    &self.file,
                    nodes,
                    feature_size,
                    self.num_nodes,
                    self.features_start_offset,
                    &mut lane,
                    self.direct_io,
                    self.dtype,
                    self.feature_dim,
                );
                self.uring = Some(lane);
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

    /// Batch read using io_uring: bounds-checks `nodes`, then gathers and
    /// decodes through [`crate::features::gather::uring_gather_rows`]
    /// (shared with `AsyncFeatureStore`) — the lane's persistent buffers
    /// land the reads, one pipelined submission covers the whole batch,
    /// and rows decode straight into the output vector.
    #[cfg(target_os = "linux")]
    #[allow(clippy::too_many_arguments)]
    fn batch_read_uring_aligned(
        file: &File,
        nodes: &[NodeId],
        feature_size: usize,
        num_nodes: usize,
        features_start_offset: u64,
        lane: &mut crate::internal::uring::UringLane,
        direct_io: bool,
        dtype: FeatureDtype,
        feature_dim: usize,
    ) -> anyhow::Result<Vec<f32>> {
        // Validate all nodes first
        for &node in nodes {
            if node as usize >= num_nodes {
                anyhow::bail!("node {} out of bounds (max {})", node, num_nodes);
            }
        }

        crate::features::gather::uring_gather_rows(
            lane,
            file.as_raw_fd(),
            nodes,
            features_start_offset,
            feature_size,
            direct_io,
            dtype,
            feature_dim,
        )
    }

    /// Sync batch read fallback.
    ///
    /// Bounds are checked up front, the landing buffer and the dtype
    /// dispatch are hoisted out of the row loop, and each row decodes as a
    /// block rather than element by element.
    fn batch_read_sync(&self, nodes: &[NodeId], feature_size: usize) -> anyhow::Result<Vec<f32>> {
        for &node in nodes {
            if node as usize >= self.num_nodes {
                anyhow::bail!("node {} out of bounds", node);
            }
        }

        let decoder = self.dtype.row_decoder();
        let mut all_features = vec![0f32; nodes.len() * self.feature_dim];
        let mut buffer = vec![0u8; feature_size];

        for (i, &node) in nodes.iter().enumerate() {
            let offset = self.features_start_offset + (node as u64 * feature_size as u64);
            self.file.read_exact_at(&mut buffer, offset)?;
            decoder.decode_row(
                &buffer,
                &mut all_features[i * self.feature_dim..(i + 1) * self.feature_dim],
            );
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

    /// Resets all counters.
    ///
    /// Each counter is cleared with an independent relaxed store, so a reset
    /// racing concurrent recorders is not atomic — a snapshot taken around it
    /// can mix pre- and post-reset values.
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

/// Sampler worker loop for the hetero pool.
///
/// Each worker owns its own [`HeteroNeighborSampler`] (the sampler's scratch
/// buffers are per-instance) over the shared `Arc<HeteroGraph>` and drains
/// the MPMC work channel until it closes or shutdown is requested.
fn worker_loop_hetero(
    graph: Arc<HeteroGraph>,
    config: HeteroSamplingConfig,
    seed_type: NodeTypeId,
    work_rx: Receiver<PrefetchWork>,
    result_tx: Sender<HeteroPrefetchResult>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<PrefetchStats>,
) {
    debug!("Hetero sampler thread started");
    let mut sampler = HeteroNeighborSampler::new(&graph, config);

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
            "hetero sampling"
        );
        let t0 = std::time::Instant::now();
        let subgraph = sampler.sample_neighbors(seed_type, &work.seeds);
        let elapsed_ns = t0.elapsed().as_nanos() as u64;
        stats
            .sample_time_ns
            .fetch_add(elapsed_ns, Ordering::Relaxed);

        let result = HeteroPrefetchResult {
            batch_idx: work.batch_idx,
            subgraph,
        };

        if result_tx.send(result).is_err() {
            break;
        }
    }

    debug!("Hetero sampler thread stopped");
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
        let features = match feature_store.get_batch(&sampled.subgraph.nodes) {
            Ok(feats) => {
                trace!(
                    batch_idx = sampled.work.batch_idx,
                    nodes = sampled.subgraph.nodes.len(),
                    "features loaded"
                );
                Ok(feats)
            }
            Err(e) => {
                warn!(batch_idx = sampled.work.batch_idx, error = %e, "feature load failed");
                Err(e)
            }
        };
        let elapsed_ns = t0.elapsed().as_nanos() as u64;
        stats
            .feature_load_time_ns
            .fetch_add(elapsed_ns, Ordering::Relaxed);

        let result = PrefetchResult {
            batch_idx: sampled.work.batch_idx,
            subgraph: sampled.subgraph,
            features: Some(features),
            feature_dim: Some(feature_store.feature_dim()),
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
    /// Receive results from prefetch thread (Option so shutdown can drop it
    /// and unblock a worker mid-`send` on a full result channel)
    result_rx: Option<Receiver<PrefetchResult>>,
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
    /// Why a worker exited on its own (captured panic or error), if it did
    worker_fault: Arc<OnceLock<String>>,
}

impl NeighborLoader {
    /// Create a new prefetching sampler for in-memory graphs (no feature loading).
    ///
    /// # Arguments
    /// * `graph` - Arc to CSR graph (shared with prefetch threads)
    /// * `config` - Sampling configuration
    /// * `prefetch_depth` - How many batches to keep ready (default: 2-3)
    /// * `sampler_threads` - Sampler worker count (0 is treated as 1). The
    ///   work channel is MPMC and results carry their `batch_idx`, so extra
    ///   workers scale sampling throughput with no ordering machinery —
    ///   consumers must already match results by content, not arrival order.
    ///
    /// # Errors
    /// Returns an error if a prefetch thread cannot be spawned.
    #[tracing::instrument(
        skip(graph, config),
        fields(num_nodes = graph.num_nodes(), prefetch_depth)
    )]
    pub fn new(
        graph: Arc<Graph>,
        config: SamplingConfig,
        prefetch_depth: usize,
        sampler_threads: usize,
    ) -> std::io::Result<Self> {
        if prefetch_depth == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "prefetch_depth must be >= 1",
            ));
        }
        let sampler_threads = sampler_threads.max(1);

        // Both channels bounded:
        //   - work channel  : producer blocks when the workers fall behind
        //                     by more than `prefetch_depth * 8` submitted
        //                     batches. Prevents unbounded RAM growth when a
        //                     producer stages many epochs at once but the
        //                     workers are slow / stuck.
        //   - result channel: workers block when the consumer falls behind
        //                     by `prefetch_depth` produced subgraphs. This is
        //                     the canonical pipeline-stall backpressure.
        let work_capacity = prefetch_depth.saturating_mul(8).max(prefetch_depth);
        let (work_tx, work_rx) = bounded::<PrefetchWork>(work_capacity);
        let (result_tx, result_rx) = bounded::<PrefetchResult>(prefetch_depth.max(sampler_threads));

        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(PrefetchStats::default());
        let worker_fault = Arc::new(OnceLock::new());

        let mut handles = Vec::with_capacity(sampler_threads);
        for t in 0..sampler_threads {
            let shutdown = shutdown.clone();
            let fault = worker_fault.clone();
            let graph = Arc::clone(&graph);
            let config = config.clone();
            let work_rx = work_rx.clone();
            let result_tx = result_tx.clone();
            handles.push(
                thread::Builder::new()
                    .name(format!("aethergraph-prefetch-{t}"))
                    .spawn(move || {
                        crate::internal::numa::pin_worker(t);
                        run_worker(&fault, move || {
                            Self::worker_loop_inmemory(
                                graph, config, work_rx, result_tx, shutdown, None,
                            );
                            Ok(())
                        });
                    })?,
            );
        }

        debug!(
            prefetch_depth,
            sampler_threads, "NeighborLoader started (in-memory mode, no features)"
        );

        Ok(Self {
            work_tx: Some(work_tx),
            result_rx: Some(result_rx),
            shutdown,
            handles,
            stats,
            prefetch_depth,
            feature_dim: None,
            worker_fault,
        })
    }

    /// Create a prefetching sampler that also loads features.
    ///
    /// Sampling and feature loading run as a pipeline: `sampler_threads`
    /// worker threads pull seed batches from the MPMC work channel, and one
    /// feature-loader thread (io_uring on Linux) drains their output, so
    /// sampling overlaps both the feature I/O and the consumer's compute.
    ///
    /// # Arguments
    /// * `graph` - Arc to CSR graph
    /// * `config` - Sampling configuration
    /// * `feature_path` - Path to feature file (AETHFEAT format)
    /// * `prefetch_depth` - How many batches to keep ready
    /// * `sampler_threads` - Sampler worker count (0 is treated as 1)
    ///
    /// # Errors
    /// Returns an error if the feature file cannot be loaded or a pipeline
    /// thread cannot be spawned.
    pub fn with_features(
        graph: Arc<Graph>,
        config: SamplingConfig,
        feature_path: impl AsRef<Path>,
        prefetch_depth: usize,
        sampler_threads: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(prefetch_depth > 0, "prefetch_depth must be >= 1");
        let sampler_threads = sampler_threads.max(1);

        // Load feature store to get metadata
        let feature_store = SyncFeatureStore::load(feature_path.as_ref())?;
        let feature_dim = feature_store.feature_dim();

        // Same bounding strategy as the in-memory path: producers see
        // backpressure rather than silently growing the work queue. The
        // sample channel holds at least one slot per sampler so a burst of
        // simultaneous completions doesn't immediately block the pool.
        let work_capacity = prefetch_depth.saturating_mul(8).max(prefetch_depth);
        let (work_tx, work_rx) = bounded::<PrefetchWork>(work_capacity);
        let (sample_tx, sample_rx) = bounded::<SampledWork>(sampler_threads.max(2));
        let (result_tx, result_rx) = bounded::<PrefetchResult>(prefetch_depth);

        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(PrefetchStats::default());
        let worker_fault = Arc::new(OnceLock::new());

        let mut handles = Vec::with_capacity(sampler_threads + 1);
        for t in 0..sampler_threads {
            let shutdown = shutdown.clone();
            let stats = stats.clone();
            let fault = worker_fault.clone();
            let graph = Arc::clone(&graph);
            let config = config.clone();
            let work_rx = work_rx.clone();
            let sample_tx = sample_tx.clone();
            handles.push(
                thread::Builder::new()
                    .name(format!("aethergraph-sampler-{t}"))
                    .spawn(move || {
                        crate::internal::numa::pin_worker(t);
                        run_worker(&fault, move || {
                            worker_loop_sampler(graph, config, work_rx, sample_tx, shutdown, stats);
                            Ok(())
                        });
                    })
                    .map_err(|e| anyhow::anyhow!("failed to spawn sampler thread: {}", e))?,
            );
        }
        // The workers own the only senders after this point, so the loader's
        // recv disconnects when they all exit.
        drop(sample_tx);

        let loader_handle = {
            let shutdown = shutdown.clone();
            let stats = stats.clone();
            let fault = worker_fault.clone();
            thread::Builder::new()
                .name("aethergraph-feat-loader".into())
                .spawn(move || {
                    run_worker(&fault, move || {
                        worker_loop_feature_loader(
                            sample_rx,
                            result_tx,
                            shutdown,
                            feature_store,
                            stats,
                        );
                        Ok(())
                    });
                })
                .map_err(|e| anyhow::anyhow!("failed to spawn feature loader thread: {}", e))?
        };
        handles.push(loader_handle);

        debug!(
            prefetch_depth,
            sampler_threads,
            feature_dim,
            "NeighborLoader started (pipeline: samplers + feature loader)"
        );

        Ok(Self {
            work_tx: Some(work_tx),
            result_rx: Some(result_rx),
            shutdown,
            handles,
            stats,
            prefetch_depth,
            feature_dim: Some(feature_dim),
            worker_fault,
        })
    }

    /// Create a prefetching sampler for NVMe-backed graphs (Linux only).
    ///
    /// Uses io_uring with SQPOLL for zero-syscall I/O. The io_uring sampling
    /// path honors only `fanout`, `replace`, `cumulative`, and `seed`; edge
    /// ids are always tracked regardless of `track_edge_ids`.
    ///
    /// # Errors
    /// Returns `InvalidInput` if `config` sets a field the io_uring path does
    /// not implement: `weighted`, `temporal_strategy`, `disjoint`,
    /// `deterministic`, `max_degree` (note: `SamplingConfig::default()` sets
    /// `max_degree`), or a non-default `subgraph_type`. Also returns an error
    /// if the prefetch thread cannot be spawned.
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
    /// The same `SamplingConfig` restrictions as [`NeighborLoader::new_nvme`]
    /// apply.
    ///
    /// # Errors
    /// Returns an error if `config` sets a field the io_uring path does not
    /// implement (see [`NeighborLoader::new_nvme`]), if the feature file
    /// cannot be loaded, or if the prefetch thread cannot be spawned.
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

    /// Rejects `SamplingConfig` fields the io_uring path does not implement,
    /// rather than silently ignoring them at sample time.
    #[cfg(target_os = "linux")]
    fn validate_nvme_config(config: &SamplingConfig) -> std::io::Result<()> {
        let unsupported = if config.weighted {
            Some("weighted = true")
        } else if config.temporal_strategy.is_some() {
            Some("temporal_strategy = Some(_)")
        } else if config.disjoint {
            Some("disjoint = true")
        } else if config.deterministic {
            Some("deterministic = true")
        } else if config.max_degree.is_some() {
            Some("max_degree = Some(_)")
        } else if config.subgraph_type != super::sampler::SubgraphType::Directional {
            Some("subgraph_type != Directional")
        } else {
            None
        };
        if let Some(field) = unsupported {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "SamplingConfig field not supported by the NVMe io_uring sampler: {field} \
                     (this path honors fanout, replace, cumulative, and seed; use the \
                     in-memory sampler for the rest)"
                ),
            ));
        }
        Ok(())
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
        Self::validate_nvme_config(&config)?;

        // Same bounding strategy as the other constructors: bounded work
        // channel applies producer-side backpressure so a runaway submit
        // loop can't grow RAM without limit.
        let work_capacity = prefetch_depth.saturating_mul(8).max(prefetch_depth);
        let (work_tx, work_rx) = bounded::<PrefetchWork>(work_capacity);
        let (result_tx, result_rx) = bounded::<PrefetchResult>(prefetch_depth);

        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(PrefetchStats::default());
        let worker_fault = Arc::new(OnceLock::new());

        let feature_dim = feature_store.as_ref().map(|s| s.feature_dim());
        let path = graph_path.to_path_buf();
        let handle = {
            let shutdown = shutdown.clone();
            let fault = worker_fault.clone();
            thread::Builder::new()
                .name("aethergraph-prefetch-nvme".into())
                .spawn(move || {
                    run_worker(&fault, move || {
                        Self::worker_loop_nvme(
                            &path,
                            config,
                            work_rx,
                            result_tx,
                            shutdown,
                            feature_store,
                        )
                    });
                })?
        };

        debug!(
            prefetch_depth,
            ?feature_dim,
            "NeighborLoader started (NVMe io_uring mode)"
        );

        Ok(Self {
            work_tx: Some(work_tx),
            result_rx: Some(result_rx),
            shutdown,
            handles: vec![handle],
            stats,
            prefetch_depth,
            feature_dim,
            worker_fault,
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
    /// Returns:
    /// - `Ok(Some(_))` when a batch is ready.
    /// - `Ok(None)` after `shutdown()` — the clean end-of-stream state.
    /// - `Err(PrefetchError::Timeout { .. })` when no result arrived within
    ///   30s (logged at warn); the worker may just be slow, so the caller may
    ///   call again.
    /// - `Err(PrefetchError::WorkerExited { .. })` when the worker exited
    ///   without `shutdown()` being requested (panic or internal error).
    pub fn next(&self) -> Result<Option<SampledSubgraph>, PrefetchError> {
        Ok(self.next_timeout(RECV_TIMEOUT)?.map(|r| r.subgraph))
    }

    /// Blocking receive with an explicit wait window (shared by the
    /// `next*` methods).
    fn next_timeout(&self, timeout: Duration) -> Result<Option<PrefetchResult>, PrefetchError> {
        let Some(result_rx) = self.result_rx.as_ref() else {
            return Ok(None);
        };
        self.stats.total.fetch_add(1, Ordering::Relaxed);

        // Try non-blocking first to track hit rate
        match result_rx.try_recv() {
            Ok(result) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                trace!(batch_idx = result.batch_idx, "prefetch hit");
                Ok(Some(result))
            }
            Err(TryRecvError::Empty) => {
                // Cache miss - need to wait (with timeout to detect issues)
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                trace!("prefetch miss - blocking");
                match result_rx.recv_timeout(timeout) {
                    Ok(r) => Ok(Some(r)),
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        warn!(
                            "Prefetch timeout after {:?} - worker may have deadlocked or I/O is very slow",
                            timeout
                        );
                        Err(PrefetchError::Timeout { waited: timeout })
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => self.disconnected(),
                }
            }
            Err(TryRecvError::Disconnected) => self.disconnected(),
        }
    }

    /// Maps a disconnected result channel to its meaning: clean end of
    /// stream when shutdown was requested, worker death otherwise.
    fn disconnected(&self) -> Result<Option<PrefetchResult>, PrefetchError> {
        if self.shutdown.load(Ordering::Relaxed) {
            debug!("prefetch channel disconnected - loader shut down");
            Ok(None)
        } else {
            Err(PrefetchError::WorkerExited {
                message: self.worker_fault.get().cloned(),
            })
        }
    }

    /// Try to get next without blocking.
    ///
    /// Returns `Ok(None)` when no batch is ready yet or the loader has been
    /// shut down, and `Err(PrefetchError::WorkerExited { .. })` when the
    /// worker exited without `shutdown()` being requested.
    pub fn try_next(&self) -> Result<Option<SampledSubgraph>, PrefetchError> {
        let Some(result_rx) = self.result_rx.as_ref() else {
            return Ok(None);
        };
        match result_rx.try_recv() {
            Ok(result) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                self.stats.total.fetch_add(1, Ordering::Relaxed);
                Ok(Some(result.subgraph))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => self.disconnected().map(|r| r.map(|r| r.subgraph)),
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
    /// Returns `(subgraph, features)` pairs; `features` is `Some` when the
    /// prefetcher was created with a feature column attached (e.g.
    /// `with_features()`) and `None` when it was not.
    ///
    /// Returns:
    /// - `Ok(Some(_))` when a batch is ready.
    /// - `Ok(None)` after `shutdown()` — the clean end-of-stream state.
    /// - `Err(PrefetchError::Timeout { .. })` when no result arrived within
    ///   30s; the caller may call again.
    /// - `Err(PrefetchError::WorkerExited { .. })` when the worker exited
    ///   without `shutdown()` being requested.
    /// - `Err(PrefetchError::FeatureLoad { .. })` when a feature column is
    ///   attached but loading this batch's features failed.
    pub fn next_with_features(&self) -> Result<Option<SubgraphWithFeatures>, PrefetchError> {
        match self.next_timeout(RECV_TIMEOUT)? {
            None => Ok(None),
            Some(result) => match result.features {
                None => Ok(Some((result.subgraph, None))),
                Some(Ok(features)) => Ok(Some((result.subgraph, Some(features)))),
                Some(Err(source)) => Err(PrefetchError::FeatureLoad {
                    batch_idx: result.batch_idx,
                    source,
                }),
            },
        }
    }

    /// Shutdown the prefetch thread(s).
    ///
    /// Closes the work channel, drops the result receiver so a worker blocked
    /// mid-`send` on a full result channel unblocks, then joins the workers.
    /// Undelivered results are discarded; subsequent consumer calls return
    /// `Ok(None)`.
    #[tracing::instrument(skip(self), fields(num_handles = self.handles.len()))]
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Drop sender to unblock workers waiting for work (take it so the
        // channel actually closes)
        drop(self.work_tx.take());
        // Drop receiver to unblock workers waiting to deliver a result
        drop(self.result_rx.take());
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
                        (Some(Ok(feats)), Some(store.feature_dim()))
                    }
                    Err(e) => {
                        warn!(batch_idx = work.batch_idx, error = %e, "feature load failed");
                        (Some(Err(e)), Some(store.feature_dim()))
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
                    Ok(feats) => (Some(Ok(feats)), Some(store.feature_dim())),
                    Err(e) => {
                        warn!(batch_idx = work.batch_idx, error = %e, "NVMe feature load failed");
                        (Some(Err(e)), Some(store.feature_dim()))
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

/// Prefetching heterogeneous neighbor sampler.
///
/// Same pipeline shape as [`NeighborLoader`]: `sampler_threads` worker
/// threads pull seed batches from a bounded MPMC work channel, sample with
/// their own [`HeteroNeighborSampler`], and deliver results tagged with
/// their `batch_idx` — unordered across the pool, no reorder buffer. The
/// seed node type is fixed at construction; every submitted batch is rooted
/// at it.
pub struct HeteroNeighborLoader {
    /// Send work to the sampler pool (Option so we can take it on shutdown)
    work_tx: Option<Sender<PrefetchWork>>,
    /// Receive results from the pool (Option so shutdown can drop it and
    /// unblock a worker mid-`send` on a full result channel)
    result_rx: Option<Receiver<HeteroPrefetchResult>>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Sampler thread handles
    handles: Vec<JoinHandle<()>>,
    /// Statistics
    stats: Arc<PrefetchStats>,
    /// Prefetch depth
    prefetch_depth: usize,
    /// Why a worker exited on its own (captured panic or error), if it did
    worker_fault: Arc<OnceLock<String>>,
}

impl HeteroNeighborLoader {
    /// Create a prefetching sampler over an in-memory [`HeteroGraph`].
    ///
    /// # Arguments
    /// * `graph` - Arc to the heterogeneous graph (shared with the workers)
    /// * `config` - Sampling configuration (per-edge-type fanout per hop)
    /// * `seed_type` - Node type every submitted seed batch is rooted at
    /// * `prefetch_depth` - How many batches to keep ready (default: 2-3)
    /// * `sampler_threads` - Sampler worker count (0 is treated as 1). The
    ///   work channel is MPMC and results carry their `batch_idx`, so extra
    ///   workers scale sampling throughput with no ordering machinery —
    ///   consumers must already match results by content, not arrival order.
    ///
    /// # Errors
    /// Returns `InvalidInput` if `prefetch_depth` is 0 or `seed_type` is not
    /// a node type of `graph`, and an error if a sampler thread cannot be
    /// spawned.
    #[tracing::instrument(
        skip(graph, config),
        fields(node_types = graph.node_type_count(), prefetch_depth)
    )]
    pub fn new(
        graph: Arc<HeteroGraph>,
        config: HeteroSamplingConfig,
        seed_type: NodeTypeId,
        prefetch_depth: usize,
        sampler_threads: usize,
    ) -> std::io::Result<Self> {
        if prefetch_depth == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "prefetch_depth must be >= 1",
            ));
        }
        if (seed_type as usize) >= graph.node_type_count() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "seed_type {} out of range ({} node types)",
                    seed_type,
                    graph.node_type_count()
                ),
            ));
        }
        let sampler_threads = sampler_threads.max(1);

        // Same bounding strategy as `NeighborLoader::new`:
        //   - work channel  : producer blocks when the workers fall behind
        //                     by more than `prefetch_depth * 8` submitted
        //                     batches, preventing unbounded RAM growth.
        //   - result channel: workers block when the consumer falls behind;
        //                     at least one slot per sampler so a burst of
        //                     simultaneous completions doesn't immediately
        //                     block the pool.
        let work_capacity = prefetch_depth.saturating_mul(8).max(prefetch_depth);
        let (work_tx, work_rx) = bounded::<PrefetchWork>(work_capacity);
        let (result_tx, result_rx) =
            bounded::<HeteroPrefetchResult>(prefetch_depth.max(sampler_threads));

        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(PrefetchStats::default());
        let worker_fault = Arc::new(OnceLock::new());

        let mut handles = Vec::with_capacity(sampler_threads);
        for t in 0..sampler_threads {
            let shutdown = shutdown.clone();
            let stats = stats.clone();
            let fault = worker_fault.clone();
            let graph = Arc::clone(&graph);
            let config = config.clone();
            let work_rx = work_rx.clone();
            let result_tx = result_tx.clone();
            handles.push(
                thread::Builder::new()
                    .name(format!("aethergraph-hetero-sampler-{t}"))
                    .spawn(move || {
                        crate::internal::numa::pin_worker(t);
                        run_worker(&fault, move || {
                            worker_loop_hetero(
                                graph, config, seed_type, work_rx, result_tx, shutdown, stats,
                            );
                            Ok(())
                        });
                    })?,
            );
        }
        // The workers own the only result senders after this point, so the
        // consumer's recv disconnects when they all exit.
        drop(result_tx);

        debug!(
            prefetch_depth,
            sampler_threads, "HeteroNeighborLoader started (sampler pool)"
        );

        Ok(Self {
            work_tx: Some(work_tx),
            result_rx: Some(result_rx),
            shutdown,
            handles,
            stats,
            prefetch_depth,
            worker_fault,
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

    /// Get next sampled subgraph (blocking).
    ///
    /// Returns:
    /// - `Ok(Some(_))` when a batch is ready.
    /// - `Ok(None)` after `shutdown()` — the clean end-of-stream state.
    /// - `Err(PrefetchError::Timeout { .. })` when no result arrived within
    ///   30s (logged at warn); the workers may just be slow, so the caller
    ///   may call again.
    /// - `Err(PrefetchError::WorkerExited { .. })` when the workers exited
    ///   without `shutdown()` being requested (panic or internal error).
    pub fn next(&self) -> Result<Option<HeteroSampledSubgraph>, PrefetchError> {
        Ok(self.next_timeout(RECV_TIMEOUT)?.map(|r| r.subgraph))
    }

    /// Blocking receive with an explicit wait window (shared by the
    /// `next*` methods).
    fn next_timeout(
        &self,
        timeout: Duration,
    ) -> Result<Option<HeteroPrefetchResult>, PrefetchError> {
        let Some(result_rx) = self.result_rx.as_ref() else {
            return Ok(None);
        };
        self.stats.total.fetch_add(1, Ordering::Relaxed);

        // Try non-blocking first to track hit rate
        match result_rx.try_recv() {
            Ok(result) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                trace!(batch_idx = result.batch_idx, "hetero prefetch hit");
                Ok(Some(result))
            }
            Err(TryRecvError::Empty) => {
                // Cache miss - need to wait (with timeout to detect issues)
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                trace!("hetero prefetch miss - blocking");
                match result_rx.recv_timeout(timeout) {
                    Ok(r) => Ok(Some(r)),
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        warn!(
                            "Hetero prefetch timeout after {:?} - workers may have deadlocked or are very slow",
                            timeout
                        );
                        Err(PrefetchError::Timeout { waited: timeout })
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => self.disconnected(),
                }
            }
            Err(TryRecvError::Disconnected) => self.disconnected(),
        }
    }

    /// Maps a disconnected result channel to its meaning: clean end of
    /// stream when shutdown was requested, worker death otherwise.
    fn disconnected(&self) -> Result<Option<HeteroPrefetchResult>, PrefetchError> {
        if self.shutdown.load(Ordering::Relaxed) {
            debug!("hetero prefetch channel disconnected - loader shut down");
            Ok(None)
        } else {
            Err(PrefetchError::WorkerExited {
                message: self.worker_fault.get().cloned(),
            })
        }
    }

    /// Try to get next without blocking.
    ///
    /// Returns `Ok(None)` when no batch is ready yet or the loader has been
    /// shut down, and `Err(PrefetchError::WorkerExited { .. })` when the
    /// workers exited without `shutdown()` being requested.
    pub fn try_next(&self) -> Result<Option<HeteroSampledSubgraph>, PrefetchError> {
        let Some(result_rx) = self.result_rx.as_ref() else {
            return Ok(None);
        };
        match result_rx.try_recv() {
            Ok(result) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                self.stats.total.fetch_add(1, Ordering::Relaxed);
                Ok(Some(result.subgraph))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => self.disconnected().map(|r| r.map(|r| r.subgraph)),
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

    /// Shutdown the sampler pool.
    ///
    /// Closes the work channel, drops the result receiver so a worker blocked
    /// mid-`send` on a full result channel unblocks, then joins the workers.
    /// Undelivered results are discarded; subsequent consumer calls return
    /// `Ok(None)`.
    #[tracing::instrument(skip(self), fields(num_handles = self.handles.len()))]
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Drop sender to unblock workers waiting for work (take it so the
        // channel actually closes)
        drop(self.work_tx.take());
        // Drop receiver to unblock workers waiting to deliver a result
        drop(self.result_rx.take());
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

impl Drop for HeteroNeighborLoader {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// io_uring-based k-hop sampling for NVMe graphs.
///
/// This path honors `fanout`, `replace`, `cumulative`, and `seed`. Edge ids
/// are always tracked (`track_edge_ids` is ignored) and edges are always
/// returned directional. The remaining `SamplingConfig` fields (`weighted`,
/// `temporal_strategy`, `disjoint`, `deterministic`, `max_degree`,
/// non-default `subgraph_type`) are rejected by the NVMe constructors;
/// callers needing those must use the in-memory [`NeighborSampler`] path.
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

        // Batch read all frontier neighbors via io_uring into one flat
        // arena; `spans` locates each node's slice.
        let (flat_neighbors, spans) =
            batch_read_neighbors_uring(handle, fd, offsets, edges_start, num_nodes, &frontier)?;

        let mut next_frontier = Vec::new();

        for (&node, &(span_start, span_len)) in frontier.iter().zip(spans.iter()) {
            if span_len == 0 {
                continue;
            }
            let neighbors =
                &flat_neighbors[span_start as usize..span_start as usize + span_len as usize];

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

        if config.cumulative {
            // Reuse the frontier allocation instead of reallocating the
            // whole accumulated list every hop.
            frontier.append(&mut next_frontier);
        } else {
            frontier = next_frontier;
        }
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

/// All neighbor lists back-to-back in one flat allocation, with
/// `spans[i]` giving `(start, len)` into it for input node `i`
/// (zero-length for invalid/zero-degree nodes).
#[cfg(target_os = "linux")]
type FlatNeighbors = (Vec<NodeId>, Vec<(u32, u32)>);

/// Batch read neighbors for multiple nodes using io_uring.
///
/// Uses SQPOLL-aware submission and registered file descriptors. All
/// neighbor lists land back-to-back in one flat `Vec<NodeId>` — a single
/// allocation whose typed backing io_uring writes into directly, so on
/// little-endian targets there is no per-node buffer, no second decode
/// pass, and no copy.
#[cfg(target_os = "linux")]
fn batch_read_neighbors_uring(
    handle: &mut crate::internal::uring::UringHandle,
    fd: i32,
    offsets: &[u64],
    edges_start: u64,
    num_nodes: usize,
    nodes: &[NodeId],
) -> anyhow::Result<FlatNeighbors> {
    use crate::internal::uring::batch_read;

    let mut spans: Vec<(u32, u32)> = Vec::with_capacity(nodes.len());
    let mut total: usize = 0;
    for &node in nodes {
        let idx = node as usize;
        if idx >= num_nodes {
            spans.push((total as u32, 0));
            continue;
        }
        let start = offsets[idx] as usize;
        let end = offsets[idx + 1] as usize;
        spans.push((total as u32, (end - start) as u32));
        total += end - start;
    }

    let mut flat: Vec<NodeId> = vec![0; total];
    if total > 0 {
        let mut reads: Vec<(u64, *mut u8, usize)> = Vec::with_capacity(nodes.len());
        {
            let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut flat);
            let base = bytes.as_mut_ptr();
            for (&node, &(span_start, span_len)) in nodes.iter().zip(spans.iter()) {
                if span_len == 0 {
                    continue;
                }
                let start = offsets[node as usize] as usize;
                let byte_offset = edges_start + (start * 4) as u64;
                // SAFETY: span offsets were accumulated to fit exactly in
                // `flat`, so `span_start * 4 .. span_start * 4 + span_len * 4`
                // is in bounds.
                let ptr = unsafe { base.add(span_start as usize * 4) };
                reads.push((byte_offset, ptr, span_len as usize * 4));
            }
        }
        // SAFETY: every ptr in `reads` points into `flat`'s backing, which
        // lives until this function returns; batch_read reaps every
        // submitted completion before returning.
        batch_read(handle, fd, &reads)?;
    }

    Ok((flat, spans))
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

        let prefetcher = NeighborLoader::new(graph, config, 2, 1).unwrap();

        prefetcher.submit(0, vec![0]).unwrap();
        prefetcher.submit(1, vec![1]).unwrap();
        prefetcher.submit(2, vec![2]).unwrap();

        let sg1 = prefetcher.next().unwrap().unwrap();
        assert_eq!(sg1.num_seeds(), 1);

        let sg2 = prefetcher.next().unwrap().unwrap();
        assert_eq!(sg2.num_seeds(), 1);

        let sg3 = prefetcher.next().unwrap().unwrap();
        assert_eq!(sg3.num_seeds(), 1);
    }

    #[test]
    fn multi_thread_sampler_pool_delivers_every_batch() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            seed: Some(7),
            ..Default::default()
        };

        let prefetcher = NeighborLoader::new(graph, config, 4, 4).unwrap();
        // Stay inside the bounded work channel (prefetch_depth * 8): the
        // producer here is also the consumer, so overfilling would just be
        // backpressure deadlocking the test, not a pool property.
        let n = 24usize;
        for i in 0..n {
            prefetcher.submit(i, vec![(i % 5) as u32]).unwrap();
        }

        // Results may arrive in any order across the pool; every batch
        // index must arrive exactly once with a valid subgraph.
        let mut seen = vec![false; n];
        for _ in 0..n {
            let r = prefetcher
                .next_timeout(Duration::from_secs(30))
                .unwrap()
                .unwrap();
            assert!(!seen[r.batch_idx], "batch {} delivered twice", r.batch_idx);
            seen[r.batch_idx] = true;
            assert_eq!(r.subgraph.num_seeds(), 1);
        }
        assert!(seen.iter().all(|&s| s), "missing batches: {seen:?}");
    }

    fn create_test_hetero_graph() -> Arc<HeteroGraph> {
        let mut edges = Vec::new();
        for user in 0u32..50 {
            for post in 0u32..4 {
                edges.push((user, post));
            }
        }
        let csr = Graph::from_edges(100, &edges, None).unwrap();
        Arc::new(HeteroGraph::from_parts(
            vec![("user".into(), 100), ("post".into(), 100)],
            vec![("user".into(), "votes".into(), "post".into(), csr)],
        ))
    }

    #[test]
    fn hetero_multi_thread_sampler_pool_delivers_every_batch() {
        let graph = create_test_hetero_graph();
        let config = HeteroSamplingConfig {
            fanout: vec![vec![2]],
            replace: false,
            seed: Some(7),
            max_degree: None,
            num_hops: 1,
        };
        let user_type: NodeTypeId = 0;

        let loader = HeteroNeighborLoader::new(graph, config, user_type, 4, 4).unwrap();
        // Stay inside the bounded work channel (prefetch_depth * 8): the
        // producer here is also the consumer, so overfilling would just be
        // backpressure deadlocking the test, not a pool property.
        let n = 24usize;
        for i in 0..n {
            loader.submit(i, vec![(i % 50) as u32]).unwrap();
        }

        // Results may arrive in any order across the pool; every batch
        // index must arrive exactly once with a valid subgraph.
        let mut seen = vec![false; n];
        for _ in 0..n {
            let r = loader
                .next_timeout(Duration::from_secs(30))
                .unwrap()
                .unwrap();
            assert!(!seen[r.batch_idx], "batch {} delivered twice", r.batch_idx);
            seen[r.batch_idx] = true;
            assert_eq!(r.subgraph.seeds, vec![(r.batch_idx % 50) as u32]);
        }
        assert!(seen.iter().all(|&s| s), "missing batches: {seen:?}");
    }

    #[test]
    fn test_prefetch_hit_rate() {
        let graph = create_test_graph();
        let config = SamplingConfig::default();

        let prefetcher = NeighborLoader::new(graph, config, 3, 1).unwrap();

        // Submit all upfront
        for i in 0..10 {
            prefetcher.submit(i, vec![i as u32 % 5]).unwrap();
        }

        // Let worker prefetch
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Consume all
        for _ in 0..10 {
            prefetcher.next().unwrap().unwrap();
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
    fn test_sync_feature_store_rejects_zero_offset_header() {
        let temp_file = NamedTempFile::new().unwrap();

        // A zero payload offset is invalid — the dtype tag lives at byte 32,
        // so the payload must start past it.
        let features = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(temp_file.path())
            .unwrap();
        file.write_all(b"AETHFEAT").unwrap();
        file.write_all(&(2u64).to_le_bytes()).unwrap();
        file.write_all(&(3u64).to_le_bytes()).unwrap();
        file.write_all(&(0u64).to_le_bytes()).unwrap();
        let feature_bytes: &[u8] = bytemuck::cast_slice(&features);
        file.write_all(feature_bytes).unwrap();
        file.sync_all().unwrap();

        assert!(SyncFeatureStore::load(temp_file.path()).is_err());
    }

    #[test]
    fn test_shutdown_unblocks_blocked_worker() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2],
            replace: false,
            seed: Some(7),
            ..Default::default()
        };

        // prefetch_depth 1 => result channel capacity 1. Submitting several
        // batches leaves the worker blocked mid-`send` on a full channel.
        let mut prefetcher = NeighborLoader::new(graph, config, 1, 1).unwrap();
        for i in 0..4 {
            prefetcher.submit(i, vec![i as u32 % 5]).unwrap();
        }
        std::thread::sleep(Duration::from_millis(100));

        // Must return promptly even though the worker is blocked in send.
        prefetcher.shutdown();

        // After shutdown, consumers see a clean end of stream.
        assert!(matches!(prefetcher.next(), Ok(None)));
        assert!(matches!(prefetcher.try_next(), Ok(None)));
        assert!(matches!(prefetcher.next_with_features(), Ok(None)));
    }

    #[test]
    fn test_next_reports_timeout() {
        let graph = create_test_graph();
        let prefetcher = NeighborLoader::new(graph, SamplingConfig::default(), 2, 1).unwrap();

        // Nothing submitted: the worker is alive but has no results.
        let waited = Duration::from_millis(50);
        match prefetcher.next_timeout(waited) {
            Err(PrefetchError::Timeout { waited: w }) => assert_eq!(w, waited),
            other => panic!("expected Timeout, got {other:?}"),
        }
    }

    #[test]
    fn test_worker_exit_surfaces_error() {
        // Build a loader whose worker panics immediately, without shutdown()
        // being requested. The fault is recorded before the channel closes,
        // so the consumer sees the panic message.
        let (work_tx, _work_rx) = bounded::<PrefetchWork>(1);
        let (result_tx, result_rx) = bounded::<PrefetchResult>(1);
        let worker_fault = Arc::new(OnceLock::new());
        let handle = {
            let fault = worker_fault.clone();
            thread::Builder::new()
                .name("aethergraph-test-worker".into())
                .spawn(move || {
                    let _result_tx = result_tx;
                    run_worker(&fault, || panic!("worker died"));
                })
                .unwrap()
        };
        let loader = NeighborLoader {
            work_tx: Some(work_tx),
            result_rx: Some(result_rx),
            shutdown: Arc::new(AtomicBool::new(false)),
            handles: vec![handle],
            stats: Arc::new(PrefetchStats::default()),
            prefetch_depth: 1,
            feature_dim: None,
            worker_fault,
        };

        match loader.next() {
            Err(PrefetchError::WorkerExited {
                message: Some(message),
            }) => {
                assert!(message.contains("worker died"), "got: {message}");
            }
            other => panic!("expected WorkerExited, got {other:?}"),
        }
        assert!(matches!(
            loader.try_next(),
            Err(PrefetchError::WorkerExited { .. })
        ));
    }

    #[test]
    fn test_feature_load_error_surfaces() {
        // Graph has 5 nodes but the feature file only covers 2, so loading
        // features for a batch that samples node 4 fails.
        let graph = create_test_graph();
        let temp_file = NamedTempFile::new().unwrap();
        let features = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        crate::features::save_features(temp_file.path(), features, 2, 3).unwrap();

        let config = SamplingConfig {
            fanout: vec![2],
            replace: false,
            seed: Some(1),
            ..Default::default()
        };
        let loader = NeighborLoader::with_features(graph, config, temp_file.path(), 2, 1).unwrap();
        loader.submit(0, vec![4]).unwrap();

        match loader.next_with_features() {
            Err(PrefetchError::FeatureLoad { batch_idx, source }) => {
                assert_eq!(batch_idx, 0);
                assert!(
                    source.to_string().contains("out of bounds"),
                    "got: {source}"
                );
            }
            other => panic!("expected FeatureLoad, got {other:?}"),
        }
    }
}
