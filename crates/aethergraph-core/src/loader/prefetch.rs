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
//! # Delivery and shutdown
//!
//! Every submission gets a loader-assigned sequence number. A seeded loader
//! reseeds per submission from that number and yields results in submission
//! order, so a multi-worker pool reproduces a single worker exactly. Every
//! blocking point — `submit`, the workers' receive and send, the consumer's
//! receive — also watches one stop signal, tripped by `shutdown` or by the
//! first worker fault, so either wakes every thread at once.
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
use super::sampler::{
    NeighborSampler, SampledSubgraph, SamplingConfig, Seeds, batch_seed, check_config,
};
use crate::features::header::{FeatureDtype, parse_feature_header};
use crate::graph::hetero::{HeteroGraph, NodeTypeId};
use crate::graph::{Graph, NodeId};
#[cfg(any(target_os = "linux", test))]
use crate::internal::genstamp::WyRand;
use crate::internal::hint;
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded, select};
use parking_lot::Mutex;
#[cfg(any(target_os = "linux", test))]
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
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

/// Rows within one page of each other share a readahead hint. A hint
/// faults whole pages anyway, so a merge wastes at most a page and saves a
/// syscall per row.
const HINT_MERGE_GAP_BYTES: u64 = 4096;

/// How long a blocking consumer call waits for the worker before reporting
/// [`PrefetchError::Timeout`].
const RECV_TIMEOUT: Duration = Duration::from_secs(30);

/// A queued item tagged with the loader-assigned submission number.
struct Sequenced<T> {
    seq: usize,
    item: T,
}

/// Holds out-of-order results until the next submission number arrives.
/// Used by seeded loaders so yield order matches submit order even with a
/// multi-worker pool.
struct BatchReorder<T> {
    next_seq: usize,
    pending: BTreeMap<usize, T>,
}

impl<T> BatchReorder<T> {
    fn new() -> Self {
        Self {
            next_seq: 0,
            pending: BTreeMap::new(),
        }
    }

    fn pop_ready(&mut self) -> Option<T> {
        let item = self.pending.remove(&self.next_seq)?;
        self.next_seq += 1;
        Some(item)
    }
}

/// Work item for the prefetch thread.
#[derive(Debug)]
pub struct PrefetchWork {
    /// Caller bookkeeping, echoed back on the result.
    pub batch_idx: usize,
    /// Seed nodes for this batch, range-checked against the loader's graph.
    pub seeds: Seeds,
}

/// Completed prefetch result.
#[derive(Debug)]
pub struct PrefetchResult {
    /// The `batch_idx` the submission carried.
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
    /// The `batch_idx` the submission carried.
    pub batch_idx: usize,
    /// Sampled heterogeneous subgraph
    pub subgraph: HeteroSampledSubgraph,
}

/// One delivered batch: the `batch_idx` its submission carried, the sampled
/// subgraph, and its features when the loader has a feature column.
#[derive(Debug)]
pub struct LoadedBatch {
    pub batch_idx: usize,
    pub subgraph: SampledSubgraph,
    pub features: Option<Vec<f32>>,
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
    /// A worker exited without `shutdown()` being requested (panic or
    /// internal error). Reported as soon as the fault happens, even while
    /// other workers keep running: the faulted batch is lost, so the stream
    /// cannot complete. `message` carries the captured panic payload or
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
            Self::Timeout { waited } => {
                write!(f, "prefetch timed out after {waited:?}")
            }
            Self::WorkerExited {
                message: Some(message),
            } => {
                write!(f, "prefetch worker exited: {message}")
            }
            Self::WorkerExited { message: None } => {
                write!(f, "prefetch worker exited unexpectedly")
            }
            Self::FeatureLoad { batch_idx, source } => {
                write!(f, "feature load failed for batch {batch_idx}: {source}")
            }
        }
    }
}

impl std::error::Error for PrefetchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FeatureLoad { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl PrefetchResult {
    fn into_loaded(self) -> Result<LoadedBatch, PrefetchError> {
        let features = match self.features {
            None => None,
            Some(Ok(features)) => Some(features),
            Some(Err(source)) => {
                return Err(PrefetchError::FeatureLoad {
                    batch_idx: self.batch_idx,
                    source,
                });
            }
        };
        Ok(LoadedBatch {
            batch_idx: self.batch_idx,
            subgraph: self.subgraph,
            features,
        })
    }
}

/// Stop state shared by one loader's threads.
///
/// Tripped at most once — by `shutdown` or by the first worker fault — and
/// every blocking point in the pipeline selects on `tripped`, which
/// disconnects on the trip and so wakes all of them.
struct StopSignal {
    stopped: AtomicBool,
    shutdown: AtomicBool,
    fault: OnceLock<String>,
    trip: Mutex<Option<Sender<()>>>,
    tripped: Receiver<()>,
}

impl StopSignal {
    fn new() -> Self {
        let (trip, tripped) = bounded(0);
        Self {
            stopped: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            fault: OnceLock::new(),
            trip: Mutex::new(Some(trip)),
            tripped,
        }
    }

    fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }

    /// Records a worker fault and stops the pipeline. The first fault wins;
    /// later ones describe its fallout.
    fn fail(&self, message: String) {
        let _ = self.fault.set(message);
        self.stop();
    }

    fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.stop();
    }

    fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
        // Dropping the only sender disconnects `tripped` for every waiter.
        drop(self.trip.lock().take());
    }

    /// What a consumer sees once the pipeline stopped or its channels
    /// closed: the clean end of stream after `shutdown`, the fault otherwise.
    fn outcome<T>(&self) -> Result<Option<T>, PrefetchError> {
        if self.shutdown.load(Ordering::Acquire) {
            Ok(None)
        } else {
            Err(PrefetchError::WorkerExited {
                message: self.fault.get().cloned(),
            })
        }
    }

    fn refusal(&self) -> SubmitError {
        if self.shutdown.load(Ordering::Acquire) {
            SubmitError::Shutdown
        } else {
            SubmitError::WorkerExited
        }
    }
}

/// Runs a worker body, recording a panic payload or error as the pipeline's
/// fault so every blocked consumer and submitter wakes with it. Workers only
/// finish cleanly after the loader stops; any earlier exit is a fault too.
fn run_worker(stop: &StopSignal, body: impl FnOnce() -> anyhow::Result<()>) {
    let message = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(Ok(())) if stop.is_stopped() => return,
        Ok(Ok(())) => "worker finished before the loader stopped".to_string(),
        Ok(Err(e)) => format!("{e:#}"),
        Err(payload) => panic_message(payload.as_ref()),
    };
    warn!("prefetch worker fault: {}", message);
    stop.fail(message);
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

/// Next queued item, or `None` once the loader stops or the queue closes.
fn recv_work<W>(rx: &Receiver<W>, stop: &StopSignal) -> Option<W> {
    if stop.is_stopped() {
        return None;
    }
    select! {
        recv(rx) -> item => item.ok(),
        recv(stop.tripped) -> _ => None,
    }
}

/// Hands `item` downstream; `false` once the loader stops or the receiving
/// side is gone.
fn deliver<T>(tx: &Sender<T>, item: T, stop: &StopSignal) -> bool {
    select! {
        send(tx, item) -> sent => sent.is_ok(),
        recv(stop.tripped) -> _ => false,
    }
}

/// The submit/deliver machinery every loader shares: a bounded work queue
/// into a thread pool, a bounded result queue back, a stop signal every
/// blocking point watches, and — for seeded loaders — a reorder buffer that
/// yields in submission order. All methods take `&self`, so one thread can
/// shut the loader down while others are blocked in `submit` or `next`.
struct Pipeline<T> {
    work_tx: Sender<Sequenced<PrefetchWork>>,
    result_rx: Receiver<Sequenced<T>>,
    /// Keeps the result queue connected for the pipeline's lifetime. A
    /// panicking worker drops its own sender while unwinding, before its
    /// fault is recorded; with this one held, consumers learn why the
    /// stream ended from the stop signal, never from a bare disconnect.
    _result_hold: Sender<Sequenced<T>>,
    stop: Arc<StopSignal>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    next_seq: AtomicUsize,
    stats: Arc<PrefetchStats>,
    prefetch_depth: usize,
    ordered: bool,
    reorder: Mutex<BatchReorder<T>>,
    /// Node count submitted seeds must be checked within: the graph's, or
    /// the seed type's.
    seed_nodes: usize,
}

/// The pool side of a [`Pipeline`]'s queues. The constructor clones what
/// each thread needs and drops the rest.
struct PoolEnds<T> {
    work_rx: Receiver<Sequenced<PrefetchWork>>,
    result_tx: Sender<Sequenced<T>>,
    stop: Arc<StopSignal>,
    stats: Arc<PrefetchStats>,
}

impl<T> Clone for PoolEnds<T> {
    fn clone(&self) -> Self {
        Self {
            work_rx: self.work_rx.clone(),
            result_tx: self.result_tx.clone(),
            stop: Arc::clone(&self.stop),
            stats: Arc::clone(&self.stats),
        }
    }
}

impl<T: Send + 'static> Pipeline<T> {
    /// Both queues are bounded. `submit` blocks once the pool falls
    /// `prefetch_depth * 8` batches behind, so a producer staging many
    /// epochs cannot grow RAM without limit; workers block once the
    /// consumer falls `result_capacity` results behind.
    fn new(
        prefetch_depth: usize,
        result_capacity: usize,
        ordered: bool,
        seed_nodes: usize,
    ) -> (Self, PoolEnds<T>) {
        let work_capacity = prefetch_depth.saturating_mul(8).max(prefetch_depth);
        let (work_tx, work_rx) = bounded(work_capacity);
        let (result_tx, result_rx) = bounded(result_capacity.max(1));
        let stop = Arc::new(StopSignal::new());
        let stats = Arc::new(PrefetchStats::default());
        let pipe = Self {
            work_tx,
            result_rx,
            _result_hold: result_tx.clone(),
            stop: Arc::clone(&stop),
            handles: Mutex::new(Vec::new()),
            next_seq: AtomicUsize::new(0),
            stats: Arc::clone(&stats),
            prefetch_depth,
            ordered,
            reorder: Mutex::new(BatchReorder::new()),
            seed_nodes,
        };
        let ends = PoolEnds {
            work_rx,
            result_tx,
            stop,
            stats,
        };
        (pipe, ends)
    }

    /// Spawns one pool thread. A panic or error `body` returns becomes the
    /// pipeline's fault.
    fn spawn(
        &self,
        name: String,
        body: impl FnOnce() -> anyhow::Result<()> + Send + 'static,
    ) -> std::io::Result<()> {
        let stop = Arc::clone(&self.stop);
        let handle = thread::Builder::new()
            .name(name)
            .spawn(move || run_worker(&stop, body))?;
        self.handles.lock().push(handle);
        Ok(())
    }
}

impl<T> Pipeline<T> {
    fn submit(&self, batch_idx: usize, seeds: Seeds) -> Result<(), SubmitError> {
        if seeds.num_nodes() > self.seed_nodes {
            return Err(SubmitError::SeedsExceedGraph {
                checked_against: seeds.num_nodes(),
                num_nodes: self.seed_nodes,
            });
        }
        if self.stop.is_stopped() {
            return Err(self.stop.refusal());
        }
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let work = Sequenced {
            seq,
            item: PrefetchWork { batch_idx, seeds },
        };
        select! {
            send(self.work_tx, work) -> sent => sent.map_err(|_| self.stop.refusal()),
            recv(self.stop.tripped) -> _ => Err(self.stop.refusal()),
        }
    }

    /// Blocking receive with an explicit wait window.
    fn next_timeout(&self, timeout: Duration) -> Result<Option<T>, PrefetchError> {
        if self.stop.is_stopped() {
            return self.stop.outcome();
        }
        let deadline = Instant::now() + timeout;
        let mut waited = false;
        loop {
            if self.ordered
                && let Some(item) = self.reorder.lock().pop_ready()
            {
                self.stats.record_delivery(waited);
                return Ok(Some(item));
            }
            let delivered = match self.result_rx.try_recv() {
                Ok(delivered) => delivered,
                Err(TryRecvError::Disconnected) => return self.stop.outcome(),
                Err(TryRecvError::Empty) => {
                    waited = true;
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    select! {
                        recv(self.result_rx) -> delivered => match delivered {
                            Ok(delivered) => delivered,
                            Err(_) => return self.stop.outcome(),
                        },
                        recv(self.stop.tripped) -> _ => return self.stop.outcome(),
                        default(remaining) => {
                            warn!(
                                "Prefetch timeout after {:?} - worker may have deadlocked or I/O is very slow",
                                timeout
                            );
                            return Err(PrefetchError::Timeout { waited: timeout });
                        }
                    }
                }
            };
            if !self.ordered {
                self.stats.record_delivery(waited);
                return Ok(Some(delivered.item));
            }
            self.reorder
                .lock()
                .pending
                .insert(delivered.seq, delivered.item);
        }
    }

    fn try_next(&self) -> Result<Option<T>, PrefetchError> {
        if self.stop.is_stopped() {
            return self.stop.outcome();
        }
        loop {
            if self.ordered
                && let Some(item) = self.reorder.lock().pop_ready()
            {
                self.stats.record_delivery(false);
                return Ok(Some(item));
            }
            match self.result_rx.try_recv() {
                Ok(delivered) if self.ordered => {
                    self.reorder
                        .lock()
                        .pending
                        .insert(delivered.seq, delivered.item);
                }
                Ok(delivered) => {
                    self.stats.record_delivery(false);
                    return Ok(Some(delivered.item));
                }
                Err(TryRecvError::Empty) => return Ok(None),
                Err(TryRecvError::Disconnected) => return self.stop.outcome(),
            }
        }
    }

    fn shutdown(&self) {
        self.stop.request_shutdown();
        let handles = std::mem::take(&mut *self.handles.lock());
        for h in handles {
            let _ = h.join();
        }
    }
}

impl<T> Drop for Pipeline<T> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Sync feature store for use in prefetch thread.
///
/// Unlike AsyncFeatureStore, this is designed for synchronous access
/// from the prefetch thread, optionally using io_uring for parallel reads.
///
/// On Linux, the ring reads through O_DIRECT with aligned buffers when the
/// layout allows it; readahead hints and the pread fallback always go
/// through a buffered descriptor on the same inode.
pub struct SyncFeatureStore {
    /// Buffered descriptor: parsed at load, and serving readahead hints and
    /// the pread fallback.
    file: Arc<File>,
    /// O_DIRECT descriptor on the same inode, used by the ring.
    #[cfg(target_os = "linux")]
    direct: Option<File>,
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
    /// Sorted-row scratch for readahead hints.
    hint_rows: Vec<NodeId>,
}

impl SyncFeatureStore {
    /// Load feature store from disk.
    ///
    /// On Linux, attempts to open a second, O_DIRECT descriptor for the
    /// io_uring IOPOLL path. Skips it when:
    /// - O_DIRECT is not supported (tmpfs, network FS)
    /// - Feature layout isn't O_DIRECT compatible (unaligned offsets)
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        debug!("Loading sync feature store from {}", path.display());

        let file = File::open(path)?;
        let header = parse_feature_header(&file)?;

        #[cfg(target_os = "linux")]
        let direct = Self::open_direct(path, &file, &header)?;

        debug!(
            "Feature store: {} nodes, {} dims, data_offset={}, O_DIRECT={}",
            header.num_nodes,
            header.feature_dim,
            header.features_start_offset,
            cfg!(target_os = "linux") && {
                #[cfg(target_os = "linux")]
                {
                    direct.is_some()
                }
                #[cfg(not(target_os = "linux"))]
                {
                    false
                }
            }
        );

        #[cfg(target_os = "linux")]
        let uring = Self::setup_uring(direct.as_ref().unwrap_or(&file));

        Ok(Self {
            file: Arc::new(file),
            #[cfg(target_os = "linux")]
            direct,
            num_nodes: header.num_nodes,
            feature_dim: header.feature_dim,
            features_start_offset: header.features_start_offset,
            dtype: header.dtype,
            #[cfg(target_os = "linux")]
            uring,
            hint_rows: Vec::new(),
        })
    }

    /// Opens the O_DIRECT descriptor the ring reads through, when the
    /// device and layout allow it.
    #[cfg(target_os = "linux")]
    fn open_direct(
        path: &Path,
        file: &File,
        header: &crate::features::header::FeatureHeader,
    ) -> anyhow::Result<Option<File>> {
        use crate::internal::uring::{
            DirectIoAlignment, is_layout_direct_io_compatible_with, open_direct_or_fallback,
        };
        use std::os::unix::fs::MetadataExt;

        // Ask the file what its device actually requires: on a 4Kn device
        // a layout that clears 512 but not 4096 would fail every read with
        // EINVAL.
        let alignment = DirectIoAlignment::probe(file);
        if !is_layout_direct_io_compatible_with(
            header.features_start_offset,
            header.feature_size,
            alignment,
        ) {
            warn!(
                "Feature layout not O_DIRECT compatible at {} alignment: offset={} (aligned={}), size={} (aligned={})",
                alignment,
                header.features_start_offset,
                (header.features_start_offset as usize).is_multiple_of(alignment.bytes()),
                header.feature_size,
                header.feature_size.is_multiple_of(alignment.bytes())
            );
            return Ok(None);
        }

        let (direct, is_direct) = open_direct_or_fallback(path)?;
        if !is_direct {
            return Ok(None);
        }
        // The header was parsed from `file`; the ring must read that inode,
        // not whatever the path names by now.
        let (parsed, reopened) = (file.metadata()?, direct.metadata()?);
        anyhow::ensure!(
            parsed.dev() == reopened.dev() && parsed.ino() == reopened.ino(),
            "feature file {} was replaced while it was being opened",
            path.display()
        );
        debug!(
            "Feature layout is O_DIRECT compatible (offset={}, size={}, alignment={})",
            header.features_start_offset, header.feature_size, alignment
        );
        Ok(Some(direct))
    }

    #[cfg(target_os = "linux")]
    fn setup_uring(file: &File) -> Option<crate::internal::uring::UringLane> {
        let mut handle = crate::internal::uring::create_feature_uring(file)?;
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

    /// Buffered handle on the feature file.
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Get features start offset (for prefetch hints)
    pub fn features_start_offset(&self) -> u64 {
        self.features_start_offset
    }

    fn row_bytes(&self) -> usize {
        self.feature_dim * self.dtype.element_size()
    }

    /// Whether this store's reads bypass the page cache.
    fn reads_bypass_cache(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.uring.is_some() && self.direct.is_some()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// Hint the kernel to read ahead the rows of `nodes`.
    ///
    /// Skipped when reads go through O_DIRECT, which bypasses the page cache
    /// the hint would fill. Otherwise one hint covers each run of rows
    /// within a page of each other, so readahead tracks the rows the batch
    /// reads rather than the span between its lowest and highest node.
    pub fn prefetch_nodes(&mut self, nodes: &[NodeId]) {
        if self.reads_bypass_cache() {
            return;
        }
        let row_bytes = self.row_bytes() as u64;
        self.hint_rows.clear();
        self.hint_rows.extend_from_slice(nodes);
        self.hint_rows.sort_unstable();
        self.hint_rows.dedup();
        let mut run: Option<(u64, u64)> = None;
        for &node in &self.hint_rows {
            let start = self.features_start_offset + u64::from(node) * row_bytes;
            let end = start + row_bytes;
            run = match run {
                Some((s, e)) if start <= e + HINT_MERGE_GAP_BYTES => Some((s, end)),
                Some((s, e)) => {
                    hint::prefetch_file_range(&*self.file, s, (e - s) as usize);
                    Some((start, end))
                }
                None => Some((start, end)),
            };
        }
        if let Some((s, e)) = run {
            hint::prefetch_file_range(&*self.file, s, (e - s) as usize);
        }
    }

    /// Load features for multiple nodes (sync, uses io_uring on Linux).
    ///
    /// With O_DIRECT + io_uring IOPOLL, this achieves near-zero-syscall I/O.
    pub fn get_batch(&mut self, nodes: &[NodeId]) -> anyhow::Result<Vec<f32>> {
        for &node in nodes {
            anyhow::ensure!(
                (node as usize) < self.num_nodes,
                "node {node} out of bounds (max {})",
                self.num_nodes
            );
        }
        let feature_size = self.row_bytes();

        #[cfg(target_os = "linux")]
        {
            if let Some(mut lane) = self.uring.take() {
                let fd = self.direct.as_ref().unwrap_or(&self.file).as_raw_fd();
                let result = crate::features::gather::uring_gather_rows(
                    &mut lane,
                    fd,
                    nodes,
                    self.features_start_offset,
                    feature_size,
                    self.direct.is_some(),
                    self.dtype,
                    self.feature_dim,
                );
                match result {
                    // The ring was accepted at setup, but this filesystem
                    // cannot serve the reads it was built for. Drop the lane
                    // — leaving `self.uring` empty — and let this call and
                    // every later one take the buffered path below.
                    Err(ref e) if crate::internal::uring::is_ring_unsupported(e) => {
                        debug!("io_uring gather unsupported for this file ({e}); using pread");
                        drop(lane);
                    }
                    _ => {
                        self.uring = Some(lane);
                        return result;
                    }
                }
            }
        }

        // Hint every row first so the kernel reads them in parallel while
        // the preads below consume them in order.
        self.prefetch_nodes(nodes);
        self.batch_read_sync(nodes, feature_size)
    }

    /// Buffered pread fallback. The landing buffer and the dtype dispatch are
    /// hoisted out of the row loop, and each row decodes as a block rather
    /// than element by element.
    fn batch_read_sync(&self, nodes: &[NodeId], feature_size: usize) -> anyhow::Result<Vec<f32>> {
        let decoder = self.dtype.row_decoder();
        let mut all_features = vec![0f32; nodes.len() * self.feature_dim];
        let mut buffer = vec![0u8; feature_size];

        for (i, &node) in nodes.iter().enumerate() {
            let offset = self.features_start_offset + (u64::from(node) * feature_size as u64);
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
    /// Batches delivered without waiting
    pub hits: AtomicU64,
    /// Batches the consumer had to wait for
    pub misses: AtomicU64,
    /// Total batches delivered
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

    fn record_delivery(&self, waited: bool) {
        self.total.fetch_add(1, Ordering::Relaxed);
        let counter = if waited { &self.misses } else { &self.hits };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn record_sampling(&self, started: Instant) {
        self.sample_time_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// A sampled batch on its way from a sampler thread to the feature loader.
struct SampledWork {
    work: PrefetchWork,
    subgraph: SampledSubgraph,
}

/// Worker loop for the in-memory pool: sample, deliver, repeat. Sampling
/// goes through [`NeighborSampler::sample`], which routes disjoint,
/// temporal, and subgraph modes; the constructor already checked the
/// config against the graph.
fn worker_loop_inmemory(
    graph: &Graph,
    config: SamplingConfig,
    ends: &PoolEnds<PrefetchResult>,
) -> anyhow::Result<()> {
    debug!("In-memory prefetch worker started");
    let base_seed = config.seed;
    let mut sampler = NeighborSampler::new(graph, config);

    while let Some(Sequenced { seq, item: work }) = recv_work(&ends.work_rx, &ends.stop) {
        trace!(
            batch_idx = work.batch_idx,
            seeds = work.seeds.len(),
            "sampling"
        );
        if let Some(base) = base_seed {
            sampler.reseed(batch_seed(base, seq));
        }
        let t0 = Instant::now();
        let subgraph = sampler.sample(&work.seeds, None)?;
        ends.stats.record_sampling(t0);

        let result = PrefetchResult {
            batch_idx: work.batch_idx,
            subgraph,
            features: None,
            feature_dim: None,
        };
        if !deliver(&ends.result_tx, Sequenced { seq, item: result }, &ends.stop) {
            break;
        }
    }

    debug!("In-memory prefetch worker stopped");
    Ok(())
}

/// Sampler-only worker loop for the feature pipeline.
fn worker_loop_sampler(
    graph: &Graph,
    config: SamplingConfig,
    work_rx: &Receiver<Sequenced<PrefetchWork>>,
    sample_tx: &Sender<Sequenced<SampledWork>>,
    stop: &StopSignal,
    stats: &PrefetchStats,
) -> anyhow::Result<()> {
    debug!("Sampler thread started");
    let base_seed = config.seed;
    let mut sampler = NeighborSampler::new(graph, config);

    while let Some(Sequenced { seq, item: work }) = recv_work(work_rx, stop) {
        trace!(
            batch_idx = work.batch_idx,
            seeds = work.seeds.len(),
            "sampling"
        );
        if let Some(base) = base_seed {
            sampler.reseed(batch_seed(base, seq));
        }
        let t0 = Instant::now();
        let subgraph = sampler.sample(&work.seeds, None)?;
        stats.record_sampling(t0);

        let item = SampledWork { work, subgraph };
        if !deliver(sample_tx, Sequenced { seq, item }, stop) {
            break;
        }
    }

    debug!("Sampler thread stopped");
    Ok(())
}

/// Sampler worker loop for the hetero pool.
///
/// Each worker owns its own [`HeteroNeighborSampler`] (the sampler's scratch
/// buffers are per-instance) over the shared `Arc<HeteroGraph>` and drains
/// the MPMC work channel until it closes or the loader stops.
fn worker_loop_hetero(
    graph: &HeteroGraph,
    config: HeteroSamplingConfig,
    seed_type: NodeTypeId,
    ends: &PoolEnds<HeteroPrefetchResult>,
) -> anyhow::Result<()> {
    debug!("Hetero sampler thread started");
    let base_seed = config.seed;
    let mut sampler = HeteroNeighborSampler::new(graph, config);

    while let Some(Sequenced { seq, item: work }) = recv_work(&ends.work_rx, &ends.stop) {
        trace!(
            batch_idx = work.batch_idx,
            seeds = work.seeds.len(),
            "hetero sampling"
        );
        if let Some(base) = base_seed {
            sampler.reseed(batch_seed(base, seq));
        }
        let t0 = Instant::now();
        let subgraph = sampler.sample(seed_type, &work.seeds)?;
        ends.stats.record_sampling(t0);

        let result = HeteroPrefetchResult {
            batch_idx: work.batch_idx,
            subgraph,
        };
        if !deliver(&ends.result_tx, Sequenced { seq, item: result }, &ends.stop) {
            break;
        }
    }

    debug!("Hetero sampler thread stopped");
    Ok(())
}

/// Feature-loader worker loop for the feature pipeline.
///
/// Loading batch N overlaps readahead for batch N+1: the loader takes the
/// next sampled batch early, hints its rows, then loads the current one.
fn worker_loop_feature_loader(
    sample_rx: &Receiver<Sequenced<SampledWork>>,
    result_tx: &Sender<Sequenced<PrefetchResult>>,
    stop: &StopSignal,
    mut feature_store: SyncFeatureStore,
    stats: &PrefetchStats,
) {
    debug!("Feature loader thread started");
    let mut pending: Option<Sequenced<SampledWork>> = None;

    while let Some(Sequenced { seq, item: sampled }) =
        pending.take().or_else(|| recv_work(sample_rx, stop))
    {
        if let Ok(next) = sample_rx.try_recv() {
            feature_store.prefetch_nodes(&next.item.subgraph.nodes);
            trace!(
                batch_idx = next.item.work.batch_idx,
                nodes = next.item.subgraph.nodes.len(),
                "prefetch hints issued for next batch"
            );
            pending = Some(next);
        }

        let t0 = Instant::now();
        let features = feature_store.get_batch(&sampled.subgraph.nodes);
        if let Err(e) = &features {
            warn!(batch_idx = sampled.work.batch_idx, error = %e, "feature load failed");
        }
        stats
            .feature_load_time_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        let result = PrefetchResult {
            batch_idx: sampled.work.batch_idx,
            subgraph: sampled.subgraph,
            features: Some(features),
            feature_dim: Some(feature_store.feature_dim()),
        };
        if !deliver(result_tx, Sequenced { seq, item: result }, stop) {
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
///
/// Every method takes `&self`: one thread may call `shutdown` while others
/// are blocked in `submit` or `next`, and all of them return promptly.
pub struct NeighborLoader {
    pipe: Pipeline<PrefetchResult>,
    /// Feature dimension (if features are being loaded)
    feature_dim: Option<usize>,
    /// Node count of the sampled graph
    num_nodes: usize,
}

impl NeighborLoader {
    /// Create a new prefetching sampler for in-memory graphs (no feature loading).
    ///
    /// # Arguments
    /// * `graph` - Arc to CSR graph (shared with prefetch threads)
    /// * `config` - Sampling configuration. `disjoint` gives each seed its
    ///   own subgraph, with the per-node `batch` vector attached.
    /// * `prefetch_depth` - How many batches to keep ready (default: 2-3)
    /// * `sampler_threads` - Sampler worker count (0 is treated as 1). The
    ///   work channel is MPMC; with `config.seed` set, results are
    ///   reordered to submission order, otherwise they arrive as workers
    ///   finish.
    ///
    /// # Errors
    /// Returns `InvalidInput` if `prefetch_depth` is 0 or `config` asks for
    /// edge weights or timestamps the graph lacks, and an error if a prefetch
    /// thread cannot be spawned.
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
        check_config(&graph, &config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        let sampler_threads = sampler_threads.max(1);
        let (pipe, ends) = Pipeline::new(
            prefetch_depth,
            prefetch_depth.max(sampler_threads),
            config.seed.is_some(),
            graph.num_nodes(),
        );

        let home = crate::internal::numa::current_node();
        for t in 0..sampler_threads {
            let ends = ends.clone();
            let graph = Arc::clone(&graph);
            let config = config.clone();
            pipe.spawn(format!("aethergraph-prefetch-{t}"), move || {
                crate::internal::numa::pin_worker(t, home);
                worker_loop_inmemory(&graph, config, &ends)
            })?;
        }

        debug!(
            prefetch_depth,
            sampler_threads, "NeighborLoader started (in-memory mode, no features)"
        );

        Ok(Self {
            pipe,
            feature_dim: None,
            num_nodes: graph.num_nodes(),
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
    /// Returns an error if `config` asks for edge data the graph lacks, the
    /// feature file cannot be loaded, or a pipeline thread cannot be spawned.
    pub fn with_features(
        graph: Arc<Graph>,
        config: SamplingConfig,
        feature_path: impl AsRef<Path>,
        prefetch_depth: usize,
        sampler_threads: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(prefetch_depth > 0, "prefetch_depth must be >= 1");
        check_config(&graph, &config)?;
        let sampler_threads = sampler_threads.max(1);

        let feature_store = SyncFeatureStore::load(feature_path.as_ref())?;
        let feature_dim = feature_store.feature_dim();

        let (pipe, ends) = Pipeline::new(
            prefetch_depth,
            prefetch_depth,
            config.seed.is_some(),
            graph.num_nodes(),
        );
        // At least one slot per sampler so a burst of simultaneous
        // completions doesn't immediately block the pool.
        let (sample_tx, sample_rx) = bounded::<Sequenced<SampledWork>>(sampler_threads.max(2));

        let home = crate::internal::numa::current_node();
        for t in 0..sampler_threads {
            let work_rx = ends.work_rx.clone();
            let sample_tx = sample_tx.clone();
            let stop = Arc::clone(&ends.stop);
            let stats = Arc::clone(&ends.stats);
            let graph = Arc::clone(&graph);
            let config = config.clone();
            pipe.spawn(format!("aethergraph-sampler-{t}"), move || {
                crate::internal::numa::pin_worker(t, home);
                worker_loop_sampler(&graph, config, &work_rx, &sample_tx, &stop, &stats)
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn sampler thread: {e}"))?;
        }
        // The samplers own the only senders after this point, so the
        // loader's recv disconnects when they all exit.
        drop(sample_tx);

        {
            let PoolEnds {
                result_tx,
                stop,
                stats,
                ..
            } = ends;
            pipe.spawn("aethergraph-feat-loader".into(), move || {
                worker_loop_feature_loader(&sample_rx, &result_tx, &stop, feature_store, &stats);
                Ok(())
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn feature loader thread: {e}"))?;
        }

        debug!(
            prefetch_depth,
            sampler_threads,
            feature_dim,
            "NeighborLoader started (pipeline: samplers + feature loader)"
        );

        Ok(Self {
            pipe,
            feature_dim: Some(feature_dim),
            num_nodes: graph.num_nodes(),
        })
    }

    /// Create a prefetching sampler for NVMe-backed graphs (Linux only).
    ///
    /// The header and offsets array are read and validated here; the edge
    /// body stays on disk, and each batch reads only the neighbor positions
    /// it samples. The io_uring sampling path honors only `fanout`,
    /// `replace`, `cumulative`, and `seed`; edge ids are always tracked
    /// regardless of `track_edge_ids`.
    ///
    /// # Errors
    /// Returns `InvalidInput` if `config` sets a field the io_uring path does
    /// not implement: `weighted`, `temporal_strategy`, `disjoint`,
    /// `deterministic`, `max_degree`, or a non-default `subgraph_type`. Also
    /// returns an error if the graph file is invalid or the prefetch thread
    /// cannot be spawned.
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
        let graph = NvmeGraph::open(graph_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e:#}")))?;
        let num_nodes = graph.num_nodes();

        let (pipe, ends) = Pipeline::new(
            prefetch_depth,
            prefetch_depth,
            config.seed.is_some(),
            num_nodes,
        );
        let feature_dim = feature_store.as_ref().map(SyncFeatureStore::feature_dim);
        pipe.spawn("aethergraph-prefetch-nvme".into(), move || {
            worker_loop_nvme(graph, &config, &ends, feature_store)
        })?;

        debug!(
            prefetch_depth,
            ?feature_dim,
            "NeighborLoader started (NVMe io_uring mode)"
        );

        Ok(Self {
            pipe,
            feature_dim,
            num_nodes,
        })
    }

    /// Submit a batch to be sampled.
    ///
    /// `seeds` carry their range check: build them with
    /// `Seeds::new(ids, loader.num_nodes())`. `batch_idx` is caller
    /// bookkeeping echoed back on the result; any value is accepted, repeats
    /// and gaps included. Ordering (for seeded loaders) follows submission
    /// order, not `batch_idx`.
    ///
    /// Blocks while the work queue is full; returns an error once the loader
    /// is shut down or a worker has faulted, or when `seeds` were checked
    /// against more nodes than this loader samples.
    pub fn submit(&self, batch_idx: usize, seeds: Seeds) -> Result<(), SubmitError> {
        self.pipe.submit(batch_idx, seeds)
    }

    /// Submit all batches for an epoch, indexed from 0.
    pub fn submit_epoch(&self, batches: Vec<Seeds>) -> Result<(), SubmitError> {
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
    /// - `Err(PrefetchError::WorkerExited { .. })` as soon as a worker
    ///   faults without `shutdown()` being requested (panic or internal
    ///   error).
    pub fn next(&self) -> Result<Option<SampledSubgraph>, PrefetchError> {
        Ok(self.pipe.next_timeout(RECV_TIMEOUT)?.map(|r| r.subgraph))
    }

    /// Get the next batch with its `batch_idx` and features (blocking).
    ///
    /// Same return contract as [`NeighborLoader::next_with_features`].
    pub fn next_batch(&self) -> Result<Option<LoadedBatch>, PrefetchError> {
        self.pipe
            .next_timeout(RECV_TIMEOUT)?
            .map(PrefetchResult::into_loaded)
            .transpose()
    }

    /// Try to get next without blocking.
    ///
    /// Returns `Ok(None)` when no batch is ready yet or the loader has been
    /// shut down, and `Err(PrefetchError::WorkerExited { .. })` when a
    /// worker faulted without `shutdown()` being requested.
    pub fn try_next(&self) -> Result<Option<SampledSubgraph>, PrefetchError> {
        Ok(self.pipe.try_next()?.map(|r| r.subgraph))
    }

    /// Get statistics.
    pub fn stats(&self) -> &PrefetchStats {
        &self.pipe.stats
    }

    /// Get prefetch depth.
    pub fn prefetch_depth(&self) -> usize {
        self.pipe.prefetch_depth
    }

    /// Get feature dimension (if features are being loaded).
    pub fn feature_dim(&self) -> Option<usize> {
        self.feature_dim
    }

    /// Node count of the graph this loader samples; seeds must lie below it.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
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
    /// - `Err(PrefetchError::WorkerExited { .. })` when a worker faulted
    ///   without `shutdown()` being requested.
    /// - `Err(PrefetchError::FeatureLoad { .. })` when a feature column is
    ///   attached but loading this batch's features failed.
    pub fn next_with_features(&self) -> Result<Option<SubgraphWithFeatures>, PrefetchError> {
        Ok(self.next_batch()?.map(|b| (b.subgraph, b.features)))
    }

    /// Shut the pipeline down and join its threads.
    ///
    /// Wakes every thread blocked in `submit`, `next`, or a worker queue;
    /// undelivered results are discarded. Afterwards consumer calls return
    /// `Ok(None)` and `submit` returns [`SubmitError::Shutdown`].
    #[tracing::instrument(skip(self))]
    pub fn shutdown(&self) {
        self.pipe.shutdown();
    }
}

/// A validated on-disk CSR: the header and offsets array read and checked
/// once at open, the edge body left on disk for the sampler to read.
#[cfg(target_os = "linux")]
struct NvmeGraph {
    file: File,
    offsets: Vec<u64>,
    edges_start: u64,
}

#[cfg(target_os = "linux")]
impl NvmeGraph {
    fn open(path: &Path) -> anyhow::Result<Self> {
        use anyhow::Context;

        // No O_DIRECT: adjacency reads are variable-sized and unaligned.
        let file = File::open(path).context("failed to open graph file")?;

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

        let offsets_size = num_nodes
            .checked_add(1)
            .and_then(|n| n.checked_mul(std::mem::size_of::<u64>()))
            .ok_or_else(|| anyhow::anyhow!("offsets array size overflow"))?;
        let offsets_start = GRAPH_HEADER_SIZE;
        let edges_start = offsets_start
            .checked_add(offsets_size as u64)
            .ok_or_else(|| anyhow::anyhow!("edges_start overflow"))?;
        let min_edges_bytes = num_edges_u64
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
        }
        anyhow::ensure!(
            offsets[num_nodes] == num_edges_u64,
            "invalid offsets tail: offsets[last]={} != num_edges {}",
            offsets[num_nodes],
            num_edges_u64
        );

        debug!(num_nodes, "Loaded offsets array for NVMe graph");
        Ok(Self {
            file,
            offsets,
            edges_start,
        })
    }

    fn num_nodes(&self) -> usize {
        self.offsets.len() - 1
    }
}

/// Worker loop for NVMe-backed graphs using io_uring.
///
/// Uses SQPOLL for reduced syscalls. Graph adjacency reads are
/// variable-sized, so the ring runs without O_DIRECT/IOPOLL (which would
/// require aligning every variable-length read).
#[cfg(target_os = "linux")]
fn worker_loop_nvme(
    graph: NvmeGraph,
    config: &SamplingConfig,
    ends: &PoolEnds<PrefetchResult>,
    mut feature_store: Option<SyncFeatureStore>,
) -> anyhow::Result<()> {
    use crate::internal::uring::{UringHandle, batch_read};

    debug!("NVMe prefetch worker starting with io_uring");

    // SQPOLL without IOPOLL: the graph file is buffered, and the kernel
    // rejects every polled read on a buffered file.
    let mut handle =
        UringHandle::new_sqpoll_only(crate::internal::uring::DEFAULT_RING_ENTRIES, 1000)?;

    // Register file descriptor for faster access
    if let Err(e) = handle.register_fd(&graph.file) {
        warn!("Failed to register graph fd: {}", e);
    }

    if handle.is_sqpoll() {
        debug!("io_uring: SQPOLL enabled (reduced syscalls)");
    } else {
        debug!("io_uring: standard mode (batched I/O)");
    }

    let fd = graph.file.as_raw_fd();
    let edges_start = graph.edges_start;
    let mut scratch = NvmeScratch::default();
    let mut rng = WyRand::new(config.seed.unwrap_or_else(rand::random));

    while let Some(Sequenced { seq, item: work }) = recv_work(&ends.work_rx, &ends.stop) {
        trace!(
            batch_idx = work.batch_idx,
            seeds = work.seeds.len(),
            "NVMe sampling"
        );

        // Per-submission reseed matches the in-memory workers, so a seeded
        // loader draws the same stream for the same submission.
        if let Some(base) = config.seed {
            rng = WyRand::new(batch_seed(base, seq));
        }
        let t0 = Instant::now();
        let subgraph = sample_from_offsets(
            &graph.offsets,
            &work.seeds,
            config,
            &mut rng,
            &mut scratch,
            |runs, landing| {
                if runs.is_empty() {
                    return Ok(());
                }
                let base = landing.as_mut_ptr().cast::<u8>();
                let reads: Vec<(u64, *mut u8, usize)> = runs
                    .iter()
                    .map(|run| {
                        // SAFETY: the plan sized `landing` to cover every
                        // run's `base..base + len` entries.
                        let ptr = unsafe { base.add(run.base * size_of::<NodeId>()) };
                        (
                            edges_start + run.start * size_of::<NodeId>() as u64,
                            ptr,
                            run.len * size_of::<NodeId>(),
                        )
                    })
                    .collect();
                // SAFETY: each ptr addresses `len` writable bytes inside
                // `landing`, which outlives this call; batch_read reaps every
                // submitted completion before returning.
                unsafe { batch_read(&mut handle, fd, &reads) }
            },
        )?;
        ends.stats.record_sampling(t0);

        let (features, feature_dim) = if let Some(ref mut store) = feature_store {
            let loaded = store.get_batch(&subgraph.nodes);
            if let Err(e) = &loaded {
                warn!(batch_idx = work.batch_idx, error = %e, "NVMe feature load failed");
            }
            (Some(loaded), Some(store.feature_dim()))
        } else {
            (None, None)
        };

        let result = PrefetchResult {
            batch_idx: work.batch_idx,
            subgraph,
            features,
            feature_dim,
        };
        if !deliver(&ends.result_tx, Sequenced { seq, item: result }, &ends.stop) {
            break;
        }
    }

    debug!("NVMe prefetch worker stopped");
    Ok(())
}

/// Picked edge positions this close share one read: a single request
/// covers a page of neighbors, so a node's picks cost one request per
/// cluster and a hub costs its picks, not its whole adjacency list.
#[cfg(any(target_os = "linux", test))]
const EDGE_RUN_GAP: u64 = 1024;

/// Upper bound on one coalesced read, in edges.
#[cfg(any(target_os = "linux", test))]
const EDGE_RUN_MAX: u64 = 1 << 20;

/// One coalesced read of the edge body: `len` entries starting at edge
/// `start`, landing at entry `base` of the landing buffer.
#[cfg(any(target_os = "linux", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EdgeRun {
    start: u64,
    len: usize,
    base: usize,
}

/// Reusable scratch for the NVMe sampler, one per worker and cleared per
/// batch, so steady-state sampling allocates only the returned subgraph.
#[cfg(any(target_os = "linux", test))]
#[derive(Default)]
struct NvmeScratch {
    local: FxHashMap<NodeId, u32>,
    nodes: Vec<NodeId>,
    frontier: Vec<NodeId>,
    next_frontier: Vec<NodeId>,
    positions: Vec<u64>,
    seen: FxHashSet<u64>,
    /// `(frontier index, absolute edge index)` per pick, in draw order.
    picks: Vec<(u32, u64)>,
    order: Vec<u32>,
    slots: Vec<usize>,
    runs: Vec<EdgeRun>,
    landing: Vec<NodeId>,
    /// Cumulative mode: hop at which each edge id was first emitted.
    emitted_at: FxHashMap<u64, usize>,
}

#[cfg(any(target_os = "linux", test))]
impl NvmeScratch {
    /// Registers `node`, returning whether it is new to this batch. Local
    /// ids follow discovery order, so the seeds come first.
    fn register(&mut self, node: NodeId) -> bool {
        let next = self.nodes.len() as u32;
        match self.local.entry(node) {
            std::collections::hash_map::Entry::Occupied(_) => false,
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(next);
                self.nodes.push(node);
                true
            }
        }
    }

    /// Plans the reads covering every pick: sorts the picked edges, merges
    /// neighbors within [`EDGE_RUN_GAP`] into one run, and records where each
    /// pick lands. Sizes `landing` to hold every run.
    fn plan_reads(&mut self) {
        self.order.clear();
        self.order.extend(0..self.picks.len() as u32);
        let picks = &self.picks;
        self.order.sort_unstable_by_key(|&i| picks[i as usize].1);
        self.slots.clear();
        self.slots.resize(self.picks.len(), 0);
        self.runs.clear();

        let mut total = 0usize;
        for &i in &self.order {
            let edge = self.picks[i as usize].1;
            let extends = self.runs.last().is_some_and(|run| {
                edge < run.start + run.len as u64 + EDGE_RUN_GAP && edge - run.start < EDGE_RUN_MAX
            });
            if !extends {
                self.runs.push(EdgeRun {
                    start: edge,
                    len: 0,
                    base: total,
                });
            }
            let run = self.runs.last_mut().expect("a run was just ensured");
            let covered = run.start + run.len as u64;
            if edge >= covered {
                let grow = (edge + 1 - covered) as usize;
                run.len += grow;
                total += grow;
            }
            self.slots[i as usize] = run.base + (edge - run.start) as usize;
        }
        self.landing.clear();
        self.landing.resize(total, 0);
    }
}

/// Draws the positions to sample from a node of degree `degree` into
/// `out`: all of them when `degree <= k`, else `k` uniform draws — with
/// replacement, or distinct via Floyd's algorithm.
#[cfg(any(target_os = "linux", test))]
fn draw_positions(
    rng: &mut WyRand,
    degree: u64,
    k: usize,
    replace: bool,
    out: &mut Vec<u64>,
    seen: &mut FxHashSet<u64>,
) {
    out.clear();
    if degree <= k as u64 {
        out.extend(0..degree);
        return;
    }
    // Multiply-high maps a 64-bit draw onto `[0, n)` without modulo bias
    // worth measuring, and reaches every position of any degree.
    let below =
        |rng: &mut WyRand, n: u64| ((u128::from(rng.next_u64()) * u128::from(n)) >> 64) as u64;
    if replace {
        out.extend((0..k).map(|_| below(rng, degree)));
        return;
    }
    seen.clear();
    for i in (degree - k as u64)..degree {
        let j = below(rng, i + 1);
        let pick = if seen.insert(j) { j } else { i };
        seen.insert(pick);
        out.push(pick);
    }
}

/// k-hop sampling over an on-disk CSR whose offsets are in memory.
///
/// Positions are drawn from the known degrees first, then `fetch` fills
/// the landing buffer for the planned runs — the NVMe worker reads them
/// through io_uring. Seeds are registered first, so they take local ids
/// `0..` in first-occurrence order; each node is expanded once per hop.
/// `num_sampled_nodes` follows PyG: the seed count, then one entry per hop.
///
/// This path honors `fanout`, `replace`, `cumulative`, and `seed` with the
/// in-memory sampler's semantics. Edge ids are always tracked and edges
/// are always returned directional; the constructors reject the remaining
/// `SamplingConfig` fields. `seeds` were checked against this graph's node
/// count at `submit`.
#[cfg(any(target_os = "linux", test))]
fn sample_from_offsets(
    offsets: &[u64],
    seeds: &Seeds,
    config: &SamplingConfig,
    rng: &mut WyRand,
    s: &mut NvmeScratch,
    mut fetch: impl FnMut(&[EdgeRun], &mut [NodeId]) -> anyhow::Result<()>,
) -> anyhow::Result<SampledSubgraph> {
    let num_nodes = offsets.len() - 1;
    s.local.clear();
    s.nodes.clear();
    s.frontier.clear();
    s.emitted_at.clear();
    for &seed in seeds.ids() {
        if s.register(seed) {
            s.frontier.push(seed);
        }
    }

    let num_hops = config.fanout.len();
    let mut edge_src = Vec::new();
    let mut edge_dst = Vec::new();
    let mut edge_ids = Vec::new();
    let mut num_sampled_nodes = Vec::with_capacity(num_hops + 1);
    let mut num_sampled_edges = Vec::with_capacity(num_hops);
    num_sampled_nodes.push(s.nodes.len());

    for (hop, &fanout) in config.fanout.iter().enumerate() {
        s.picks.clear();
        for (fi, &node) in s.frontier.iter().enumerate() {
            let start = offsets[node as usize];
            let degree = offsets[node as usize + 1] - start;
            draw_positions(
                rng,
                degree,
                fanout,
                config.replace,
                &mut s.positions,
                &mut s.seen,
            );
            s.picks
                .extend(s.positions.iter().map(|&p| (fi as u32, start + p)));
        }
        s.plan_reads();
        fetch(&s.runs, &mut s.landing)?;

        let edges_before = edge_src.len();
        s.next_frontier.clear();
        for pick in 0..s.picks.len() {
            let (fi, edge) = s.picks[pick];
            let src = s.frontier[fi as usize];
            let dst = u32::from_le(s.landing[s.slots[pick]]);
            // The edge body is not validated at open; a corrupt entry is
            // dropped rather than minting a node past the graph.
            if dst as usize >= num_nodes {
                continue;
            }
            // Cumulative mode re-expands earlier nodes: an edge an earlier
            // hop emitted is dropped, while repeats within this hop (drawn
            // with replacement) stay.
            if config.cumulative && *s.emitted_at.entry(edge).or_insert(hop) != hop {
                continue;
            }
            edge_src.push(src);
            edge_dst.push(dst);
            edge_ids.push(edge);
            if s.register(dst) {
                s.next_frontier.push(dst);
            }
        }
        num_sampled_nodes.push(s.next_frontier.len());
        num_sampled_edges.push(edge_src.len() - edges_before);

        if config.cumulative {
            s.frontier.extend_from_slice(&s.next_frontier);
        } else {
            std::mem::swap(&mut s.frontier, &mut s.next_frontier);
        }
    }

    let next_capacity = super::planned_capacity(s.nodes.len(), 0);
    let nodes = std::mem::replace(&mut s.nodes, Vec::with_capacity(next_capacity));
    Ok(SampledSubgraph::from_parts(
        nodes,
        edge_src,
        edge_dst,
        edge_ids,
        seeds.ids().to_vec(),
        num_sampled_nodes,
        num_sampled_edges,
    ))
}

/// Prefetching heterogeneous neighbor sampler.
///
/// Same pipeline shape as [`NeighborLoader`]: `sampler_threads` worker
/// threads pull seed batches from a bounded MPMC work channel, sample with
/// their own [`HeteroNeighborSampler`], and deliver results tagged with
/// their `batch_idx`. When `config.seed` is set, `next*` yields in
/// submission order so multi-worker pools stay bit-identical to a single
/// worker. The seed node type is fixed at construction; every submitted
/// batch is rooted at it.
pub struct HeteroNeighborLoader {
    pipe: Pipeline<HeteroPrefetchResult>,
}

impl HeteroNeighborLoader {
    /// Create a prefetching sampler over an in-memory [`HeteroGraph`].
    ///
    /// # Arguments
    /// * `graph` - Arc to the heterogeneous graph (shared with the workers)
    /// * `config` - Sampling configuration (per-edge-type fanout per hop)
    /// * `seed_type` - Node type every submitted seed batch is rooted at
    /// * `prefetch_depth` - How many batches to keep ready (default: 2-3)
    /// * `sampler_threads` - Sampler worker count (0 is treated as 1)
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
        let (pipe, ends) = Pipeline::new(
            prefetch_depth,
            prefetch_depth.max(sampler_threads),
            config.seed.is_some(),
            graph.num_nodes(seed_type),
        );

        let home = crate::internal::numa::current_node();
        for t in 0..sampler_threads {
            let ends = ends.clone();
            let graph = Arc::clone(&graph);
            let config = config.clone();
            pipe.spawn(format!("aethergraph-hetero-sampler-{t}"), move || {
                crate::internal::numa::pin_worker(t, home);
                worker_loop_hetero(&graph, config, seed_type, &ends)
            })?;
        }

        debug!(
            prefetch_depth,
            sampler_threads, "HeteroNeighborLoader started (sampler pool)"
        );

        Ok(Self { pipe })
    }

    /// Submit a batch of seeds of the loader's seed type, checked with
    /// `Seeds::new(ids, loader.seed_nodes())`. `batch_idx` is caller
    /// bookkeeping echoed back on the result; see [`NeighborLoader::submit`].
    pub fn submit(&self, batch_idx: usize, seeds: Seeds) -> Result<(), SubmitError> {
        self.pipe.submit(batch_idx, seeds)
    }

    /// Node count of the seed type; seeds must lie below it.
    pub fn seed_nodes(&self) -> usize {
        self.pipe.seed_nodes
    }

    /// Get next sampled subgraph (blocking).
    ///
    /// Returns:
    /// - `Ok(Some(_))` when a batch is ready.
    /// - `Ok(None)` after `shutdown()` — the clean end-of-stream state.
    /// - `Err(PrefetchError::Timeout { .. })` when no result arrived within
    ///   30s (logged at warn); the workers may just be slow, so the caller
    ///   may call again.
    /// - `Err(PrefetchError::WorkerExited { .. })` as soon as a worker
    ///   faults without `shutdown()` being requested.
    pub fn next(&self) -> Result<Option<HeteroSampledSubgraph>, PrefetchError> {
        Ok(self.next_batch()?.map(|r| r.subgraph))
    }

    /// Get the next batch with the `batch_idx` its submission carried
    /// (blocking). Same return contract as [`HeteroNeighborLoader::next`].
    pub fn next_batch(&self) -> Result<Option<HeteroPrefetchResult>, PrefetchError> {
        self.pipe.next_timeout(RECV_TIMEOUT)
    }

    /// Try to get next without blocking.
    ///
    /// Returns `Ok(None)` when no batch is ready yet or the loader has been
    /// shut down, and `Err(PrefetchError::WorkerExited { .. })` when a
    /// worker faulted without `shutdown()` being requested.
    pub fn try_next(&self) -> Result<Option<HeteroSampledSubgraph>, PrefetchError> {
        Ok(self.pipe.try_next()?.map(|r| r.subgraph))
    }

    /// Get statistics.
    pub fn stats(&self) -> &PrefetchStats {
        &self.pipe.stats
    }

    /// Get prefetch depth.
    pub fn prefetch_depth(&self) -> usize {
        self.pipe.prefetch_depth
    }

    /// Shut the sampler pool down and join its threads. Same contract as
    /// [`NeighborLoader::shutdown`].
    #[tracing::instrument(skip(self))]
    pub fn shutdown(&self) {
        self.pipe.shutdown();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    /// `shutdown()` was requested.
    Shutdown,
    /// A worker faulted; the loader accepts no more work.
    WorkerExited,
    /// The seeds were checked against more nodes than the loader samples.
    SeedsExceedGraph {
        checked_against: usize,
        num_nodes: usize,
    },
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shutdown => write!(f, "prefetcher shut down"),
            Self::WorkerExited => write!(f, "prefetch worker exited"),
            Self::SeedsExceedGraph {
                checked_against,
                num_nodes,
            } => write!(
                f,
                "seeds were checked against {checked_against} nodes, but the loader samples {num_nodes}"
            ),
        }
    }
}

impl std::error::Error for SubmitError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// A loader the tests submit raw IDs to, checked against its bound.
    trait Submits {
        fn seed_bound(&self) -> usize;
        fn submit_seeds(&self, batch_idx: usize, seeds: Seeds) -> Result<(), SubmitError>;
    }

    impl Submits for NeighborLoader {
        fn seed_bound(&self) -> usize {
            self.num_nodes()
        }
        fn submit_seeds(&self, batch_idx: usize, seeds: Seeds) -> Result<(), SubmitError> {
            self.submit(batch_idx, seeds)
        }
    }

    impl Submits for HeteroNeighborLoader {
        fn seed_bound(&self) -> usize {
            self.seed_nodes()
        }
        fn submit_seeds(&self, batch_idx: usize, seeds: Seeds) -> Result<(), SubmitError> {
            self.submit(batch_idx, seeds)
        }
    }

    impl<T> Submits for Pipeline<T> {
        fn seed_bound(&self) -> usize {
            self.seed_nodes
        }
        fn submit_seeds(&self, batch_idx: usize, seeds: Seeds) -> Result<(), SubmitError> {
            self.submit(batch_idx, seeds)
        }
    }

    fn submit(loader: &impl Submits, batch_idx: usize, ids: &[NodeId]) -> Result<(), SubmitError> {
        let seeds = Seeds::new(ids.to_vec(), loader.seed_bound()).expect("test seeds in range");
        loader.submit_seeds(batch_idx, seeds)
    }

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

    /// Node 0 points at every other node, so a small fanout from it draws
    /// from a large pool and repeated draws are distinguishable.
    fn create_star_graph(leaves: u32) -> Arc<Graph> {
        let edges: Vec<(NodeId, NodeId)> = (1..=leaves).map(|leaf| (0, leaf)).collect();
        Arc::new(Graph::from_edges(leaves as usize + 1, &edges, None).unwrap())
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

        submit(&prefetcher, 0, &[0]).unwrap();
        submit(&prefetcher, 1, &[1]).unwrap();
        submit(&prefetcher, 2, &[2]).unwrap();

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
            submit(&prefetcher, i, &[(i % 5) as u32]).unwrap();
        }

        // Seeded: batches come back in submission order across the pool.
        for i in 0..n {
            let b = prefetcher.next_batch().unwrap().unwrap();
            assert_eq!(b.batch_idx, i);
            assert_eq!(b.subgraph.seeds, vec![(i % 5) as u32]);
        }
    }

    #[test]
    fn unseeded_pool_delivers_every_batch_once() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            seed: None,
            ..Default::default()
        };
        let prefetcher = NeighborLoader::new(graph, config, 4, 4).unwrap();
        let n = 24usize;
        for i in 0..n {
            submit(&prefetcher, i, &[(i % 5) as u32]).unwrap();
        }
        let mut seen = vec![false; n];
        for _ in 0..n {
            let b = prefetcher.next_batch().unwrap().unwrap();
            assert!(!seen[b.batch_idx], "batch {} delivered twice", b.batch_idx);
            seen[b.batch_idx] = true;
            assert_eq!(b.subgraph.seeds, vec![(b.batch_idx % 5) as u32]);
        }
        assert!(seen.iter().all(|&s| s), "missing batches: {seen:?}");
    }

    #[test]
    fn seeded_loader_accepts_repeated_and_gapped_batch_indices() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2],
            seed: Some(3),
            ..Default::default()
        };
        let loader = NeighborLoader::new(graph, config, 2, 3).unwrap();
        for idx in [5usize, 5, 9, 1] {
            submit(&loader, idx, &[0]).unwrap();
        }
        let got: Vec<usize> = (0..4)
            .map(|_| loader.next_batch().unwrap().unwrap().batch_idx)
            .collect();
        assert_eq!(got, vec![5, 5, 9, 1]);
    }

    #[test]
    fn seeded_loader_is_reusable_across_epochs() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2],
            seed: Some(11),
            ..Default::default()
        };
        let loader = NeighborLoader::new(graph, config, 4, 2).unwrap();
        for _epoch in 0..3 {
            loader
                .submit_epoch((0..5u32).map(|s| Seeds::new(vec![s], 5).unwrap()).collect())
                .unwrap();
            for idx in 0..5 {
                let b = loader.next_batch().unwrap().unwrap();
                assert_eq!(b.batch_idx, idx);
            }
        }
    }

    #[test]
    fn resubmitted_seeds_draw_fresh_samples() {
        // Reseeding follows submission order, so resubmitting a batch (a
        // new epoch on the same loader) draws a new neighborhood.
        let graph = create_star_graph(1000);
        let config = SamplingConfig {
            fanout: vec![5],
            seed: Some(1),
            ..Default::default()
        };
        let loader = NeighborLoader::new(graph, config, 2, 1).unwrap();
        submit(&loader, 0, &[0]).unwrap();
        submit(&loader, 0, &[0]).unwrap();
        let a = loader.next().unwrap().unwrap().nodes;
        let b = loader.next().unwrap().unwrap().nodes;
        assert_ne!(a, b);
    }

    #[test]
    fn same_seed_reproduces_the_stream_across_pool_sizes() {
        let graph = create_star_graph(500);
        let run = |threads: usize| -> Vec<Vec<NodeId>> {
            let config = SamplingConfig {
                fanout: vec![4, 2],
                seed: Some(99),
                ..Default::default()
            };
            let loader = NeighborLoader::new(Arc::clone(&graph), config, 4, threads).unwrap();
            for i in 0..16u32 {
                submit(&loader, i as usize, &[i % 7]).unwrap();
            }
            (0..16)
                .map(|_| loader.next().unwrap().unwrap().nodes)
                .collect()
        };
        assert_eq!(run(1), run(4));
    }

    #[test]
    fn disjoint_loader_attaches_per_seed_batch() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            fanout: vec![2],
            disjoint: true,
            seed: Some(5),
            ..Default::default()
        };
        let loader = NeighborLoader::new(graph, config, 2, 2).unwrap();
        submit(&loader, 0, &[0, 0, 2]).unwrap();
        let sg = loader.next().unwrap().unwrap();
        let batch = sg
            .batch
            .expect("disjoint subgraph carries its batch vector");
        assert_eq!(batch.len(), sg.nodes.len());
        // Each seed owns its own copy of itself: no dedup across seeds.
        assert!(sg.nodes.iter().filter(|&&n| n == 0).count() >= 2);
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
        let n = 24usize;
        for i in 0..n {
            submit(&loader, i, &[(i % 50) as u32]).unwrap();
        }

        // Seeded: submission order is preserved across the pool.
        for i in 0..n {
            let r = loader.next_batch().unwrap().unwrap();
            assert_eq!(r.batch_idx, i);
            assert_eq!(r.subgraph.seeds, vec![(i % 50) as u32]);
        }
    }

    #[test]
    fn test_prefetch_hit_rate() {
        let graph = create_test_graph();
        let config = SamplingConfig::default();

        let prefetcher = NeighborLoader::new(graph, config, 3, 1).unwrap();

        // Submit all upfront
        for i in 0..10 {
            submit(&prefetcher, i, &[i as u32 % 5]).unwrap();
        }

        // Let worker prefetch
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Consume all
        for _ in 0..10 {
            prefetcher.next().unwrap().unwrap();
        }

        let hit_rate = prefetcher.stats().hit_rate();
        // With prefetch_depth=3 and 10 batches, we expect some hits
        // The exact rate depends on timing, so just verify we got some hits
        assert!(hit_rate >= 0.3, "Expected hit_rate >= 0.3, got {hit_rate}");
        assert_eq!(prefetcher.stats().total.load(Ordering::Relaxed), 10);
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
    fn sync_feature_store_gathers_scattered_rows() {
        let temp_file = NamedTempFile::new().unwrap();
        let dim = 3;
        let features: Vec<f32> = (0..20 * dim).map(|v| v as f32).collect();
        crate::features::save_features(temp_file.path(), features.clone(), 20, dim).unwrap();

        let mut store = SyncFeatureStore::load(temp_file.path()).unwrap();
        let nodes = [19u32, 0, 7, 7, 3];
        store.prefetch_nodes(&nodes);
        let got = store.get_batch(&nodes).unwrap();
        let want: Vec<f32> = nodes
            .iter()
            .flat_map(|&n| features[n as usize * dim..(n as usize + 1) * dim].to_vec())
            .collect();
        assert_eq!(got, want);
        assert!(store.get_batch(&[20]).is_err());
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
        let prefetcher = NeighborLoader::new(graph, config, 1, 1).unwrap();
        for i in 0..4 {
            submit(&prefetcher, i, &[i as u32 % 5]).unwrap();
        }
        std::thread::sleep(Duration::from_millis(100));

        // Must return promptly even though the worker is blocked in send.
        prefetcher.shutdown();

        // After shutdown, consumers see a clean end of stream.
        assert!(matches!(prefetcher.next(), Ok(None)));
        assert!(matches!(prefetcher.try_next(), Ok(None)));
        assert!(matches!(prefetcher.next_with_features(), Ok(None)));
        assert_eq!(submit(&prefetcher, 9, &[0]), Err(SubmitError::Shutdown));
    }

    #[test]
    fn shutdown_wakes_a_blocked_submitter_and_consumer() {
        let graph = create_star_graph(100);
        let config = SamplingConfig {
            fanout: vec![5],
            seed: Some(2),
            ..Default::default()
        };
        // Tiny queues so the submitter blocks: nothing consumes results.
        let loader = NeighborLoader::new(graph, config, 1, 1).unwrap();
        let started = Instant::now();
        std::thread::scope(|scope| {
            let submitter = scope.spawn(|| {
                let mut outcome = Ok(());
                for i in 0..1000 {
                    outcome = submit(&loader, i, &[0]);
                    if outcome.is_err() {
                        break;
                    }
                }
                outcome
            });
            std::thread::sleep(Duration::from_millis(100));
            loader.shutdown();
            assert_eq!(submitter.join().unwrap(), Err(SubmitError::Shutdown));
        });

        // A consumer blocked on an empty stream wakes on shutdown too.
        let graph = create_test_graph();
        let loader = NeighborLoader::new(graph, SamplingConfig::default(), 2, 1).unwrap();
        std::thread::scope(|scope| {
            let consumer = scope.spawn(|| loader.next());
            std::thread::sleep(Duration::from_millis(100));
            loader.shutdown();
            assert!(matches!(consumer.join().unwrap(), Ok(None)));
        });
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn test_next_reports_timeout() {
        let graph = create_test_graph();
        let prefetcher = NeighborLoader::new(graph, SamplingConfig::default(), 2, 1).unwrap();

        // Nothing submitted: the worker is alive but has no results.
        let waited = Duration::from_millis(50);
        match prefetcher.pipe.next_timeout(waited) {
            Err(PrefetchError::Timeout { waited: w }) => assert_eq!(w, waited),
            other => panic!("expected Timeout, got {other:?}"),
        }
    }

    #[test]
    fn worker_fault_surfaces_while_other_workers_live() {
        // One worker panics; its peer stays alive and keeps the result queue
        // open, so only the stop signal can tell the consumer.
        let (pipe, ends) = Pipeline::<PrefetchResult>::new(2, 2, false, 1);
        {
            let ends = ends.clone();
            pipe.spawn("aethergraph-test-survivor".into(), move || {
                while recv_work(&ends.work_rx, &ends.stop).is_some() {}
                Ok(())
            })
            .unwrap();
        }
        pipe.spawn("aethergraph-test-faulty".into(), || panic!("worker died"))
            .unwrap();
        drop(ends);

        let started = Instant::now();
        match pipe.next_timeout(Duration::from_secs(30)) {
            Err(PrefetchError::WorkerExited {
                message: Some(message),
            }) => assert!(message.contains("worker died"), "got: {message}"),
            other => panic!("expected WorkerExited, got {other:?}"),
        }
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(matches!(
            pipe.try_next(),
            Err(PrefetchError::WorkerExited { .. })
        ));
        assert_eq!(submit(&pipe, 0, &[0]), Err(SubmitError::WorkerExited));
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
        submit(&loader, 0, &[4]).unwrap();

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

    #[test]
    fn feature_pipeline_delivers_rows_matching_nodes() {
        let graph = create_test_graph();
        let temp_file = NamedTempFile::new().unwrap();
        let dim = 2;
        let features: Vec<f32> = (0..5 * dim).map(|v| v as f32).collect();
        crate::features::save_features(temp_file.path(), features.clone(), 5, dim).unwrap();

        let config = SamplingConfig {
            fanout: vec![3],
            seed: Some(4),
            ..Default::default()
        };
        let loader = NeighborLoader::with_features(graph, config, temp_file.path(), 2, 2).unwrap();
        for i in 0..6 {
            submit(&loader, i, &[(i % 5) as u32]).unwrap();
        }
        for i in 0..6 {
            let b = loader.next_batch().unwrap().unwrap();
            assert_eq!(b.batch_idx, i);
            let x = b.features.expect("feature column attached");
            let want: Vec<f32> = b
                .subgraph
                .nodes
                .iter()
                .flat_map(|&n| features[n as usize * dim..(n as usize + 1) * dim].to_vec())
                .collect();
            assert_eq!(x, want);
        }
    }

    /// Offsets and edges for the NVMe sampler tests: node `i` points at
    /// `i+1..=i+degree[i]` (mod n).
    fn offsets_graph(degrees: &[u64]) -> (Vec<u64>, Vec<NodeId>) {
        let n = degrees.len() as u64;
        let mut offsets = vec![0u64];
        let mut edges = Vec::new();
        for (i, &d) in degrees.iter().enumerate() {
            for j in 1..=d {
                edges.push(((i as u64 + j) % n) as NodeId);
            }
            offsets.push(offsets.last().unwrap() + d);
        }
        (offsets, edges)
    }

    fn in_memory_fetch(
        edges: &[NodeId],
    ) -> impl FnMut(&[EdgeRun], &mut [NodeId]) -> anyhow::Result<()> {
        move |runs, landing| {
            for run in runs {
                let src = &edges[run.start as usize..run.start as usize + run.len];
                landing[run.base..run.base + run.len].copy_from_slice(src);
            }
            Ok(())
        }
    }

    #[test]
    fn nvme_sampler_registers_seeds_first_and_reads_true_neighbors() {
        let (offsets, edges) = offsets_graph(&[3, 1, 4, 0, 2, 5, 1]);
        let config = SamplingConfig {
            fanout: vec![2, 2],
            cumulative: false,
            max_degree: None,
            seed: Some(9),
            ..Default::default()
        };
        let mut rng = WyRand::new(9);
        let mut scratch = NvmeScratch::default();
        let seeds = Seeds::new(vec![5, 2, 5], 7).unwrap();
        let sg = sample_from_offsets(
            &offsets,
            &seeds,
            &config,
            &mut rng,
            &mut scratch,
            in_memory_fetch(&edges),
        )
        .unwrap();

        // Unique seeds take the first local ids, in first-occurrence order.
        assert_eq!(&sg.nodes[..2], &[5, 2]);
        let unique: FxHashSet<NodeId> = sg.nodes.iter().copied().collect();
        assert_eq!(unique.len(), sg.nodes.len(), "nodes must be deduplicated");
        // Every emitted edge is a real edge with its true CSR id.
        for ((&s, &d), &e) in sg.edge_src.iter().zip(&sg.edge_dst).zip(&sg.edge_ids) {
            let (lo, hi) = (offsets[s as usize], offsets[s as usize + 1]);
            assert!((lo..hi).contains(&e), "edge id {e} outside node {s}'s list");
            assert_eq!(edges[e as usize], d);
        }
        assert_eq!(sg.seed_indices_local().unwrap().into_owned(), vec![0, 1, 0]);
        // PyG layout: the seed count first, then one entry per hop.
        assert_eq!(sg.num_sampled_nodes.len(), 3);
        assert_eq!(sg.num_sampled_nodes[0], 2);
        assert_eq!(sg.num_sampled_nodes.iter().sum::<usize>(), sg.nodes.len());
    }

    #[test]
    fn nvme_cumulative_sampling_emits_each_edge_once() {
        // Every degree is within the fanout, so each hop re-expanding the
        // earlier nodes draws their whole lists again.
        let (offsets, edges) = offsets_graph(&[2, 2, 2, 2, 2]);
        let config = SamplingConfig {
            fanout: vec![4, 4, 4],
            cumulative: true,
            max_degree: None,
            ..Default::default()
        };
        let mut rng = WyRand::new(3);
        let mut scratch = NvmeScratch::default();
        let seeds = Seeds::new(vec![0], 5).unwrap();
        let sg = sample_from_offsets(
            &offsets,
            &seeds,
            &config,
            &mut rng,
            &mut scratch,
            in_memory_fetch(&edges),
        )
        .unwrap();
        let unique: FxHashSet<u64> = sg.edge_ids.iter().copied().collect();
        assert_eq!(unique.len(), sg.edge_ids.len(), "an edge was emitted twice");
        assert_eq!(
            sg.num_sampled_edges.iter().sum::<usize>(),
            sg.edge_ids.len()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn nvme_loader_samples_real_edges_from_disk() {
        let mut edges: Vec<(NodeId, NodeId)> = (1..=300u32).map(|leaf| (0, leaf)).collect();
        edges.extend((1..300u32).map(|n| (n, n + 1)));
        let graph = Graph::from_edges(301, &edges, None).unwrap();
        let file = NamedTempFile::new().unwrap();
        crate::internal::mmap::save_graph(&graph, file.path()).unwrap();

        let config = SamplingConfig {
            fanout: vec![5, 2],
            max_degree: None,
            seed: Some(1),
            ..Default::default()
        };
        let loader = NeighborLoader::new_nvme(file.path(), config, 2).unwrap();
        assert_eq!(loader.num_nodes(), 301);
        for i in 0..4 {
            submit(&loader, i, &[0, 7]).unwrap();
        }
        for i in 0..4 {
            let batch = match loader.next_batch() {
                Ok(Some(batch)) => batch,
                Err(PrefetchError::WorkerExited { message }) => {
                    eprintln!("io_uring unavailable ({message:?}); skipping");
                    return;
                }
                other => panic!("expected a batch, got {other:?}"),
            };
            assert_eq!(batch.batch_idx, i);
            let sg = batch.subgraph;
            assert_eq!(&sg.nodes[..2], &[0, 7]);
            assert!(!sg.edge_src.is_empty());
            let (offsets, body) = (graph.offsets(), graph.edges());
            for ((&s, &d), &e) in sg.edge_src.iter().zip(&sg.edge_dst).zip(&sg.edge_ids) {
                let range = offsets[s as usize]..offsets[s as usize + 1];
                assert!(range.contains(&e), "edge id {e} not in node {s}'s list");
                assert_eq!(body[e as usize], d);
            }
        }
    }

    #[test]
    fn submit_refuses_seeds_checked_against_a_larger_graph() {
        let graph = create_test_graph();
        let loader = NeighborLoader::new(graph, SamplingConfig::default(), 2, 1).unwrap();
        let seeds = Seeds::new(vec![7], 10).unwrap();
        assert_eq!(
            loader.submit(0, seeds),
            Err(SubmitError::SeedsExceedGraph {
                checked_against: 10,
                num_nodes: 5
            })
        );
        // The loader stays usable after refusing a batch.
        submit(&loader, 1, &[4]).unwrap();
        assert_eq!(loader.next_batch().unwrap().unwrap().batch_idx, 1);
    }

    #[test]
    fn construction_rejects_configs_the_graph_cannot_serve() {
        let graph = create_test_graph();
        let config = SamplingConfig {
            weighted: true,
            ..Default::default()
        };
        let err = NeighborLoader::new(graph, config, 2, 1).err().unwrap();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn nvme_read_plan_touches_only_picked_clusters() {
        // A hub of degree 10M with three picks costs three small reads, not
        // a 40 MB list.
        let mut s = NvmeScratch {
            picks: vec![(0, 9_000_000), (0, 12), (0, 9_000_001), (1, 5_000_000)],
            ..NvmeScratch::default()
        };
        s.plan_reads();
        assert_eq!(
            s.runs,
            vec![
                EdgeRun {
                    start: 12,
                    len: 1,
                    base: 0
                },
                EdgeRun {
                    start: 5_000_000,
                    len: 1,
                    base: 1
                },
                EdgeRun {
                    start: 9_000_000,
                    len: 2,
                    base: 2
                },
            ]
        );
        assert_eq!(s.landing.len(), 4);
        assert_eq!(s.slots, vec![2, 0, 3, 1]);
    }

    #[test]
    fn draw_positions_is_distinct_and_in_range() {
        let mut rng = WyRand::new(123);
        let mut out = Vec::new();
        let mut seen = FxHashSet::default();
        for degree in [0u64, 1, 5, 6, 1_000, 5_000_000_000] {
            draw_positions(&mut rng, degree, 5, false, &mut out, &mut seen);
            assert_eq!(out.len() as u64, degree.min(5));
            assert!(out.iter().all(|&p| p < degree.max(1)));
            let distinct: FxHashSet<u64> = out.iter().copied().collect();
            assert_eq!(distinct.len(), out.len());
        }
        draw_positions(&mut rng, 3, 8, true, &mut out, &mut seen);
        assert_eq!(out, vec![0, 1, 2]);
        draw_positions(&mut rng, 100, 8, true, &mut out, &mut seen);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|&p| p < 100));
    }
}
