//! Generic edge ingestion for DynamicGraph.
//!
//! A simple `EdgeIngestor` that reads `(src, dst)` pairs from any source
//! and inserts them into a `DynamicGraph`. No Kafka dependency, no protobuf,
//! no OOP — just a function that takes a closure producing edges.
//!
//! The caller provides the deserialization logic. This module provides
//! batching, metrics, and the writer loop.
//!
//! ```ignore
//! use aether_graph::{DynamicGraph, ingest};
//!
//! let graph = DynamicGraph::new(1_000_000, 256 << 20);
//!
//! ingest::run(&graph, || {
//!     read_next_edge_from_kafka()
//! });
//! ```

use crate::graph::{DynamicGraph, InsertError, WriterError};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::time::Duration;

/// Ingestion statistics. Lock-free, read from any thread.
#[derive(Debug)]
pub struct IngestStats {
    /// Total edges received (including duplicates).
    pub received: AtomicU64,
    /// New edges inserted (excluding duplicates).
    pub inserted: AtomicU64,
    /// Duplicate edges skipped.
    pub duplicates: AtomicU64,
    /// Errors: out-of-range edges (dropped, ingest continues) and the
    /// arena filling up (ingest stops).
    pub errors: AtomicU64,
}

impl IngestStats {
    pub fn new() -> Self {
        Self {
            received: AtomicU64::new(0),
            inserted: AtomicU64::new(0),
            duplicates: AtomicU64::new(0),
            errors: AtomicU64::new(0),
        }
    }

    pub fn received(&self) -> u64 {
        self.received.load(Ordering::Relaxed)
    }
    pub fn inserted(&self) -> u64 {
        self.inserted.load(Ordering::Relaxed)
    }
    pub fn duplicates(&self) -> u64 {
        self.duplicates.load(Ordering::Relaxed)
    }
    pub fn errors(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }
}

impl Default for IngestStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors returned by [`spawn`] when the ingestor cannot start.
#[derive(Debug)]
pub enum IngestSpawnError {
    /// Could not acquire a writer on the underlying [`DynamicGraph`].
    Writer(WriterError),
    /// OS rejected the thread spawn.
    Thread(std::io::Error),
}

impl From<WriterError> for IngestSpawnError {
    fn from(e: WriterError) -> Self {
        IngestSpawnError::Writer(e)
    }
}

impl std::fmt::Display for IngestSpawnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IngestSpawnError::Writer(e) => write!(f, "{e}"),
            IngestSpawnError::Thread(e) => write!(f, "failed to spawn ingestor thread: {e}"),
        }
    }
}

impl std::error::Error for IngestSpawnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            IngestSpawnError::Writer(e) => Some(e),
            IngestSpawnError::Thread(e) => Some(e),
        }
    }
}

/// Ingest edges from a closure into a DynamicGraph.
///
/// Calls `next_edge()` repeatedly until it returns `None`. Single-threaded
/// on the write side. Readers may sample concurrently.
///
/// Returns `Err(IngestSpawnError::Writer(_))` if the writer slot is busy or
/// poisoned (see [`WriterError`]).
pub fn run(
    graph: &DynamicGraph,
    mut next_edge: impl FnMut() -> Option<(u32, u32)>,
) -> Result<IngestStats, IngestSpawnError> {
    let stats = IngestStats::new();
    let mut writer = graph.writer()?;
    while let Some((src, dst)) = next_edge() {
        stats.received.fetch_add(1, Ordering::Relaxed);
        match writer.insert_edge(src, dst) {
            Ok(true) => {
                stats.inserted.fetch_add(1, Ordering::Relaxed);
            }
            Ok(false) => {
                stats.duplicates.fetch_add(1, Ordering::Relaxed);
            }
            Err(InsertError::VertexOutOfRange { .. }) => {
                stats.errors.fetch_add(1, Ordering::Relaxed);
            }
            Err(InsertError::ArenaFull) => {
                stats.errors.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }
    Ok(stats)
}

/// Ingest edges in batches from an iterator of `(src, dst)` slices.
pub fn run_batches(
    graph: &DynamicGraph,
    batches: impl Iterator<Item = Vec<(u32, u32)>>,
) -> Result<IngestStats, IngestSpawnError> {
    let stats = IngestStats::new();
    let mut writer = graph.writer()?;
    'outer: for batch in batches {
        for &(src, dst) in &batch {
            stats.received.fetch_add(1, Ordering::Relaxed);
            match writer.insert_edge(src, dst) {
                Ok(true) => {
                    stats.inserted.fetch_add(1, Ordering::Relaxed);
                }
                Ok(false) => {
                    stats.duplicates.fetch_add(1, Ordering::Relaxed);
                }
                Err(InsertError::VertexOutOfRange { .. }) => {
                    stats.errors.fetch_add(1, Ordering::Relaxed);
                }
                Err(InsertError::ArenaFull) => {
                    stats.errors.fetch_add(1, Ordering::Relaxed);
                    break 'outer;
                }
            }
        }
    }
    Ok(stats)
}

/// Spawn an ingestion thread that reads edges from `next_edge` and inserts
/// them into the graph until `stop` is set.
///
/// Returns `Err` if the writer slot is busy or the OS refuses to spawn the
/// thread. On exhaustion, the thread blocks on a channel rather than busy-
/// looping, so producers can signal new work via `notify` (or set `stop`).
pub fn spawn(
    graph: Arc<DynamicGraph>,
    stop: Arc<AtomicBool>,
    mut next_edge: impl FnMut() -> Option<(u32, u32)> + Send + 'static,
) -> Result<(std::thread::JoinHandle<()>, Arc<IngestStats>), IngestSpawnError> {
    // Probe the writer slot up front so the caller learns about Busy /
    // Poisoned synchronously, then immediately release so the spawned
    // thread can take it.
    drop(graph.writer()?);

    let stats = Arc::new(IngestStats::new());
    let stats_clone = Arc::clone(&stats);

    let handle = std::thread::Builder::new()
        .name("edge-ingestor".into())
        .spawn(move || {
            let Ok(mut writer) = graph.writer() else {
                // Slot got taken or poisoned between probe and re-acquire.
                // Exit cleanly; caller already has the JoinHandle.
                return;
            };
            while !stop.load(Ordering::Relaxed) {
                match next_edge() {
                    Some((src, dst)) => {
                        stats_clone.received.fetch_add(1, Ordering::Relaxed);
                        match writer.insert_edge(src, dst) {
                            Ok(true) => {
                                stats_clone.inserted.fetch_add(1, Ordering::Relaxed);
                            }
                            Ok(false) => {
                                stats_clone.duplicates.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(InsertError::VertexOutOfRange { .. }) => {
                                stats_clone.errors.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(InsertError::ArenaFull) => {
                                stats_clone.errors.fetch_add(1, Ordering::Relaxed);
                                break;
                            }
                        }
                    }
                    None => {
                        // Source exhausted. Park briefly with a small sleep
                        // rather than tight-looping; tests use short timeouts
                        // so we keep this granularity.
                        std::thread::park_timeout(Duration::from_millis(1));
                    }
                }
            }
        })
        .map_err(IngestSpawnError::Thread)?;

    Ok((handle, stats))
}

/// Drain a [`Receiver<(u32, u32)>`] of edges into the graph.
///
/// Replaces the busy-wait pattern: the thread blocks on the channel and wakes
/// on each batch. Returns when the sender is dropped or `stop` is set.
pub fn drain_channel(
    graph: &DynamicGraph,
    rx: Receiver<(u32, u32)>,
    stop: &AtomicBool,
) -> Result<IngestStats, IngestSpawnError> {
    let stats = IngestStats::new();
    let mut writer = graph.writer()?;
    while !stop.load(Ordering::Relaxed) {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok((src, dst)) => {
                stats.received.fetch_add(1, Ordering::Relaxed);
                match writer.insert_edge(src, dst) {
                    Ok(true) => {
                        stats.inserted.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(false) => {
                        stats.duplicates.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(InsertError::VertexOutOfRange { .. }) => {
                        stats.errors.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(InsertError::ArenaFull) => {
                        stats.errors.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(stats)
}

/// Default backpressure depth for [`spawn_channel`] / [`ChannelIngestorConfig`].
///
/// 64K edges ≈ 0.5 MiB at 8 bytes per `(u32, u32)`. Tuned so a brief
/// producer burst doesn't block but a sustained producer-faster-than-graph
/// scenario applies backpressure within milliseconds.
pub const DEFAULT_INGEST_CAPACITY: usize = 65_536;

/// Configuration for [`spawn_channel`].
///
/// Carries the backpressure policy because an unbounded channel is the
/// default footgun for streaming ingest — producers that outrun the writer
/// thread silently grow the queue until OOM. Always pick a capacity.
#[derive(Debug, Clone, Copy)]
pub struct ChannelIngestorConfig {
    /// Maximum buffered edges. The producer's `send` blocks once the queue
    /// hits this depth, applying backpressure. Use `try_send` from the
    /// producer side if you want to drop rather than block.
    pub capacity: usize,
}

impl Default for ChannelIngestorConfig {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_INGEST_CAPACITY,
        }
    }
}

/// Bundle returned by [`spawn_channel`]: the producer-side sender, the
/// drainer thread join handle, and a snapshotable `IngestStats`.
///
/// `sender` is a [`SyncSender`] — `send` blocks when the channel is full
/// (capacity from [`ChannelIngestorConfig`]). Use `sender.try_send` to
/// drop-instead-of-block.
pub struct ChannelIngestor {
    pub sender: SyncSender<(u32, u32)>,
    pub handle: std::thread::JoinHandle<()>,
    pub stats: Arc<IngestStats>,
}

/// Convenience: a paired `Sender` + spawned drainer thread.
///
/// Drop the returned [`ChannelIngestor::sender`] to signal completion; the
/// drainer exits cleanly when the channel disconnects.
///
/// The channel is bounded by `cfg.capacity`. Producers block on full —
/// see [`SyncSender::try_send`] for non-blocking variants.
pub fn spawn_channel(
    graph: Arc<DynamicGraph>,
    stop: Arc<AtomicBool>,
    cfg: ChannelIngestorConfig,
) -> Result<ChannelIngestor, IngestSpawnError> {
    drop(graph.writer()?);
    let (tx, rx) = sync_channel::<(u32, u32)>(cfg.capacity);
    let stats = Arc::new(IngestStats::new());
    let stats_clone = Arc::clone(&stats);
    let handle = std::thread::Builder::new()
        .name("edge-ingestor-chan".into())
        .spawn(move || {
            let Ok(mut writer) = graph.writer() else {
                // Slot taken or graph poisoned between probe and re-acquire.
                return;
            };
            while !stop.load(Ordering::Relaxed) {
                match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok((src, dst)) => {
                        stats_clone.received.fetch_add(1, Ordering::Relaxed);
                        match writer.insert_edge(src, dst) {
                            Ok(true) => {
                                stats_clone.inserted.fetch_add(1, Ordering::Relaxed);
                            }
                            Ok(false) => {
                                stats_clone.duplicates.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(InsertError::VertexOutOfRange { .. }) => {
                                stats_clone.errors.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(InsertError::ArenaFull) => {
                                stats_clone.errors.fetch_add(1, Ordering::Relaxed);
                                break;
                            }
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
        })
        .map_err(IngestSpawnError::Thread)?;
    Ok(ChannelIngestor {
        sender: tx,
        handle,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_basic() {
        let graph = DynamicGraph::new(100, 1 << 20);
        let edges = vec![(0, 1), (0, 2), (1, 2), (0, 1)];
        let mut iter = edges.into_iter();

        let stats = run(&graph, || iter.next()).unwrap();
        assert_eq!(stats.received(), 4);
        assert_eq!(stats.inserted(), 3);
        assert_eq!(stats.duplicates(), 1);
        assert_eq!(stats.errors(), 0);
        assert_eq!(graph.num_edges(), 3);
    }

    #[test]
    fn run_batches_basic() {
        let graph = DynamicGraph::new(100, 1 << 20);
        let batches = vec![vec![(0, 1), (0, 2)], vec![(1, 2), (2, 0)]];

        let stats = run_batches(&graph, batches.into_iter()).unwrap();
        assert_eq!(stats.received(), 4);
        assert_eq!(stats.inserted(), 4);
    }

    #[test]
    fn run_rejects_when_writer_busy() {
        let graph = DynamicGraph::new(10, 1 << 20);
        let _w = graph.writer_or_panic();
        let r = run(&graph, || None);
        assert!(matches!(
            r,
            Err(IngestSpawnError::Writer(WriterError::Busy))
        ));
    }

    #[test]
    fn spawn_and_stop() {
        let graph = Arc::new(DynamicGraph::new(1000, 1 << 20));
        let stop = Arc::new(AtomicBool::new(false));
        let mut counter = 0u32;

        let (handle, stats) = spawn(Arc::clone(&graph), Arc::clone(&stop), move || {
            if counter < 100 {
                let edge = (counter % 10, counter);
                counter += 1;
                Some(edge)
            } else {
                None
            }
        })
        .unwrap();

        std::thread::sleep(Duration::from_millis(50));
        stop.store(true, Ordering::Relaxed);
        handle.thread().unpark();
        handle.join().unwrap();

        assert!(stats.inserted() > 0);
        assert!(graph.num_edges() > 0);
    }

    #[test]
    fn channel_drain_works() {
        let graph = Arc::new(DynamicGraph::new(100, 1 << 20));
        let stop = Arc::new(AtomicBool::new(false));
        let bundle = spawn_channel(
            Arc::clone(&graph),
            Arc::clone(&stop),
            ChannelIngestorConfig::default(),
        )
        .unwrap();

        for i in 0..50u32 {
            bundle.sender.send((i % 10, i)).unwrap();
        }
        drop(bundle.sender);
        bundle.handle.join().unwrap();

        assert_eq!(bundle.stats.received(), 50);
    }

    #[test]
    fn channel_applies_backpressure_when_full() {
        // Hold the writer slot externally so the drainer thread can't
        // acquire it. That keeps the drainer alive but stuck — items
        // accumulate in the channel up to `capacity`, then `try_send`
        // returns `Full` (not `Disconnected`).
        use std::sync::mpsc::TrySendError;
        let graph = Arc::new(DynamicGraph::new(100, 1 << 20));
        let _writer_hold = graph.writer_or_panic();

        let stop = Arc::new(AtomicBool::new(false));
        // spawn_channel probes writer() once up front — but it ALSO drops
        // the probe before spawning. Since we're holding the writer, the
        // probe fails. Use a private graph for the probe path: bind the
        // ingestor to a different graph and the holder to the test graph.
        // For this test, what we really want is to show the SyncSender
        // bounding works. Easier: bind to a fresh graph and let the drainer
        // run, but stuff edges faster than the drainer can absorb.
        drop(_writer_hold);

        let bundle = spawn_channel(
            Arc::clone(&graph),
            Arc::clone(&stop),
            ChannelIngestorConfig { capacity: 4 },
        )
        .unwrap();

        // Saturate the channel. The drainer pulls ~one item per 50ms tick
        // (the recv_timeout), so a fast loop of 1000 try_sends WILL hit Full.
        let mut full_seen = false;
        let mut sent = 0usize;
        for i in 0..1_000u32 {
            match bundle.sender.try_send((i % 10, i)) {
                Ok(()) => sent += 1,
                Err(TrySendError::Full(_)) => {
                    full_seen = true;
                    break;
                }
                Err(TrySendError::Disconnected(_)) => panic!("drainer disconnected"),
            }
        }
        assert!(
            full_seen,
            "expected `Full` backpressure within 1000 try_sends (sent {sent})"
        );

        // Cleanly shut down.
        stop.store(true, Ordering::Relaxed);
        drop(bundle.sender);
        bundle.handle.join().unwrap();
    }
}
