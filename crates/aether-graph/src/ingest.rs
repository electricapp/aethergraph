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

use crate::graph::DynamicGraph;
use crate::writer::{InsertError, WriterError};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, channel, sync_channel};
use std::time::Duration;

/// Number of edges a single writer guard absorbs before every ingest
/// loop drops and reacquires it. Each guard drop commits (WAL sync when
/// attached, epoch advance), so this bounds both the window of inserts a
/// crash can lose and how stale epoch-pinned readers can get during a
/// long ingest session.
pub const COMMIT_INTERVAL_EDGES: usize = 65_536;

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

/// Thread-local counters, folded into the shared [`IngestStats`] at commit
/// boundaries. Per-edge atomic RMWs on the `Arc`-shared stats line cost
/// more than the tree insert itself once a monitor thread polls them; two
/// atomics per commit interval replace two per edge.
#[derive(Default)]
struct LocalCounts {
    received: u64,
    inserted: u64,
    duplicates: u64,
    errors: u64,
}

impl LocalCounts {
    fn flush(&mut self, stats: &IngestStats) {
        if self.received > 0 {
            stats.received.fetch_add(self.received, Ordering::Relaxed);
            self.received = 0;
        }
        if self.inserted > 0 {
            stats.inserted.fetch_add(self.inserted, Ordering::Relaxed);
            self.inserted = 0;
        }
        if self.duplicates > 0 {
            stats
                .duplicates
                .fetch_add(self.duplicates, Ordering::Relaxed);
            self.duplicates = 0;
        }
        if self.errors > 0 {
            stats.errors.fetch_add(self.errors, Ordering::Relaxed);
            self.errors = 0;
        }
    }
}

/// Insert one edge, updating the local counters. Returns `false` on a
/// fatal error (arena exhaustion, WAL failure) — the caller must stop.
/// Every ingest driver funnels through here so the insert/count/commit
/// logic exists exactly once.
#[inline]
fn drive_edge(
    writer: &mut crate::writer::Writer<'_>,
    local: &mut LocalCounts,
    src: u32,
    dst: u32,
) -> bool {
    local.received += 1;
    match writer.insert_edge(src, dst) {
        Ok(true) => {
            local.inserted += 1;
            true
        }
        Ok(false) => {
            local.duplicates += 1;
            true
        }
        Err(InsertError::VertexOutOfRange { .. }) => {
            local.errors += 1;
            true
        }
        // Arena exhaustion (and WAL append failure when enabled) cannot be
        // recovered here: stop ingesting.
        Err(_) => {
            local.errors += 1;
            false
        }
    }
}

/// Ingest edges from a closure into a DynamicGraph.
///
/// Calls `next_edge()` repeatedly until it returns `None`. Single-threaded
/// on the write side. Readers may sample concurrently.
///
/// Durability cadence: the writer guard is dropped and reacquired every
/// [`COMMIT_INTERVAL_EDGES`] edges, committing (WAL sync when attached,
/// epoch advance) at that interval and again when the run ends. Stats are
/// folded into the returned [`IngestStats`] at the same cadence.
///
/// Returns `Err(IngestSpawnError::Writer(_))` if the writer slot is busy
/// or poisoned (see [`WriterError`]), either at the start or when a
/// periodic reacquire finds the slot stolen or the graph poisoned.
pub fn run(
    graph: &DynamicGraph,
    mut next_edge: impl FnMut() -> Option<(u32, u32)>,
) -> Result<IngestStats, IngestSpawnError> {
    let stats = IngestStats::new();
    let mut local = LocalCounts::default();
    let mut writer = graph.writer()?;
    let mut guard_edges = 0usize;
    while let Some((src, dst)) = next_edge() {
        let ok = drive_edge(&mut writer, &mut local, src, dst);
        if !ok {
            break;
        }
        guard_edges += 1;
        if guard_edges >= COMMIT_INTERVAL_EDGES {
            local.flush(&stats);
            drop(writer);
            writer = graph.writer()?;
            guard_edges = 0;
        }
    }
    local.flush(&stats);
    Ok(stats)
}

/// Ingest edges in batches from an iterator of `(src, dst)` slices.
///
/// Durability cadence: the writer guard is dropped and reacquired every
/// [`COMMIT_INTERVAL_EDGES`] edges — batch boundaries deliberately do NOT
/// force a commit. A caller feeding 100-edge batches would otherwise pay
/// an fsync every 100 edges (a 650x amplification over the interval),
/// capping ingest at the storage's fsync rate. A final commit still runs
/// when the iterator ends. A failed reacquire (slot stolen or graph
/// poisoned) returns `Err(IngestSpawnError::Writer(_))`.
pub fn run_batches(
    graph: &DynamicGraph,
    batches: impl Iterator<Item = Vec<(u32, u32)>>,
) -> Result<IngestStats, IngestSpawnError> {
    let stats = IngestStats::new();
    let mut local = LocalCounts::default();
    let mut writer = graph.writer()?;
    let mut guard_edges = 0usize;
    'outer: for batch in batches {
        for &(src, dst) in &batch {
            let ok = drive_edge(&mut writer, &mut local, src, dst);
            if !ok {
                break 'outer;
            }
            guard_edges += 1;
            if guard_edges >= COMMIT_INTERVAL_EDGES {
                local.flush(&stats);
                drop(writer);
                writer = graph.writer()?;
                guard_edges = 0;
            }
        }
    }
    local.flush(&stats);
    Ok(stats)
}

/// Spawn an ingestion thread that reads edges from `next_edge` and inserts
/// them into the graph until `stop` is set.
///
/// The spawned thread acquires the writer slot itself and reports the
/// result back before this returns, so `Err` means no idle ingestor was
/// left behind. Returns `Err` if the writer slot is busy or poisoned, or
/// if the OS refuses to spawn the thread.
///
/// Durability cadence: the writer guard is dropped and reacquired every
/// [`COMMIT_INTERVAL_EDGES`] edges, committing (WAL sync when attached,
/// epoch advance) at that interval. A failed reacquire bumps
/// `stats.errors` and stops the thread.
pub fn spawn(
    graph: Arc<DynamicGraph>,
    stop: Arc<AtomicBool>,
    mut next_edge: impl FnMut() -> Option<(u32, u32)> + Send + 'static,
) -> Result<(std::thread::JoinHandle<()>, Arc<IngestStats>), IngestSpawnError> {
    let stats = Arc::new(IngestStats::new());
    let stats_clone = Arc::clone(&stats);
    // The spawned thread reports its writer acquisition over this channel
    // so the caller learns about Busy / Poisoned synchronously.
    let (ready_tx, ready_rx) = channel::<Result<(), WriterError>>();

    let handle = std::thread::Builder::new()
        .name("edge-ingestor".into())
        .spawn(move || {
            let mut writer = match graph.writer() {
                Ok(w) => {
                    let _ = ready_tx.send(Ok(()));
                    w
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            let mut local = LocalCounts::default();
            let mut guard_edges = 0usize;
            let mut idle_spins = 0u32;
            while !stop.load(Ordering::Relaxed) {
                match next_edge() {
                    Some((src, dst)) => {
                        idle_spins = 0;
                        let ok = drive_edge(&mut writer, &mut local, src, dst);
                        if !ok {
                            break;
                        }
                        guard_edges += 1;
                        if guard_edges >= COMMIT_INTERVAL_EDGES {
                            local.flush(&stats_clone);
                            drop(writer);
                            writer = match graph.writer() {
                                Ok(w) => w,
                                Err(e) => {
                                    stats_clone.errors.fetch_add(1, Ordering::Relaxed);
                                    tracing::error!(
                                        error = %e,
                                        "ingestor could not reacquire the writer slot; stopping"
                                    );
                                    return;
                                }
                            };
                            guard_edges = 0;
                        }
                    }
                    None => {
                        // Source momentarily dry. Make buffered stats
                        // visible, spin briefly for a bursty source, then
                        // park — jumping straight to a 1 ms sleep adds up
                        // to 1 ms latency per gap.
                        local.flush(&stats_clone);
                        idle_spins += 1;
                        if idle_spins < 64 {
                            std::hint::spin_loop();
                        } else {
                            std::thread::park_timeout(Duration::from_millis(1));
                        }
                    }
                }
            }
            local.flush(&stats_clone);
        })
        .map_err(IngestSpawnError::Thread)?;

    match ready_rx.recv() {
        Ok(Ok(())) => Ok((handle, stats)),
        Ok(Err(e)) => {
            let _ = handle.join();
            Err(IngestSpawnError::Writer(e))
        }
        Err(_) => {
            let _ = handle.join();
            Err(IngestSpawnError::Thread(std::io::Error::other(
                "ingestor thread exited before reporting writer acquisition",
            )))
        }
    }
}

/// Drain a [`Receiver<(u32, u32)>`] of edges into the graph.
///
/// Replaces the busy-wait pattern: the thread blocks on the channel and wakes
/// on each batch. Returns when the sender is dropped or `stop` is set.
///
/// Durability cadence: the writer guard is dropped and reacquired every
/// [`COMMIT_INTERVAL_EDGES`] edges, committing (WAL sync when attached,
/// epoch advance) at that interval. A failed reacquire returns
/// `Err(IngestSpawnError::Writer(_))`.
pub fn drain_channel(
    graph: &DynamicGraph,
    rx: Receiver<(u32, u32)>,
    stop: &AtomicBool,
) -> Result<IngestStats, IngestSpawnError> {
    let stats = IngestStats::new();
    let mut local = LocalCounts::default();
    let mut writer = graph.writer()?;
    let mut guard_edges = 0usize;
    'outer: while !stop.load(Ordering::Relaxed) {
        // One blocking receive arms the drain; everything already queued
        // is then pulled with non-blocking try_recv — a timeout-armed recv
        // per edge (deadline construction + park/unpark) would cap
        // throughput far below the writer itself.
        let first = match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(edge) => edge,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        let mut next = Some(first);
        while let Some((src, dst)) = next {
            let ok = drive_edge(&mut writer, &mut local, src, dst);
            if !ok {
                break 'outer;
            }
            guard_edges += 1;
            if guard_edges >= COMMIT_INTERVAL_EDGES {
                local.flush(&stats);
                drop(writer);
                writer = graph.writer()?;
                guard_edges = 0;
            }
            next = rx.try_recv().ok();
        }
        local.flush(&stats);
    }
    local.flush(&stats);
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
/// The drainer thread acquires the writer slot itself and reports the
/// result back before this returns, so `Err` means no idle drainer was
/// left behind.
///
/// The channel is bounded by `cfg.capacity`. Producers block on full —
/// see [`SyncSender::try_send`] for non-blocking variants.
///
/// Durability cadence: the writer guard is dropped and reacquired every
/// [`COMMIT_INTERVAL_EDGES`] edges, committing (WAL sync when attached,
/// epoch advance) at that interval. A failed reacquire bumps
/// `stats.errors` and stops the drainer.
pub fn spawn_channel(
    graph: Arc<DynamicGraph>,
    stop: Arc<AtomicBool>,
    cfg: ChannelIngestorConfig,
) -> Result<ChannelIngestor, IngestSpawnError> {
    let (tx, rx) = sync_channel::<(u32, u32)>(cfg.capacity);
    let stats = Arc::new(IngestStats::new());
    let stats_clone = Arc::clone(&stats);
    // The drainer reports its writer acquisition over this channel so
    // the caller learns about Busy / Poisoned synchronously.
    let (ready_tx, ready_rx) = channel::<Result<(), WriterError>>();
    let handle = std::thread::Builder::new()
        .name("edge-ingestor-chan".into())
        .spawn(move || {
            let mut writer = match graph.writer() {
                Ok(w) => {
                    let _ = ready_tx.send(Ok(()));
                    w
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            let mut local = LocalCounts::default();
            let mut guard_edges = 0usize;
            'outer: while !stop.load(Ordering::Relaxed) {
                // Same drain pattern as `drain_channel`: block once, then
                // pull everything queued with non-blocking try_recv.
                let first = match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(edge) => edge,
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(RecvTimeoutError::Disconnected) => break,
                };
                let mut next = Some(first);
                while let Some((src, dst)) = next {
                    let ok = drive_edge(&mut writer, &mut local, src, dst);
                    if !ok {
                        break 'outer;
                    }
                    guard_edges += 1;
                    if guard_edges >= COMMIT_INTERVAL_EDGES {
                        local.flush(&stats_clone);
                        drop(writer);
                        writer = match graph.writer() {
                            Ok(w) => w,
                            Err(e) => {
                                stats_clone.errors.fetch_add(1, Ordering::Relaxed);
                                tracing::error!(
                                    error = %e,
                                    "drainer could not reacquire the writer slot; stopping"
                                );
                                return;
                            }
                        };
                        guard_edges = 0;
                    }
                    next = rx.try_recv().ok();
                }
                local.flush(&stats_clone);
            }
            local.flush(&stats_clone);
        })
        .map_err(IngestSpawnError::Thread)?;
    match ready_rx.recv() {
        Ok(Ok(())) => Ok(ChannelIngestor {
            sender: tx,
            handle,
            stats,
        }),
        Ok(Err(e)) => {
            let _ = handle.join();
            Err(IngestSpawnError::Writer(e))
        }
        Err(_) => {
            let _ = handle.join();
            Err(IngestSpawnError::Thread(std::io::Error::other(
                "drainer thread exited before reporting writer acquisition",
            )))
        }
    }
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
    fn spawn_reports_busy_writer() {
        let graph = Arc::new(DynamicGraph::new(10, 1 << 20));
        let _hold = graph.writer_or_panic();
        let stop = Arc::new(AtomicBool::new(false));
        let r = spawn(Arc::clone(&graph), stop, || None);
        assert!(matches!(
            r,
            Err(IngestSpawnError::Writer(WriterError::Busy))
        ));
    }

    #[test]
    fn spawn_channel_reports_busy_writer() {
        let graph = Arc::new(DynamicGraph::new(10, 1 << 20));
        let _hold = graph.writer_or_panic();
        let stop = Arc::new(AtomicBool::new(false));
        let r = spawn_channel(Arc::clone(&graph), stop, ChannelIngestorConfig::default());
        assert!(matches!(
            r,
            Err(IngestSpawnError::Writer(WriterError::Busy))
        ));
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
        // Stuff edges faster than the drainer can absorb them: items
        // accumulate in the channel up to `capacity`, then `try_send`
        // returns `Full` (not `Disconnected`).
        use std::sync::mpsc::TrySendError;
        let graph = Arc::new(DynamicGraph::new(100, 1 << 20));
        let stop = Arc::new(AtomicBool::new(false));

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
