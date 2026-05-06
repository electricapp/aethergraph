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
//! // Your Kafka consumer, stdin, network socket, whatever:
//! ingest::run(&graph, || {
//!     // Return Some((src, dst)) for each edge, None when done
//!     read_next_edge_from_kafka()
//! });
//! ```

use crate::graph::{ArenaFull, DynamicGraph};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Ingestion statistics. Lock-free, read from any thread.
pub struct IngestStats {
    /// Total edges received (including duplicates).
    pub received: AtomicU64,
    /// New edges inserted (excluding duplicates).
    pub inserted: AtomicU64,
    /// Duplicate edges skipped.
    pub duplicates: AtomicU64,
    /// Errors (arena full, etc.).
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

/// Ingest edges from a closure into a DynamicGraph.
///
/// Calls `next_edge()` repeatedly until it returns `None`. Each edge is
/// inserted into the graph. Duplicates are silently skipped. Returns
/// stats when the source is exhausted.
///
/// Single-threaded on the write side (DynamicGraph is single-writer).
/// Readers can sample concurrently — the graph is lock-free for reads.
pub fn run(graph: &DynamicGraph, mut next_edge: impl FnMut() -> Option<(u32, u32)>) -> IngestStats {
    let stats = IngestStats::new();
    while let Some((src, dst)) = next_edge() {
        stats.received.fetch_add(1, Ordering::Relaxed);
        match graph.insert_edge(src, dst) {
            Ok(true) => {
                stats.inserted.fetch_add(1, Ordering::Relaxed);
            }
            Ok(false) => {
                stats.duplicates.fetch_add(1, Ordering::Relaxed);
            }
            Err(ArenaFull) => {
                stats.errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    stats
}

/// Ingest edges in batches from an iterator of `(src, dst)` slices.
///
/// More efficient than `run()` when edges arrive in batches (e.g., from
/// a Kafka consumer that polls N messages at once). Avoids per-edge
/// closure overhead.
pub fn run_batches(
    graph: &DynamicGraph,
    batches: impl Iterator<Item = Vec<(u32, u32)>>,
) -> IngestStats {
    let stats = IngestStats::new();
    for batch in batches {
        for &(src, dst) in &batch {
            stats.received.fetch_add(1, Ordering::Relaxed);
            match graph.insert_edge(src, dst) {
                Ok(true) => {
                    stats.inserted.fetch_add(1, Ordering::Relaxed);
                }
                Ok(false) => {
                    stats.duplicates.fetch_add(1, Ordering::Relaxed);
                }
                Err(ArenaFull) => {
                    stats.errors.fetch_add(1, Ordering::Relaxed);
                    return stats; // arena full, stop accepting
                }
            }
        }
    }
    stats
}

/// Spawn an ingestion thread that reads edges from `next_edge` and inserts
/// them into the graph until `stop` is set.
///
/// Returns a handle to the thread and shared stats. The thread runs until
/// `stop` is set to `true` or `next_edge` returns `None`.
pub fn spawn(
    graph: Arc<DynamicGraph>,
    stop: Arc<AtomicBool>,
    mut next_edge: impl FnMut() -> Option<(u32, u32)> + Send + 'static,
) -> (std::thread::JoinHandle<()>, Arc<IngestStats>) {
    let stats = Arc::new(IngestStats::new());
    let stats_clone = Arc::clone(&stats);

    let handle = std::thread::Builder::new()
        .name("edge-ingestor".into())
        .spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                match next_edge() {
                    Some((src, dst)) => {
                        stats_clone.received.fetch_add(1, Ordering::Relaxed);
                        match graph.insert_edge(src, dst) {
                            Ok(true) => {
                                stats_clone.inserted.fetch_add(1, Ordering::Relaxed);
                            }
                            Ok(false) => {
                                stats_clone.duplicates.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(ArenaFull) => {
                                stats_clone.errors.fetch_add(1, Ordering::Relaxed);
                                break; // arena full, stop
                            }
                        }
                    }
                    None => {
                        // Source exhausted, sleep briefly and retry
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
            }
        })
        .expect("spawn ingestor thread");

    (handle, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_basic() {
        let graph = DynamicGraph::new(100, 1 << 20);
        let edges = vec![(0, 1), (0, 2), (1, 2), (0, 1)]; // one duplicate
        let mut iter = edges.into_iter();

        let stats = run(&graph, || iter.next());
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

        let stats = run_batches(&graph, batches.into_iter());
        assert_eq!(stats.received(), 4);
        assert_eq!(stats.inserted(), 4);
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
                None // will cause thread to sleep+retry
            }
        });

        // Let it run
        std::thread::sleep(std::time::Duration::from_millis(50));
        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();

        assert!(stats.inserted() > 0);
        assert!(graph.num_edges() > 0);
    }
}
