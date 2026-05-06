//! Lightweight telemetry for production observability.
//!
//! Uses lock-free atomic counters for minimal overhead. No automatic logging.
//! Call `.summary()` to get metrics on-demand.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Telemetry collector for graph sampling operations.
///
/// Thread-safe, lock-free counters with negligible performance impact.
/// Designed for high-throughput production systems (Reddit scale).
#[derive(Debug, Default)]
pub struct SamplingTelemetry {
    /// Total sampling operations
    pub total_samples: AtomicU64,

    /// Number of hub nodes encountered (degree > max_degree)
    pub hub_nodes_capped: AtomicU64,

    /// Total nodes sampled across all operations
    pub total_nodes_sampled: AtomicU64,

    /// Total edges sampled across all operations
    pub total_edges_sampled: AtomicU64,

    /// Cumulative sampling latency (microseconds)
    cumulative_latency_us: AtomicU64,
}

impl SamplingTelemetry {
    /// Creates a new telemetry collector
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a sampling operation
    #[inline]
    pub fn record_sample(&self, nodes: u64, edges: u64, duration: Duration) {
        self.total_samples.fetch_add(1, Ordering::Relaxed);
        self.total_nodes_sampled.fetch_add(nodes, Ordering::Relaxed);
        self.total_edges_sampled.fetch_add(edges, Ordering::Relaxed);
        self.cumulative_latency_us
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
    }

    /// Records a hub node that was degree-capped
    #[inline]
    pub fn record_hub_node(&self) {
        self.hub_nodes_capped.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns a human-readable summary of metrics (call on-demand)
    pub fn summary(&self) -> TelemetrySummary {
        let total = self.total_samples.load(Ordering::Relaxed);
        let cumulative_us = self.cumulative_latency_us.load(Ordering::Relaxed);

        TelemetrySummary {
            total_samples: total,
            hub_nodes_capped: self.hub_nodes_capped.load(Ordering::Relaxed),
            total_nodes_sampled: self.total_nodes_sampled.load(Ordering::Relaxed),
            total_edges_sampled: self.total_edges_sampled.load(Ordering::Relaxed),
            avg_latency_us: cumulative_us.checked_div(total).unwrap_or(0),
        }
    }

    /// Resets all counters (useful for benchmarking)
    pub fn reset(&self) {
        self.total_samples.store(0, Ordering::Relaxed);
        self.hub_nodes_capped.store(0, Ordering::Relaxed);
        self.total_nodes_sampled.store(0, Ordering::Relaxed);
        self.total_edges_sampled.store(0, Ordering::Relaxed);
        self.cumulative_latency_us.store(0, Ordering::Relaxed);
    }
}

/// Snapshot of telemetry metrics
#[derive(Debug, Clone, Copy)]
pub struct TelemetrySummary {
    pub total_samples: u64,
    pub hub_nodes_capped: u64,
    pub total_nodes_sampled: u64,
    pub total_edges_sampled: u64,
    pub avg_latency_us: u64,
}

impl std::fmt::Display for TelemetrySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Sampling Telemetry:\n\
             - Total operations: {}\n\
             - Hub nodes capped: {} ({:.1}%)\n\
             - Avg latency: {}µs\n\
             - Total nodes: {}, edges: {}",
            self.total_samples,
            self.hub_nodes_capped,
            if self.total_samples > 0 {
                (self.hub_nodes_capped as f64 / self.total_samples as f64) * 100.0
            } else {
                0.0
            },
            self.avg_latency_us,
            self.total_nodes_sampled,
            self.total_edges_sampled,
        )
    }
}

/// RAII guard for measuring operation latency
pub struct SamplingTimer {
    start: Instant,
}

impl SamplingTimer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Default for SamplingTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_telemetry_initial_state() {
        let telemetry = SamplingTelemetry::new();
        let summary = telemetry.summary();

        assert_eq!(summary.total_samples, 0);
        assert_eq!(summary.hub_nodes_capped, 0);
        assert_eq!(summary.total_nodes_sampled, 0);
        assert_eq!(summary.total_edges_sampled, 0);
        assert_eq!(summary.avg_latency_us, 0);
    }

    #[test]
    fn test_record_sample() {
        let telemetry = SamplingTelemetry::new();

        telemetry.record_sample(10, 50, Duration::from_micros(100));
        telemetry.record_sample(20, 80, Duration::from_micros(200));
        telemetry.record_sample(30, 70, Duration::from_micros(300));

        let summary = telemetry.summary();
        assert_eq!(summary.total_samples, 3);
        assert_eq!(summary.total_nodes_sampled, 60);
        assert_eq!(summary.total_edges_sampled, 200);
        assert_eq!(summary.avg_latency_us, 200); // (100+200+300)/3 = 200
    }

    #[test]
    fn test_record_hub_node() {
        let telemetry = SamplingTelemetry::new();

        telemetry.record_hub_node();
        telemetry.record_hub_node();
        telemetry.record_hub_node();

        let summary = telemetry.summary();
        assert_eq!(summary.hub_nodes_capped, 3);
    }

    #[test]
    fn test_reset() {
        let telemetry = SamplingTelemetry::new();

        telemetry.record_sample(10, 50, Duration::from_micros(100));
        telemetry.record_sample(20, 80, Duration::from_micros(200));
        telemetry.record_hub_node();

        // Verify non-zero before reset
        let before = telemetry.summary();
        assert!(before.total_samples > 0);
        assert!(before.hub_nodes_capped > 0);

        telemetry.reset();

        let after = telemetry.summary();
        assert_eq!(after.total_samples, 0);
        assert_eq!(after.hub_nodes_capped, 0);
        assert_eq!(after.total_nodes_sampled, 0);
        assert_eq!(after.total_edges_sampled, 0);
        assert_eq!(after.avg_latency_us, 0);
    }

    #[test]
    fn test_telemetry_thread_safe() {
        let telemetry = Arc::new(SamplingTelemetry::new());
        let mut handles = Vec::new();

        for _ in 0..4 {
            let t = Arc::clone(&telemetry);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    t.record_sample(1, 2, Duration::from_micros(10));
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let summary = telemetry.summary();
        assert_eq!(summary.total_samples, 400);
        assert_eq!(summary.total_nodes_sampled, 400);
        assert_eq!(summary.total_edges_sampled, 800);
    }
}
