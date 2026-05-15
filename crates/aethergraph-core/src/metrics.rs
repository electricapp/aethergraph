//! Cross-subsystem metric snapshots for production observability.
//!
//! Each subsystem owns its own lock-free atomic counter struct
//! ([`TelemetrySummary`] from the sampler, [`FeatureLoadTelemetry`] from the
//! feature store, [`PrefetchSummary`] from the loader). This module rolls
//! them up into one [`MetricsSnapshot`] for export to Prometheus, OTEL, or
//! plain JSON logs.
//!
//! # Pattern
//!
//! Subsystems publish via `Arc<…Telemetry>` (already returned by their
//! `telemetry()` methods). Callers gather references and call
//! [`MetricsSnapshot::collect`] periodically — typically on a scrape from
//! Prometheus or once per training-epoch summary log.
//!
//! ```no_run
//! use aethergraph_core::metrics::MetricsSnapshot;
//! use aethergraph_core::SamplingTelemetry;
//! use std::sync::Arc;
//!
//! let sampling = Arc::new(SamplingTelemetry::new());
//! // ... sampler.record_sample(...) ...
//!
//! let snap = MetricsSnapshot::collect(
//!     Some(&sampling),
//!     None,    // no feature store wired up
//!     None,    // no prefetch loader wired up
//! );
//! println!("{}", snap.to_prometheus());
//! ```

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::features::FeatureLoadTelemetry;
use crate::internal::telemetry::{SamplingTelemetry, TelemetrySummary};
use crate::loader::PrefetchStats;

/// Plain-data snapshot of every subsystem's metrics at one point in time.
#[derive(Debug, Clone, Default)]
pub struct MetricsSnapshot {
    /// Sampler stats — populated when the sampler is given a telemetry
    /// collector. `None` means the sampler ran without telemetry attached.
    pub sampling: Option<TelemetrySummary>,
    /// Feature-store stats — populated when the store was loaded with
    /// telemetry enabled.
    pub feature_load: Option<FeatureLoadTelemetry>,
    /// Prefetch-loader stats — always available when a loader is alive.
    pub prefetch: Option<PrefetchSummary>,
}

/// Snapshot of a [`PrefetchStats`] at one instant. `PrefetchStats` itself
/// is mutable atomic state; this is the read-only roll-up.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrefetchSummary {
    pub hits: u64,
    pub misses: u64,
    pub total: u64,
    pub hit_rate: f64,
    pub sample_time_ns: u64,
    pub feature_load_time_ns: u64,
}

impl From<&PrefetchStats> for PrefetchSummary {
    fn from(stats: &PrefetchStats) -> Self {
        let hits = stats.hits.load(Ordering::Relaxed);
        let misses = stats.misses.load(Ordering::Relaxed);
        let total = stats.total.load(Ordering::Relaxed);
        Self {
            hits,
            misses,
            total,
            hit_rate: stats.hit_rate(),
            sample_time_ns: stats.sample_time_ns.load(Ordering::Relaxed),
            feature_load_time_ns: stats.feature_load_time_ns.load(Ordering::Relaxed),
        }
    }
}

impl MetricsSnapshot {
    /// Collect a snapshot from whichever subsystems the caller has access
    /// to. `None` arguments are skipped in the output rather than
    /// reported as zero — distinguishing "wired and idle" from "not
    /// wired".
    pub fn collect(
        sampling: Option<&Arc<SamplingTelemetry>>,
        feature_load: Option<FeatureLoadTelemetry>,
        prefetch: Option<&Arc<PrefetchStats>>,
    ) -> Self {
        Self {
            sampling: sampling.map(|t| t.summary()),
            feature_load,
            prefetch: prefetch.map(|p| PrefetchSummary::from(p.as_ref())),
        }
    }

    /// Serialize as Prometheus text-exposition format (0.0.4). Suitable
    /// for HTTP scrape endpoints. Counter metrics use `_total` suffix per
    /// Prometheus convention; gauges have no suffix.
    pub fn to_prometheus(&self) -> String {
        let mut out = String::new();
        if let Some(s) = self.sampling {
            push_counter(
                &mut out,
                "aethergraph_sampling_operations_total",
                "Total neighborhood-sampling operations.",
                s.total_samples,
            );
            push_counter(
                &mut out,
                "aethergraph_sampling_hub_nodes_capped_total",
                "Sampling operations that hit the max-degree cap.",
                s.hub_nodes_capped,
            );
            push_counter(
                &mut out,
                "aethergraph_sampling_nodes_total",
                "Total nodes touched by sampling.",
                s.total_nodes_sampled,
            );
            push_counter(
                &mut out,
                "aethergraph_sampling_edges_total",
                "Total edges touched by sampling.",
                s.total_edges_sampled,
            );
            push_gauge(
                &mut out,
                "aethergraph_sampling_avg_latency_microseconds",
                "Average sample-call latency.",
                s.avg_latency_us as f64,
            );
        }
        if let Some(f) = &self.feature_load {
            push_counter(
                &mut out,
                "aethergraph_feature_get_calls_total",
                "Single-node feature lookups.",
                f.single_gets,
            );
            push_counter(
                &mut out,
                "aethergraph_feature_batch_calls_total",
                "Batch feature lookups.",
                f.batch_gets,
            );
            push_counter(
                &mut out,
                "aethergraph_feature_nodes_loaded_total",
                "Nodes whose features were read from the store.",
                f.total_nodes_loaded,
            );
            push_counter(
                &mut out,
                "aethergraph_feature_bytes_total",
                "Bytes read from the feature mmap.",
                f.total_bytes_loaded,
            );
        }
        if let Some(p) = self.prefetch {
            push_counter(
                &mut out,
                "aethergraph_prefetch_hits_total",
                "Batches already ready when the consumer called next().",
                p.hits,
            );
            push_counter(
                &mut out,
                "aethergraph_prefetch_misses_total",
                "Times the consumer waited for the prefetcher.",
                p.misses,
            );
            push_counter(
                &mut out,
                "aethergraph_prefetch_batches_total",
                "Total batches processed by the prefetcher.",
                p.total,
            );
            push_gauge(
                &mut out,
                "aethergraph_prefetch_hit_rate",
                "Fraction of next() calls that did not wait.",
                p.hit_rate,
            );
            push_counter(
                &mut out,
                "aethergraph_prefetch_sample_time_nanoseconds_total",
                "Cumulative time the prefetcher spent sampling.",
                p.sample_time_ns,
            );
            push_counter(
                &mut out,
                "aethergraph_prefetch_feature_load_time_nanoseconds_total",
                "Cumulative time the prefetcher spent loading features.",
                p.feature_load_time_ns,
            );
        }
        out
    }
}

fn push_counter(out: &mut String, name: &str, help: &str, value: u64) {
    use std::fmt::Write;
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} counter");
    let _ = writeln!(out, "{name} {value}");
}

fn push_gauge(out: &mut String, name: &str, help: &str, value: f64) {
    use std::fmt::Write;
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} gauge");
    let _ = writeln!(out, "{name} {value}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn empty_snapshot_is_empty_string() {
        let snap = MetricsSnapshot::default();
        assert_eq!(snap.to_prometheus(), "");
    }

    #[test]
    fn sampling_snapshot_emits_prometheus_lines() {
        let t = Arc::new(SamplingTelemetry::new());
        t.record_sample(10, 50, Duration::from_micros(100));
        t.record_sample(20, 80, Duration::from_micros(200));

        let snap = MetricsSnapshot::collect(Some(&t), None, None);
        let exp = snap.to_prometheus();
        assert!(exp.contains("aethergraph_sampling_operations_total 2"));
        assert!(exp.contains("aethergraph_sampling_nodes_total 30"));
        assert!(exp.contains("aethergraph_sampling_avg_latency_microseconds 150"));
    }

    #[test]
    fn prefetch_summary_round_trips() {
        let stats = Arc::new(PrefetchStats::default());
        stats.hits.store(5, Ordering::Relaxed);
        stats.misses.store(1, Ordering::Relaxed);
        stats.total.store(6, Ordering::Relaxed);

        let snap = MetricsSnapshot::collect(None, None, Some(&stats));
        let p = snap.prefetch.unwrap();
        assert_eq!(p.hits, 5);
        assert_eq!(p.misses, 1);
        assert!((p.hit_rate - 5.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn prometheus_help_and_type_are_present() {
        let t = Arc::new(SamplingTelemetry::new());
        t.record_sample(1, 1, Duration::from_micros(1));
        let snap = MetricsSnapshot::collect(Some(&t), None, None);
        let exp = snap.to_prometheus();
        for line in [
            "# HELP aethergraph_sampling_operations_total",
            "# TYPE aethergraph_sampling_operations_total counter",
        ] {
            assert!(exp.contains(line), "missing line: {line}\nfull:\n{exp}");
        }
    }
}
