//! Python bindings for cross-subsystem metric snapshots.
//!
//! `MetricsSnapshot` rolls up the three independent telemetry channels —
//! sampler, feature store, prefetch loader — into one object that can be
//! serialized as Prometheus text-exposition format. Consumers feed the
//! result to whatever HTTP scrape handler they're running (`/metrics`).

use aethergraph_core::metrics::{MetricsSnapshot, PrefetchSummary};
use pyo3::prelude::*;

use crate::feature_telemetry::PyFeatureLoadTelemetry;
use crate::prefetch::PyPrefetchStats;
use crate::sampler::PySamplingTelemetry;

/// Python wrapper for `aethergraph_core::metrics::MetricsSnapshot`.
///
/// Built from optional Python wrappers around the three telemetry channels.
/// The snapshot is frozen at construction time — call `collect` again to
/// re-scrape live counters.
#[pyclass(name = "MetricsSnapshot", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMetricsSnapshot {
    inner: MetricsSnapshot,
}

#[pymethods]
impl PyMetricsSnapshot {
    /// Collect a fresh snapshot. Each argument is optional — `None` means
    /// "subsystem not wired up" and is left out of the export entirely
    /// (distinguishing it from "wired and idle").
    ///
    /// Args:
    ///     sampling: Live sampler telemetry collector.
    ///     feature_load: A `FeatureLoadTelemetry` from `FeatureStore.telemetry()`.
    ///     prefetch: A `PrefetchStats` snapshot from `NeighborLoader.stats()`.
    #[staticmethod]
    #[pyo3(signature = (sampling=None, feature_load=None, prefetch=None))]
    fn collect(
        sampling: Option<PySamplingTelemetry>,
        feature_load: Option<PyFeatureLoadTelemetry>,
        prefetch: Option<PyRef<'_, PyPrefetchStats>>,
    ) -> Self {
        let prefetch_summary = prefetch.map(|p| PrefetchSummary {
            hits: p.hits,
            misses: p.misses,
            total: p.total,
            hit_rate: if p.total == 0 {
                1.0
            } else {
                p.hits as f64 / p.total as f64
            },
            sample_time_ns: p.sample_time_ns,
            feature_load_time_ns: p.feature_load_time_ns,
        });

        Self {
            inner: MetricsSnapshot {
                sampling: sampling.map(|s| s.inner.summary()),
                feature_load: feature_load.map(|f| f.inner),
                prefetch: prefetch_summary,
            },
        }
    }

    /// Serialize as Prometheus text-exposition format (0.0.4). Suitable to
    /// return verbatim from an HTTP `/metrics` handler.
    fn to_prometheus(&self) -> String {
        self.inner.to_prometheus()
    }

    fn __repr__(&self) -> String {
        let sections: Vec<&'static str> = [
            self.inner.sampling.as_ref().map(|_| "sampling"),
            self.inner.feature_load.as_ref().map(|_| "feature_load"),
            self.inner.prefetch.as_ref().map(|_| "prefetch"),
        ]
        .into_iter()
        .flatten()
        .collect();
        format!("MetricsSnapshot(sections={sections:?})")
    }
}
