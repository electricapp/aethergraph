"""Tests for the MetricsSnapshot Prometheus exporter."""

from __future__ import annotations

import numpy as np
import pytest

from aethergraph import Graph, MetricsSnapshot
from aethergraph._core import NeighborSampler, SamplingConfig, SamplingTelemetry


@pytest.fixture
def small_graph_for_sampling() -> Graph:
    """A small graph wired for sampling, separate from conftest's fixture so
    metrics tests stay self-contained."""
    rng = np.random.default_rng(0)
    src = rng.integers(0, 50, size=200, dtype=np.uint32)
    dst = rng.integers(0, 50, size=200, dtype=np.uint32)
    return Graph.from_edges(50, src, dst)


def test_empty_snapshot_is_empty_string() -> None:
    """Snapshot with no subsystems wired emits no Prometheus lines."""
    snap = MetricsSnapshot.collect()
    assert snap.to_prometheus() == ""


def test_sampling_only_snapshot_round_trips(small_graph_for_sampling: Graph) -> None:
    """A live sampler's telemetry surfaces in the Prometheus output."""
    telem = SamplingTelemetry()
    cfg = SamplingConfig(num_neighbors=[5, 3], telemetry=telem)
    sampler = NeighborSampler(small_graph_for_sampling, cfg)
    for _ in range(4):
        sampler.sample([0, 1, 2])

    snap = MetricsSnapshot.collect(sampling=telem)
    text = snap.to_prometheus()
    assert "aethergraph_sampling_operations_total 4" in text
    assert "# TYPE aethergraph_sampling_operations_total counter" in text
    # Feature/prefetch sections should be omitted entirely when not provided.
    assert "aethergraph_feature_" not in text
    assert "aethergraph_prefetch_" not in text


def test_snapshot_repr_lists_active_sections() -> None:
    """`repr` names the sections actually present in the snapshot."""
    telem = SamplingTelemetry()
    snap = MetricsSnapshot.collect(sampling=telem)
    assert "sampling" in repr(snap)
    assert "feature_load" not in repr(snap)


def test_snapshot_is_a_freeze(small_graph_for_sampling: Graph) -> None:
    """`collect` is a snapshot — later sampling does not mutate the export."""
    telem = SamplingTelemetry()
    cfg = SamplingConfig(num_neighbors=[5], telemetry=telem)
    sampler = NeighborSampler(small_graph_for_sampling, cfg)
    sampler.sample([0])

    snap = MetricsSnapshot.collect(sampling=telem)
    text_before = snap.to_prometheus()

    sampler.sample([1])
    sampler.sample([2])
    text_after = snap.to_prometheus()

    assert text_before == text_after, "snapshot must not pick up later samples"
