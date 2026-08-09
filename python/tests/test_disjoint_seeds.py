"""
Tests for disjoint-mode seed indices.

In disjoint mode, each seed produces its own isolated subgraph and the
sampler emits a `batch` vector. The sampler records each seed's local
index at build time (a seed is always the first node of its block), and
`seed_indices` exposes them. These tests pin that contract: one index per
seed, each pointing at the first node of its batch block.
"""

from __future__ import annotations

import numpy as np
import pytest

from aethergraph import Graph, NeighborSampler, SamplingConfig


@pytest.fixture
def small_graph() -> Graph:
    """Small graph with explicit out-edges from 0,1,2,3,4."""
    src = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.uint32)
    dst = np.array([5, 6, 5, 7, 5, 8, 5, 9, 5, 10], dtype=np.uint32)
    return Graph.from_edges(11, src, dst)


def test_disjoint_single_seed(small_graph: Graph) -> None:
    cfg = SamplingConfig(num_neighbors=[2], disjoint=True, seed=0)
    sub = NeighborSampler(small_graph, cfg).sample([0])
    assert sub.num_seeds == 1
    batch = sub.batch
    assert batch is not None
    # All nodes in the subgraph belong to seed 0.
    assert np.all(np.asarray(batch) == 0)
    seed_indices = np.asarray(sub.seed_indices)
    assert seed_indices.tolist() == [0]


def test_disjoint_multiple_seeds(small_graph: Graph) -> None:
    cfg = SamplingConfig(num_neighbors=[2], disjoint=True, seed=0)
    sub = NeighborSampler(small_graph, cfg).sample([0, 1, 2])
    assert sub.num_seeds == 3
    batch = np.asarray(sub.batch)
    assert batch is not None
    # Batch vector is monotonic non-decreasing — the invariant the
    # reconstruction relies on.
    assert np.all(batch[:-1] <= batch[1:])
    seed_indices = np.asarray(sub.seed_indices)
    assert seed_indices.tolist() == sorted(seed_indices.tolist())
    # First occurrence of each seed id 0,1,2 corresponds to indices in `batch`.
    for sid in range(3):
        first_occurrence = int(np.where(batch == sid)[0][0])
        assert seed_indices[sid] == first_occurrence


def test_disjoint_empty_seeds(small_graph: Graph) -> None:
    cfg = SamplingConfig(num_neighbors=[2], disjoint=True, seed=0)
    sub = NeighborSampler(small_graph, cfg).sample([])
    assert sub.num_seeds == 0
    # Batch vector may be None or an empty array — both are acceptable.
    batch = sub.batch
    if batch is not None:
        assert len(np.asarray(batch)) == 0


def test_disjoint_repeated_seed(small_graph: Graph) -> None:
    """When the same node appears twice as a seed, disjoint mode gives each
    one its own subgraph (it does NOT dedupe seeds)."""
    cfg = SamplingConfig(num_neighbors=[2], disjoint=True, seed=0)
    sub = NeighborSampler(small_graph, cfg).sample([0, 0])
    assert sub.num_seeds == 2
    batch = np.asarray(sub.batch)
    # Each instance of seed 0 produces its own block, so batch is still
    # monotonic but now has two distinct seed IDs (0 and 1).
    assert np.all(batch[:-1] <= batch[1:])
    assert int(batch.max()) == 1


def test_seed_indices_match_nodes(small_graph: Graph) -> None:
    """seed_indices[k] should point to the k-th seed's entry in `nodes`."""
    cfg = SamplingConfig(num_neighbors=[2], disjoint=True, seed=0)
    sub = NeighborSampler(small_graph, cfg).sample([0, 1, 2])
    nodes = np.asarray(sub.nodes)
    seeds = np.asarray(sub.seeds)
    seed_indices = np.asarray(sub.seed_indices)
    # nodes[seed_indices[k]] must equal seeds[k]
    for k in range(len(seeds)):
        assert nodes[seed_indices[k]] == seeds[k]
