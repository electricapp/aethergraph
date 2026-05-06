"""Property-based tests for AetherGraph using Hypothesis."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from aethergraph import Graph, NeighborSampler, SamplingConfig


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=20)
def test_sampling_deterministic_with_seed(seed: int) -> None:
    """Sampling with the same seed produces identical results."""
    rng = np.random.default_rng(42)
    src = rng.integers(0, 50, size=100, dtype=np.uint32)
    dst = rng.integers(0, 50, size=100, dtype=np.uint32)
    graph = Graph.from_edges(50, src, dst)

    config = SamplingConfig(num_neighbors=[5, 3], seed=seed)
    sampler1 = NeighborSampler(graph, config)
    sampler2 = NeighborSampler(graph, config)

    seeds = np.array([0, 1, 2], dtype=np.uint32)
    result1 = sampler1.sample(seeds)
    result2 = sampler2.sample(seeds)

    assert np.array_equal(result1.nodes, result2.nodes)


@given(seeds=st.lists(st.integers(0, 49), min_size=1, max_size=10, unique=True))
@settings(max_examples=20)
def test_sampled_nodes_exist_in_graph(seeds: list[int]) -> None:
    """All sampled nodes must exist in the original graph."""
    rng = np.random.default_rng(42)
    src = rng.integers(0, 50, size=100, dtype=np.uint32)
    dst = rng.integers(0, 50, size=100, dtype=np.uint32)
    graph = Graph.from_edges(50, src, dst)

    config = SamplingConfig(num_neighbors=[5], seed=123)
    sampler = NeighborSampler(graph, config)
    result = sampler.sample(np.array(seeds, dtype=np.uint32))

    assert all(0 <= n < 50 for n in result.nodes)
