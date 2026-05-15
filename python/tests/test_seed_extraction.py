"""
Tests for the shared seed-coercion path used by NeighborSampler,
HeteroNeighborSampler, and NeighborLoader.submit.

These cover the edge cases in `error::extract_seeds` from the Rust side:
- uint32 numpy array (zero-copy fast path)
- int64 numpy array (range-checked widening)
- Python list of ints (generic fallback)
- empty seeds
- negative i64 values (must raise)
- i64 values exceeding u32::MAX (must raise)
"""

from __future__ import annotations

import numpy as np
import pytest

from aethergraph import Graph, NeighborSampler, SamplingConfig
from aethergraph._core import SamplingError


@pytest.fixture
def chain_graph() -> Graph:
    """A 100-node directed chain 0→1→2→...→99."""
    src = np.arange(0, 99, dtype=np.uint32)
    dst = np.arange(1, 100, dtype=np.uint32)
    return Graph.from_edges(100, src, dst)


@pytest.fixture
def sampler(chain_graph: Graph) -> NeighborSampler:
    cfg = SamplingConfig(num_neighbors=[2], seed=42)
    return NeighborSampler(chain_graph, cfg)


def test_seeds_uint32_array(sampler: NeighborSampler) -> None:
    seeds = np.array([0, 5, 10], dtype=np.uint32)
    sub = sampler.sample(seeds)
    assert sub.num_seeds == 3


def test_seeds_int64_array(sampler: NeighborSampler) -> None:
    seeds = np.array([0, 5, 10], dtype=np.int64)
    sub = sampler.sample(seeds)
    assert sub.num_seeds == 3


def test_seeds_python_list(sampler: NeighborSampler) -> None:
    sub = sampler.sample([0, 5, 10])
    assert sub.num_seeds == 3


def test_seeds_empty(sampler: NeighborSampler) -> None:
    seeds = np.array([], dtype=np.uint32)
    sub = sampler.sample(seeds)
    assert sub.num_seeds == 0
    assert sub.num_nodes == 0


def test_seeds_negative_i64_raises(sampler: NeighborSampler) -> None:
    seeds = np.array([0, -1, 5], dtype=np.int64)
    with pytest.raises(SamplingError, match="out of u32 range"):
        sampler.sample(seeds)


def test_seeds_overflow_i64_raises(sampler: NeighborSampler) -> None:
    seeds = np.array([0, int(np.iinfo(np.uint32).max) + 1], dtype=np.int64)
    with pytest.raises(SamplingError, match="out of u32 range"):
        sampler.sample(seeds)
