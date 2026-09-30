"""Sampling semantics at the Python boundary: PyG edge orientation, seed and
config validation, per-hop counts, and the parallel sampler's RNG stream."""

from __future__ import annotations

import numpy as np
import pytest

from aethergraph import Graph, ParallelBatchSampler, Sampler, SamplingConfig
from aethergraph._core import SamplingConfig as CoreSamplingConfig
from aethergraph._core import SamplingError


@pytest.fixture
def ring() -> Graph:
    """200 nodes, each with 6 distinct out-neighbors."""
    n = 200
    src = np.repeat(np.arange(n, dtype=np.int64), 6)
    step = np.tile(np.arange(1, 7, dtype=np.int64) * 7, n)
    dst = (src + step) % n
    return Graph.from_edges(n, src.astype(np.uint32), dst.astype(np.uint32))


def test_edge_index_runs_from_neighbor_to_expanded_node(ring: Graph) -> None:
    """Row 1 holds the node that sampled row 0, so hop-1 messages reach the
    seeds, and each edge id names the stored edge row 1 -> row 0."""
    sub = Sampler(ring, SamplingConfig(num_neighbors=[3], seed=1)).sample([0, 50, 100])
    ei = sub.edge_index_local
    assert ei.shape == (2, 9)
    assert set(ei[1].tolist()) <= set(sub.seed_indices.tolist())

    nodes = sub.nodes
    np.testing.assert_array_equal(sub.edge_index, nodes[ei])
    row_start = np.concatenate([[0], np.cumsum(ring.degrees(), dtype=np.int64)])
    for (neighbor, expanded), eid in zip(ei.T.tolist(), sub.edge_ids.tolist(), strict=True):
        src, dst = int(nodes[expanded]), int(nodes[neighbor])
        assert row_start[src] <= eid < row_start[src + 1]
        assert int(ring.neighbors(src)[eid - row_start[src]]) == dst


def test_num_sampled_counts_follow_pyg_layout(ring: Graph) -> None:
    sub = Sampler(ring, SamplingConfig(num_neighbors=[3, 2], seed=2)).sample([0, 0, 5])
    per_hop = sub.num_sampled_nodes_per_hop
    assert len(per_hop) == 3
    assert per_hop[0] == 2  # unique seed nodes
    assert sum(per_hop) == sub.num_nodes
    assert len(sub.num_sampled_edges_per_hop) == 2
    assert sum(sub.num_sampled_edges_per_hop) == sub.num_edges


def test_out_of_range_seed_raises(ring: Graph) -> None:
    cfg = SamplingConfig(num_neighbors=[2])
    with pytest.raises(SamplingError, match="out of range"):
        Sampler(ring, cfg).sample([0, 200])
    with pytest.raises(SamplingError, match="out of range"):
        ParallelBatchSampler(ring, cfg).sample_batches([[1], np.array([999], dtype=np.int64)])


def test_input_times_need_a_temporal_strategy(ring: Graph) -> None:
    sampler = Sampler(ring, SamplingConfig(num_neighbors=[2]))
    with pytest.raises(SamplingError, match="temporal_strategy"):
        sampler.sample([0], input_times=np.array([1.0]))


def test_config_graph_mismatch_fails_at_construction(ring: Graph) -> None:
    with pytest.raises(ValueError, match="timestamps"):
        Sampler(ring, SamplingConfig(num_neighbors=[2], temporal_strategy="uniform"))
    with pytest.raises(ValueError, match="weights"):
        ParallelBatchSampler(ring, SamplingConfig(num_neighbors=[2], weighted=True))


def test_temporal_without_times_leaves_every_seed_unbounded() -> None:
    """Seed 1's recent edges stay eligible however seed 0's old edges look."""
    g = Graph.from_edges(
        6, np.array([0, 0, 1, 1], dtype=np.uint32), np.array([2, 3, 4, 5], dtype=np.uint32)
    )
    g.set_timestamps(np.array([1.0, 2.0, 100.0, 200.0]))
    cfg = SamplingConfig(num_neighbors=[10], temporal_strategy="uniform", seed=1)
    sub = Sampler(g, cfg).sample([0, 1])
    assert sorted(sub.nodes.tolist()) == [0, 1, 2, 3, 4, 5]

    timed = Sampler(g, cfg).sample([0, 1], input_times=np.array([1.5, 150.0]))
    assert sorted(timed.nodes.tolist()) == [0, 1, 2, 4]
    with pytest.raises(SamplingError, match="input_times"):
        Sampler(g, cfg).sample([0, 1], input_times=np.array([1.5]))


def test_disjoint_sampler_returns_batch_vector(ring: Graph) -> None:
    cfg = SamplingConfig(num_neighbors=[2], disjoint=True, seed=4)
    sub = Sampler(ring, cfg).sample([0, 1])
    assert sub.batch is not None
    assert len(sub.batch) == sub.num_nodes
    subs = ParallelBatchSampler(ring, cfg).sample_batches([[0, 1], [2]])
    assert all(s.batch is not None for s in subs)


def test_parallel_sampler_stream_is_reproducible_and_advances(ring: Graph) -> None:
    cfg = SamplingConfig(num_neighbors=[3, 2], seed=9)
    batches = [np.arange(i, i + 4, dtype=np.uint32) for i in range(0, 64, 4)]
    a = ParallelBatchSampler(ring, cfg)
    b = ParallelBatchSampler(ring, cfg)

    first = [s.edge_ids.tolist() for s in a.sample_batches(batches)]
    assert first == [s.edge_ids.tolist() for s in b.sample_batches(batches)]
    second = [s.edge_ids.tolist() for s in a.sample_batches(batches)]
    assert first != second, "a second call replayed the first"
    # Equal batches in one call draw independently.
    same = a.sample_batches([batches[0]] * 8)
    assert len({tuple(s.edge_ids.tolist()) for s in same}) > 1


def test_bidirectional_lists_each_ordered_pair_once(ring: Graph) -> None:
    cfg = SamplingConfig(num_neighbors=[4, 3], subgraph_type="bidirectional", seed=6)
    sub = Sampler(ring, cfg).sample([0, 1, 2])
    pairs = set(map(tuple, sub.edge_index_local.T.tolist()))
    assert len(pairs) == sub.num_edges
    assert all((v, u) in pairs for u, v in pairs)


def test_induced_holds_every_edge_between_sampled_nodes(ring: Graph) -> None:
    cfg = SamplingConfig(num_neighbors=[4, 3], subgraph_type="induced", seed=6)
    sub = Sampler(ring, cfg).sample([0, 1, 2])
    members = set(sub.nodes.tolist())
    expected = {
        (int(v), u) for u in members for v in ring.neighbors(u).tolist() if int(v) in members
    }
    got = set(map(tuple, sub.edge_index.T.tolist()))
    assert got == expected


def test_core_config_is_the_single_validator() -> None:
    with pytest.raises(ValueError, match="max_degree"):
        CoreSamplingConfig([2], max_degree=0)
    with pytest.raises(ValueError, match="non-negative"):
        CoreSamplingConfig([2, -1])
    with pytest.raises(ValueError, match="temporal_strategy"):
        SamplingConfig(num_neighbors=[2], temporal_strategy="latest")  # type: ignore[arg-type]
    assert SamplingConfig(num_neighbors=[2]).cumulative is False


def test_to_arrow_follows_edge_index_with_uint64_ids(ring: Graph) -> None:
    pa = pytest.importorskip("pyarrow")
    sub = Sampler(ring, SamplingConfig(num_neighbors=[3], seed=5)).sample([0])
    edges = sub.to_arrow()["edges"]
    assert edges.schema.field("edge_id").type == pa.uint64()
    np.testing.assert_array_equal(edges.column("edge_src").to_numpy(), sub.edge_index[0])
    np.testing.assert_array_equal(edges.column("edge_dst").to_numpy(), sub.edge_index[1])
    np.testing.assert_array_equal(edges.column("edge_id").to_numpy(), sub.edge_ids)
