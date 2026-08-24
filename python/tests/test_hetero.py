"""Tests for heterogeneous graph support.

Tests cover:
- HeteroGraph construction from edge arrays
- HeteroGraph metadata (node types, edge types, counts)
- HeteroNeighborSampler: typed multi-hop sampling
- HeteroSampledSubgraph: node/edge access, local edge indices
- HeteroNeighborLoader: full PyG HeteroData pipeline (requires torch + pyg)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from aethergraph import HeteroGraph
from aethergraph._core import (
    HeteroCsrGraph,
    HeteroNeighborSampler,
    HeteroSamplingConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def reddit_graph(rng: np.random.Generator) -> HeteroCsrGraph:
    """Reddit-shaped heterogeneous graph.

    Node types: user(100), post(500), comment(1000), subreddit(50)
    Edge types:
      - (user, votes, post):           2000 edges
      - (user, writes, comment):       3000 edges
      - (comment, reply_to, comment):  1500 edges
      - (post, belongs_to, subreddit): 500 edges
    """
    return HeteroCsrGraph.from_edge_arrays(
        node_types={"user": 100, "post": 500, "comment": 1000, "subreddit": 50},
        edge_types=[
            (
                "user",
                "votes",
                "post",
                rng.integers(0, 100, 2000).astype(np.uint32),
                rng.integers(0, 500, 2000).astype(np.uint32),
            ),
            (
                "user",
                "writes",
                "comment",
                rng.integers(0, 100, 3000).astype(np.uint32),
                rng.integers(0, 1000, 3000).astype(np.uint32),
            ),
            (
                "comment",
                "reply_to",
                "comment",
                rng.integers(0, 1000, 1500).astype(np.uint32),
                rng.integers(0, 1000, 1500).astype(np.uint32),
            ),
            (
                "post",
                "belongs_to",
                "subreddit",
                rng.integers(0, 500, 500).astype(np.uint32),
                rng.integers(0, 50, 500).astype(np.uint32),
            ),
        ],
    )


@pytest.fixture
def reddit_hetero_graph(reddit_graph: HeteroCsrGraph) -> HeteroGraph:
    """Python HeteroGraph builder around the Rust CSR fixture."""
    return HeteroGraph(reddit_graph)


@pytest.fixture
def reddit_hetero_features(
    rng: np.random.Generator,
) -> dict[str, npt.NDArray[np.float32]]:
    """Per-node-type feature arrays sized to the reddit hetero fixture.

    Kept separate from the graph fixture so tests opt in to features
    explicitly, mirroring the homo `small_graph_with_features` pattern."""
    return {
        "user": rng.standard_normal((100, 64)).astype(np.float32),
        "post": rng.standard_normal((500, 128)).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# HeteroCsrGraph construction
# ---------------------------------------------------------------------------


class TestHeteroCsrGraph:
    def test_node_types(self, reddit_graph: HeteroCsrGraph) -> None:
        types = reddit_graph.node_types()
        assert set(types) == {"user", "post", "comment", "subreddit"}

    def test_edge_types(self, reddit_graph: HeteroCsrGraph) -> None:
        types = reddit_graph.edge_types()
        expected = {
            ("user", "votes", "post"),
            ("user", "writes", "comment"),
            ("comment", "reply_to", "comment"),
            ("post", "belongs_to", "subreddit"),
        }
        assert set(types) == expected

    def test_node_counts(self, reddit_graph: HeteroCsrGraph) -> None:
        assert reddit_graph.num_nodes("user") == 100
        assert reddit_graph.num_nodes("post") == 500
        assert reddit_graph.num_nodes("comment") == 1000
        assert reddit_graph.num_nodes("subreddit") == 50

    def test_total_counts(self, reddit_graph: HeteroCsrGraph) -> None:
        assert reddit_graph.total_nodes() == 1650
        assert reddit_graph.total_edges() == 7000

    def test_edge_counts(self, reddit_graph: HeteroCsrGraph) -> None:
        assert reddit_graph.num_edges("user", "votes", "post") == 2000
        assert reddit_graph.num_edges("user", "writes", "comment") == 3000
        assert reddit_graph.num_edges("comment", "reply_to", "comment") == 1500
        assert reddit_graph.num_edges("post", "belongs_to", "subreddit") == 500

    def test_unknown_node_type_raises(self, reddit_graph: HeteroCsrGraph) -> None:
        with pytest.raises(KeyError, match="unknown node type"):
            reddit_graph.num_nodes("nonexistent")

    def test_unknown_edge_type_raises(self, reddit_graph: HeteroCsrGraph) -> None:
        with pytest.raises(KeyError, match="unknown edge type"):
            reddit_graph.num_edges("user", "follows", "user")

    def test_empty_edge_type(self) -> None:
        """Edge type with zero edges should work."""
        g = HeteroCsrGraph.from_edge_arrays(
            node_types={"a": 10, "b": 10},
            edge_types=[
                ("a", "rel", "b", np.array([], dtype=np.uint32), np.array([], dtype=np.uint32)),
            ],
        )
        assert g.num_edges("a", "rel", "b") == 0

    def test_self_loop_edge_type(self) -> None:
        """Edge type where src and dst are the same type."""
        g = HeteroCsrGraph.from_edge_arrays(
            node_types={"node": 50},
            edge_types=[
                (
                    "node",
                    "connects",
                    "node",
                    np.array([0, 1, 2], dtype=np.uint32),
                    np.array([1, 2, 0], dtype=np.uint32),
                ),
            ],
        )
        assert g.num_edges("node", "connects", "node") == 3


# ---------------------------------------------------------------------------
# HeteroNeighborSampler
# ---------------------------------------------------------------------------


class TestHeteroSampler:
    def test_basic_sampling(self, reddit_graph: HeteroCsrGraph) -> None:
        config = HeteroSamplingConfig(
            num_neighbors={
                ("user", "votes", "post"): [10],
                ("user", "writes", "comment"): [5],
                ("comment", "reply_to", "comment"): [3],
                ("post", "belongs_to", "subreddit"): [2],
            },
        )
        sampler = HeteroNeighborSampler(reddit_graph, config)
        seeds = np.array([0, 1, 2], dtype=np.uint32)
        sub = sampler.sample("user", seeds)

        assert sub.seed_type == "user"
        assert list(sub.seeds) == [0, 1, 2]
        # Should have sampled users (seeds) + posts + comments
        assert len(sub.nodes("user")) >= 3  # at least the seeds

    def test_two_hop_sampling(self, reddit_graph: HeteroCsrGraph) -> None:
        config = HeteroSamplingConfig(
            num_neighbors={
                ("user", "votes", "post"): [5, 3],
                ("user", "writes", "comment"): [5, 3],
                ("comment", "reply_to", "comment"): [2, 2],
                ("post", "belongs_to", "subreddit"): [2, 2],
            },
        )
        sampler = HeteroNeighborSampler(reddit_graph, config)
        seeds = np.array([0, 1, 2, 3, 4], dtype=np.uint32)
        sub = sampler.sample("user", seeds)

        # 2-hop from users should reach posts, comments, and subreddits
        assert len(sub.nodes("user")) >= 5
        assert len(sub.nodes("post")) > 0
        # Subreddits reached via user->votes->post->belongs_to->subreddit (2 hops)
        assert len(sub.nodes("subreddit")) > 0

    def test_edge_index_local_shape(self, reddit_graph: HeteroCsrGraph) -> None:
        config = HeteroSamplingConfig(
            num_neighbors={
                ("user", "votes", "post"): [10],
                ("user", "writes", "comment"): [5],
                ("comment", "reply_to", "comment"): [3],
                ("post", "belongs_to", "subreddit"): [2],
            },
        )
        sampler = HeteroNeighborSampler(reddit_graph, config)
        sub = sampler.sample("user", np.array([0, 1], dtype=np.uint32))

        for src, rel, dst in sub.edge_types:
            ei = sub.edge_index_local(src, rel, dst)
            assert ei.ndim == 2
            assert ei.shape[0] == 2
            # Local indices should be < number of sampled nodes of that type
            if ei.shape[1] > 0:
                src_nodes = sub.nodes(src)
                dst_nodes = sub.nodes(dst)
                assert ei[0].max() < len(src_nodes)
                assert ei[1].max() < len(dst_nodes)

    def test_empty_seeds(self, reddit_graph: HeteroCsrGraph) -> None:
        config = HeteroSamplingConfig(
            num_neighbors={
                ("user", "votes", "post"): [10],
                ("user", "writes", "comment"): [5],
                ("comment", "reply_to", "comment"): [3],
                ("post", "belongs_to", "subreddit"): [2],
            },
        )
        sampler = HeteroNeighborSampler(reddit_graph, config)
        sub = sampler.sample("user", np.array([], dtype=np.uint32))
        assert len(sub.seeds) == 0

    def test_reproducible_with_seed(self, reddit_graph: HeteroCsrGraph) -> None:
        config = HeteroSamplingConfig(
            num_neighbors={
                ("user", "votes", "post"): [10],
                ("user", "writes", "comment"): [5],
                ("comment", "reply_to", "comment"): [3],
                ("post", "belongs_to", "subreddit"): [2],
            },
            seed=123,
        )
        seeds = np.array([0, 1, 2], dtype=np.uint32)

        s1 = HeteroNeighborSampler(reddit_graph, config)
        sub1 = s1.sample("user", seeds)

        s2 = HeteroNeighborSampler(reddit_graph, config)
        sub2 = s2.sample("user", seeds)

        for nt in sub1.node_types:
            np.testing.assert_array_equal(sub1.nodes(nt), sub2.nodes(nt))

    def test_single_edge_type(self) -> None:
        """Graph with only one edge type."""
        g = HeteroCsrGraph.from_edge_arrays(
            node_types={"user": 50, "post": 100},
            edge_types=[
                (
                    "user",
                    "likes",
                    "post",
                    np.arange(50, dtype=np.uint32) % 50,
                    np.arange(50, dtype=np.uint32) % 100,
                ),
            ],
        )
        config = HeteroSamplingConfig(
            num_neighbors={("user", "likes", "post"): [5]},
        )
        sampler = HeteroNeighborSampler(g, config)
        sub = sampler.sample("user", np.array([0, 1, 2], dtype=np.uint32))
        assert len(sub.nodes("user")) >= 3
        assert len(sub.nodes("post")) > 0


# ---------------------------------------------------------------------------
# HeteroGraph Python wrapper
# ---------------------------------------------------------------------------


class TestHeteroGraphPython:
    def test_from_edge_arrays(self) -> None:
        g = HeteroGraph.from_edge_arrays(
            node_types={"a": 10, "b": 20},
            edge_types=[
                (
                    "a",
                    "connects",
                    "b",
                    np.array([0, 1, 2], dtype=np.uint32),
                    np.array([5, 6, 7], dtype=np.uint32),
                ),
            ],
        )
        assert set(g.node_types) == {"a", "b"}
        assert g.num_nodes("a") == 10
        assert g.num_nodes("b") == 20

    def test_properties(self, reddit_hetero_graph: HeteroGraph) -> None:
        g = reddit_hetero_graph
        assert g.total_nodes == 1650
        assert g.total_edges == 7000


# ---------------------------------------------------------------------------
# HeteroNeighborLoader (requires torch + pyg)
# ---------------------------------------------------------------------------


@pytest.mark.requires_pyg
class TestHeteroNeighborLoader:
    def test_basic_iteration(self, reddit_hetero_graph: HeteroGraph) -> None:
        import torch
        from torch_geometric.data import HeteroData

        from aethergraph.pytorch import HeteroNeighborLoader

        loader = HeteroNeighborLoader(
            reddit_hetero_graph,
            num_neighbors={
                ("user", "votes", "post"): [10, 5],
                ("user", "writes", "comment"): [5, 3],
                ("comment", "reply_to", "comment"): [3, 2],
                ("post", "belongs_to", "subreddit"): [2, 1],
            },
            input_nodes=("user", torch.arange(20)),
            batch_size=10,
        )

        batches = list(loader)
        assert len(batches) == 2  # 20 seeds / 10 batch_size

        for batch in batches:
            assert isinstance(batch, HeteroData)
            # Seed type should have nodes
            assert batch["user"].num_nodes > 0
            assert hasattr(batch["user"], "n_id")
            # Should have edge indices
            assert ("user", "votes", "post") in batch.edge_types
            ei = batch["user", "votes", "post"].edge_index
            assert ei.shape[0] == 2

    def test_features_attached(
        self,
        reddit_hetero_graph: HeteroGraph,
        reddit_hetero_features: dict[str, npt.NDArray[np.float32]],
    ) -> None:
        import torch

        from aethergraph.pytorch import HeteroNeighborLoader

        loader = HeteroNeighborLoader(
            reddit_hetero_graph,
            num_neighbors={
                ("user", "votes", "post"): [5],
                ("user", "writes", "comment"): [3],
                ("comment", "reply_to", "comment"): [2],
                ("post", "belongs_to", "subreddit"): [1],
            },
            input_nodes=("user", torch.arange(10)),
            batch_size=10,
            features=reddit_hetero_features,
        )

        batch = next(iter(loader))
        # user features should be attached (64-dim)
        assert batch["user"].x is not None
        assert batch["user"].x.shape[1] == 64
        # post features should be attached (128-dim)
        assert batch["post"].x is not None
        assert batch["post"].x.shape[1] == 128
        # comment and subreddit have no features
        assert not hasattr(batch["comment"], "x") or batch["comment"].x is None

    def test_batch_size_correct(self, reddit_hetero_graph: HeteroGraph) -> None:
        import torch

        from aethergraph.pytorch import HeteroNeighborLoader

        loader = HeteroNeighborLoader(
            reddit_hetero_graph,
            num_neighbors={
                ("user", "votes", "post"): [5],
                ("user", "writes", "comment"): [3],
                ("comment", "reply_to", "comment"): [2],
                ("post", "belongs_to", "subreddit"): [1],
            },
            input_nodes=("user", torch.arange(50)),
            batch_size=16,
        )

        batches = list(loader)
        # 50 seeds / 16 batch_size = 4 batches (16, 16, 16, 2)
        assert len(batches) == 4
        assert batches[0]["user"].batch_size == 16
        assert batches[-1]["user"].batch_size == 2

    def test_shuffle_produces_different_orders(self, reddit_hetero_graph: HeteroGraph) -> None:
        import torch

        from aethergraph.pytorch import HeteroNeighborLoader

        loader = HeteroNeighborLoader(
            reddit_hetero_graph,
            num_neighbors={
                ("user", "votes", "post"): [5],
                ("user", "writes", "comment"): [3],
                ("comment", "reply_to", "comment"): [2],
                ("post", "belongs_to", "subreddit"): [1],
            },
            input_nodes=("user", torch.arange(50)),
            batch_size=10,
            shuffle=True,
        )

        epoch1_seeds = [b["user"].n_id[: b["user"].batch_size].numpy() for b in loader]
        epoch2_seeds = [b["user"].n_id[: b["user"].batch_size].numpy() for b in loader]

        # With shuffle, at least one batch should have different seeds
        any_different = any(
            not np.array_equal(s1, s2) for s1, s2 in zip(epoch1_seeds, epoch2_seeds)
        )
        assert any_different

    def test_num_workers_pool_delivers_every_batch(self, reddit_hetero_graph: HeteroGraph) -> None:
        """num_workers=4 sizes the Rust sampler pool; every batch arrives exactly once."""
        import torch

        from aethergraph.pytorch import HeteroNeighborLoader

        loader = HeteroNeighborLoader(
            reddit_hetero_graph,
            num_neighbors={
                ("user", "votes", "post"): [5],
                ("user", "writes", "comment"): [3],
                ("comment", "reply_to", "comment"): [2],
                ("post", "belongs_to", "subreddit"): [1],
            },
            input_nodes=("user", torch.arange(60)),
            batch_size=10,
            shuffle=False,
            num_workers=4,
        )

        batches = list(loader)
        assert len(batches) == 6

        # Delivery order across the pool is unordered; each subgraph carries
        # its own seeds. input_id is local indices (homo contract); recover
        # globals via n_id[input_id].
        seen: list[int] = []
        for batch in batches:
            n_id = batch["user"].n_id
            input_id = batch["user"].input_id
            seen.extend(n_id[input_id].tolist())
        assert sorted(seen) == list(range(60))

    def test_pin_memory(self, reddit_hetero_graph: HeteroGraph) -> None:
        import torch

        from aethergraph.pytorch import HeteroNeighborLoader

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        loader = HeteroNeighborLoader(
            reddit_hetero_graph,
            num_neighbors={
                ("user", "votes", "post"): [5],
                ("user", "writes", "comment"): [3],
                ("comment", "reply_to", "comment"): [2],
                ("post", "belongs_to", "subreddit"): [1],
            },
            input_nodes=("user", torch.arange(10)),
            batch_size=10,
            pin_memory=True,
        )

        batch = next(iter(loader))
        assert batch["user"].n_id.is_pinned()

    def test_duplicate_seeds_preserve_batch_size(self, reddit_hetero_graph: HeteroGraph) -> None:
        """Duplicate input seeds must not shrink batch_size vs n_id contract."""
        import torch

        from aethergraph.pytorch import HeteroNeighborLoader

        loader = HeteroNeighborLoader(
            reddit_hetero_graph,
            num_neighbors={
                ("user", "votes", "post"): [5],
                ("user", "writes", "comment"): [3],
                ("comment", "reply_to", "comment"): [2],
                ("post", "belongs_to", "subreddit"): [1],
            },
            input_nodes=("user", torch.tensor([1, 1, 2])),
            batch_size=3,
            shuffle=False,
        )
        batch = next(iter(loader))
        u = batch["user"]
        assert u.batch_size == 3
        assert len(u.input_id) == 3
        # Locals: first two seeds collide → same local index.
        assert u.input_id[0].item() == u.input_id[1].item()
        assert u.n_id[u.input_id].tolist() == [1, 1, 2]

    def test_rejects_empty_and_negative_fanout(self, reddit_hetero_graph: HeteroGraph) -> None:
        import torch

        from aethergraph.pytorch import HeteroNeighborLoader

        with pytest.raises(ValueError, match="non-empty"):
            HeteroNeighborLoader(
                reddit_hetero_graph,
                num_neighbors={},
                input_nodes=("user", torch.arange(4)),
            )
        with pytest.raises(ValueError, match="non-empty"):
            HeteroNeighborLoader(
                reddit_hetero_graph,
                num_neighbors={("user", "votes", "post"): []},
                input_nodes=("user", torch.arange(4)),
            )
        with pytest.raises(ValueError, match="non-negative"):
            HeteroNeighborLoader(
                reddit_hetero_graph,
                num_neighbors={("user", "votes", "post"): [5, -1]},
                input_nodes=("user", torch.arange(4)),
            )
        with pytest.raises(ValueError, match="max_degree"):
            HeteroNeighborLoader(
                reddit_hetero_graph,
                num_neighbors={("user", "votes", "post"): [5]},
                input_nodes=("user", torch.arange(4)),
                max_degree=0,
            )

    def test_multi_worker_seed_content_stable(self, reddit_hetero_graph: HeteroGraph) -> None:
        """Same seed + multi-worker → bit-identical ordered epoch stream."""
        import torch

        from aethergraph.pytorch import HeteroNeighborLoader

        g = reddit_hetero_graph
        fanout = {
            ("user", "votes", "post"): [5, 3],
            ("user", "writes", "comment"): [3, 2],
            ("comment", "reply_to", "comment"): [2, 2],
            ("post", "belongs_to", "subreddit"): [1, 1],
        }
        seeds = torch.arange(40)

        def epoch(workers: int) -> list[tuple[tuple[int, ...], ...]]:
            ld = HeteroNeighborLoader(
                g,
                num_neighbors=fanout,
                input_nodes=("user", seeds),
                batch_size=8,
                seed=42,
                num_workers=workers,
                shuffle=False,
                replace=True,
            )
            out: list[tuple[tuple[int, ...], ...]] = []
            for batch in ld:
                u = batch["user"]
                globals_ = tuple(u.n_id[u.input_id].tolist())
                n_id = tuple(u.n_id.tolist())
                out.append((globals_, n_id))
            return out

        a = epoch(1)
        assert a == epoch(1)
        assert a == epoch(4)
