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
    """Reddit-shaped heterogeneous graph with PyG-style reverse relations.

    Node types: user(100), post(500), comment(1000), subreddit(50)
    Edge types:
      - (user, votes, post):               2000 edges
      - (user, writes, comment):           3000 edges
      - (comment, reply_to, comment):      1500 edges
      - (post, belongs_to, subreddit):     500 edges
      - (post, rev_votes, user):           2000 edges (transpose of votes)
      - (comment, rev_writes, user):       3000 edges (transpose of writes)
      - (subreddit, rev_belongs_to, post): 500 edges (transpose of belongs_to)

    Sampling expands a node along the relations that point into its type,
    so the reverse relations are what give user seeds neighbors.
    """
    votes = (
        rng.integers(0, 100, 2000).astype(np.uint32),
        rng.integers(0, 500, 2000).astype(np.uint32),
    )
    writes = (
        rng.integers(0, 100, 3000).astype(np.uint32),
        rng.integers(0, 1000, 3000).astype(np.uint32),
    )
    reply_to = (
        rng.integers(0, 1000, 1500).astype(np.uint32),
        rng.integers(0, 1000, 1500).astype(np.uint32),
    )
    belongs_to = (
        rng.integers(0, 500, 500).astype(np.uint32),
        rng.integers(0, 50, 500).astype(np.uint32),
    )
    return HeteroCsrGraph.from_edge_arrays(
        node_types={"user": 100, "post": 500, "comment": 1000, "subreddit": 50},
        edge_types=[
            ("user", "votes", "post", *votes),
            ("user", "writes", "comment", *writes),
            ("comment", "reply_to", "comment", *reply_to),
            ("post", "belongs_to", "subreddit", *belongs_to),
            ("post", "rev_votes", "user", votes[1], votes[0]),
            ("comment", "rev_writes", "user", writes[1], writes[0]),
            ("subreddit", "rev_belongs_to", "post", belongs_to[1], belongs_to[0]),
        ],
    )


FANOUT_1HOP: dict[tuple[str, str, str], list[int]] = {
    ("user", "votes", "post"): [10],
    ("user", "writes", "comment"): [5],
    ("comment", "reply_to", "comment"): [3],
    ("post", "belongs_to", "subreddit"): [2],
    ("post", "rev_votes", "user"): [10],
    ("comment", "rev_writes", "user"): [5],
    ("subreddit", "rev_belongs_to", "post"): [2],
}


def fanout(*hops: int) -> dict[tuple[str, str, str], list[int]]:
    """Every relation of the reddit fixture with the same per-hop counts."""
    return {edge_type: list(hops) for edge_type in FANOUT_1HOP}


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
        assert set(types) == set(FANOUT_1HOP)

    def test_node_counts(self, reddit_graph: HeteroCsrGraph) -> None:
        assert reddit_graph.num_nodes("user") == 100
        assert reddit_graph.num_nodes("post") == 500
        assert reddit_graph.num_nodes("comment") == 1000
        assert reddit_graph.num_nodes("subreddit") == 50

    def test_total_counts(self, reddit_graph: HeteroCsrGraph) -> None:
        assert reddit_graph.total_nodes() == 1650
        assert reddit_graph.total_edges() == 12500

    def test_edge_counts(self, reddit_graph: HeteroCsrGraph) -> None:
        assert reddit_graph.num_edges("user", "votes", "post") == 2000
        assert reddit_graph.num_edges("user", "writes", "comment") == 3000
        assert reddit_graph.num_edges("comment", "reply_to", "comment") == 1500
        assert reddit_graph.num_edges("post", "belongs_to", "subreddit") == 500
        assert reddit_graph.num_edges("post", "rev_votes", "user") == 2000

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
        config = HeteroSamplingConfig(num_neighbors=FANOUT_1HOP)
        sampler = HeteroNeighborSampler(reddit_graph, config)
        seeds = np.array([0, 1, 2], dtype=np.uint32)
        sub = sampler.sample("user", seeds)

        assert sub.seed_type == "user"
        assert list(sub.seeds) == [0, 1, 2]
        assert len(sub.nodes("user")) >= 3  # at least the seeds
        assert len(sub.nodes("post")) > 0

    def test_two_hop_sampling(self, reddit_graph: HeteroCsrGraph) -> None:
        config = HeteroSamplingConfig(num_neighbors=fanout(5, 3))
        sampler = HeteroNeighborSampler(reddit_graph, config)
        seeds = np.array([0, 1, 2, 3, 4], dtype=np.uint32)
        sub = sampler.sample("user", seeds)

        assert len(sub.nodes("user")) >= 5
        assert len(sub.nodes("post")) > 0
        # Subreddits reached via user <- rev_votes <- post <- rev_belongs_to.
        assert len(sub.nodes("subreddit")) > 0

    def test_edge_index_local_shape(self, reddit_graph: HeteroCsrGraph) -> None:
        config = HeteroSamplingConfig(num_neighbors=fanout(5, 3))
        sampler = HeteroNeighborSampler(reddit_graph, config)
        sub = sampler.sample("user", np.array([0, 1], dtype=np.uint32))

        for src, rel, dst in sub.edge_types:
            ei = sub.edge_index_local(src, rel, dst)
            assert ei.ndim == 2
            assert ei.shape[0] == 2
            # Local indices should be < number of sampled nodes of that type
            if ei.shape[1] > 0:
                assert ei[0].max() < len(sub.nodes(src))
                assert ei[1].max() < len(sub.nodes(dst))

    def test_edges_flow_toward_seeds(self, reddit_graph: HeteroCsrGraph) -> None:
        """Hop-1 edges end at seeds; nothing points out of them."""
        config = HeteroSamplingConfig(num_neighbors=FANOUT_1HOP, seed=3)
        sampler = HeteroNeighborSampler(reddit_graph, config)
        sub = sampler.sample("user", np.array([4, 7, 9], dtype=np.uint32))

        ei = sub.edge_index_local("post", "rev_votes", "user")
        assert ei.shape[1] > 0
        assert set(ei[1].tolist()) <= set(sub.seed_indices.tolist())
        assert sub.edge_index_local("user", "votes", "post").shape[1] == 0

    def test_edges_keep_stored_direction(self) -> None:
        """Each sampled edge is a stored edge of its relation, source first."""
        posts = np.arange(4, dtype=np.uint32)
        voters = posts % 2  # post p was voted on by user p % 2
        g = HeteroCsrGraph.from_edge_arrays(
            node_types={"user": 2, "post": 4},
            edge_types=[
                ("user", "votes", "post", voters, posts),
                ("post", "rev_votes", "user", posts, voters),
            ],
        )
        config = HeteroSamplingConfig(
            num_neighbors={("post", "rev_votes", "user"): [10], ("user", "votes", "post"): [10]},
            seed=3,
        )
        sub = HeteroNeighborSampler(g, config).sample("user", np.array([1], dtype=np.uint32))
        ei = sub.edge_index_local("post", "rev_votes", "user")
        post_ids, user_ids = sub.nodes("post"), sub.nodes("user")
        edges = {
            (int(post_ids[p]), int(user_ids[u]))
            for p, u in zip(ei[0].tolist(), ei[1].tolist(), strict=True)
        }
        assert edges == {(1, 1), (3, 1)}

    def test_empty_seeds(self, reddit_graph: HeteroCsrGraph) -> None:
        config = HeteroSamplingConfig(num_neighbors=FANOUT_1HOP)
        sampler = HeteroNeighborSampler(reddit_graph, config)
        sub = sampler.sample("user", np.array([], dtype=np.uint32))
        assert len(sub.seeds) == 0

    def test_out_of_range_seed_raises(self, reddit_graph: HeteroCsrGraph) -> None:
        from aethergraph._core import SamplingError

        config = HeteroSamplingConfig(num_neighbors=FANOUT_1HOP)
        sampler = HeteroNeighborSampler(reddit_graph, config)
        # 100 users: 100 is out of range even though it fits other types.
        with pytest.raises(SamplingError, match="out of range"):
            sampler.sample("user", np.array([0, 100], dtype=np.int64))

    def test_reproducible_with_seed(self, reddit_graph: HeteroCsrGraph) -> None:
        config = HeteroSamplingConfig(num_neighbors=FANOUT_1HOP, seed=123)
        seeds = np.array([0, 1, 2], dtype=np.uint32)

        s1 = HeteroNeighborSampler(reddit_graph, config)
        sub1 = s1.sample("user", seeds)

        s2 = HeteroNeighborSampler(reddit_graph, config)
        sub2 = s2.sample("user", seeds)

        for nt in sub1.node_types:
            np.testing.assert_array_equal(sub1.nodes(nt), sub2.nodes(nt))

    def test_single_edge_type(self) -> None:
        """Graph with only one edge type, sampled from its destination type."""
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
        sub = sampler.sample("post", np.array([0, 1, 2], dtype=np.uint32))
        assert len(sub.nodes("post")) >= 3
        assert len(sub.nodes("user")) > 0
        # A type no relation points into gains no neighbors.
        users_only = sampler.sample("user", np.array([0, 1], dtype=np.uint32))
        assert len(users_only.nodes("post")) == 0


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
        assert g.total_edges == 12500


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
            num_neighbors=fanout(10, 5),
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
            # Hop 1 draws posts into the seeds; hop 2 draws users into posts.
            assert ("post", "rev_votes", "user") in batch.edge_types
            assert ("user", "votes", "post") in batch.edge_types
            ei = batch["post", "rev_votes", "user"].edge_index
            assert ei.shape[0] == 2

    def test_seeds_receive_messages(self, reddit_hetero_graph: HeteroGraph) -> None:
        """Every hop-1 edge into the seed type ends at a seed (PyG flow)."""
        import torch

        from aethergraph.pytorch import HeteroNeighborLoader

        loader = HeteroNeighborLoader(
            reddit_hetero_graph,
            num_neighbors=FANOUT_1HOP,
            input_nodes=("user", torch.arange(10)),
            batch_size=10,
        )
        batch = next(iter(loader))
        user = batch["user"]
        ei = batch["post", "rev_votes", "user"].edge_index
        assert ei.shape[1] > 0
        assert set(ei[1].tolist()) <= set(user.input_id.tolist())

    def test_features_attached(
        self,
        reddit_hetero_graph: HeteroGraph,
        reddit_hetero_features: dict[str, npt.NDArray[np.float32]],
    ) -> None:
        import torch

        from aethergraph.pytorch import HeteroNeighborLoader

        loader = HeteroNeighborLoader(
            reddit_hetero_graph,
            num_neighbors=fanout(5),
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
            num_neighbors=fanout(5),
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
            num_neighbors=fanout(5),
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
            num_neighbors=fanout(5),
            input_nodes=("user", torch.arange(60)),
            batch_size=10,
            shuffle=False,
            num_workers=4,
        )

        batches = list(loader)
        assert len(batches) == 6

        # Delivery order across the pool is unordered; each batch's input_id
        # still names its own seeds' positions in input_nodes (PyG), and
        # seed_index their local indices into n_id.
        seen: list[int] = []
        positions: list[int] = []
        for batch in batches:
            u = batch["user"]
            seeds = u.n_id[u.seed_index]
            seen.extend(seeds.tolist())
            positions.extend(u.input_id.tolist())
            assert u.input_id.tolist() == seeds.tolist()  # input_nodes is arange(60)
        assert sorted(seen) == list(range(60))
        assert sorted(positions) == list(range(60))

    def test_pin_memory(self, reddit_hetero_graph: HeteroGraph) -> None:
        import torch

        from aethergraph.pytorch import HeteroNeighborLoader

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        loader = HeteroNeighborLoader(
            reddit_hetero_graph,
            num_neighbors=fanout(5),
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
            num_neighbors=fanout(5),
            input_nodes=("user", torch.tensor([1, 1, 2])),
            batch_size=3,
            shuffle=False,
        )
        batch = next(iter(loader))
        u = batch["user"]
        assert u.batch_size == 3
        assert u.input_id.tolist() == [0, 1, 2]
        # Locals: first two seeds collide → same local index.
        assert u.seed_index[0].item() == u.seed_index[1].item()
        assert u.n_id[u.seed_index].tolist() == [1, 1, 2]

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
        seeds = torch.arange(40)

        def epoch(workers: int) -> list[tuple[tuple[int, ...], ...]]:
            ld = HeteroNeighborLoader(
                g,
                num_neighbors=fanout(3, 2),
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
                globals_ = tuple(u.n_id[u.seed_index].tolist())
                n_id = tuple(u.n_id.tolist())
                out.append((globals_, n_id))
            return out

        a = epoch(1)
        assert a == epoch(1)
        assert a == epoch(4)

    def test_seeded_epochs_resample_neighborhoods(self, reddit_hetero_graph: HeteroGraph) -> None:
        """A seeded loader draws fresh neighborhoods every epoch, reproducibly."""
        import torch

        from aethergraph.pytorch import HeteroNeighborLoader

        def make() -> HeteroNeighborLoader:
            return HeteroNeighborLoader(
                reddit_hetero_graph,
                num_neighbors={("post", "rev_votes", "user"): [2]},
                input_nodes=("user", torch.arange(40)),
                batch_size=40,
                seed=5,
                shuffle=False,
            )

        def post_ids(ld: HeteroNeighborLoader) -> list[int]:
            return next(iter(ld))["post"].n_id.tolist()

        loader = make()
        first, second = post_ids(loader), post_ids(loader)
        assert first != second
        replay = make()
        assert post_ids(replay) == first
        assert post_ids(replay) == second


class TestHeteroRawLoader:
    """The `_core.HeteroNeighborLoader` pipeline contract."""

    def test_next_batch_echoes_batch_idx_and_reuses_across_epochs(
        self, reddit_hetero_graph: HeteroGraph
    ) -> None:
        from aethergraph._core import HeteroNeighborLoader, HeteroSamplingConfig

        cfg = HeteroSamplingConfig({("post", "rev_votes", "user"): [2]}, seed=1)
        with HeteroNeighborLoader(reddit_hetero_graph.csr, cfg, "user", 2, 3) as loader:
            for _epoch in range(2):
                for i in range(6):
                    loader.submit(batch_idx=i, seeds=[i])
                got = [loader.next_batch() for _ in range(6)]
                assert [g[0] for g in got if g is not None] == list(range(6))
        assert loader.next() is None

    def test_seed_past_its_type_fails_at_submit(self, reddit_hetero_graph: HeteroGraph) -> None:
        """Seeds are checked against the seed type's count, not the graph's."""
        from aethergraph._core import HeteroNeighborLoader, HeteroSamplingConfig, SamplingError

        cfg = HeteroSamplingConfig({("post", "rev_votes", "user"): [2]})
        with HeteroNeighborLoader(reddit_hetero_graph.csr, cfg, "user") as loader:
            # 100 users, 500 posts: 150 is a valid post but not a valid user.
            with pytest.raises(SamplingError, match="out of range"):
                loader.submit(0, [150])
            loader.submit(1, [99])
            got = loader.next_batch()
            assert got is not None and got[0] == 1

    def test_duplicate_edge_type_raises_value_error(self) -> None:
        src = np.array([0], dtype=np.uint32)
        with pytest.raises(ValueError, match="duplicate edge type"):
            HeteroGraph.from_edge_arrays(
                node_types={"a": 2},
                edge_types=[("a", "r", "a", src, src), ("a", "r", "a", src, src)],
            )

    def test_too_many_edge_types_raises_value_error(self) -> None:
        src = np.array([0], dtype=np.uint32)
        with pytest.raises(ValueError, match="too many edge types"):
            HeteroGraph.from_edge_arrays(
                node_types={"a": 2},
                edge_types=[("a", f"r{i}", "a", src, src) for i in range(256)],
            )

    def test_endpoint_past_its_type_raises_value_error(self) -> None:
        src = np.array([0], dtype=np.uint32)
        dst = np.array([4], dtype=np.uint32)
        with pytest.raises(ValueError, match=r"\(a, r, b\)"):
            HeteroCsrGraph.from_edge_arrays(
                node_types={"a": 1, "b": 4},
                edge_types=[("a", "r", "b", src, dst)],
            )

    def test_lopsided_edge_type_samples_far_destinations(self) -> None:
        """A 2-user x 200k-item relation reaches the highest item ids."""
        users = np.array([0, 1], dtype=np.uint32)
        items = np.array([7, 199_999], dtype=np.uint32)
        g = HeteroCsrGraph.from_edge_arrays(
            node_types={"user": 2, "item": 200_000},
            edge_types=[("user", "buys", "item", users, items)],
        )
        assert g.num_edges("user", "buys", "item") == 2
        config = HeteroSamplingConfig(num_neighbors={("user", "buys", "item"): [4]}, seed=0)
        sub = HeteroNeighborSampler(g, config).sample("item", np.array([199_999], dtype=np.uint32))
        assert sub.nodes("user").tolist() == [1]
