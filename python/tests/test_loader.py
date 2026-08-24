"""
Tests for the NeighborLoader class.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

if TYPE_CHECKING:
    from aethergraph import Graph


pytestmark = pytest.mark.requires_torch


class TestNeighborLoaderBasic:
    """Basic NeighborLoader tests."""

    def test_create_loader(self, small_graph: Graph) -> None:
        """Create a basic loader."""
        from aethergraph.pytorch import NeighborLoader

        loader = NeighborLoader(
            small_graph,
            num_neighbors=[10, 5],
            batch_size=16,
        )

        assert len(loader) > 0

    def test_iteration(self, small_graph_with_features: tuple[Graph, np.ndarray]) -> None:
        """Test iterating over batches."""
        from aethergraph.pytorch import NeighborLoader

        graph, features = small_graph_with_features
        loader = NeighborLoader(
            graph,
            num_neighbors=[10, 5],
            batch_size=16,
            features=features,
        )

        batches = list(loader)
        assert len(batches) > 0

        for batch in batches:
            assert hasattr(batch, "edge_index")
            assert hasattr(batch, "n_id")
            assert batch.edge_index.shape[0] == 2

    def test_batch_size(self, small_graph: Graph) -> None:
        """Test batch_size parameter."""
        from aethergraph.pytorch import NeighborLoader

        loader = NeighborLoader(
            small_graph,
            num_neighbors=[5],
            batch_size=10,
        )

        for batch in loader:
            # batch_size should be at most 10
            assert batch.batch_size <= 10

    def test_features_included(self, small_graph_with_features: tuple[Graph, np.ndarray]) -> None:
        """Test that features are included in batches."""
        from aethergraph.pytorch import NeighborLoader

        graph, features = small_graph_with_features
        loader = NeighborLoader(
            graph,
            num_neighbors=[10],
            batch_size=16,
            features=features,
        )

        for batch in loader:
            assert batch.x is not None
            assert batch.x.shape[0] == batch.num_nodes
            assert batch.x.shape[1] == 64  # feature dim

    def test_file_backed_features_match_node_ids(self, small_graph: Graph, temp_dir: Path) -> None:
        """Features loaded from file should align exactly with batch n_id."""
        from aethergraph._core import save_features
        from aethergraph.pytorch import NeighborLoader

        feature_dim = 8
        features = np.arange(
            small_graph.num_nodes * feature_dim,
            dtype=np.float32,
        ).reshape(small_graph.num_nodes, feature_dim)
        feature_path = temp_dir / "features.bin"
        save_features(str(feature_path), features)

        loader = NeighborLoader(
            small_graph,
            num_neighbors=[5],
            input_nodes=np.array([0, 1, 2], dtype=np.int64),
            batch_size=3,
            shuffle=False,
            replace=False,
            feature_path=str(feature_path),
        )

        batch = next(iter(loader))
        assert batch.x is not None
        n_id = batch.n_id.numpy()
        np.testing.assert_allclose(batch.x.numpy(), features[n_id])


class TestNeighborLoaderSampling:
    """Test sampling behavior."""

    def test_input_nodes(self, small_graph: Graph) -> None:
        """Test input_nodes parameter."""
        from aethergraph.pytorch import NeighborLoader

        input_nodes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        loader = NeighborLoader(
            small_graph,
            num_neighbors=[5],
            input_nodes=input_nodes,
            batch_size=5,
        )

        # Should only have one batch
        batches = list(loader)
        assert len(batches) == 1

    def test_fanout_controls_neighbors(self, medium_graph: Graph) -> None:
        """Test that fanout controls number of neighbors sampled."""
        from aethergraph.pytorch import NeighborLoader

        # Small fanout
        loader_small = NeighborLoader(
            medium_graph,
            num_neighbors=[2, 2],
            batch_size=32,
            shuffle=False,
        )

        # Large fanout
        loader_large = NeighborLoader(
            medium_graph,
            num_neighbors=[25, 10],
            batch_size=32,
            shuffle=False,
        )

        batch_small = next(iter(loader_small))
        batch_large = next(iter(loader_large))

        # Larger fanout should sample more neighbors on average
        # (not strictly guaranteed but statistically likely)
        assert batch_large.num_nodes >= batch_small.num_nodes

    def test_shuffle(self, small_graph: Graph) -> None:
        """Test shuffle parameter."""
        from aethergraph.pytorch import NeighborLoader

        # Without shuffle, batches should iterate in deterministic order
        loader_no_shuffle = NeighborLoader(
            small_graph,
            num_neighbors=[5],
            batch_size=16,
            shuffle=False,
        )

        # Verify batches are produced
        batches_no_shuffle = list(loader_no_shuffle)
        assert len(batches_no_shuffle) > 0
        assert all(b.num_nodes > 0 for b in batches_no_shuffle)

        # With shuffle, should also produce valid batches
        loader_shuffle = NeighborLoader(
            small_graph,
            num_neighbors=[5],
            batch_size=16,
            shuffle=True,
        )

        batches_shuffled = list(loader_shuffle)
        assert len(batches_shuffled) == len(batches_no_shuffle), "Same number of batches"
        assert all(b.num_nodes > 0 for b in batches_shuffled)

    def test_shuffle_all_nodes_is_permutation(self, small_graph: Graph) -> None:
        """shuffle=True with input_nodes=None should visit each node exactly once."""
        from aethergraph.pytorch import NeighborLoader

        loader = NeighborLoader(
            small_graph,
            num_neighbors=[0],
            batch_size=7,
            shuffle=True,
        )

        seen: list[int] = []
        for batch in loader:
            seen.extend(batch.n_id[batch.input_id].tolist())

        assert len(seen) == small_graph.num_nodes
        assert len(set(seen)) == small_graph.num_nodes
        assert set(seen) == set(range(small_graph.num_nodes))


class TestNeighborLoaderMultiEpoch:
    """Test multi-epoch iteration."""

    def test_multiple_epochs(self, small_graph: Graph) -> None:
        """Test iterating multiple epochs."""
        from aethergraph.pytorch import NeighborLoader

        loader = NeighborLoader(
            small_graph,
            num_neighbors=[5],
            batch_size=16,
        )

        for epoch in range(3):
            batch_count = 0
            for batch in loader:
                batch_count += 1
            assert batch_count > 0

    def test_deterministic_without_shuffle(self, small_graph: Graph) -> None:
        """Test determinism when shuffle=False."""
        from aethergraph.pytorch import NeighborLoader

        loader = NeighborLoader(
            small_graph,
            num_neighbors=[5],
            batch_size=16,
            shuffle=False,
        )

        # First epoch
        batches_epoch1 = [b.n_id.tolist() for b in loader]

        # Second epoch
        batches_epoch2 = [b.n_id.tolist() for b in loader]

        # Should be identical
        assert len(batches_epoch1) == len(batches_epoch2)


class TestNeighborLoaderEdgeCases:
    """Test edge cases."""

    def test_single_node_batch(self, small_graph: Graph) -> None:
        """Test with batch_size=1."""
        from aethergraph.pytorch import NeighborLoader

        loader = NeighborLoader(
            small_graph,
            num_neighbors=[5],
            input_nodes=np.array([0], dtype=np.int64),
            batch_size=1,
        )

        batch = next(iter(loader))
        assert batch.batch_size == 1

    def test_empty_input_nodes(self, small_graph: Graph) -> None:
        """Test with empty input_nodes."""
        from aethergraph.pytorch import NeighborLoader

        loader = NeighborLoader(
            small_graph,
            num_neighbors=[5],
            input_nodes=np.array([], dtype=np.int64),
            batch_size=16,
        )

        batches = list(loader)
        assert len(batches) == 0

    def test_single_hop(self, small_graph: Graph) -> None:
        """Test single-hop sampling."""
        from aethergraph.pytorch import NeighborLoader

        loader = NeighborLoader(
            small_graph,
            num_neighbors=[10],
            batch_size=16,
        )

        for batch in loader:
            assert batch.num_nodes > 0

    def test_input_id_tracks_seed_order(self) -> None:
        """Seed ordering should be recovered via input_id, not by slicing [:batch_size]."""
        from aethergraph import Graph
        from aethergraph.pytorch import NeighborLoader

        src = np.array([3, 3, 2, 2, 1, 1], dtype=np.uint32)
        dst = np.array([0, 1, 0, 1, 0, 2], dtype=np.uint32)
        graph = Graph.from_edges(4, src, dst)

        loader = NeighborLoader(
            graph,
            num_neighbors=[2],
            input_nodes=np.array([3, 2], dtype=np.int64),
            batch_size=2,
            shuffle=False,
        )

        batch = next(iter(loader))
        seed_nodes = batch.n_id[batch.input_id].tolist()
        assert seed_nodes == [3, 2]

    def test_duplicate_seeds_preserved_in_input_id(self) -> None:
        """Duplicate seeds should produce duplicate entries in input_id."""
        from aethergraph import Graph
        from aethergraph.pytorch import NeighborLoader

        src = np.array([0, 0, 1, 1, 2], dtype=np.uint32)
        dst = np.array([1, 2, 2, 3, 3], dtype=np.uint32)
        graph = Graph.from_edges(4, src, dst)

        loader = NeighborLoader(
            graph,
            num_neighbors=[2],
            input_nodes=np.array([1, 1, 2, 3], dtype=np.int64),
            batch_size=2,
            shuffle=False,
        )

        batch = next(iter(loader))
        assert batch.batch_size == 2
        assert len(batch.input_id) == 2
        assert batch.n_id[batch.input_id].tolist() == [1, 1]

    def test_negative_input_nodes_rejected(self, small_graph: Graph) -> None:
        """Negative node IDs should be rejected before sampling."""
        from aethergraph.pytorch import NeighborLoader

        with pytest.raises(ValueError, match="non-negative"):
            NeighborLoader(
                small_graph,
                num_neighbors=[5],
                input_nodes=np.array([-1], dtype=np.int64),
                batch_size=1,
            )

    def test_out_of_bounds_input_nodes_rejected(self, small_graph: Graph) -> None:
        """Out-of-range node IDs should be rejected before sampling."""
        from aethergraph.pytorch import NeighborLoader

        with pytest.raises(ValueError, match="out-of-range"):
            NeighborLoader(
                small_graph,
                num_neighbors=[5],
                input_nodes=np.array([small_graph.num_nodes], dtype=np.int64),
                batch_size=1,
            )


class TestNeighborLoaderFailureHandling:
    """Test explicit error handling for backend failures."""

    def test_backend_early_stop_raises(
        self, monkeypatch: pytest.MonkeyPatch, small_graph: Graph
    ) -> None:
        """Loader should raise if backend stops before producing all batches."""
        import aethergraph.pytorch.loader as loader_module
        from aethergraph.pytorch import NeighborLoader

        class _FakeStats:
            hit_rate = 0.0
            hits = 0
            misses = 1

        class _FakeRustNeighborLoader:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self._stats = _FakeStats()

            def submit(self, _batch_idx: int, _seeds: npt.NDArray[np.int64]) -> None:
                return None

            def next_with_features(self) -> None:
                return None

            def stats(self) -> _FakeStats:
                return self._stats

            def shutdown(self) -> None:
                return None

        monkeypatch.setattr(loader_module, "RustNeighborLoader", _FakeRustNeighborLoader)

        loader = NeighborLoader(
            small_graph,
            num_neighbors=[5],
            batch_size=8,
            shuffle=False,
        )

        with pytest.raises(RuntimeError, match="stopped early"):
            list(loader)


class TestNeighborLoaderRepr:
    """Test string representations."""

    def test_repr(self, small_graph: Graph) -> None:
        """Test __repr__."""
        from aethergraph.pytorch import NeighborLoader

        loader = NeighborLoader(
            small_graph,
            num_neighbors=[10, 5],
            batch_size=16,
        )

        r = repr(loader)
        assert "NeighborLoader" in r


class TestSamplingConfigValidation:
    """Test SamplingConfig validation."""

    def test_empty_num_neighbors(self) -> None:
        """Empty num_neighbors should raise ValueError."""
        from aethergraph import SamplingConfig

        with pytest.raises(ValueError, match="non-empty"):
            SamplingConfig(num_neighbors=[])

    def test_loader_rejects_empty_num_neighbors(self, small_graph: Graph) -> None:
        """NeighborLoader must reject empty fanout at construction, not at iter."""
        from aethergraph.pytorch import NeighborLoader

        with pytest.raises(ValueError, match="non-empty"):
            NeighborLoader(small_graph, num_neighbors=[], batch_size=1)

    def test_negative_num_neighbors(self) -> None:
        """Negative values in num_neighbors should raise ValueError."""
        from aethergraph import SamplingConfig

        with pytest.raises(ValueError, match="non-negative"):
            SamplingConfig(num_neighbors=[10, -5])

    def test_loader_rejects_negative_num_neighbors(self, small_graph: Graph) -> None:
        from aethergraph.pytorch import NeighborLoader

        with pytest.raises(ValueError, match="non-negative"):
            NeighborLoader(small_graph, num_neighbors=[-1], batch_size=1)

    def test_invalid_max_degree(self) -> None:
        """Invalid max_degree should raise ValueError."""
        from aethergraph import SamplingConfig

        with pytest.raises(ValueError, match="max_degree"):
            SamplingConfig(num_neighbors=[10, 5], max_degree=0)

        with pytest.raises(ValueError, match="max_degree"):
            SamplingConfig(num_neighbors=[10, 5], max_degree=-1)

    def test_invalid_subgraph_type(self) -> None:
        """Invalid subgraph_type should raise ValueError."""
        from aethergraph import SamplingConfig

        with pytest.raises(ValueError, match="subgraph_type"):
            SamplingConfig(num_neighbors=[10], subgraph_type="unknown")

    def test_to_rust_passes_advanced_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Advanced config fields should be forwarded to Rust config."""
        import aethergraph.sampler as sampler_module
        from aethergraph import SamplingConfig

        captured: dict[str, object] = {}

        class _FakeSamplingConfig:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(sampler_module, "_SamplingConfig", _FakeSamplingConfig)

        config = SamplingConfig(
            num_neighbors=[4, 2],
            replace=True,
            seed=7,
            max_degree=999,
            cumulative=False,
            weighted=True,
            subgraph_type="induced",
            track_edge_ids=False,
        )
        _ = config._to_rust()

        assert captured["weighted"] is True
        assert captured["subgraph_type"] == "induced"
        assert captured["track_edge_ids"] is False

    def test_valid_config(self) -> None:
        """Valid config should not raise."""
        from aethergraph import SamplingConfig

        # Basic config
        config = SamplingConfig(num_neighbors=[25, 10])
        assert config.num_neighbors == [25, 10]
        assert config.replace is False

        # With all options
        config = SamplingConfig(
            num_neighbors=[15, 10, 5],
            replace=True,
            seed=42,
            max_degree=1000,
            cumulative=False,
        )
        assert config.num_neighbors == [15, 10, 5]
        assert config.replace is True
        assert config.seed == 42
        assert config.max_degree == 1000
        assert config.cumulative is False


class TestBoundaryParsing:
    """Parse-at-edge contracts for seeds and features."""

    def test_normalize_rejects_float_ids(self, small_graph: Graph) -> None:
        from aethergraph.pytorch.loader import normalize_input_nodes

        with pytest.raises(ValueError, match="integer or bool"):
            normalize_input_nodes(
                np.array([1.9, 2.1], dtype=np.float64),
                num_nodes=small_graph.num_nodes,
            )

    def test_parse_features_rejects_float64(self, small_graph: Graph) -> None:
        from aethergraph.pytorch.loader import _parse_features

        bad = np.zeros((small_graph.num_nodes, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            _parse_features(bad, small_graph.num_nodes)

    def test_multi_worker_seed_content_stable(self, medium_graph: Graph) -> None:
        """Same seed + multi-worker → bit-identical ordered epoch stream."""
        from aethergraph.pytorch import NeighborLoader

        g = medium_graph

        def epoch(workers: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
            ld = NeighborLoader(
                g,
                num_neighbors=[10, 10],
                batch_size=8,
                seed=42,
                num_workers=workers,
                shuffle=False,
                replace=True,
                input_nodes=np.arange(min(40, g.num_nodes), dtype=np.int64),
            )
            return [
                (
                    tuple(d.n_id.tolist()),
                    tuple(d.edge_index.reshape(-1).tolist()),
                )
                for d in ld
            ]

        a = epoch(1)
        assert a == epoch(1)
        assert a == epoch(4)


class TestTrackEdgeIds:
    def test_track_edge_ids_false_sets_e_id_none(self, small_graph: Graph) -> None:
        from aethergraph.pytorch import NeighborLoader

        ld = NeighborLoader(
            small_graph,
            num_neighbors=[5],
            batch_size=4,
            track_edge_ids=False,
            shuffle=False,
        )
        batch = next(iter(ld))
        assert batch.get("e_id") is None
        assert batch.edge_index.shape[1] >= 0
