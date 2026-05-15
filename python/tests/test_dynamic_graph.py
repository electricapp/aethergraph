"""Tests for the DynamicGraph class."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from aethergraph import DynamicGraph


class TestCreateEmpty:
    """Test empty graph construction."""

    def test_basic(self) -> None:
        """Create an empty graph and verify properties."""
        g = DynamicGraph(num_vertices=100)
        assert g.num_vertices == 100
        assert g.num_edges == 0
        assert g.arena_capacity > 0
        assert g.arena_used == 0

    def test_custom_arena(self) -> None:
        """Arena size should match requested MB."""
        g = DynamicGraph(num_vertices=10, arena_mb=64)
        assert g.arena_capacity == 64 * 1024 * 1024

    def test_len(self) -> None:
        """len() returns num_vertices."""
        g = DynamicGraph(num_vertices=42)
        assert len(g) == 42


class TestInsertAndQuery:
    """Test edge insertion and querying."""

    def test_insert_single(self) -> None:
        """Insert a single edge and verify all query methods."""
        g = DynamicGraph(num_vertices=100)
        assert g.insert_edge(0, 10) is True
        assert g.num_edges == 1
        assert g.degree(0) == 1
        assert g.has_edge(0, 10) is True
        assert g.has_edge(0, 11) is False

    def test_insert_multiple(self) -> None:
        """Insert multiple edges to the same source."""
        g = DynamicGraph(num_vertices=100)
        g.insert_edge(0, 10)
        g.insert_edge(0, 20)
        g.insert_edge(0, 5)

        assert g.degree(0) == 3
        assert g.has_edge(0, 10)
        assert g.has_edge(0, 20)
        assert g.has_edge(0, 5)

        neighbors = g.neighbors(0)
        np.testing.assert_array_equal(neighbors, [5, 10, 20])

    def test_insert_different_sources(self) -> None:
        """Insert edges from different sources."""
        g = DynamicGraph(num_vertices=100)
        g.insert_edge(0, 10)
        g.insert_edge(0, 20)
        g.insert_edge(1, 30)
        g.insert_edge(1, 40)
        g.insert_edge(1, 50)

        assert g.degree(0) == 2
        assert g.degree(1) == 3
        assert g.degree(2) == 0
        assert g.num_edges == 5

    def test_empty_neighbors(self) -> None:
        """Neighbors of isolated vertex is empty array."""
        g = DynamicGraph(num_vertices=100)
        neighbors = g.neighbors(50)
        assert len(neighbors) == 0
        assert neighbors.dtype == np.int64


class TestInsertDuplicate:
    """Test duplicate edge handling."""

    def test_duplicate_returns_false(self) -> None:
        """Duplicate edge returns False."""
        g = DynamicGraph(num_vertices=100)
        assert g.insert_edge(0, 10) is True
        assert g.insert_edge(0, 10) is False
        assert g.degree(0) == 1
        assert g.num_edges == 1

    def test_batch_with_duplicates(self) -> None:
        """Batch insert skips duplicates and returns correct count."""
        g = DynamicGraph(num_vertices=100)
        src = np.array([0, 0, 0, 0], dtype=np.uint32)
        dst = np.array([10, 20, 10, 30], dtype=np.uint32)
        count = g.insert_edges(src, dst)
        assert count == 3  # 10 is duplicate
        assert g.num_edges == 3


class TestBatchInsert:
    """Test batch edge insertion from numpy arrays."""

    def test_basic(self) -> None:
        """Batch insert from numpy arrays."""
        g = DynamicGraph(num_vertices=100)
        src = np.array([0, 0, 1, 2], dtype=np.uint32)
        dst = np.array([1, 2, 2, 0], dtype=np.uint32)
        count = g.insert_edges(src, dst)
        assert count == 4
        assert g.num_edges == 4

    def test_dtype_conversion(self) -> None:
        """Arrays should accept int64 and convert to uint32."""
        g = DynamicGraph(num_vertices=100)
        src = np.array([0, 1, 2], dtype=np.int64)
        dst = np.array([1, 2, 0], dtype=np.int64)
        count = g.insert_edges(src, dst)
        assert count == 3

    def test_length_mismatch(self) -> None:
        """Mismatched array lengths should raise ValueError."""
        g = DynamicGraph(num_vertices=100)
        src = np.array([0, 1], dtype=np.uint32)
        dst = np.array([1, 2, 3], dtype=np.uint32)
        with pytest.raises(ValueError, match="same length"):
            g.insert_edges(src, dst)

    def test_empty_arrays(self) -> None:
        """Empty arrays should be a no-op."""
        g = DynamicGraph(num_vertices=100)
        src = np.array([], dtype=np.uint32)
        dst = np.array([], dtype=np.uint32)
        count = g.insert_edges(src, dst)
        assert count == 0
        assert g.num_edges == 0


class TestFromEdges:
    """Test static constructor from edge arrays."""

    def test_basic(self) -> None:
        """Build graph from edge arrays."""
        src = np.array([0, 0, 1, 2], dtype=np.uint32)
        dst = np.array([1, 2, 2, 0], dtype=np.uint32)
        g = DynamicGraph.from_edges(5, src, dst)

        assert g.num_vertices == 5
        assert g.num_edges == 4
        assert g.degree(0) == 2
        assert g.has_edge(0, 1)
        assert g.has_edge(2, 0)

    def test_from_edges_dtype_conversion(self) -> None:
        """from_edges should accept int64 arrays."""
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([1, 0], dtype=np.int64)
        g = DynamicGraph.from_edges(2, src, dst)
        assert g.num_edges == 2

    def test_from_edges_length_mismatch(self) -> None:
        """from_edges should reject mismatched array lengths."""
        src = np.array([0, 1], dtype=np.uint32)
        dst = np.array([1], dtype=np.uint32)
        with pytest.raises(ValueError, match="same length"):
            DynamicGraph.from_edges(2, src, dst)


class TestNeighborsSorted:
    """Test that neighbors are always returned in sorted order."""

    def test_sorted_small(self) -> None:
        """Neighbors are sorted for small degree."""
        g = DynamicGraph(num_vertices=100)
        for dst in [50, 10, 30, 20, 40]:
            g.insert_edge(0, dst)
        neighbors = g.neighbors(0)
        np.testing.assert_array_equal(neighbors, [10, 20, 30, 40, 50])

    def test_sorted_medium(self) -> None:
        """Neighbors are sorted for medium degree (multiple chunks)."""
        g = DynamicGraph(num_vertices=1000)
        rng = np.random.default_rng(42)
        dsts = rng.choice(999, size=100, replace=False).astype(np.uint32) + 1
        for d in dsts:
            g.insert_edge(0, int(d))

        neighbors = g.neighbors(0)
        assert len(neighbors) == 100
        # Verify sorted
        for i in range(len(neighbors) - 1):
            assert neighbors[i] < neighbors[i + 1]


class TestHighDegree:
    """Test with high-degree vertices (triggers multi-chunk C-trees)."""

    def test_500_neighbors(self) -> None:
        """Insert 500 edges to one vertex."""
        g = DynamicGraph(num_vertices=1000, arena_mb=4)
        for i in range(500):
            assert g.insert_edge(0, i) is True
        assert g.degree(0) == 500

        neighbors = g.neighbors(0)
        assert len(neighbors) == 500
        assert neighbors.dtype == np.int64
        # Verify sorted
        for i in range(len(neighbors) - 1):
            assert neighbors[i] < neighbors[i + 1]

    def test_arena_usage_grows(self) -> None:
        """Arena usage should increase with inserts."""
        g = DynamicGraph(num_vertices=1000, arena_mb=4)
        initial = g.arena_used
        for i in range(100):
            g.insert_edge(0, i)
        assert g.arena_used > initial


class TestConcurrentReadWrite:
    """Test concurrent read/write from multiple threads."""

    def test_reader_sees_consistent_snapshot(self) -> None:
        """Reader always sees sorted, consistent neighbor list."""
        g = DynamicGraph(num_vertices=2000, arena_mb=16)
        errors: list[str] = []
        stop = threading.Event()

        def writer() -> None:
            for i in range(1000):
                g.insert_edge(0, i)
            stop.set()

        def reader() -> None:
            while not stop.is_set():
                neighbors = g.neighbors(0)
                # Must always be sorted
                for j in range(len(neighbors) - 1):
                    if neighbors[j] >= neighbors[j + 1]:
                        errors.append(
                            f"unsorted at index {j}: {neighbors[j]} >= {neighbors[j + 1]}"
                        )
                        return

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)
        reader_thread.start()
        writer_thread.start()
        writer_thread.join()
        reader_thread.join()

        assert errors == [], f"Reader saw inconsistency: {errors}"
        assert g.degree(0) == 1000


class TestRepr:
    """Test string representations."""

    def test_repr(self) -> None:
        """__repr__ contains key info."""
        g = DynamicGraph(num_vertices=100, arena_mb=1)
        r = repr(g)
        assert "DynamicGraph" in r
        assert "100" in r

    def test_str(self) -> None:
        """__str__ is human-readable."""
        g = DynamicGraph(num_vertices=100)
        s = str(g)
        assert "vertices" in s
        assert "edges" in s


class TestSnapshot:
    """Test DynamicGraph.snapshot() producing a static Graph."""

    def test_snapshot_returns_graph(self) -> None:
        """snapshot() produces a Graph with correct edges."""
        from aethergraph import Graph

        g = DynamicGraph(num_vertices=5)
        g.insert_edge(0, 1)
        g.insert_edge(0, 2)
        g.insert_edge(0, 3)
        g.insert_edge(1, 0)
        g.insert_edge(1, 2)
        g.insert_edge(2, 0)

        snap = g.snapshot()
        assert isinstance(snap, Graph)
        assert snap.num_nodes == 5
        assert snap.num_edges == 6

        # Check degrees match
        assert snap.degree(0) == 3
        assert snap.degree(1) == 2
        assert snap.degree(2) == 1
        assert snap.degree(3) == 0
        assert snap.degree(4) == 0

        # Check neighbor lists
        np.testing.assert_array_equal(sorted(snap.neighbors(0)), [1, 2, 3])
        np.testing.assert_array_equal(sorted(snap.neighbors(1)), [0, 2])
        np.testing.assert_array_equal(sorted(snap.neighbors(2)), [0])

    def test_snapshot_empty_graph(self) -> None:
        """snapshot() works on a graph with no edges."""
        from aethergraph import Graph

        g = DynamicGraph(num_vertices=10)
        snap = g.snapshot()
        assert isinstance(snap, Graph)
        assert snap.num_nodes == 10
        assert snap.num_edges == 0
        assert snap.degree(0) == 0

    def test_snapshot_is_frozen(self) -> None:
        """Inserting edges after snapshot doesn't affect the snapshot."""
        g = DynamicGraph(num_vertices=10)
        g.insert_edge(0, 1)

        snap = g.snapshot()
        assert snap.num_edges == 1
        assert snap.degree(0) == 1

        # Insert more edges into the dynamic graph
        g.insert_edge(0, 2)
        g.insert_edge(0, 3)

        # Snapshot should be unchanged
        assert snap.num_edges == 1
        assert snap.degree(0) == 1

        # New snapshot should reflect the new edges
        snap2 = g.snapshot()
        assert snap2.num_edges == 3
        assert snap2.degree(0) == 3

    def test_snapshot_high_degree(self) -> None:
        """snapshot() handles high-degree vertices (multi-chunk C-trees)."""
        g = DynamicGraph(num_vertices=1000, arena_mb=4)
        for i in range(1, 500):
            g.insert_edge(0, i)

        snap = g.snapshot()
        assert snap.degree(0) == 499
        assert snap.num_edges == 499


@pytest.mark.requires_torch
class TestNeighborLoaderWithDynamicGraph:
    """Test NeighborLoader accepts DynamicGraph and snapshots per epoch."""

    def test_loader_accepts_dynamic_graph(self) -> None:
        """NeighborLoader.__init__ accepts a DynamicGraph."""
        from aethergraph.pytorch import NeighborLoader

        g = DynamicGraph(num_vertices=100)
        for i in range(1, 20):
            g.insert_edge(0, i)
            g.insert_edge(i, 0)

        loader = NeighborLoader(
            g,
            num_neighbors=[5],
            batch_size=10,
            shuffle=False,
        )
        assert len(loader) > 0

    def test_loader_yields_batches(self) -> None:
        """NeighborLoader iterates and produces PyG Data batches."""
        from aethergraph.pytorch import NeighborLoader

        # Build a small connected graph: star with node 0 at center
        g = DynamicGraph(num_vertices=50)
        for i in range(1, 50):
            g.insert_edge(0, i)
            g.insert_edge(i, 0)

        loader = NeighborLoader(
            g,
            num_neighbors=[10],
            batch_size=10,
            shuffle=False,
        )

        batches = list(loader)
        assert len(batches) > 0

        # Each batch should have the expected PyG attributes
        batch = batches[0]
        assert hasattr(batch, "edge_index")
        assert hasattr(batch, "n_id")
        assert batch.num_nodes > 0

    def test_snapshot_refreshes_per_epoch(self) -> None:
        """Each __iter__ call re-snapshots, picking up new edges."""
        from aethergraph.pytorch import NeighborLoader

        g = DynamicGraph(num_vertices=50)
        # Start with a simple graph
        for i in range(1, 10):
            g.insert_edge(0, i)
            g.insert_edge(i, 0)

        loader = NeighborLoader(
            g,
            num_neighbors=[5],
            input_nodes=[0],
            batch_size=1,
            shuffle=False,
        )

        # First epoch: node 0 has degree 9
        batches_1 = list(loader)
        assert len(batches_1) == 1

        # Add more edges
        for i in range(10, 30):
            g.insert_edge(0, i)
            g.insert_edge(i, 0)

        # Second epoch: re-snapshot picks up new edges
        batches_2 = list(loader)
        assert len(batches_2) == 1
        # The subgraph in epoch 2 should potentially include more neighbors
        # (since node 0 now has degree 29 instead of 9)
        # We verify that iteration works and the loader didn't error out.
        assert batches_2[0].num_nodes > 0


class TestWalDurability:
    """WAL-backed DynamicGraph: open, append, replay round-trip."""

    def test_open_creates_fresh_wal(self, tmp_path: Path) -> None:
        """A fresh WAL path produces an empty graph at epoch 0."""
        wal = tmp_path / "graph.wal"
        g = DynamicGraph.open_with_wal(wal, num_vertices=100)
        assert g.num_vertices == 100
        assert g.num_edges == 0
        assert g.current_epoch == 0
        assert wal.exists(), "WAL header is written on open"

    def test_inserts_replay_after_reopen(self, tmp_path: Path) -> None:
        """Edges inserted before close are recovered on the next open."""
        wal = tmp_path / "graph.wal"
        g = DynamicGraph.open_with_wal(wal, num_vertices=100)
        for src, dst in [(0, 1), (0, 2), (3, 7)]:
            g.insert_edge(src, dst)
        del g

        g2 = DynamicGraph.open_with_wal(wal, num_vertices=100)
        assert g2.num_edges == 3
        assert g2.has_edge(0, 1)
        assert g2.has_edge(0, 2)
        assert g2.has_edge(3, 7)
        assert not g2.has_edge(1, 0)
        # Three writer-guard commits during replay → epoch advances three times.
        assert g2.current_epoch == 3

    def test_current_epoch_advances_on_commit(self, tmp_path: Path) -> None:
        """Each writer-guard close (one per `insert_edge`) bumps the epoch."""
        wal = tmp_path / "graph.wal"
        g = DynamicGraph.open_with_wal(wal, num_vertices=10)
        start = g.current_epoch
        g.insert_edge(0, 1)
        g.insert_edge(0, 2)
        assert g.current_epoch == start + 2

    def test_bad_magic_raises_value_error(self, tmp_path: Path) -> None:
        """Existing file without WAL magic surfaces as ValueError."""
        wal = tmp_path / "graph.wal"
        wal.write_bytes(b"NOT_A_WAL_FILE\0\0\0\0\0\0\0\0\0\0\0\0")
        with pytest.raises(ValueError, match="bad magic"):
            DynamicGraph.open_with_wal(wal, num_vertices=10)
