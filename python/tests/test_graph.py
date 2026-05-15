"""
Tests for the Graph class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aethergraph import Graph


class TestGraphCreation:
    """Test graph creation methods."""

    def test_from_edges_basic(self) -> None:
        """Create a simple graph from edge arrays."""
        src = np.array([0, 0, 1, 2], dtype=np.uint32)
        dst = np.array([1, 2, 2, 0], dtype=np.uint32)
        graph = Graph.from_edges(3, src, dst)

        assert graph.num_nodes == 3
        assert graph.num_edges == 4

    def test_from_edges_empty(self) -> None:
        """Create a graph with no edges."""
        src = np.array([], dtype=np.uint32)
        dst = np.array([], dtype=np.uint32)
        graph = Graph.from_edges(10, src, dst)

        assert graph.num_nodes == 10
        assert graph.num_edges == 0

    def test_from_edges_strict_dtype(self) -> None:
        """The Rust binding requires `uint32` arrays. Non-uint32 input must
        raise TypeError — callers convert at the call site."""
        src = np.array([0, 1, 2], dtype=np.int64)
        dst = np.array([1, 2, 0], dtype=np.int64)
        with pytest.raises(TypeError):
            # The intentional wrong dtype is the whole point of this test —
            # silence mypy so the strict check survives.
            Graph.from_edges(3, src, dst)  # type: ignore[arg-type]

    def test_from_edges_list_input(self) -> None:
        """Python lists should work as edge input."""
        src = [0, 1, 2]
        dst = [1, 2, 0]
        graph = Graph.from_edges(3, np.array(src, dtype=np.uint32), np.array(dst, dtype=np.uint32))

        assert graph.num_edges == 3

    def test_core_from_edges_rejects_length_mismatch(self) -> None:
        """Low-level binding should reject mismatched src/dst lengths."""
        from aethergraph._core import CsrGraph

        src = np.array([0, 1, 2], dtype=np.uint32)
        dst = np.array([1, 2], dtype=np.uint32)
        with pytest.raises(ValueError, match="same length"):
            CsrGraph.from_edges(3, src, dst)


class TestGraphPersistence:
    """Test save/load functionality."""

    def test_save_and_load(self, small_graph: Graph, temp_dir: Path) -> None:
        """Save and reload a graph."""
        path = temp_dir / "test_graph.bin"
        small_graph.save(path)

        loaded = Graph.load(path)
        assert loaded.num_nodes == small_graph.num_nodes
        assert loaded.num_edges == small_graph.num_edges

    def test_load_explicit_storage_modes(self, small_graph: Graph, temp_dir: Path) -> None:
        """Load using explicit storage modes via the unified `load(storage=...)` API."""
        path = temp_dir / "test_graph_modes.bin"
        small_graph.save(path)

        mmap_graph = Graph.load(path, storage="mmap", validation="offsets_only")
        owned_graph = Graph.load(path, storage="owned", validation="full")

        assert mmap_graph.num_nodes == small_graph.num_nodes
        assert mmap_graph.num_edges == small_graph.num_edges
        assert owned_graph.num_nodes == small_graph.num_nodes
        assert owned_graph.num_edges == small_graph.num_edges

    def test_invalid_validation_mode(self, small_graph: Graph, temp_dir: Path) -> None:
        """Invalid validation mode should raise ValueError."""
        path = temp_dir / "test_graph_invalid_validation.bin"
        small_graph.save(path)

        with pytest.raises(ValueError, match="Invalid validation mode"):
            Graph.load(path, validation="bad_mode")

    def test_load_nonexistent_file(self, temp_dir: Path) -> None:
        """Loading non-existent file should raise GraphLoadError."""
        from aethergraph import GraphLoadError

        with pytest.raises(GraphLoadError):
            Graph.load(temp_dir / "nonexistent.bin")


class TestGraphStructure:
    """Test graph structure queries."""

    def test_degree(self, small_graph: Graph) -> None:
        """Test degree queries."""
        for node in range(min(10, small_graph.num_nodes)):
            degree = small_graph.degree(node)
            neighbors = small_graph.neighbors(node)
            assert degree == len(neighbors)

    def test_neighbors(self, small_graph: Graph) -> None:
        """Test neighbor queries."""
        neighbors = small_graph.neighbors(0)
        assert isinstance(neighbors, np.ndarray)
        assert neighbors.dtype == np.uint32

    def test_batch_neighbors(self, small_graph: Graph) -> None:
        """Test batch neighbor queries."""
        nodes = [0, 1, 2, 3, 4]
        batch_result = small_graph.batch_neighbors(nodes)

        assert len(batch_result) == len(nodes)
        for i, node in enumerate(nodes):
            expected = small_graph.neighbors(node)
            np.testing.assert_array_equal(batch_result[i], expected)

    def test_stats(self, small_graph: Graph) -> None:
        """Test stats method."""
        stats = small_graph.stats()

        assert "num_nodes" in stats
        assert "num_edges" in stats
        assert "max_degree" in stats
        assert "avg_degree" in stats
        assert stats["num_nodes"] == small_graph.num_nodes
        assert stats["num_edges"] == small_graph.num_edges


# Feature handling lives on the loader, not the graph — see
# `tests/test_loader.py` for the constructor-arg shape.


class TestGraphRepr:
    """Test string representations."""

    def test_repr(self, small_graph: Graph) -> None:
        """Test __repr__."""
        r = repr(small_graph)
        assert "Graph" in r
        assert str(small_graph.num_nodes) in r

    def test_str(self, small_graph: Graph) -> None:
        """Test __str__."""
        s = str(small_graph)
        assert "nodes" in s
        assert "edges" in s

    def test_len(self, small_graph: Graph) -> None:
        """Test len(graph)."""
        assert len(small_graph) == small_graph.num_nodes
