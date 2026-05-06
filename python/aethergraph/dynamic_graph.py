"""Lock-free dynamic graph for live GNN training.

This module provides a Pythonic wrapper around the Rust DynamicGraph,
which uses C-tree neighbor lists with arena allocation for zero-alloc
reads and lock-free concurrent access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from aethergraph._core import DynamicGraph as _DynamicGraph

if TYPE_CHECKING:
    from aethergraph.graph import Graph


class DynamicGraph:
    """Lock-free dynamic graph with C-tree neighbor lists.

    Supports concurrent edge inserts and neighbor reads for live
    GNN training on evolving graphs (e.g., Reddit's social graph).

    Each vertex's neighbor list is a balanced tree of sorted,
    cache-line-sized chunks. 90% of nodes (degree < 15) fit in a
    single chunk -- one cache line read, identical cost to static CSR.

    Attributes:
        _inner: The underlying Rust DynamicGraph instance.

    Example:
        >>> g = DynamicGraph(num_vertices=1_000_000, arena_mb=512)
        >>> g.insert_edge(0, 42)
        True
        >>> g.degree(0)
        1
        >>> g.neighbors(0)
        array([42])
    """

    _inner: _DynamicGraph

    def __init__(self, num_vertices: int, arena_mb: int = 256) -> None:
        """Create an empty dynamic graph.

        Args:
            num_vertices: Number of vertices (fixed at construction).
            arena_mb: Arena capacity in megabytes. At ~80 bytes per edge,
                256 MB supports ~3.2M edges.
        """
        self._inner = _DynamicGraph(num_vertices=num_vertices, arena_mb=arena_mb)

    @classmethod
    def from_edges(
        cls,
        num_vertices: int,
        src: npt.NDArray[Any],
        dst: npt.NDArray[Any],
        arena_mb: int = 256,
    ) -> DynamicGraph:
        """Build a DynamicGraph from edge arrays.

        Args:
            num_vertices: Number of vertices.
            src: Source vertex array. Will be converted to uint32.
            dst: Destination vertex array. Will be converted to uint32.
            arena_mb: Arena capacity in megabytes.

        Returns:
            DynamicGraph with all edges inserted.

        Raises:
            ValueError: If src and dst have different lengths.
        """
        src_arr: npt.NDArray[np.uint32] = np.asarray(src, dtype=np.uint32)
        dst_arr: npt.NDArray[np.uint32] = np.asarray(dst, dtype=np.uint32)

        obj = cls.__new__(cls)
        obj._inner = _DynamicGraph.from_edges(num_vertices, src_arr, dst_arr, arena_mb)
        return obj

    def insert_edge(self, src: int, dst: int) -> bool:
        """Insert a directed edge from src to dst.

        Args:
            src: Source vertex ID.
            dst: Destination vertex ID.

        Returns:
            True if the edge was new, False if it already existed.

        Raises:
            RuntimeError: If the arena is full.
        """
        return self._inner.insert_edge(src, dst)

    def insert_edges(
        self, src: npt.NDArray[Any], dst: npt.NDArray[Any]
    ) -> int:
        """Batch-insert edges from arrays.

        Args:
            src: Source vertex array. Will be converted to uint32.
            dst: Destination vertex array. Will be converted to uint32.

        Returns:
            Number of new edges inserted (duplicates are skipped).

        Raises:
            ValueError: If src and dst have different lengths.
            RuntimeError: If the arena is full.
        """
        src_arr: npt.NDArray[np.uint32] = np.asarray(src, dtype=np.uint32)
        dst_arr: npt.NDArray[np.uint32] = np.asarray(dst, dtype=np.uint32)
        return self._inner.insert_edges(src_arr, dst_arr)

    @property
    def num_vertices(self) -> int:
        """Number of vertices (fixed at construction)."""
        return self._inner.num_vertices

    @property
    def num_edges(self) -> int:
        """Total number of edges."""
        return self._inner.num_edges

    @property
    def arena_used(self) -> int:
        """Arena bytes currently used."""
        return self._inner.arena_used

    @property
    def arena_capacity(self) -> int:
        """Arena total capacity in bytes."""
        return self._inner.arena_capacity

    def degree(self, vertex: int) -> int:
        """Get the degree (number of outgoing edges) for a vertex.

        Args:
            vertex: Vertex ID to query.

        Returns:
            Number of outgoing edges from this vertex.
        """
        return self._inner.degree(vertex)

    def has_edge(self, src: int, dst: int) -> bool:
        """Check whether edge (src -> dst) exists.

        Args:
            src: Source vertex ID.
            dst: Destination vertex ID.

        Returns:
            True if the edge exists.
        """
        return self._inner.has_edge(src, dst)

    def neighbors(self, vertex: int) -> npt.NDArray[np.int64]:
        """Get sorted neighbor array for a vertex.

        Returns int64 dtype for PyTorch compatibility.

        Args:
            vertex: Vertex ID to query.

        Returns:
            Sorted numpy array of neighbor IDs (dtype=int64).
        """
        return self._inner.neighbors(vertex)

    def snapshot(self) -> Graph:
        """Create a frozen CSR snapshot for use with NeighborLoader.

        Collects all edges from the C-tree neighbor lists into a static
        CSR graph. O(V + E) time -- call once per epoch, not per batch.

        The returned Graph is completely independent of this DynamicGraph.
        Edges inserted after the snapshot is taken will not appear in it.

        Returns:
            A static :class:`~aethergraph.graph.Graph` usable with
            :class:`~aethergraph.pytorch.loader.NeighborLoader` and all
            existing sampling infrastructure.

        Example:
            >>> dg = DynamicGraph(num_vertices=1000)
            >>> dg.insert_edge(0, 1)
            True
            >>> graph = dg.snapshot()
            >>> graph.num_nodes
            1000
            >>> graph.degree(0)
            1
        """
        from aethergraph.graph import Graph

        csr_graph = self._inner.snapshot()
        return Graph(csr_graph)

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"DynamicGraph(num_vertices={self.num_vertices}, "
            f"num_edges={self.num_edges}, "
            f"arena={self.arena_used // (1024 * 1024)}/{self.arena_capacity // (1024 * 1024)}MB)"
        )

    def __str__(self) -> str:
        """Return short string description."""
        return f"DynamicGraph with {self.num_vertices:,} vertices and {self.num_edges:,} edges"

    def __len__(self) -> int:
        """Return number of vertices."""
        return self.num_vertices
