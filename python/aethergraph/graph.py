"""High-level Python API for graph operations.

This module provides a Pythonic wrapper around the Rust CsrGraph implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from aethergraph._core import CsrGraph, GraphLoadError

if TYPE_CHECKING:
    import torch


class Graph:
    """High-level graph interface with Pythonic API.

    This class wraps the low-level CsrGraph Rust implementation and provides
    additional convenience methods and better error messages.

    Attributes:
        _inner: The underlying Rust CsrGraph instance.
        _features: In-memory node features array, if set directly.
        _feature_path: Path to feature file, if loaded via load_features().

    Example:
        >>> graph = Graph.load("reddit_graph.bin")
        >>> print(f"Graph has {graph.num_nodes} nodes and {graph.num_edges} edges")
        >>> neighbors = graph.neighbors(0)
        >>> print(f"Node 0 has {len(neighbors)} neighbors")
    """

    _inner: CsrGraph
    _features: npt.NDArray[np.float32] | None
    _feature_path: str | None

    def __init__(self, inner: CsrGraph) -> None:
        """Initialize a Graph from a CsrGraph instance.

        Args:
            inner: The underlying Rust CsrGraph instance.
        """
        self._inner = inner
        self._features = None
        self._feature_path = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        validation: Literal["auto", "header_only", "offsets_only", "full"] = "auto",
    ) -> Graph:
        """Load a graph from a binary file created by AetherGraph CLI.

        Args:
            path: Path to the binary graph file.
            validation: Validation mode. ``"auto"`` selects ``"full"`` for small
                files and ``"offsets_only"`` for large files.

        Returns:
            A Graph instance backed by mmap storage.

        Raises:
            GraphLoadError: If the file cannot be loaded or is invalid.

        Example:
            >>> graph = Graph.load("my_graph.bin")
            >>> graph = Graph.load(Path("/data/graphs/reddit.bin"))
        """
        path_str = str(path)
        try:
            inner = CsrGraph.load(path_str, validation)
            return cls(inner)
        except GraphLoadError as e:
            raise GraphLoadError(f"Failed to load graph from {path}: {e}") from e

    @classmethod
    def load_mmap(
        cls,
        path: str | Path,
        *,
        validation: Literal["header_only", "offsets_only", "full"] = "offsets_only",
    ) -> Graph:
        """Load a graph with explicit mmap-backed storage.

        Args:
            path: Path to the binary graph file.
            validation: Validation mode for load-time checks.
        """
        path_str = str(path)
        try:
            inner = CsrGraph.load_mmap(path_str, validation)
            return cls(inner)
        except GraphLoadError as e:
            raise GraphLoadError(f"Failed to mmap-load graph from {path}: {e}") from e

    @classmethod
    def load_owned(
        cls,
        path: str | Path,
        *,
        validation: Literal["auto", "header_only", "offsets_only", "full"] = "full",
    ) -> Graph:
        """Load a graph into owned in-memory storage.

        Args:
            path: Path to the binary graph file.
            validation: Validation mode. ``"auto"`` mirrors ``Graph.load`` policy.
        """
        path_str = str(path)
        try:
            inner = CsrGraph.load_owned(path_str, validation)
            return cls(inner)
        except GraphLoadError as e:
            raise GraphLoadError(f"Failed to owned-load graph from {path}: {e}") from e

    @classmethod
    def from_edges(
        cls,
        num_nodes: int,
        src: npt.NDArray[Any],
        dst: npt.NDArray[Any],
    ) -> Graph:
        """Create a graph from edge arrays.

        Validates that num_nodes is positive, arrays have matching lengths,
        and all node IDs are within bounds. Arrays are converted to uint32.

        Args:
            num_nodes: Number of nodes in the graph. Must be greater than 0.
            src: Source node array. Will be converted to dtype=uint32.
            dst: Destination node array. Will be converted to dtype=uint32.

        Returns:
            A Graph instance containing the constructed graph.

        Raises:
            ValueError: If num_nodes <= 0, arrays have different lengths,
                or node IDs are out of bounds.

        Example:
            >>> src = np.array([0, 0, 1], dtype=np.uint32)
            >>> dst = np.array([1, 2, 2], dtype=np.uint32)
            >>> graph = Graph.from_edges(3, src, dst)
        """
        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be > 0, got {num_nodes}")

        src_arr: npt.NDArray[np.uint32] = np.asarray(src, dtype=np.uint32)
        dst_arr: npt.NDArray[np.uint32] = np.asarray(dst, dtype=np.uint32)

        if len(src_arr) != len(dst_arr):
            raise ValueError(
                f"src and dst arrays must have same length, got {len(src_arr)} and {len(dst_arr)}"
            )

        if len(src_arr) > 0:
            max_src = int(src_arr.max())
            max_dst = int(dst_arr.max())
            if max_src >= num_nodes:
                raise ValueError(f"src contains node ID {max_src} >= num_nodes ({num_nodes})")
            if max_dst >= num_nodes:
                raise ValueError(f"dst contains node ID {max_dst} >= num_nodes ({num_nodes})")

        inner = CsrGraph.from_edges(num_nodes, src_arr, dst_arr)
        return cls(inner)

    def save(self, path: str | Path) -> None:
        """Save graph to a binary file.

        Args:
            path: Path to save the binary graph file.

        Example:
            >>> graph.save("my_graph.bin")
        """
        self._inner.save(str(path))

    def load_features(self, path: str | Path) -> None:
        """Load node features from a binary file.

        Features are loaded lazily by NeighborLoader using the most efficient
        strategy for your platform (io_uring on Linux, mmap elsewhere). This
        method only stores the path; actual loading happens during sampling.

        Args:
            path: Path to feature file in AETHFEAT format.

        Raises:
            FileNotFoundError: If the feature file does not exist.

        Example:
            >>> graph = Graph.load("graph.bin")
            >>> graph.load_features("features.bin")
        """
        feature_path = Path(path)
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        self._feature_path = str(feature_path)

    @property
    def features(self) -> npt.NDArray[np.float32] | None:
        """Node features as a 2D numpy array.

        When setting features directly (not via load_features), they will be
        stored in memory and indexed during sampling. Features must be 2D with
        shape [num_nodes, feature_dim].

        Returns:
            Node features array with shape [num_nodes, feature_dim], or None
            if no features have been set.

        Example:
            >>> graph.features = np.random.randn(graph.num_nodes, 128).astype(np.float32)
        """
        return self._features

    @features.setter
    def features(self, value: npt.NDArray[Any] | torch.Tensor) -> None:
        """Set node features.

        Converts torch.Tensor to numpy and ensures dtype is float32. Validates
        that the array is 2D with shape [num_nodes, feature_dim]. Clears any
        previously set feature_path.

        Args:
            value: Node features array or tensor with shape [num_nodes, feature_dim].

        Raises:
            ValueError: If array is not 2D or first dimension doesn't match num_nodes.
        """
        if hasattr(value, "numpy"):
            value = value.numpy()
        arr: npt.NDArray[np.float32] = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"features must be a 2D array with shape (num_nodes, feature_dim), "
                f"got {arr.ndim}D array with shape {arr.shape}"
            )
        if arr.shape[0] != self.num_nodes:
            raise ValueError(
                f"features first dimension must equal num_nodes={self.num_nodes}, "
                f"got shape {arr.shape}"
            )
        self._features = arr
        self._feature_path = None

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._inner.num_nodes()

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self._inner.num_edges()

    @property
    def has_weights(self) -> bool:
        """Whether the graph has edge weights."""
        return self._inner.has_weights()

    def degree(self, node: int) -> int:
        """Get the degree (number of outgoing edges) for a node.

        Args:
            node: Node ID to query.

        Returns:
            Number of outgoing edges from this node.

        Example:
            >>> graph = Graph.load("graph.bin")
            >>> print(f"Node 0 degree: {graph.degree(0)}")
        """
        return self._inner.degree(node)

    def neighbors(self, node: int) -> npt.NDArray[np.uint32]:
        """Get the neighbor IDs for a given node.

        This is a zero-copy operation. The numpy array views the underlying
        Rust data without duplication.

        Args:
            node: Node ID to query.

        Returns:
            Numpy array of neighbor node IDs with dtype=uint32.

        Example:
            >>> graph = Graph.load("graph.bin")
            >>> neighbors = graph.neighbors(0)
            >>> print(f"Node 0 has neighbors: {neighbors}")
        """
        return self._inner.neighbors(node)

    def batch_neighbors(self, nodes: list[int]) -> list[npt.NDArray[np.uint32]]:
        """Get neighbors for multiple nodes in a batch.

        Args:
            nodes: List of node IDs to query.

        Returns:
            List of numpy arrays, one per input node, each containing
            neighbor IDs with dtype=uint32.

        Example:
            >>> graph = Graph.load("graph.bin")
            >>> batch_neighbors = graph.batch_neighbors([0, 1, 2])
            >>> for i, neighbors in enumerate(batch_neighbors):
            ...     print(f"Node {i} has {len(neighbors)} neighbors")
        """
        return self._inner.batch_neighbors(nodes)

    def neighbor_weights(self, node: int) -> npt.NDArray[np.float32] | None:
        """Get edge weights for a node's neighbors, if available.

        Args:
            node: Node ID to query.

        Returns:
            Numpy array of edge weights with dtype=float32, or None if the
            graph is unweighted.

        Example:
            >>> graph = Graph.load("weighted_graph.bin")
            >>> weights = graph.neighbor_weights(0)
            >>> if weights is not None:
            ...     print(f"Edge weights: {weights}")
        """
        return self._inner.weights(node)

    def stats(self) -> dict[str, Any]:
        """Get statistics about the graph structure.

        Returns:
            Dictionary containing:
                - num_nodes: Number of nodes in the graph.
                - num_edges: Number of edges in the graph.
                - max_degree: Maximum node degree.
                - avg_degree: Average node degree.
                - has_weights: Whether the graph has edge weights.

        Example:
            >>> graph = Graph.load("graph.bin")
            >>> stats = graph.stats()
            >>> print(f"Average degree: {stats['avg_degree']:.2f}")
            >>> print(f"Max degree: {stats['max_degree']}")
        """
        return self._inner.stats()

    def __repr__(self) -> str:
        """Return string representation of the graph."""
        return (
            f"Graph(num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, "
            f"weighted={self.has_weights})"
        )

    def __str__(self) -> str:
        """Return short string description of the graph."""
        return f"Graph with {self.num_nodes:,} nodes and {self.num_edges:,} edges"

    def __len__(self) -> int:
        """Return number of nodes in the graph."""
        return self.num_nodes

    @property
    def feature_path(self) -> str | None:
        """Path to the feature file, if loaded via load_features().

        Returns:
            Feature file path, or None if features were set directly or
            not loaded.
        """
        return self._feature_path

    @property
    def _rust_graph(self) -> CsrGraph:
        """Access the underlying Rust CsrGraph.

        For internal use by other AetherGraph modules.

        Returns:
            The underlying CsrGraph instance.
        """
        return self._inner
