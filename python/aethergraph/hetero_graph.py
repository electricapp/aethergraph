"""High-level Python API for heterogeneous graph operations.

This module provides a Pythonic wrapper around the Rust HeteroCsrGraph
implementation for multi-relational graph neural network training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from aethergraph._core import HeteroCsrGraph

__all__ = ["HeteroGraph"]


class HeteroGraph:
    """Heterogeneous graph with multiple node and edge types.

    Thin Python builder over the Rust :class:`HeteroCsrGraph`. Adds three
    things the Rust class doesn't provide:

    1. ``from_pyg`` — converts a ``torch_geometric.data.HeteroData`` object,
       which can't live on the Rust side (importing torch from a Rust
       extension is a bad idea).
    2. Tuple-friendly ``num_edges((src, rel, dst))`` — the Rust API takes
       three positional args, which is less Pythonic.
    3. ``@property`` accessors for ``node_types`` / ``edge_types`` /
       ``total_nodes`` / ``total_edges`` — the Rust calls are methods.

    Features (per-type ``x`` arrays) are NOT held on this class; pass them
    to :class:`HeteroNeighborLoader` directly via its ``features=`` kwarg,
    matching the homogeneous-graph pattern.

    Example:
        >>> graph = HeteroGraph.from_edge_arrays(
        ...     node_types={"user": 1000, "post": 5000},
        ...     edge_types=[
        ...         ("user", "votes", "post", src_ids, dst_ids),
        ...         ("post", "belongs_to", "subreddit", src_ids, dst_ids),
        ...     ],
        ... )
        >>> print(graph.node_types)
        ['post', 'user']
        >>> print(graph.num_nodes("user"))
        1000
    """

    _inner: HeteroCsrGraph

    def __init__(self, inner: HeteroCsrGraph) -> None:
        """Initialize a HeteroGraph from a HeteroCsrGraph instance.

        Args:
            inner: The underlying Rust HeteroCsrGraph instance.
        """
        self._inner = inner

    @classmethod
    def from_edge_arrays(
        cls,
        node_types: dict[str, int],
        edge_types: list[tuple[str, str, str, npt.NDArray[Any], npt.NDArray[Any]]],
    ) -> HeteroGraph:
        """Create a heterogeneous graph from edge arrays.

        Args:
            node_types: Dict mapping node type name to node count.
                Example: {"user": 1000, "post": 5000}
            edge_types: List of (src_type, relation, dst_type, src_array, dst_array)
                tuples. Arrays are converted to uint32.
                Example: [("user", "votes", "post", src_ids, dst_ids)]

        Returns:
            A HeteroGraph instance.

        Raises:
            ValueError: If arrays have mismatched lengths or node types are unknown.

        Example:
            >>> src = np.array([0, 0, 1], dtype=np.uint32)
            >>> dst = np.array([0, 1, 2], dtype=np.uint32)
            >>> graph = HeteroGraph.from_edge_arrays(
            ...     node_types={"user": 3, "post": 5},
            ...     edge_types=[("user", "votes", "post", src, dst)],
            ... )
        """
        converted: list[tuple[str, str, str, npt.NDArray[np.uint32], npt.NDArray[np.uint32]]] = []
        for src_type, rel, dst_type, src_arr, dst_arr in edge_types:
            src_np: npt.NDArray[np.uint32] = np.asarray(src_arr, dtype=np.uint32)
            dst_np: npt.NDArray[np.uint32] = np.asarray(dst_arr, dtype=np.uint32)
            converted.append((src_type, rel, dst_type, src_np, dst_np))

        inner = HeteroCsrGraph.from_edge_arrays(node_types, converted)
        return cls(inner)

    @classmethod
    def from_pyg(cls, data: Any) -> HeteroGraph:
        """Create a HeteroGraph from a PyTorch Geometric HeteroData object.

        Args:
            data: A torch_geometric.data.HeteroData instance.

        Returns:
            A HeteroGraph instance.

        Raises:
            ImportError: If torch_geometric is not installed.
            TypeError: If data is not a HeteroData instance.

        Example:
            >>> from torch_geometric.data import HeteroData
            >>> pyg_data = HeteroData()
            >>> pyg_data["user"].num_nodes = 100
            >>> pyg_data["post"].num_nodes = 200
            >>> pyg_data["user", "votes", "post"].edge_index = edge_index
            >>> graph = HeteroGraph.from_pyg(pyg_data)
        """
        try:
            from torch_geometric.data import HeteroData
        except ImportError as e:
            raise ImportError(
                "HeteroGraph.from_pyg requires torch-geometric>=2.4. "
                "Install with: pip install aethergraph[pytorch-geometric]"
            ) from e

        if not isinstance(data, HeteroData):
            raise TypeError(f"expected HeteroData, got {type(data).__name__}")

        # Extract node types and counts
        node_types: dict[str, int] = {}
        for nt in data.node_types:
            store = data[nt]
            if hasattr(store, "num_nodes") and store.num_nodes is not None:
                node_types[nt] = int(store.num_nodes)
            elif hasattr(store, "x") and store.x is not None:
                node_types[nt] = store.x.size(0)
            else:
                raise ValueError(
                    f"cannot determine num_nodes for node type '{nt}'. "
                    f"Set data['{nt}'].num_nodes or provide data['{nt}'].x"
                )

        # Extract edge types
        edge_list: list[tuple[str, str, str, npt.NDArray[np.uint32], npt.NDArray[np.uint32]]] = []
        for et in data.edge_types:
            src_type, rel, dst_type = et
            store = data[et]
            ei = store.edge_index
            src_arr = ei[0].cpu().numpy().astype(np.uint32)
            dst_arr = ei[1].cpu().numpy().astype(np.uint32)
            edge_list.append((src_type, rel, dst_type, src_arr, dst_arr))

        return cls.from_edge_arrays(node_types, edge_list)

    @property
    def node_types(self) -> list[str]:
        """Names of all node types."""
        return self._inner.node_types()

    @property
    def edge_types(self) -> list[tuple[str, str, str]]:
        """All edge types as (src_type, relation, dst_type) triples."""
        return self._inner.edge_types()

    def num_nodes(self, node_type: str) -> int:
        """Number of nodes of a given type.

        Args:
            node_type: Node type name.

        Returns:
            Number of nodes.

        Raises:
            KeyError: If node type is unknown.
        """
        return self._inner.num_nodes(node_type)

    def num_edges(self, edge_type: tuple[str, str, str]) -> int:
        """Number of edges of a given edge type.

        Args:
            edge_type: (src_type, relation, dst_type) triple.

        Returns:
            Number of edges.

        Raises:
            KeyError: If edge type is unknown.
        """
        src, rel, dst = edge_type
        return self._inner.num_edges(src, rel, dst)

    @property
    def total_nodes(self) -> int:
        """Total number of nodes across all types."""
        return self._inner.total_nodes()

    @property
    def total_edges(self) -> int:
        """Total number of edges across all edge types."""
        return self._inner.total_edges()

    @property
    def csr(self) -> HeteroCsrGraph:
        """The underlying Rust :class:`HeteroCsrGraph`.

        Pass this to :class:`HeteroNeighborSampler` or
        :class:`HeteroNeighborLoader` constructors when they expect the
        Rust type directly.
        """
        return self._inner

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"HeteroGraph(node_types={self.node_types}, "
            f"edge_types={self.edge_types}, "
            f"total_nodes={self.total_nodes}, "
            f"total_edges={self.total_edges})"
        )

    def __str__(self) -> str:
        """Return short string description."""
        return (
            f"HeteroGraph with {len(self.node_types)} node types, "
            f"{len(self.edge_types)} edge types, "
            f"{self.total_nodes:,} nodes, {self.total_edges:,} edges"
        )
