"""High-level Python API for graph sampling.

This module provides Pythonic wrappers around the Rust sampling implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from aethergraph._core import (
    NeighborSampler as _NeighborSampler,
    ParallelBatchSampler as _ParallelBatchSampler,
    SampledSubgraph as _SampledSubgraph,
    SamplingConfig as _SamplingConfig,
    SamplingError,
)
from aethergraph.graph import Graph

__all__ = [
    "SamplingConfig",
    "SampledSubgraph",
    "Sampler",
    "NeighborSampler",  # PyG-compatible alias
    "ParallelBatchSampler",
    "SamplingError",
]

SeedInput = Sequence[int] | npt.NDArray[np.integer[Any]]


@dataclass
class SamplingConfig:
    """Configuration for neighborhood sampling.

    Attributes:
        num_neighbors: Number of neighbors to sample per node at each hop.
            For example, [25, 10] samples 25 neighbors at hop 1 and 10 at hop 2.
        replace: Whether to sample with replacement.
        seed: Random seed for reproducibility. None uses random seed.
        max_degree: Maximum degree to process for hub nodes. None means no limit.
        cumulative: Whether to use cumulative sampling (PyG-style). When True,
            samples from all nodes seen so far at each hop (larger subgraphs).
            When False, samples only from new frontier at each hop (smaller).
        weighted: Whether to use edge weights for weighted sampling.
        subgraph_type: Type of subgraph to extract.
            Must be one of {"directional", "induced", "bidirectional"}.
        track_edge_ids: Whether to track original edge IDs in sampled subgraph.

    Example:
        >>> config = SamplingConfig(num_neighbors=[25, 10], replace=False)
        >>> config = SamplingConfig(num_neighbors=[15, 10, 5], replace=True, seed=42)
        >>> config = SamplingConfig(num_neighbors=[25, 10], max_degree=10000)
        >>> config = SamplingConfig(num_neighbors=[25, 10], cumulative=False)
    """

    num_neighbors: list[int]
    replace: bool = False
    seed: int | None = None
    max_degree: int | None = None
    cumulative: bool = True
    weighted: bool = False
    subgraph_type: str = "directional"
    track_edge_ids: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If num_neighbors is empty, contains negative values,
                or max_degree is non-positive when specified.
        """
        if not self.num_neighbors:
            raise ValueError("num_neighbors must be a non-empty list")
        if any(n < 0 for n in self.num_neighbors):
            raise ValueError(f"num_neighbors values must be non-negative, got {self.num_neighbors}")
        if self.max_degree is not None and self.max_degree <= 0:
            raise ValueError(f"max_degree must be > 0 if specified, got {self.max_degree}")
        if self.subgraph_type not in {"directional", "induced", "bidirectional"}:
            raise ValueError(
                "subgraph_type must be one of {'directional', 'induced', 'bidirectional'}, "
                f"got '{self.subgraph_type}'"
            )

    def _to_rust(self) -> _SamplingConfig:
        """Convert to Rust SamplingConfig.

        Returns:
            Rust SamplingConfig instance with equivalent settings.
        """
        return _SamplingConfig(
            num_neighbors=self.num_neighbors,
            replace=self.replace,
            seed=self.seed,
            max_degree=self.max_degree,
            cumulative=self.cumulative,
            weighted=self.weighted,
            subgraph_type=self.subgraph_type,
            track_edge_ids=self.track_edge_ids,
        )


class SampledSubgraph:
    """A sampled subgraph containing nodes and edges.

    This is a lightweight wrapper around the Rust SampledSubgraph that provides
    convenient access to the sampled data.

    Attributes:
        _inner: The underlying Rust SampledSubgraph instance.

    Example:
        >>> sampler = Sampler(graph, SamplingConfig(num_neighbors=[10, 5]))
        >>> subgraph = sampler.sample([0, 1, 2])
        >>> print(f"Sampled {subgraph.num_nodes} nodes and {subgraph.num_edges} edges")
        >>> edge_index = subgraph.edge_index
    """

    _inner: _SampledSubgraph

    def __init__(self, inner: _SampledSubgraph) -> None:
        """Initialize a SampledSubgraph from a Rust SampledSubgraph.

        Args:
            inner: The underlying Rust SampledSubgraph instance.
        """
        self._inner = inner

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the subgraph."""
        return self._inner.num_nodes()

    @property
    def num_edges(self) -> int:
        """Number of edges in the subgraph."""
        return self._inner.num_edges()

    @property
    def num_seeds(self) -> int:
        """Number of seed nodes."""
        return self._inner.num_seeds()

    @property
    def nodes(self) -> npt.NDArray[np.uint32]:
        """All node IDs in the subgraph (sorted).

        Returns:
            Numpy array of node IDs with dtype=uint32.
        """
        return self._inner.nodes()

    @property
    def seeds(self) -> npt.NDArray[np.uint32]:
        """Original seed node IDs.

        Returns:
            Numpy array of seed node IDs with dtype=uint32.
        """
        return self._inner.seeds()

    @property
    def edge_index(self) -> npt.NDArray[np.int64]:
        """Edge index in PyTorch Geometric COO format with global node IDs.

        Returns:
            Numpy array with shape [2, num_edges] and dtype=int64.
            Row 0 contains source node IDs (global).
            Row 1 contains destination node IDs (global).
        """
        return self._inner.edge_index()

    @property
    def edge_index_local(self) -> npt.NDArray[np.int64]:
        """Edge index with local (remapped) node IDs in [0, num_nodes).

        Local IDs are pre-computed in Rust using O(E log N) binary search,
        avoiding Python dict/list overhead.

        Returns:
            Numpy array with shape [2, num_edges] and dtype=int64.
            Row 0 contains source node IDs (local, 0-indexed).
            Row 1 contains destination node IDs (local, 0-indexed).
        """
        return self._inner.edge_index_local()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary containing num_nodes, num_edges, nodes, seeds, and
            edge_index keys.
        """
        return self._inner.to_dict()

    def to_arrow(self) -> Any:
        """Convert to Apache Arrow RecordBatch for zero-copy data transfer.

        Requires pyarrow to be installed.

        Returns:
            pyarrow.RecordBatch with columns edge_src (source node IDs),
            edge_dst (destination node IDs), and nodes (all node IDs in
            subgraph).

        Example:
            >>> import pyarrow as pa
            >>> subgraph = sampler.sample([0, 1, 2])
            >>> batch = subgraph.to_arrow()
            >>> print(batch.schema)
        """
        return self._inner.to_arrow()

    def __repr__(self) -> str:
        """Return string representation of the subgraph."""
        return (
            f"SampledSubgraph(num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, "
            f"num_seeds={self.num_seeds})"
        )

    def __str__(self) -> str:
        """Return short string description of the subgraph."""
        return (
            f"Subgraph with {self.num_nodes:,} nodes, "
            f"{self.num_edges:,} edges (from {self.num_seeds} seeds)"
        )


class Sampler:
    """High-level neighborhood sampler for GNN training.

    This sampler implements the neighborhood sampling strategy used in GraphSAGE
    and similar GNN architectures. It wraps the high-performance Rust
    implementation.

    Attributes:
        graph: The graph to sample from.
        config: The sampling configuration.
        _inner: The underlying Rust NeighborSampler instance.

    Example:
        >>> from aethergraph import Graph, Sampler, SamplingConfig
        >>> graph = Graph.load("graph.bin")
        >>> config = SamplingConfig(num_neighbors=[25, 10], replace=False, seed=42)
        >>> sampler = Sampler(graph, config)
        >>> subgraph = sampler.sample([0, 100, 200, 300])
        >>> print(f"Sampled {subgraph.num_nodes} nodes")
        >>> edge_index = subgraph.edge_index
    """

    graph: Graph
    config: SamplingConfig
    _inner: _NeighborSampler

    def __init__(self, graph: Graph, config: SamplingConfig) -> None:
        """Initialize a neighborhood sampler.

        Args:
            graph: Graph to sample from.
            config: Sampling configuration specifying num_neighbors, replace,
                and seed parameters.

        Example:
            >>> graph = Graph.load("graph.bin")
            >>> sampler = Sampler(graph, SamplingConfig(num_neighbors=[10, 5]))
        """
        self.graph = graph
        self.config = config
        self._inner = _NeighborSampler(
            graph._rust_graph,
            config._to_rust(),
        )

    def sample(self, seeds: SeedInput) -> SampledSubgraph:
        """Sample k-hop neighborhoods for a batch of seed nodes.

        Args:
            seeds: List of seed node IDs to sample from.

        Returns:
            SampledSubgraph containing the sampled nodes and edges.

        Example:
            >>> sampler = Sampler(graph, SamplingConfig(num_neighbors=[25, 10]))
            >>> subgraph = sampler.sample([0, 1, 2, 3])
            >>> print(f"Sampled {subgraph.num_nodes} nodes from 4 seeds")
        """
        seed_list = seeds.tolist() if isinstance(seeds, np.ndarray) else list(seeds)
        rust_subgraph = self._inner.sample(seed_list)
        return SampledSubgraph(rust_subgraph)

    def __repr__(self) -> str:
        """Return string representation of the sampler."""
        return f"Sampler(num_neighbors={self.config.num_neighbors}, replace={self.config.replace})"


# PyG-compatible alias
NeighborSampler = Sampler


class ParallelBatchSampler:
    """Parallel batch sampler for high-throughput GNN training.

    This sampler uses Rayon (Rust's parallel iterator library) to sample
    multiple subgraphs in parallel, maximizing CPU utilization during data
    loading.

    Attributes:
        graph: The graph to sample from.
        config: The sampling configuration.
        _inner: The underlying Rust ParallelBatchSampler instance.

    Example:
        >>> graph = Graph.load("graph.bin")
        >>> config = SamplingConfig(num_neighbors=[25, 10])
        >>> sampler = ParallelBatchSampler(graph, config)
        >>> batches = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        >>> subgraphs = sampler.sample_batches(batches)
        >>> print(f"Sampled {len(subgraphs)} subgraphs in parallel")
    """

    graph: Graph
    config: SamplingConfig
    _inner: _ParallelBatchSampler

    def __init__(self, graph: Graph, config: SamplingConfig) -> None:
        """Initialize a parallel batch sampler.

        Args:
            graph: Graph to sample from.
            config: Sampling configuration specifying num_neighbors, replace,
                and seed parameters.
        """
        self.graph = graph
        self.config = config
        self._inner = _ParallelBatchSampler(
            graph._rust_graph,
            config._to_rust(),
        )

    def sample_batches(self, batches: list[list[int]]) -> list[SampledSubgraph]:
        """Sample neighborhoods for multiple batches in parallel.

        Args:
            batches: List of seed node ID lists, one per batch.

        Returns:
            List of SampledSubgraphs, one per input batch.

        Example:
            >>> sampler = ParallelBatchSampler(graph, config)
            >>> batches = [[0, 1], [2, 3], [4, 5]]
            >>> subgraphs = sampler.sample_batches(batches)
            >>> for i, subgraph in enumerate(subgraphs):
            ...     print(f"Batch {i}: {subgraph.num_nodes} nodes")
        """
        rust_subgraphs = self._inner.sample_batches(batches)
        return [SampledSubgraph(sg) for sg in rust_subgraphs]

    def __repr__(self) -> str:
        """Return string representation of the sampler."""
        return (
            f"ParallelBatchSampler(num_neighbors={self.config.num_neighbors}, "
            f"replace={self.config.replace})"
        )
