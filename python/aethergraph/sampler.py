"""High-level Python API for graph sampling.

This module provides Pythonic wrappers around the Rust sampling implementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from aethergraph._core import (
    NeighborSampler as _NeighborSampler,
)
from aethergraph._core import (
    ParallelBatchSampler as _ParallelBatchSampler,
)
from aethergraph._core import (
    SampledSubgraph as _SampledSubgraph,
)
from aethergraph._core import (
    SamplingConfig as _SamplingConfig,
)
from aethergraph._core import (
    SamplingError,
    SamplingTelemetry,
)
from aethergraph.graph import Graph

__all__ = [
    "NeighborSampler",  # PyG-compatible alias
    "ParallelBatchSampler",
    "SampledSubgraph",
    "Sampler",
    "SamplingConfig",
    "SamplingError",
    "SamplingTelemetry",
]

SeedInput = Sequence[int] | npt.NDArray[np.int64] | npt.NDArray[np.uint32]


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
        temporal_strategy: ``"uniform"`` or ``"last"`` to enable temporal
            sampling; ``None`` disables it. Requires `Graph.set_timestamps`.
        disjoint: When ``True``, each seed produces its own isolated subgraph
            (no node dedup across seeds). The returned subgraph carries a
            ``batch`` vector mapping each node to its seed index.
        deterministic: When ``True``, sampling is bit-deterministic given a
            ``seed`` (at some throughput cost). When ``False``, output is
            statistically reproducible given a ``seed`` but the exact node
            ordering may vary across parallel runs.
        telemetry: Optional :class:`SamplingTelemetry` collector. When set,
            per-sample metrics are recorded into it; ``None`` disables
            collection.

    Example:
        >>> config = SamplingConfig(num_neighbors=[25, 10], replace=False)
        >>> config = SamplingConfig(num_neighbors=[15, 10, 5], replace=True, seed=42)
        >>> config = SamplingConfig(num_neighbors=[25, 10], max_degree=10000)
        >>> config = SamplingConfig(num_neighbors=[25, 10], cumulative=False)
        >>> config = SamplingConfig(num_neighbors=[25, 10], disjoint=True)
    """

    num_neighbors: list[int]
    replace: bool = False
    seed: int | None = None
    max_degree: int | None = None
    cumulative: bool = True
    weighted: bool = False
    subgraph_type: str = "directional"
    track_edge_ids: bool = True
    temporal_strategy: str | None = None
    disjoint: bool = False
    deterministic: bool = False
    telemetry: SamplingTelemetry | None = None

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
        if self.temporal_strategy is not None and self.temporal_strategy not in {
            "uniform",
            "last",
        }:
            raise ValueError(
                "temporal_strategy must be 'uniform', 'last', or None, "
                f"got '{self.temporal_strategy}'"
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
            temporal_strategy=self.temporal_strategy,
            disjoint=self.disjoint,
            deterministic=self.deterministic,
            telemetry=self.telemetry,
        )


class SampledSubgraph:
    """A sampled subgraph containing nodes and edges.

    Thin typed wrapper around the Rust SampledSubgraph; every data accessor is
    a property that forwards to the underlying Rust getter. The Rust class is
    also accessible directly via `aethergraph._core.SampledSubgraph` for code
    that prefers to avoid the wrapper.

    Example:
        >>> sampler = Sampler(graph, SamplingConfig(num_neighbors=[10, 5]))
        >>> subgraph = sampler.sample([0, 1, 2])
        >>> print(f"Sampled {subgraph.num_nodes} nodes and {subgraph.num_edges} edges")
        >>> edge_index = subgraph.edge_index
    """

    _inner: _SampledSubgraph

    def __init__(self, inner: _SampledSubgraph) -> None:
        self._inner = inner

    @property
    def num_nodes(self) -> int:
        """Number of unique nodes in the subgraph."""
        return self._inner.num_nodes

    @property
    def num_edges(self) -> int:
        """Number of edges in the subgraph."""
        return self._inner.num_edges

    @property
    def num_seeds(self) -> int:
        """Number of seed nodes used to produce this subgraph."""
        return self._inner.num_seeds

    @property
    def nodes(self) -> npt.NDArray[np.int64]:
        """All node IDs (dtype=int64 to match PyTorch's index dtype)."""
        return self._inner.nodes

    @property
    def seeds(self) -> npt.NDArray[np.int64]:
        """Seed node IDs (dtype=int64)."""
        return self._inner.seeds

    @property
    def edge_index(self) -> npt.NDArray[np.int64]:
        """Edge index `[2, num_edges]` with global node IDs (dtype=int64)."""
        return self._inner.edge_index

    @property
    def edge_index_local(self) -> npt.NDArray[np.int64]:
        """Edge index `[2, num_edges]` with local IDs in `[0, num_nodes)` (dtype=int64)."""
        return self._inner.edge_index_local

    @property
    def edge_ids(self) -> npt.NDArray[np.int64]:
        """Global edge IDs (positions in the CSR edges array, dtype=int64)."""
        return self._inner.edge_ids

    @property
    def seed_indices(self) -> npt.NDArray[np.int64]:
        """Indices of seeds inside `nodes` (dtype=int64)."""
        return self._inner.seed_indices

    @property
    def batch(self) -> npt.NDArray[np.int64] | None:
        """Per-node seed index (disjoint mode only), else `None`."""
        return self._inner.batch

    @property
    def num_sampled_nodes_per_hop(self) -> list[int]:
        """PyG-compatible: new nodes added at each hop."""
        return self._inner.num_sampled_nodes_per_hop

    @property
    def num_sampled_edges_per_hop(self) -> list[int]:
        """PyG-compatible: edges added at each hop."""
        return self._inner.num_sampled_edges_per_hop

    def to_dict(self) -> dict[str, Any]:
        """Dictionary view: `{num_nodes, num_edges, nodes, seeds, edge_index}`."""
        return self._inner.to_dict()

    def to_arrow(self) -> Any:
        """Convert to PyArrow `RecordBatch`es.

        Returns a `dict[str, pyarrow.RecordBatch]` with keys `"edges"`
        (length E, columns `edge_src` / `edge_dst` / `edge_id`), `"nodes"`
        (length N, column `nodes`), and `"seeds"` (length S, column `seeds`).
        Three batches rather than one because Arrow `RecordBatch` requires
        equal-length columns.

        Requires pyarrow.
        """
        return self._inner.to_arrow()

    def __repr__(self) -> str:
        return (
            f"SampledSubgraph(num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, "
            f"num_seeds={self.num_seeds})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        """`len(subgraph)` returns the node count (matches PyG convention)."""
        return self.num_nodes


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
            graph,
            config._to_rust(),
        )

    def sample(
        self,
        seeds: SeedInput,
        input_times: npt.NDArray[np.float64] | None = None,
    ) -> SampledSubgraph:
        """Sample k-hop neighborhoods for a batch of seed nodes.

        Args:
            seeds: Seed node IDs. Accepts numpy `uint32` arrays (zero-copy),
                numpy `int64` arrays (range-checked widening), or any Python
                sequence of ints. Dispatch happens at the FFI boundary, not
                here.
            input_times: Per-seed timestamps (`float64`), required when
                `config.temporal_strategy` is set.

        Returns:
            SampledSubgraph containing the sampled nodes and edges.

        Example:
            >>> sampler = Sampler(graph, SamplingConfig(num_neighbors=[25, 10]))
            >>> subgraph = sampler.sample([0, 1, 2, 3])
            >>> print(f"Sampled {subgraph.num_nodes} nodes from 4 seeds")
        """
        rust_subgraph = self._inner.sample(seeds, input_times=input_times)
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
            graph,
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
