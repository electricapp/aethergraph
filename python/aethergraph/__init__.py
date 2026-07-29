"""AetherGraph: High-Performance Graph Sampling for GNN Training.

A Rust-powered library for billion-scale graph neural network training.
Provides memory-mapped graph loading, high-performance neighborhood sampling,
and seamless PyTorch integration.

Example:
    >>> from aethergraph import Graph
    >>> from aethergraph.pytorch import NeighborLoader
    >>>
    >>> graph = Graph.load("graph.bin")
    >>> loader = NeighborLoader(
    ...     graph,
    ...     num_neighbors=[15, 10],
    ...     batch_size=128,
    ...     feature_path="features.bin",
    ... )
    >>> for data in loader:
    ...     out = model(data.x, data.edge_index)
"""

from aethergraph._core import (
    ArrowConversionError,
    CacheError,
    GraphLoadError,
    MetricsSnapshot,
    SamplingError,
    __version__,
    save_features,
)
from aethergraph.dynamic_graph import DynamicGraph
from aethergraph.graph import Graph
from aethergraph.hetero_graph import HeteroGraph
from aethergraph.sampler import (
    NeighborSampler,
    ParallelBatchSampler,
    SampledSubgraph,
    Sampler,
    SamplingConfig,
)

__all__ = [
    "ArrowConversionError",
    "CacheError",
    "DynamicGraph",
    "Graph",
    "GraphLoadError",
    "HeteroGraph",
    "MetricsSnapshot",
    "NeighborSampler",  # PyG-compatible alias
    "ParallelBatchSampler",
    "SampledSubgraph",
    "Sampler",
    "SamplingConfig",
    "SamplingError",
    "__version__",
    "save_features",
]
