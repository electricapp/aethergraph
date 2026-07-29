"""
PyTorch integration for AetherGraph.

Usage:
    >>> from aethergraph import Graph
    >>> from aethergraph.pytorch import NeighborLoader
    >>>
    >>> graph = Graph.load("graph.bin")
    >>> graph.load_features("features.bin")  # optional
    >>>
    >>> loader = NeighborLoader(graph, num_neighbors=[15, 10], batch_size=128)
    >>> for data in loader:
    ...     out = model(data.x, data.edge_index)
"""

from aethergraph.pytorch.hetero_loader import HeteroNeighborLoader
from aethergraph.pytorch.loader import NeighborLoader

__all__ = ["HeteroNeighborLoader", "NeighborLoader"]
