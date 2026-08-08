"""PyTorch Geometric compatibility utilities.

This module provides functions to convert AetherGraph sampled subgraphs to
PyTorch Geometric Data objects for use with standard PyG models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch integration requires torch>=2.0. Install with: pip install aethergraph[torch]"
    ) from e

try:
    from torch_geometric.data import Data
except ImportError as e:
    raise ImportError(
        "PyTorch Geometric integration requires torch-geometric>=2.4. "
        "Install with: pip install aethergraph[pytorch-geometric]"
    ) from e

if TYPE_CHECKING:
    from aethergraph.sampler import SampledSubgraph

__all__ = ["batch_to_pyg", "subgraph_to_pyg", "to_edge_index"]


def to_edge_index(subgraph: SampledSubgraph) -> torch.Tensor:
    """Convert sampled subgraph edges to PyG edge_index format.

    Performs a zero-copy conversion from the numpy array to a PyTorch tensor.
    The numpy array is already int64, so torch shares the underlying memory.

    Args:
        subgraph: Sampled subgraph from AetherGraph containing edge data.

    Returns:
        PyTorch tensor with shape [2, num_edges] in COO format where row 0
        contains source node IDs and row 1 contains destination node IDs.

    Example:
        >>> sampler = Sampler(graph, SamplingConfig(num_neighbors=[10, 5]))
        >>> subgraph = sampler.sample([0, 1, 2])
        >>> edge_index = to_edge_index(subgraph)
        >>> print(edge_index.shape)
        torch.Size([2, 150])
    """
    return torch.from_numpy(subgraph.edge_index)


def subgraph_to_pyg(
    subgraph: SampledSubgraph,
    features: torch.Tensor | npt.NDArray[np.float32] | None = None,
    labels: torch.Tensor | npt.NDArray[np.int64] | None = None,
    remap_nodes: bool = True,
) -> Data:
    """Convert an AetherGraph SampledSubgraph to a PyTorch Geometric Data object.

    Creates a PyG Data object from a sampled subgraph, optionally including
    node features and labels. Node IDs can be remapped to local indices
    [0, num_nodes) for compatibility with PyG message passing layers.

    When remap_nodes is True (default), uses pre-computed local edge indices
    from Rust, which uses O(E log N) binary search and avoids Python dict
    overhead.

    Args:
        subgraph: Sampled subgraph from AetherGraph containing nodes and edges.
        features: Optional node features for **all nodes in the original
            graph**, as a torch.Tensor or numpy array. Rows are gathered by
            the subgraph's global node IDs. Callers holding features already
            gathered for this subgraph should assign ``data.x`` directly.
        labels: Optional node labels for all nodes in the original graph,
            gathered the same way as ``features``.
        remap_nodes: Whether to use local node IDs [0, num_nodes) in edge_index.
            When True, uses pre-computed local indices from Rust. When False,
            uses global node IDs.

    Returns:
        PyTorch Geometric Data object with attributes:
            - edge_index: Edge connectivity [2, num_edges]
            - num_nodes: Number of nodes in subgraph
            - node_ids: Original global node IDs [num_nodes]
            - x: Node features [num_nodes, feature_dim] if provided
            - y: Node labels [num_nodes] or [num_nodes, num_classes] if provided

    Example:
        >>> subgraph = sampler.sample([0, 1, 2])
        >>> data = subgraph_to_pyg(subgraph)
        >>>
        >>> features = torch.randn(graph.num_nodes, 128)
        >>> labels = torch.randint(0, 10, (graph.num_nodes,))
        >>> data = subgraph_to_pyg(subgraph, features=features, labels=labels)
        >>> print(data)
        Data(edge_index=[2, E], x=[N, 128], y=[N], node_ids=[N])
    """
    node_ids = torch.from_numpy(subgraph.nodes)
    num_nodes = len(node_ids)

    if remap_nodes:
        edge_index = torch.from_numpy(subgraph.edge_index_local)
    else:
        edge_index = torch.from_numpy(subgraph.edge_index)

    data = Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_ids=node_ids,
    )

    if features is not None:
        features_tensor = (
            torch.from_numpy(features) if isinstance(features, np.ndarray) else features
        )
        data.x = torch.index_select(features_tensor, 0, node_ids)

    if labels is not None:
        labels_tensor = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels
        data.y = torch.index_select(labels_tensor, 0, node_ids)

    return data


def batch_to_pyg(
    subgraphs: list[SampledSubgraph],
    features: torch.Tensor | npt.NDArray[np.float32] | None = None,
    labels: torch.Tensor | npt.NDArray[np.int64] | None = None,
) -> list[Data]:
    """Convert a batch of sampled subgraphs to PyG Data objects.

    Convenience function that applies subgraph_to_pyg to each subgraph in
    a batch. Useful when using ParallelBatchSampler for parallel sampling.

    Args:
        subgraphs: List of sampled subgraphs from ParallelBatchSampler.
        features: Optional node features for all nodes in the original graph.
            Will be indexed by node IDs for each subgraph.
        labels: Optional node labels for all nodes in the original graph.
            Will be indexed by node IDs for each subgraph.

    Returns:
        List of PyG Data objects, one per input subgraph.

    Example:
        >>> sampler = ParallelBatchSampler(graph, config)
        >>> batches = [[0, 1], [2, 3], [4, 5]]
        >>> subgraphs = sampler.sample_batches(batches)
        >>>
        >>> data_list = batch_to_pyg(subgraphs, features=features, labels=labels)
        >>> for data in data_list:
        ...     out = model(data.x, data.edge_index)
    """
    return [subgraph_to_pyg(sg, features=features, labels=labels) for sg in subgraphs]
