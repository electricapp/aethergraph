"""High-level Ray Dataset API for distributed graph sampling.

This module provides convenience functions for creating Ray Datasets from the
AetherGraphDatasource, simplifying distributed GNN training workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

try:
    import ray
    import ray.data
except ImportError as e:
    raise ImportError(
        "Ray Data integration requires ray[data]>=2.9. Install with: pip install aethergraph[ray]"
    ) from e

from aethergraph.ray.datasource import AetherGraphDatasource

if TYPE_CHECKING:
    import torch
    from torch_geometric.data import Data  # type: ignore[import-untyped]

__all__ = ["collate_to_pyg", "create_sampling_dataset"]


def collate_to_pyg(batch: dict[str, Any]) -> Data:
    """Convert a Ray batch to a PyG Data object.

    This function is designed to run on GPU workers just before the forward
    pass. It reconstructs the PyG Data object from the batch format returned
    by Ray's iter_batches().

    Arrays are copied during conversion because Ray iter_batches may return
    non-writable numpy arrays, and PyTorch requires writable backing arrays
    for tensor creation.

    Args:
        batch: Dictionary from Ray dataset iteration containing exactly one row.
            Expected keys: edge_index_0, edge_index_1, n_id, e_id, batch_size.
            Optional: x, x_shape_0, x_shape_1.

    Returns:
        torch_geometric.data.Data object ready for model.forward().

    Raises:
        ImportError: If torch or torch_geometric are not installed.
        ValueError: If the input batch contains multiple rows.

    Example:
        >>> for batch in dataset.iter_batches(batch_size=1):
        ...     data = collate_to_pyg(batch)
        ...     out = model(data.x, data.edge_index)
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as e:
        raise ImportError("collate_to_pyg requires torch and torch_geometric") from e

    num_rows = len(batch["batch_size"])
    if num_rows != 1:
        raise ValueError(
            "collate_to_pyg expects exactly one row. "
            f"Got {num_rows}; iterate with dataset.iter_batches(batch_size=1)."
        )

    edge_src = np.array(batch["edge_index_0"][0], dtype=np.int64, copy=True)
    edge_dst = np.array(batch["edge_index_1"][0], dtype=np.int64, copy=True)
    n_id_arr = np.array(batch["n_id"][0], dtype=np.int64, copy=True)
    e_id_arr = np.array(batch["e_id"][0], dtype=np.int64, copy=True)

    edge_index = torch.stack([torch.from_numpy(edge_src), torch.from_numpy(edge_dst)], dim=0)
    n_id = torch.from_numpy(n_id_arr)
    e_id = torch.from_numpy(e_id_arr)

    x: torch.Tensor | None = None
    if "x" in batch and len(batch["x"][0]) > 0:
        x_flat = np.array(batch["x"][0], dtype=np.float32, copy=True)
        shape_0 = int(batch["x_shape_0"][0])
        shape_1 = int(batch["x_shape_1"][0])
        x = torch.from_numpy(x_flat.reshape(shape_0, shape_1))

    return Data(
        x=x,
        edge_index=edge_index,
        n_id=n_id,
        e_id=e_id,
        batch_size=int(batch["batch_size"][0]),
    )


def create_sampling_dataset(
    graph_path: str | Path,
    num_neighbors: list[int],
    input_nodes: list[int] | npt.NDArray[Any] | None = None,
    batch_size: int = 1024,
    replace: bool = False,
    features_path: str | Path | None = None,
    parallelism: int | None = None,
) -> ray.data.Dataset:
    """Create a Ray Dataset for distributed graph sampling.

    Each Ray worker loads the graph via memory-mapping and runs the existing
    NeighborLoader to sample its partition of seed nodes. No network shuffle
    is required for graph data.

    Args:
        graph_path: Path to the binary graph file.
        num_neighbors: Number of neighbors per hop. For example, [25, 10]
            samples 25 neighbors at hop 1 and 10 at hop 2.
        input_nodes: Seed nodes to sample from. If None, uses all nodes.
        batch_size: Number of seed nodes per batch.
        replace: Whether to sample with replacement.
        features_path: Optional path to features file.
        parallelism: Number of parallel tasks. If None, uses available CPUs.

    Returns:
        Ray Dataset yielding sampled subgraphs as dictionaries.

    Example:
        >>> import ray
        >>> from aethergraph.ray import create_sampling_dataset
        >>>
        >>> ray.init()
        >>> dataset = create_sampling_dataset(
        ...     graph_path="reddit_graph.bin",
        ...     features_path="features.bin",
        ...     num_neighbors=[25, 10],
        ...     input_nodes=train_nodes,
        ...     batch_size=1024,
        ... )
        >>>
        >>> for batch in dataset.iter_rows():
        ...     train_step(batch)
    """
    datasource = AetherGraphDatasource(
        graph_path=str(graph_path),
        num_neighbors=num_neighbors,
        input_nodes=input_nodes,
        batch_size=batch_size,
        replace=replace,
        features_path=str(features_path) if features_path else None,
    )

    num_blocks = parallelism or int(ray.available_resources().get("CPU", 1))
    result: ray.data.Dataset = ray.data.read_datasource(datasource, override_num_blocks=num_blocks)
    return result
