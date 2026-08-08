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

from aethergraph.ray import _schema
from aethergraph.ray.datasource import AetherGraphDatasource

if TYPE_CHECKING:
    import torch
    from torch_geometric.data import Data

__all__ = ["collate_to_pyg", "create_sampling_dataset"]


def collate_to_pyg(batch: dict[str, Any]) -> Data:
    """Convert a Ray batch to a PyG Data object.

    This function is designed to run on GPU workers just before the forward
    pass. It reconstructs the PyG Data object from the batch format returned
    by Ray's iter_batches().

    Contract: the datasource emits one PyG batch per Ray row (each row's
    variable-length arrays live in single list cells). This collate therefore
    expects exactly one row and must be driven with
    ``dataset.iter_batches(batch_size=1)``; passing more than one row stacks
    unrelated subgraphs and is rejected.

    Arrays are copied only when Ray hands back read-only views (PyTorch
    requires writable backing arrays); writable arrays are shared with the
    resulting tensors zero-copy.

    Args:
        batch: Dictionary from Ray dataset iteration containing exactly one
            row, with the columns defined in :mod:`aethergraph.ray._schema`.

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

    num_rows = len(batch[_schema.BATCH_SIZE])
    if num_rows != 1:
        raise ValueError(
            "collate_to_pyg expects exactly one row: the datasource packs one "
            f"PyG batch per Ray row, so rows cannot be stacked. Got {num_rows} "
            "rows; iterate with dataset.iter_batches(batch_size=1)."
        )

    def _writable(arr: Any, dtype: np.dtype[Any]) -> npt.NDArray[Any]:
        # Ray may hand back read-only Arrow-backed views; torch needs
        # writable backing memory. Copy only when the view is read-only or
        # needs a dtype change — never unconditionally.
        out = np.asarray(arr, dtype=dtype)
        if not out.flags.writeable:
            out = out.copy()
        return out

    edge_src = batch[_schema.EDGE_SRC][0]
    edge_dst = batch[_schema.EDGE_DST][0]
    # One fresh (2, E) destination filled row-by-row: a single copy per row
    # from the (possibly read-only) source, with no torch.stack allocating
    # and copying a second [2, E] on top.
    num_edges = len(edge_src)
    ei = np.empty((2, num_edges), dtype=np.int64)
    ei[0] = edge_src
    ei[1] = edge_dst
    edge_index = torch.from_numpy(ei)

    n_id = torch.from_numpy(_writable(batch[_schema.N_ID][0], np.dtype(np.int64)))
    e_id = torch.from_numpy(_writable(batch[_schema.E_ID][0], np.dtype(np.int64)))

    x: torch.Tensor | None = None
    if _schema.X in batch:
        x_flat = _writable(batch[_schema.X][0], np.dtype(np.float32))
        rows = int(batch[_schema.X_ROWS][0])
        cols = int(batch[_schema.X_COLS][0])
        x = torch.from_numpy(x_flat.reshape(rows, cols))

    return Data(
        x=x,
        edge_index=edge_index,
        n_id=n_id,
        e_id=e_id,
        batch_size=int(batch[_schema.BATCH_SIZE][0]),
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
