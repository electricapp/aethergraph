"""Ray Data custom datasource for distributed graph sampling.

This module implements a Ray Data datasource that uses the existing NeighborLoader
internally. Each Ray worker loads the graph via memory-mapping and runs the same
battle-tested NeighborLoader without code duplication.

The key insight: instead of partitioning the graph across workers (which incurs
massive network overhead), we replicate the graph topology on each worker's NVMe
and only distribute the seed node workload.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

try:
    from ray.data import Datasource
    from ray.data.block import BlockMetadata
    from ray.data.datasource import ReadTask
except ImportError as e:
    raise ImportError(
        "Ray Data integration requires ray[data]>=2.9. Install with: pip install aethergraph[ray]"
    ) from e

try:
    import pyarrow as pa  # type: ignore[import-untyped]
except ImportError as e:
    raise ImportError(
        "Ray Data integration requires pyarrow. Install with: pip install pyarrow"
    ) from e

if TYPE_CHECKING:
    from torch_geometric.data import Data

__all__ = ["AetherGraphDatasource"]


class AetherGraphDatasource(Datasource):
    """Ray Data datasource for distributed graph sampling.

    This datasource splits seed nodes across Ray workers, where each worker:
    1. Loads the graph via memory-mapping (instant, zero RAM overhead)
    2. Uses the existing NeighborLoader to sample k-hop neighborhoods
    3. Returns Arrow tables (converted from PyG Data) for Ray Data integration

    The graph is NOT partitioned - it's replicated on each worker's local storage.
    This sidesteps the "distributed graph partitioning" problem entirely.

    Attributes:
        graph_path: Path to the binary graph file.
        features_path: Optional path to the features file.
        num_neighbors: Number of neighbors to sample at each hop.
        batch_size: Number of seed nodes per batch.
        replace: Whether to sample with replacement.
        _input_nodes: Array of seed node IDs to sample from.

    Example:
        >>> import ray
        >>> from aethergraph.ray import AetherGraphDatasource
        >>>
        >>> ray.init()
        >>> ds = ray.data.read_datasource(
        ...     AetherGraphDatasource(
        ...         graph_path="graph.bin",
        ...         features_path="features.bin",
        ...         num_neighbors=[25, 10],
        ...         input_nodes=list(range(100000)),
        ...         batch_size=1024,
        ...     )
        ... )
        >>> for batch in ds.iter_rows():
        ...     train_step(batch)
    """

    graph_path: str
    features_path: str | None
    num_neighbors: list[int]
    batch_size: int
    replace: bool
    _input_nodes: npt.NDArray[np.int64] | None
    _num_input_nodes: int

    def __init__(
        self,
        graph_path: str | Path,
        num_neighbors: list[int],
        input_nodes: list[int] | npt.NDArray[Any] | None = None,
        batch_size: int = 1024,
        replace: bool = False,
        features_path: str | Path | None = None,
    ) -> None:
        """Initialize the AetherGraph datasource.

        Args:
            graph_path: Path to the binary graph file (.bin).
            num_neighbors: Number of neighbors per hop. For example, [25, 10]
                samples 25 neighbors at hop 1 and 10 at hop 2.
            input_nodes: Seed nodes to sample from. If None, uses all nodes
                in the graph. Can be a list or numpy array.
            batch_size: Number of seed nodes per batch.
            replace: Whether to sample with replacement.
            features_path: Optional path to features file (.bin).
        """
        self.graph_path = str(graph_path)
        self.features_path = str(features_path) if features_path else None
        self.num_neighbors = list(num_neighbors)
        self.batch_size = batch_size
        self.replace = replace

        if input_nodes is None:
            from aethergraph import Graph

            # Only the node count is needed here. Open mmap-backed with
            # header-only validation so the driver reads just the file header
            # instead of paging in (and validating) the whole graph; topology
            # pages stay on disk and are never touched.
            graph = Graph.load(self.graph_path, storage="mmap", validation="header_only")
            self._input_nodes = None
            self._num_input_nodes = graph.num_nodes
        elif isinstance(input_nodes, np.ndarray):
            self._input_nodes = input_nodes.astype(np.int64)
            self._num_input_nodes = len(self._input_nodes)
        else:
            self._input_nodes = np.array(input_nodes, dtype=np.int64)
            self._num_input_nodes = len(self._input_nodes)

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: int | None = None,
        data_context: Any | None = None,
    ) -> list[ReadTask]:
        """Generate read tasks that Ray will execute in parallel.

        Splits the input nodes across the requested parallelism level, creating
        one ReadTask per split. Each task will run the NeighborLoader on its
        partition of seed nodes.

        Args:
            parallelism: Number of parallel tasks requested by Ray.
            per_task_row_limit: Optional row limit per task (ignored).
            data_context: Ray's DataContext (Ray >= 2.46). Typed as ``Any`` so
                we don't take a hard import on ``ray.data.context.DataContext``
                whose location has shifted between Ray versions. We don't read
                it; accepting it satisfies the Liskov override.
        """
        del per_task_row_limit, data_context
        total_nodes = self._num_input_nodes
        num_tasks = min(parallelism, total_nodes)
        if num_tasks == 0:
            return []

        tasks: list[ReadTask] = []
        if self._input_nodes is not None:
            node_splits = np.array_split(self._input_nodes, num_tasks)

            for task_nodes in node_splits:
                if len(task_nodes) == 0:
                    continue

                def make_read_fn(
                    task_nodes: npt.NDArray[np.int64] = task_nodes,
                    graph_path: str = self.graph_path,
                    features_path: str | None = self.features_path,
                    num_neighbors: list[int] = self.num_neighbors,
                    batch_size: int = self.batch_size,
                    replace: bool = self.replace,
                ) -> Callable[[], Iterator[pa.Table]]:
                    """Create a read function closure for Ray task execution.

                    Uses default arguments to capture values at definition time,
                    avoiding late binding issues when closures are sent to workers.

                    Returns:
                        Callable that yields Arrow tables from neighborhood sampling.
                    """

                    def read_fn() -> Iterator[pa.Table]:
                        """Execute neighborhood sampling on a Ray worker.

                        Loads graph via memory-mapping and yields Arrow tables
                        for each sampled batch.
                        """
                        return _run_neighbor_loader(
                            task_nodes,
                            graph_path,
                            features_path,
                            num_neighbors,
                            batch_size,
                            replace,
                        )

                    return read_fn

                num_batches = (len(task_nodes) + self.batch_size - 1) // self.batch_size
                task = ReadTask(
                    read_fn=make_read_fn(),
                    metadata=BlockMetadata(
                        num_rows=num_batches,
                        size_bytes=None,
                        input_files=(self.graph_path,),
                        exec_stats=None,
                    ),
                )
                tasks.append(task)
        else:
            for task_idx in range(num_tasks):
                start = (task_idx * total_nodes) // num_tasks
                end = ((task_idx + 1) * total_nodes) // num_tasks
                if start >= end:
                    continue

                def make_range_read_fn(
                    start: int = start,
                    end: int = end,
                    graph_path: str = self.graph_path,
                    features_path: str | None = self.features_path,
                    num_neighbors: list[int] = self.num_neighbors,
                    batch_size: int = self.batch_size,
                    replace: bool = self.replace,
                ) -> Callable[[], Iterator[pa.Table]]:
                    """Create a read function closure for range-partitioned sampling."""

                    def read_fn() -> Iterator[pa.Table]:
                        task_nodes = np.arange(start, end, dtype=np.int64)
                        return _run_neighbor_loader(
                            task_nodes,
                            graph_path,
                            features_path,
                            num_neighbors,
                            batch_size,
                            replace,
                        )

                    return read_fn

                num_task_nodes = end - start
                num_batches = (num_task_nodes + self.batch_size - 1) // self.batch_size
                task = ReadTask(
                    read_fn=make_range_read_fn(),
                    metadata=BlockMetadata(
                        num_rows=num_batches,
                        size_bytes=None,
                        input_files=(self.graph_path,),
                        exec_stats=None,
                    ),
                )
                tasks.append(task)

        return tasks

    def estimate_inmemory_data_size(self) -> int | None:
        """Estimate in-memory size for Ray scheduling.

        Returns:
            None to indicate unknown size, letting Ray use defaults.
        """
        return None

    def get_name(self) -> str:
        """Return the datasource name for logging.

        Returns:
            The string "AetherGraph".
        """
        return "AetherGraph"


def _run_neighbor_loader(
    input_nodes: npt.NDArray[np.int64],
    graph_path: str,
    features_path: str | None,
    num_neighbors: list[int],
    batch_size: int,
    replace: bool,
) -> Iterator[pa.Table]:
    """Run the existing NeighborLoader on a Ray worker.

    This function is executed by Ray workers. It loads the graph via
    memory-mapping (instant, zero RAM overhead) and uses the existing
    NeighborLoader to sample neighborhoods.

    If OpenTelemetry is configured, emits a "ray_worker" span with attributes
    for the worker's assigned nodes and graph path.

    Args:
        input_nodes: Seed node IDs assigned to this worker.
        graph_path: Path to the binary graph file.
        features_path: Optional path to features file.
        num_neighbors: Number of neighbors per hop.
        batch_size: Number of seed nodes per batch.
        replace: Whether to sample with replacement.

    Yields:
        PyArrow Table for each sampled batch.

    Raises:
        RuntimeError: If sampling fails on this worker.
    """
    from aethergraph import Graph, GraphLoadError, SamplingError
    from aethergraph.pytorch import NeighborLoader
    from aethergraph.tracing import get_tracer

    tracer = get_tracer()
    worker_span = tracer.start_span("ray_worker") if tracer else None
    if worker_span:
        worker_span.set_attribute("num_input_nodes", len(input_nodes))
        worker_span.set_attribute("graph_path", graph_path)
        worker_span.set_attribute("num_neighbors", str(num_neighbors))
        worker_span.set_attribute("batch_size", batch_size)

    batches_yielded = 0
    try:
        graph = Graph.load(graph_path)
        loader = NeighborLoader(
            graph,
            num_neighbors=num_neighbors,
            input_nodes=input_nodes,
            batch_size=batch_size,
            shuffle=False,
            replace=replace,
            feature_path=features_path,
        )

        for data in loader:
            batches_yielded += 1
            yield _pyg_data_to_arrow(data)

    except GraphLoadError as e:
        if worker_span:
            worker_span.set_attribute("error", str(e))
        raise RuntimeError(f"Ray worker failed to load graph '{graph_path}': {e}") from e
    except SamplingError as e:
        if worker_span:
            worker_span.set_attribute("error", str(e))
        raise RuntimeError(f"Ray worker sampling failed for graph '{graph_path}': {e}") from e
    except OSError as e:
        if worker_span:
            worker_span.set_attribute("error", str(e))
        raise RuntimeError(f"Ray worker I/O error for graph '{graph_path}': {e}") from e
    except ValueError as e:
        if worker_span:
            worker_span.set_attribute("error", str(e))
        raise RuntimeError(f"Ray worker config error for graph '{graph_path}': {e}") from e
    finally:
        if worker_span:
            worker_span.set_attribute("batches_yielded", batches_yielded)
            worker_span.end()


def _numpy_to_list_array(arr: npt.NDArray[Any], value_type: pa.DataType) -> pa.ListArray:
    """Create a single-element list array from a numpy array.

    Creates a PyArrow ListArray containing a single list element from the
    input numpy array. This is zero-copy for the values array.

    Args:
        arr: Numpy array to wrap in a list array.
        value_type: PyArrow data type for the array values.

    Returns:
        ListArray containing one element (the input array as a list).
    """
    arr = np.ascontiguousarray(arr)
    values = pa.array(arr, type=value_type)
    offsets = pa.array([0, len(arr)], type=pa.int32())
    return pa.ListArray.from_arrays(offsets, values)


def _pyg_data_to_arrow(data: Data) -> pa.Table:
    """Convert a PyG Data object to an Arrow table.

    Creates a single-row Arrow table from the PyG Data object with list
    columns for variable-length arrays. Uses zero-copy where possible.

    Contract: one PyG batch maps to exactly one Arrow row. The variable-length
    arrays (edge_index, n_id, e_id, x) are packed into single list cells so a
    whole batch survives as one row. :func:`aethergraph.ray.collate_to_pyg`
    relies on this and must be fed one row at a time
    (``iter_batches(batch_size=1)``).

    Args:
        data: PyG Data object from NeighborLoader containing edge_index,
            n_id, e_id, batch_size, and optionally x (features).

    Returns:
        PyArrow Table with columns: edge_index_0, edge_index_1, n_id, e_id,
        batch_size, and optionally x, x_shape_0, x_shape_1.
    """
    # These tensors are already int64/float32 from the loader; the only
    # array that isn't C-contiguous is the edge_index row slice, and
    # `_numpy_to_list_array` makes its input contiguous anyway, so no extra
    # dtype cast or copy is needed here.
    edge_src: npt.NDArray[np.int64] = data.edge_index[0].numpy()
    edge_dst: npt.NDArray[np.int64] = data.edge_index[1].numpy()
    n_id: npt.NDArray[np.int64] = data.n_id.numpy()
    e_id: npt.NDArray[np.int64]
    if data.e_id is not None:
        e_id = data.e_id.numpy()
    else:
        e_id = np.array([], dtype=np.int64)

    columns: dict[str, pa.Array] = {
        "edge_index_0": _numpy_to_list_array(edge_src, pa.int64()),
        "edge_index_1": _numpy_to_list_array(edge_dst, pa.int64()),
        "n_id": _numpy_to_list_array(n_id, pa.int64()),
        "e_id": _numpy_to_list_array(e_id, pa.int64()),
        "batch_size": pa.array([data.batch_size], type=pa.int32()),
    }

    if data.x is not None:
        x_np: npt.NDArray[np.float32] = data.x.numpy().ravel()
        columns["x"] = _numpy_to_list_array(x_np, pa.float32())
        columns["x_shape_0"] = pa.array([data.x.shape[0]], type=pa.int32())
        columns["x_shape_1"] = pa.array([data.x.shape[1]], type=pa.int32())

    return pa.table(columns)
