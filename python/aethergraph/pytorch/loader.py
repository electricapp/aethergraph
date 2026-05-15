"""PyTorch data loader for GNN training.

This module provides NeighborLoader, a drop-in replacement for PyTorch
Geometric's NeighborLoader that uses AetherGraph's Rust sampling backend.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

try:
    import torch
    from torch.utils.data import IterableDataset
except ImportError as e:
    raise ImportError(
        "PyTorch integration requires torch>=2.0. Install with: pip install aethergraph[torch]"
    ) from e

try:
    from torch_geometric.data import Data
except ImportError as e:
    raise ImportError(
        "NeighborLoader requires torch-geometric>=2.4. "
        "Install with: pip install aethergraph[pytorch-geometric]"
    ) from e

from aethergraph._core import HAS_GPUDIRECT
from aethergraph._core import NeighborLoader as RustNeighborLoader
from aethergraph._core import SampledSubgraph as RustSampledSubgraph
from aethergraph._core import SamplingConfig as RustSamplingConfig
from aethergraph.dynamic_graph import DynamicGraph
from aethergraph.tracing import get_tracer

if TYPE_CHECKING:
    from aethergraph.graph import Graph

__all__ = ["NeighborLoader", "LoaderMetrics"]


@dataclass
class LoaderMetrics:
    """Observability metrics for NeighborLoader.

    Provides production monitoring data for identifying bottlenecks in the
    sampling pipeline. Expose these via Prometheus or your metrics system.

    Attributes:
        batches_processed: Total number of batches yielded this epoch.
        total_time_ms: Wall-clock time for the entire epoch in milliseconds.
        avg_batch_time_ms: Average time per batch in milliseconds.
        prefetch_hit_rate: Fraction of batches ready without waiting (0.0-1.0).
            Low hit rate indicates sampling is slower than training.
        prefetch_hits: Number of batches that were immediately available.
        prefetch_misses: Number of times the consumer had to wait.

    Example:
        >>> loader = NeighborLoader(graph, num_neighbors=[15, 10])
        >>> for batch in loader:
        ...     train_step(batch)
        >>> metrics = loader.metrics
        >>> print(f"Hit rate: {metrics.prefetch_hit_rate:.1%}")
        >>> prometheus_gauge.set(metrics.avg_batch_time_ms)
    """

    batches_processed: int = 0
    total_time_ms: float = 0.0
    avg_batch_time_ms: float = 0.0
    prefetch_hit_rate: float = 0.0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    _epoch_start: float = field(default=0.0, repr=False)

    def to_dict(self) -> dict[str, float | int]:
        """Convert metrics to dictionary for JSON/logging.

        Returns:
            Dictionary with all metric values.
        """
        return {
            "batches_processed": self.batches_processed,
            "total_time_ms": self.total_time_ms,
            "avg_batch_time_ms": self.avg_batch_time_ms,
            "prefetch_hit_rate": self.prefetch_hit_rate,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
        }


class NeighborLoader(IterableDataset[Data]):
    """Sample k-hop neighborhoods from a graph for GNN training.

    Drop-in replacement for PyTorch Geometric's NeighborLoader that uses
    AetherGraph's high-performance Rust sampling backend with prefetching.

    Supports weighted sampling, temporal sampling (uniform/last), disjoint
    subgraphs, and custom sampler injection. For heterogeneous graphs, use
    :class:`HeteroNeighborLoader`. See the README for a complete comparison:
    https://github.com/electricapp/aethergraph#differences-from-pytorch-geometric

    The loader handles parallelism internally via Rust/Rayon, so num_workers
    must be 0. A prefetch pipeline ensures batches are ready before they're
    needed, minimizing GPU idle time.

    Attributes:
        graph: The graph to sample from.
        num_neighbors: Number of neighbors to sample at each hop.
        input_nodes: Node IDs to iterate over as seeds.
        batch_size: Number of seed nodes per batch.
        shuffle: Whether to shuffle nodes between epochs.
        replace: Whether to sample with replacement.
        prefetch_factor: Number of batches to prefetch ahead.
        transform: Optional transform to apply to each batch.
        rng: Random number generator for shuffling.

    Example:
        >>> from aethergraph import Graph
        >>> from aethergraph.pytorch import NeighborLoader
        >>>
        >>> graph = Graph.load("graph.bin")
        >>> loader = NeighborLoader(
        ...     graph,
        ...     num_neighbors=[15, 10],
        ...     batch_size=128,
        ...     shuffle=True,
        ...     feature_path="features.bin",
        ... )
        >>>
        >>> for batch in loader:
        ...     out = model(batch.x, batch.edge_index)
    """

    graph: Graph
    _dynamic_graph: DynamicGraph | None
    num_neighbors: list[int]
    _input_nodes: npt.NDArray[np.int64] | None
    _num_input_nodes: int
    batch_size: int
    shuffle: bool
    replace: bool
    weighted: bool
    prefetch_factor: int
    pin_memory: bool
    transform: Callable[[Data], Data] | None
    rng: np.random.Generator
    _metrics: LoaderMetrics
    _input_time: npt.NDArray[np.float64] | None
    _temporal_strategy: Literal["uniform", "last"] | None
    _disjoint: bool
    _neighbor_sampler: Callable[..., Data] | None

    def __init__(
        self,
        data: Graph | DynamicGraph,
        num_neighbors: list[int],
        input_nodes: torch.Tensor | npt.NDArray[Any] | list[int] | None = None,
        batch_size: int = 128,
        shuffle: bool = True,
        replace: bool = False,
        weighted: bool = False,
        num_workers: int = 0,
        prefetch_factor: int = 3,
        pin_memory: bool = False,
        transform: Callable[[Data], Data] | None = None,
        feature_source: str | None = None,
        gpu_id: int = 0,
        input_time: npt.NDArray[np.float64] | None = None,
        temporal_strategy: Literal["uniform", "last"] | None = None,
        disjoint: bool = False,
        neighbor_sampler: Callable[..., Data] | None = None,
        features: npt.NDArray[np.float32] | None = None,
        feature_path: str | Path | None = None,
    ) -> None:
        """Initialize the neighbor loader.

        Args:
            data: Graph to sample from. Can be a static :class:`Graph` or a
                :class:`DynamicGraph`. When a ``DynamicGraph`` is passed,
                a CSR snapshot is created automatically at the start of each
                epoch (in ``__iter__``), so the sampler always works with a
                consistent, frozen view. Features are not supported on
                DynamicGraph snapshots.
            num_neighbors: Number of neighbors per hop. For example, [15, 10]
                samples 15 neighbors at hop 1 and 10 at hop 2.
            input_nodes: Node IDs to sample from. If None, uses all nodes in
                the graph. Can be a torch.Tensor or numpy array.
            batch_size: Number of seed nodes per batch.
            shuffle: Whether to shuffle nodes between epochs.
            replace: Whether to sample neighbors with replacement.
            num_workers: Must be 0. AetherGraph handles parallelism internally
                via Rust/Rayon. A warning is issued if non-zero.
            prefetch_factor: Number of batches to prefetch ahead in the
                sampling pipeline. Higher values reduce GPU stalls but use
                more memory.
            pin_memory: If True, tensors are allocated in pinned (page-locked)
                memory, enabling faster async GPU transfers with non_blocking=True.
            transform: Optional callable to apply to each batch after sampling.
                Receives a PyG Data object and should return a Data object.
            feature_source: RDMA feature server address as "rdma://host:port".
                When set, features are gathered via GPUDirect RDMA directly
                into VRAM. Requires Linux, libibverbs, CUDA, and nvidia-peermem.
            gpu_id: CUDA device ordinal for RDMA features (default: 0).
                Only used when feature_source is set.
            input_time: Per-seed timestamps as a float64 ndarray for temporal
                sampling. Must have the same length as input_nodes. Only used
                when temporal_strategy is set.
            temporal_strategy: ``"uniform"`` or ``"last"``. When set, only
                edges with timestamp < seed time are eligible. Requires
                ``graph.set_timestamps()``.
            disjoint: If True, each seed gets an isolated subgraph with no
                node dedup across seeds. Output includes a ``batch`` tensor
                mapping each node to its seed index.
            neighbor_sampler: Custom sampling callable. If provided, bypasses
                the Rust backend entirely. Signature:
                ``(graph, seeds: np.ndarray) -> Data``.
            features: In-memory node features as a 2D ``float32`` array of
                shape ``[num_nodes, feature_dim]``. Mutually exclusive with
                ``feature_path``.
            feature_path: Path to an AETHFEAT file for lazy feature loading.
                The loader uses io_uring on Linux and mmap elsewhere.
                Mutually exclusive with ``features``.

        Raises:
            UserWarning: If num_workers is non-zero.
            ValueError: If both ``features`` and ``feature_path`` are set, or
                if ``features.shape[0]`` does not match the graph's node count.
        """
        super().__init__()

        if num_workers != 0:
            warnings.warn(
                "num_workers must be 0 for AetherGraph (Rust handles parallelism)",
                UserWarning,
                stacklevel=2,
            )

        # DynamicGraph: snapshot at each epoch for fresh edges.
        # Using match on the union discriminant — mypy narrows the type.
        match data:
            case DynamicGraph() as dg:
                self._dynamic_graph: DynamicGraph | None = dg
                self.graph = dg.snapshot()
            case _:
                self._dynamic_graph = None
                self.graph = data

        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replace = replace
        self.weighted = weighted
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.transform = transform
        self.feature_source = feature_source
        self.gpu_id = gpu_id
        if (
            feature_source is not None
            and feature_source.startswith("rdma://")
            and not HAS_GPUDIRECT
        ):
            raise RuntimeError(
                "feature_source='rdma://' requires a wheel built with the "
                "`gpudirect` Cargo feature (maturin develop --features gpudirect)."
            )
        self._temporal_strategy = temporal_strategy
        self._disjoint = disjoint
        self._neighbor_sampler = neighbor_sampler

        if features is not None and feature_path is not None:
            raise ValueError("pass either `features` or `feature_path`, not both")
        self._features: npt.NDArray[np.float32] | None = (
            np.asarray(features, dtype=np.float32) if features is not None else None
        )
        self._feature_path: Path | None = Path(feature_path) if feature_path is not None else None
        if self._features is not None and self._features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {self._features.shape}")

        num_nodes = self.graph.num_nodes

        if self._features is not None and self._features.shape[0] != num_nodes:
            raise ValueError(
                f"features.shape[0]={self._features.shape[0]} does not match graph "
                f"num_nodes={num_nodes}"
            )

        if input_nodes is None:
            self._input_nodes = None
            self._num_input_nodes = num_nodes
        elif isinstance(input_nodes, torch.Tensor):
            self._input_nodes = input_nodes.cpu().numpy().astype(np.int64)
            self._num_input_nodes = len(self._input_nodes)
        else:
            self._input_nodes = np.asarray(input_nodes, dtype=np.int64)
            self._num_input_nodes = len(self._input_nodes)

        if self._input_nodes is not None:
            if self._input_nodes.ndim != 1:
                raise ValueError(
                    f"input_nodes must be a 1D array-like, got shape {self._input_nodes.shape}"
                )
            if np.any(self._input_nodes < 0):
                raise ValueError("input_nodes must contain non-negative node IDs")
            if self._input_nodes.size > 0 and int(self._input_nodes.max()) >= num_nodes:
                raise ValueError(
                    f"input_nodes contains out-of-range node IDs for graph with "
                    f"num_nodes={num_nodes}"
                )

        self._input_time = input_time

        self.rng = np.random.default_rng()
        self._metrics = LoaderMetrics()

    @property
    def metrics(self) -> LoaderMetrics:
        """Get metrics from the most recent epoch.

        Returns:
            LoaderMetrics with timing and prefetch statistics.
        """
        return self._metrics

    def __iter__(self) -> Iterator[Data]:
        """Iterate over batches of sampled subgraphs.

        Each iteration yields a PyG Data object containing the sampled
        subgraph with node features (if available), edge indices, and
        metadata.

        Uses a sliding window prefetch strategy: while processing batch N,
        batches N+1 through N+prefetch_factor are being sampled in parallel
        by the Rust backend.

        Dispatch order:
            1. If ``neighbor_sampler`` is set, bypasses Rust entirely and
               delegates each batch to the user-provided callable.
            2. If ``feature_source`` starts with ``"rdma://"``, uses the
               GPUDirect RDMA path (features gathered directly into VRAM
               via DLPack zero-copy).
            3. Otherwise, uses the standard Rust prefetch pipeline with
               optional file-backed or in-memory features.

        For ``DynamicGraph`` inputs, a fresh CSR snapshot is taken at the
        start of each epoch (O(V + E)) so newly inserted edges are visible.

        Input nodes are generated lazily to avoid O(N) memory allocation at
        init time. For billion-node graphs with ``input_nodes=None``, this
        saves ~8GB per process. When ``shuffle=True`` with all nodes, an
        O(1)-memory coprime-stride permutation is used instead of a full
        shuffle array.

        Yields:
            PyG Data objects with attributes:
                - x: Node features [num_nodes, feature_dim] if available
                - edge_index: Edge connectivity [2, num_edges]
                - e_id: Global edge IDs [num_edges]
                - n_id: Global node IDs [num_nodes]
                - batch_size: Number of seed nodes in this batch
                - input_id: Local indices of seed nodes
                - num_nodes: Total nodes in subgraph
                - num_sampled_nodes: Nodes sampled per hop
                - num_sampled_edges: Edges sampled per hop
                - batch: (disjoint only) Node-to-seed mapping [num_nodes]
        """
        if self._dynamic_graph is not None:
            self.graph = self._dynamic_graph.snapshot()

        batch_size = self.batch_size
        num_batches = len(self)
        num_nodes = self._num_input_nodes

        if self._input_nodes is not None:
            epoch_nodes = self._input_nodes.copy()
            if self.shuffle:
                self.rng.shuffle(epoch_nodes)

            def get_batch(batch_idx: int) -> npt.NDArray[np.int64]:
                start = batch_idx * batch_size
                end = min(start + batch_size, num_nodes)
                return epoch_nodes[start:end]

        elif self.shuffle and num_nodes > 0:
            offset = int(self.rng.integers(0, num_nodes))
            stride = self._random_coprime_stride(num_nodes)

            def get_batch(batch_idx: int) -> npt.NDArray[np.int64]:
                start = batch_idx * batch_size
                end = min(start + batch_size, num_nodes)
                count = end - start
                return np.fromiter(
                    ((offset + (start + i) * stride) % num_nodes for i in range(count)),
                    dtype=np.int64,
                    count=count,
                )

        else:

            def get_batch(batch_idx: int) -> npt.NDArray[np.int64]:
                start = batch_idx * batch_size
                end = min(start + batch_size, num_nodes)
                return np.arange(start, end, dtype=np.int64)

        if self._neighbor_sampler is not None:
            for batch_idx in range(num_batches):
                seeds = get_batch(batch_idx)
                data = self._neighbor_sampler(self.graph, seeds)
                if self.transform is not None:
                    data = self.transform(data)
                yield data
            return

        rust_config = RustSamplingConfig(
            num_neighbors=self.num_neighbors,
            replace=self.replace,
            weighted=self.weighted,
            temporal_strategy=self._temporal_strategy,
            disjoint=self._disjoint,
        )

        if self.feature_source and self.feature_source.startswith("rdma://"):
            server_addr = self.feature_source[len("rdma://") :]
            sampler = RustNeighborLoader.with_rdma_features(
                self.graph,
                rust_config,
                server_addr,
                self.gpu_id,
                # Upper bound: batch_size * product of fanout at each hop
                max_batch_nodes=self.batch_size * max(1, int(np.prod(self.num_neighbors))) * 2,
                prefetch_depth=self.prefetch_factor,
            )
            yield from self._iter_rdma(sampler, num_batches, get_batch)
            return

        feature_path = self._feature_path
        in_memory_features = self._features

        if feature_path is not None:
            sampler = RustNeighborLoader.with_features(
                self.graph,
                rust_config,
                feature_path,
                self.prefetch_factor,
            )
        else:
            sampler = RustNeighborLoader(
                self.graph,
                rust_config,
                self.prefetch_factor,
            )

        submitted = 0
        received = 0
        epoch_start = time.perf_counter()

        tracer = get_tracer()
        epoch_span = tracer.start_span("epoch") if tracer else None
        if epoch_span:
            epoch_span.set_attribute("num_batches", num_batches)
            epoch_span.set_attribute("batch_size", self.batch_size)
            epoch_span.set_attribute("num_neighbors", str(self.num_neighbors))
            epoch_span.set_attribute("prefetch_factor", self.prefetch_factor)

        try:
            while submitted < min(self.prefetch_factor, num_batches):
                sampler.submit(submitted, get_batch(submitted))
                submitted += 1

            while received < num_batches:
                result = sampler.next_with_features()
                if result is None:
                    raise RuntimeError(
                        f"NeighborLoader backend stopped early: received {received} "
                        f"of {num_batches} batches"
                    )

                subgraph, features = result
                received += 1

                if submitted < num_batches:
                    sampler.submit(submitted, get_batch(submitted))
                    submitted += 1

                data = self._to_pyg_data(subgraph, features, in_memory_features)
                if self.transform is not None:
                    data = self.transform(data)
                yield data
        finally:
            stats = sampler.stats()
            sampler.shutdown()

            total_time_ms = (time.perf_counter() - epoch_start) * 1000
            self._metrics = LoaderMetrics(
                batches_processed=received,
                total_time_ms=total_time_ms,
                avg_batch_time_ms=total_time_ms / received if received > 0 else 0.0,
                prefetch_hit_rate=stats.hit_rate,
                prefetch_hits=stats.hits,
                prefetch_misses=stats.misses,
            )

            if epoch_span:
                epoch_span.set_attribute("batches_processed", received)
                epoch_span.set_attribute("total_time_ms", total_time_ms)
                epoch_span.set_attribute("prefetch_hit_rate", stats.hit_rate)
                epoch_span.end()

    def _to_pyg_data(
        self,
        subgraph: RustSampledSubgraph,
        file_features: npt.NDArray[np.float32] | None,
        in_memory_features: npt.NDArray[np.float32] | None,
    ) -> Data:
        """Convert a sampled subgraph to a PyG Data object.

        Feature sources are checked in priority order:
            1. File-based features (loaded by Rust sampler via io_uring/mmap)
            2. In-memory features (``graph.features`` numpy array)
            3. No features (``x`` will be ``None``)

        Edge indices use pre-computed local IDs from Rust's
        ``edge_index_local()``, which is O(1) HashMap lookup per edge.
        Seeds are extracted from the subgraph itself, eliminating any FIFO
        ordering dependency between batch submission and result retrieval.

        In disjoint mode, a ``batch`` tensor is attached mapping each node
        to its seed index. All tensors are optionally pinned to page-locked
        memory when ``pin_memory=True`` for faster async GPU transfer.

        Args:
            subgraph: Rust SampledSubgraph containing nodes, edges, and seeds.
            file_features: Features loaded from file by Rust sampler, or None.
            in_memory_features: Features set directly on Graph, or None.

        Returns:
            PyG Data object with x, edge_index, e_id, n_id, input_id, and
            optionally batch (disjoint mode).
        """
        nodes_arr: npt.NDArray[np.int64] = np.asarray(subgraph.nodes, dtype=np.int64)
        n_id = torch.from_numpy(nodes_arr.copy())
        num_nodes = len(n_id)

        local_edge_index: npt.NDArray[np.int64] = subgraph.edge_index_local
        edge_index = torch.from_numpy(local_edge_index.copy())

        e_id = torch.from_numpy(np.asarray(subgraph.edge_ids, dtype=np.int64))

        x: torch.Tensor | None = None
        if file_features is not None:
            x = torch.from_numpy(np.asarray(file_features, dtype=np.float32))
        elif in_memory_features is not None:
            x = torch.from_numpy(in_memory_features[nodes_arr].copy())

        seed_indices_arr: npt.NDArray[np.int64] = np.asarray(subgraph.seed_indices, dtype=np.int64)
        input_id = torch.from_numpy(seed_indices_arr.copy())

        if self.pin_memory and torch.cuda.is_available():
            edge_index = edge_index.pin_memory()
            e_id = e_id.pin_memory()
            n_id = n_id.pin_memory()
            input_id = input_id.pin_memory()
            if x is not None:
                x = x.pin_memory()

        data = Data(
            x=x,
            edge_index=edge_index,
            e_id=e_id,
            n_id=n_id,
            batch_size=len(seed_indices_arr),
            input_id=input_id,
            num_nodes=num_nodes,
            num_sampled_nodes=subgraph.num_sampled_nodes_per_hop,
            num_sampled_edges=subgraph.num_sampled_edges_per_hop,
        )

        batch_arr = subgraph.batch
        if batch_arr is not None:
            batch_tensor = torch.from_numpy(np.asarray(batch_arr, dtype=np.int64))
            if self.pin_memory and torch.cuda.is_available():
                batch_tensor = batch_tensor.pin_memory()
            data.batch = batch_tensor

        return data

    def _iter_rdma(
        self,
        sampler: RustNeighborLoader,
        num_batches: int,
        get_batch: Callable[[int], npt.NDArray[np.int64]],
    ) -> Iterator[Data]:
        """Yield batches with RDMA-gathered GPU features.

        Features are gathered via GPUDirect RDMA directly into VRAM and
        returned as CUDA tensors via DLPack (zero-copy, never touches CPU).
        Uses the same sliding-window prefetch strategy as the standard path.

        Args:
            sampler: Rust NeighborLoader configured with RDMA feature source.
            num_batches: Total batches to yield this epoch.
            get_batch: Callable returning seed node IDs for a given batch index.

        Yields:
            PyG Data objects with ``x`` as a CUDA tensor and all metadata
            on the same device.
        """
        submitted = 0
        received = 0
        epoch_start = time.perf_counter()

        tracer = get_tracer()
        epoch_span = tracer.start_span("epoch_rdma") if tracer else None

        try:
            while submitted < min(self.prefetch_factor, num_batches):
                sampler.submit(submitted, get_batch(submitted))
                submitted += 1

            while received < num_batches:
                result = sampler.next_with_gpu_features()
                if result is None:
                    raise RuntimeError(
                        f"RDMA sampler stopped early: received {received} of {num_batches} batches"
                    )

                subgraph, dlpack_capsule = result
                received += 1

                if submitted < num_batches:
                    sampler.submit(submitted, get_batch(submitted))
                    submitted += 1

                x = torch.from_dlpack(dlpack_capsule)  # type: ignore[attr-defined]

                data = self._to_pyg_data_gpu(subgraph, x)
                if self.transform is not None:
                    data = self.transform(data)
                yield data
        finally:
            stats = sampler.stats()
            sampler.shutdown()

            total_time_ms = (time.perf_counter() - epoch_start) * 1000
            self._metrics = LoaderMetrics(
                batches_processed=received,
                total_time_ms=total_time_ms,
                avg_batch_time_ms=total_time_ms / received if received > 0 else 0.0,
                prefetch_hit_rate=stats.hit_rate,
                prefetch_hits=stats.hits,
                prefetch_misses=stats.misses,
            )

            if epoch_span:
                epoch_span.set_attribute("batches_processed", received)
                epoch_span.set_attribute("total_time_ms", total_time_ms)
                epoch_span.set_attribute("prefetch_hit_rate", stats.hit_rate)
                epoch_span.end()

    def _to_pyg_data_gpu(
        self,
        subgraph: RustSampledSubgraph,
        x_gpu: torch.Tensor,
    ) -> Data:
        """Build PyG Data with features already on GPU via RDMA.

        Unlike ``_to_pyg_data``, features arrive as a CUDA tensor from
        GPUDirect RDMA (zero-copy DLPack capsule). All metadata tensors
        (edge_index, e_id, n_id, input_id) are moved to the same CUDA
        device with ``non_blocking=True`` for async H2D transfer.

        Args:
            subgraph: Rust SampledSubgraph containing nodes, edges, and seeds.
            x_gpu: CUDA tensor of node features from RDMA gather.

        Returns:
            PyG Data object with all tensors on the same CUDA device.
        """
        nodes_arr: npt.NDArray[np.int64] = np.asarray(subgraph.nodes, dtype=np.int64)
        n_id = torch.from_numpy(nodes_arr.copy())
        num_nodes = len(n_id)

        local_edge_index: npt.NDArray[np.int64] = subgraph.edge_index_local
        edge_index = torch.from_numpy(local_edge_index.copy())

        e_id = torch.from_numpy(np.asarray(subgraph.edge_ids, dtype=np.int64))

        seed_indices_arr: npt.NDArray[np.int64] = np.asarray(subgraph.seed_indices, dtype=np.int64)
        input_id = torch.from_numpy(seed_indices_arr.copy())

        device = x_gpu.device
        edge_index = edge_index.to(device, non_blocking=True)
        e_id = e_id.to(device, non_blocking=True)
        n_id = n_id.to(device, non_blocking=True)
        input_id = input_id.to(device, non_blocking=True)

        return Data(
            x=x_gpu,
            edge_index=edge_index,
            e_id=e_id,
            n_id=n_id,
            batch_size=len(seed_indices_arr),
            input_id=input_id,
            num_nodes=num_nodes,
            num_sampled_nodes=subgraph.num_sampled_nodes_per_hop,
            num_sampled_edges=subgraph.num_sampled_edges_per_hop,
        )

    def _random_coprime_stride(self, modulus: int) -> int:
        """Pick a random stride coprime with modulus for O(1)-memory shuffling.

        Used to generate a full permutation of ``[0, modulus)`` without
        allocating an array: ``node = (offset + pos * stride) % modulus``
        visits every index exactly once when ``gcd(stride, modulus) == 1``.

        Args:
            modulus: The range size (number of input nodes).

        Returns:
            A random integer in ``[1, modulus)`` coprime with modulus.
        """
        if modulus <= 1:
            return 1
        while True:
            stride = int(self.rng.integers(1, modulus))
            if math.gcd(stride, modulus) == 1:
                return stride

    def __len__(self) -> int:
        """Return the number of batches per epoch.

        Returns:
            Number of batches, computed as ceil(num_input_nodes / batch_size).
        """
        return (self._num_input_nodes + self.batch_size - 1) // self.batch_size

    def __repr__(self) -> str:
        """Return a string representation of the loader.

        Returns:
            String showing num_nodes, num_neighbors, and batch_size.
        """
        return (
            f"NeighborLoader(num_nodes={self._num_input_nodes}, "
            f"num_neighbors={self.num_neighbors}, batch_size={self.batch_size})"
        )
