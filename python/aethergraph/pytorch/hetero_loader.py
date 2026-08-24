"""PyTorch data loader for heterogeneous GNN training.

This module provides HeteroNeighborLoader, a drop-in replacement for PyTorch
Geometric's NeighborLoader with HeteroData that uses AetherGraph's Rust
sampling backend.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
    from torch_geometric.data import HeteroData
except ImportError as e:
    raise ImportError(
        "HeteroNeighborLoader requires torch-geometric>=2.4. "
        "Install with: pip install aethergraph[pytorch-geometric]"
    ) from e

from aethergraph._core import HeteroNeighborLoader as RustHeteroLoader
from aethergraph._core import HeteroSampledSubgraph as RustHeteroSubgraph
from aethergraph._core import HeteroSamplingConfig as RustHeteroSamplingConfig
from aethergraph.pytorch.device_pipeline import DeviceTransferPipeline
from aethergraph.pytorch.loader import _parse_features, make_batch_getter, normalize_input_nodes
from aethergraph.tracing import get_tracer

if TYPE_CHECKING:
    from aethergraph.hetero_graph import HeteroGraph

__all__ = ["HeteroLoaderMetrics", "HeteroNeighborLoader"]


@dataclass
class HeteroLoaderMetrics:
    """Observability metrics for HeteroNeighborLoader.

    Attributes:
        batches_processed: Total number of batches yielded this epoch.
        total_time_ms: Wall-clock time for the entire epoch in milliseconds.
        avg_batch_time_ms: Average time per batch in milliseconds.
    """

    batches_processed: int = 0
    total_time_ms: float = 0.0
    avg_batch_time_ms: float = 0.0
    _epoch_start: float = field(default=0.0, repr=False)

    def to_dict(self) -> dict[str, float | int]:
        """Convert metrics to dictionary for JSON/logging."""
        return {
            "batches_processed": self.batches_processed,
            "total_time_ms": self.total_time_ms,
            "avg_batch_time_ms": self.avg_batch_time_ms,
        }


class HeteroNeighborLoader(IterableDataset[HeteroData]):
    """Heterogeneous neighbor sampling for GNN training.

    Drop-in replacement for PyG's NeighborLoader with HeteroData.
    Uses AetherGraph's Rust sampling backend for multi-relational graphs.

    Sampling runs in the Rust ``HeteroNeighborLoader`` pipeline — the same
    worker pool behind bounded channels the homogeneous loader uses — so
    batch N+1 through N+prefetch_factor are sampled while Python converts
    and trains on batch N.

    Parallelism lives in Rust: ``num_workers`` sizes the sampler thread
    pool (PyG-compatible semantics, but threads in the Rust backend instead
    of DataLoader worker processes).

    Attributes:
        graph: The heterogeneous graph to sample from.
        num_neighbors: Per-edge-type neighbor counts per hop.
        batch_size: Number of seed nodes per batch.
        shuffle: Whether to shuffle nodes between epochs.
        pin_memory: Whether to pin tensors in page-locked memory.
        prefetch_factor: Number of batches sampled ahead of the consumer.
        transform: Optional transform to apply to each batch.

    Example:
        >>> loader = HeteroNeighborLoader(
        ...     hetero_graph,
        ...     num_neighbors={
        ...         ("user", "votes", "post"): [15, 10],
        ...         ("user", "replies", "comment"): [10, 5],
        ...     },
        ...     input_nodes=("user", train_user_ids),
        ...     batch_size=128,
        ... )
        >>> for batch in loader:
        ...     user_x = batch["user"].x
        ...     ei = batch["user", "votes", "post"].edge_index
    """

    graph: HeteroGraph
    num_neighbors: dict[tuple[str, str, str], list[int]]
    _seed_type: str
    _input_nodes: npt.NDArray[np.int64] | None
    _num_input_nodes: int
    batch_size: int
    shuffle: bool
    replace: bool
    pin_memory: bool
    prefetch_factor: int
    transform: Callable[[HeteroData], HeteroData] | None
    _rng: np.random.Generator
    _metrics: HeteroLoaderMetrics

    def __init__(
        self,
        data: HeteroGraph,
        num_neighbors: dict[tuple[str, str, str], list[int]],
        input_nodes: tuple[str, torch.Tensor | npt.NDArray[Any] | list[int] | None] | None = None,
        batch_size: int = 128,
        shuffle: bool = True,
        replace: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        device: str | None = None,
        transfer_depth: int = 2,
        prefetch_factor: int = 2,
        seed: int | None = None,
        max_degree: int | None = None,
        transform: Callable[[HeteroData], HeteroData] | None = None,
        features: dict[str, npt.NDArray[np.float32]] | None = None,
    ) -> None:
        """Initialize the heterogeneous neighbor loader.

        Args:
            data: HeteroGraph to sample from.
            num_neighbors: Dict mapping (src_type, rel, dst_type) to per-hop
                neighbor counts. Example: {("user","votes","post"): [15, 10]}
            input_nodes: Tuple of (node_type, node_ids) specifying which nodes
                to iterate over as seeds. If None, raises ValueError.
                node_ids can be a Tensor, numpy array, or list; a boolean
                tensor/array is treated as a PyG-style mask and converted to
                the indices of its ``True`` entries. If node_ids is None, all
                nodes of that type are used.
            batch_size: Number of seed nodes per batch.
            shuffle: Whether to shuffle seed nodes between epochs.
            replace: Whether to sample neighbors with replacement.
            num_workers: Number of Rust sampler threads feeding the
                prefetch pipeline (PyG-compatible semantics, but threads in
                the Rust backend instead of DataLoader worker processes).
                0 and 1 both mean a single sampler thread; higher values
                scale sampling throughput when sampling is the bottleneck.
            pin_memory: If True, tensors are allocated in pinned memory for
                faster async GPU transfers.
            device: Optional CUDA device (e.g. ``"cuda"``). When set, each
                batch is moved to it on a dedicated copy stream, double-
                buffered so batch N+1's transfer overlaps batch N's
                training; implies pinning. A no-op passthrough without CUDA.
            transfer_depth: Batches kept in flight on the copy stream when
                ``device`` is set (default 2 = double buffering).
            prefetch_factor: Number of batches the Rust prefetch pipeline
                keeps ready ahead of the consumer.
            seed: Random seed for sampling reproducibility (passed to Rust).
            max_degree: Maximum degree cap for hub nodes.
            transform: Optional callable to apply to each batch after sampling.
            features: Per-node-type in-memory features. Normalized once to
                contiguous ``float32`` here so the per-batch gather never
                re-converts dtypes.

        Raises:
            ValueError: If input_nodes is None or has invalid format, or if
                ``num_workers`` is negative.
        """
        super().__init__()

        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")
        # PyG semantics, Rust execution: instead of forking DataLoader
        # worker processes, num_workers sizes the Rust sampler thread pool
        # feeding the prefetch pipeline. 0 keeps the single-threaded
        # pipeline (sampling still overlaps training).
        self._sampler_threads: int = max(1, num_workers)

        if not num_neighbors:
            raise ValueError("num_neighbors must be a non-empty dict")
        hop_lens = {len(hops) for hops in num_neighbors.values()}
        if len(hop_lens) != 1:
            raise ValueError(
                "all edge types in num_neighbors must have the same number of hops, "
                f"got lengths {sorted(hop_lens)}"
            )
        for edge_type, hops in num_neighbors.items():
            if not hops:
                raise ValueError(
                    f"num_neighbors[{edge_type!r}] must be a non-empty list, got {hops!r}"
                )
            if any(n < 0 for n in hops):
                raise ValueError(
                    f"num_neighbors values must be non-negative, got {hops} for {edge_type!r}"
                )
        if max_degree is not None and max_degree <= 0:
            raise ValueError(f"max_degree must be > 0 if specified, got {max_degree}")

        if input_nodes is None:
            raise ValueError(
                "input_nodes is required for HeteroNeighborLoader. "
                "Pass a (node_type, node_ids) tuple."
            )

        try:
            seed_type, node_ids = input_nodes
        except (TypeError, ValueError) as e:
            raise ValueError(
                "input_nodes must be a (node_type, node_ids) tuple, "
                f"got {type(input_nodes).__name__}"
            ) from e

        self.graph = data
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replace = replace
        self.pin_memory = pin_memory
        self.prefetch_factor = max(1, prefetch_factor)
        self.transform = transform
        self._seed = seed
        self._max_degree = max_degree

        # Parse features once into their canonical form: zero-copy torch
        # views over contiguous float32 arrays, shape-checked against each
        # type's node count — the one representation the per-batch gather
        # consumes. `_parse_features` is the same edge parser the
        # homogeneous loader uses.
        self._feat_torch: dict[str, torch.Tensor] | None = (
            {k: _parse_features(v, data.num_nodes(k)) for k, v in features.items()}
            if features is not None
            else None
        )

        # Device availability is immutable for the process; resolve the
        # pinning decision once instead of per tensor per type per batch.
        self._pin: bool = bool(pin_memory and torch.cuda.is_available())

        # Optional stream-pipelined device transfer, matching NeighborLoader:
        # each batch moves to the device on a dedicated copy stream, double-
        # buffered so batch N+1 crosses PCIe while batch N trains. Moving to
        # a CUDA device implies pinning the host source.
        self._device: torch.device | None = torch.device(device) if device is not None else None
        self._transfer_depth = transfer_depth
        if self._device is not None and self._device.type == "cuda":
            self._pin = self._pin or torch.cuda.is_available()

        # Type lists are graph-wide and immutable: resolved here once, so
        # batch conversion never asks Rust to rebuild the string lists.
        self._node_types: list[str] = data.node_types
        self._edge_types: list[tuple[str, str, str]] = data.edge_types

        self._seed_type = seed_type

        if node_ids is None:
            self._input_nodes = None
            self._num_input_nodes = data.num_nodes(seed_type)
        else:
            self._input_nodes = normalize_input_nodes(
                node_ids,
                num_nodes=data.num_nodes(seed_type),
                what=f"input_nodes for node type '{seed_type}'",
            )
            self._num_input_nodes = len(self._input_nodes)

        # Seed the shuffle generator from `seed` so epochs are reproducible;
        # `__iter__` spawns a fresh child per epoch so concurrent iterators
        # don't share mutable RNG state.
        self._rng = np.random.default_rng(seed)
        self._metrics = HeteroLoaderMetrics()

    @property
    def metrics(self) -> HeteroLoaderMetrics:
        """Get metrics from the most recent epoch."""
        return self._metrics

    def __iter__(self) -> Iterator[HeteroData]:
        """Iterate over batches, optionally pipelined onto a CUDA device.

        Delegates to :meth:`_iter_batches` for sampling; when ``device`` was
        given, wraps the batches in a :class:`DeviceTransferPipeline` so each
        one's host-to-device copy overlaps the consumer's compute on the
        previous batch. Without a device, host batches are yielded directly.
        """
        batches = self._iter_batches()
        if self._device is not None:
            return iter(DeviceTransferPipeline(batches, self._device, depth=self._transfer_depth))
        return batches

    def _iter_batches(self) -> Iterator[HeteroData]:
        """Iterate over batches of sampled heterogeneous subgraphs.

        Uses the same sliding-window prefetch strategy as the homogeneous
        loader: ``prefetch_factor`` batches are primed into the Rust
        ``HeteroNeighborLoader`` pipeline, then each received batch is
        replaced by the next submission, so the Rust worker pool samples
        ahead while the consumer converts and trains on the current batch.

        Yields:
            PyG HeteroData objects with per-type attributes:
                - data[node_type].n_id: Global node IDs
                - data[node_type].num_nodes: Number of nodes of this type
                - data[node_type].x: Node features (if available)
                - data[src, rel, dst].edge_index: Local edge connectivity
                - data[seed_type].batch_size: Number of seed nodes
                - data[seed_type].input_id: Local indices of seeds in
                  ``n_id`` (homo contract; globals are ``n_id[input_id]``)
        """
        batch_size = self.batch_size
        num_batches = len(self)
        num_nodes = self._num_input_nodes

        # Fresh child generator per epoch: concurrent iterators don't share
        # mutable RNG state, and the sequence is reproducible when seeded.
        rng = self._rng.spawn(1)[0]
        get_batch = make_batch_getter(self._input_nodes, num_nodes, batch_size, self.shuffle, rng)

        # Build Rust config and prefetching loader
        rust_config = RustHeteroSamplingConfig(
            num_neighbors=self.num_neighbors,
            replace=self.replace,
            seed=self._seed,
            max_degree=self._max_degree,
        )
        loader = RustHeteroLoader(
            self.graph.csr,
            rust_config,
            self._seed_type,
            prefetch_depth=self.prefetch_factor,
            sampler_threads=self._sampler_threads,
        )

        in_memory_features = self._feat_torch
        epoch_start = time.perf_counter()
        submitted = 0
        received = 0

        tracer = get_tracer()
        epoch_span = tracer.start_span("hetero_epoch") if tracer else None
        if epoch_span:
            epoch_span.set_attribute("num_batches", num_batches)
            epoch_span.set_attribute("batch_size", self.batch_size)
            epoch_span.set_attribute("seed_type", self._seed_type)
            epoch_span.set_attribute("prefetch_factor", self.prefetch_factor)

        try:
            while submitted < min(self.prefetch_factor, num_batches):
                loader.submit(submitted, get_batch(submitted))
                submitted += 1

            while received < num_batches:
                subgraph = loader.next()
                if subgraph is None:
                    raise RuntimeError(
                        f"hetero sampler stopped early: received {received} "
                        f"of {num_batches} batches"
                    )
                received += 1

                if submitted < num_batches:
                    loader.submit(submitted, get_batch(submitted))
                    submitted += 1

                data = self._to_pyg_hetero_data(subgraph, in_memory_features)
                if self.transform is not None:
                    data = self.transform(data)
                yield data
        finally:
            stats = loader.stats()
            loader.shutdown()

            total_time_ms = (time.perf_counter() - epoch_start) * 1000
            self._metrics = HeteroLoaderMetrics(
                batches_processed=received,
                total_time_ms=total_time_ms,
                avg_batch_time_ms=total_time_ms / received if received > 0 else 0.0,
            )

            if epoch_span:
                epoch_span.set_attribute("batches_processed", received)
                epoch_span.set_attribute("total_time_ms", total_time_ms)
                epoch_span.set_attribute("prefetch_hit_rate", stats.hit_rate)
                epoch_span.end()

    def _to_pyg_hetero_data(
        self,
        subgraph: RustHeteroSubgraph,
        in_memory_features: dict[str, torch.Tensor] | None,
    ) -> HeteroData:
        """Convert a heterogeneous sampled subgraph to a PyG HeteroData object.

        Feature rows gather in a single ``index_select`` pass straight into
        the final (optionally pinned) buffer.

        Args:
            subgraph: Rust HeteroSampledSubgraph.
            in_memory_features: Per-type feature tensors (zero-copy torch
                views over the normalized float32 arrays), or None.

        Returns:
            PyG HeteroData object ready for GNN forward pass.
        """
        data = HeteroData()
        pin = self._pin
        node_types = self._node_types
        edge_types = self._edge_types

        for nt in node_types:
            nodes = subgraph.nodes(nt)
            n_id = torch.from_numpy(nodes)
            if pin:
                n_id = n_id.pin_memory()
            store = data[nt]
            store.n_id = n_id
            store.num_nodes = len(nodes)

            if in_memory_features is not None:
                feat = in_memory_features.get(nt)
                if feat is not None:
                    out = torch.empty(
                        (len(nodes), feat.shape[1]), dtype=torch.float32, pin_memory=pin
                    )
                    torch.index_select(feat, 0, n_id, out=out)
                    store.x = out

        for src, rel, dst in edge_types:
            edge_index = torch.from_numpy(subgraph.edge_index_local(src, rel, dst))
            if pin:
                edge_index = edge_index.pin_memory()
            data[src, rel, dst].edge_index = edge_index

        # Homo contract: input_id = local seed indices, batch_size = len(seeds)
        # including duplicates. seed_indices preserves one entry per input seed.
        seed_type = subgraph.seed_type
        seed_indices = torch.from_numpy(subgraph.seed_indices)
        if pin:
            seed_indices = seed_indices.pin_memory()
        seed_store = data[seed_type]
        seed_store.batch_size = len(seed_indices)
        seed_store.input_id = seed_indices

        return data

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return (self._num_input_nodes + self.batch_size - 1) // self.batch_size

    def __repr__(self) -> str:
        """Return a string representation of the loader."""
        return (
            f"HeteroNeighborLoader(seed_type='{self._seed_type}', "
            f"num_input_nodes={self._num_input_nodes}, "
            f"num_neighbors=<{len(self.num_neighbors)} edge types>, "
            f"batch_size={self.batch_size})"
        )
