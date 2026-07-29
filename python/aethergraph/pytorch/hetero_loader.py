"""PyTorch data loader for heterogeneous GNN training.

This module provides HeteroNeighborLoader, a drop-in replacement for PyTorch
Geometric's NeighborLoader with HeteroData that uses AetherGraph's Rust
sampling backend.
"""

from __future__ import annotations

import math
import time
import warnings
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

from aethergraph._core import HeteroNeighborSampler as RustHeteroSampler
from aethergraph._core import HeteroSampledSubgraph as RustHeteroSubgraph
from aethergraph._core import HeteroSamplingConfig as RustHeteroSamplingConfig
from aethergraph.pytorch.loader import normalize_input_nodes
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

    The loader handles parallelism internally via Rust, so num_workers
    must be 0.

    Attributes:
        graph: The heterogeneous graph to sample from.
        num_neighbors: Per-edge-type neighbor counts per hop.
        batch_size: Number of seed nodes per batch.
        shuffle: Whether to shuffle nodes between epochs.
        pin_memory: Whether to pin tensors in page-locked memory.
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
            num_workers: Must be 0. AetherGraph handles parallelism internally.
            pin_memory: If True, tensors are allocated in pinned memory for
                faster async GPU transfers.
            seed: Random seed for sampling reproducibility (passed to Rust).
            max_degree: Maximum degree cap for hub nodes.
            transform: Optional callable to apply to each batch after sampling.

        Raises:
            ValueError: If input_nodes is None or has invalid format.
            UserWarning: If num_workers is non-zero.
        """
        super().__init__()

        if num_workers != 0:
            warnings.warn(
                "num_workers must be 0 for AetherGraph (Rust handles parallelism)",
                UserWarning,
                stacklevel=2,
            )

        if input_nodes is None:
            raise ValueError(
                "input_nodes is required for HeteroNeighborLoader. "
                "Pass a (node_type, node_ids) tuple."
            )

        if not isinstance(input_nodes, tuple) or len(input_nodes) != 2:
            raise ValueError(
                "input_nodes must be a (node_type, node_ids) tuple, "
                f"got {type(input_nodes).__name__}"
            )

        self.graph = data
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replace = replace
        self.pin_memory = pin_memory
        self.transform = transform
        self._seed = seed
        self._max_degree = max_degree
        self._features: dict[str, npt.NDArray[np.float32]] | None = features

        seed_type, node_ids = input_nodes
        self._seed_type = seed_type

        if node_ids is None:
            self._input_nodes = None
            self._num_input_nodes = data.num_nodes(seed_type)
        else:
            self._input_nodes = normalize_input_nodes(node_ids)
            self._num_input_nodes = len(self._input_nodes)

        if self._input_nodes is not None:
            if self._input_nodes.ndim != 1:
                raise ValueError(
                    f"input_nodes must be a 1D array-like, got shape {self._input_nodes.shape}"
                )
            if np.any(self._input_nodes < 0):
                raise ValueError("input_nodes must contain non-negative node IDs")
            max_node = data.num_nodes(seed_type)
            if self._input_nodes.size > 0 and int(self._input_nodes.max()) >= max_node:
                raise ValueError(
                    f"input_nodes contains out-of-range node IDs for node type "
                    f"'{seed_type}' with num_nodes={max_node}"
                )

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
        """Iterate over batches of sampled heterogeneous subgraphs.

        Each iteration yields a PyG HeteroData object containing the sampled
        subgraph with per-type node features (if available), per-edge-type
        edge indices, and metadata.

        Yields:
            PyG HeteroData objects with per-type attributes:
                - data[node_type].n_id: Global node IDs
                - data[node_type].num_nodes: Number of nodes of this type
                - data[node_type].x: Node features (if available)
                - data[src, rel, dst].edge_index: Local edge connectivity
                - data[seed_type].batch_size: Number of seed nodes
                - data[seed_type].input_id: Global IDs of seed nodes
        """
        batch_size = self.batch_size
        num_batches = len(self)
        num_nodes = self._num_input_nodes

        # Fresh child generator per epoch: concurrent iterators don't share
        # mutable RNG state, and the sequence is reproducible when seeded.
        rng = self._rng.spawn(1)[0]

        # Build batch getter (same lazy/shuffle pattern as NeighborLoader)
        if self._input_nodes is not None:
            epoch_nodes = self._input_nodes.copy()
            if self.shuffle:
                rng.shuffle(epoch_nodes)

            def get_batch(batch_idx: int) -> npt.NDArray[np.int64]:
                start = batch_idx * batch_size
                end = min(start + batch_size, num_nodes)
                return epoch_nodes[start:end]

        elif self.shuffle and num_nodes > 0:
            offset = int(rng.integers(0, num_nodes))
            stride = self._random_coprime_stride(num_nodes, rng)

            def get_batch(batch_idx: int) -> npt.NDArray[np.int64]:
                start = batch_idx * batch_size
                end = min(start + batch_size, num_nodes)
                idx = np.arange(start, end, dtype=np.int64)
                return (offset + idx * stride) % num_nodes

        else:

            def get_batch(batch_idx: int) -> npt.NDArray[np.int64]:
                start = batch_idx * batch_size
                end = min(start + batch_size, num_nodes)
                return np.arange(start, end, dtype=np.int64)

        # Build Rust config and sampler
        rust_config = RustHeteroSamplingConfig(
            num_neighbors=self.num_neighbors,
            replace=self.replace,
            seed=self._seed,
            max_degree=self._max_degree,
        )
        sampler = RustHeteroSampler(self.graph.csr, rust_config)

        in_memory_features = self._features
        epoch_start = time.perf_counter()
        received = 0

        tracer = get_tracer()
        epoch_span = tracer.start_span("hetero_epoch") if tracer else None
        if epoch_span:
            epoch_span.set_attribute("num_batches", num_batches)
            epoch_span.set_attribute("batch_size", self.batch_size)
            epoch_span.set_attribute("seed_type", self._seed_type)

        try:
            for batch_idx in range(num_batches):
                seeds = get_batch(batch_idx)

                # Pass int64 seeds straight through: the Rust sampler accepts
                # int64 (range-checked at the FFI boundary). Narrowing to
                # uint32 here would wrap node IDs >= 2**32.
                subgraph = sampler.sample(self._seed_type, seeds)
                received += 1

                data = self._to_pyg_hetero_data(subgraph, in_memory_features)
                if self.transform is not None:
                    data = self.transform(data)
                yield data
        finally:
            total_time_ms = (time.perf_counter() - epoch_start) * 1000
            self._metrics = HeteroLoaderMetrics(
                batches_processed=received,
                total_time_ms=total_time_ms,
                avg_batch_time_ms=total_time_ms / received if received > 0 else 0.0,
            )

            if epoch_span:
                epoch_span.set_attribute("batches_processed", received)
                epoch_span.set_attribute("total_time_ms", total_time_ms)
                epoch_span.end()

    def _to_pyg_hetero_data(
        self,
        subgraph: RustHeteroSubgraph,
        in_memory_features: dict[str, npt.NDArray[np.float32]] | None,
    ) -> HeteroData:
        """Convert a heterogeneous sampled subgraph to a PyG HeteroData object.

        Args:
            subgraph: Rust HeteroSampledSubgraph.
            in_memory_features: Per-type in-memory features, or None.

        Returns:
            PyG HeteroData object ready for GNN forward pass.
        """
        data = HeteroData()

        for nt in subgraph.node_types:
            nodes = np.asarray(subgraph.nodes(nt), dtype=np.int64)
            n_id = torch.from_numpy(nodes.copy())
            data[nt].n_id = n_id
            data[nt].num_nodes = len(nodes)

            # Load features for this node type
            if in_memory_features and nt in in_memory_features:
                # Index into the in-memory feature array using global node IDs
                feat = in_memory_features[nt][nodes].copy()
                x = torch.from_numpy(np.asarray(feat, dtype=np.float32))
                if self.pin_memory and torch.cuda.is_available():
                    x = x.pin_memory()
                data[nt].x = x

        for src, rel, dst in subgraph.edge_types:
            ei = np.asarray(subgraph.edge_index_local(src, rel, dst), dtype=np.int64)
            edge_index = torch.from_numpy(ei.copy())
            if self.pin_memory and torch.cuda.is_available():
                edge_index = edge_index.pin_memory()
            data[src, rel, dst].edge_index = edge_index

        # Set batch_size and input_id for the seed node type
        seed_type = subgraph.seed_type
        seeds = np.asarray(subgraph.seeds, dtype=np.int64)
        data[seed_type].batch_size = len(seeds)

        # Compute input_id: the global IDs of the seed nodes
        input_id = torch.from_numpy(seeds.copy())
        if self.pin_memory and torch.cuda.is_available():
            input_id = input_id.pin_memory()
            n_id = data[seed_type].n_id
            if n_id is not None:
                data[seed_type].n_id = n_id.pin_memory()
        data[seed_type].input_id = input_id

        return data

    def _random_coprime_stride(self, modulus: int, rng: np.random.Generator) -> int:
        """Pick a random stride in [1, modulus) coprime with modulus."""
        if modulus <= 1:
            return 1
        while True:
            stride = int(rng.integers(1, modulus))
            if math.gcd(stride, modulus) == 1:
                return stride

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
