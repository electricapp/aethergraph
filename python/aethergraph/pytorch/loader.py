"""PyTorch data loader for GNN training.

This module provides NeighborLoader, a drop-in replacement for PyTorch
Geometric's NeighborLoader that uses AetherGraph's Rust sampling backend.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

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
from aethergraph._types import SubgraphType, TemporalStrategy
from aethergraph.dynamic_graph import DynamicGraph
from aethergraph.pytorch.device_pipeline import DeviceTransferPipeline
from aethergraph.sampler import SamplingConfig
from aethergraph.tracing import get_tracer

if TYPE_CHECKING:
    from aethergraph.graph import Graph

__all__ = ["LoaderMetrics", "NeighborLoader"]

# What a backend hands the epoch driver per batch, before PyG conversion.
_Received = TypeVar("_Received")


def normalize_input_nodes(
    input_nodes: torch.Tensor | npt.NDArray[Any] | list[int],
    *,
    num_nodes: int,
    what: str = "input_nodes",
) -> npt.NDArray[np.int64]:
    """Parse ``input_nodes`` into a validated, contiguous 1D int64 array.

    This is the single choke point for seed-node input: it accepts a torch
    tensor, numpy array, or Python sequence, and returns an array whose type
    carries the full contract — 1D, contiguous, ``int64``, every ID in
    ``[0, num_nodes)``. Callers use the result without further checks.

    A boolean tensor/array is treated as a PyG-style mask and converted to the
    indices of its ``True`` entries via ``np.nonzero``; everything else is
    taken as explicit node IDs.

    Args:
        input_nodes: Mask or explicit node IDs.
        num_nodes: Exclusive upper bound for valid node IDs.
        what: Description of the array for error messages.

    Returns:
        Contiguous 1D ``int64`` array of in-range node IDs.

    Raises:
        ValueError: If the input is not 1D after mask conversion, has a
            non-integer dtype (floats are rejected — no silent truncation),
            or contains IDs outside ``[0, num_nodes)``.
    """
    if isinstance(input_nodes, torch.Tensor):
        if input_nodes.dtype == torch.bool:
            arr: npt.NDArray[np.int64] = np.nonzero(input_nodes.cpu().numpy())[0].astype(np.int64)
        elif input_nodes.dtype.is_floating_point:
            raise ValueError(f"{what} must be integer or bool IDs, got dtype {input_nodes.dtype}")
        else:
            arr = input_nodes.cpu().numpy().astype(np.int64, copy=False)
    else:
        a = np.asarray(input_nodes)
        if a.dtype == np.bool_:
            arr = np.nonzero(a)[0].astype(np.int64)
        elif np.issubdtype(a.dtype, np.integer):
            arr = a.astype(np.int64, copy=False)
        else:
            raise ValueError(f"{what} must be integer or bool IDs, got dtype {a.dtype}")
    if arr.ndim != 1:
        raise ValueError(f"{what} must be a 1D array-like, got shape {arr.shape}")
    if arr.size > 0:
        lo = int(arr.min())
        hi = int(arr.max())
        if lo < 0:
            raise ValueError(f"{what} must contain non-negative node IDs (min {lo})")
        if hi >= num_nodes:
            raise ValueError(
                f"{what} contains out-of-range node IDs (max {hi} >= num_nodes={num_nodes})"
            )
    return np.ascontiguousarray(arr)


def _random_coprime_stride(modulus: int, rng: np.random.Generator) -> int:
    """Pick a random stride in ``[1, modulus)`` coprime with ``modulus``.

    Used to walk a full permutation of ``[0, modulus)`` in O(1) memory:
    ``node = (offset + pos * stride) % modulus`` visits every index exactly
    once when ``gcd(stride, modulus) == 1``.
    """
    if modulus <= 1:
        return 1
    while True:
        stride = int(rng.integers(1, modulus))
        if math.gcd(stride, modulus) == 1:
            return stride


@dataclass(frozen=True)
class SeedBatch:
    """One batch of seeds and where they sit in ``input_nodes``.

    ``input_id`` is PyG's: each seed's position in the loader's
    ``input_nodes`` (the node IDs themselves when iterating all nodes).
    """

    seeds: npt.NDArray[np.int64]
    input_id: npt.NDArray[np.int64]


def make_batch_getter(
    input_nodes: npt.NDArray[np.int64] | None,
    num_nodes: int,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Callable[[int], SeedBatch]:
    """Build the per-epoch ``batch_idx -> SeedBatch`` function.

    With explicit ``input_nodes``, batches slice a (shuffled) permutation of
    their positions. With all nodes and ``shuffle``, an O(1)-memory
    coprime-stride permutation replaces the full shuffle array — for
    billion-node graphs that saves ~8 GB per epoch. Otherwise batches are
    plain ``arange`` ranges.
    """
    if input_nodes is not None:
        nodes = input_nodes
        order = rng.permutation(num_nodes) if shuffle else None

        def get_batch_sliced(batch_idx: int) -> SeedBatch:
            start = batch_idx * batch_size
            end = min(start + batch_size, num_nodes)
            if order is None:
                return SeedBatch(nodes[start:end], np.arange(start, end, dtype=np.int64))
            positions = order[start:end]
            return SeedBatch(nodes[positions], positions)

        return get_batch_sliced

    if shuffle and num_nodes > 0:
        offset = int(rng.integers(0, num_nodes))
        stride = _random_coprime_stride(num_nodes, rng)

        def get_batch_strided(batch_idx: int) -> SeedBatch:
            start = batch_idx * batch_size
            end = min(start + batch_size, num_nodes)
            # In-place ops on the arange result: one allocation for the
            # batch instead of four temporaries.
            idx = np.arange(start, end, dtype=np.int64)
            idx *= stride
            idx += offset
            idx %= num_nodes
            return SeedBatch(idx, idx)

        return get_batch_strided

    def get_batch_range(batch_idx: int) -> SeedBatch:
        start = batch_idx * batch_size
        end = min(start + batch_size, num_nodes)
        idx = np.arange(start, end, dtype=np.int64)
        return SeedBatch(idx, idx)

    return get_batch_range


def epoch_sampling_seed(seeds: np.random.SeedSequence | None) -> int | None:
    """Draw the next epoch's neighbor-sampling seed from a loader's sequence.

    Every epoch gets a fresh stream, so seeded training still resamples
    neighborhoods each epoch, and a loader built with the same seed replays
    the same sequence of epochs. ``None`` (an unseeded loader) stays None.
    """
    if seeds is None:
        return None
    (child,) = seeds.spawn(1)
    return int(child.generate_state(1, dtype=np.uint64)[0])


def pinned_block(specs: Sequence[tuple[tuple[int, ...], torch.dtype]]) -> list[torch.Tensor]:
    """Carve one pinned host allocation into empty tensors of ``specs``.

    A batch's tensors then cost one page-locked allocation instead of one
    each, and every view still copies to the device asynchronously.
    Regions are 8-byte aligned so each view keeps its dtype's alignment.
    """
    offsets: list[int] = []
    total = 0
    for shape, dtype in specs:
        total = (total + 7) & ~7
        offsets.append(total)
        total += math.prod(shape) * dtype.itemsize
    block = torch.empty(max(total, 1), dtype=torch.uint8, pin_memory=True)
    return [
        block[offset : offset + math.prod(shape) * dtype.itemsize].view(dtype).view(shape)
        for offset, (shape, dtype) in zip(offsets, specs, strict=True)
    ]


@dataclass(frozen=True)
class _RdmaFeatureSource:
    """Parsed ``feature_source`` URI.

    Constructed once by :func:`_parse_feature_source`; past the constructor
    no code inspects the URI string again.
    """

    server_addr: str


def _parse_features(features: npt.NDArray[Any], num_nodes: int) -> torch.Tensor:
    """Parse an in-memory feature matrix into its canonical torch view.

    Returns a zero-copy torch view over a contiguous ``float32`` array of
    shape ``[num_nodes, dim]`` — the one representation the per-batch gather
    consumes, so no batch ever re-converts dtype or layout.

    Raises:
        ValueError: If the array is not 2D, not ``float32``, or its row count
            does not match the graph. Callers that hold float64 must cast
            explicitly before this boundary.
    """
    arr = np.asarray(features)
    if arr.dtype != np.float32:
        raise ValueError(
            f"features must be float32 (got {arr.dtype}); cast explicitly before passing"
        )
    arr = np.ascontiguousarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {arr.shape}")
    if arr.shape[0] != num_nodes:
        raise ValueError(
            f"features.shape[0]={arr.shape[0]} does not match graph num_nodes={num_nodes}"
        )
    return torch.from_numpy(arr)


_RDMA_SCHEME = "rdma://"


def _parse_feature_source(feature_source: str | None) -> _RdmaFeatureSource | None:
    """Parse the ``feature_source`` URI into its typed form.

    Raises:
        ValueError: If the string does not use a recognized scheme — an
            unrecognized source must fail here rather than be silently
            ignored by the sampling path.
        RuntimeError: If RDMA is requested but the extension was built
            without the ``gpudirect`` Cargo feature.
    """
    if feature_source is None:
        return None
    if not feature_source.startswith(_RDMA_SCHEME):
        raise ValueError(
            f"unrecognized feature_source {feature_source!r}: expected 'rdma://host:port'"
        )
    if not HAS_GPUDIRECT:
        raise RuntimeError(
            "feature_source='rdma://' requires a wheel built with the "
            "`gpudirect` Cargo feature (maturin develop --features gpudirect)."
        )
    addr = feature_source[len(_RDMA_SCHEME) :]
    if not addr:
        raise ValueError("feature_source 'rdma://' is missing the host:port address")
    return _RdmaFeatureSource(server_addr=addr)


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

    Each ``__iter__`` derives a fresh child generator from the loader's seed
    so epochs are reproducible (when ``seed`` is set) and concurrent
    iterations do not share mutable RNG state.

    Batches follow PyG's conventions: the first ``batch_size`` entries of
    ``n_id`` are the seeds and ``input_id`` holds each seed's position in
    ``input_nodes``. ``seed_index`` additionally gives each seed's local
    index into ``n_id``, one per input seed, so duplicate seeds map to the
    same local node.

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
    _seed: int | None
    _rng: np.random.Generator
    _max_degree: int | None
    _cumulative: bool
    _subgraph_type: SubgraphType
    _track_edge_ids: bool
    _metrics: LoaderMetrics
    _temporal_strategy: TemporalStrategy | None
    _disjoint: bool
    _neighbor_sampler: Callable[..., Data] | None
    _rdma: _RdmaFeatureSource | None

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
        device: str | None = None,
        transfer_depth: int = 2,
        transform: Callable[[Data], Data] | None = None,
        feature_source: str | None = None,
        gpu_id: int = 0,
        input_time: npt.NDArray[np.float64] | None = None,
        temporal_strategy: TemporalStrategy | None = None,
        disjoint: bool = False,
        seed: int | None = None,
        generator: np.random.Generator | None = None,
        max_degree: int | None = None,
        cumulative: bool = False,
        subgraph_type: SubgraphType = "directional",
        track_edge_ids: bool = True,
        neighbor_sampler: Callable[..., Data] | None = None,
        features: npt.NDArray[Any] | None = None,
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
                the graph. Can be a torch.Tensor, numpy array, or list. A
                boolean tensor/array is treated as a PyG-style mask and
                converted to the indices of its ``True`` entries.
            batch_size: Number of seed nodes per batch.
            shuffle: Whether to shuffle nodes between epochs.
            replace: Whether to sample neighbors with replacement.
            num_workers: Number of Rust sampler threads feeding the
                prefetch pipeline (PyG-compatible semantics, but threads in
                the Rust backend instead of DataLoader worker processes).
                0 and 1 both mean a single sampler thread; higher values
                scale sampling throughput when sampling is the bottleneck.
            prefetch_factor: Number of batches to prefetch ahead in the
                sampling pipeline. Higher values reduce GPU stalls but use
                more memory.
            pin_memory: If True, tensors are allocated in pinned (page-locked)
                memory, enabling faster async GPU transfers with non_blocking=True.
            device: Optional CUDA device (e.g. ``"cuda"`` or ``"cuda:1"``).
                When set, each batch is moved to it on a dedicated copy
                stream, double-buffered so batch N+1's transfer overlaps
                batch N's training. Implies pinning the host tensors. A
                non-CUDA device or a machine without CUDA makes this a no-op
                passthrough (batches stay on the host).
            transfer_depth: Batches kept in flight on the copy stream when
                ``device`` is set (default 2 = classic double buffering).
            transform: Optional callable to apply to each batch after sampling.
                Receives a PyG Data object and should return a Data object.
            feature_source: RDMA feature server address as "rdma://host:port".
                When set, features are gathered via GPUDirect RDMA directly
                into VRAM. Requires Linux, libibverbs, CUDA, and nvidia-peermem.
            gpu_id: CUDA device ordinal for RDMA features (default: 0).
                Only used when feature_source is set.
            input_time: Not supported — the prefetch pipeline has no channel
                for per-seed timestamps, so passing this raises
                ``NotImplementedError``. Use ``aethergraph.Sampler.sample``
                with ``input_times`` for per-seed temporal sampling.
            temporal_strategy: ``"uniform"`` or ``"last"``. When set, seeds
                sample with unbounded time (for ``"last"``, the k most recent
                edges). Requires ``graph.set_timestamps()``.
            disjoint: If True, each seed gets an isolated subgraph with no
                node dedup across seeds. Output includes a ``batch`` tensor
                mapping each node to its seed's position in the batch.
            seed: Random seed. Controls both seed-node shuffling (a fresh
                child generator is derived per epoch, so epochs are
                reproducible) and the Rust sampler's neighbor selection.
                When set, each epoch draws a fresh sampling seed from it,
                so neighborhoods are resampled every epoch while the whole
                run stays reproducible; within an epoch each batch reseeds
                from its submission order, and the Rust prefetch layer
                yields in that order, so the stream is bit-identical to
                ``num_workers=1``. None uses OS entropy (non-reproducible).
            generator: Explicit numpy ``Generator`` for seed-node shuffling.
                Takes precedence over ``seed`` for shuffling. The Rust sampler
                still uses ``seed`` for neighbor selection.
            max_degree: Degree above which a node counts as a hub. Uniform
                draws never cap; weighted and uniform-temporal sampling draw
                from ``max_degree`` random positions of a hub's row. None
                never caps.
            cumulative: When False (the default, PyG semantics), each hop
                expands only the previous hop's new nodes. When True, each
                hop re-expands every node seen so far, adding only edges no
                earlier hop emitted.
            subgraph_type: One of ``"directional"``, ``"induced"``,
                ``"bidirectional"``. Controls which edges are kept in the
                output subgraph.
            track_edge_ids: Whether to track original edge IDs (``e_id``).
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
            ValueError: If ``num_workers`` is negative, if both ``features``
                and ``feature_path`` are set, or if ``features.shape[0]``
                does not match the graph's node count.
        """
        super().__init__()

        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")
        # PyG semantics, Rust execution: instead of forking DataLoader
        # worker processes, num_workers sizes the Rust sampler thread pool
        # feeding the prefetch pipeline. 0 keeps the single-threaded
        # pipeline (sampling still overlaps training).
        self._sampler_threads: int = max(1, num_workers)

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
        self._rdma = _parse_feature_source(feature_source)
        self._temporal_strategy = temporal_strategy
        self._disjoint = disjoint
        self._seed = seed
        self._max_degree = max_degree
        self._cumulative = cumulative
        self._subgraph_type = subgraph_type
        self._track_edge_ids = track_edge_ids
        self._neighbor_sampler = neighbor_sampler

        # Parse fanout / strategy once at construction so illegal configs
        # fail here rather than on the first ``__iter__``.
        self._sampling = SamplingConfig(
            num_neighbors=num_neighbors,
            replace=replace,
            seed=seed,
            max_degree=max_degree,
            cumulative=cumulative,
            weighted=weighted,
            subgraph_type=subgraph_type,
            track_edge_ids=track_edge_ids,
            temporal_strategy=temporal_strategy,
            disjoint=disjoint,
            deterministic=False,
        )
        if features is not None and feature_path is not None:
            raise ValueError("pass either `features` or `feature_path`, not both")
        self._feature_path: Path | None = Path(feature_path) if feature_path is not None else None

        num_nodes = self.graph.num_nodes
        self._feat_torch: torch.Tensor | None = (
            _parse_features(features, num_nodes) if features is not None else None
        )

        if input_nodes is None:
            self._input_nodes = None
            self._num_input_nodes = num_nodes
        else:
            self._input_nodes = normalize_input_nodes(input_nodes, num_nodes=num_nodes)
            self._num_input_nodes = len(self._input_nodes)

        # The prefetch pipeline submits seeds only — there is no channel for
        # per-seed timestamps — so accepting `input_time` here would silently
        # sample as if every seed had unbounded time. Fail loudly and point at
        # the API that does carry per-seed times. `temporal_strategy` alone is
        # fine: seeds sample with unbounded time (for "last", the k most
        # recent edges).
        if input_time is not None:
            raise NotImplementedError(
                "per-seed input_time is not supported by NeighborLoader's prefetch "
                "pipeline; use aethergraph.Sampler.sample(seeds, input_times=...) "
                "for temporal sampling with per-seed times"
            )

        # Device availability is immutable for the process lifetime; resolve
        # the pinning decision once instead of per tensor per batch.
        self._pin: bool = bool(pin_memory and torch.cuda.is_available())

        # Optional device transfer: when a CUDA device is requested, each
        # yielded batch is moved to it on a dedicated copy stream, double-
        # buffered so batch N+1 crosses PCIe while batch N trains. Moving to
        # the device implies pinning the host source (async H2D needs it).
        self._device: torch.device | None = torch.device(device) if device is not None else None
        self._transfer_depth = transfer_depth
        if self._device is not None and self._device.type == "cuda":
            self._pin = self._pin or torch.cuda.is_available()

        # Master generator for seed-node shuffling. An explicit `generator`
        # wins; otherwise seed from `seed` (reproducible) or OS entropy.
        # `__iter__` spawns a fresh child from this per epoch so iteration is
        # reproducible and never shares mutable RNG state across concurrent
        # iterators.
        self._rng = generator if generator is not None else np.random.default_rng(seed)
        # Neighbor sampling draws each epoch's seed from its own sequence,
        # independent of the shuffle generator.
        self._sampling_seeds = np.random.SeedSequence(seed) if seed is not None else None
        self._metrics = LoaderMetrics()

    @property
    def metrics(self) -> LoaderMetrics:
        """Get metrics from the most recent epoch.

        Returns:
            LoaderMetrics with timing and prefetch statistics.
        """
        return self._metrics

    def __iter__(self) -> Iterator[Data]:
        """Iterate over batches, optionally pipelined onto a CUDA device.

        Delegates to :meth:`_iter_batches` for sampling. When ``device`` was
        given, the batches are wrapped in a :class:`DeviceTransferPipeline`
        so each one's host-to-device copy runs on a dedicated stream and
        overlaps the consumer's compute on the previous batch. Without a
        device, the sampled (host) batches are yielded directly.
        """
        batches = self._iter_batches()
        if self._device is not None:
            return iter(DeviceTransferPipeline(batches, self._device, depth=self._transfer_depth))
        return batches

    def _iter_batches(self) -> Iterator[Data]:
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
                - n_id: Global node IDs [num_nodes], seeds first
                - batch_size: Number of seed nodes in this batch
                - input_id: Each seed's position in ``input_nodes``
                - seed_index: Each seed's local index into ``n_id``
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

        # Fresh child generator per epoch: concurrent iterators don't share
        # mutable RNG state, and the sequence is reproducible when the loader
        # was seeded.
        rng = self._rng.spawn(1)[0]
        get_batch = make_batch_getter(self._input_nodes, num_nodes, batch_size, self.shuffle, rng)

        if self._neighbor_sampler is not None:
            for batch_idx in range(num_batches):
                data = self._neighbor_sampler(self.graph, get_batch(batch_idx).seeds)
                if self.transform is not None:
                    data = self.transform(data)
                yield data
            return

        sampling = dataclasses.replace(
            self._sampling, seed=epoch_sampling_seed(self._sampling_seeds)
        )
        rust_config = sampling._to_rust()

        if self._rdma is not None:
            # Upper bound on nodes per batch: batch_size * product of the
            # per-hop fanout, doubled for slack. `math.prod` is arbitrary
            # precision so it can't overflow into a negative int64, and the
            # fanout is clamped to >= 1 so a hop of 0 doesn't zero it out.
            fanout = max(1, math.prod(self.num_neighbors))
            max_batch_nodes = self.batch_size * fanout * 2
            sampler = RustNeighborLoader.with_rdma_features(
                self.graph,
                rust_config,
                self._rdma.server_addr,
                self.gpu_id,
                max_batch_nodes=max_batch_nodes,
                prefetch_depth=self.prefetch_factor,
                sampler_threads=self._sampler_threads,
            )

            def receive_gpu(
                s: RustNeighborLoader,
            ) -> tuple[int, tuple[RustSampledSubgraph, torch.Tensor]] | None:
                result = s.next_batch_gpu()
                if result is None:
                    return None
                batch_idx, subgraph, dlpack_capsule = result
                # torch re-exports from_dlpack without an explicit re-export.
                x_gpu = torch.from_dlpack(dlpack_capsule)  # type: ignore[attr-defined]
                return batch_idx, (subgraph, x_gpu)

            yield from self._drive_epoch(
                sampler,
                num_batches,
                get_batch,
                receive_gpu,
                lambda got, input_id: self._to_pyg_data_gpu(got[0], got[1], input_id),
                "epoch_rdma",
            )
            return

        feature_path = self._feature_path

        if feature_path is not None:
            sampler = RustNeighborLoader.with_features(
                self.graph,
                rust_config,
                feature_path,
                self.prefetch_factor,
                self._sampler_threads,
            )
        else:
            sampler = RustNeighborLoader(
                self.graph,
                rust_config,
                self.prefetch_factor,
                self._sampler_threads,
            )

        def receive_cpu(
            s: RustNeighborLoader,
        ) -> tuple[int, tuple[RustSampledSubgraph, npt.NDArray[np.float32] | None]] | None:
            result = s.next_batch()
            if result is None:
                return None
            batch_idx, subgraph, features = result
            return batch_idx, (subgraph, features)

        yield from self._drive_epoch(
            sampler,
            num_batches,
            get_batch,
            receive_cpu,
            lambda got, input_id: self._to_pyg_data(got[0], got[1], input_id),
            "epoch",
        )

    def _drive_epoch(
        self,
        sampler: RustNeighborLoader,
        num_batches: int,
        get_batch: Callable[[int], SeedBatch],
        receive: Callable[[RustNeighborLoader], tuple[int, _Received] | None],
        to_data: Callable[[_Received, npt.NDArray[np.int64]], Data],
        span_name: str,
    ) -> Iterator[Data]:
        """Run one epoch through the Rust prefetch pipeline.

        Primes the sliding window, then interleaves receive and resubmit so
        ``prefetch_factor`` batches are always in flight. ``receive`` pulls
        one finished batch and the ``batch_idx`` it was submitted under
        (``None`` means the backend stopped early); ``to_data`` converts it,
        given that batch's ``input_id``, to a PyG ``Data`` object. Metrics
        and the tracing span are finalized in all exit paths, including
        early ``break`` by the consumer.
        """
        submitted = 0
        received = 0
        epoch_start = time.perf_counter()
        # Results arrive in completion order across an unseeded pool; each
        # carries its batch_idx, which finds its input_id here.
        input_ids: dict[int, npt.NDArray[np.int64]] = {}

        def submit_next() -> None:
            nonlocal submitted
            batch = get_batch(submitted)
            input_ids[submitted] = batch.input_id
            sampler.submit(submitted, batch.seeds)
            submitted += 1

        tracer = get_tracer()
        epoch_span = tracer.start_span(span_name) if tracer else None
        if epoch_span:
            epoch_span.set_attribute("num_batches", num_batches)
            epoch_span.set_attribute("batch_size", self.batch_size)
            epoch_span.set_attribute("num_neighbors", str(self.num_neighbors))
            epoch_span.set_attribute("prefetch_factor", self.prefetch_factor)

        try:
            while submitted < min(self.prefetch_factor, num_batches):
                submit_next()

            while received < num_batches:
                got = receive(sampler)
                if got is None:
                    raise RuntimeError(
                        f"NeighborLoader backend stopped early: received {received} "
                        f"of {num_batches} batches"
                    )
                batch_idx, payload = got
                received += 1

                if submitted < num_batches:
                    submit_next()

                data = to_data(payload, input_ids.pop(batch_idx))
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
        input_id: npt.NDArray[np.int64],
    ) -> Data:
        """Convert a sampled subgraph to a PyG Data object.

        Feature sources are checked in priority order:
            1. File-based features (loaded by Rust sampler via io_uring/mmap)
            2. In-memory features (the loader's ``features`` matrix)
            3. No features (``x`` will be ``None``)

        Edge indices use pre-computed local IDs from Rust's
        ``edge_index_local()``. In disjoint mode, a ``batch`` tensor is
        attached mapping each node to its seed.

        With pinning, every tensor of the batch is a view into one pinned
        block (:func:`pinned_block`): one page-locked allocation per batch,
        and each tensor still transfers to the device asynchronously.
        Without it, the numpy arrays are wrapped zero-copy.

        Args:
            subgraph: Rust SampledSubgraph containing nodes, edges, and seeds.
            file_features: Features gathered by the Rust sampler, or None.
            input_id: Each seed's position in ``input_nodes``.

        Returns:
            PyG Data object with x, edge_index, e_id, n_id, input_id,
            seed_index, and optionally batch (disjoint mode).
        """
        arrays: dict[str, npt.NDArray[np.int64]] = {
            "n_id": subgraph.nodes,
            "edge_index": subgraph.edge_index_local,
            "seed_index": subgraph.seed_indices,
            "input_id": input_id,
        }
        if self._track_edge_ids:
            arrays["e_id"] = subgraph.edge_ids
        batch_arr = subgraph.batch
        if batch_arr is not None:
            arrays["batch"] = batch_arr
        num_nodes = arrays["n_id"].shape[0]

        feature_dim: int | None = None
        if file_features is not None:
            feature_dim = file_features.shape[1]
        elif self._feat_torch is not None:
            feature_dim = self._feat_torch.shape[1]

        x: torch.Tensor | None = None
        tensors: dict[str, torch.Tensor]
        if self._pin:
            specs = [(a.shape, torch.int64) for a in arrays.values()]
            if feature_dim is not None:
                specs.append(((num_nodes, feature_dim), torch.float32))
            views = pinned_block(specs)
            tensors = {}
            for (name, arr), view in zip(arrays.items(), views, strict=False):
                view.copy_(torch.from_numpy(arr))
                tensors[name] = view
            if feature_dim is not None:
                x = views[-1]
        else:
            tensors = {name: torch.from_numpy(arr) for name, arr in arrays.items()}
            if feature_dim is not None:
                x = torch.empty((num_nodes, feature_dim), dtype=torch.float32)

        if x is not None:
            if file_features is not None:
                if self._pin:
                    x.copy_(torch.from_numpy(file_features))
                else:
                    x = torch.from_numpy(file_features)
            elif self._feat_torch is not None:
                # One gather pass straight into the destination.
                torch.index_select(self._feat_torch, 0, tensors["n_id"], out=x)

        seed_index = tensors["seed_index"]
        data = Data(
            x=x,
            edge_index=tensors["edge_index"],
            e_id=tensors.get("e_id"),
            n_id=tensors["n_id"],
            batch_size=seed_index.shape[0],
            input_id=tensors["input_id"],
            seed_index=seed_index,
            num_nodes=num_nodes,
            num_sampled_nodes=subgraph.num_sampled_nodes_per_hop,
            num_sampled_edges=subgraph.num_sampled_edges_per_hop,
        )
        if "batch" in tensors:
            data.batch = tensors["batch"]
        return data

    def _to_pyg_data_gpu(
        self,
        subgraph: RustSampledSubgraph,
        x_gpu: torch.Tensor,
        input_id: npt.NDArray[np.int64],
    ) -> Data:
        """Build PyG Data with features already on GPU via RDMA.

        Unlike ``_to_pyg_data``, features arrive as a CUDA tensor from
        GPUDirect RDMA (zero-copy DLPack capsule). The index arrays are
        packed into one pinned staging buffer and moved with a single
        asynchronous H2D copy; the returned tensors are on-device views
        into that transfer.

        Args:
            subgraph: Rust SampledSubgraph containing nodes, edges, and seeds.
            x_gpu: CUDA tensor of node features from RDMA gather.
            input_id: Each seed's position in ``input_nodes``.

        Returns:
            PyG Data object with all tensors on the same CUDA device.
        """
        arrays: dict[str, npt.NDArray[np.int64]] = {
            "n_id": subgraph.nodes,
            "edge_index": subgraph.edge_index_local,
            "seed_index": subgraph.seed_indices,
            "input_id": input_id,
        }
        if self._track_edge_ids:
            arrays["e_id"] = subgraph.edge_ids
        batch_arr = subgraph.batch
        if batch_arr is not None:
            arrays["batch"] = batch_arr

        # `non_blocking=True` is only honoured from pinned host memory —
        # pageable tensors silently fall back to synchronous copies.
        total = sum(a.size for a in arrays.values())
        stage = torch.empty(total, dtype=torch.int64, pin_memory=True)
        stage_np = stage.numpy()
        pos = 0
        for arr in arrays.values():
            stage_np[pos : pos + arr.size] = arr.reshape(-1)
            pos += arr.size
        packed = stage.to(x_gpu.device, non_blocking=True)

        on_device: dict[str, torch.Tensor] = {}
        pos = 0
        for name, arr in arrays.items():
            on_device[name] = packed[pos : pos + arr.size].view(arr.shape)
            pos += arr.size

        seed_index = on_device["seed_index"]
        data = Data(
            x=x_gpu,
            edge_index=on_device["edge_index"],
            e_id=on_device.get("e_id"),
            n_id=on_device["n_id"],
            batch_size=seed_index.shape[0],
            input_id=on_device["input_id"],
            seed_index=seed_index,
            num_nodes=arrays["n_id"].shape[0],
            num_sampled_nodes=subgraph.num_sampled_nodes_per_hop,
            num_sampled_edges=subgraph.num_sampled_edges_per_hop,
        )
        if "batch" in on_device:
            data.batch = on_device["batch"]
        return data

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
