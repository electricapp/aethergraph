"""Type stubs for `aethergraph._core` Rust extension module.

These stubs are hand-authored to mirror the PyO3 surface in
`crates/aethergraph-py/src/`. Keep them in sync — `mypy --strict` runs
against this file in CI.

Path-accepting APIs declare `str | os.PathLike[str]` since PyO3's
`PathBuf` accepts anything implementing `__fspath__`.
"""

import os
from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from aethergraph._types import SubgraphType as SubgraphType
from aethergraph._types import TemporalStrategy as TemporalStrategy

__version__: str
__author__: str

# Compile-time Cargo feature flags exposed as data. True when the wheel
# was built with `--features gpudirect`. Callers should test this before
# constructing RDMA-backed loaders rather than probing for method names.
HAS_GPUDIRECT: bool

# Test-only helper, only present in `gpudirect` builds. Wraps a raw CUDA
# device pointer in a DLPack capsule so `torch.from_dlpack` can ingest it.
def _dlpack_capsule_from_cuda_ptr(
    ptr: int, num_nodes: int, feature_dim: int, gpu_id: int
) -> Any: ...

SeedArray: TypeAlias = npt.NDArray[np.int64] | npt.NDArray[np.uint32]
PathLike: TypeAlias = str | os.PathLike[str]

# -- Exceptions --------------------------------------------------------------

class GraphLoadError(Exception):
    """Raised when graph loading fails."""

class SamplingError(Exception):
    """Raised when sampling fails."""

class CacheError(Exception):
    """Raised when cache operations fail."""

class ArrowConversionError(Exception):
    """Raised when Arrow conversion fails."""

# -- Graph -------------------------------------------------------------------

class CsrGraph:
    """Compressed-Sparse-Row graph. The canonical Python type for static graphs
    (re-exported as :class:`aethergraph.Graph`)."""

    @staticmethod
    def load(
        path: PathLike,
        *,
        storage: str = "auto",
        validation: str = "auto",
    ) -> CsrGraph:
        """Load a graph from disk. `storage` is one of
        ``"auto" | "mmap" | "owned"``; `validation` is one of
        ``"auto" | "header_only" | "offsets_only" | "full"``."""

    @staticmethod
    def from_edges(
        num_nodes: int,
        src: npt.NDArray[np.uint32],
        dst: npt.NDArray[np.uint32],
        weights: npt.NDArray[np.float32] | None = ...,
    ) -> CsrGraph: ...
    def save(self, path: PathLike) -> None: ...

    # The next 4 are #[getter]s — attribute access, no parens.
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def has_weights(self) -> bool: ...
    @property
    def has_timestamps(self) -> bool: ...

    # Methods (take an argument).
    def degree(self, node: int) -> int: ...
    def degrees(self) -> npt.NDArray[np.uint32]: ...
    def neighbors(self, node: int) -> npt.NDArray[np.uint32]: ...
    def batch_neighbors(self, nodes: list[int]) -> list[npt.NDArray[np.uint32]]: ...
    def neighbor_weights(self, node: int) -> npt.NDArray[np.float32] | None: ...
    def set_timestamps(self, timestamps: npt.NDArray[np.float64]) -> None: ...
    def stats(self) -> dict[str, Any]: ...
    def reorder_rabbit(self) -> npt.NDArray[np.uint32]: ...
    def permute(self, perm: npt.NDArray[np.uint32]) -> CsrGraph: ...
    def __len__(self) -> int: ...
    def __getitem__(self, node: int) -> npt.NDArray[np.uint32]: ...

# -- Free functions ----------------------------------------------------------

def save_features(path: PathLike, features: npt.NDArray[np.float32]) -> None:
    """Save node features in AETHFEAT format."""

# -- Sampling ----------------------------------------------------------------

class SamplingConfig:
    """Configuration for neighborhood sampling."""

    def __init__(
        self,
        num_neighbors: list[int],
        replace: bool = False,
        seed: int | None = None,
        max_degree: int | None = None,
        cumulative: bool = True,
        weighted: bool = False,
        subgraph_type: SubgraphType = "directional",
        track_edge_ids: bool = True,
        temporal_strategy: TemporalStrategy | None = None,
        disjoint: bool = False,
        deterministic: bool = False,
        telemetry: SamplingTelemetry | None = None,
    ) -> None: ...
    @property
    def num_neighbors(self) -> list[int]: ...
    @property
    def replace(self) -> bool: ...
    @property
    def seed(self) -> int | None: ...
    @property
    def max_degree(self) -> int | None: ...
    @property
    def cumulative(self) -> bool: ...
    @property
    def weighted(self) -> bool: ...
    @property
    def subgraph_type(self) -> SubgraphType: ...
    @property
    def track_edge_ids(self) -> bool: ...
    @property
    def temporal_strategy(self) -> TemporalStrategy | None: ...
    @property
    def disjoint(self) -> bool: ...
    @property
    def deterministic(self) -> bool: ...

class SamplingTelemetry:
    """Lightweight, lock-free sampling metrics collector."""

    def __init__(self) -> None: ...
    def summary(self) -> dict[str, Any]: ...
    def reset(self) -> None: ...

class SampledSubgraph:
    """One sampled subgraph. All array accessors are `@property` — no parens.

    The arrays are `int64` because PyTorch's index dtype is `int64`.

    Ownership contract: every array accessor returns a Python-owned numpy
    array that never aliases Rust memory, so wrapping it zero-copy
    (``torch.from_numpy``) needs no defensive copy. The *same* array object
    may be returned on every access (accessors cache), so treat the result
    as immutable — a mutation would be visible through every later access
    and through ``to_dict()``. Copy first if you need to write."""

    # Counts.
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def num_seeds(self) -> int: ...

    # Arrays.
    @property
    def nodes(self) -> npt.NDArray[np.int64]: ...
    @property
    def seeds(self) -> npt.NDArray[np.int64]: ...
    @property
    def edge_index(self) -> npt.NDArray[np.int64]: ...
    @property
    def edge_index_local(self) -> npt.NDArray[np.int64]: ...
    @property
    def edge_ids(self) -> npt.NDArray[np.int64]: ...
    @property
    def seed_indices(self) -> npt.NDArray[np.int64]: ...
    @property
    def batch(self) -> npt.NDArray[np.int64] | None: ...
    @property
    def num_sampled_nodes_per_hop(self) -> list[int]: ...
    @property
    def num_sampled_edges_per_hop(self) -> list[int]: ...
    def to_dict(self) -> dict[str, Any]: ...
    def to_arrow(self) -> dict[str, Any]: ...
    def __len__(self) -> int: ...

class NeighborSampler:
    """Single-graph neighborhood sampler."""

    def __init__(self, graph: CsrGraph, config: SamplingConfig) -> None: ...
    def sample(
        self,
        seeds: Sequence[int] | SeedArray,
        input_times: npt.NDArray[np.float64] | None = None,
    ) -> SampledSubgraph: ...

class ParallelBatchSampler:
    """Rayon-parallel batch sampler for high-throughput training.

    Output is statistically reproducible given a seed; for bit-deterministic
    output set ``SamplingConfig.deterministic = True``."""

    def __init__(self, graph: CsrGraph, config: SamplingConfig) -> None: ...
    def sample_batches(self, batches: list[Sequence[int] | SeedArray]) -> list[SampledSubgraph]: ...

# -- DynamicGraph + writer guard --------------------------------------------

class DynamicGraph:
    """Lock-free dynamic graph supporting concurrent inserts + reads."""

    def __init__(self, num_vertices: int, arena_mb: int = 256) -> None: ...
    @staticmethod
    def from_edges(
        num_vertices: int,
        src: npt.NDArray[np.uint32],
        dst: npt.NDArray[np.uint32],
        arena_mb: int = 256,
    ) -> DynamicGraph: ...
    @staticmethod
    def open_with_wal(
        path: PathLike,
        num_vertices: int,
        arena_mb: int = 256,
    ) -> DynamicGraph:
        """Open a graph backed by an append-only write-ahead log. Existing
        records at ``path`` are replayed before this returns; torn tails are
        truncated."""
    def insert_edge(self, src: int, dst: int) -> bool: ...
    def insert_edges(
        self,
        src: npt.NDArray[np.uint32],
        dst: npt.NDArray[np.uint32],
    ) -> int: ...
    def degree(self, vertex: int) -> int: ...
    def has_edge(self, src: int, dst: int) -> bool: ...
    def neighbors(self, vertex: int) -> npt.NDArray[np.int64]: ...
    def neighbors_u32(self, vertex: int) -> npt.NDArray[np.uint32]: ...
    @property
    def current_epoch(self) -> int:
        """Monotonic version counter — advances on every successful writer
        commit. Pin before a multi-source read to coordinate consistency
        with other subsystems sharing the same `EpochClock`."""
    @property
    def num_vertices(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def arena_used(self) -> int: ...
    @property
    def arena_capacity(self) -> int: ...
    def snapshot(self) -> CsrGraph: ...
    def __len__(self) -> int: ...

# -- Prefetch loader (low-level) --------------------------------------------

class PrefetchStats:
    """Counters from the prefetch worker."""

    @property
    def hits(self) -> int: ...
    @property
    def misses(self) -> int: ...
    @property
    def total(self) -> int: ...
    @property
    def hit_rate(self) -> float: ...
    @property
    def sample_time_ns(self) -> int: ...
    @property
    def feature_load_time_ns(self) -> int: ...
    def to_dict(self) -> dict[str, Any]: ...

class NeighborLoader:
    """Prefetching neighbor loader. Spawns `sampler_threads` worker
    threads over an MPMC work queue (plus one feature-loader thread when
    features are configured); bounded submission and result channels apply
    backpressure both ways. Results arrive unordered across the pool —
    each subgraph carries its own seeds, so consumers never rely on
    arrival order."""

    def __init__(
        self,
        graph: CsrGraph,
        config: SamplingConfig,
        prefetch_depth: int = 2,
        sampler_threads: int = 1,
    ) -> None: ...
    @staticmethod
    def with_features(
        graph: CsrGraph,
        config: SamplingConfig,
        feature_path: PathLike,
        prefetch_depth: int = 2,
        sampler_threads: int = 1,
    ) -> NeighborLoader: ...
    @staticmethod
    def new_nvme(
        graph_path: PathLike,
        config: SamplingConfig,
        prefetch_depth: int = 2,
    ) -> NeighborLoader: ...
    def submit(self, batch_id: int, seeds: SeedArray | list[int]) -> None: ...
    def submit_epoch(self, batches: list[Any]) -> None: ...
    def next(self) -> SampledSubgraph | None: ...
    def try_next(self) -> SampledSubgraph | None: ...
    def next_with_features(
        self,
    ) -> tuple[SampledSubgraph, npt.NDArray[np.float32] | None] | None: ...

    # The two methods below only exist on wheels built with the `gpudirect`
    # Cargo feature. Default builds raise AttributeError on access. Callers
    # gate their usage on the module-level `HAS_GPUDIRECT` constant.
    @staticmethod
    def with_rdma_features(
        graph: CsrGraph,
        config: SamplingConfig,
        server_addr: str,
        gpu_id: int = 0,
        max_batch_nodes: int = 65536,
        prefetch_depth: int = 2,
        gid_index: int = 1,
        sampler_threads: int = 1,
    ) -> NeighborLoader: ...
    def next_with_gpu_features(self) -> tuple[SampledSubgraph, Any] | None: ...
    @property
    def feature_dim(self) -> int | None: ...
    @property
    def has_features(self) -> bool: ...
    @property
    def prefetch_depth(self) -> int: ...
    def shutdown(self) -> None: ...
    def stats(self) -> PrefetchStats: ...

# -- Metrics rollup ----------------------------------------------------------

class MetricsSnapshot:
    """Cross-subsystem metrics rollup. Built from optional telemetry / stats
    wrappers; serializes as Prometheus text-exposition format."""

    @staticmethod
    def collect(
        sampling: SamplingTelemetry | None = None,
        feature_load: FeatureLoadTelemetry | None = None,
        prefetch: PrefetchStats | None = None,
    ) -> MetricsSnapshot: ...
    def to_prometheus(self) -> str: ...

# -- FeatureStore ------------------------------------------------------------

class FeatureStore:
    """Memory-mapped feature lookup."""

    @staticmethod
    def load(path: PathLike, telemetry: bool = False) -> FeatureStore: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def feature_dim(self) -> int: ...
    def get(self, node: int) -> npt.NDArray[np.float32]: ...
    def get_batch(self, nodes: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]: ...
    def features(self) -> npt.NDArray[np.float32]: ...
    def telemetry(self) -> FeatureLoadTelemetry | None: ...

class FeatureData:
    """Mutable feature builder."""

    def __init__(self, num_nodes: int, feature_dim: int) -> None: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def feature_dim(self) -> int: ...
    def get(self, node: int) -> npt.NDArray[np.float32]: ...
    def set(self, node: int, features: npt.NDArray[np.float32]) -> None: ...
    def save(self, path: PathLike) -> None: ...

class FeatureLoadTelemetry:
    """Counters from FeatureStore reads."""

    @property
    def single_gets(self) -> int: ...
    @property
    def batch_gets(self) -> int: ...
    @property
    def total_nodes_loaded(self) -> int: ...
    @property
    def total_features_loaded(self) -> int: ...
    @property
    def total_bytes_loaded(self) -> int: ...
    def total_time_secs(self) -> float: ...
    def throughput_features_per_sec(self) -> float: ...
    def throughput_gb_per_sec(self) -> float: ...
    def avg_batch_size(self) -> float: ...
    def summary(self) -> dict[str, Any]: ...

# -- Hetero graph ------------------------------------------------------------

class HeteroCsrGraph:
    """Heterogeneous CSR graph: multiple node + edge types."""

    @staticmethod
    def from_edge_arrays(
        node_types: dict[str, int],
        edge_types: list[tuple[str, str, str, npt.NDArray[np.uint32], npt.NDArray[np.uint32]]],
    ) -> HeteroCsrGraph: ...
    def node_types(self) -> list[str]: ...
    def edge_types(self) -> list[tuple[str, str, str]]: ...
    def num_nodes(self, node_type: str) -> int: ...
    def num_edges(self, src_type: str, rel: str, dst_type: str) -> int: ...
    def total_nodes(self) -> int: ...
    def total_edges(self) -> int: ...

class HeteroSamplingConfig:
    """Configuration for heterogeneous neighborhood sampling."""

    def __init__(
        self,
        num_neighbors: dict[tuple[str, str, str], list[int]],
        replace: bool = False,
        seed: int | None = None,
        max_degree: int | None = None,
    ) -> None: ...
    @property
    def num_neighbors(self) -> dict[tuple[str, str, str], list[int]]: ...
    @property
    def replace(self) -> bool: ...
    @property
    def seed(self) -> int | None: ...
    @property
    def max_degree(self) -> int | None: ...
    @property
    def num_hops(self) -> int: ...

class HeteroSampledSubgraph:
    """One sampled subgraph from `HeteroNeighborSampler`.

    Array accessors return Python-owned arrays that never alias Rust
    memory; here each access allocates fresh, so results are independently
    mutable (unlike `SampledSubgraph`, whose accessors cache). `sample`
    accepts int64 seed arrays directly — IDs are range-checked at this
    boundary, so callers never pre-narrow to uint32."""

    @property
    def node_types(self) -> list[str]: ...
    @property
    def edge_types(self) -> list[tuple[str, str, str]]: ...
    @property
    def seed_type(self) -> str: ...
    @property
    def seeds(self) -> npt.NDArray[np.int64]: ...
    def nodes(self, node_type: str) -> npt.NDArray[np.int64]: ...
    def nodes_u32(self, node_type: str) -> npt.NDArray[np.uint32]: ...
    def edge_index_local(self, src: str, rel: str, dst: str) -> npt.NDArray[np.int64]: ...

class HeteroNeighborSampler:
    """Heterogeneous neighborhood sampler."""

    def __init__(self, graph: HeteroCsrGraph, config: HeteroSamplingConfig) -> None: ...
    def sample(
        self,
        seed_type: str,
        seeds: SeedArray | list[int],
    ) -> HeteroSampledSubgraph: ...

class HeteroNeighborLoader:
    """Prefetching heterogeneous neighbor loader. Spawns `sampler_threads`
    worker threads over an MPMC work queue — the same pipeline as
    `NeighborLoader`; bounded submission and result channels apply
    backpressure both ways. Results arrive unordered across the pool —
    each subgraph carries its own seed type and seeds, so consumers never
    rely on arrival order. Every submitted batch is rooted at the
    `seed_type` fixed at construction."""

    def __init__(
        self,
        graph: HeteroCsrGraph,
        config: HeteroSamplingConfig,
        seed_type: str,
        prefetch_depth: int = 2,
        sampler_threads: int = 1,
    ) -> None: ...
    def submit(self, batch_id: int, seeds: SeedArray | list[int]) -> None: ...
    def next(self) -> HeteroSampledSubgraph | None: ...
    @property
    def prefetch_depth(self) -> int: ...
    def shutdown(self) -> None: ...
    def stats(self) -> PrefetchStats: ...

# -- FeatureCache (async) ----------------------------------------------------

class FeatureCacheConfig:
    """Configuration for the tiered feature cache."""

    def __init__(
        self,
        gpu_capacity: int = 10_000,
        cpu_capacity: int = 1_000_000,
        feature_dim: int = 128,
        nvme_path: PathLike | None = None,
    ) -> None: ...
    @property
    def gpu_capacity(self) -> int: ...
    @property
    def cpu_capacity(self) -> int: ...
    @property
    def feature_dim(self) -> int: ...
    @property
    def nvme_path(self) -> str | None: ...

class FeatureCache:
    """Tiered (GPU/CPU/NVMe) feature cache. Async API."""

    @staticmethod
    async def create(config: FeatureCacheConfig) -> FeatureCache: ...
    async def get(self, node: int) -> npt.NDArray[np.float32]: ...
    async def get_batch(self, nodes: list[int]) -> npt.NDArray[np.float32]: ...
    async def insert(self, node: int, features: npt.NDArray[np.float32]) -> None: ...
    def stats(self) -> dict[str, int | float]: ...
    def print_stats(self) -> None: ...

# -- AsyncFeatureStore -------------------------------------------------------

class AsyncFeatureStore:
    """Async, io_uring-accelerated feature lookup (Linux fastest)."""

    @staticmethod
    async def load(path: PathLike, telemetry: bool = False) -> AsyncFeatureStore: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def feature_dim(self) -> int: ...
    async def get(self, node: int) -> npt.NDArray[np.float32]: ...
    async def get_batch(self, nodes: list[int]) -> npt.NDArray[np.float32]: ...
    def telemetry(self) -> FeatureLoadTelemetry | None: ...
