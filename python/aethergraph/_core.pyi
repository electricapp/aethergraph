"""Type stubs for `aethergraph._core` Rust extension module.

These stubs are hand-authored to mirror the PyO3 surface in
`crates/aethergraph-py/src/`. Keep them in sync — `mypy --strict` runs
against this file in CI.

Path-accepting APIs declare `str | os.PathLike[str]` since PyO3's
`PathBuf` accepts anything implementing `__fspath__`.
"""

import os
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Literal, Self, TypeAlias

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

# True when this build can interleave the graph body across NUMA nodes and
# pin the sampler pool to them. Says the code is compiled in, not that it
# took effect: placement is inert on a single-node machine.
HAS_NUMA: bool

# True on Linux builds carrying the memfd-backed cross-process store.
# `SharedFeatureStore` is absent entirely when this is False.
HAS_SHARED_STORE: bool

# True on Linux builds carrying perf_event_open counters. `PerfCounters`
# is absent entirely when this is False.
HAS_PERF_COUNTERS: bool

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
        storage: Literal["auto", "mmap", "owned"] = "auto",
        validation: Literal["auto", "header_only", "offsets_only", "full"] = "auto",
    ) -> CsrGraph:
        """Load a graph from disk, in either the flat or the compressed
        format.

        `validation` sets what an mmap load proves before returning; an
        owned or compressed load reads every byte and always proves the
        whole structure (`validation` then only selects checksum
        verification). A compressed file always decodes into owned
        arrays, so ``storage="mmap"`` raises on one."""

    @staticmethod
    def from_edges(
        num_nodes: int,
        src: npt.NDArray[np.uint32],
        dst: npt.NDArray[np.uint32],
        weights: npt.NDArray[np.float32] | None = ...,
    ) -> CsrGraph: ...
    def save(self, path: PathLike, *, compressed: bool = False) -> None:
        """Write the graph. `compressed` selects the succinct-coded format
        (Elias-Fano offsets, StreamVByte edges), typically 2-4x smaller at
        rest and readable by `load`."""

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
    def degrees_of(self, nodes: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        """Degrees of just these nodes, in input order (out-of-range → 0).
        Touches only the requested nodes, unlike `degrees()`."""
    def neighbors(self, node: int) -> npt.NDArray[np.uint32]: ...
    def batch_neighbors(self, nodes: list[int]) -> list[npt.NDArray[np.uint32]]: ...
    def neighbor_weights(self, node: int) -> npt.NDArray[np.float32] | None: ...
    def set_timestamps(self, timestamps: npt.NDArray[np.float64]) -> None: ...
    def stats(self) -> dict[str, Any]: ...
    def reorder_rabbit(self) -> npt.NDArray[np.uint32]:
        """Rabbit Order permutation, ``perm[new_id] = old_id``. Deterministic.
        Raises `GraphLoadError` if the graph's edges fail validation."""
    def rabbit_partitions(self) -> npt.NDArray[np.uint32]:
        """Rabbit Order community of every node, dense ids from 0."""
    def reorder_rabbit_with_partitions(
        self,
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        """``(reorder_rabbit(), rabbit_partitions())`` from one pass."""
    def permute(self, perm: npt.NDArray[np.uint32]) -> CsrGraph: ...
    def __len__(self) -> int: ...
    def __getitem__(self, node: int) -> npt.NDArray[np.uint32]: ...

# -- Free functions ----------------------------------------------------------

def save_features(
    path: PathLike,
    features: npt.NDArray[np.float32],
    dtype: Literal["f32", "f16", "bf16"] = "f32",
) -> None:
    """Save node features in AETHFEAT format.

    `dtype` sets the on-disk element type. ``"f16"`` and ``"bf16"`` both
    halve the file; bf16 keeps f32's exponent range (so small magnitudes
    don't flush to zero and large ones don't saturate) at the cost of
    mantissa bits, which is the usual trade for trained embeddings.
    Readers pick the decoder up from the header — nothing at the call site
    changes."""

# -- Sampling ----------------------------------------------------------------

class SamplingConfig:
    """Configuration for neighborhood sampling.

    The constructor validates every field and raises ``ValueError`` on an
    empty or negative ``num_neighbors``, a non-positive ``max_degree``, or an
    unknown ``subgraph_type`` / ``temporal_strategy``. ``cumulative=False``
    is PyG's semantics (each hop expands the previous hop's new nodes);
    ``deterministic`` has no effect, since a fixed ``seed`` is already
    bit-reproducible."""

    def __init__(
        self,
        num_neighbors: list[int],
        replace: bool = False,
        seed: int | None = None,
        max_degree: int | None = None,
        cumulative: bool = False,
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
    and through ``to_dict()``. Copy first if you need to write.

    Edge orientation follows PyG's source-to-target convention: row 0 of
    ``edge_index`` / ``edge_index_local`` is the sampled neighbor and row 1
    the node whose adjacency row was expanded, so messages flow toward the
    seeds. ``edge_ids[i]`` is the CSR position of the stored edge from
    ``edge_index[1, i]`` to ``edge_index[0, i]``. A CSR row is read as the
    nodes a node aggregates from: for PyG parity on a directed graph, build
    the CSR from ``(dst, src)``; for an undirected graph (both directions
    stored) the two coincide. ``to_arrow()`` edge columns use the same
    orientation.

    ``num_sampled_nodes_per_hop`` is ``[seed nodes, hop 1, ..., hop k]`` and
    ``num_sampled_edges_per_hop`` has one entry per hop, as PyG expects."""

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
    """Single-graph neighborhood sampler.

    The constructor raises ``ValueError`` when ``weighted`` or
    ``temporal_strategy`` asks for edge data the graph lacks. ``sample``
    raises ``SamplingError`` for a seed outside the graph, or for
    ``input_times`` given without a temporal strategy or with a length other
    than the seed count; ``input_times=None`` leaves every seed unbounded."""

    def __init__(self, graph: CsrGraph, config: SamplingConfig) -> None: ...
    def sample(
        self,
        seeds: Sequence[int] | SeedArray,
        input_times: npt.NDArray[np.float64] | None = None,
    ) -> SampledSubgraph: ...

class ParallelBatchSampler:
    """Rayon-parallel batch sampler for high-throughput training.

    Each batch draws from an RNG stream derived from the config's seed and
    the batch's position among every batch this sampler has handled, so a
    fixed seed reproduces the same sequence of calls bit-for-bit at any
    thread count, and repeated calls draw fresh samples."""

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
    def acquire(self) -> GraphSnapshot:
        """Pin the latest committed snapshot. Immutable and strictly
        serializable; reads are lock-free while inserts continue. Holding
        it defers arena recycling of its state — drop it when done."""
    def __len__(self) -> int: ...

class GraphSnapshot:
    """Pinned, immutable view of a DynamicGraph at one committed epoch."""

    @property
    def epoch(self) -> int: ...
    @property
    def num_vertices(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    def degree(self, vertex: int) -> int: ...
    def has_edge(self, src: int, dst: int) -> bool: ...
    def neighbors(self, vertex: int) -> npt.NDArray[np.int64]: ...
    def neighbors_u32(self, vertex: int) -> npt.NDArray[np.uint32]: ...
    def to_static(self) -> CsrGraph:
        """Freeze into a static CSR graph — an atomic cut at this commit."""
    def __len__(self) -> int: ...

# -- Prefetch loader (low-level) --------------------------------------------

class PrefetchStats:
    """Counters from the prefetch worker. ``hits`` counts batches that were
    ready when requested, ``misses`` batches the consumer waited for;
    ``total`` is their sum."""

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
    backpressure both ways.

    With a config seed, results come back in submission order and each
    batch reseeds from its position in that order, so any pool size yields
    the same stream and one loader serves any number of epochs. Without a
    seed, results arrive as workers finish; `next_batch()` reports each
    one's `batch_idx`.

    Every method is safe from any thread: `shutdown()` (or leaving a `with`
    block) wakes threads blocked in `submit()` or `next*()`, and a worker
    failure is raised by the next call rather than after a timeout."""

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
    def submit(self, batch_idx: int, seeds: SeedArray | list[int]) -> None:
        """`batch_idx` is echoed back by `next_batch()`; any value is
        accepted, repeats and gaps included."""
    def submit_epoch(self, batches: list[Any]) -> None: ...
    def next(self) -> SampledSubgraph | None: ...
    def try_next(self) -> SampledSubgraph | None: ...
    def next_with_features(
        self,
    ) -> tuple[SampledSubgraph, npt.NDArray[np.float32] | None] | None: ...
    def next_batch(
        self,
    ) -> tuple[int, SampledSubgraph, npt.NDArray[np.float32] | None] | None:
        """`(batch_idx, subgraph, features)`, or None after `shutdown()`."""

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
    def next_batch_gpu(self) -> tuple[int, SampledSubgraph, Any] | None:
        """`(batch_idx, subgraph, capsule)`: the capsule is a DLPack tensor
        owning its VRAM, for `torch.from_dlpack`. None after `shutdown()`."""
    @property
    def feature_dim(self) -> int | None: ...
    @property
    def has_features(self) -> bool: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def prefetch_depth(self) -> int: ...
    def shutdown(self) -> None: ...
    def stats(self) -> PrefetchStats: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

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
    @staticmethod
    def load_paged(
        path: PathLike,
        budget_pages: int,
        degrees: npt.NDArray[np.uint32] | None = None,
        telemetry: bool = False,
    ) -> FeatureStore:
        """Demand-page the store from disk with at most `budget_pages`
        pages resident, instead of mapping it and letting the kernel choose.
        `budget_pages` must be at least 2, since one row can straddle a page
        boundary. `degrees` (one per node) makes eviction degree-weighted so
        hub nodes outlive leaves.

        Open the store in the process that reads it: a paged store opened
        before a fork raises in the child rather than reading zeros.

        Linux only; raises OSError where the kernel withholds userfaultfd,
        and `load` is the fallback."""

    @property
    def num_nodes(self) -> int: ...
    @property
    def feature_dim(self) -> int: ...
    def get(self, node: int) -> npt.NDArray[np.float32]: ...
    def get_batch(self, nodes: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]: ...
    def features(self) -> npt.NDArray[np.float32]: ...
    def telemetry(self) -> FeatureLoadTelemetry | None: ...
    def pager_stats(self) -> tuple[int, int] | None:
        """`(faults, evictions)` for a store opened with `load_paged`, or
        None for a mapped store. Linux only."""

class SharedFeatureStore:
    """A feature store held once in shared memory and mapped by many
    processes.

    The owner publishes a feature file into a sealed memfd and serves the
    descriptor on a Unix socket; each worker attaches and maps the same
    physical pages read-only, so N workers cost one copy of the matrix and
    attaching is a mmap rather than a read.

    Linux only — present only when `HAS_SHARED_STORE` is True."""

    @staticmethod
    def publish(path: PathLike) -> SharedFeatureStore:
        """Copy a feature file's payload into a fresh sealed shared region."""

    @staticmethod
    def attach(socket_path: PathLike) -> SharedFeatureStore:
        """Map a store being served at `socket_path` read-only."""

    def serve(self, socket_path: PathLike) -> None:
        """Serve this store to workers until `stop_serving()` or drop."""

    def stop_serving(self) -> None: ...
    @property
    def is_serving(self) -> bool: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def feature_dim(self) -> int: ...
    @property
    def shared_bytes(self) -> int: ...
    def get_batch(self, nodes: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]: ...

class PerfCounters:
    """Hardware performance counters around a block of work.

    Counters are thread-scoped and count user space only. Hosts that
    withhold PMU access grant fewer counters; those readings come back
    None rather than raising, so `active` reports what was granted.

    Linux only — present only when `HAS_PERF_COUNTERS` is True."""

    def __init__(self, counters: list[str] | None = None) -> None: ...
    @property
    def active(self) -> int: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def readings(self) -> dict[str, Any] | None:
        """Latest readings, or None before the first `stop()`."""

    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...

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
    """Configuration for heterogeneous neighborhood sampling.

    ``num_neighbors[(src, rel, dst)]`` is, per hop, how many ``src`` nodes
    with an edge into each expanded ``dst`` node to draw — PyG's meaning.
    Edge types left out draw nothing. ``max_degree`` has no effect: every
    heterogeneous draw is uniform over the whole in-neighborhood."""

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
    accepts int64 seed arrays directly — IDs are range-checked against the
    seed type's node count at this boundary, so callers never pre-narrow
    to uint32.

    Sampling follows PyG: a node of type ``T`` is expanded along every edge
    type whose destination is ``T``. ``edge_index_local(src, rel, dst)``
    keeps the stored direction (row 0 indexes ``nodes(src)``, row 1
    ``nodes(dst)``), with the expanded node as destination, so messages
    flow toward the seeds. Seeds of a type no edge type points into gain
    no neighbors — add reverse relations, as PyG's ``ToUndirected`` does."""

    @property
    def node_types(self) -> list[str]: ...
    @property
    def edge_types(self) -> list[tuple[str, str, str]]: ...
    @property
    def seed_type(self) -> str: ...
    @property
    def seeds(self) -> npt.NDArray[np.int64]: ...
    @property
    def seed_indices(self) -> npt.NDArray[np.int64]: ...
    def nodes(self, node_type: str) -> npt.NDArray[np.int64]: ...
    def nodes_u32(self, node_type: str) -> npt.NDArray[np.uint32]: ...
    def edge_index_local(self, src: str, rel: str, dst: str) -> npt.NDArray[np.int64]: ...

class HeteroNeighborSampler:
    """Heterogeneous neighborhood sampler. ``sample`` raises
    ``SamplingError`` for a seed outside ``seed_type``'s node range."""

    def __init__(self, graph: HeteroCsrGraph, config: HeteroSamplingConfig) -> None: ...
    def sample(
        self,
        seed_type: str,
        seeds: SeedArray | list[int],
    ) -> HeteroSampledSubgraph: ...

class HeteroNeighborLoader:
    """Prefetching heterogeneous neighbor loader. Spawns `sampler_threads`
    worker threads over an MPMC work queue — the same pipeline, ordering,
    and shutdown contract as `NeighborLoader`. Every submitted batch is
    rooted at the `seed_type` fixed at construction."""

    def __init__(
        self,
        graph: HeteroCsrGraph,
        config: HeteroSamplingConfig,
        seed_type: str,
        prefetch_depth: int = 2,
        sampler_threads: int = 1,
    ) -> None: ...
    def submit(self, batch_idx: int, seeds: SeedArray | list[int]) -> None: ...
    def next(self) -> HeteroSampledSubgraph | None: ...
    def next_batch(self) -> tuple[int, HeteroSampledSubgraph] | None:
        """`(batch_idx, subgraph)`, or None after `shutdown()`."""
    @property
    def prefetch_depth(self) -> int: ...
    def shutdown(self) -> None: ...
    def stats(self) -> PrefetchStats: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

# -- FeatureCache (async) ----------------------------------------------------

class FeatureCacheConfig:
    """Configuration for the tiered feature cache."""

    def __init__(
        self,
        gpu_capacity: int = 10_000,
        cpu_capacity: int = 1_000_000,
        feature_dim: int = 128,
        nvme_path: PathLike | None = None,
        cold_store_path: PathLike | None = None,
        cold_level: int = 12,
    ) -> None:
        """`cold_store_path` points at an AETHFEAT file to compress into a
        resident backing tier, making the cache a complete feature source:
        a node in no other tier is served from its zstd block instead of
        raising. Requires a build with the `zstd-tier` feature."""

    @property
    def gpu_capacity(self) -> int: ...
    @property
    def cpu_capacity(self) -> int: ...
    @property
    def feature_dim(self) -> int: ...
    @property
    def nvme_path(self) -> str | None: ...
    @property
    def cold_store_path(self) -> str | None: ...
    @property
    def cold_level(self) -> int: ...

class FeatureCache:
    """Tiered (GPU/CPU/NVMe) feature cache. Async API."""

    @staticmethod
    async def create(config: FeatureCacheConfig) -> FeatureCache: ...
    async def get(self, node: int) -> npt.NDArray[np.float32]: ...
    async def get_batch(self, nodes: list[int]) -> npt.NDArray[np.float32]: ...
    async def insert(self, node: int, features: npt.NDArray[np.float32]) -> None:
        """`features` must hold exactly `feature_dim` values; any other
        length raises before anything is written."""
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
