"""Type stubs for aethergraph._core Rust extension module."""

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

__version__: str

SeedArray = npt.NDArray[np.int64] | npt.NDArray[np.uint32]

# Exceptions
class GraphLoadError(Exception):
    """Raised when graph loading fails."""

    ...

class SamplingError(Exception):
    """Raised when sampling fails."""

    ...

class CacheError(Exception):
    """Raised when cache operations fail."""

    ...

class ArrowConversionError(Exception):
    """Raised when Arrow conversion fails."""

    ...

# Graph
class CsrGraph:
    """Compressed Sparse Row graph representation."""

    @staticmethod
    def load(path: str, validation: str = "auto") -> CsrGraph:
        """Load mmap-backed graph from binary file."""
        ...

    @staticmethod
    def load_mmap(path: str, validation: str = "offsets_only") -> CsrGraph:
        """Explicit mmap-backed graph load."""
        ...

    @staticmethod
    def load_owned(path: str, validation: str = "full") -> CsrGraph:
        """Explicit owned in-memory graph load."""
        ...

    @staticmethod
    def from_edges(
        num_nodes: int,
        src: npt.NDArray[np.uint32],
        dst: npt.NDArray[np.uint32],
    ) -> CsrGraph:
        """Create graph from edge arrays."""
        ...

    def save(self, path: str) -> None:
        """Save graph to binary file."""
        ...

    def num_nodes(self) -> int:
        """Return number of nodes."""
        ...

    def num_edges(self) -> int:
        """Return number of edges."""
        ...

    def has_weights(self) -> bool:
        """Return whether graph has edge weights."""
        ...

    def degree(self, node: int) -> int:
        """Return degree of a node."""
        ...

    def neighbors(self, node: int) -> npt.NDArray[np.uint32]:
        """Return neighbors of a node."""
        ...

    def batch_neighbors(self, nodes: list[int]) -> list[npt.NDArray[np.uint32]]:
        """Return neighbors for multiple nodes."""
        ...

    def weights(self, node: int) -> npt.NDArray[np.float32] | None:
        """Return edge weights for a node."""
        ...

    def stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        ...

def save_features(path: str, features: npt.NDArray[np.float32]) -> None:
    """Save node features to AETHFEAT format."""
    ...

# Sampling
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
        subgraph_type: str = "directional",
        track_edge_ids: bool = True,
    ) -> None:
        """Initialize sampling configuration."""
        ...

    @property
    def num_neighbors(self) -> list[int]:
        """Return num_neighbors configuration."""
        ...

    @property
    def replace(self) -> bool:
        """Return replace flag."""
        ...

    @property
    def seed(self) -> int | None:
        """Return random seed."""
        ...

    @property
    def max_degree(self) -> int | None:
        """Return max degree cap."""
        ...

    @property
    def cumulative(self) -> bool:
        """Return cumulative flag."""
        ...

class SampledSubgraph:
    """A sampled subgraph from neighborhood sampling."""

    def num_nodes(self) -> int:
        """Return number of nodes in subgraph."""
        ...

    def num_edges(self) -> int:
        """Return number of edges in subgraph."""
        ...

    def num_seeds(self) -> int:
        """Return number of seed nodes."""
        ...

    def nodes(self) -> npt.NDArray[np.uint32]:
        """Return all node IDs in subgraph."""
        ...

    def seeds(self) -> npt.NDArray[np.uint32]:
        """Return seed node IDs."""
        ...

    def edge_index(self) -> npt.NDArray[np.int64]:
        """Return edge index in COO format."""
        ...

    def edge_index_local(self) -> npt.NDArray[np.int64]:
        """Return edge index with local node IDs."""
        ...

    def edge_ids(self) -> npt.NDArray[np.int64]:
        """Return global edge IDs."""
        ...

    def seed_indices(self) -> npt.NDArray[np.int64]:
        """Return local indices of seed nodes in the sorted node array."""
        ...

    def num_sampled_nodes_per_hop(self) -> list[int]:
        """Return number of sampled nodes per hop."""
        ...

    def num_sampled_edges_per_hop(self) -> list[int]:
        """Return number of sampled edges per hop."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        ...

    def to_arrow(self) -> Any:
        """Convert to Arrow RecordBatch."""
        ...

class NeighborSampler:
    """Neighborhood sampler for GNN training."""

    def __init__(self, graph: CsrGraph, config: SamplingConfig) -> None:
        """Initialize sampler."""
        ...

    def sample(self, seeds: Sequence[int] | SeedArray) -> SampledSubgraph:
        """Sample neighborhoods for seed nodes."""
        ...

class ParallelBatchSampler:
    """Parallel batch sampler for high-throughput training."""

    def __init__(self, graph: CsrGraph, config: SamplingConfig) -> None:
        """Initialize parallel sampler."""
        ...

    def sample_batches(self, batches: list[list[int]]) -> list[SampledSubgraph]:
        """Sample neighborhoods for multiple batches in parallel."""
        ...

# Prefetch stats
class PrefetchStats:
    """Statistics from prefetch sampler."""

    @property
    def hits(self) -> int:
        """Number of batches immediately available."""
        ...

    @property
    def misses(self) -> int:
        """Number of times consumer had to wait."""
        ...

    @property
    def total(self) -> int:
        """Total batches processed."""
        ...

    @property
    def hit_rate(self) -> float:
        """Hit rate (0.0 to 1.0)."""
        ...

# Prefetch loader
class NeighborLoader:
    """Prefetching neighbor loader."""

    def __init__(
        self,
        graph: CsrGraph,
        config: SamplingConfig,
        prefetch_depth: int = 2,
    ) -> None:
        """Initialize loader."""
        ...

    @staticmethod
    def with_features(
        graph: CsrGraph,
        config: SamplingConfig,
        features_path: str,
        prefetch_depth: int = 2,
    ) -> NeighborLoader:
        """Create loader with feature loading."""
        ...

    @staticmethod
    def new_nvme(
        graph_path: str,
        config: SamplingConfig,
        prefetch_depth: int = 2,
    ) -> NeighborLoader:
        """Create loader for NVMe-backed graphs (Linux only)."""
        ...

    def submit(
        self, batch_id: int, seeds: npt.NDArray[np.int64] | npt.NDArray[np.uint32] | list[int]
    ) -> None:
        """Submit a batch for sampling."""
        ...

    def submit_epoch(self, batches: list[Any]) -> None:
        """Submit all batches for an epoch."""
        ...

    def next(self) -> SampledSubgraph | None:
        """Get next sampled subgraph, blocking."""
        ...

    def try_next(self) -> SampledSubgraph | None:
        """Get next sampled subgraph without blocking."""
        ...

    def next_with_features(
        self,
    ) -> tuple[SampledSubgraph, npt.NDArray[np.float32] | None] | None:
        """Get next sampled batch with features."""
        ...

    @property
    def feature_dim(self) -> int | None:
        """Feature dimension if this loader is feature-enabled."""
        ...

    @property
    def has_features(self) -> bool:
        """Whether this loader is feature-enabled."""
        ...

    @property
    def prefetch_depth(self) -> int:
        """Configured prefetch depth."""
        ...

    def shutdown(self) -> None:
        """Shutdown the loader."""
        ...

    def stats(self) -> PrefetchStats:
        """Get prefetch statistics."""
        ...
