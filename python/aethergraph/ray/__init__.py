"""Ray Data integration for distributed graph sampling.

This module provides a custom Ray Datasource that uses the existing NeighborLoader
internally for distributed GNN training at massive scale.

The Key Insight:
    Most distributed GNN systems partition the graph across workers, which incurs
    massive network overhead during sampling. AetherGraph takes a different approach:
    replicate the graph topology on each worker's local NVMe, and only distribute
    the seed node workload. This keeps GPUs 100% saturated with zero network shuffle.

Example:
    >>> import ray
    >>> from aethergraph.ray import AetherGraphDatasource, create_sampling_dataset
    >>>
    >>> ray.init()
    >>>
    >>> # Low-level: use datasource directly
    >>> ds = ray.data.read_datasource(
    ...     AetherGraphDatasource(
    ...         graph_path="graph.bin",
    ...         num_neighbors=[25, 10],
    ...         batch_size=1024,
    ...     )
    ... )
    >>>
    >>> # High-level: convenience function
    >>> dataset = create_sampling_dataset(
    ...     graph_path="graph.bin",
    ...     num_neighbors=[25, 10],
    ...     batch_size=1024,
    ... )
    >>>
    >>> for batch in dataset.iter_rows():
    ...     train_step(batch)
"""

from aethergraph.ray.dataset import collate_to_pyg, create_sampling_dataset
from aethergraph.ray.datasource import AetherGraphDatasource

__all__ = [
    "AetherGraphDatasource",
    "create_sampling_dataset",
    "collate_to_pyg",
]
