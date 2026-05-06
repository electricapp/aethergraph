#!/usr/bin/env python3
"""Simple GNN training with AetherGraph.

Demonstrates the minimal API: Graph + NeighborLoader for iterating over
mini-batches of sampled subgraphs.
"""

import logging

import numpy as np

from aethergraph import Graph
from aethergraph.pytorch import NeighborLoader

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the simple training example.

    Creates a synthetic graph with random features, iterates over mini-batches
    using NeighborLoader, and demonstrates multi-epoch iteration.
    """
    logger.info("AetherGraph - Simple GNN Training")
    logger.info("=" * 50)

    num_nodes, num_edges, feature_dim = 10_000, 50_000, 64
    rng = np.random.default_rng(42)

    src = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    dst = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    graph = Graph.from_edges(num_nodes, src, dst)
    graph.features = rng.standard_normal((num_nodes, feature_dim)).astype(np.float32)
    logger.info(f"Graph: {num_nodes:,} nodes, {num_edges:,} edges")

    loader = NeighborLoader(graph, num_neighbors=[10, 5], batch_size=128, shuffle=True)
    logger.info(f"Loader: {len(loader)} batches per epoch")

    total_nodes, total_edges = 0, 0
    for i, data in enumerate(loader):
        total_nodes += data.num_nodes
        total_edges += data.edge_index.shape[1]
        if i < 3:
            logger.info(f"Batch {i}: {data.num_nodes} nodes, x={data.x.shape}")

    logger.info(f"Totals: {total_nodes:,} nodes, {total_edges:,} edges")

    for epoch in range(3):
        batch_count = sum(1 for _ in loader)
        logger.info(f"Epoch {epoch + 1}: {batch_count} batches")


if __name__ == "__main__":
    main()
