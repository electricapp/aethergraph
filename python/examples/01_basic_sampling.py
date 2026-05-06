"""Basic graph sampling with AetherGraph.

Demonstrates loading a graph, configuring a sampler, and sampling neighborhoods.
"""

import logging

from aethergraph import Graph, Sampler, SamplingConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the basic sampling example.

    Loads a graph from disk, creates a sampler with 2-hop configuration,
    samples neighborhoods for seed nodes, and displays the results.
    """
    logger.info("AetherGraph - Basic Sampling")
    logger.info("=" * 50)

    graph = Graph.load("../test_data/simple_graph.bin")
    logger.info(f"Loaded graph: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")

    stats = graph.stats()
    logger.info(f"Avg degree: {stats['avg_degree']:.2f}, max: {stats['max_degree']}")

    config = SamplingConfig(num_neighbors=[3, 2], replace=False, seed=42)
    sampler = Sampler(graph, config)
    logger.info(f"Sampler: {config.num_neighbors} neighbors per hop")

    seed_nodes = [0, 1, 2]
    subgraph = sampler.sample(seed_nodes)
    logger.info(f"Sampled {subgraph.num_nodes} nodes, {subgraph.num_edges} edges")
    logger.info(f"Edge index shape: {subgraph.edge_index.shape}")

    for node in seed_nodes:
        neighbors = graph.neighbors(node)
        logger.info(f"Node {node}: {len(neighbors)} neighbors")


if __name__ == "__main__":
    main()
