"""Distributed graph sampling with Ray Data.

Demonstrates creating a distributed sampling dataset, processing batches
in parallel across Ray workers, and converting to PyG format.

Requires: pip install aethergraph[ray]
"""

import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

try:
    import ray

    from aethergraph.ray import collate_to_pyg, create_sampling_dataset
except ImportError:
    logger.error("Requires ray[data]. Install with: pip install aethergraph[ray]")
    sys.exit(1)


def main() -> None:
    """Run the distributed sampling example.

    Initializes a local Ray cluster, creates a sampling dataset that distributes
    work across workers, and processes batches showing throughput metrics.
    """
    logger.info("AetherGraph - Ray Distributed Sampling")
    logger.info("=" * 50)

    ray.init(num_cpus=4, ignore_reinit_error=True, runtime_env={"working_dir": None})
    logger.info(f"Ray initialized: {ray.available_resources().get('CPU', 0):.0f} CPUs")

    graph_path = "../test_data/simple_graph.bin"
    seed_nodes = list(range(100))

    dataset = create_sampling_dataset(
        graph_path=graph_path,
        num_neighbors=[3, 2],
        input_nodes=seed_nodes,
        batch_size=10,
        parallelism=4,
    )
    logger.info(f"Dataset created with {len(seed_nodes)} seed nodes")

    start = time.perf_counter()
    batch_count, total_nodes, total_edges = 0, 0, 0

    for batch in dataset.iter_batches(batch_size=1):
        data = collate_to_pyg(batch)
        batch_count += 1
        total_nodes += data.n_id.shape[0]
        total_edges += data.edge_index.shape[1]

    elapsed = time.perf_counter() - start
    logger.info(f"Processed {batch_count} batches in {elapsed:.2f}s")
    logger.info(f"Total: {total_nodes:,} nodes, {total_edges:,} edges")
    logger.info(f"Throughput: {batch_count / elapsed:.1f} batches/sec")

    ray.shutdown()


if __name__ == "__main__":
    main()
