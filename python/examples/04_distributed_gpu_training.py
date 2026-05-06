"""Distributed GPU training with Ray Data and PyTorch.

Demonstrates combining Ray Data for distributed graph sampling across CPU workers
with PyTorch GPU training for GraphSAGE. This is the recommended pattern for
large-scale GNN training:

1. Ray workers sample subgraphs in parallel (CPU-bound, scales horizontally)
2. GPU worker receives batches and runs forward/backward passes
3. Graph is memory-mapped, so workers share the same data without copying

Requires: pip install aethergraph[ray,pytorch-geometric]
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

try:
    from typing import cast

    import ray
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data  # type: ignore[import-untyped]
    from torch_geometric.nn import SAGEConv  # type: ignore[import-untyped]

    from aethergraph import Graph
    from aethergraph.ray import collate_to_pyg, create_sampling_dataset
except ImportError:
    logger.error("Requires Ray and PyG. Install: pip install aethergraph[ray,pytorch-geometric]")
    sys.exit(1)


class GraphSAGE(torch.nn.Module):
    """Two-layer GraphSAGE model for node classification."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        """Initialize the GraphSAGE model.

        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden layer dimension.
            out_channels: Number of output classes.
        """
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Node features of shape [num_nodes, in_channels].
            edge_index: Edge connectivity of shape [2, num_edges].

        Returns:
            Output logits of shape [num_nodes, out_channels].
        """
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return cast(torch.Tensor, self.conv2(x, edge_index))


def train_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: Data,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    """Train on a single batch.

    Args:
        model: The GNN model to train.
        optimizer: Optimizer for parameter updates.
        data: PyG Data object containing the sampled subgraph.
        labels: Ground truth labels tensor (indexed by global node IDs).
        device: Device to run training on.

    Returns:
        Loss value for this batch.
    """
    model.train()
    data = data.to(device)
    batch_labels = labels[data.n_id].to(device)

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    seed_idx = data.input_id
    loss = F.cross_entropy(out[seed_idx], batch_labels[seed_idx])
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()

    return loss.item()


def create_synthetic_graph(
    num_nodes: int, num_edges: int, feature_dim: int, graph_path: Path
) -> tuple[torch.Tensor, int]:
    """Create and save a synthetic graph for training.

    Args:
        num_nodes: Number of nodes in the graph.
        num_edges: Number of edges in the graph.
        feature_dim: Dimension of node features.
        graph_path: Path to save the graph binary file.

    Returns:
        Tuple of (labels tensor, number of classes).
    """
    rng = np.random.default_rng(42)
    num_classes = 10

    src = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    dst = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    graph = Graph.from_edges(num_nodes, src, dst)

    graph.features = rng.standard_normal((num_nodes, feature_dim)).astype(np.float32)
    labels = torch.from_numpy(rng.integers(0, num_classes, num_nodes)).long()

    graph.save(graph_path)
    logger.info(f"Created graph: {num_nodes:,} nodes, {num_edges:,} edges")

    return labels, num_classes


def main() -> None:
    """Run the distributed GPU training example.

    Sets up Ray for distributed sampling, creates a synthetic graph, and trains
    a GraphSAGE model with GPU acceleration. Ray workers handle CPU-bound
    sampling while the main process runs GPU training.
    """
    logger.info("AetherGraph - Distributed GPU Training")
    logger.info("=" * 50)

    num_cpus = 4
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True, runtime_env={"working_dir": None})
    logger.info(f"Ray initialized: {num_cpus} CPU workers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    torch.manual_seed(42)
    num_nodes, num_edges, feature_dim = 100_000, 500_000, 64
    graph_path = Path("/tmp/aethergraph_distributed_example.bin")
    labels, num_classes = create_synthetic_graph(num_nodes, num_edges, feature_dim, graph_path)

    train_nodes = np.arange(int(num_nodes * 0.8), dtype=np.int64)
    val_nodes = np.arange(int(num_nodes * 0.8), num_nodes, dtype=np.int64)
    logger.info(f"Split: {len(train_nodes):,} train, {len(val_nodes):,} val")

    model = GraphSAGE(feature_dim, 128, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    num_epochs = 3
    batch_size = 512
    num_neighbors = [15, 10]

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()

        dataset = create_sampling_dataset(
            graph_path=graph_path,
            num_neighbors=num_neighbors,
            input_nodes=train_nodes,
            batch_size=batch_size,
            parallelism=num_cpus,
        )

        total_loss, num_batches = 0.0, 0
        for batch in dataset.iter_batches(batch_size=1):
            data = collate_to_pyg(batch)
            total_loss += train_batch(model, optimizer, data, labels, device)
            num_batches += 1

        elapsed = time.perf_counter() - epoch_start
        avg_loss = total_loss / num_batches
        throughput = num_batches / elapsed

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"loss={avg_loss:.4f}, "
            f"time={elapsed:.1f}s, "
            f"{throughput:.0f} batches/sec"
        )

    model.eval()
    val_dataset = create_sampling_dataset(
        graph_path=graph_path,
        num_neighbors=num_neighbors,
        input_nodes=val_nodes,
        batch_size=batch_size,
        parallelism=num_cpus,
    )

    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_dataset.iter_batches(batch_size=1):
            data = collate_to_pyg(batch).to(device)
            batch_labels = labels[data.n_id].to(device)
            logits = model(data.x, data.edge_index)
            seed_idx = data.input_id
            pred = logits[seed_idx].argmax(dim=1)
            correct += (pred == batch_labels[seed_idx]).sum().item()
            total += int(seed_idx.numel())

    logger.info(f"Validation accuracy: {correct / total:.2%}")

    ray.shutdown()
    graph_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
