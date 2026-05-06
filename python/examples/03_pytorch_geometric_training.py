"""PyTorch Geometric GNN training with AetherGraph.

Demonstrates using NeighborLoader as a drop-in replacement for PyG's loader
to train a GraphSAGE model with standard PyTorch workflow.

Requires: pip install aethergraph[pytorch-geometric]
"""

import logging
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

try:
    from typing import cast

    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv  # type: ignore[import-untyped]

    from aethergraph import Graph
    from aethergraph.pytorch import NeighborLoader
except ImportError:
    logger.error("Requires PyTorch and PyG. Install: pip install aethergraph[pytorch-geometric]")
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


def train_epoch(
    model: torch.nn.Module,
    loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    """Train the model for one epoch.

    Args:
        model: The GNN model to train.
        loader: NeighborLoader yielding mini-batches.
        optimizer: Optimizer for parameter updates.
        labels: Ground truth labels for all nodes.
        device: Device to run training on.

    Returns:
        Average loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        batch_labels = labels[data.n_id].to(device)

        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        seed_idx = data.input_id
        loss = F.cross_entropy(out[seed_idx], batch_labels[seed_idx])
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: NeighborLoader,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    """Evaluate the model on a dataset.

    Args:
        model: The trained GNN model.
        loader: NeighborLoader for evaluation data.
        labels: Ground truth labels for all nodes.
        device: Device to run evaluation on.

    Returns:
        Accuracy as a fraction between 0 and 1.
    """
    model.eval()
    correct, total = 0, 0

    for data in loader:
        data = data.to(device)
        batch_labels = labels[data.n_id].to(device)
        logits = model(data.x, data.edge_index)
        seed_idx = data.input_id
        pred = logits[seed_idx].argmax(dim=1)
        correct += (pred == batch_labels[seed_idx]).sum().item()
        total += int(seed_idx.numel())

    return correct / total if total > 0 else 0.0


def main() -> None:
    """Run the PyG training example.

    Creates a synthetic graph with random features and labels, trains a
    GraphSAGE model using AetherGraph's NeighborLoader, and reports metrics.
    """
    logger.info("AetherGraph - PyTorch Geometric Training")
    logger.info("=" * 50)

    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    graph = Graph.load("../test_data/simple_graph.bin")
    num_nodes, feature_dim, num_classes = graph.num_nodes, 16, 3

    graph.features = rng.standard_normal((num_nodes, feature_dim)).astype(np.float32)
    labels = torch.from_numpy(rng.integers(0, num_classes, num_nodes)).long()
    logger.info(f"Graph: {num_nodes} nodes, features [{num_nodes}, {feature_dim}]")

    train_idx = np.arange(num_nodes // 2, dtype=np.int64)
    val_idx = np.arange(num_nodes // 2, 3 * num_nodes // 4, dtype=np.int64)
    logger.info(f"Split: {len(train_idx)} train, {len(val_idx)} val")

    train_loader = NeighborLoader(
        graph, num_neighbors=[3, 2], input_nodes=train_idx, batch_size=4, shuffle=True
    )
    val_loader = NeighborLoader(
        graph, num_neighbors=[3, 2], input_nodes=val_idx, batch_size=4, shuffle=False
    )

    model = GraphSAGE(feature_dim, 32, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(3):
        start = time.perf_counter()
        loss = train_epoch(model, train_loader, optimizer, labels, device)
        elapsed = time.perf_counter() - start
        logger.info(f"Epoch {epoch + 1}: loss={loss:.4f}, time={elapsed:.2f}s")

    acc = evaluate(model, val_loader, labels, device)
    logger.info(f"Validation accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()
