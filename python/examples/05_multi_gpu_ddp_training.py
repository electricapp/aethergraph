"""Multi-GPU training with PyTorch DDP and local sampling.

Demonstrates the recommended pattern for large-scale GNN training:
- Each GPU samples from a local NVMe graph replica (no network for graph data)
- PyTorch DDP synchronizes gradients via NCCL/NVLINK
- Linear scaling with number of GPUs

Launch with torchrun:
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 05_multi_gpu_ddp_training.py

    # Multi-node (each node needs graph on local NVMe at same path)
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d \
        --rdzv_endpoint=master:29500 05_multi_gpu_ddp_training.py

Requires: pip install aethergraph[pytorch-geometric]
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

try:
    from typing import cast

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
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


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialize the distributed process group.

    Returns:
        Tuple of (rank, world_size, device) for this process.
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def cleanup_distributed() -> None:
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def partition_nodes(
    all_nodes: np.ndarray, rank: int, world_size: int, shuffle: bool = True
) -> np.ndarray:
    """Partition nodes across ranks for distributed training.

    Args:
        all_nodes: Array of all training node IDs.
        rank: Current process rank.
        world_size: Total number of processes.
        shuffle: Whether to shuffle before partitioning.

    Returns:
        Node IDs assigned to this rank.
    """
    if shuffle:
        rng = np.random.default_rng(42)
        all_nodes = rng.permutation(all_nodes)
    return np.array_split(all_nodes, world_size)[rank]


def train_epoch(
    model: torch.nn.Module,
    loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    """Train the model for one epoch.

    Args:
        model: The DDP-wrapped GNN model.
        loader: NeighborLoader yielding mini-batches from local graph.
        optimizer: Optimizer for parameter updates.
        labels: Ground truth labels for all nodes.
        device: Device to run training on.

    Returns:
        Average loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device, non_blocking=True)
        batch_labels = labels[data.n_id].to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        seed_idx = data.input_id
        loss = F.cross_entropy(out[seed_idx], batch_labels[seed_idx])
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def create_synthetic_graph(
    graph_path: Path, num_nodes: int, num_edges: int, feature_dim: int
) -> None:
    """Create and save a synthetic graph.

    Args:
        graph_path: Path to save the graph binary file.
        num_nodes: Number of nodes in the graph.
        num_edges: Number of edges in the graph.
        feature_dim: Dimension of node features.
    """
    rng = np.random.default_rng(42)
    src = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    dst = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    graph = Graph.from_edges(num_nodes, src, dst)
    graph.features = rng.standard_normal((num_nodes, feature_dim)).astype(np.float32)
    graph.save(graph_path)


def main() -> None:
    """Run the multi-GPU DDP training example.

    Each rank loads the graph from local NVMe, samples its partition of nodes,
    and trains with DDP gradient synchronization. This pattern avoids network
    transfer of graph data - only gradients cross the network.
    """
    rank, world_size, device = setup_distributed()
    is_main = rank == 0

    if is_main:
        logger.info("AetherGraph - Multi-GPU DDP Training")
        logger.info("=" * 50)
        logger.info(f"World size: {world_size}, Device: {device}")

    torch.manual_seed(42 + rank)
    num_nodes, num_edges, feature_dim, num_classes = 100_000, 500_000, 64, 10
    graph_path = Path("/tmp/aethergraph_ddp_example.bin")

    if is_main:
        create_synthetic_graph(graph_path, num_nodes, num_edges, feature_dim)
        logger.info(f"Graph: {num_nodes:,} nodes, {num_edges:,} edges")

    if world_size > 1:
        dist.barrier()

    graph = Graph.load(graph_path)
    rng = np.random.default_rng(42)
    labels = torch.from_numpy(rng.integers(0, num_classes, num_nodes)).long()

    all_train_nodes = np.arange(int(num_nodes * 0.8), dtype=np.int64)
    my_train_nodes = partition_nodes(all_train_nodes, rank, world_size)

    if is_main:
        logger.info(f"Train nodes per rank: ~{len(my_train_nodes):,}")

    loader = NeighborLoader(
        graph,
        num_neighbors=[15, 10],
        input_nodes=my_train_nodes,
        batch_size=512,
        shuffle=True,
        pin_memory=True,
    )

    model: GraphSAGE | DDP = GraphSAGE(feature_dim, 128, num_classes).to(device)
    if world_size > 1:
        model = DDP(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if is_main:
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {params:,} parameters")

    for epoch in range(3):
        start = time.perf_counter()
        loss = train_epoch(model, loader, optimizer, labels, device)
        elapsed = time.perf_counter() - start

        if is_main:
            logger.info(f"Epoch {epoch + 1}: loss={loss:.4f}, time={elapsed:.1f}s")

    cleanup_distributed()

    if is_main:
        graph_path.unlink(missing_ok=True)
        logger.info("Training complete")


if __name__ == "__main__":
    main()
