#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "aethergraph[pytorch-geometric]",
#     "wandb",
#     "typer",
#     "rich",
#     "pydantic",
# ]
# ///
"""GNN training with Weights & Biases experiment tracking.

Trains a GraphSAGE model using AetherGraph's NeighborLoader and logs
metrics, hyperparameters, sampling throughput, and model checkpoints
to W&B for experiment tracking and comparison.

Usage::

    wandb login
    uv run examples/06_wandb_experiment_tracking.py
    uv run examples/06_wandb_experiment_tracking.py --epochs 20 --lr 0.005
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import torch
import torch.nn.functional as F
import typer
import wandb
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from torch_geometric.nn import SAGEConv  # type: ignore[import-untyped]

from aethergraph import Graph
from aethergraph.pytorch import NeighborLoader

app = typer.Typer(help="AetherGraph GNN training with W&B experiment tracking.")
console = Console()


# ---------------------------------------------------------------------------
# Pydantic models for structured config / metrics
# ---------------------------------------------------------------------------


class TrainConfig(BaseModel):
    """Training hyperparameters — logged to W&B as run config."""

    num_nodes: int = 10_000
    feature_dim: int = 64
    hidden_dim: int = 128
    num_classes: int = 7
    epochs: int = 10
    batch_size: int = 128
    lr: float = 0.01
    num_neighbors: list[int] = [15, 10]
    device: str = "cpu"
    sampler: str = "aethergraph"


class EpochMetrics(BaseModel):
    """Per-epoch metrics — logged to W&B after each epoch."""

    epoch: int
    train_loss: float
    val_accuracy: float
    epoch_time_s: float
    num_batches: int
    sampling_throughput_batches_per_sec: float
    sampling_avg_batch_ms: float
    sampling_prefetch_hit_rate: float


class CheckpointPayload(BaseModel):
    """Model checkpoint metadata — saved alongside state dict."""

    config: TrainConfig
    final_loss: float
    final_val_accuracy: float
    total_epochs: int


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return cast(torch.Tensor, self.conv2(x, edge_index))


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_synthetic_graph(
    num_nodes: int, avg_degree: int, feature_dim: int, num_classes: int,
) -> tuple[Graph, torch.Tensor]:
    """Build a synthetic power-law graph with random features and labels."""
    rng = np.random.default_rng(42)
    edges = []
    for src in range(num_nodes):
        degree = int(rng.exponential(avg_degree)) + 1
        for dst in rng.integers(0, num_nodes, size=degree):
            if src != dst:
                edges.append((src, int(dst)))

    src_arr = np.array([e[0] for e in edges], dtype=np.uint32)
    dst_arr = np.array([e[1] for e in edges], dtype=np.uint32)

    from aethergraph._core import CsrGraph

    rust_graph = CsrGraph.from_edges(num_nodes, src_arr, dst_arr)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    rust_graph.save(path)
    graph = Graph.load(path)
    Path(path).unlink()

    graph.features = rng.standard_normal((num_nodes, feature_dim)).astype(np.float32)
    labels = torch.from_numpy(rng.integers(0, num_classes, num_nodes)).long()

    return graph, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@app.command()
def train(
    num_nodes: Annotated[int, typer.Option(help="Number of graph nodes")] = 10_000,
    feature_dim: Annotated[int, typer.Option(help="Node feature dimension")] = 64,
    hidden_dim: Annotated[int, typer.Option(help="Hidden layer dimension")] = 128,
    num_classes: Annotated[int, typer.Option(help="Number of output classes")] = 7,
    epochs: Annotated[int, typer.Option(help="Training epochs")] = 10,
    batch_size: Annotated[int, typer.Option(help="Batch size (seed nodes)")] = 128,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 0.01,
    num_neighbors: Annotated[list[int], typer.Option(help="Neighbors per hop")] = [15, 10],  # noqa: B006
) -> None:
    """Train a GraphSAGE model with AetherGraph sampling and W&B logging."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = TrainConfig(
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_neighbors=num_neighbors,
        device=str(device),
    )

    run = wandb.init(project="aethergraph-gnn", config=config.model_dump())

    console.print("[bold]AetherGraph + W&B Training[/bold]")
    console.print(f"Device: {device} | W&B run: {run.name}")

    # Build graph
    graph, labels = build_synthetic_graph(
        config.num_nodes, avg_degree=20,
        feature_dim=config.feature_dim, num_classes=config.num_classes,
    )
    wandb.log({"graph/num_nodes": graph.num_nodes, "graph/num_edges": graph.num_edges})
    console.print(f"Graph: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")

    # Train/val split
    rng = np.random.default_rng(42)
    perm = rng.permutation(graph.num_nodes)
    split = int(0.8 * graph.num_nodes)
    train_idx = np.sort(perm[:split]).astype(np.int64)
    val_idx = np.sort(perm[split:]).astype(np.int64)

    train_loader = NeighborLoader(
        graph, num_neighbors=config.num_neighbors, input_nodes=train_idx,
        batch_size=config.batch_size, shuffle=True,
    )
    val_loader = NeighborLoader(
        graph, num_neighbors=config.num_neighbors, input_nodes=val_idx,
        batch_size=config.batch_size, shuffle=False,
    )

    model = GraphSAGE(config.feature_dim, config.hidden_dim, config.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Results table
    table = Table(title="Training Progress")
    table.add_column("Epoch", justify="right", style="cyan")
    table.add_column("Loss", justify="right")
    table.add_column("Val Acc", justify="right", style="green")
    table.add_column("Time", justify="right")
    table.add_column("Sampling", justify="right", style="dim")
    table.add_column("Prefetch", justify="right", style="dim")

    last_metrics: EpochMetrics | None = None

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.perf_counter()
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            batch_labels = labels[batch.n_id].to(device)

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            seed_idx = batch.input_id
            loss = F.cross_entropy(out[seed_idx], batch_labels[seed_idx])
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_time = time.perf_counter() - epoch_start
        avg_loss = epoch_loss / num_batches

        loader_metrics = train_loader.metrics
        sampling_throughput = (
            num_batches / (loader_metrics.total_time_ms / 1000)
            if loader_metrics.total_time_ms > 0
            else 0.0
        )

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch_labels = labels[batch.n_id].to(device)
                logits = model(batch.x, batch.edge_index)
                seed_idx = batch.input_id
                pred = logits[seed_idx].argmax(dim=1)
                correct += (pred == batch_labels[seed_idx]).sum().item()
                total += int(seed_idx.numel())
        val_acc = correct / total if total > 0 else 0.0

        m = EpochMetrics(
            epoch=epoch + 1,
            train_loss=avg_loss,
            val_accuracy=val_acc,
            epoch_time_s=epoch_time,
            num_batches=num_batches,
            sampling_throughput_batches_per_sec=sampling_throughput,
            sampling_avg_batch_ms=loader_metrics.avg_batch_time_ms,
            sampling_prefetch_hit_rate=loader_metrics.prefetch_hit_rate,
        )
        last_metrics = m

        wandb.log({
            "epoch": m.epoch,
            "train/loss": m.train_loss,
            "train/epoch_time_s": m.epoch_time_s,
            "train/batches": m.num_batches,
            "val/accuracy": m.val_accuracy,
            "sampling/throughput_batches_per_sec": m.sampling_throughput_batches_per_sec,
            "sampling/avg_batch_ms": m.sampling_avg_batch_ms,
            "sampling/prefetch_hit_rate": m.sampling_prefetch_hit_rate,
        })

        table.add_row(
            str(m.epoch),
            f"{m.train_loss:.4f}",
            f"{m.val_accuracy:.2%}",
            f"{m.epoch_time_s:.2f}s",
            f"{m.sampling_avg_batch_ms:.1f}ms/b",
            f"{m.sampling_prefetch_hit_rate:.0%}",
        )

    console.print(table)

    # Save model checkpoint with structured metadata
    checkpoint_path = Path(tempfile.mkdtemp()) / "model.pt"
    meta = CheckpointPayload(
        config=config,
        final_loss=last_metrics.train_loss if last_metrics else 0.0,
        final_val_accuracy=last_metrics.val_accuracy if last_metrics else 0.0,
        total_epochs=config.epochs,
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": meta.model_dump(),
    }, checkpoint_path)

    artifact = wandb.Artifact("graphsage-model", type="model")
    artifact.add_file(str(checkpoint_path))
    run.log_artifact(artifact)

    run.finish()
    console.print(f"\n[bold green]Done.[/bold green] W&B run: {run.url}")


if __name__ == "__main__":
    app()
