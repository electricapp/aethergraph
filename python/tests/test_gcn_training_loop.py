"""Smoke-train a 2-layer GCN through NeighborLoader.

Validates that batches reach the model with correct shapes, backward +
optimizer step run without errors, and the average loss measurably
decreases from the first epoch to the last (coarse signal that the
loader / model / optimizer wiring is intact).

Intentionally tiny so it stays fast. Skipped if torch, torch_geometric,
or CUDA are missing.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_torch, pytest.mark.requires_pyg]


def _skip_unless_cuda() -> None:
    """Skip the whole test if torch has no CUDA backend."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")


def _synthetic_classification_problem(
    rng: np.random.Generator, num_nodes: int, feature_dim: int, num_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a linearly-separable problem so loss visibly drops in 5 epochs.

    Returns `(src, dst, features, labels)`. Each class is a Gaussian cluster
    in feature space; edges connect nodes within the same class with high
    probability, giving the GCN useful signal to aggregate.
    """
    labels = rng.integers(0, num_classes, size=num_nodes, dtype=np.int64)
    centers = rng.standard_normal((num_classes, feature_dim)).astype(np.float32) * 5.0
    features = centers[labels] + rng.standard_normal((num_nodes, feature_dim)).astype(np.float32)

    edges_per_node = 6
    src_list = []
    dst_list = []
    for n in range(num_nodes):
        # Connect to other nodes of the same class with high prob.
        same_class = np.where(labels == labels[n])[0]
        choices = rng.choice(same_class, size=edges_per_node, replace=True)
        src_list.append(np.full(edges_per_node, n, dtype=np.uint32))
        dst_list.append(choices.astype(np.uint32))
    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    return src, dst, features, labels


def test_2_layer_gcn_loss_decreases_over_5_epochs() -> None:
    _skip_unless_cuda()

    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    from aethergraph import Graph
    from aethergraph.pytorch import NeighborLoader

    rng = np.random.default_rng(42)
    num_nodes = 256
    feature_dim = 16
    num_classes = 4

    src, dst, features, labels = _synthetic_classification_problem(
        rng, num_nodes, feature_dim, num_classes
    )
    graph = Graph.from_edges(num_nodes, src, dst)
    labels_t = torch.from_numpy(labels).long().cuda()

    class GCN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = GCNConv(feature_dim, 32)
            self.conv2 = GCNConv(32, num_classes)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            h = F.relu(self.conv1(x, edge_index))
            out: torch.Tensor = self.conv2(h, edge_index)
            return out

    # Pin torch's RNG so loader / model init don't make the convergence
    # threshold flaky across runs.
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    model = GCN().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=5e-2)

    # Build the loader once. Reusing it across epochs means each epoch sees
    # the same mini-batch decomposition and noise is bounded to gradient
    # variance rather than re-sampling variance.
    loader = NeighborLoader(
        graph,
        num_neighbors=[10, 5],
        batch_size=32,
        features=features,
        shuffle=False,
    )

    epoch_losses: list[float] = []
    for _ in range(5):
        running = 0.0
        steps = 0
        for batch in loader:
            x = batch.x.cuda()
            edge_index = batch.edge_index.cuda()
            n_id = batch.n_id.cuda()
            y = labels_t[n_id[: batch.batch_size]]

            logits = model(x, edge_index)
            loss = F.cross_entropy(logits[: batch.batch_size], y)

            optim.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optim.step()

            running += loss.item()
            steps += 1

        assert steps > 0, "loader produced zero batches"
        epoch_losses.append(running / steps)

    # Coarse signal: last epoch's average loss is meaningfully below the first.
    # 5 epochs on a separable problem with a 4-class GCN should easily clear
    # 30% reduction; the threshold is loose to avoid flakes.
    assert epoch_losses[-1] < epoch_losses[0] * 0.7, (
        f"loss did not decrease enough: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f}"
    )
    for epoch_loss in epoch_losses:
        assert np.isfinite(epoch_loss), f"non-finite loss observed: {epoch_loss}"
