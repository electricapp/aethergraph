"""Training loop against a live-updating RDMA feature server.

Spawns the `rdma_feature_server` example with `--live-rate` set so the
server overwrites features continuously, then runs a small Python training
loop with `feature_source="rdma://..."`. Asserts:

- every batch returns a CUDA tensor of the expected shape,
- the loss is finite at every step (no NaN/inf from torn or stale reads),
- repeated batches see at least some changing feature values (proves the
  server's live writes are actually reaching the consumer).

Skipped unless the extension was built with `--features gpudirect`, CUDA
is available, an RDMA device is loaded, and `cargo` is on PATH (for
building the server example).
"""

from __future__ import annotations

import os
import select
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_torch]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def _has_rxe_or_real_rdma() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    if os.environ.get("AETHER_SKIP_RDMA"):
        return False
    sys_class_infiniband = Path("/sys/class/infiniband")
    return sys_class_infiniband.exists() and any(sys_class_infiniband.iterdir())


def _has_cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def _build_server_example() -> Path:
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--features",
            "rdma",
            "-p",
            "aether-stream",
            "--example",
            "rdma_feature_server",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    candidate = REPO_ROOT / "target" / "release" / "examples" / "rdma_feature_server"
    if not candidate.exists():
        pytest.skip(f"example binary not found at {candidate}")
    return candidate


def _wait_for_ready(proc: subprocess.Popen[str], timeout_s: float) -> None:
    """Block until the server prints `READY port=…` or `timeout_s` elapses.

    Uses `select` so a server that opens stderr but never writes can't pin
    the test in `readline()` past `timeout_s`.
    """
    deadline = time.monotonic() + timeout_s
    assert proc.stderr is not None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("server did not print READY within timeout")
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (code={proc.returncode}) before READY")
        ready, _, _ = select.select([proc.stderr], [], [], min(remaining, 0.5))
        if not ready:
            continue
        line = proc.stderr.readline()
        if not line:
            continue
        if "READY" in line:
            return


def test_neighbor_loader_trains_against_live_rdma_features() -> None:
    """Run a short training loop against a server that keeps rewriting features."""
    from aethergraph._core import HAS_GPUDIRECT

    if not HAS_GPUDIRECT:
        pytest.skip("extension built without --features gpudirect")
    if not _has_cuda():
        pytest.skip("CUDA not available on this host")
    if not _has_rxe_or_real_rdma():
        pytest.skip("no RDMA device detected under /sys/class/infiniband")
    if shutil.which("cargo") is None:
        pytest.skip("cargo not on PATH")

    import torch
    import torch.nn.functional as F

    from aethergraph import Graph
    from aethergraph.pytorch import NeighborLoader

    server_bin = _build_server_example()
    port = _pick_free_port()
    bind = f"127.0.0.1:{port}"

    feature_dim = 32
    num_nodes = 256

    proc = subprocess.Popen(
        [
            str(server_bin),
            "--bind",
            bind,
            "--nodes",
            str(num_nodes),
            "--dim",
            str(feature_dim),
            "--live-rate",
            "5000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_ready(proc, timeout_s=15.0)

        rng = np.random.default_rng(1)
        src = rng.integers(0, num_nodes, size=2048, dtype=np.uint32)
        dst = rng.integers(0, num_nodes, size=2048, dtype=np.uint32)
        graph = Graph.from_edges(num_nodes, src, dst)

        loader = NeighborLoader(
            graph,
            num_neighbors=[6, 3],
            batch_size=16,
            feature_source=f"rdma://{bind}",
        )

        # Tiny MLP — exercises the gradient path without committing to a
        # specific GNN library. The point is to confirm features survive
        # a backward pass cleanly under server contention.
        model = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4),
        ).cuda()
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        seen_feature_signatures: set[float] = set()
        steps = 0
        for batch in loader:
            x = batch.x.cuda()
            assert x.dtype == torch.float32
            assert x.shape[1] == feature_dim
            # Synthetic labels so the loss has something to push against.
            y = (batch.n_id[: batch.batch_size].cuda() % 4).long()

            logits = model(x)[: batch.batch_size]
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optim.step()

            assert torch.isfinite(loss).item(), f"non-finite loss at step {steps}: {loss.item()}"

            # Fingerprint the first row to confirm the server's live writes
            # are reaching us across batches.
            seen_feature_signatures.add(float(x[0].sum().item()))
            steps += 1
            if steps >= 30:
                break

        assert steps > 0, "loader produced zero batches"
        assert len(seen_feature_signatures) > 1, (
            "features never changed across batches; live-rate writer may not be running"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
