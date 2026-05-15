"""Python NeighborLoader against an RDMA `feature_source="rdma://..."`.

Two layers:

1. **CPU-only error path** (always runnable). Verify that constructing
   `NeighborLoader(..., feature_source="rdma://...")` against an extension
   built without `--features gpudirect` raises a clear `RuntimeError`.
2. **Full end-to-end** (skipped unless CUDA + RoCE present). Spawns the
   `rdma_feature_server` example, points the loader at it, iterates one
   batch, and asserts `batch.x` is on CUDA with the right shape.
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


# ---------------------------------------------------------------------------
# Layer 1 — CPU-only error path
# ---------------------------------------------------------------------------


def test_feature_source_rdma_requires_gpudirect_build() -> None:
    """Building NeighborLoader with `rdma://` and no gpudirect must raise."""
    from aethergraph._core import HAS_GPUDIRECT

    if HAS_GPUDIRECT:
        pytest.skip("extension was built with --features gpudirect; error path not reachable")

    from aethergraph import Graph
    from aethergraph.pytorch import NeighborLoader

    rng = np.random.default_rng(0)
    num_nodes, num_edges = 32, 64
    src = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    dst = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    graph = Graph.from_edges(num_nodes, src, dst)

    with pytest.raises(RuntimeError, match="rdma://"):
        NeighborLoader(
            graph,
            num_neighbors=[5],
            batch_size=4,
            feature_source="rdma://127.0.0.1:9999",
        )


# ---------------------------------------------------------------------------
# Layer 2 — full subprocess E2E (gated)
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def _has_rxe_or_real_rdma() -> bool:
    """Best-effort detection of a usable RDMA device for SoftRoCE / real HCAs."""
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
    """Build the `rdma_feature_server` example and return its path."""
    cmd = [
        "cargo",
        "build",
        "--release",
        "--features",
        "rdma",
        "-p",
        "aether-stream",
        "--example",
        "rdma_feature_server",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
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
            # EOF before READY — process is exiting; loop will catch it via poll().
            continue
        if "READY" in line:
            return


def test_neighbor_loader_gathers_features_over_rdma() -> None:
    """End-to-end: server populates table, Python loader gathers via RDMA."""
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

    from aethergraph import Graph
    from aethergraph.pytorch import NeighborLoader

    server_bin = _build_server_example()
    port = _pick_free_port()
    bind = f"127.0.0.1:{port}"

    feature_dim = 32
    num_nodes = 128

    proc = subprocess.Popen(
        [
            str(server_bin),
            "--bind",
            bind,
            "--nodes",
            str(num_nodes),
            "--dim",
            str(feature_dim),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_ready(proc, timeout_s=15.0)

        # Build a small graph and ask the loader to gather features from the
        # remote server. The graph nodes line up with the server's table.
        rng = np.random.default_rng(7)
        src = rng.integers(0, num_nodes, size=512, dtype=np.uint32)
        dst = rng.integers(0, num_nodes, size=512, dtype=np.uint32)
        graph = Graph.from_edges(num_nodes, src, dst)

        loader = NeighborLoader(
            graph,
            num_neighbors=[5, 3],
            batch_size=8,
            feature_source=f"rdma://{bind}",
        )

        batch = next(iter(loader))
        assert batch.x is not None, "feature_source set, but batch.x is None"
        assert batch.x.device.type == "cuda", f"expected CUDA tensor, got {batch.x.device}"
        assert batch.x.dtype == torch.float32
        assert batch.x.shape[1] == feature_dim, (
            f"expected feature_dim={feature_dim}, got {batch.x.shape[1]}"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
