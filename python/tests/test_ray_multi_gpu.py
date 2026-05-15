"""Ray Data sampling across multiple GPU workers.

Runs the high-level `create_sampling_dataset` pipeline with a parallelism
matching the number of visible CUDA devices, then iterates batches across
all workers and confirms aggregate throughput scales meaningfully versus a
single-worker baseline.

Skipped unless `torch.cuda.device_count() >= 2`. On a single-GPU box the
parallel iteration would still work but the scaling assertion is the
point of the test, so we require at least two GPUs.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_torch, pytest.mark.requires_ray]


def _skip_unless_multi_gpu() -> int:
    """Return the GPU count, or skip the test if fewer than two are visible."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")
    n = torch.cuda.device_count()
    if n < 2:
        pytest.skip(f"need at least 2 GPUs, found {n}")
    return n


def _drain(dataset: object, max_batches: int) -> tuple[int, float]:
    """Iterate up to `max_batches` from `dataset`, returning (count, elapsed_s)."""
    start = time.perf_counter()
    count = 0
    for _ in dataset.iter_batches(batch_size=1):  # type: ignore[attr-defined]
        count += 1
        if count >= max_batches:
            break
    return count, time.perf_counter() - start


def test_ray_dataset_scales_across_gpus(temp_dir: Path) -> None:
    """Aggregate batches/sec with N-worker parallelism > 1-worker baseline.

    The exact scaling factor is sensitive to PCIe topology and per-batch
    cost, so the assertion is loose (≥ 1.5× of the single-worker rate at
    N≥2 GPUs). The test's main value is catching regressions where
    parallelism collapses to serial execution, not measuring the actual
    speedup curve.
    """
    n_gpus = _skip_unless_multi_gpu()

    import ray

    from aethergraph import Graph
    from aethergraph.ray import create_sampling_dataset

    rng = np.random.default_rng(0)
    num_nodes = 4096
    src = rng.integers(0, num_nodes, size=32_768, dtype=np.uint32)
    dst = rng.integers(0, num_nodes, size=32_768, dtype=np.uint32)
    graph = Graph.from_edges(num_nodes, src, dst)

    graph_path = temp_dir / "graph.bin"
    graph.save(graph_path)

    if not ray.is_initialized():
        ray.init(num_cpus=max(4, n_gpus * 2), runtime_env={"working_dir": None})
    try:
        baseline = create_sampling_dataset(
            graph_path=graph_path,
            num_neighbors=[15, 10],
            batch_size=128,
            parallelism=1,
        )
        baseline_count, baseline_secs = _drain(baseline, max_batches=64)
        baseline_rate = baseline_count / max(baseline_secs, 1e-6)

        parallel = create_sampling_dataset(
            graph_path=graph_path,
            num_neighbors=[15, 10],
            batch_size=128,
            parallelism=n_gpus,
        )
        parallel_count, parallel_secs = _drain(parallel, max_batches=64 * n_gpus)
        parallel_rate = parallel_count / max(parallel_secs, 1e-6)

        # Loose threshold: parallel should be at least 1.5× single-worker.
        # Anything below that is suspicious of serialization (one worker
        # doing all the work, or GIL contention if the loader sneaks back
        # into Python under the hood).
        assert parallel_rate >= baseline_rate * 1.5, (
            f"parallel {parallel_rate:.1f} batches/s vs baseline "
            f"{baseline_rate:.1f} batches/s — no scaling observed"
        )
    finally:
        ray.shutdown()
