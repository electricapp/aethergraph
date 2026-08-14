"""Head-to-head benchmark: AetherGraph vs DGL-GraphBolt.

GraphBolt is the closest competitor, so it's the comparison worth running.

Held equal: the edge list, the seed batches and their order, the fanout,
batch size, replacement policy, and thread count. The graph is symmetrized
because GraphBolt samples in-edges over CSC and we sample out-edges over
CSR — on a directed graph the two sample different neighbourhoods.

Not equal: GraphBolt compacts node IDs inside `sample_neighbors`, so its
number includes work our raw `sample()` defers. Treat the sample-only row
as directional.

Keep `--threads 1`. A single `sample()` is serial here (raising the flag
moves the number by less than noise), so above 1 the setting only frees
GraphBolt's threads and the comparison tilts the other way.

Note this runs entirely in memory, which is where we have the least to
offer — io_uring, O_DIRECT, hugepages and the rest only matter once the
graph outgrows page cache. Read a loss here as "competitive on table
stakes", not as the headline.

Requires: aethergraph; torch and dgl for the GraphBolt side.

Usage::

    uv run python benchmarks/bench_vs_graphbolt.py
    uv run python benchmarks/bench_vs_graphbolt.py --scale large --json out.json
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="AetherGraph vs DGL-GraphBolt head-to-head benchmark.")
console = Console()


class Scale(str, Enum):
    """Graph size for benchmarking."""

    small = "small"
    medium = "medium"
    large = "large"


SCALES = {
    Scale.small: {"nodes": 10_000, "edges": 100_000, "iters": 200},
    Scale.medium: {"nodes": 100_000, "edges": 1_000_000, "iters": 100},
    Scale.large: {"nodes": 1_000_000, "edges": 10_000_000, "iters": 30},
}


@dataclass
class BenchResult:
    """One measurement, with the conditions that produced it."""

    name: str
    framework: str
    us_per_iter: float
    batches_per_sec: float
    threads: int
    note: str = ""


def _symmetric_edges(
    num_nodes: int, num_edges: int, seed: int
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """A symmetric edge list, so CSR out-edges and CSC in-edges agree.

    Without this the two frameworks sample different neighbourhoods and the
    comparison measures graph orientation rather than sampler speed.
    """
    rng = np.random.default_rng(seed)
    src = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    dst = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    both_src = np.concatenate([src, dst])
    both_dst = np.concatenate([dst, src])
    return both_src, both_dst


def _timed(fn: Callable[[], None], iters: int, warmup: int = 5) -> float:
    """Warm up, then time `iters` calls; return microseconds per call."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1e6


def _seed_batches(
    num_nodes: int, batch_size: int, count: int, seed: int
) -> list[npt.NDArray[np.uint32]]:
    """Fixed seed batches, replayed identically to both frameworks."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, num_nodes, size=batch_size, dtype=np.uint32) for _ in range(count)]


def _cycle(batches: list[npt.NDArray[np.uint32]]) -> Iterator[npt.NDArray[np.uint32]]:
    while True:
        yield from batches


def bench_aethergraph(
    src: npt.NDArray[np.uint32],
    dst: npt.NDArray[np.uint32],
    num_nodes: int,
    batches: list[npt.NDArray[np.uint32]],
    fanout: list[int],
    iters: int,
    threads: int,
) -> list[BenchResult]:
    """Raw sampling throughput on the shared graph."""
    from aethergraph._core import CsrGraph, NeighborSampler, SamplingConfig

    graph = CsrGraph.from_edges(num_nodes, src, dst)
    sampler = NeighborSampler(graph, SamplingConfig(num_neighbors=fanout, replace=False))
    feed = _cycle(batches)

    def run() -> None:
        sampler.sample(next(feed))

    us = _timed(run, iters)
    return [
        BenchResult(
            name="sample only",
            framework="AetherGraph",
            us_per_iter=us,
            batches_per_sec=1e6 / us,
            threads=threads,
            note="no ID compaction",
        )
    ]


def bench_graphbolt(
    src: npt.NDArray[np.uint32],
    dst: npt.NDArray[np.uint32],
    num_nodes: int,
    batches: list[npt.NDArray[np.uint32]],
    fanout: list[int],
    iters: int,
    threads: int,
) -> list[BenchResult]:
    """The same sampling through GraphBolt's fused CSC sampler."""
    import torch
    from dgl import graphbolt as gb

    # GraphBolt wants a CSC. `from_csc` needs sorted indptr/indices, which
    # is what the shared symmetric edge list is converted into here — the
    # same edges both sides sample, laid out the way each expects.
    order = np.argsort(dst, kind="stable")
    sorted_src = src[order]
    sorted_dst = dst[order]
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(indptr, sorted_dst.astype(np.int64) + 1, 1)
    np.cumsum(indptr, out=indptr)

    graph = gb.fused_csc_sampling_graph(
        torch.from_numpy(indptr),
        torch.from_numpy(sorted_src.astype(np.int64)),
    )
    feed = _cycle(batches)
    fanout_t = [torch.LongTensor([f]) for f in fanout]

    def run() -> None:
        seeds = torch.from_numpy(next(feed).astype(np.int64))
        for f in fanout_t:
            seeds = graph.sample_neighbors(seeds, f).indices

    us = _timed(run, iters)
    return [
        BenchResult(
            name="sample only",
            framework="GraphBolt",
            us_per_iter=us,
            batches_per_sec=1e6 / us,
            threads=threads,
            note="includes fused ID compaction",
        )
    ]


def _render(results: list[BenchResult]) -> None:
    table = Table(title="AetherGraph vs DGL-GraphBolt")
    table.add_column("stage")
    table.add_column("framework")
    table.add_column("µs/batch", justify="right")
    table.add_column("batches/s", justify="right")
    table.add_column("threads", justify="right")
    table.add_column("caveat")
    for r in results:
        table.add_row(
            r.name,
            r.framework,
            f"{r.us_per_iter:,.1f}",
            f"{r.batches_per_sec:,.0f}",
            str(r.threads),
            r.note,
        )
    console.print(table)

    by_stage: dict[str, dict[str, float]] = {}
    for r in results:
        by_stage.setdefault(r.name, {})[r.framework] = r.us_per_iter
    for stage, entry in by_stage.items():
        if len(entry) == 2:
            ag = entry["AetherGraph"]
            gbt = entry["GraphBolt"]
            faster, ratio = ("AetherGraph", gbt / ag) if ag < gbt else ("GraphBolt", ag / gbt)
            console.print(f"[bold]{stage}[/bold]: {faster} by {ratio:.2f}×")


@app.command()
def main(
    scale: Scale = Scale.medium,
    batch_size: int = 1024,
    fanout: str = "25,10",
    threads: int = 1,
    seed: int = 42,
    json_out: Path | None = typer.Option(None, "--json", help="Write results as JSON."),
) -> None:
    """Run both frameworks over one shared graph and report the delta."""
    # Pin our side too. `torch.set_num_threads` only binds GraphBolt; rayon
    # reads this when it builds its pool, which happens on first use, so it
    # has to be set before aethergraph is imported below. Without it a
    # parallel sampling path would take every core while GraphBolt ran on
    # one, and the benchmark would claim a control it never applied.
    os.environ.setdefault("RAYON_NUM_THREADS", str(threads))

    # torch is GraphBolt's dependency, not ours — importing it up front would
    # make our own numbers unobtainable on a machine that only has us.
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        console.print("[yellow]torch not installed; thread count not pinned[/]")

    cfg = SCALES[scale]
    fan = [int(x) for x in fanout.split(",")]
    src, dst = _symmetric_edges(cfg["nodes"], cfg["edges"], seed)
    batches = _seed_batches(cfg["nodes"], batch_size, 64, seed)

    console.print(
        f"nodes={cfg['nodes']:,} edges={cfg['edges']:,} (symmetrized to "
        f"{len(src):,}) batch={batch_size} fanout={fan} threads={threads}"
    )

    results = bench_aethergraph(src, dst, cfg["nodes"], batches, fan, cfg["iters"], threads)
    try:
        results += bench_graphbolt(src, dst, cfg["nodes"], batches, fan, cfg["iters"], threads)
    except ImportError:
        console.print("[yellow]dgl.graphbolt not installed; AetherGraph numbers only[/]")

    _render(results)
    if json_out is not None:
        json_out.write_text(json.dumps([asdict(r) for r in results], indent=2))
        console.print(f"wrote {json_out}")


if __name__ == "__main__":
    app()
