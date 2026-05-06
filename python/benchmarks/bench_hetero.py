"""Heterogeneous graph sampling benchmarks.

Measures Python->Rust->Python roundtrip for the full HeteroNeighborLoader
pipeline. Quick feedback on hot paths.

Usage::

    uv run python benchmarks/bench_hetero.py              # quick (10s)
    uv run python benchmarks/bench_hetero.py --scale large # heavier (60s)
    uv run python benchmarks/bench_hetero.py --scale huge  # stress test
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from aethergraph._core import (
    HeteroCsrGraph,
    HeteroNeighborSampler,
    HeteroSamplingConfig,
)

app = typer.Typer(help="Heterogeneous graph sampling benchmarks.")
console = Console()


class Scale(str, Enum):
    """Benchmark intensity level.

    Attributes:
        quick: 10K users, 260K nodes, 650K edges, 100 iters. ~10 seconds.
        large: 100K users, 2.6M nodes, 6.5M edges, 500 iters. ~60 seconds.
        huge: 1M users, 26M nodes, 65M edges, 200 iters. Stress test.
    """

    quick = "quick"
    large = "large"
    huge = "huge"


SCALES = {
    Scale.quick: {"users": 10_000, "posts": 50_000, "comments": 200_000, "subs": 1_000, "iters": 100},
    Scale.large: {"users": 100_000, "posts": 500_000, "comments": 2_000_000, "subs": 10_000, "iters": 500},
    Scale.huge:  {"users": 1_000_000, "posts": 5_000_000, "comments": 20_000_000, "subs": 100_000, "iters": 200},
}


@dataclass
class BenchResult:
    """Single benchmark measurement.

    Attributes:
        name: Human-readable benchmark identifier.
        iters: Number of iterations measured.
        total_ms: Wall-clock time for all iterations in milliseconds.
        per_iter_us: Average time per iteration in microseconds.
        throughput: Human-readable throughput string (e.g. "10,000,000 nodes/s").
    """

    name: str
    iters: int
    total_ms: float
    per_iter_us: float
    throughput: str


def build_graph(scale: dict[str, int]) -> HeteroCsrGraph:
    """Build a Reddit-shaped heterogeneous graph with synthetic edges.

    Creates four node types (user, post, comment, subreddit) and four edge
    types (votes, writes, reply_to, belongs_to) with random connectivity.
    Edge counts scale linearly with user count: 20 votes/user, 30 writes/user,
    0.5 replies/comment, 1 belongs_to/post.

    Args:
        scale: Dict with keys "users", "posts", "comments", "subs" mapping to
            node counts for each type.

    Returns:
        A ``HeteroCsrGraph`` ready for sampling benchmarks.
    """
    rng = np.random.default_rng(42)
    u, p, c, s = scale["users"], scale["posts"], scale["comments"], scale["subs"]

    with console.status(f"Building graph: {u:,} users, {p:,} posts, {c:,} comments, {s:,} subreddits..."):
        t0 = time.perf_counter()
        graph = HeteroCsrGraph.from_edge_arrays(
            node_types={"user": u, "post": p, "comment": c, "subreddit": s},
            edge_types=[
                ("user", "votes", "post",
                 rng.integers(0, u, u * 20).astype(np.uint32),
                 rng.integers(0, p, u * 20).astype(np.uint32)),
                ("user", "writes", "comment",
                 rng.integers(0, u, u * 30).astype(np.uint32),
                 rng.integers(0, c, u * 30).astype(np.uint32)),
                ("comment", "reply_to", "comment",
                 rng.integers(0, c, c // 2).astype(np.uint32),
                 rng.integers(0, c, c // 2).astype(np.uint32)),
                ("post", "belongs_to", "subreddit",
                 rng.integers(0, p, p).astype(np.uint32),
                 rng.integers(0, s, p).astype(np.uint32)),
            ],
        )
        build_ms = (time.perf_counter() - t0) * 1000

    console.print(f"  Built {graph.total_edges():,} edges in {build_ms:.0f}ms")
    return graph


def bench_sample(
    graph: HeteroCsrGraph,
    fanout: list[int],
    batch_size: int,
    iters: int,
) -> BenchResult:
    """Benchmark multi-hop heterogeneous neighbor sampling.

    Samples ``iters`` batches of ``batch_size`` user seeds through all edge
    types with the given per-hop fanout. Measures total sampled nodes per
    second as the throughput metric.

    Runs 5 warmup iterations before timing to eliminate JIT and cache effects.

    Args:
        graph: The heterogeneous graph to sample from.
        fanout: Per-hop fanout list (e.g. ``[15, 10]`` for 2-hop).
            Applied to user edge types directly; reply_to gets half, belongs_to
            gets one-third to reflect realistic asymmetric fanout.
        batch_size: Number of seed user nodes per sample call.
        iters: Number of timed iterations.

    Returns:
        A ``BenchResult`` with per-iteration latency and node throughput.
    """
    config = HeteroSamplingConfig(
        num_neighbors={
            ("user", "votes", "post"): fanout,
            ("user", "writes", "comment"): fanout,
            ("comment", "reply_to", "comment"): [f // 2 for f in fanout],
            ("post", "belongs_to", "subreddit"): [max(1, f // 3) for f in fanout],
        },
        seed=42,
    )
    sampler = HeteroNeighborSampler(graph, config)
    users = graph.num_nodes("user")
    rng = np.random.default_rng(99)

    for _ in range(5):
        seeds = rng.integers(0, users, batch_size).astype(np.uint32)
        sampler.sample("user", seeds)

    t0 = time.perf_counter()
    total_nodes = 0
    for _ in range(iters):
        seeds = rng.integers(0, users, batch_size).astype(np.uint32)
        sub = sampler.sample("user", seeds)
        total_nodes += sum(len(sub.nodes(nt)) for nt in sub.node_types())
    elapsed = time.perf_counter() - t0

    hops = len(fanout)
    return BenchResult(
        name=f"sample_{hops}hop_fan{fanout[0]}_batch{batch_size}",
        iters=iters,
        total_ms=elapsed * 1000,
        per_iter_us=elapsed / iters * 1e6,
        throughput=f"{total_nodes / elapsed:,.0f} nodes/s",
    )


def bench_edge_index_local(graph: HeteroCsrGraph, iters: int) -> BenchResult:
    """Benchmark edge index local remapping across all edge types.

    Pre-samples a 2-hop subgraph from 128 user seeds, then measures the
    time to remap all edge types' source/destination IDs to local indices
    via binary search. This is the operation that converts sampled global
    node IDs into the ``[0, N)`` range expected by PyG's ``edge_index``.

    Args:
        graph: The heterogeneous graph to sample from.
        iters: Number of timed iterations (each remaps all edge types once).

    Returns:
        A ``BenchResult`` with per-iteration latency and remap throughput.
    """
    config = HeteroSamplingConfig(
        num_neighbors={
            ("user", "votes", "post"): [15, 10],
            ("user", "writes", "comment"): [15, 10],
            ("comment", "reply_to", "comment"): [5, 5],
            ("post", "belongs_to", "subreddit"): [3, 3],
        },
        seed=42,
    )
    sampler = HeteroNeighborSampler(graph, config)
    seeds = np.arange(128, dtype=np.uint32)
    sub = sampler.sample("user", seeds)

    for _ in range(5):
        for et in sub.edge_types():
            sub.edge_index_local(*et)

    t0 = time.perf_counter()
    for _ in range(iters):
        for et in sub.edge_types():
            sub.edge_index_local(*et)
    elapsed = time.perf_counter() - t0

    return BenchResult(
        name="edge_index_local_all_types",
        iters=iters,
        total_ms=elapsed * 1000,
        per_iter_us=elapsed / iters * 1e6,
        throughput=f"{iters * len(sub.edge_types()) / elapsed:,.0f} remap/s",
    )


def bench_construction() -> BenchResult:
    """Benchmark heterogeneous graph construction from edge arrays.

    Measures the time to build a single-edge-type graph with 10K nodes and
    100K edges, including CSR sorting. This is the graph ingestion hot path
    for both file loading and ``from_pyg()`` conversion.

    Returns:
        A ``BenchResult`` with per-iteration latency and edge throughput.
    """
    rng = np.random.default_rng(42)
    n, e, iters = 10_000, 100_000, 50
    src = rng.integers(0, n, e).astype(np.uint32)
    dst = rng.integers(0, n, e).astype(np.uint32)

    t0 = time.perf_counter()
    for _ in range(iters):
        HeteroCsrGraph.from_edge_arrays(
            node_types={"a": n, "b": n},
            edge_types=[("a", "rel", "b", src, dst)],
        )
    elapsed = time.perf_counter() - t0

    return BenchResult(
        name="construction_10k_100k_edges",
        iters=iters,
        total_ms=elapsed * 1000,
        per_iter_us=elapsed / iters * 1e6,
        throughput=f"{iters * e / elapsed:,.0f} edges/s",
    )


@app.command()
def run(
    scale: Scale = typer.Option(Scale.quick, help="Benchmark scale: quick (~10s), large (~60s), huge (stress)"),
) -> None:
    """Run heterogeneous graph sampling benchmarks.

    Exercises the full Python->Rust->Python roundtrip for graph construction,
    multi-hop typed sampling, and edge index remapping. Results are displayed
    as a rich table with per-iteration latency and throughput.

    All benchmarks use a Reddit-shaped graph with user/post/comment/subreddit
    node types and votes/writes/reply_to/belongs_to edge types.
    """
    cfg = SCALES[scale]
    iters = cfg["iters"]
    graph = build_graph(cfg)

    results: list[BenchResult] = []

    with console.status("Running benchmarks..."):
        for fanout in [[15], [15, 10], [15, 10, 5]]:
            for batch_size in [32, 128, 512]:
                if len(fanout) == 3 and batch_size == 512:
                    continue
                results.append(bench_sample(graph, fanout, batch_size, iters))

        results.append(bench_edge_index_local(graph, iters * 5))
        results.append(bench_construction())

    table = Table(title=f"Hetero Sampling Benchmarks ({scale.value})")
    table.add_column("Benchmark", style="cyan", no_wrap=True)
    table.add_column("us/iter", justify="right", style="green")
    table.add_column("Throughput", justify="right", style="yellow")
    table.add_column("Total ms", justify="right", style="dim")

    for r in results:
        table.add_row(r.name, f"{r.per_iter_us:.1f}", r.throughput, f"{r.total_ms:.0f}")

    console.print()
    console.print(table)
    console.print(f"\n[bold]Total: {sum(r.total_ms for r in results):.0f}ms[/bold]")


if __name__ == "__main__":
    app()
