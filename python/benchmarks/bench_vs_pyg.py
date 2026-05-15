"""Head-to-head benchmark: AetherGraph vs PyTorch Geometric.

Apples-to-apples comparison on identical graphs. Measures raw sampling
throughput (no edge remapping) and full pipeline (including Data
construction). Both frameworks do the exact same work.

Requires: torch, torch_geometric (with pyg-lib), aethergraph

Usage::

    uv run python benchmarks/bench_vs_pyg.py
    uv run python benchmarks/bench_vs_pyg.py --scale medium
    uv run python benchmarks/bench_vs_pyg.py --scale large
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="AetherGraph vs PyG head-to-head benchmark.")
console = Console()


class Scale(str, Enum):
    """Graph size for benchmarking."""

    small = "small"
    medium = "medium"
    large = "large"


SCALES = {
    Scale.small: {"nodes": 10_000, "edges": 100_000, "iters": 500},
    Scale.medium: {"nodes": 100_000, "edges": 1_000_000, "iters": 200},
    Scale.large: {"nodes": 1_000_000, "edges": 10_000_000, "iters": 50},
}


@dataclass
class BenchResult:
    """Single benchmark measurement."""

    name: str
    framework: str
    us_per_iter: float


def _timed_iters(fn: Callable[[], None], iters: int, warmup: int = 10) -> float:
    """Run warmup + timed iterations, return total seconds."""
    for _ in range(warmup):
        fn()
    gc.disable()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - t0
    gc.enable()
    return elapsed


def _make_graph(
    scale: dict[str, int],
) -> tuple[int, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Build identical random edges for both frameworks."""
    rng = np.random.default_rng(42)
    n, e = scale["nodes"], scale["edges"]
    src = rng.integers(0, n, e).astype(np.int64)
    dst = rng.integers(0, n, e).astype(np.int64)
    return n, src, dst


# ---------------------------------------------------------------------------
# Homogeneous benchmarks
# ---------------------------------------------------------------------------


def bench_homo_ag(scale: dict[str, int], fanout: list[int], batch_size: int) -> BenchResult:
    """AetherGraph homogeneous: sample only (no edge remapping)."""
    from aethergraph._core import CsrGraph, NeighborSampler, SamplingConfig

    n, src, dst = _make_graph(scale)
    graph = CsrGraph.from_edges(n, src.astype(np.uint32), dst.astype(np.uint32))
    config = SamplingConfig(num_neighbors=fanout, replace=False)
    sampler = NeighborSampler(graph, config)

    rng = np.random.default_rng(99)
    iters = scale["iters"]
    seeds = [rng.integers(0, n, batch_size).astype(np.int64) for _ in range(iters + 15)]
    idx = [0]

    def run() -> None:
        sampler.sample(seeds[idx[0]])
        idx[0] += 1

    elapsed = _timed_iters(run, iters)
    return BenchResult(f"homo_{len(fanout)}hop_b{batch_size}", "ag", elapsed / iters * 1e6)


def bench_homo_pyg(scale: dict[str, int], fanout: list[int], batch_size: int) -> BenchResult:
    """PyG homogeneous: sample only (low-level NeighborSampler)."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.sampler import NeighborSampler, NodeSamplerInput, NumNeighbors

    n, src, dst = _make_graph(scale)
    edge_index = torch.stack([torch.from_numpy(src), torch.from_numpy(dst)])
    data = Data(edge_index=edge_index, num_nodes=n)
    sampler = NeighborSampler(data, num_neighbors=NumNeighbors(fanout))

    rng = np.random.default_rng(99)
    iters = scale["iters"]
    seeds = [
        torch.from_numpy(rng.integers(0, n, batch_size).astype(np.int64)) for _ in range(iters + 15)
    ]
    idx = [0]

    def run() -> None:
        inp = NodeSamplerInput(input_id=torch.arange(batch_size), node=seeds[idx[0]])
        sampler.sample_from_nodes(inp)
        idx[0] += 1

    elapsed = _timed_iters(run, iters)
    return BenchResult(f"homo_{len(fanout)}hop_b{batch_size}", "pyg", elapsed / iters * 1e6)


# ---------------------------------------------------------------------------
# Weighted benchmarks
# ---------------------------------------------------------------------------


def _make_weighted_graph(
    scale: dict[str, int],
) -> tuple[
    int,
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float32],
]:
    """Build identical random edges + weights for both frameworks."""
    rng = np.random.default_rng(42)
    n, e = scale["nodes"], scale["edges"]
    src = rng.integers(0, n, e).astype(np.int64)
    dst = rng.integers(0, n, e).astype(np.int64)
    weights = rng.random(e).astype(np.float32) * 10.0
    return n, src, dst, weights


def bench_weighted_ag(scale: dict[str, int], fanout: list[int], batch_size: int) -> BenchResult:
    """AetherGraph weighted sampling."""
    from aethergraph._core import CsrGraph, NeighborSampler, SamplingConfig

    n, src, dst, weights = _make_weighted_graph(scale)
    graph = CsrGraph.from_edges(n, src.astype(np.uint32), dst.astype(np.uint32), weights)
    config = SamplingConfig(num_neighbors=fanout, replace=False, weighted=True)
    sampler = NeighborSampler(graph, config)

    rng = np.random.default_rng(99)
    iters = scale["iters"]
    seeds = [rng.integers(0, n, batch_size).astype(np.int64) for _ in range(iters + 15)]
    idx = [0]

    def run() -> None:
        sampler.sample(seeds[idx[0]])
        idx[0] += 1

    elapsed = _timed_iters(run, iters)
    return BenchResult(f"weighted_{len(fanout)}hop_b{batch_size}", "ag", elapsed / iters * 1e6)


def bench_weighted_pyg(scale: dict[str, int], fanout: list[int], batch_size: int) -> BenchResult:
    """PyG weighted sampling."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.sampler import NeighborSampler, NodeSamplerInput, NumNeighbors

    n, src, dst, weights = _make_weighted_graph(scale)
    edge_index = torch.stack([torch.from_numpy(src), torch.from_numpy(dst)])
    data = Data(edge_index=edge_index, num_nodes=n)
    data.edge_weight = torch.from_numpy(weights)
    sampler = NeighborSampler(data, num_neighbors=NumNeighbors(fanout), weight_attr="edge_weight")

    rng = np.random.default_rng(99)
    iters = scale["iters"]
    seeds = [
        torch.from_numpy(rng.integers(0, n, batch_size).astype(np.int64)) for _ in range(iters + 15)
    ]
    idx = [0]

    def run() -> None:
        inp = NodeSamplerInput(input_id=torch.arange(batch_size), node=seeds[idx[0]])
        sampler.sample_from_nodes(inp)
        idx[0] += 1

    elapsed = _timed_iters(run, iters)
    return BenchResult(f"weighted_{len(fanout)}hop_b{batch_size}", "pyg", elapsed / iters * 1e6)


# ---------------------------------------------------------------------------
# Temporal benchmarks
# ---------------------------------------------------------------------------


def _make_temporal_graph(
    scale: dict[str, int],
) -> tuple[
    int,
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
]:
    """Build identical random edges + timestamps for both frameworks."""
    rng = np.random.default_rng(42)
    n, e = scale["nodes"], scale["edges"]
    src = rng.integers(0, n, e).astype(np.int64)
    dst = rng.integers(0, n, e).astype(np.int64)
    # Monotonically increasing timestamps
    timestamps = np.sort(rng.random(e).astype(np.float64) * 1000.0)
    return n, src, dst, timestamps


def bench_temporal_ag(
    scale: dict[str, int],
    fanout: list[int],
    batch_size: int,
    strategy: str,
) -> BenchResult:
    """AetherGraph temporal sampling."""
    from aethergraph._core import CsrGraph, NeighborSampler, SamplingConfig

    n, src, dst, timestamps = _make_temporal_graph(scale)
    graph = CsrGraph.from_edges(n, src.astype(np.uint32), dst.astype(np.uint32))
    graph.set_timestamps(timestamps)
    config = SamplingConfig(
        num_neighbors=fanout,
        replace=False,
        temporal_strategy=strategy,
    )
    sampler = NeighborSampler(graph, config)

    mid_time = float(np.median(timestamps))
    rng = np.random.default_rng(99)
    iters = scale["iters"]
    seeds = [rng.integers(0, n, batch_size).astype(np.int64) for _ in range(iters + 15)]
    input_times = np.full(batch_size, mid_time, dtype=np.float64)
    idx = [0]

    def run() -> None:
        sampler.sample(seeds[idx[0]], input_times=input_times)
        idx[0] += 1

    elapsed = _timed_iters(run, iters)
    return BenchResult(
        f"temporal_{strategy}_{len(fanout)}hop_b{batch_size}", "ag", elapsed / iters * 1e6
    )


def bench_temporal_pyg(
    scale: dict[str, int],
    fanout: list[int],
    batch_size: int,
    strategy: str,
) -> BenchResult:
    """PyG temporal sampling."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.sampler import NeighborSampler, NodeSamplerInput, NumNeighbors

    n, src, dst, timestamps = _make_temporal_graph(scale)
    edge_index = torch.stack([torch.from_numpy(src), torch.from_numpy(dst)])
    data = Data(edge_index=edge_index, num_nodes=n)
    # PyG expects edge-level time as int (or long) tensor
    edge_time = torch.from_numpy((timestamps * 1000).astype(np.int64))
    data.edge_time = edge_time

    sampler = NeighborSampler(
        data,
        num_neighbors=NumNeighbors(fanout),
        time_attr="edge_time",
        temporal_strategy=strategy,
    )

    mid_time = int(np.median(timestamps) * 1000)
    rng = np.random.default_rng(99)
    iters = scale["iters"]
    seeds = [
        torch.from_numpy(rng.integers(0, n, batch_size).astype(np.int64)) for _ in range(iters + 15)
    ]
    seed_times = torch.full((batch_size,), mid_time, dtype=torch.long)
    idx = [0]

    def run() -> None:
        inp = NodeSamplerInput(
            input_id=torch.arange(batch_size),
            node=seeds[idx[0]],
            time=seed_times,
        )
        sampler.sample_from_nodes(inp)
        idx[0] += 1

    elapsed = _timed_iters(run, iters)
    return BenchResult(
        f"temporal_{strategy}_{len(fanout)}hop_b{batch_size}", "pyg", elapsed / iters * 1e6
    )


# ---------------------------------------------------------------------------
# Disjoint benchmarks
# ---------------------------------------------------------------------------


def bench_disjoint_ag(scale: dict[str, int], fanout: list[int], batch_size: int) -> BenchResult:
    """AetherGraph disjoint sampling."""
    from aethergraph._core import CsrGraph, NeighborSampler, SamplingConfig

    n, src, dst = _make_graph(scale)
    graph = CsrGraph.from_edges(n, src.astype(np.uint32), dst.astype(np.uint32))
    config = SamplingConfig(num_neighbors=fanout, replace=False, disjoint=True)
    sampler = NeighborSampler(graph, config)

    rng = np.random.default_rng(99)
    iters = scale["iters"]
    seeds = [rng.integers(0, n, batch_size).astype(np.int64) for _ in range(iters + 15)]
    idx = [0]

    def run() -> None:
        sampler.sample(seeds[idx[0]])
        idx[0] += 1

    elapsed = _timed_iters(run, iters)
    return BenchResult(f"disjoint_{len(fanout)}hop_b{batch_size}", "ag", elapsed / iters * 1e6)


def bench_disjoint_pyg(scale: dict[str, int], fanout: list[int], batch_size: int) -> BenchResult:
    """PyG disjoint sampling."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.sampler import NeighborSampler, NodeSamplerInput, NumNeighbors

    n, src, dst = _make_graph(scale)
    edge_index = torch.stack([torch.from_numpy(src), torch.from_numpy(dst)])
    data = Data(edge_index=edge_index, num_nodes=n)
    sampler = NeighborSampler(data, num_neighbors=NumNeighbors(fanout), disjoint=True)

    rng = np.random.default_rng(99)
    iters = scale["iters"]
    seeds = [
        torch.from_numpy(rng.integers(0, n, batch_size).astype(np.int64)) for _ in range(iters + 15)
    ]
    idx = [0]

    def run() -> None:
        inp = NodeSamplerInput(input_id=torch.arange(batch_size), node=seeds[idx[0]])
        sampler.sample_from_nodes(inp)
        idx[0] += 1

    elapsed = _timed_iters(run, iters)
    return BenchResult(f"disjoint_{len(fanout)}hop_b{batch_size}", "pyg", elapsed / iters * 1e6)


# ---------------------------------------------------------------------------
# Heterogeneous benchmarks
# ---------------------------------------------------------------------------


def _make_hetero_edges(
    scale: dict[str, int],
) -> tuple[
    int,
    int,
    int,
    int,
    dict[
        tuple[str, str, str],
        tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
    ],
]:
    """Build identical hetero edges for both frameworks."""
    rng = np.random.default_rng(42)
    n = scale["nodes"]
    u = n // 10
    p = n // 5
    c = n // 2
    s = n // 100
    return (
        u,
        p,
        c,
        s,
        {
            ("user", "votes", "post"): (
                rng.integers(0, u, u * 10).astype(np.int64),
                rng.integers(0, p, u * 10).astype(np.int64),
            ),
            ("user", "writes", "comment"): (
                rng.integers(0, u, u * 15).astype(np.int64),
                rng.integers(0, c, u * 15).astype(np.int64),
            ),
            ("comment", "reply_to", "comment"): (
                rng.integers(0, c, c // 2).astype(np.int64),
                rng.integers(0, c, c // 2).astype(np.int64),
            ),
            ("post", "belongs_to", "subreddit"): (
                rng.integers(0, p, p).astype(np.int64),
                rng.integers(0, s, p).astype(np.int64),
            ),
        },
    )


HETERO_FANOUT = {
    ("user", "votes", "post"): [15, 10],
    ("user", "writes", "comment"): [15, 10],
    ("comment", "reply_to", "comment"): [5, 5],
    ("post", "belongs_to", "subreddit"): [3, 3],
}


def bench_hetero_ag(scale: dict[str, int], batch_size: int) -> BenchResult:
    """AetherGraph heterogeneous: sample only."""
    from aethergraph._core import HeteroCsrGraph, HeteroNeighborSampler, HeteroSamplingConfig

    u, p, c, s, edges = _make_hetero_edges(scale)
    graph = HeteroCsrGraph.from_edge_arrays(
        node_types={"user": u, "post": p, "comment": c, "subreddit": s},
        edge_types=[
            (st, r, dt, es.astype(np.uint32), ed.astype(np.uint32))
            for (st, r, dt), (es, ed) in edges.items()
        ],
    )
    config = HeteroSamplingConfig(num_neighbors=HETERO_FANOUT)
    sampler = HeteroNeighborSampler(graph, config)

    rng = np.random.default_rng(99)
    iters = scale["iters"]
    seeds = [rng.integers(0, u, batch_size).astype(np.uint32) for _ in range(iters + 15)]
    idx = [0]

    def run() -> None:
        sampler.sample("user", seeds[idx[0]])
        idx[0] += 1

    elapsed = _timed_iters(run, iters)
    return BenchResult(f"hetero_2hop_b{batch_size}", "ag", elapsed / iters * 1e6)


def bench_hetero_pyg(scale: dict[str, int], batch_size: int) -> BenchResult:
    """PyG heterogeneous: sample only (low-level NeighborSampler)."""
    import torch
    from torch_geometric.data import HeteroData
    from torch_geometric.sampler import NeighborSampler, NodeSamplerInput, NumNeighbors

    u, p, c, s, edges = _make_hetero_edges(scale)

    data = HeteroData()
    data["user"].num_nodes = u
    data["post"].num_nodes = p
    data["comment"].num_nodes = c
    data["subreddit"].num_nodes = s

    for (st, r, dt), (es, ed) in edges.items():
        data[st, r, dt].edge_index = torch.stack([torch.from_numpy(es), torch.from_numpy(ed)])

    sampler = NeighborSampler(data, num_neighbors=NumNeighbors(HETERO_FANOUT))

    rng = np.random.default_rng(99)
    iters = scale["iters"]
    seeds = [
        torch.from_numpy(rng.integers(0, u, batch_size).astype(np.int64)) for _ in range(iters + 15)
    ]
    idx = [0]

    # Verify PyG is actually sampling (not returning empty)
    test_inp = NodeSamplerInput(input_id=torch.arange(batch_size), node=seeds[0], input_type="user")
    test_out = sampler.sample_from_nodes(test_inp)
    total_nodes = (
        sum(v.numel() for v in test_out.node.values())
        if isinstance(test_out.node, dict)
        else test_out.node.numel()
    )
    if total_nodes <= batch_size:
        raise RuntimeError(
            f"PyG hetero sampler returned only {total_nodes} nodes for {batch_size} seeds — "
            "sampling backend may not be working. Check pyg-lib installation."
        )

    def run() -> None:
        inp = NodeSamplerInput(
            input_id=torch.arange(batch_size), node=seeds[idx[0]], input_type="user"
        )
        sampler.sample_from_nodes(inp)
        idx[0] += 1

    elapsed = _timed_iters(run, iters)
    return BenchResult(f"hetero_2hop_b{batch_size}", "pyg", elapsed / iters * 1e6)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _display_results(results: list[BenchResult], title: str) -> None:
    """Render a rich table pairing AG and PyG results."""
    table = Table(title=title)
    table.add_column("Benchmark", style="cyan")
    table.add_column("AetherGraph (us)", justify="right", style="green")
    table.add_column("PyG (us)", justify="right", style="yellow")
    table.add_column("Speedup", justify="right", style="bold")

    seen = []
    for r in results:
        if r.name not in seen:
            seen.append(r.name)

    for name in seen:
        ag = next((x for x in results if x.name == name and x.framework == "ag"), None)
        pyg = next((x for x in results if x.name == name and x.framework == "pyg"), None)
        ag_str = f"{ag.us_per_iter:.0f}" if ag else "—"
        pyg_str = f"{pyg.us_per_iter:.0f}" if pyg else "ERROR"
        if ag and pyg:
            speedup = pyg.us_per_iter / ag.us_per_iter
            color = "green" if speedup > 1 else "red"
            speedup_str = f"[{color}]{speedup:.1f}x[/{color}]"
        else:
            speedup_str = "—"
        table.add_row(name, ag_str, pyg_str, speedup_str)

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@app.command()
def run(
    scale: Scale = typer.Option(Scale.small, help="Graph scale"),
) -> None:
    """Run head-to-head AetherGraph vs PyG sampling benchmarks.

    Builds identical graphs for both frameworks and measures raw sampling
    throughput. Both frameworks do the same work: k-hop neighborhood
    sampling from seed nodes. No edge remapping or Data construction
    is included — this measures the pure sampling kernel.
    """
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        import torch_geometric.typing as tgt
    except ImportError:
        console.print("[red]torch and torch_geometric required[/red]")
        raise typer.Exit(1)

    console.print(f"PyG {torch_geometric.__version__} | torch {torch.__version__}")
    console.print(f"pyg-lib: {tgt.WITH_PYG_LIB} | torch-sparse: {tgt.WITH_TORCH_SPARSE}")
    console.print(f"Scale: {scale.value}\n")

    cfg = SCALES[scale]
    results: list[BenchResult] = []

    console.print("[bold]Homogeneous sampling (raw kernel)[/bold]")
    for fanout, bs in [([15], 128), ([15, 10], 128), ([15, 10], 512)]:
        label = f"{len(fanout)}-hop b={bs}"
        console.print(f"  {label}...", end=" ")
        ag = bench_homo_ag(cfg, fanout, bs)
        try:
            pyg = bench_homo_pyg(cfg, fanout, bs)
            sp = pyg.us_per_iter / ag.us_per_iter
            console.print(f"AG={ag.us_per_iter:.0f}us  PyG={pyg.us_per_iter:.0f}us  {sp:.1f}x")
            results.extend([ag, pyg])
        except Exception as e:
            console.print(f"AG={ag.us_per_iter:.0f}us  PyG=ERROR ({e})")
            results.append(ag)

    console.print("\n[bold]Weighted sampling (Efraimidis-Spirakis vs PyG)[/bold]")
    for fanout, bs in [([15, 10], 128)]:
        label = f"{len(fanout)}-hop b={bs}"
        console.print(f"  {label}...", end=" ")
        ag = bench_weighted_ag(cfg, fanout, bs)
        try:
            pyg = bench_weighted_pyg(cfg, fanout, bs)
            sp = pyg.us_per_iter / ag.us_per_iter
            console.print(f"AG={ag.us_per_iter:.0f}us  PyG={pyg.us_per_iter:.0f}us  {sp:.1f}x")
            results.extend([ag, pyg])
        except Exception as e:
            console.print(f"AG={ag.us_per_iter:.0f}us  PyG=ERROR ({e})")
            results.append(ag)

    console.print("\n[bold]Temporal sampling (uniform + last vs PyG)[/bold]")
    for strategy in ["uniform", "last"]:
        for fanout, bs in [([15, 10], 128)]:
            label = f"{strategy} {len(fanout)}-hop b={bs}"
            console.print(f"  {label}...", end=" ")
            ag = bench_temporal_ag(cfg, fanout, bs, strategy)
            try:
                pyg = bench_temporal_pyg(cfg, fanout, bs, strategy)
                sp = pyg.us_per_iter / ag.us_per_iter
                console.print(f"AG={ag.us_per_iter:.0f}us  PyG={pyg.us_per_iter:.0f}us  {sp:.1f}x")
                results.extend([ag, pyg])
            except Exception as e:
                console.print(f"AG={ag.us_per_iter:.0f}us  PyG=ERROR ({e})")
                results.append(ag)

    console.print("\n[bold]Disjoint sampling (per-seed isolation vs PyG)[/bold]")
    for fanout, bs in [([15, 10], 128)]:
        label = f"{len(fanout)}-hop b={bs}"
        console.print(f"  {label}...", end=" ")
        ag = bench_disjoint_ag(cfg, fanout, bs)
        try:
            pyg = bench_disjoint_pyg(cfg, fanout, bs)
            sp = pyg.us_per_iter / ag.us_per_iter
            console.print(f"AG={ag.us_per_iter:.0f}us  PyG={pyg.us_per_iter:.0f}us  {sp:.1f}x")
            results.extend([ag, pyg])
        except Exception as e:
            console.print(f"AG={ag.us_per_iter:.0f}us  PyG=ERROR ({e})")
            results.append(ag)

    console.print("\n[bold]Heterogeneous sampling (raw kernel, 4 edge types)[/bold]")
    for bs in [128, 512]:
        label = f"2-hop b={bs}"
        console.print(f"  {label}...", end=" ")
        ag = bench_hetero_ag(cfg, bs)
        try:
            pyg = bench_hetero_pyg(cfg, bs)
            sp = pyg.us_per_iter / ag.us_per_iter
            console.print(f"AG={ag.us_per_iter:.0f}us  PyG={pyg.us_per_iter:.0f}us  {sp:.1f}x")
            results.extend([ag, pyg])
        except Exception as e:
            console.print(f"AG={ag.us_per_iter:.0f}us  PyG=ERROR ({e})")
            results.append(ag)

    _display_results(results, f"AetherGraph vs PyG ({scale.value})")


if __name__ == "__main__":
    app()
