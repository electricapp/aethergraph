"""AetherGraph CLI - Graph conversion and inspection tools.

This module provides command-line tools for converting edge lists to binary
format and inspecting graph files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from aethergraph import Graph

app = typer.Typer(
    name="aethergraph",
    help="High-performance graph sampling for GNN training",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()
error_console = Console(stderr=True)

SPLASH = r"""
█████╗ ███████╗████████╗██╗  ██╗███████╗██████╗  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗
██╔══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║
███████║█████╗     ██║   ███████║█████╗  ██████╔╝██║  ███╗██████╔╝███████║██████╔╝███████║
██╔══██║██╔══╝     ██║   ██╔══██║██╔══╝  ██╔══██╗██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║
██║  ██║███████╗   ██║   ██║  ██║███████╗██║  ██║╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║
╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝

   High-Performance Async Neighborhood Sampling for Billions-Scale GNNs
"""


class Logger:
    """Simple logger with verbosity levels.

    Attributes:
        verbose: Verbosity level (0=info only, 1=debug, 2=trace).
        quiet: If True, suppress all non-error output.
    """

    verbose: int
    quiet: bool

    def __init__(self, verbose: int = 0, quiet: bool = False) -> None:
        """Initialize the logger.

        Args:
            verbose: Verbosity level (0=info only, 1=debug, 2=trace).
            quiet: If True, suppress all non-error output.
        """
        self.verbose = verbose
        self.quiet = quiet

    def info(self, msg: str) -> None:
        """Log an info message.

        Args:
            msg: Message to log.
        """
        if not self.quiet:
            console.print(msg)

    def debug(self, msg: str) -> None:
        """Log a debug message (requires -v).

        Args:
            msg: Message to log.
        """
        if self.verbose >= 1 and not self.quiet:
            console.print(msg)

    def trace(self, msg: str) -> None:
        """Log a trace message (requires -vv).

        Args:
            msg: Message to log.
        """
        if self.verbose >= 2 and not self.quiet:
            console.print(msg)

    def warn(self, msg: str) -> None:
        """Log a warning message.

        Args:
            msg: Message to log.
        """
        if not self.quiet:
            console.print(f"[yellow]WARN:[/yellow] {msg}")


def _print_error(msg: str) -> None:
    """Print an error message to stderr.

    Args:
        msg: Error message to print.
    """
    error_console.print(f"[red]Error:[/red] {msg}")


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Verbose output (-v, -vv, -vvv)"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-error output"),
) -> None:
    """High-performance graph sampling for GNN training."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["log"] = Logger(verbose, quiet)

    if not quiet and ctx.invoked_subcommand is not None:
        console.print(SPLASH)


@app.command()
def convert(
    ctx: typer.Context,
    input_file: Path = typer.Option(..., "--input", "-i", help="Input edge list file (TSV/CSV)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output binary graph file"),
    num_nodes: int = typer.Option(..., "--num-nodes", "-n", help="Number of nodes in the graph"),
    delimiter: str | None = typer.Option(
        None, "--delimiter", "-d", help="Delimiter (auto-detect if not set)"
    ),
    skip_lines: int = typer.Option(0, "--skip-lines", help="Skip first N lines (for headers)"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite the output file if it already exists"
    ),
) -> None:
    """Convert an edge list file to AetherGraph binary format.

    Reads a text file containing edges (one per line) and converts it to
    AetherGraph's efficient binary CSR format. Supports TSV, CSV, and
    space-delimited formats with auto-detection.
    """
    log: Logger = ctx.obj["log"]
    quiet: bool = ctx.obj["quiet"]

    resolved_input = input_file.resolve()
    resolved_output = output.resolve()

    if not resolved_input.exists():
        _print_error(f"File not found: {resolved_input}")
        raise typer.Exit(1)

    if resolved_output.exists() and not force:
        _print_error(f"Output file already exists: {resolved_output} (use --force to overwrite)")
        raise typer.Exit(1)

    log.info("Converting edge list to AetherGraph format")
    log.debug(f"Input: {resolved_input}")
    log.debug(f"Output: {resolved_output}")
    log.debug(f"Nodes: {num_nodes}")

    edges: list[tuple[int, int]] = []
    errors: list[str] = []
    max_errors = 10  # Collect up to this many errors before stopping

    # Delimiter is detected once from the first data line and reused for the
    # whole file, so a stray tab/comma in one row can't switch the parser
    # mid-stream.
    delim: str | None = delimiter

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        disable=quiet,
    ) as progress:
        task = progress.add_task("Reading edges...", total=None)

        with open(resolved_input) as f:
            for i, line in enumerate(f):
                if i < skip_lines:
                    continue
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if delim is None:
                    delim = _detect_delimiter(line, delimiter)

                parts = line.split(delim)
                if len(parts) >= 2:
                    try:
                        src, dst = int(parts[0]), int(parts[1])
                    except ValueError:
                        errors.append(
                            f"bad token at line {i + 1}: could not parse "
                            f"'{parts[0]}' / '{parts[1]}' as integers"
                        )
                        if len(errors) >= max_errors:
                            break
                        continue

                    error = _validate_edge(src, dst, num_nodes, i + 1)
                    if error:
                        errors.append(error)
                        if len(errors) >= max_errors:
                            break
                    else:
                        edges.append((src, dst))

                if len(edges) % 100_000 == 0:
                    progress.update(task, description=f"Read {len(edges):,} edges")
                    log.trace(f"Read {len(edges):,} edges")

    # Report all collected errors
    if errors:
        for error in errors:
            _print_error(error)
        if len(errors) >= max_errors:
            _print_error(f"... stopping after {max_errors} errors")
        raise typer.Exit(1)

    log.info(f"Read {len(edges):,} edges from input file")

    log.debug("Building CSR graph structure")
    src_arr: npt.NDArray[np.uint32] = np.array([e[0] for e in edges], dtype=np.uint32)
    dst_arr: npt.NDArray[np.uint32] = np.array([e[1] for e in edges], dtype=np.uint32)
    graph = Graph.from_edges(num_nodes, src_arr, dst_arr)

    log.debug("Writing binary file")
    graph.save(resolved_output)

    log.info("Conversion complete")
    log.info(f"  Nodes: {graph.num_nodes:,}")
    log.info(f"  Edges: {graph.num_edges:,}")
    if graph.num_nodes > 0:
        log.info(f"  Avg degree: {graph.num_edges / graph.num_nodes:.2f}")

    file_size = resolved_output.stat().st_size
    log.info(f"  File size: {file_size / 1_000_000:.2f} MB")


def _detect_delimiter(line: str, explicit_delimiter: str | None) -> str:
    """Detect the delimiter used in a line.

    If an explicit delimiter is provided, uses that. Otherwise, auto-detects
    based on presence of tab, comma, or defaults to space.

    Args:
        line: The line to analyze.
        explicit_delimiter: User-specified delimiter, or None for auto-detect.

    Returns:
        The delimiter string to use for splitting.
    """
    if explicit_delimiter is not None:
        return explicit_delimiter
    if "\t" in line:
        return "\t"
    if "," in line:
        return ","
    return " "


def _validate_edge(src: int, dst: int, num_nodes: int, line_num: int) -> str | None:
    """Validate that an edge has valid node IDs.

    Args:
        src: Source node ID.
        dst: Destination node ID.
        num_nodes: Total number of nodes in the graph.
        line_num: Line number in the input file (for error messages).

    Returns:
        Error message string if validation fails, None if valid.
    """
    if src < 0:
        return f"negative source node {src} at line {line_num}"
    if dst < 0:
        return f"negative dest node {dst} at line {line_num}"
    if src >= num_nodes:
        return f"source node {src} exceeds num_nodes {num_nodes} at line {line_num}"
    if dst >= num_nodes:
        return f"dest node {dst} exceeds num_nodes {num_nodes} at line {line_num}"
    return None


@app.command()
def info(
    ctx: typer.Context,
    path: Path = typer.Argument(..., help="Binary graph file"),
) -> None:
    """Display information about a binary graph file.

    Shows basic statistics including node count, edge count, average degree,
    and file size.
    """
    log: Logger = ctx.obj["log"]

    resolved_path = path.resolve()
    if not resolved_path.exists():
        _print_error(f"File not found: {resolved_path}")
        raise typer.Exit(1)

    log.debug(f"Loading graph from {resolved_path}")
    graph = Graph.load(str(resolved_path))

    log.info("Graph Information:")
    log.info(f"  Nodes: {graph.num_nodes:,}")
    log.info(f"  Edges: {graph.num_edges:,}")
    if graph.num_nodes > 0:
        log.info(f"  Avg degree: {graph.num_edges / graph.num_nodes:.2f}")
    log.info(f"  Has weights: {graph.has_weights}")

    file_size = resolved_path.stat().st_size
    log.info(f"  File size: {file_size / 1_000_000:.2f} MB")


@app.command()
def stats(
    ctx: typer.Context,
    path: Path = typer.Argument(..., help="Binary graph file"),
) -> None:
    """Display detailed statistics about a binary graph file.

    Shows comprehensive statistics including degree distribution percentiles,
    bytes per node/edge, and isolated node warnings.
    """
    log: Logger = ctx.obj["log"]
    quiet: bool = ctx.obj["quiet"]

    resolved_path = path.resolve()
    if not resolved_path.exists():
        _print_error(f"File not found: {resolved_path}")
        raise typer.Exit(1)

    log.debug(f"Loading graph from {resolved_path}")
    graph = Graph.load(str(resolved_path))

    # Headline aggregates come from the Rust `stats()` call, which computes
    # max/avg degree in parallel over the CSR offsets without crossing the
    # FFI boundary per node.
    summary = graph.stats()
    num_nodes = int(summary["num_nodes"])
    num_edges = int(summary["num_edges"])
    max_degree = int(summary["max_degree"])
    avg_degree = float(summary["avg_degree"])

    log.info("Graph Statistics:")
    log.info(f"  Nodes: {num_nodes:,}")
    log.info(f"  Edges: {num_edges:,}")

    log.info(f"  Max degree: {max_degree:,}")
    log.info(f"  Avg degree: {avg_degree:.2f}")
    log.info(f"  Has weights: {bool(summary['has_weights'])}")

    # Percentiles and the isolated-node count need the full per-node degree
    # distribution, which `_core` does not expose as a bulk offsets/degrees
    # array. Build it with a per-node loop; for very large graphs this is the
    # slow part and would benefit from a degrees accessor on CsrGraph.
    log.debug("Analyzing degree distribution")

    degrees: list[int] = []

    with Progress(
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} nodes"),
        transient=True,
        disable=quiet,
    ) as progress:
        task = progress.add_task("Analyzing", total=num_nodes)

        for node in range(num_nodes):
            degrees.append(graph.degree(node))
            if node % 10000 == 0:
                progress.update(task, completed=node)
        progress.update(task, completed=num_nodes)

    degrees_arr: npt.NDArray[np.int64] = np.array(degrees, dtype=np.int64)

    file_size = resolved_path.stat().st_size
    log.info("")
    log.info("File Information:")
    log.info(f"  Size: {file_size / 1_000_000:.2f} MB")
    if num_nodes > 0:
        log.info(f"  Bytes per node: {file_size / num_nodes:.2f}")
    if num_edges > 0:
        log.info(f"  Bytes per edge: {file_size / num_edges:.2f}")

    log.info("")
    log.info("Degree Distribution:")
    p50 = int(np.percentile(degrees_arr, 50))
    p90 = int(np.percentile(degrees_arr, 90))
    p99 = int(np.percentile(degrees_arr, 99))
    log.info(f"  50th percentile: {p50}")
    log.info(f"  90th percentile: {p90}")
    log.info(f"  99th percentile: {p99}")

    isolated = int(np.sum(degrees_arr == 0))
    if isolated > 0 and num_nodes > 0:
        pct = 100.0 * isolated / num_nodes
        log.warn(f"Isolated nodes (degree 0): {isolated:,} ({pct:.2f}%)")


def main() -> None:
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
