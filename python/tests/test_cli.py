"""Tests for the AetherGraph CLI.

This module tests the CLI commands for graph conversion and inspection.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from aethergraph import Graph
from aethergraph.cli import app

runner = CliRunner()


class TestConvertCommand:
    """Tests for the 'convert' command."""

    def test_convert_tsv_file(self, temp_dir: Path) -> None:
        """Convert a TSV edge list to binary format.

        Args:
            temp_dir: Temporary directory fixture for test files.
        """
        input_file = temp_dir / "edges.tsv"
        output_file = temp_dir / "graph.bin"

        input_file.write_text("0\t1\n0\t2\n1\t2\n2\t0\n")

        result = runner.invoke(
            app,
            ["-q", "convert", "-i", str(input_file), "-o", str(output_file), "-n", "3"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output_file.exists()

        graph = Graph.load(output_file)
        assert graph.num_nodes == 3
        assert graph.num_edges == 4

    def test_convert_csv_file(self, temp_dir: Path) -> None:
        """Convert a CSV edge list to binary format.

        Args:
            temp_dir: Temporary directory fixture for test files.
        """
        input_file = temp_dir / "edges.csv"
        output_file = temp_dir / "graph.bin"

        input_file.write_text("0,1\n0,2\n1,2\n")

        result = runner.invoke(
            app,
            ["-q", "convert", "-i", str(input_file), "-o", str(output_file), "-n", "3"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output_file.exists()

        graph = Graph.load(output_file)
        assert graph.num_edges == 3

    def test_convert_space_delimited(self, temp_dir: Path) -> None:
        """Convert a space-delimited edge list to binary format.

        Args:
            temp_dir: Temporary directory fixture for test files.
        """
        input_file = temp_dir / "edges.txt"
        output_file = temp_dir / "graph.bin"

        input_file.write_text("0 1\n1 2\n")

        result = runner.invoke(
            app,
            ["-q", "convert", "-i", str(input_file), "-o", str(output_file), "-n", "3"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        graph = Graph.load(output_file)
        assert graph.num_edges == 2

    def test_convert_with_header_skip(self, temp_dir: Path) -> None:
        """Convert an edge list with header lines.

        Args:
            temp_dir: Temporary directory fixture for test files.
        """
        input_file = temp_dir / "edges.tsv"
        output_file = temp_dir / "graph.bin"

        input_file.write_text("src\tdst\n0\t1\n1\t2\n")

        result = runner.invoke(
            app,
            [
                "-q",
                "convert",
                "-i",
                str(input_file),
                "-o",
                str(output_file),
                "-n",
                "3",
                "--skip-lines",
                "1",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        graph = Graph.load(output_file)
        assert graph.num_edges == 2

    def test_convert_with_comments(self, temp_dir: Path) -> None:
        """Convert an edge list with comment lines.

        Args:
            temp_dir: Temporary directory fixture for test files.
        """
        input_file = temp_dir / "edges.tsv"
        output_file = temp_dir / "graph.bin"

        input_file.write_text("# this is a comment\n0\t1\n# another comment\n1\t2\n")

        result = runner.invoke(
            app,
            ["-q", "convert", "-i", str(input_file), "-o", str(output_file), "-n", "3"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        graph = Graph.load(output_file)
        assert graph.num_edges == 2

    def test_convert_nonexistent_input(self, temp_dir: Path) -> None:
        """Convert with a non-existent input file should fail.

        Args:
            temp_dir: Temporary directory fixture for test files.
        """
        output_file = temp_dir / "graph.bin"

        result = runner.invoke(
            app,
            [
                "-q",
                "convert",
                "-i",
                str(temp_dir / "nonexistent.tsv"),
                "-o",
                str(output_file),
                "-n",
                "3",
            ],
        )

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_convert_invalid_node_id(self, temp_dir: Path) -> None:
        """Convert with out-of-bounds node IDs should fail.

        Args:
            temp_dir: Temporary directory fixture for test files.
        """
        input_file = temp_dir / "edges.tsv"
        output_file = temp_dir / "graph.bin"

        input_file.write_text("0\t1\n0\t10\n")

        result = runner.invoke(
            app,
            ["-q", "convert", "-i", str(input_file), "-o", str(output_file), "-n", "3"],
        )

        assert result.exit_code == 1
        assert "exceeds" in result.output.lower()

    def test_convert_negative_node_id(self, temp_dir: Path) -> None:
        """Convert with negative node IDs should fail.

        Args:
            temp_dir: Temporary directory fixture for test files.
        """
        input_file = temp_dir / "edges.tsv"
        output_file = temp_dir / "graph.bin"

        input_file.write_text("0\t1\n-1\t2\n")

        result = runner.invoke(
            app,
            ["-q", "convert", "-i", str(input_file), "-o", str(output_file), "-n", "3"],
        )

        assert result.exit_code == 1
        assert "negative" in result.output.lower()

    def test_convert_collects_multiple_errors(self, temp_dir: Path) -> None:
        """Convert should report multiple validation errors.

        Args:
            temp_dir: Temporary directory fixture for test files.
        """
        input_file = temp_dir / "edges.tsv"
        output_file = temp_dir / "graph.bin"

        input_file.write_text("0\t10\n1\t20\n2\t30\n")

        result = runner.invoke(
            app,
            ["-q", "convert", "-i", str(input_file), "-o", str(output_file), "-n", "5"],
        )

        assert result.exit_code == 1
        assert result.output.lower().count("exceeds") >= 2


class TestInfoCommand:
    """Tests for the 'info' command."""

    def test_info_basic(self, small_graph: Graph, temp_dir: Path) -> None:
        """Display info for a graph file.

        Args:
            small_graph: Small graph fixture.
            temp_dir: Temporary directory fixture for test files.
        """
        path = temp_dir / "graph.bin"
        small_graph.save(path)

        result = runner.invoke(app, ["info", str(path)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "100" in result.output
        assert "500" in result.output

    def test_info_nonexistent_file(self, temp_dir: Path) -> None:
        """Info on non-existent file should fail.

        Args:
            temp_dir: Temporary directory fixture for test files.
        """
        result = runner.invoke(app, ["info", str(temp_dir / "nonexistent.bin")])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestStatsCommand:
    """Tests for the 'stats' command."""

    def test_stats_basic(self, small_graph: Graph, temp_dir: Path) -> None:
        """Display stats for a graph file.

        Args:
            small_graph: Small graph fixture.
            temp_dir: Temporary directory fixture for test files.
        """
        path = temp_dir / "graph.bin"
        small_graph.save(path)

        result = runner.invoke(app, ["stats", str(path)])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "100" in result.output
        assert "percentile" in result.output.lower()

    def test_stats_nonexistent_file(self, temp_dir: Path) -> None:
        """Stats on non-existent file should fail.

        Args:
            temp_dir: Temporary directory fixture for test files.
        """
        result = runner.invoke(app, ["stats", str(temp_dir / "nonexistent.bin")])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestVerbosity:
    """Tests for verbosity flags."""

    def test_verbose_flag(self, small_graph: Graph, temp_dir: Path) -> None:
        """Verbose flag should produce more output.

        Args:
            small_graph: Small graph fixture.
            temp_dir: Temporary directory fixture for test files.
        """
        path = temp_dir / "graph.bin"
        small_graph.save(path)

        quiet_result = runner.invoke(app, ["-q", "info", str(path)])
        verbose_result = runner.invoke(app, ["-v", "info", str(path)])

        assert len(verbose_result.output) > len(quiet_result.output)

    def test_quiet_flag(self, small_graph: Graph, temp_dir: Path) -> None:
        """Quiet flag should suppress output.

        Args:
            small_graph: Small graph fixture.
            temp_dir: Temporary directory fixture for test files.
        """
        path = temp_dir / "graph.bin"
        small_graph.save(path)

        result = runner.invoke(app, ["-q", "info", str(path)])

        assert result.exit_code == 0
        assert "AETHERGRAPH" not in result.output
