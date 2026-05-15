"""Shared fixtures for AetherGraph test suite.

This module configures pytest fixtures and markers for optional dependencies.
Environment variables are set before imports to configure Ray's behavior.
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from aethergraph import Graph


@pytest.fixture
def rng() -> np.random.Generator:
    """Create a seeded random number generator for reproducibility.

    Returns:
        Numpy random generator seeded with 42.
    """
    return np.random.default_rng(42)


@pytest.fixture
def small_graph() -> Graph:
    """Create a small graph fixture for unit tests.

    Creates a random graph with 100 nodes and 500 edges using a fixed
    seed for reproducibility.

    Returns:
        Graph instance with 100 nodes and approximately 500 edges.
    """
    from aethergraph import Graph

    rng = np.random.default_rng(42)
    num_nodes = 100
    num_edges = 500
    src = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    dst = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    return Graph.from_edges(num_nodes, src, dst)


@pytest.fixture
def small_graph_with_features(
    small_graph: Graph, rng: np.random.Generator
) -> tuple[Graph, np.ndarray]:
    """Return `(graph, features)` for loader tests that need both.

    Features are passed to the loader constructor; the graph itself
    carries no feature state.
    """
    features = rng.standard_normal((small_graph.num_nodes, 64)).astype(np.float32)
    return small_graph, features


@pytest.fixture
def medium_graph() -> Graph:
    """Create a medium-sized graph fixture for performance tests.

    Creates a random graph with 10,000 nodes and 50,000 edges using a
    fixed seed for reproducibility.

    Returns:
        Graph instance with 10K nodes and approximately 50K edges.
    """
    from aethergraph import Graph

    rng = np.random.default_rng(42)
    num_nodes = 10_000
    num_edges = 50_000
    src = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    dst = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    return Graph.from_edges(num_nodes, src, dst)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.

    The directory is automatically cleaned up after the test completes.

    Yields:
        Path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def pytest_configure(config: Any) -> None:
    """Register custom pytest markers for optional dependencies.

    Adds markers that can be used to skip tests when optional dependencies
    (torch, ray, torch_geometric) are not installed.

    Args:
        config: Pytest configuration object.
    """
    config.addinivalue_line("markers", "requires_torch: skip test if torch is not installed")
    config.addinivalue_line("markers", "requires_ray: skip test if ray is not installed")
    config.addinivalue_line(
        "markers", "requires_pyg: skip test if torch_geometric is not installed"
    )


def has_torch() -> bool:
    """Check if PyTorch is installed.

    Returns:
        True if torch can be imported, False otherwise.
    """
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def has_ray() -> bool:
    """Check if Ray is installed.

    Returns:
        True if ray can be imported, False otherwise.
    """
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        return False


def has_pyg() -> bool:
    """Check if PyTorch Geometric is installed.

    Returns:
        True if torch_geometric can be imported, False otherwise.
    """
    try:
        import torch_geometric  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(autouse=True)
def skip_by_requirement(request: pytest.FixtureRequest) -> None:
    """Skip tests based on availability of optional dependencies.

    Automatically applied to all tests. Checks for requires_torch,
    requires_ray, and requires_pyg markers and skips the test if
    the corresponding dependency is not installed.

    Args:
        request: Pytest fixture request object containing test metadata.
    """
    if request.node.get_closest_marker("requires_torch") and not has_torch():
        pytest.skip("torch not installed")
    if request.node.get_closest_marker("requires_ray") and not has_ray():
        pytest.skip("ray not installed")
    if request.node.get_closest_marker("requires_pyg") and not has_pyg():
        pytest.skip("torch_geometric not installed")
