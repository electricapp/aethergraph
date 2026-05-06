"""Tests for Ray Data integration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from aethergraph import Graph

pytestmark = [pytest.mark.requires_ray, pytest.mark.requires_torch]


class TestAetherGraphDatasource:
    """Test the Ray Data datasource."""

    def test_datasource_creation(self, small_graph: Graph, temp_dir: Path) -> None:
        """Test creating the datasource."""
        from aethergraph.ray import AetherGraphDatasource

        graph_path = temp_dir / "graph.bin"
        small_graph.save(graph_path)

        datasource = AetherGraphDatasource(
            graph_path=str(graph_path),
            num_neighbors=[10, 5],
            batch_size=16,
        )

        assert datasource.graph_path == str(graph_path)

    def test_datasource_with_input_nodes(self, small_graph: Graph, temp_dir: Path) -> None:
        """Test datasource with explicit input_nodes."""
        from aethergraph.ray import AetherGraphDatasource

        graph_path = temp_dir / "graph.bin"
        small_graph.save(graph_path)

        input_nodes = np.arange(50, dtype=np.int64)
        datasource = AetherGraphDatasource(
            graph_path=str(graph_path),
            num_neighbors=[10],
            input_nodes=input_nodes,
            batch_size=16,
        )

        assert datasource._input_nodes is not None and len(datasource._input_nodes) == 50


class TestCreateSamplingDataset:
    """Test the high-level create_sampling_dataset function."""

    def test_create_dataset(self, small_graph: Graph, temp_dir: Path) -> None:
        """Test creating a sampling dataset."""
        import ray

        from aethergraph.ray import create_sampling_dataset

        if not ray.is_initialized():
            ray.init(num_cpus=2, runtime_env={"working_dir": None})

        try:
            graph_path = temp_dir / "graph.bin"
            small_graph.save(graph_path)

            dataset = create_sampling_dataset(
                graph_path=graph_path,
                num_neighbors=[10, 5],
                batch_size=16,
                parallelism=1,
            )

            count = dataset.count()
            assert count > 0

        finally:
            ray.shutdown()

    def test_dataset_with_features(
        self,
        small_graph_with_features: Graph,
        temp_dir: Path,
        rng: np.random.Generator,
    ) -> None:
        """Test dataset with features."""
        import ray

        from aethergraph.ray import create_sampling_dataset

        if not ray.is_initialized():
            ray.init(num_cpus=2, runtime_env={"working_dir": None})

        try:
            graph_path = temp_dir / "graph.bin"
            small_graph_with_features.save(graph_path)

            dataset = create_sampling_dataset(
                graph_path=graph_path,
                num_neighbors=[10],
                batch_size=16,
                parallelism=1,
            )

            count = dataset.count()
            assert count > 0

        finally:
            ray.shutdown()


class TestCollateToPyg:
    """Test the collate_to_pyg function."""

    @pytest.mark.requires_pyg
    def test_collate_basic(self, small_graph: Graph, temp_dir: Path) -> None:
        """Test collating Arrow batch to PyG Data."""
        import ray

        from aethergraph.ray import collate_to_pyg, create_sampling_dataset

        if not ray.is_initialized():
            ray.init(num_cpus=2, runtime_env={"working_dir": None})

        try:
            graph_path = temp_dir / "graph.bin"
            small_graph.save(graph_path)

            dataset = create_sampling_dataset(
                graph_path=graph_path,
                num_neighbors=[10],
                batch_size=16,
                parallelism=1,
            )

            for batch in dataset.iter_batches(batch_size=1):
                pyg_data = collate_to_pyg(batch)

                assert hasattr(pyg_data, "edge_index")
                assert pyg_data.edge_index.shape[0] == 2
                assert hasattr(pyg_data, "n_id")
                break

        finally:
            ray.shutdown()

    @pytest.mark.requires_pyg
    def test_collate_rejects_multi_row_batch(self) -> None:
        """collate_to_pyg should fail fast on multi-row inputs."""
        from aethergraph.ray import collate_to_pyg

        batch = {
            "edge_index_0": [
                np.array([0, 1], dtype=np.int64),
                np.array([5, 6], dtype=np.int64),
            ],
            "edge_index_1": [
                np.array([1, 2], dtype=np.int64),
                np.array([6, 7], dtype=np.int64),
            ],
            "n_id": [
                np.array([0, 1, 2], dtype=np.int64),
                np.array([5, 6, 7], dtype=np.int64),
            ],
            "e_id": [
                np.array([10, 11], dtype=np.int64),
                np.array([20, 21], dtype=np.int64),
            ],
            "batch_size": [3, 3],
        }

        with pytest.raises(ValueError, match="exactly one row"):
            collate_to_pyg(batch)


class TestRayIntegration:
    """Integration tests for full Ray workflow."""

    def test_distributed_sampling_flow(self, medium_graph: Graph, temp_dir: Path) -> None:
        """Test complete distributed sampling flow."""
        import ray

        from aethergraph.ray import create_sampling_dataset

        if not ray.is_initialized():
            ray.init(num_cpus=2, runtime_env={"working_dir": None})

        try:
            graph_path = temp_dir / "graph.bin"
            medium_graph.save(graph_path)

            dataset = create_sampling_dataset(
                graph_path=graph_path,
                num_neighbors=[15, 10],
                input_nodes=np.arange(1000),
                batch_size=128,
                parallelism=2,
            )

            total_batches = 0
            for batch in dataset.iter_batches(batch_size=1):
                total_batches += 1
                if total_batches >= 5:
                    break

            assert total_batches > 0

        finally:
            ray.shutdown()
