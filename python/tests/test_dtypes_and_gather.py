"""Half-width feature dtypes and the batched degree gather."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aethergraph import Graph
from aethergraph._core import FeatureStore, save_features


class TestFeatureDtypes:
    @pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
    def test_round_trip(self, dtype: str, temp_dir: Path) -> None:
        features = np.linspace(-8.0, 8.0, 64 * 16, dtype=np.float32).reshape(64, 16)
        path = temp_dir / f"features-{dtype}.bin"
        save_features(path, features, dtype)

        store = FeatureStore.load(path)
        assert store.num_nodes == 64
        assert store.feature_dim == 16

        got = store.get_batch(np.array([0, 31, 63], dtype=np.int64))
        # f32 is exact; the half-width formats keep ~3 decimal digits.
        tol = 0.0 if dtype == "f32" else 0.1
        np.testing.assert_allclose(got, features[[0, 31, 63]], atol=tol)

    def test_half_width_files_are_smaller(self, temp_dir: Path) -> None:
        features = np.zeros((1000, 32), dtype=np.float32)
        sizes = {}
        for dtype in ("f32", "f16", "bf16"):
            path = temp_dir / f"features-{dtype}.bin"
            save_features(path, features, dtype)
            sizes[dtype] = path.stat().st_size

        assert sizes["f16"] == sizes["bf16"]
        assert sizes["f16"] < sizes["f32"]

    def test_bf16_keeps_f32_exponent_range(self, temp_dir: Path) -> None:
        """The reason to pick bf16 over f16: no flush-to-zero, no saturation."""
        features = np.array([[1e-30, 1e30, -1e-30, -1e30]], dtype=np.float32)

        bf16_path = temp_dir / "bf16.bin"
        save_features(bf16_path, features, "bf16")
        bf16_row = FeatureStore.load(bf16_path).get_batch(np.array([0], dtype=np.int64))
        assert np.all(np.isfinite(bf16_row)), f"bf16 lost range: {bf16_row}"
        np.testing.assert_allclose(bf16_row, features, rtol=0.01)

        f16_path = temp_dir / "f16.bin"
        save_features(f16_path, features, "f16")
        f16_row = FeatureStore.load(f16_path).get_batch(np.array([0], dtype=np.int64))
        assert f16_row[0, 0] == 0.0, "f16 should flush 1e-30 to zero"
        assert np.isinf(f16_row[0, 1]), "f16 should saturate 1e30"

    def test_unknown_dtype_raises(self, temp_dir: Path) -> None:
        features = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="unknown dtype"):
            save_features(temp_dir / "bad.bin", features, "float8")


class TestDegreesOf:
    def test_matches_full_degrees(self) -> None:
        rng = np.random.default_rng(0)
        num_nodes = 2000
        src = rng.integers(0, num_nodes, size=20_000, dtype=np.uint32)
        dst = rng.integers(0, num_nodes, size=20_000, dtype=np.uint32)
        graph = Graph.from_edges(num_nodes, src, dst)

        all_degrees = graph.degrees()
        # A scattered batch, plus lengths that exercise the vector tail.
        for count in (0, 1, 3, 4, 5, 17, 512):
            nodes = rng.integers(0, num_nodes, size=count, dtype=np.uint32)
            got = graph.degrees_of(nodes)
            assert got.dtype == np.uint32
            np.testing.assert_array_equal(got, all_degrees[nodes])

    def test_out_of_range_is_zero(self) -> None:
        src = np.array([0, 1, 2], dtype=np.uint32)
        dst = np.array([1, 2, 0], dtype=np.uint32)
        graph = Graph.from_edges(3, src, dst)

        nodes = np.array([0, 3, 99, 2], dtype=np.uint32)
        got = graph.degrees_of(nodes)
        assert got[1] == 0 and got[2] == 0
        assert got[0] == graph.degree(0)
        assert got[3] == graph.degree(2)
