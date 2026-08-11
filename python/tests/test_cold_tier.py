"""Tests for the compressed backing tier of the feature cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aethergraph._core import FeatureCache, FeatureCacheConfig, save_features


def _write_features(path: Path, num_nodes: int, dim: int) -> np.ndarray:
    features = np.arange(num_nodes * dim, dtype=np.float32).reshape(num_nodes, dim) * 0.25
    save_features(path, features)
    return features


class TestColdStoreConfig:
    def test_defaults_leave_tier_disabled(self, temp_dir: Path) -> None:
        config = FeatureCacheConfig(nvme_path=str(temp_dir / "spill"))
        assert config.cold_store_path is None
        assert config.cold_level == 12

    def test_rejects_out_of_range_level(self, temp_dir: Path) -> None:
        with pytest.raises(ValueError, match="cold_level"):
            FeatureCacheConfig(nvme_path=str(temp_dir / "spill"), cold_level=99)


class TestColdStoreServing:
    @pytest.mark.asyncio
    async def test_serves_nodes_never_inserted(self, temp_dir: Path) -> None:
        """The tier makes the cache a complete source, not a best-effort one."""
        num_nodes, dim = 800, 16
        store = temp_dir / "features.bin"
        features = _write_features(store, num_nodes, dim)

        config = FeatureCacheConfig(
            gpu_capacity=4,
            cpu_capacity=8,
            feature_dim=dim,
            nvme_path=str(temp_dir / "spill"),
            cold_store_path=str(store),
        )
        cache = await FeatureCache.create(config)

        got = await cache.get(700)
        np.testing.assert_allclose(got, features[700])
        assert cache.stats()["cold_hits"] == 1

        # Promoted on the way out, so a re-read never decompresses again.
        np.testing.assert_allclose(await cache.get(700), features[700])
        assert cache.stats()["cold_hits"] == 1

        batch = await cache.get_batch([1, 300, 799, 300])
        for row, node in zip(batch, [1, 300, 799, 300]):
            np.testing.assert_allclose(row, features[node])

    @pytest.mark.asyncio
    async def test_without_tier_a_missing_node_raises(self, temp_dir: Path) -> None:
        config = FeatureCacheConfig(feature_dim=16, nvme_path=str(temp_dir / "spill"))
        cache = await FeatureCache.create(config)

        with pytest.raises(Exception, match="no cache tier"):
            await cache.get(700)

    @pytest.mark.asyncio
    async def test_dim_mismatch_fails_at_create(self, temp_dir: Path) -> None:
        store = temp_dir / "features.bin"
        _write_features(store, 100, 8)

        config = FeatureCacheConfig(
            feature_dim=16,
            nvme_path=str(temp_dir / "spill"),
            cold_store_path=str(store),
        )
        with pytest.raises(Exception, match="feature_dim"):
            await FeatureCache.create(config)
