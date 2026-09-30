"""Tests for the tiered feature cache and the async feature store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aethergraph._core import (
    AsyncFeatureStore,
    CacheError,
    FeatureCache,
    FeatureCacheConfig,
    save_features,
)


def _config(spill: Path, dim: int = 4) -> FeatureCacheConfig:
    # Tiny memory tiers push most rows down to the spill file.
    return FeatureCacheConfig(gpu_capacity=1, cpu_capacity=1, feature_dim=dim, nvme_path=spill)


class TestFeatureCache:
    def test_accepts_pathlike_paths(self, temp_dir: Path) -> None:
        config = FeatureCacheConfig(nvme_path=temp_dir / "spill", cold_store_path=temp_dir / "cold")
        assert config.nvme_path == str(temp_dir / "spill")
        assert config.cold_store_path == str(temp_dir / "cold")

    @pytest.mark.asyncio
    async def test_rejects_rows_of_the_wrong_width(self, temp_dir: Path) -> None:
        cache = await FeatureCache.create(_config(temp_dir / "spill"))
        for node in range(4):
            await cache.insert(node, np.full(4, node, dtype=np.float32))
        with pytest.raises(CacheError, match="feature_dim"):
            await cache.insert(1, np.zeros(8, dtype=np.float32))
        with pytest.raises(CacheError, match="feature_dim"):
            await cache.insert(2, np.zeros(2, dtype=np.float32))
        for node in range(4):
            np.testing.assert_array_equal(await cache.get(node), np.full(4, node))

    @pytest.mark.asyncio
    async def test_caches_sharing_a_spill_directory_stay_independent(self, temp_dir: Path) -> None:
        spill = temp_dir / "shared-spill"
        a = await FeatureCache.create(_config(spill))
        for node in range(6):
            await a.insert(node, np.full(4, node, dtype=np.float32))
        b = await FeatureCache.create(_config(spill))
        for node in range(6):
            await b.insert(node, np.full(4, 100 + node, dtype=np.float32))

        rows_a = await a.get_batch(list(range(6)))
        rows_b = await b.get_batch(list(range(6)))
        np.testing.assert_array_equal(rows_a[:, 0], np.arange(6))
        np.testing.assert_array_equal(rows_b[:, 0], 100 + np.arange(6))


class TestAsyncFeatureStore:
    @pytest.mark.asyncio
    async def test_round_trips_through_every_read_path(self, temp_dir: Path) -> None:
        # 128 lanes: 512-byte rows, the layout the O_DIRECT path serves.
        features = np.arange(300 * 128, dtype=np.float32).reshape(300, 128) * 0.5
        path = temp_dir / "features.bin"
        save_features(path, features)

        store = await AsyncFeatureStore.load(path)
        assert (store.num_nodes, store.feature_dim) == (300, 128)
        np.testing.assert_array_equal(await store.get(17), features[17])
        nodes = [299, 0, 17, 17, 150]
        np.testing.assert_array_equal(await store.get_batch(nodes), features[nodes])
