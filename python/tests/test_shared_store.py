"""Cross-process feature sharing over a sealed memfd.

Linux only: the module omits `SharedFeatureStore` elsewhere, which
`HAS_SHARED_STORE` reports.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from aethergraph._core import HAS_SHARED_STORE, save_features

pytestmark = pytest.mark.skipif(not HAS_SHARED_STORE, reason="shared feature store is Linux-only")

if HAS_SHARED_STORE:
    from aethergraph._core import SharedFeatureStore


def _write_features(path: Path, num_nodes: int, dim: int) -> np.ndarray:
    features = (np.arange(num_nodes * dim, dtype=np.float32) * 0.5).reshape(num_nodes, dim)
    save_features(path, features)
    return features


class TestSharedFeatureStore:
    def test_publish_and_read(self, temp_dir: Path) -> None:
        path = temp_dir / "features.bin"
        features = _write_features(path, 500, 16)

        store = SharedFeatureStore.publish(path)
        assert store.num_nodes == 500
        assert store.feature_dim == 16
        assert store.shared_bytes == 500 * 16 * 4
        assert not store.is_serving

        nodes = np.array([0, 17, 499, 17], dtype=np.int64)
        np.testing.assert_allclose(store.get_batch(nodes), features[nodes])

    def test_attach_in_same_process(self, temp_dir: Path) -> None:
        path = temp_dir / "features.bin"
        features = _write_features(path, 300, 8)
        socket = temp_dir / "store.sock"

        owner = SharedFeatureStore.publish(path)
        owner.serve(socket)
        assert owner.is_serving

        peer = SharedFeatureStore.attach(socket)
        assert peer.num_nodes == owner.num_nodes
        assert peer.shared_bytes == owner.shared_bytes

        nodes = np.array([1, 100, 299], dtype=np.int64)
        np.testing.assert_allclose(peer.get_batch(nodes), features[nodes])

        owner.stop_serving()
        assert not owner.is_serving

    def test_second_process_maps_the_same_pages(self, temp_dir: Path) -> None:
        """The actual claim: another OS process reads the owner's memory."""
        path = temp_dir / "features.bin"
        features = _write_features(path, 1000, 8)
        socket = temp_dir / "store.sock"

        owner = SharedFeatureStore.publish(path)
        owner.serve(socket)

        worker = textwrap.dedent(f"""
            import numpy as np
            from aethergraph._core import SharedFeatureStore

            store = SharedFeatureStore.attach({str(socket)!r})
            nodes = np.array([3, 111, 999], dtype=np.int64)
            rows = store.get_batch(nodes)
            print(store.num_nodes, store.feature_dim, store.shared_bytes)
            print(",".join(f"{{v:.4f}}" for v in rows.ravel()))
        """)
        result = subprocess.run(
            [sys.executable, "-c", worker],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, f"worker failed: {result.stderr}"

        geometry, values = result.stdout.strip().splitlines()
        num_nodes, feature_dim, shared_bytes = (int(v) for v in geometry.split())
        assert (num_nodes, feature_dim) == (1000, 8)
        assert shared_bytes == owner.shared_bytes

        got = np.array([float(v) for v in values.split(",")], dtype=np.float32)
        np.testing.assert_allclose(got, features[[3, 111, 999]].ravel(), rtol=1e-4)

    def test_attach_without_server_raises(self, temp_dir: Path) -> None:
        with pytest.raises(OSError):
            SharedFeatureStore.attach(temp_dir / "nobody-here.sock")

    def test_out_of_range_node_raises(self, temp_dir: Path) -> None:
        path = temp_dir / "features.bin"
        _write_features(path, 50, 4)
        store = SharedFeatureStore.publish(path)

        with pytest.raises(ValueError):
            store.get_batch(np.array([50], dtype=np.int64))
