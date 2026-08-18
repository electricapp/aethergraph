"""Tests for the stream-pipelined device transfer.

The overlap itself is CUDA-only, but every ordering and correctness
property is exercised on CPU: the pipeline must preserve batch order and
count, refill to its configured depth, and pass batches through unchanged
when no CUDA device is present.
"""

from __future__ import annotations

import pytest

# Bound through importorskip rather than imported: torch and PyG are optional
# extras, and a module-level import of them fails collection for the whole
# suite in environments that install only the dev group.
torch = pytest.importorskip("torch")
Data = pytest.importorskip("torch_geometric.data").Data
_device_pipeline = pytest.importorskip("aethergraph.pytorch.device_pipeline")

DeviceTransferPipeline = _device_pipeline.DeviceTransferPipeline
move_data_to_device = _device_pipeline.move_data_to_device


def _batch(seed: int) -> Data:
    """A small PyG batch tagged so order is checkable."""
    return Data(
        x=torch.full((4, 3), float(seed)),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        n_id=torch.tensor([seed, seed + 1, seed + 2, seed + 3], dtype=torch.long),
        batch_size=1,
        num_nodes=4,
    )


def test_cpu_passthrough_preserves_order_and_count() -> None:
    src = [_batch(i * 10) for i in range(5)]
    out = list(DeviceTransferPipeline(src, "cpu", depth=2))
    assert len(out) == 5
    for i, data in enumerate(out):
        assert int(data.x[0, 0].item()) == i * 10
        # CPU passthrough must not have moved anything.
        assert data.x.device.type == "cpu"


def test_passthrough_is_the_same_objects_on_cpu() -> None:
    # With no CUDA device the pipeline yields the source batches verbatim.
    src = [_batch(i) for i in range(3)]
    out = list(DeviceTransferPipeline(src, "cpu"))
    assert [id(d) for d in out] == [id(d) for d in src]


def test_empty_source() -> None:
    assert list(DeviceTransferPipeline([], "cpu")) == []


def test_depth_validation() -> None:
    with pytest.raises(ValueError, match="depth"):
        DeviceTransferPipeline([_batch(0)], "cpu", depth=0)


def test_move_data_to_device_cpu_noop() -> None:
    # Moving a CPU batch to "cpu" leaves tensors on the host and metadata
    # intact; exercises the PyG `.to` delegation on the CPU path.
    data = _batch(7)
    moved = move_data_to_device(data, torch.device("cpu"), non_blocking=False)
    assert moved.x.device.type == "cpu"
    assert moved.n_id.device.type == "cpu"
    assert int(moved.n_id[0].item()) == 7
    # Non-tensor / missing attributes are left untouched.
    assert moved.batch_size == 1
    assert moved.num_nodes == 4


def test_depth_one_still_yields_all() -> None:
    src = [_batch(i) for i in range(4)]
    out = list(DeviceTransferPipeline(src, "cpu", depth=1))
    assert [int(d.x[0, 0].item()) for d in out] == [0, 1, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_moves_and_preserves_order() -> None:
    src = [_batch(i * 100) for i in range(6)]
    out = list(DeviceTransferPipeline(src, "cuda", depth=2))
    assert len(out) == 6
    for i, data in enumerate(out):
        assert data.x.is_cuda
        assert data.n_id.is_cuda
        # Order preserved and values intact after the async copy.
        torch.cuda.synchronize()
        assert int(data.x[0, 0].item()) == i * 100
