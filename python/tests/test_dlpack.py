"""DLPack capsule round-trip into PyTorch.

Builds a DLPack capsule from an existing CUDA tensor's `data_ptr` and asserts
that `torch.from_dlpack` produces a view with the correct device/dtype/shape
and the same underlying memory. Skipped automatically when:
    - PyTorch is not installed (`requires_torch` marker),
    - CUDA is not available at runtime,
    - the extension was built without `--features gpudirect` (HAS_GPUDIRECT False).
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.requires_torch]


def _skip_if_no_cuda_or_gpudirect() -> None:
    """Skip the test if CUDA or the gpudirect build feature is missing."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")

    from aethergraph._core import HAS_GPUDIRECT

    if not HAS_GPUDIRECT:
        pytest.skip("extension built without --features gpudirect")


def test_dlpack_round_trip_matches_source_tensor() -> None:
    """Capsule built from a CUDA tensor's data_ptr round-trips byte-for-byte."""
    _skip_if_no_cuda_or_gpudirect()

    import torch

    from aethergraph._core import _dlpack_capsule_from_cuda_ptr

    num_nodes = 7
    feature_dim = 16

    src = (
        torch.arange(num_nodes * feature_dim, dtype=torch.float32, device="cuda:0")
        .reshape(num_nodes, feature_dim)
        .contiguous()
    )

    capsule = _dlpack_capsule_from_cuda_ptr(src.data_ptr(), num_nodes, feature_dim, 0)
    dst = torch.from_dlpack(capsule)  # type: ignore[attr-defined]

    assert dst.device.type == "cuda", f"expected CUDA tensor, got {dst.device}"
    assert dst.device.index == 0
    assert dst.dtype == torch.float32
    assert tuple(dst.shape) == (num_nodes, feature_dim)
    assert dst.data_ptr() == src.data_ptr(), (
        "DLPack view does not share storage with the source tensor"
    )
    assert torch.equal(src, dst), "DLPack view does not match source contents"


def test_dlpack_view_is_zero_copy() -> None:
    """Writing through the source tensor is visible through the DLPack view."""
    _skip_if_no_cuda_or_gpudirect()

    import torch

    from aethergraph._core import _dlpack_capsule_from_cuda_ptr

    num_nodes = 4
    feature_dim = 8

    src = torch.zeros(num_nodes, feature_dim, dtype=torch.float32, device="cuda:0")
    capsule = _dlpack_capsule_from_cuda_ptr(src.data_ptr(), num_nodes, feature_dim, 0)
    dst = torch.from_dlpack(capsule)  # type: ignore[attr-defined]

    # Mutate the source after the view is created. Zero-copy means dst sees it.
    src.fill_(3.5)
    assert torch.all(dst == 3.5), "DLPack view did not observe source mutation"
