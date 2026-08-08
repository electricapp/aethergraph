"""Shared helpers for validating and converting node/vertex ID arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

_U32_MAX = np.iinfo(np.uint32).max


def _to_uint32_ids(arr: npt.NDArray[Any], what: str) -> npt.NDArray[np.uint32]:
    """Convert node IDs to ``uint32`` after range-checking.

    The Rust graph types index nodes with ``u32``. A bare
    ``astype(np.uint32)`` would silently wrap IDs >= 2**32 and turn negative
    sentinels (e.g. ``-1``) into huge IDs, so the range is validated first.

    Args:
        arr: Array-like of node IDs.
        what: Human-readable description of the array, used in error messages.

    Returns:
        A ``uint32`` array of node IDs. Arrays that are already ``uint32``
        are returned without copying.

    Raises:
        ValueError: If any ID is negative or exceeds ``2**32 - 1``.
    """
    a = np.asarray(arr)
    # Already-uint32 arrays cannot be out of range: return without scanning.
    # This is the common case for streaming edge updates, where this helper
    # runs per insert batch.
    if a.dtype == np.uint32:
        return a
    if a.size:
        a_min = int(a.min())
        a_max = int(a.max())
        if a_min < 0:
            raise ValueError(f"{what} contains negative node IDs (min {a_min})")
        if a_max > _U32_MAX:
            raise ValueError(
                f"{what} contains node IDs exceeding uint32 range (max {a_max} > {_U32_MAX})"
            )
    return a.astype(np.uint32, copy=False)
