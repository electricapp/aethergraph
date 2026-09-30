"""Shared closed-vocabulary types for the Python API surface.

These aliases give enum-ish parameters a static domain: mypy rejects a typo
like ``subgraph_type="directed"`` at the call site, and the single runtime
validation for untyped callers lives in the Rust FFI converter that consumes
the value (``_core.SamplingConfig``).
"""

from __future__ import annotations

from typing import Literal

SubgraphType = Literal["directional", "induced", "bidirectional"]
"""Which edges the sampled subgraph keeps."""

TemporalStrategy = Literal["uniform", "last"]
"""Neighbor selection under timestamps; ``None`` at use sites disables it."""
