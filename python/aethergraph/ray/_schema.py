"""Row schema shared by the Ray datasource (producer) and collate (consumer).

One PyG batch maps to exactly one Ray row; each variable-length array lives in
a single list cell. Both sides of the pipeline import these names, so the
column contract exists in exactly one place — a rename here is a rename
everywhere, and a drifted producer/consumer cannot type-check.
"""

from __future__ import annotations

from typing import Final

EDGE_SRC: Final = "edge_index_0"
"""Edge source node IDs (int64 list cell)."""

EDGE_DST: Final = "edge_index_1"
"""Edge destination node IDs (int64 list cell)."""

N_ID: Final = "n_id"
"""Global node IDs of the subgraph (int64 list cell)."""

E_ID: Final = "e_id"
"""Global edge IDs (int64 list cell; empty when tracking is off)."""

BATCH_SIZE: Final = "batch_size"
"""Number of seed nodes (int32 scalar)."""

X: Final = "x"
"""Row-major flattened features (float32 list cell); absent when the loader
has no feature source."""

X_ROWS: Final = "x_shape_0"
"""Feature matrix row count (int32 scalar); present iff ``X`` is present."""

X_COLS: Final = "x_shape_1"
"""Feature matrix column count (int32 scalar); present iff ``X`` is present."""
