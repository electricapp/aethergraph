"""Public ``Graph`` API — a thin re-export of the Rust ``CsrGraph`` class.

The Rust ``CsrGraph`` IS the public Python API. This module exists only to
preserve the historical import path ``from aethergraph import Graph``.

The previous Python wrapper (manual property forwarders, an attached
``features`` numpy attribute, etc.) was removed because it duplicated state
already on the Rust side and drifted from it. The ``features`` and
``feature_path`` state that used to live on ``Graph`` is now a constructor
argument on the loaders that consume it
(:class:`aethergraph.pytorch.NeighborLoader`, :class:`aethergraph.Sampler`).

Example:
    >>> from aethergraph import Graph
    >>> g = Graph.load("graph.bin")
    >>> g.num_nodes                                   # property on Rust class
    100000
    >>> g.neighbors(0)                                # method on Rust class
    array([1, 2, 3, ...], dtype=uint32)
"""

from __future__ import annotations

from aethergraph._core import CsrGraph as Graph

__all__ = ["Graph"]
