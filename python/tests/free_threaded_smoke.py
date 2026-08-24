"""Free-threaded (no-GIL) smoke test for the aethergraph extension.

Run under a free-threaded CPython (``python3.14t``). It proves two things:

1. Importing ``aethergraph._core`` does NOT re-enable the GIL — i.e. the
   ``#[pymodule(gil_used = false)]`` declaration took effect. A module that
   omits it flips the GIL back on at import (with a runtime warning), which
   this asserts against.
2. N threads sampling one shared graph concurrently produce correct
   results with no crashes or data races. The sampler releases the GIL for
   the Rust work, so on a free-threaded build these run in true parallel.
3. Returned numpy arrays stay valid after further sampling on other threads
   (regression for "borrow across ``py.detach``" UAF under ``gil_used =
   false`` — accessors must hand back owned copies, not views into Rust).

Imports ``aethergraph._core`` directly (not the torch-dependent
``aethergraph.pytorch``), so it needs only numpy — which ships
free-threaded wheels.
"""

from __future__ import annotations

import sys
import threading

import numpy as np

from aethergraph._core import CsrGraph, NeighborSampler, SamplingConfig


def main() -> None:
    # (1) The GIL must be off both before and after importing the extension.
    gil_probe = getattr(sys, "_is_gil_enabled", None)
    if gil_probe is None:
        print("interpreter has no _is_gil_enabled; not a 3.13+ build — skipping")
        return
    assert not gil_probe(), (
        "GIL is enabled after importing aethergraph._core; gil_used = false did not take effect"
    )
    print(f"free-threaded OK: {sys.version}")

    # (2) Build a shared graph and hammer it from many threads.
    rng = np.random.default_rng(0)
    num_nodes = 5000
    num_edges = 200_000
    src = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    dst = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    graph = CsrGraph.from_edges(num_nodes, src, dst)

    config = SamplingConfig(num_neighbors=[15, 10], seed=42)
    n_threads = 8
    iters = 200
    errors: list[str] = []
    barrier = threading.Barrier(n_threads)
    # (3) Snapshots of returned arrays kept alive across concurrent samples.
    snapshots: list[tuple[np.ndarray, np.ndarray]] = []
    snap_lock = threading.Lock()

    def worker(tid: int) -> None:
        try:
            # Each thread owns its sampler; the graph is shared read-only.
            sampler = NeighborSampler(graph, config)
            seeds = rng.integers(0, num_nodes, size=64, dtype=np.int64)
            barrier.wait()  # maximize real overlap under no-GIL
            for i in range(iters):
                sub = sampler.sample(seeds)
                n = sub.num_nodes
                # Every seed is always present, so the subgraph is never
                # smaller than the seed set.
                if n < len(seeds):
                    raise AssertionError(f"num_nodes {n} < seeds {len(seeds)}")
                nodes = np.array(sub.nodes, copy=True)
                edge_index = np.array(sub.edge_index, copy=True)
                if i == 0:
                    # Keep a live reference without copying so a borrowed
                    # view into freed Rust memory would be detectable when
                    # we re-check after the barrier below.
                    with snap_lock:
                        snapshots.append((sub.nodes, sub.edge_index))
                # Touch the arrays again after more Rust work on this thread.
                if nodes.shape[0] != n:
                    raise AssertionError("nodes length drifted after copy")
                if edge_index.ndim != 2:
                    raise AssertionError("edge_index rank drifted after copy")
        except Exception as e:  # noqa: BLE001 - report, don't crash the run
            errors.append(f"thread {tid}: {e!r}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        for e in errors:
            print("ERROR:", e)
        raise SystemExit(1)

    # After all workers finish, held arrays must still be readable and sane.
    for nodes, edge_index in snapshots:
        _ = int(nodes.sum()) + int(edge_index.sum())
        if nodes.ndim != 1 or edge_index.ndim != 2:
            raise SystemExit("held sample arrays corrupted after concurrent sampling")
    print(f"{n_threads} threads x {iters} samples: OK ({len(snapshots)} held arrays)")


if __name__ == "__main__":
    main()
