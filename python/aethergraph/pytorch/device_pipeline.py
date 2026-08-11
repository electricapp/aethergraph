"""Stream-pipelined host-to-device transfer for sampled batches.

A neighbor loader that pins its batches already lets the training loop
issue an async H2D copy, but if the copy runs on the default stream it
still serializes against compute: batch N+1 cannot start crossing PCIe
until batch N's kernels drain. :class:`DeviceTransferPipeline` moves each
batch on a dedicated copy stream and double-buffers, so batch N+1's
transfer overlaps batch N's training — the single biggest GPU-utilization
win available without touching the model.

The overlap is CUDA-specific; when no CUDA device is available the
pipeline is a transparent passthrough, so the same loader code runs
unchanged on CPU (and in CI).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

try:
    import torch
except ImportError as e:  # pragma: no cover - torch is a hard dep of this module
    raise ImportError("aethergraph.pytorch requires torch") from e

if TYPE_CHECKING:
    from torch_geometric.data import Data, HeteroData

    Batch = Data | HeteroData


def move_data_to_device(
    data: Batch,
    device: torch.device,
    *,
    non_blocking: bool,
) -> Batch:
    """Move every tensor a PyG batch carries to ``device``, in place.

    Delegates to PyG's own ``.to`` so both homogeneous ``Data`` and
    heterogeneous ``HeteroData`` (per node/edge type) move uniformly;
    non-tensor metadata is left alone. ``non_blocking`` is honored only for
    pinned source tensors — pageable tensors copy synchronously regardless,
    which is why the loader pins.
    """
    return data.to(device, non_blocking=non_blocking)


def _record_stream(data: Batch, stream: torch.cuda.Stream) -> None:
    """Tie ``data``'s device tensors to ``stream``.

    Tensors produced on the copy stream must not have their memory reused
    by the caching allocator until work queued on the consumer stream has
    run. ``record_stream`` records that dependency so the allocator waits.
    Walks every PyG storage so it covers HeteroData's per-type tensors too.
    """
    for store in data.stores:
        for value in store.values():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                value.record_stream(stream)


class DeviceTransferPipeline:
    """Overlap batch H2D transfers with consumer compute.

    Wraps an iterator of PyG ``Data`` batches. Each batch is moved to the
    device on a dedicated copy stream; the pipeline keeps ``depth`` batches
    in flight so, while the consumer trains on batch N, batches N+1..N+depth
    are already crossing (or have crossed) PCIe. The consumer's default
    stream is made to wait on each batch's copy-completion event before the
    batch is yielded, so the returned tensors are safe to use immediately.

    On a machine with no CUDA device the wrapper yields the source batches
    unchanged — same iteration, no streams, no copies.
    """

    def __init__(
        self,
        source: Iterable[Data],
        device: torch.device | str,
        *,
        depth: int = 2,
    ) -> None:
        if depth < 1:
            raise ValueError(f"pipeline depth must be >= 1, got {depth}")
        self._source = source
        self._device = torch.device(device)
        self._depth = depth
        self._enabled = self._device.type == "cuda" and torch.cuda.is_available()

    def __iter__(self) -> Iterator[Data]:
        if not self._enabled:
            # CPU / no-CUDA: transparent passthrough.
            yield from self._source
            return
        yield from self._run_cuda()

    def _run_cuda(self) -> Iterator[Data]:
        copy_stream = torch.cuda.Stream(device=self._device)
        # Each in-flight entry is (data_on_device, copy_done_event).
        inflight: deque[tuple[Data, torch.cuda.Event]] = deque()
        source = iter(self._source)

        def issue_next() -> bool:
            """Queue one batch's H2D on the copy stream. False when drained."""
            try:
                data = next(source)
            except StopIteration:
                return False
            with torch.cuda.stream(copy_stream):
                data = move_data_to_device(data, self._device, non_blocking=True)
                done = torch.cuda.Event()
                done.record(copy_stream)
            inflight.append((data, done))
            return True

        # Prime the pipeline so `depth` transfers are in flight before the
        # first yield.
        for _ in range(self._depth):
            if not issue_next():
                break

        consumer = torch.cuda.current_stream(device=self._device)
        while inflight:
            data, done = inflight.popleft()
            # The consumer stream must not touch the tensors until their
            # copy has completed; then hand the allocator the dependency.
            consumer.wait_event(done)
            _record_stream(data, consumer)
            # Refill so the pipeline stays `depth` deep behind this yield.
            issue_next()
            yield data


__all__ = ["DeviceTransferPipeline", "move_data_to_device"]
