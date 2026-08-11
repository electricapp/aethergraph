"""Hardware performance counters around Python work.

Linux only, and the PMU itself may be withheld (containers, high
`perf_event_paranoid`). The API contract holds either way: readings come
back None rather than raising, so these tests assert shape and behaviour
without requiring a real PMU.
"""

from __future__ import annotations

import numpy as np
import pytest

from aethergraph import Graph
from aethergraph._core import HAS_PERF_COUNTERS

pytestmark = pytest.mark.skipif(not HAS_PERF_COUNTERS, reason="perf counters are Linux-only")

if HAS_PERF_COUNTERS:
    from aethergraph._core import PerfCounters

READING_KEYS = {
    "cycles",
    "instructions",
    "llc_misses",
    "dtlb_misses",
    "task_clock_ns",
    "ipc",
    "llc_misses_per_kilo_instruction",
}


class TestPerfCounters:
    def test_readings_none_before_stop(self) -> None:
        counters = PerfCounters()
        assert counters.readings() is None
        assert counters.active >= 0

    def test_context_manager_latches_readings(self) -> None:
        rng = np.random.default_rng(0)
        src = rng.integers(0, 5000, size=50_000, dtype=np.uint32)
        dst = rng.integers(0, 5000, size=50_000, dtype=np.uint32)
        graph = Graph.from_edges(5000, src, dst)

        with PerfCounters() as counters:
            for _ in range(20):
                graph.degrees()

        readings = counters.readings()
        assert readings is not None
        assert set(readings) == READING_KEYS

        # Where the host granted a counter, the value must be a real
        # measurement of the block; where it didn't, None.
        for key in ("cycles", "instructions", "task_clock_ns"):
            value = readings[key]
            assert value is None or (isinstance(value, int) and value >= 0)

        if readings["instructions"] and readings["cycles"]:
            assert readings["ipc"] > 0
        else:
            assert readings["ipc"] is None

    def test_exception_inside_block_still_latches(self) -> None:
        counters = PerfCounters()
        with pytest.raises(RuntimeError, match="boom"), counters:
            raise RuntimeError("boom")

        # The measurement of a failing block is still a measurement, and
        # __exit__ must not swallow the exception (pytest.raises above).
        assert counters.readings() is not None

    def test_explicit_start_stop(self) -> None:
        counters = PerfCounters(["instructions", "task_clock"])
        counters.start()
        sum(range(10_000))
        counters.stop()

        readings = counters.readings()
        assert readings is not None
        # Counters that were never opened read None.
        assert readings["llc_misses"] is None

    def test_restart_clears_previous_readings(self) -> None:
        counters = PerfCounters()
        with counters:
            sum(range(1000))
        assert counters.readings() is not None

        counters.start()
        assert counters.readings() is None

    def test_unknown_counter_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown counter"):
            PerfCounters(["branch_mispredicts"])
