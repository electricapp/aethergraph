//! Python bindings for hardware performance counters.

use aethergraph_core::{Counter, CounterReadings, CounterSet};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Hardware performance counters for a block of Python work.
///
/// Wraps `perf_event_open` counters counting user space only, so the
/// readings describe the measured code rather than surrounding kernel work.
/// Every counter is best-effort: hosts that withhold PMU access (most
/// containers, and any machine with `perf_event_paranoid > 2`) yield a set
/// with fewer counters, and the corresponding readings come back None
/// instead of raising.
///
/// **This measures the calling thread only.** The events are opened with
/// `pid = 0`, which counts the one thread that opened them. Work the
/// loader does on its sampler threads, and io_uring gathers running on
/// their own lane threads, are not included — a block that mostly waits on
/// them reports very few instructions, which looks like a fast path and is
/// not one. Read it as "what this thread did", not "what this batch cost".
///
/// `readings()["multiplexed"]` being True means the PMU was shared and the
/// counts are scaled estimates rather than exact totals; open fewer
/// counters when a measurement has to be exact.
///
/// Linux only — present only when `HAS_PERF_COUNTERS` is True.
///
/// # Example
/// ```python
/// # Measures the training step, which runs on this thread. The loader's
/// # sampling runs elsewhere and is deliberately not in this window.
/// with PerfCounters() as counters:
///     train_step(batch)
/// r = counters.readings()
/// print(r["ipc"], "estimated" if r["multiplexed"] else "exact")
/// ```
#[pyclass(name = "PerfCounters")]
pub struct PyPerfCounters {
    inner: CounterSet,
    last: Option<CounterReadings>,
}

#[pymethods]
impl PyPerfCounters {
    /// Open a counter set.
    ///
    /// # Arguments
    /// - counters: Names to open, from "cycles", "instructions",
    ///   "llc_misses", "dtlb_misses", "task_clock". Defaults to all of
    ///   them.
    #[new]
    #[pyo3(signature = (counters=None))]
    fn new(counters: Option<Vec<String>>) -> PyResult<Self> {
        let inner = match counters {
            None => CounterSet::open_default(),
            Some(names) => {
                let parsed: Vec<Counter> = names
                    .iter()
                    .map(|n| match n.as_str() {
                        "cycles" => Ok(Counter::Cycles),
                        "instructions" => Ok(Counter::Instructions),
                        "llc_misses" => Ok(Counter::LlcMisses),
                        "dtlb_misses" => Ok(Counter::DtlbMisses),
                        "task_clock" => Ok(Counter::TaskClockNs),
                        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "unknown counter {other:?}: expected \"cycles\", \
                             \"instructions\", \"llc_misses\", \"dtlb_misses\", \
                             or \"task_clock\""
                        ))),
                    })
                    .collect::<PyResult<_>>()?;
                CounterSet::open(&parsed)
            }
        };
        Ok(Self { inner, last: None })
    }

    /// How many counters the host actually granted. Zero means the PMU is
    /// unavailable here and every reading will be None.
    #[getter]
    fn active(&self) -> usize {
        self.inner.active()
    }

    /// Zero the counters and start counting.
    fn start(&mut self) {
        self.last = None;
        self.inner.start();
    }

    /// Stop counting and latch the readings.
    fn stop(&mut self) {
        self.last = Some(self.inner.stop());
    }

    /// The latest readings as a dict, or None before the first `stop()`.
    ///
    /// Keys: cycles, instructions, llc_misses, dtlb_misses,
    /// task_clock_ns (each int or None), plus the derived ipc and
    /// llc_misses_per_kilo_instruction (float or None), and multiplexed
    /// (bool) — True when the PMU was shared and the counts are scaled
    /// estimates rather than exact totals.
    fn readings(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Some(r) = self.last else {
            return Ok(None);
        };
        let dict = PyDict::new(py);
        dict.set_item("cycles", r.cycles)?;
        dict.set_item("instructions", r.instructions)?;
        dict.set_item("llc_misses", r.llc_misses)?;
        dict.set_item("dtlb_misses", r.dtlb_misses)?;
        dict.set_item("task_clock_ns", r.task_clock_ns)?;
        dict.set_item("multiplexed", r.multiplexed)?;
        dict.set_item("ipc", r.ipc())?;
        dict.set_item(
            "llc_misses_per_kilo_instruction",
            r.llc_misses_per_kilo_instruction(),
        )?;
        Ok(Some(dict.into()))
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.start();
        slf
    }

    /// Latches the readings on the way out, including when the block
    /// exits by exception — the measurement is still valid, and
    /// swallowing it would lose the numbers for the failing case.
    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        _exc_type: Option<Bound<'_, PyAny>>,
        _exc_value: Option<Bound<'_, PyAny>>,
        _traceback: Option<Bound<'_, PyAny>>,
    ) -> bool {
        slf.stop();
        // False: never suppress an exception raised inside the block.
        false
    }

    fn __repr__(&self) -> String {
        format!(
            "PerfCounters(active={}, measured={})",
            self.inner.active(),
            self.last.is_some()
        )
    }
}
