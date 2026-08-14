//! Self-profiling hardware counters via `perf_event_open`.
//!
//! Wraps a small group of performance counters around a hot loop —
//! cycles, instructions, LLC misses, and dTLB-load misses — so the loader
//! can report *why* a batch was slow, not just that it was. The
//! dTLB-miss counter in particular is the hugepage-payoff meter: it drops
//! sharply once the feature table sits on 2 MiB pages.
//!
//! Counters degrade to absent, never to error. A virtualized host with no
//! PMU passthrough, or a `perf_event_paranoid` setting that forbids
//! user-space counting, simply yields fewer (or no) hardware counters;
//! the always-available software task-clock keeps the timing field live.
//! Callers read [`CounterSet::measure`]'s snapshot and treat any missing
//! field as "not measured here".

#![cfg(all(target_os = "linux", feature = "perf"))]

use std::os::unix::io::{FromRawFd, OwnedFd, RawFd};

// perf_event ABI (linux/perf_event.h). The struct and constants are
// stable kernel ABI; defined locally to avoid a bindgen dependency.

const PERF_TYPE_HARDWARE: u32 = 0;
const PERF_TYPE_SOFTWARE: u32 = 1;
const PERF_TYPE_HW_CACHE: u32 = 3;

const PERF_COUNT_HW_CPU_CYCLES: u64 = 0;
const PERF_COUNT_HW_INSTRUCTIONS: u64 = 1;
const PERF_COUNT_SW_TASK_CLOCK: u64 = 1;

// Cache-counter config is (id) | (op << 8) | (result << 16).
const PERF_COUNT_HW_CACHE_LL: u64 = 2;
const PERF_COUNT_HW_CACHE_DTLB: u64 = 3;
const PERF_COUNT_HW_CACHE_OP_READ: u64 = 0;
const PERF_COUNT_HW_CACHE_RESULT_MISS: u64 = 1;

const fn cache_config(id: u64, op: u64, result: u64) -> u64 {
    id | (op << 8) | (result << 16)
}

// perf_event_attr flag bits we set (bitfield word after `config`).
// Adds `time_enabled` / `time_running` after the value in every read, so a
// count taken while the PMU was shared can be scaled back to the full
// measurement window.
const PERF_FORMAT_TOTAL_TIME_ENABLED: u64 = 1 << 0;
const PERF_FORMAT_TOTAL_TIME_RUNNING: u64 = 1 << 1;

const ATTR_DISABLED: u64 = 1 << 0;
const ATTR_EXCLUDE_KERNEL: u64 = 1 << 5;
const ATTR_EXCLUDE_HV: u64 = 1 << 6;

// ioctl request numbers for enable/disable/reset (_IO('$', n)).
const PERF_EVENT_IOC_ENABLE: libc::c_ulong = 0x2400;
const PERF_EVENT_IOC_DISABLE: libc::c_ulong = 0x2401;
const PERF_EVENT_IOC_RESET: libc::c_ulong = 0x2403;

/// `struct perf_event_attr`, trimmed to the head fields we set (the kernel
/// zero-fills the tail). `size` tells the kernel the real struct size so
/// the layout stays forward-compatible.
#[repr(C)]
#[derive(Default)]
struct PerfEventAttr {
    type_: u32,
    size: u32,
    config: u64,
    sample_period_or_freq: u64,
    sample_type: u64,
    read_format: u64,
    flags: u64,
    wakeup: u32,
    bp_type: u32,
    bp_addr_or_config1: u64,
    bp_len_or_config2: u64,
    branch_sample_type: u64,
    sample_regs_user: u64,
    sample_stack_user: u32,
    clockid: i32,
    sample_regs_intr: u64,
    aux_watermark: u32,
    sample_max_stack: u16,
    __reserved_2: u16,
}

/// One kind of counter we try to open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counter {
    /// Total CPU cycles (hardware).
    Cycles,
    /// Retired instructions (hardware).
    Instructions,
    /// Last-level cache read misses (hardware).
    LlcMisses,
    /// dTLB load misses (hardware) — the hugepage-payoff meter.
    DtlbMisses,
    /// Task wall-clock in nanoseconds (software; always available).
    TaskClockNs,
}

impl Counter {
    fn attr(self) -> PerfEventAttr {
        let (type_, config) = match self {
            Counter::Cycles => (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES),
            Counter::Instructions => (PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS),
            Counter::LlcMisses => (
                PERF_TYPE_HW_CACHE,
                cache_config(
                    PERF_COUNT_HW_CACHE_LL,
                    PERF_COUNT_HW_CACHE_OP_READ,
                    PERF_COUNT_HW_CACHE_RESULT_MISS,
                ),
            ),
            Counter::DtlbMisses => (
                PERF_TYPE_HW_CACHE,
                cache_config(
                    PERF_COUNT_HW_CACHE_DTLB,
                    PERF_COUNT_HW_CACHE_OP_READ,
                    PERF_COUNT_HW_CACHE_RESULT_MISS,
                ),
            ),
            Counter::TaskClockNs => (PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK),
        };
        PerfEventAttr {
            type_,
            size: std::mem::size_of::<PerfEventAttr>() as u32,
            config,
            // Ask for the enabled/running times alongside each count. A PMU
            // has few hardware slots, so with several events open the kernel
            // time-slices them and the raw count covers only the fraction of
            // the window the event was scheduled. Without these two numbers
            // that scaling is invisible and a ratio like IPC silently
            // compares counts taken over different slices.
            read_format: PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING,
            // Start disabled; count only user space so the numbers reflect
            // the measured code, not surrounding kernel/hypervisor work.
            flags: ATTR_DISABLED | ATTR_EXCLUDE_KERNEL | ATTR_EXCLUDE_HV,
            ..Default::default()
        }
    }
}

/// A snapshot of counter readings; a field is `None` when that counter
/// couldn't be opened on this host.
#[derive(Debug, Default, Clone, Copy)]
pub struct CounterReadings {
    pub cycles: Option<u64>,
    pub instructions: Option<u64>,
    pub llc_misses: Option<u64>,
    pub dtlb_misses: Option<u64>,
    pub task_clock_ns: Option<u64>,
    /// At least one counter shared the PMU with another event and its
    /// value was extrapolated from the fraction of the window it ran for.
    /// The numbers are then estimates: treat them as indicative, and open
    /// fewer counters for a measurement that has to be exact.
    pub multiplexed: bool,
}

/// A set of opened counters, measured together around a closure.
pub struct CounterSet {
    fds: Vec<(Counter, OwnedFd)>,
}

impl CounterSet {
    /// Open the default hot-loop counter set. Every counter is best-effort:
    /// the returned set contains only those the host actually granted.
    pub fn open_default() -> Self {
        Self::open(&[
            Counter::Cycles,
            Counter::Instructions,
            Counter::LlcMisses,
            Counter::DtlbMisses,
            Counter::TaskClockNs,
        ])
    }

    /// Try to open each of `counters`; silently drop any the host refuses.
    pub fn open(counters: &[Counter]) -> Self {
        let mut fds = Vec::new();
        for &c in counters {
            if let Some(fd) = open_counter(c) {
                fds.push((c, fd));
            }
        }
        Self { fds }
    }

    /// Number of counters actually opened.
    pub fn active(&self) -> usize {
        self.fds.len()
    }

    /// Reset, enable, run `f`, disable, and read every counter. The
    /// readings bracket exactly `f`'s work (plus a fixed few syscalls of
    /// enable/disable overhead, identical across calls).
    pub fn measure<R>(&self, f: impl FnOnce() -> R) -> (R, CounterReadings) {
        self.start();
        let out = f();
        (out, self.stop())
    }

    /// Zero every counter and begin counting.
    ///
    /// Paired with [`Self::stop`] for callers that can't wrap their work
    /// in a closure — a Python `with` block, or a region spanning an
    /// `await`. [`Self::measure`] is the safer choice where it fits, since
    /// it cannot leave counters running.
    pub fn start(&self) {
        for (_, fd) in &self.fds {
            ioctl(fd.as_raw_fd(), PERF_EVENT_IOC_RESET);
            ioctl(fd.as_raw_fd(), PERF_EVENT_IOC_ENABLE);
        }
    }

    /// Stop counting and read every counter.
    ///
    /// Reading without a preceding [`Self::start`] yields whatever the
    /// counters hold — zero for a freshly opened set.
    pub fn stop(&self) -> CounterReadings {
        let mut readings = CounterReadings::default();
        for (c, fd) in &self.fds {
            ioctl(fd.as_raw_fd(), PERF_EVENT_IOC_DISABLE);
            let read = read_counter(fd.as_raw_fd());
            if let Some((_, true)) = read {
                readings.multiplexed = true;
            }
            let value = read.map(|(v, _)| v);
            match c {
                Counter::Cycles => readings.cycles = value,
                Counter::Instructions => readings.instructions = value,
                Counter::LlcMisses => readings.llc_misses = value,
                Counter::DtlbMisses => readings.dtlb_misses = value,
                Counter::TaskClockNs => readings.task_clock_ns = value,
            }
        }
        readings
    }
}

impl CounterReadings {
    /// Instructions per cycle, when both counters were available.
    ///
    /// The headline efficiency number: below ~1.0 on this workload means
    /// the core is stalled on memory rather than retiring work.
    pub fn ipc(&self) -> Option<f64> {
        match (self.instructions, self.cycles) {
            (Some(i), Some(c)) if c > 0 => Some(i as f64 / c as f64),
            _ => None,
        }
    }

    /// LLC misses per thousand instructions, when both were available.
    ///
    /// The cache-behaviour number to watch when changing layout or
    /// prefetch distance.
    pub fn llc_misses_per_kilo_instruction(&self) -> Option<f64> {
        match (self.llc_misses, self.instructions) {
            (Some(m), Some(i)) if i > 0 => Some(m as f64 * 1000.0 / i as f64),
            _ => None,
        }
    }
}

use std::os::unix::io::AsRawFd;

fn open_counter(counter: Counter) -> Option<OwnedFd> {
    let attr = counter.attr();
    // perf_event_open(attr, pid=0 (this thread), cpu=-1 (any), group_fd=-1,
    // flags=PERF_FLAG_FD_CLOEXEC=8).
    // SAFETY: `attr` is a valid perf_event_attr with its `size` set; the
    // remaining scalar args are the documented "count this thread on any
    // CPU" form.
    let fd = unsafe {
        libc::syscall(
            libc::SYS_perf_event_open,
            &attr as *const PerfEventAttr,
            0,
            -1,
            -1,
            8,
        )
    };
    if fd < 0 {
        return None;
    }
    // SAFETY: `fd` is a fresh perf event fd we exclusively own.
    Some(unsafe { OwnedFd::from_raw_fd(fd as RawFd) })
}

/// One counter's value, scaled for PMU multiplexing.
///
/// Returns `None` when the read fails or the event never got scheduled at
/// all (`time_running == 0`), which is not the same as a genuine zero.
/// `scaled` reports whether the event shared the PMU, meaning the value is
/// an extrapolation rather than an exact count.
fn read_counter(fd: RawFd) -> Option<(u64, bool)> {
    // Layout for read_format = TOTAL_TIME_ENABLED | TOTAL_TIME_RUNNING:
    // { u64 value; u64 time_enabled; u64 time_running; }
    let mut buf = [0u8; 24];
    // SAFETY: reading 24 bytes into a local buffer from a perf event fd
    // opened with both time fields in its read format.
    let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
    if n != buf.len() as isize {
        return None;
    }
    let field = |i: usize| {
        let mut b = [0u8; 8];
        b.copy_from_slice(&buf[i * 8..(i + 1) * 8]);
        u64::from_ne_bytes(b)
    };
    let (value, enabled, running) = (field(0), field(1), field(2));
    if running == 0 {
        // Enabled but never scheduled — reporting 0 would read as "this
        // never happened" rather than "this was not measured".
        return None;
    }
    if running >= enabled {
        return Some((value, false));
    }
    // Extrapolate to the full window: value * enabled / running, in u128 so
    // a long measurement cannot overflow the intermediate product.
    let scaled = (value as u128 * enabled as u128 / running as u128).min(u64::MAX as u128) as u64;
    Some((scaled, true))
}

fn ioctl(fd: RawFd, request: libc::c_ulong) {
    // SAFETY: `fd` is an open perf event fd; the request is a no-argument
    // enable/disable/reset control code.
    unsafe {
        libc::ioctl(fd, request, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_clock_always_measures_time() {
        // The software task-clock counter is available even in a VM with
        // no PMU; it must open and report a positive duration for real work.
        let set = CounterSet::open(&[Counter::TaskClockNs]);
        if set.active() == 0 {
            eprintln!("perf_event_open unavailable (paranoid setting?); skipping");
            return;
        }
        let (sum, readings) = set.measure(|| {
            let mut s = 0u64;
            for i in 0..2_000_000u64 {
                s = s.wrapping_add(i);
            }
            s
        });
        assert_eq!(sum, 1_999_999u64 * 2_000_000 / 2);
        let ns = readings.task_clock_ns.expect("task clock opened");
        assert!(ns > 0, "task clock should record positive time");
    }

    #[test]
    fn hardware_counters_are_best_effort() {
        // Opening HW counters must never panic; on a runner without PMU
        // access the set is simply smaller. When cycles opened, a busy
        // loop must show a nonzero count.
        let set = CounterSet::open_default();
        let (_out, readings) = set.measure(|| {
            let mut s = 0u64;
            for i in 0..1_000_000u64 {
                s = s.wrapping_add(i.wrapping_mul(3));
            }
            s
        });
        if let Some(cycles) = readings.cycles {
            assert!(cycles > 0, "a million iterations must burn cycles");
        } else {
            eprintln!("HW cycle counter unavailable here; degraded as designed");
        }
    }

    #[test]
    fn open_default_never_panics_and_reports_active() {
        let set = CounterSet::open_default();
        // active() is between 0 and 5; the call itself is the assertion
        // that construction degrades gracefully.
        assert!(set.active() <= 5);
    }
}
