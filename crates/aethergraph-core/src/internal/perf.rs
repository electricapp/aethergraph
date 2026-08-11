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
        for (_, fd) in &self.fds {
            ioctl(fd.as_raw_fd(), PERF_EVENT_IOC_RESET);
            ioctl(fd.as_raw_fd(), PERF_EVENT_IOC_ENABLE);
        }
        let out = f();
        let mut readings = CounterReadings::default();
        for (c, fd) in &self.fds {
            ioctl(fd.as_raw_fd(), PERF_EVENT_IOC_DISABLE);
            let value = read_counter(fd.as_raw_fd());
            match c {
                Counter::Cycles => readings.cycles = value,
                Counter::Instructions => readings.instructions = value,
                Counter::LlcMisses => readings.llc_misses = value,
                Counter::DtlbMisses => readings.dtlb_misses = value,
                Counter::TaskClockNs => readings.task_clock_ns = value,
            }
        }
        (out, readings)
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

fn read_counter(fd: RawFd) -> Option<u64> {
    let mut buf = [0u8; 8];
    // SAFETY: reading 8 bytes into a local buffer from a perf event fd,
    // whose default read format is a single u64 count.
    let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, 8) };
    if n == 8 {
        Some(u64::from_ne_bytes(buf))
    } else {
        None
    }
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
