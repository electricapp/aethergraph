//! Software prefetch hints for latency-bound random-access loops.
//!
//! The sampler's dedup probes and the feature gather touch cache lines whose
//! addresses are known several iterations ahead of use; issuing a prefetch
//! hint at that distance overlaps the miss latency with useful work. Hints
//! are advisory — an invalid address is simply ignored by the hardware — so
//! the helper is safe to call with any pointer.

/// Hint the CPU to pull the line containing `ptr` toward L1.
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetch is a pure hint with no memory effects; any address,
    // valid or not, is architecturally safe.
    unsafe {
        core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0)
    };
    #[cfg(target_arch = "aarch64")]
    // SAFETY: prfm is a pure hint with no memory effects; any address,
    // valid or not, is architecturally safe.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{p}]",
            p = in(reg) ptr,
            options(nostack, preserves_flags),
        );
    };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = ptr;
}
