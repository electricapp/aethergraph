//! Store ordering for memory a device reads by DMA.

use std::sync::atomic::{Ordering, fence};

/// Release fence whose ordering also holds for DMA observers.
///
/// Stores before the fence become visible — to other CPUs and to devices
/// reading coherent host memory (an RDMA NIC, a GPU over PCIe) — before
/// stores after it. On aarch64 `fence(Release)` alone is `dmb ish`, which
/// orders only for the CPUs' inner-shareable domain; the outer-shareable
/// store barrier (Linux's `dma_wmb`) extends it to devices. x86 stores are
/// observed in program order by every agent, so there it is a compiler
/// barrier.
#[inline(always)]
pub fn dma_release_fence() {
    fence(Ordering::Release);
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    // SAFETY: a barrier instruction; it reads and writes no memory or
    // registers.
    unsafe {
        std::arch::asm!("dmb oshst", options(nostack, preserves_flags));
    }
}
