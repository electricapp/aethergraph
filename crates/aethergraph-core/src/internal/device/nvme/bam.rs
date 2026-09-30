//! K1.1 BaM / GIDS-style GPU-initiated NVMe path.
//!
//! Host side: map controller BAR0 as I/O memory, place SQ/CQ rings in a
//! DMA-visible buffer (VRAM via GPUDirect or pinned host), submit
//! [`super::NvmeRwSqe`] entries, and ring the submission-queue doorbell with
//! a volatile MMIO store.
//!
//! GPU threads reuse the same ring protocol (`submit_sqe` + `ring_doorbell`).
//! Mapping BAR0 through `cudaHostRegisterIoMemory` is the hardware step —
//! this module owns the queue arithmetic and MMIO contract.

use super::NvmeRwSqe;
use core::sync::atomic::{AtomicU32, Ordering};

/// NVMe doorbell stride is typically 4 or more dwords; CAP.DSTRD encodes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvmeDoorbellLayout {
    /// Byte offset of SQ0 tail doorbell from BAR0.
    pub sq0_tdbl_bytes: u32,
    /// Doorbel stride in bytes (`4 << CAP.DSTRD`).
    pub stride_bytes: u32,
}

impl NvmeDoorbellLayout {
    /// Spec-default: SQ0 TDBL at 0x1000, stride 4.
    pub const fn legacy() -> Self {
        Self {
            sq0_tdbl_bytes: 0x1000,
            stride_bytes: 4,
        }
    }

    /// Byte offset of the submission-queue tail doorbell for `qid`.
    pub const fn sq_tdbl_offset(self, qid: u16) -> u32 {
        self.sq0_tdbl_bytes + (qid as u32) * 2 * self.stride_bytes
    }

    /// Byte offset of the completion-queue head doorbell for `qid`.
    pub const fn cq_hdbl_offset(self, qid: u16) -> u32 {
        self.sq0_tdbl_bytes + (qid as u32) * 2 * self.stride_bytes + self.stride_bytes
    }
}

/// Power-of-two submission / completion queue pair living in shared memory.
#[derive(Debug)]
pub struct BamQueuePair {
    pub qid: u16,
    pub depth: u32,
    pub sq_tail: AtomicU32,
    pub cq_head: AtomicU32,
    pub phase: AtomicU32,
}

impl BamQueuePair {
    /// Largest NVMe I/O queue a controller can expose (`CAP.MQES` is a
    /// zero-based 16-bit field).
    pub const MAX_DEPTH: u32 = 1 << 16;

    /// `depth` must be a non-zero power of two no larger than
    /// [`Self::MAX_DEPTH`] (NVMe queue size = depth).
    pub fn new(qid: u16, depth: u32) -> Option<Self> {
        if depth == 0 || !depth.is_power_of_two() || depth > Self::MAX_DEPTH {
            return None;
        }
        Some(Self {
            qid,
            depth,
            sq_tail: AtomicU32::new(0),
            cq_head: AtomicU32::new(0),
            phase: AtomicU32::new(1),
        })
    }

    /// In-flight SQEs: `sq_tail - cq_head`.
    pub fn in_flight(&self) -> u32 {
        self.sq_tail
            .load(Ordering::Acquire)
            .wrapping_sub(self.cq_head.load(Ordering::Acquire))
    }

    /// True when another submit would overwrite an uncompleted slot.
    pub fn is_full(&self) -> bool {
        self.in_flight() >= self.depth
    }

    /// Atomically claim the next SQ slot index, or `None` if the queue is full.
    ///
    /// Returns the pre-claim tail (slot = `prev & (depth-1)`). Concurrent
    /// submitters cannot share a slot.
    pub fn try_claim_sq_slot(&self) -> Option<u32> {
        loop {
            let prev = self.sq_tail.load(Ordering::Acquire);
            let head = self.cq_head.load(Ordering::Acquire);
            if prev.wrapping_sub(head) >= self.depth {
                return None;
            }
            match self.sq_tail.compare_exchange_weak(
                prev,
                prev.wrapping_add(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(prev),
                Err(_) => continue,
            }
        }
    }

    /// Retire `n` completed commands (advances CQ head).
    pub fn retire(&self, n: u32) {
        self.cq_head.fetch_add(n, Ordering::Release);
    }

    /// The SQ tail doorbell value for the current tail counter.
    pub fn doorbell_tail(&self) -> u16 {
        self.doorbell_value(self.sq_tail.load(Ordering::Acquire))
    }

    /// The doorbell value for tail counter `tail`: the slot index the next
    /// submission will use. NVMe requires it to be below the queue size;
    /// the controller treats anything else as an invalid doorbell write
    /// and disables the queue.
    fn doorbell_value(&self, tail: u32) -> u16 {
        // `depth <= MAX_DEPTH`, so the masked slot fits 16 bits.
        (tail & (self.depth - 1)) as u16
    }
}

/// Order every prior store — the SQE, in DMA-visible memory — before a
/// following MMIO doorbell store, as seen by the device.
///
/// An atomic release fence orders stores for other CPUs only; on aarch64
/// it is `dmb ish`, which does not cover a device reading memory. This is
/// the kernel's `wmb()`.
#[inline(always)]
fn device_write_barrier() {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `sfence` has no preconditions.
    unsafe {
        core::arch::x86_64::_mm_sfence();
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `dsb st` has no preconditions; it only waits for prior stores.
    unsafe {
        core::arch::asm!("dsb st", options(nostack, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    core::sync::atomic::fence(Ordering::SeqCst);
}

/// Host-visible BaM controller view: BAR0 + doorbell layout + one I/O QP.
#[derive(Debug)]
pub struct BamController {
    pub doorbells: NvmeDoorbellLayout,
    pub qp: BamQueuePair,
    /// Mapped BAR0 base (I/O memory). Null until `attach_bar0`.
    bar0: *mut u8,
    bar0_len: usize,
}

// SAFETY: doorbell stores are explicitly volatile; the pointer is only
// written from the owning thread after attach.
unsafe impl Send for BamController {}

impl BamController {
    /// Create an unbound controller (no BAR yet).
    pub fn new(qid: u16, depth: u32, doorbells: NvmeDoorbellLayout) -> Option<Self> {
        Some(Self {
            doorbells,
            qp: BamQueuePair::new(qid, depth)?,
            bar0: core::ptr::null_mut(),
            bar0_len: 0,
        })
    }

    /// Attach a previously mapped BAR0 I/O region.
    ///
    /// # Safety
    /// `bar0` must point at `len` bytes of device MMIO (e.g. from
    /// `mmap(/sys/bus/pci/.../resource0)` or `cudaHostRegisterIoMemory`).
    pub unsafe fn attach_bar0(&mut self, bar0: *mut u8, len: usize) {
        self.bar0 = bar0;
        self.bar0_len = len;
    }

    /// Write `sqe` into `sq_ring[slot]` and ring the SQ doorbell.
    ///
    /// # Safety
    /// `sq_ring` must be a live DMA-visible queue of `qp.depth` entries.
    /// BAR0 must be attached.
    pub unsafe fn submit_sqe(
        &self,
        sq_ring: *mut NvmeRwSqe,
        sqe: NvmeRwSqe,
    ) -> Result<u16, BamError> {
        if self.bar0.is_null() {
            return Err(BamError::BarNotMapped);
        }
        let prev = self.qp.try_claim_sq_slot().ok_or(BamError::QueueFull)?;
        let slot = (prev & (self.qp.depth - 1)) as usize;
        // SAFETY: caller guarantees `sq_ring` covers `depth` entries.
        let dst = unsafe { sq_ring.add(slot) };
        // SAFETY: `dst` is within the caller-provided ring.
        unsafe {
            core::ptr::write_volatile(dst, sqe);
        }
        // The controller fetches the SQE by DMA once the doorbell lands.
        device_write_barrier();
        let doorbell = self.qp.doorbell_value(prev.wrapping_add(1));
        self.ring_sq_doorbell(doorbell)?;
        Ok(doorbell)
    }

    /// Retire completed CQ entries so new submits can reuse SQ slots.
    pub fn retire_completions(&self, n: u32) {
        self.qp.retire(n);
    }

    /// `st.relaxed.mmio` equivalent: volatile 32-bit store to SQ TDBL.
    pub fn ring_sq_doorbell(&self, tail: u16) -> Result<(), BamError> {
        if self.bar0.is_null() {
            return Err(BamError::BarNotMapped);
        }
        let off = self.doorbells.sq_tdbl_offset(self.qp.qid) as usize;
        if off + 4 > self.bar0_len {
            return Err(BamError::DoorbellOob);
        }
        // SAFETY: attach_bar0 established MMIO mapping covering this offset.
        let ptr = unsafe { self.bar0.add(off) as *mut u32 };
        // SAFETY: `ptr` is a device MMIO doorbell dword inside the mapped BAR.
        unsafe {
            core::ptr::write_volatile(ptr, u32::from(tail));
        }
        Ok(())
    }
}

/// Errors from the BaM host path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BamError {
    BarNotMapped,
    DoorbellOob,
    QueueFull,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::device::nvme::{NvmeDataPointer, NvmeRwSqe};

    #[test]
    fn doorbell_offsets_match_legacy_cap() {
        let d = NvmeDoorbellLayout::legacy();
        assert_eq!(d.sq_tdbl_offset(0), 0x1000);
        assert_eq!(d.cq_hdbl_offset(0), 0x1004);
        assert_eq!(d.sq_tdbl_offset(1), 0x1008);
    }

    #[test]
    fn claim_advances_distinct_slots() {
        let qp = BamQueuePair::new(1, 16).unwrap();
        assert_eq!(qp.try_claim_sq_slot(), Some(0));
        assert_eq!(qp.try_claim_sq_slot(), Some(1));
    }

    #[test]
    fn submit_requires_bar() {
        let ctl = BamController::new(0, 8, NvmeDoorbellLayout::legacy()).unwrap();
        let mut ring = [NvmeRwSqe::read(1, 0, 0, 0, NvmeDataPointer::Prp { prp1: 0, prp2: 0 }); 8];
        // SAFETY: `ring` outlives the call and has the queue's slot count.
        let err = unsafe { ctl.submit_sqe(ring.as_mut_ptr(), ring[0]) };
        assert_eq!(err, Err(BamError::BarNotMapped));
    }

    #[test]
    fn submit_rejects_when_full_until_retire() {
        let mut ctl = BamController::new(0, 4, NvmeDoorbellLayout::legacy()).unwrap();
        let mut bar = [0u8; 0x2000];
        // SAFETY: `bar` is a live 0x2000 buffer covering every doorbell
        // offset the legacy layout addresses, and outlives `ctl`.
        unsafe { ctl.attach_bar0(bar.as_mut_ptr(), bar.len()) };
        let mut ring = [NvmeRwSqe::read(1, 0, 0, 0, NvmeDataPointer::Prp { prp1: 0, prp2: 0 }); 4];
        for _ in 0..4 {
            // SAFETY: `ring` outlives the call and has the queue's slot count.
            assert!(unsafe { ctl.submit_sqe(ring.as_mut_ptr(), ring[0]) }.is_ok());
        }
        assert_eq!(
            // SAFETY: as above.
            unsafe { ctl.submit_sqe(ring.as_mut_ptr(), ring[0]) },
            Err(BamError::QueueFull)
        );
        ctl.retire_completions(2);
        // SAFETY: as above.
        assert!(unsafe { ctl.submit_sqe(ring.as_mut_ptr(), ring[0]) }.is_ok());
    }

    /// The tail doorbell is a slot index, always below the queue size: the
    /// fifth submission to a depth-4 queue rings 1, not 5.
    #[test]
    fn doorbell_wraps_at_the_queue_depth() {
        let mut ctl = BamController::new(0, 4, NvmeDoorbellLayout::legacy()).unwrap();
        let mut bar = [0u8; 0x2000];
        // SAFETY: `bar` covers every legacy doorbell offset and outlives `ctl`.
        unsafe { ctl.attach_bar0(bar.as_mut_ptr(), bar.len()) };
        let mut ring = [NvmeRwSqe::read(1, 0, 0, 0, NvmeDataPointer::Prp { prp1: 0, prp2: 0 }); 4];
        let mut rung = Vec::new();
        for _ in 0..6 {
            // SAFETY: `ring` outlives the call and has the queue's slot count.
            rung.push(unsafe { ctl.submit_sqe(ring.as_mut_ptr(), ring[0]) }.unwrap());
            ctl.retire_completions(1);
        }
        assert_eq!(rung, [1, 2, 3, 0, 1, 2]);
        assert_eq!(ctl.qp.doorbell_tail(), 2);
        let off = ctl.doorbells.sq_tdbl_offset(0) as usize;
        assert_eq!(u32::from_ne_bytes(bar[off..off + 4].try_into().unwrap()), 2);
    }

    #[test]
    fn queue_depth_is_bounded_by_the_spec() {
        assert!(BamQueuePair::new(0, 1 << 16).is_some());
        assert!(BamQueuePair::new(0, 1 << 17).is_none());
        assert!(BamQueuePair::new(0, 3).is_none());
    }
}
