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
    /// `depth` must be a non-zero power of two (NVMe queue size = depth).
    pub fn new(qid: u16, depth: u32) -> Option<Self> {
        if depth == 0 || !depth.is_power_of_two() {
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

    /// Masked doorbell value (NVMe uses the raw tail counter modulo 2^16
    /// for queue sizes ≤ 64K; we keep the full counter and let callers
    /// truncate).
    pub fn doorbell_tail(&self) -> u16 {
        (self.sq_tail.load(Ordering::Acquire) & 0xffff) as u16
    }
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
        // Ensure the SQE is visible to the controller before the doorbell.
        core::sync::atomic::fence(Ordering::Release);
        let tail = prev.wrapping_add(1);
        self.ring_sq_doorbell((tail & 0xffff) as u16)?;
        Ok((tail & 0xffff) as u16)
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
}
