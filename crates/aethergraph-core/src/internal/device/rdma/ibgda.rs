//! K2.1 IBGDA — GPU-constructed mlx5 WQEs + doorbell record.
//!
//! Host publishes a doorbell record and BlueFlame/DBR mapping; GPU warps
//! (see `aether-stream` `kernels/ibgda`) write [`super::Mlx5RdmaReadWqe`]
//! into the QP ring and bump the doorbell record.

use super::Mlx5RdmaReadWqe;
use core::sync::atomic::{AtomicU16, Ordering};

/// mlx5 doorbell record: 8 bytes, `recv_db` then `send_db` (BE16 each in HW;
/// we keep host-endian counters and document the byte swap at the MMIO edge).
#[repr(C, align(8))]
#[derive(Debug, Default)]
pub struct Mlx5DoorbellRecord {
    pub recv_db: AtomicU16,
    pub send_db: AtomicU16,
}

impl Mlx5DoorbellRecord {
    /// Bump the send doorbell to `wqe_index` (mod 64K).
    pub fn ring_send(&self, wqe_index: u16) {
        self.send_db.store(wqe_index, Ordering::Release);
    }

    pub fn send_index(&self) -> u16 {
        self.send_db.load(Ordering::Acquire)
    }
}

/// Host-side IBGDA queue view used as the CPU oracle for GPU producers.
#[derive(Debug)]
pub struct IbgdaQueue {
    pub qpn: u32,
    pub depth: u16,
    pub next_wqe: AtomicU16,
    pub dbr: Mlx5DoorbellRecord,
}

impl IbgdaQueue {
    pub fn new(qpn: u32, depth: u16) -> Option<Self> {
        if depth == 0 || !depth.is_power_of_two() || qpn > 0x00ff_ffff {
            return None;
        }
        Some(Self {
            qpn,
            depth,
            next_wqe: AtomicU16::new(0),
            dbr: Mlx5DoorbellRecord::default(),
        })
    }

    /// CPU-path: build WQE, write into `ring[slot]`, ring DBR.
    ///
    /// # Safety
    /// `ring` must hold `depth` WQE slots of 64 bytes stride (3 segments + pad).
    pub unsafe fn post_rdma_read(
        &self,
        ring: *mut Mlx5RdmaReadWqe,
        local_address: u64,
        lkey: u32,
        byte_count: u32,
        remote_address: u64,
        rkey: u32,
    ) -> u16 {
        let idx = self.next_wqe.fetch_add(1, Ordering::Relaxed);
        let slot = (idx & (self.depth - 1)) as usize;
        let wqe = Mlx5RdmaReadWqe::new(
            self.qpn,
            idx,
            local_address,
            lkey,
            byte_count,
            remote_address,
            rkey,
        );
        // SAFETY: caller guarantees `ring` covers `depth` slots.
        let dst = unsafe { ring.add(slot) };
        // SAFETY: `dst` is within the caller-provided WQE ring.
        unsafe {
            core::ptr::write_volatile(dst, wqe);
        }
        core::sync::atomic::fence(Ordering::Release);
        self.dbr.ring_send(idx.wrapping_add(1));
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_rejects_bad_depth_or_qpn() {
        assert!(IbgdaQueue::new(1, 0).is_none());
        assert!(IbgdaQueue::new(1, 3).is_none());
        assert!(IbgdaQueue::new(0x0100_0000, 16).is_none());
        assert!(IbgdaQueue::new(7, 16).is_some());
    }

    #[test]
    fn post_advances_doorbell() {
        let q = IbgdaQueue::new(0x42, 8).unwrap();
        let mut ring = [Mlx5RdmaReadWqe::new(0x42, 0, 0, 0, 0, 0, 0); 8];
        let idx = unsafe { q.post_rdma_read(ring.as_mut_ptr(), 0x1000, 1, 64, 0x2000, 2) };
        assert_eq!(idx, 0);
        assert_eq!(q.dbr.send_index(), 1);
        assert_eq!(ring[0].qpn(), 0x42);
    }
}
