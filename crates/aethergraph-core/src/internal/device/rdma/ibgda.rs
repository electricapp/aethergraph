//! K2.1 IBGDA — GPU-constructed mlx5 WQEs + doorbell record.
//!
//! Host publishes a doorbell record and BlueFlame/DBR mapping; GPU warps
//! (see `aether-stream` `kernels/ibgda`) write [`super::Mlx5RdmaReadWqe`]
//! into the QP ring at **64-byte basic-block** stride and bump the doorbell
//! record monotonically.

use super::Mlx5RdmaReadWqe;
use core::sync::atomic::{AtomicU16, Ordering};

/// mlx5 WQE basic-block size (bytes). One RDMA READ WQE occupies one BB
/// (48-byte payload + 16-byte pad).
pub const MLX5_WQE_BB: usize = 64;

/// mlx5 doorbell record: 8 bytes, `recv_db` then `send_db`.
///
/// Logical indices are stored **host-endian** in the atomics for CPU oracles.
/// Use [`Mlx5DoorbellRecord::send_db_be`] when writing a HCA-mapped DBR /
/// BlueFlame record (device expects BE16).
#[repr(C, align(8))]
#[derive(Debug, Default)]
pub struct Mlx5DoorbellRecord {
    pub recv_db: AtomicU16,
    pub send_db: AtomicU16,
}

impl Mlx5DoorbellRecord {
    /// Monotonically advance the send doorbell to at least `wqe_index`.
    ///
    /// Multi-producer safe: a stale lower index never overwrites a newer one
    /// (wrapping half-range compare).
    pub fn ring_send_monotonic(&self, wqe_index: u16) {
        let mut cur = self.send_db.load(Ordering::Acquire);
        loop {
            let ahead = wqe_index.wrapping_sub(cur);
            // Equal or more than half the u16 space behind → do not store.
            if ahead == 0 || ahead > 0x8000 {
                return;
            }
            match self.send_db.compare_exchange_weak(
                cur,
                wqe_index,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(v) => cur = v,
            }
        }
    }

    /// Host-endian send index (CPU oracle / tests).
    pub fn send_index(&self) -> u16 {
        self.send_db.load(Ordering::Acquire)
    }

    /// Big-endian send index for a device-mapped DBR dword.
    pub fn send_db_be(&self) -> u16 {
        self.send_index().to_be()
    }
}

/// Errors from the IBGDA host post path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IbgdaError {
    /// More than `depth` WQEs are outstanding; retire CQEs first.
    QueueFull,
}

/// Host-side IBGDA queue view used as the CPU oracle for GPU producers.
#[derive(Debug)]
pub struct IbgdaQueue {
    pub qpn: u32,
    pub depth: u16,
    pub next_wqe: AtomicU16,
    /// Lowest incomplete WQE index (advanced by [`Self::retire`]).
    pub cq_head: AtomicU16,
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
            cq_head: AtomicU16::new(0),
            dbr: Mlx5DoorbellRecord::default(),
        })
    }

    /// In-flight posts: `next_wqe - cq_head`.
    pub fn in_flight(&self) -> u16 {
        self.next_wqe
            .load(Ordering::Acquire)
            .wrapping_sub(self.cq_head.load(Ordering::Acquire))
    }

    /// Retire `n` completed WQEs so slots can be reused.
    pub fn retire(&self, n: u16) {
        self.cq_head.fetch_add(n, Ordering::Release);
    }

    /// Claim the next WQE index, or [`IbgdaError::QueueFull`].
    fn try_claim_wqe(&self) -> Result<u16, IbgdaError> {
        loop {
            let idx = self.next_wqe.load(Ordering::Acquire);
            let head = self.cq_head.load(Ordering::Acquire);
            if idx.wrapping_sub(head) >= self.depth {
                return Err(IbgdaError::QueueFull);
            }
            match self.next_wqe.compare_exchange_weak(
                idx,
                idx.wrapping_add(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(idx),
                Err(_) => continue,
            }
        }
    }

    /// CPU-path: build WQE, write into `ring` at 64-byte BB stride, ring DBR.
    ///
    /// # Safety
    /// `ring` must be a byte buffer of at least `depth * MLX5_WQE_BB` bytes.
    pub unsafe fn post_rdma_read(
        &self,
        ring: *mut u8,
        local_address: u64,
        lkey: u32,
        byte_count: u32,
        remote_address: u64,
        rkey: u32,
    ) -> Result<u16, IbgdaError> {
        let idx = self.try_claim_wqe()?;
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
        // SAFETY: caller sized `ring` for `depth` 64-byte BBs.
        let dst = unsafe { ring.add(slot * MLX5_WQE_BB) as *mut Mlx5RdmaReadWqe };
        // SAFETY: `dst` is the start of a BB; WQE is 48 bytes within it.
        unsafe {
            core::ptr::write_volatile(dst, wqe);
        }
        core::sync::atomic::fence(Ordering::Release);
        self.dbr.ring_send_monotonic(idx.wrapping_add(1));
        Ok(idx)
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
    fn post_advances_doorbell_at_64b_stride() {
        let q = IbgdaQueue::new(0x42, 8).unwrap();
        let mut ring = vec![0u8; 8 * MLX5_WQE_BB];
        let idx = unsafe { q.post_rdma_read(ring.as_mut_ptr(), 0x1000, 1, 64, 0x2000, 2) }.unwrap();
        assert_eq!(idx, 0);
        assert_eq!(q.dbr.send_index(), 1);
        let wqe = unsafe { *(ring.as_ptr() as *const Mlx5RdmaReadWqe) };
        assert_eq!(wqe.qpn(), 0x42);
    }

    #[test]
    fn post_rejects_when_full_until_retire() {
        let q = IbgdaQueue::new(0x42, 4).unwrap();
        let mut ring = vec![0u8; 4 * MLX5_WQE_BB];
        for _ in 0..4 {
            assert!(unsafe { q.post_rdma_read(ring.as_mut_ptr(), 0, 0, 0, 0, 0) }.is_ok());
        }
        assert_eq!(
            unsafe { q.post_rdma_read(ring.as_mut_ptr(), 0, 0, 0, 0, 0) },
            Err(IbgdaError::QueueFull)
        );
        q.retire(2);
        assert!(unsafe { q.post_rdma_read(ring.as_mut_ptr(), 0, 0, 0, 0, 0) }.is_ok());
    }

    #[test]
    fn doorbell_monotonic_ignores_stale_index() {
        let dbr = Mlx5DoorbellRecord::default();
        dbr.ring_send_monotonic(5);
        dbr.ring_send_monotonic(3);
        assert_eq!(dbr.send_index(), 5);
        dbr.ring_send_monotonic(7);
        assert_eq!(dbr.send_index(), 7);
        assert_eq!(dbr.send_db_be(), 7u16.to_be());
    }
}
