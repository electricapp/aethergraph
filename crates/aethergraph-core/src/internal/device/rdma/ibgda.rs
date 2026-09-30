//! K2.1 IBGDA — GPU-constructed mlx5 WQEs + doorbell record.
//!
//! Host publishes a doorbell record and BlueFlame/DBR mapping; GPU warps
//! (see `aether-stream` `kernels/ibgda`) write [`super::Mlx5RdmaReadWqe`]
//! into the QP ring at **64-byte basic-block** stride and publish them
//! through the doorbell record in claim order.

use super::Mlx5RdmaReadWqe;
use core::sync::atomic::{AtomicU16, AtomicU32, Ordering};

/// mlx5 WQE basic-block size (bytes). One RDMA READ WQE occupies one BB
/// (48-byte payload + 16-byte pad).
pub const MLX5_WQE_BB: usize = 64;

/// mlx5 doorbell record: two big-endian 32-bit words, receive counter then
/// send counter (`__be32 dbrec[2]`; the HCA reads `dbrec[1]` as
/// `htobe32(sq_cur_post & 0xffff)`).
#[repr(C, align(8))]
#[derive(Debug, Default)]
pub struct Mlx5DoorbellRecord {
    recv_db: AtomicU32,
    send_db: AtomicU32,
}

const _: () = assert!(core::mem::offset_of!(Mlx5DoorbellRecord, send_db) == 4);
const _: () = assert!(core::mem::size_of::<Mlx5DoorbellRecord>() == 8);

impl Mlx5DoorbellRecord {
    /// Publish `posted` — the count of WQEs the HCA may now fetch — as the
    /// send counter. Callers publish in claim order (see
    /// [`IbgdaQueue::post_rdma_read`]), so the value only ever moves past
    /// WQEs that are fully written.
    fn publish_send(&self, posted: u16) {
        self.send_db
            .store(u32::from(posted).to_be(), Ordering::Release);
    }

    /// Host-endian send counter (CPU oracle / tests).
    pub fn send_index(&self) -> u16 {
        u32::from_be(self.send_db.load(Ordering::Acquire)) as u16
    }

    /// The send word exactly as the HCA reads it.
    pub fn send_db_raw(&self) -> u32 {
        self.send_db.load(Ordering::Acquire)
    }

    /// Host-endian receive counter.
    pub fn recv_index(&self) -> u16 {
        u32::from_be(self.recv_db.load(Ordering::Acquire)) as u16
    }
}

/// Errors from the IBGDA host post path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IbgdaError {
    /// More than `depth` WQEs are outstanding; retire CQEs first.
    QueueFull,
}

/// Host-side IBGDA queue view used as the CPU oracle for GPU producers.
///
/// Producers claim indices concurrently but publish in claim order: a
/// producer rings the doorbell only once every lower index is written, so
/// the HCA never fetches a slot that is still stale.
#[derive(Debug)]
pub struct IbgdaQueue {
    pub qpn: u32,
    pub depth: u16,
    /// Next index to claim.
    pub next_wqe: AtomicU16,
    /// Count of WQEs written and published through the doorbell record.
    pub ready: AtomicU16,
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
            ready: AtomicU16::new(0),
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

    /// CPU-path: build WQE, write into `ring` at 64-byte BB stride, then
    /// publish it through the doorbell record once every earlier claim is
    /// published.
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
        // Publish in claim order: wait until every lower index is out, so
        // the doorbell never covers a slot another producer is still
        // writing. The producer holding `ready` has already claimed, so it
        // is running and will get there.
        while self.ready.load(Ordering::Acquire) != idx {
            core::hint::spin_loop();
        }
        let next = idx.wrapping_add(1);
        // Release: the WQE write above is visible before the doorbell.
        self.dbr.publish_send(next);
        self.ready.store(next, Ordering::Release);
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
        // SAFETY: `ring` is 8 work-queue BBs — the queue's depth — and outlives the call.
        let idx = unsafe { q.post_rdma_read(ring.as_mut_ptr(), 0x1000, 1, 64, 0x2000, 2) }.unwrap();
        assert_eq!(idx, 0);
        assert_eq!(q.dbr.send_index(), 1);
        // SAFETY: the post above initialized BB 0 as a valid WQE, and
        // `Mlx5RdmaReadWqe` is POD no larger than one BB.
        let wqe = unsafe { *(ring.as_ptr() as *const Mlx5RdmaReadWqe) };
        assert_eq!(wqe.qpn(), 0x42);
    }

    #[test]
    fn post_rejects_when_full_until_retire() {
        let q = IbgdaQueue::new(0x42, 4).unwrap();
        let mut ring = vec![0u8; 4 * MLX5_WQE_BB];
        for _ in 0..4 {
            // SAFETY: `ring` is 4 work-queue BBs — the queue's depth — and outlives the call.
            assert!(unsafe { q.post_rdma_read(ring.as_mut_ptr(), 0, 0, 0, 0, 0) }.is_ok());
        }
        assert_eq!(
            // SAFETY: as above.
            unsafe { q.post_rdma_read(ring.as_mut_ptr(), 0, 0, 0, 0, 0) },
            Err(IbgdaError::QueueFull)
        );
        q.retire(2);
        // SAFETY: as above.
        assert!(unsafe { q.post_rdma_read(ring.as_mut_ptr(), 0, 0, 0, 0, 0) }.is_ok());
    }

    /// The HCA reads `dbrec[1]` as a big-endian 32-bit counter.
    #[test]
    fn doorbell_record_is_two_big_endian_words() {
        let dbr = Mlx5DoorbellRecord::default();
        dbr.publish_send(0x1234);
        assert_eq!(dbr.send_db_raw(), 0x1234u32.to_be());
        assert_eq!(dbr.send_index(), 0x1234);
        assert_eq!(dbr.recv_index(), 0);
        // SAFETY: the record is 8 bytes of two u32 words.
        let words: [u32; 2] = unsafe { core::mem::transmute_copy(&dbr) };
        assert_eq!(words[1], 0x1234u32.to_be());
        assert_eq!(words[0], 0);
    }

    /// Concurrent producers: however their writes interleave, the doorbell
    /// ends at the total posted and never runs ahead of a written slot.
    #[test]
    fn concurrent_producers_publish_contiguously() {
        const THREADS: usize = 4;
        const PER: usize = 16;
        let q = IbgdaQueue::new(0x42, 128).unwrap();
        let ring = std::sync::Mutex::new(vec![0u8; 128 * MLX5_WQE_BB]);
        let base = ring.lock().unwrap().as_mut_ptr() as usize;
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let q = &q;
                s.spawn(move || {
                    for k in 0..PER {
                        let tag = (t * PER + k) as u64 + 1;
                        // SAFETY: the ring holds `depth` BBs and outlives the scope;
                        // producers write disjoint claimed slots.
                        unsafe { q.post_rdma_read(base as *mut u8, tag, 1, 8, tag, 2) }.unwrap();
                        // Whatever is published is fully written.
                        let published = q.dbr.send_index() as usize;
                        for slot in 0..published {
                            // SAFETY: published slots are written and no longer mutated.
                            let w =
                                unsafe { *((base + slot * MLX5_WQE_BB) as *const Mlx5RdmaReadWqe) };
                            assert_eq!(w.wqe_index() as usize, slot);
                            assert_ne!(u64::from_be_bytes(w.local_address), 0);
                        }
                    }
                });
            }
        });
        assert_eq!(q.dbr.send_index() as usize, THREADS * PER);
        assert_eq!(q.ready.load(Ordering::Acquire) as usize, THREADS * PER);
    }
}
