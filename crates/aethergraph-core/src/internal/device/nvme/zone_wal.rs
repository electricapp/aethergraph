//! K1.3 zone-append WAL: device-assigned LBAs replace host offset counters.
//!
//! Writers build [`super::ZoneAppendSqe`] commands against a fixed zone start.
//! Completions carry the LBA the controller assigned; [`ZoneAppendWal`] records
//! those LBAs without a shared bump allocator.
//!
//! `durable_high_water` advances only over a **contiguous** prefix of completed
//! ranges starting at the previous durable cursor, so out-of-order CQEs cannot
//! expose unfinished LBAs to readers.

use super::ZoneAppendSqe;
use core::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use parking_lot::Mutex;
use std::collections::BTreeMap;

/// One completed append: device LBA + writer-visible cookie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZoneAppendCompletion {
    pub command_id: u16,
    /// LBA returned in the CQ entry (controller-assigned).
    pub assigned_lba: u64,
    pub nlb: u16,
}

/// Multi-writer WAL state for a single open zone.
///
/// Command ids are allocated atomically (no shared bump of LBA). Durable
/// high-water merges completed ranges under a short mutex so OOO CQEs stay
/// correct.
#[derive(Debug)]
pub struct ZoneAppendWal {
    pub nsid: u32,
    pub zone_start_lba: u64,
    next_cid: AtomicU16,
    high_water: AtomicU64,
    /// Pending completed ranges keyed by start LBA → exclusive end.
    pending: Mutex<BTreeMap<u64, u64>>,
}

impl ZoneAppendWal {
    /// Open a WAL on `zone_start_lba` of `nsid`.
    pub fn new(nsid: u32, zone_start_lba: u64) -> Self {
        Self {
            nsid,
            zone_start_lba,
            next_cid: AtomicU16::new(1),
            high_water: AtomicU64::new(zone_start_lba),
            pending: Mutex::new(BTreeMap::new()),
        }
    }

    /// Allocate a command id and build a Zone Append SQE for `nlb` blocks.
    pub fn prepare_append(&self, nlb: u16, prp1: u64, prp2: u64) -> (u16, ZoneAppendSqe) {
        let cid = self.next_cid.fetch_add(1, Ordering::Relaxed);
        let sqe = ZoneAppendSqe::new(self.nsid, cid, self.zone_start_lba, nlb, prp1, prp2);
        (cid, sqe)
    }

    /// Record a controller completion. Advances durable high-water only for
    /// the contiguous prefix of completed ranges.
    pub fn complete(&self, cqe: ZoneAppendCompletion) {
        let start = cqe.assigned_lba;
        let end = cqe.assigned_lba.saturating_add(u64::from(cqe.nlb) + 1);
        let mut pending = self.pending.lock();
        pending.insert(start, end);
        let mut hw = self.high_water.load(Ordering::Relaxed);
        while let Some(next_end) = pending.remove(&hw) {
            hw = next_end;
            self.high_water.store(hw, Ordering::Release);
        }
    }

    /// Exclusive end of the contiguous durable prefix (safe for readers).
    pub fn durable_high_water(&self) -> u64 {
        self.high_water.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_uses_zone_start_not_host_offset() {
        let wal = ZoneAppendWal::new(3, 0x1_0000);
        let (cid, sqe) = wal.prepare_append(7, 0x1000, 0);
        assert_eq!(cid, 1);
        assert_eq!(sqe.zone_start_lba(), 0x1_0000);
        assert_eq!(sqe.nlb(), 7);
    }

    #[test]
    fn completions_advance_high_water_without_shared_offset() {
        let wal = ZoneAppendWal::new(1, 100);
        wal.complete(ZoneAppendCompletion {
            command_id: 1,
            assigned_lba: 100,
            nlb: 3, // 4 blocks → end 104
        });
        assert_eq!(wal.durable_high_water(), 104);
        wal.complete(ZoneAppendCompletion {
            command_id: 2,
            assigned_lba: 104,
            nlb: 0,
        });
        assert_eq!(wal.durable_high_water(), 105);
    }

    #[test]
    fn out_of_order_completion_does_not_skip_gap() {
        let wal = ZoneAppendWal::new(1, 100);
        // Later LBA completes first.
        wal.complete(ZoneAppendCompletion {
            command_id: 2,
            assigned_lba: 104,
            nlb: 0,
        });
        assert_eq!(wal.durable_high_water(), 100);
        wal.complete(ZoneAppendCompletion {
            command_id: 1,
            assigned_lba: 100,
            nlb: 3,
        });
        assert_eq!(wal.durable_high_water(), 105);
    }
}
