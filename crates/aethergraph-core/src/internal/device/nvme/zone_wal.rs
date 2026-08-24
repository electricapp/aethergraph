//! K1.3 zone-append WAL: device-assigned LBAs replace host offset counters.
//!
//! Writers build [`super::ZoneAppendSqe`] commands against a fixed zone start.
//! Completions carry the LBA the controller assigned; [`ZoneAppendWal`] records
//! those LBAs without a shared bump allocator.

use super::ZoneAppendSqe;
use core::sync::atomic::{AtomicU16, AtomicU64, Ordering};

/// One completed append: device LBA + writer-visible cookie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZoneAppendCompletion {
    pub command_id: u16,
    /// LBA returned in the CQ entry (controller-assigned).
    pub assigned_lba: u64,
    pub nlb: u16,
}

/// Lockless multi-writer WAL state for a single open zone.
///
/// The only shared mutable state is a command-id counter and a completion
/// cursor. Write offsets come from the drive.
#[derive(Debug)]
pub struct ZoneAppendWal {
    pub nsid: u32,
    pub zone_start_lba: u64,
    next_cid: AtomicU16,
    /// Highest exclusive end LBA observed from completions (telemetry).
    high_water: AtomicU64,
}

impl ZoneAppendWal {
    /// Open a WAL on `zone_start_lba` of `nsid`.
    pub const fn new(nsid: u32, zone_start_lba: u64) -> Self {
        Self {
            nsid,
            zone_start_lba,
            next_cid: AtomicU16::new(1),
            high_water: AtomicU64::new(zone_start_lba),
        }
    }

    /// Allocate a command id and build a Zone Append SQE for `nlb` blocks.
    pub fn prepare_append(&self, nlb: u16, prp1: u64, prp2: u64) -> (u16, ZoneAppendSqe) {
        let cid = self.next_cid.fetch_add(1, Ordering::Relaxed);
        let sqe = ZoneAppendSqe::new(self.nsid, cid, self.zone_start_lba, nlb, prp1, prp2);
        (cid, sqe)
    }

    /// Record a controller completion. Updates the high-water mark.
    pub fn complete(&self, cqe: ZoneAppendCompletion) {
        let end = cqe.assigned_lba.saturating_add(u64::from(cqe.nlb) + 1);
        let mut cur = self.high_water.load(Ordering::Relaxed);
        while end > cur {
            match self.high_water.compare_exchange_weak(
                cur,
                end,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => cur = v,
            }
        }
    }

    /// Exclusive end of the highest completed append (for readers).
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
}
