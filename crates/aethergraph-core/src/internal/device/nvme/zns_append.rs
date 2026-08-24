//! K1.3: NVMe Zoned Namespace Zone Append command layout.
//!
//! Zone Append uses the ordinary 64-byte I/O command format. CDW10/11 carry
//! the zone's start LBA; CDW12 low 16 bits carry the zero-based data size.

/// NVMe Zoned Namespace Management Send: Zone Append opcode.
pub const NVME_OPC_ZONE_APPEND: u8 = 0x7d;

/// A packed 64-byte Zone Append SQE.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZoneAppendSqe {
    pub opcode: u8,
    pub flags: u8,
    pub command_id: [u8; 2],
    pub nsid: [u8; 4],
    pub reserved: [u8; 16],
    pub prp1: [u8; 8],
    pub prp2: [u8; 8],
    pub cdw10_15: [u8; 24],
}

const _: () = assert!(core::mem::size_of::<ZoneAppendSqe>() == 64);

impl ZoneAppendSqe {
    /// Build a Zone Append. `nlb` is zero-based: zero transfers one LBA.
    pub fn new(
        nsid: u32,
        command_id: u16,
        zone_start_lba: u64,
        nlb: u16,
        prp1: u64,
        prp2: u64,
    ) -> Self {
        let mut cdw10_15 = [0; 24];
        cdw10_15[..8].copy_from_slice(&zone_start_lba.to_le_bytes());
        cdw10_15[8..10].copy_from_slice(&nlb.to_le_bytes());
        Self {
            opcode: NVME_OPC_ZONE_APPEND,
            flags: 0,
            command_id: command_id.to_le_bytes(),
            nsid: nsid.to_le_bytes(),
            reserved: [0; 16],
            prp1: prp1.to_le_bytes(),
            prp2: prp2.to_le_bytes(),
            cdw10_15,
        }
    }

    /// The zone start LBA, not an arbitrary write offset.
    pub fn zone_start_lba(self) -> u64 {
        u64::from_le_bytes(self.cdw10_15[..8].try_into().expect("fixed layout"))
    }

    /// The command's zero-based number of logical blocks.
    pub fn nlb(self) -> u16 {
        u16::from_le_bytes(self.cdw10_15[8..10].try_into().expect("fixed layout"))
    }
}

// WAL path: [`super::zone_wal::ZoneAppendWal`].
// TODO(HARDWARE): Submit concurrent Zone Append commands on a ZNS drive or
// QEMU ZNS namespace and verify each completion's returned LBA is allocated by
// the device, monotonically advances within its zone, and matches written data.

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    #[test]
    fn zone_append_layout_matches_io_command() {
        assert_eq!(size_of::<ZoneAppendSqe>(), 64);
        assert_eq!(offset_of!(ZoneAppendSqe, prp1), 24);
        assert_eq!(offset_of!(ZoneAppendSqe, prp2), 32);
        assert_eq!(offset_of!(ZoneAppendSqe, cdw10_15), 40);
    }

    #[test]
    fn zone_append_encodes_zone_and_data_size() {
        let sqe = ZoneAppendSqe::new(9, 4, 0x1_0000_0200, 31, 0x4000, 0x5000);
        assert_eq!(sqe.opcode, NVME_OPC_ZONE_APPEND);
        assert_eq!(u32::from_le_bytes(sqe.nsid), 9);
        assert_eq!(sqe.zone_start_lba(), 0x1_0000_0200);
        assert_eq!(sqe.nlb(), 31);
    }
}
