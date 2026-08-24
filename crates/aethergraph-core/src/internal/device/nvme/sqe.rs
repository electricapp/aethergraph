//! K1.1: NVMe I/O command SQEs (NVMe Base Specification, Figure 68).
//!
//! The command is byte-addressable because a GPU or a doorbell path writes it
//! as DMA-visible bytes. `NvmeRwSqe` is exactly one 64-byte NVMe SQ entry.

/// NVMe READ opcode.
pub const NVME_OPC_READ: u8 = 0x02;
/// NVMe WRITE opcode.
pub const NVME_OPC_WRITE: u8 = 0x01;
const PSDT_SGL: u8 = 0b01 << 6;

/// The command data pointer selected for an NVMe READ or WRITE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvmeDataPointer {
    /// Physical Region Page addresses.
    Prp { prp1: u64, prp2: u64 },
    /// A pre-encoded, 16-byte SGL descriptor in NVMe wire order.
    Sgl([u8; 16]),
}

/// A packed 64-byte NVMe READ/WRITE submission queue entry.
///
/// Multibyte members are explicitly little-endian byte arrays so this packed
/// layout can be inspected without creating unaligned references.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvmeRwSqe {
    pub opcode: u8,
    pub flags: u8,
    pub command_id: [u8; 2],
    pub nsid: [u8; 4],
    pub cdw2: [u8; 4],
    pub cdw3: [u8; 4],
    pub metadata: [u8; 8],
    /// PRP1 or the first eight bytes of an SGL descriptor.
    pub dptr1: [u8; 8],
    /// PRP2 or the final eight bytes of an SGL descriptor.
    pub dptr2: [u8; 8],
    /// CDW10-15.
    pub cdw10_15: [u8; 24],
}

const _: () = assert!(core::mem::size_of::<NvmeRwSqe>() == 64);

impl NvmeRwSqe {
    /// Build an NVMe READ command. `nlb` is the spec's zero-based block count.
    pub fn read(nsid: u32, command_id: u16, slba: u64, nlb: u16, data: NvmeDataPointer) -> Self {
        Self::rw(NVME_OPC_READ, nsid, command_id, slba, nlb, data)
    }

    /// Build an NVMe WRITE command. `nlb` is the spec's zero-based block count.
    pub fn write(nsid: u32, command_id: u16, slba: u64, nlb: u16, data: NvmeDataPointer) -> Self {
        Self::rw(NVME_OPC_WRITE, nsid, command_id, slba, nlb, data)
    }

    fn rw(
        opcode: u8,
        nsid: u32,
        command_id: u16,
        slba: u64,
        nlb: u16,
        data: NvmeDataPointer,
    ) -> Self {
        let (flags, dptr1, dptr2) = match data {
            NvmeDataPointer::Prp { prp1, prp2 } => (0, prp1.to_le_bytes(), prp2.to_le_bytes()),
            NvmeDataPointer::Sgl(sgl) => {
                let mut first = [0; 8];
                let mut second = [0; 8];
                first.copy_from_slice(&sgl[..8]);
                second.copy_from_slice(&sgl[8..]);
                (PSDT_SGL, first, second)
            }
        };
        let mut cdw10_15 = [0; 24];
        cdw10_15[..8].copy_from_slice(&slba.to_le_bytes());
        cdw10_15[8..10].copy_from_slice(&nlb.to_le_bytes());
        Self {
            opcode,
            flags,
            command_id: command_id.to_le_bytes(),
            nsid: nsid.to_le_bytes(),
            cdw2: [0; 4],
            cdw3: [0; 4],
            metadata: [0; 8],
            dptr1,
            dptr2,
            cdw10_15,
        }
    }

    /// Starting LBA encoded in CDW10/11.
    pub fn slba(self) -> u64 {
        u64::from_le_bytes(self.cdw10_15[..8].try_into().expect("fixed layout"))
    }

    /// Zero-based logical block count encoded in CDW12.
    pub fn nlb(self) -> u16 {
        u16::from_le_bytes(self.cdw10_15[8..10].try_into().expect("fixed layout"))
    }

    /// Set CDW13 directive type + DSPEC (streams / FDP).
    #[must_use]
    pub fn with_directive(mut self, dtype: super::NvmeDirective, dspec: u16) -> Self {
        let packed = super::fdp::pack_directive(dtype, dspec).to_le_bytes();
        self.cdw10_15[12..16].copy_from_slice(&packed);
        self
    }

    /// Convenience: FDP placement identifier in CDW13.
    #[must_use]
    pub fn with_fdp_placement(self, placement: super::FdpPlacementId) -> Self {
        self.with_directive(super::NvmeDirective::FlexibleDataPlacement, placement.0)
    }

    /// CDW13 as packed u32.
    pub fn cdw13(self) -> u32 {
        u32::from_le_bytes(self.cdw10_15[12..16].try_into().expect("fixed layout"))
    }
}

// BaM path: [`super::bam::BamController`]. FDP: [`Self::with_fdp_placement`].
// TODO(HARDWARE): On the bare-metal ConnectX + sacrificial-NVMe rig, map BAR0
// and VRAM SQ/CQ rings, ring the controller doorbell, and verify BaM READ/WRITE
// completion and data integrity without the host NVMe driver bound.

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    #[test]
    fn rw_sqe_has_nvme_base_spec_offsets() {
        assert_eq!(size_of::<NvmeRwSqe>(), 64);
        assert_eq!(offset_of!(NvmeRwSqe, opcode), 0);
        assert_eq!(offset_of!(NvmeRwSqe, nsid), 4);
        assert_eq!(offset_of!(NvmeRwSqe, metadata), 16);
        assert_eq!(offset_of!(NvmeRwSqe, dptr1), 24);
        assert_eq!(offset_of!(NvmeRwSqe, dptr2), 32);
        assert_eq!(offset_of!(NvmeRwSqe, cdw10_15), 40);
    }

    #[test]
    fn read_builder_encodes_prps_lba_and_zero_based_nlb() {
        let sqe = NvmeRwSqe::read(
            7,
            0x1234,
            0x1234_5678_9abc_def0,
            15,
            NvmeDataPointer::Prp {
                prp1: 0x1000,
                prp2: 0x2000,
            },
        );
        assert_eq!(sqe.opcode, NVME_OPC_READ);
        assert_eq!(u32::from_le_bytes(sqe.nsid), 7);
        assert_eq!(u16::from_le_bytes(sqe.command_id), 0x1234);
        assert_eq!(sqe.slba(), 0x1234_5678_9abc_def0);
        assert_eq!(sqe.nlb(), 15);
        assert_eq!(u64::from_le_bytes(sqe.dptr1), 0x1000);
    }

    #[test]
    fn write_with_fdp_sets_cdw13() {
        use super::super::NvmeDirective;
        use super::super::fdp::{FdpPlacementId, directive_specific, directive_type};
        let sqe = NvmeRwSqe::write(1, 2, 3, 0, NvmeDataPointer::Prp { prp1: 0, prp2: 0 })
            .with_fdp_placement(FdpPlacementId(0xAB));
        assert_eq!(
            directive_type(sqe.cdw13()),
            NvmeDirective::FlexibleDataPlacement as u8
        );
        assert_eq!(directive_specific(sqe.cdw13()), 0xAB);
    }

    #[test]
    fn write_builder_selects_sgl_pointer_type() {
        let sgl = [0xA5; 16];
        let sqe = NvmeRwSqe::write(1, 2, 3, 0, NvmeDataPointer::Sgl(sgl));
        assert_eq!(sqe.opcode, NVME_OPC_WRITE);
        assert_eq!(sqe.flags, PSDT_SGL);
        assert_eq!(sqe.dptr1, [0xA5; 8]);
        assert_eq!(sqe.dptr2, [0xA5; 8]);
    }
}
