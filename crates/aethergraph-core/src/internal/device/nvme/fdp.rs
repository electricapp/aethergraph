//! K1.2: NVMe directive words for streams and Flexible Data Placement (FDP).
//!
//! NVMe encodes Directive Type (DTYPE) in CDW13 bits 7:0 and Directive
//! Specific (DSPEC) in bits 31:16. Bits 15:8 are reserved and always zero.

/// Directive type as carried in CDW13.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NvmeDirective {
    /// NVMe Streams directive.
    Streams = 0x01,
    /// NVMe Flexible Data Placement directive.
    FlexibleDataPlacement = 0x02,
}

/// Controller-assigned FDP placement identifier.
///
/// The identifier's interpretation is namespace-specific; this type prevents
/// accidentally treating it as a host-side reclaim-unit index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FdpPlacementId(pub u16);

/// Pack a directive type and its 16-bit directive-specific value into CDW13.
pub const fn pack_directive(dtype: NvmeDirective, dspec: u16) -> u32 {
    (u32::from_le_bytes([dtype as u8, 0, 0, 0])) | ((dspec as u32) << 16)
}

/// Return the DTYPE field from CDW13.
pub const fn directive_type(cdw13: u32) -> u8 {
    cdw13 as u8
}

/// Return the DSPEC field from CDW13.
pub const fn directive_specific(cdw13: u32) -> u16 {
    (cdw13 >> 16) as u16
}

/// Pack an FDP placement identifier as the directive-specific field.
pub const fn pack_fdp_placement(placement: FdpPlacementId) -> u32 {
    pack_directive(NvmeDirective::FlexibleDataPlacement, placement.0)
}

/// Hot adjacency vs cold features — conventional placement ids for writers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FdpReclaimSplit {
    pub hot_adjacency: FdpPlacementId,
    pub cold_features: FdpPlacementId,
}

impl FdpReclaimSplit {
    pub const fn standard() -> Self {
        Self {
            hot_adjacency: FdpPlacementId(1),
            cold_features: FdpPlacementId(2),
        }
    }
}

// Wired onto [`super::NvmeRwSqe::with_fdp_placement`].
// TODO(HARDWARE): On a real FDP-capable NVMe drive in the bare-metal rig,
// allocate reclaim units, issue writes with each placement identifier, and
// verify controller-reported reclaim-unit accounting and write amplification.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directive_packing_preserves_reserved_gap() {
        let cdw13 = pack_directive(NvmeDirective::Streams, 0xBEEF);
        assert_eq!(cdw13, 0xBEEF_0001);
        assert_eq!(directive_type(cdw13), NvmeDirective::Streams as u8);
        assert_eq!(directive_specific(cdw13), 0xBEEF);
        assert_eq!(cdw13 & 0x0000_ff00, 0);
    }

    #[test]
    fn fdp_placement_is_dspec() {
        let cdw13 = pack_fdp_placement(FdpPlacementId(0x1234));
        assert_eq!(cdw13, 0x1234_0002);
        assert_eq!(directive_specific(cdw13), 0x1234);
    }
}
