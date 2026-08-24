//! K2.3 BlueField-3 FlexIO / DPA host control plane.
//!
//! Stages a [`super::DpaParsePipeline`] for a DPA process. The on-device
//! program lives in `modules/aether_dpa/` (FlexIO C). This module owns the
//! host-side buffer layout and load contract.

use super::dpa::{DpaParsePipeline, EdgeField};

/// Host-visible staging region the DPA writes CSR deltas into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DpaStagingRegion {
    pub host_va: u64,
    pub bytes: u32,
}

impl DpaStagingRegion {
    pub fn new(host_va: u64, bytes: u32) -> Option<Self> {
        (host_va != 0 && bytes > 0).then_some(Self { host_va, bytes })
    }
}

/// Packed header the DPA program and host agree on (little-endian).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DpaDeltaHeader {
    pub magic: u32,
    pub n_edges: u32,
    pub flags: u32,
    pub reserved: u32,
}

pub const DPA_DELTA_MAGIC: u32 = 0x4145_4450; // "ADEP"
pub const DPA_FLAG_DEDUP: u32 = 1;

impl DpaDeltaHeader {
    pub fn new(n_edges: u32, deduplicate: bool) -> Self {
        Self {
            magic: DPA_DELTA_MAGIC,
            n_edges,
            flags: if deduplicate { DPA_FLAG_DEDUP } else { 0 },
            reserved: 0,
        }
    }

    pub fn valid(self) -> bool {
        self.magic == DPA_DELTA_MAGIC
    }
}

/// Field order encoded as a bit mask for the DPA program.
pub fn field_mask(fields: &[EdgeField]) -> u32 {
    let mut m = 0u32;
    for f in fields {
        m |= 1u32
            << match f {
                EdgeField::SourceNode => 0,
                EdgeField::DestinationNode => 1,
                EdgeField::EdgeType => 2,
                EdgeField::EventTimestamp => 3,
            };
    }
    m
}

/// Host handle that would load `aether_dpa` onto a BlueField-3.
#[derive(Debug)]
pub struct FlexIoHost {
    pub pipeline: DpaParsePipeline,
    pub staging: DpaStagingRegion,
    pub field_mask: u32,
}

impl FlexIoHost {
    pub fn plan(pipeline: DpaParsePipeline, staging: DpaStagingRegion) -> Option<Self> {
        if staging.bytes < pipeline.csr_delta_bytes {
            return None;
        }
        let mask = field_mask(&pipeline.fields);
        Some(Self {
            field_mask: mask,
            pipeline,
            staging,
        })
    }

    /// Serialize the control block the DPA entry reads at attach time.
    pub fn control_block(&self) -> [u64; 4] {
        [
            self.staging.host_va,
            u64::from(self.staging.bytes),
            u64::from(self.field_mask)
                | (u64::from(self.pipeline.deduplicate) << 32)
                | (u64::from(self.pipeline.csr_delta_bytes) << 33),
            u64::from(DPA_DELTA_MAGIC),
        ]
    }

    /// Load onto BlueField — returns Unsupported until FlexIO libs are linked.
    pub fn attach(&self) -> std::io::Result<()> {
        let _ = self.control_block();
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "FlexIoHost::attach needs BlueField-3 FlexIO runtime \
             (modules/aether_dpa + DOCA FlexIO)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_rejects_undersized_staging() {
        let pipe = DpaParsePipeline::new(vec![EdgeField::SourceNode], true, 4096).unwrap();
        let small = DpaStagingRegion::new(0x1000, 100).unwrap();
        assert!(FlexIoHost::plan(pipe.clone(), small).is_none());
        let ok = DpaStagingRegion::new(0x1000, 8192).unwrap();
        let host = FlexIoHost::plan(pipe, ok).unwrap();
        assert!(host.field_mask & 1 != 0);
        assert_eq!(host.control_block()[3] as u32, DPA_DELTA_MAGIC);
    }
}
