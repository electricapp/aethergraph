//! K2.2: descriptors for GPU-resident DEVX QP/CQ rings and flow steering.
//!
//! These are intent descriptions, not DEVX handles. A future Linux/ConnectX
//! adapter translates them into DEVX object creation and mlx5 flow tables.

/// A GPU virtual-address range used for a QP or CQ ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuRingPlacement {
    /// GPU virtual address of the first ring byte.
    pub gpu_va: u64,
    /// Ring allocation size; must be a non-zero multiple of `entry_bytes`.
    pub bytes: u32,
    /// Hardware entry stride.
    pub entry_bytes: u16,
}

impl GpuRingPlacement {
    /// Build a placement descriptor when the ring has an integral entry count.
    pub fn new(gpu_va: u64, bytes: u32, entry_bytes: u16) -> Option<Self> {
        (gpu_va != 0
            && entry_bytes != 0
            && bytes != 0
            && bytes.is_multiple_of(u32::from(entry_bytes)))
        .then_some(Self {
            gpu_va,
            bytes,
            entry_bytes,
        })
    }

    /// Number of hardware entries in the ring.
    pub const fn entries(self) -> u32 {
        self.bytes / self.entry_bytes as u32
    }
}

/// Five-tuple match fields for an mlx5 flow-steering rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FlowMatch {
    pub ethernet_type: Option<u16>,
    pub ip_protocol: Option<u8>,
    pub udp_dst_port: Option<u16>,
}

/// A steering rule that directs matching traffic to a GPU-backed receive QP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlowSteeringRule {
    pub priority: u16,
    pub matcher: FlowMatch,
    pub destination_qp: u32,
}

impl FlowSteeringRule {
    /// Construct a rule only for a real QP number.
    pub const fn new(priority: u16, matcher: FlowMatch, destination_qp: u32) -> Option<Self> {
        if destination_qp == 0 {
            None
        } else {
            Some(Self {
                priority,
                matcher,
                destination_qp,
            })
        }
    }
}

// DEVX path: [`super::devx::DevxGpuEthPlan`] + [`super::devx::Mlx5DevxBackend`].
// TODO(HARDWARE): On the bare-metal ConnectX rig, create DEVX QP/CQ objects
// backed by CUDA memory and validate DOCA GPUNetIO-equivalent packet delivery,
// doorbell records, and flow steering directly into VRAM.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_ring_requires_integral_nonzero_entries() {
        let ring = GpuRingPlacement::new(0x1000, 4096, 64).expect("valid ring");
        assert_eq!(ring.entries(), 64);
        assert_eq!(GpuRingPlacement::new(0, 4096, 64), None);
        assert_eq!(GpuRingPlacement::new(0x1000, 4095, 64), None);
    }

    #[test]
    fn flow_rule_rejects_null_qp() {
        let matcher = FlowMatch {
            ethernet_type: Some(0x0800),
            ip_protocol: Some(17),
            udp_dst_port: Some(9000),
        };
        assert!(FlowSteeringRule::new(1, matcher, 0).is_none());
        assert_eq!(
            FlowSteeringRule::new(1, matcher, 42)
                .unwrap()
                .destination_qp,
            42
        );
    }
}
