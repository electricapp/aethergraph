//! K3.1: policy for an out-of-tree GPU-VRAM peer-DMA validation ioctl.
//!
//! The future module contract is: userspace passes producer and consumer PCI
//! BDFs plus a dma-buf; the module imports the dma-buf, calls
//! `pci_p2pdma_distance()`, and returns one of [`P2pdmaPath`] with the
//! peer-bus address only for [`P2pdmaPath::Ok`]. Consumers must not infer
//! topology from BDF strings themselves.

/// Outcome of validating a peer-DMA path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum P2pdmaPath {
    /// The path is usable; `distance` is the kernel topology cost.
    Ok { distance: u32 },
    /// Devices are reachable but exceed the caller's policy distance.
    TooFar { distance: u32, maximum: u32 },
    /// No IOMMU domain permits this peer transaction.
    NoIommu,
    /// PCIe ACS redirects or blocks peer transactions.
    AcsRedirected,
    /// Either device does not support the required peer-DMA capability.
    Unsupported,
}

/// Userspace policy applied to a successful `pci_p2pdma_distance()` query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct P2pdmaPolicy {
    pub maximum_distance: u32,
    pub require_iommu: bool,
}

impl P2pdmaPolicy {
    /// Classify a kernel-provided topology distance and platform constraints.
    pub const fn classify(
        self,
        distance: Option<u32>,
        iommu_present: bool,
        acs_redirected: bool,
    ) -> P2pdmaPath {
        if self.require_iommu && !iommu_present {
            P2pdmaPath::NoIommu
        } else if acs_redirected {
            P2pdmaPath::AcsRedirected
        } else if let Some(distance) = distance {
            if distance <= self.maximum_distance {
                P2pdmaPath::Ok { distance }
            } else {
                P2pdmaPath::TooFar {
                    distance,
                    maximum: self.maximum_distance,
                }
            }
        } else {
            P2pdmaPath::Unsupported
        }
    }
}

// Module: `modules/aether_p2pdma/`. Userspace: [`super::p2pdma_ioctl`].
// TODO(HARDWARE): Crash-iterate the out-of-tree module under virtme-ng/QEMU
// before using bare metal; then validate actual ACS routing and GPU dma-buf to
// NVMe peer-DMA on the bare-metal ConnectX + spare-NVMe rig.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_exposes_topology_failure_reason() {
        let policy = P2pdmaPolicy {
            maximum_distance: 2,
            require_iommu: true,
        };
        assert_eq!(
            policy.classify(Some(1), true, false),
            P2pdmaPath::Ok { distance: 1 }
        );
        assert_eq!(
            policy.classify(Some(3), true, false),
            P2pdmaPath::TooFar {
                distance: 3,
                maximum: 2
            }
        );
        assert_eq!(policy.classify(Some(1), false, false), P2pdmaPath::NoIommu);
        assert_eq!(
            policy.classify(Some(1), true, true),
            P2pdmaPath::AcsRedirected
        );
    }
}
