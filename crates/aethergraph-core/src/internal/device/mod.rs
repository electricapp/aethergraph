//! Pure-logic + device-path modules for the roadmap in `KERNELS.md`.
//!
//! - [`nvme`] — K1.* BaM / FDP / ZNS (+ zone-append WAL)
//! - [`rdma`] — K2.* IBGDA / DEVX / FlexIO
//! - [`pcie`] — K3.* p2pdma module client / CXL mbind / NVLink-C2C hints
//! - [`host`] — K4.* sched_ext loader / DAMON / provided buffers

#![allow(dead_code, unused_imports)]

pub mod host;
pub mod nvme;
pub mod pcie;
pub mod rdma;

pub use host::damon::{AccessFrequencyScheme, DamonConfig};
pub use host::damon_sysfs::DamonSysfs;
pub use host::provided_buffers::{IoUringBuf, ProvidedBufferRingSpec};
pub use host::sched_ext::{SchedExtLoader, SchedExtPolicy};
pub use nvme::bam::{BamController, BamQueuePair, NvmeDoorbellLayout};
pub use nvme::fdp::{FdpPlacementId, FdpReclaimSplit, NvmeDirective};
pub use nvme::sqe::{NvmeDataPointer, NvmeRwSqe};
pub use nvme::zns_append::ZoneAppendSqe;
pub use nvme::zone_wal::{ZoneAppendCompletion, ZoneAppendWal};
pub use pcie::cxl::CxlNumaBinding;
pub use pcie::nvlink_c2c::{CoherentAllocation, CoherentPlacementHint};
pub use pcie::p2pdma::{P2pdmaPath, P2pdmaPolicy};
pub use pcie::p2pdma_ioctl::{P2pdmaValidateResult, validate_p2pdma_path};
pub use rdma::devx::{DevxGpuEthPlan, MockDevxBackend};
pub use rdma::dpa::DpaParsePipeline;
pub use rdma::flexio::FlexIoHost;
pub use rdma::gpu_eth::{FlowSteeringRule, GpuRingPlacement};
pub use rdma::ibgda::{IbgdaError, IbgdaQueue};
pub use rdma::mlx5_wqe::Mlx5RdmaReadWqe;

/// Compatibility path used by `uring` before the nest.
pub mod provided_buffers {
    pub use super::host::provided_buffers::*;
}
