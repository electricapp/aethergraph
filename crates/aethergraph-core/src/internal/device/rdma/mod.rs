//! K2.* RDMA / GPU-ethernet / BlueField DPA.

pub mod devx;
pub mod dpa;
pub mod flexio;
pub mod gpu_eth;
pub mod ibgda;
pub mod mlx5_wqe;

pub use devx::{
    DevxGpuEthBackend, DevxGpuEthPlan, DevxSessionId, Mlx5DevxBackend, MockDevxBackend,
};
pub use dpa::DpaParsePipeline;
pub use flexio::{DpaStagingRegion, FlexIoHost};
pub use gpu_eth::{FlowSteeringRule, GpuRingPlacement};
pub use ibgda::{IbgdaError, IbgdaQueue, MLX5_WQE_BB, Mlx5DoorbellRecord};
pub use mlx5_wqe::Mlx5RdmaReadWqe;
