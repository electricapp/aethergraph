//! K3.* peer-DMA / CXL / coherent-placement.

pub mod cxl;
pub mod nvlink_c2c;
pub mod p2pdma;
pub mod p2pdma_ioctl;

pub use cxl::{CxlBindError, CxlNumaBinding};
pub use nvlink_c2c::{CoherentAllocation, CoherentPlacementHint};
pub use p2pdma::{P2pdmaPath, P2pdmaPolicy};
pub use p2pdma_ioctl::{P2pdmaValidateResult, validate_p2pdma_path, validate_p2pdma_path_default};
