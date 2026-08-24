//! K1.* NVMe command layouts and device paths (BaM / FDP / ZNS).

pub mod bam;
pub mod fdp;
pub mod sqe;
pub mod zns_append;
pub mod zone_wal;

pub use bam::{BamController, BamError, BamQueuePair, NvmeDoorbellLayout};
pub use fdp::{FdpPlacementId, FdpReclaimSplit, NvmeDirective};
pub use sqe::{NvmeDataPointer, NvmeRwSqe};
pub use zns_append::ZoneAppendSqe;
pub use zone_wal::{ZoneAppendCompletion, ZoneAppendWal};
