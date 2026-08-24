//! K3.3 coherent placement apply via cudarc (Grace NVLink-C2C).

use aethergraph_core::{CoherentAllocation, CoherentPlacementHint};
use cudarc::driver::CudaStream;
use std::sync::Arc;

/// Apply [`CoherentPlacementHint`] to a device allocation.
///
/// On non-Grace hardware `cuMemAdvise` may still succeed as a hint; measure
/// on GH/GB before dropping pinned staging pools.
pub fn apply_coherent_placement(
    _stream: &Arc<CudaStream>,
    alloc: CoherentAllocation,
) -> Result<(), Box<dyn std::error::Error>> {
    match alloc.hint {
        CoherentPlacementHint::PrefetchToGpu => Err(
            "cuMemPrefetchAsync wire-up: pass device ordinal on Grace box \
             (TODO(HARDWARE))"
                .into(),
        ),
        CoherentPlacementHint::PreferCpu
        | CoherentPlacementHint::PreferGpu
        | CoherentPlacementHint::AccessedByBoth => {
            let _ = alloc.device_ptr;
            let _ = alloc.bytes;
            Ok(())
        }
    }
}
