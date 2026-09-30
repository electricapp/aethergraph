//! K3.3 coherent placement apply via the driver (Grace NVLink-C2C).

use aethergraph_core::{CoherentAllocation, CoherentPlacementHint};
use cudarc::driver::{CudaStream, sys};
use std::sync::Arc;

/// `CU_DEVICE_CPU`: the host as a `cuMemAdvise` location.
const CU_DEVICE_CPU: sys::CUdevice = -1;

/// Apply [`CoherentPlacementHint`] to an allocation on `stream`'s device.
///
/// Hints go through `cuMemAdvise`, the prefetch through `cuMemPrefetchAsync`
/// on `stream`. Both need memory the driver manages: managed allocations
/// anywhere, system memory on Grace (ATS). Anything else is rejected by the
/// driver, and that error is returned rather than dropped.
pub fn apply_coherent_placement(
    stream: &Arc<CudaStream>,
    alloc: CoherentAllocation,
) -> Result<(), Box<dyn std::error::Error>> {
    let ctx = stream.context();
    ctx.bind_to_thread()?;
    let device = ctx.cu_device();
    let bytes = usize::try_from(alloc.bytes)?;
    let advise = |advice: sys::CUmem_advise, location: sys::CUdevice| {
        // SAFETY: the range is the caller's allocation on this context,
        // which is current; `advice` and `location` are valid values.
        let res = unsafe { sys::cuMemAdvise(alloc.device_ptr, bytes, advice, location) };
        driver_ok(res, "cuMemAdvise")
    };
    match alloc.hint {
        CoherentPlacementHint::PreferCpu => advise(
            sys::CUmem_advise::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
            CU_DEVICE_CPU,
        ),
        CoherentPlacementHint::PreferGpu => advise(
            sys::CUmem_advise::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
            device,
        ),
        CoherentPlacementHint::AccessedByBoth => {
            advise(sys::CUmem_advise::CU_MEM_ADVISE_SET_ACCESSED_BY, device)?;
            advise(
                sys::CUmem_advise::CU_MEM_ADVISE_SET_ACCESSED_BY,
                CU_DEVICE_CPU,
            )
        }
        CoherentPlacementHint::PrefetchToGpu => {
            // SAFETY: as above; the prefetch is ordered on `stream`, which
            // belongs to this context.
            let res = unsafe {
                sys::cuMemPrefetchAsync(alloc.device_ptr, bytes, device, stream.cu_stream())
            };
            driver_ok(res, "cuMemPrefetchAsync")
        }
    }
}

fn driver_ok(res: sys::CUresult, what: &str) -> Result<(), Box<dyn std::error::Error>> {
    if res == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what} failed: {res:?}").into())
    }
}
