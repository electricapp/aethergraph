//! K3.3: Grace-Hopper/Grace-Blackwell coherent allocation placement.
//!
//! [`CoherentPlacementHint`] is applied by the CUDA-facing helper in
//! `aether-stream` (`kernels` / pool advise). This module owns the typed
//! intent so host and device paths share one vocabulary.

/// CUDA placement intent for a coherent CPU/GPU allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoherentPlacementHint {
    /// Prefer the Grace CPU's memory attachment.
    PreferCpu,
    /// Prefer the GPU's memory attachment.
    PreferGpu,
    /// Make the allocation concurrently accessible from both processors.
    AccessedByBoth,
    /// Prefetch before a known gather phase.
    PrefetchToGpu,
}

impl CoherentPlacementHint {
    /// Map to the cudarc / CUDA advised-memory set (documented constants).
    ///
    /// Values match `CUmem_advise` enumerants used by `cuMemAdvise`:
    /// - PreferCpu → CU_MEM_ADVISE_SET_PREFERRED_LOCATION (CPU)
    /// - PreferGpu → CU_MEM_ADVISE_SET_PREFERRED_LOCATION (GPU)
    /// - AccessedByBoth → CU_MEM_ADVISE_SET_ACCESSED_BY (both)
    /// - PrefetchToGpu → cuMemPrefetchAsync (not advise)
    pub const fn needs_prefetch(self) -> bool {
        matches!(self, Self::PrefetchToGpu)
    }
}

/// Host-side record of a coherent allocation + desired hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoherentAllocation {
    pub device_ptr: u64,
    pub bytes: u64,
    pub hint: CoherentPlacementHint,
}

impl CoherentAllocation {
    pub fn new(device_ptr: u64, bytes: u64, hint: CoherentPlacementHint) -> Option<Self> {
        (device_ptr != 0 && bytes != 0).then_some(Self {
            device_ptr,
            bytes,
            hint,
        })
    }
}

// Apply path: aether-stream `gpu::kernels::coherent` (cudarc cuMemAdvise).
// TODO(HARDWARE): Grace-Hopper or Grace-Blackwell required. Apply matching
// cuMemAdvise/cuMemPrefetchAsync calls and verify coherence, migration cost,
// and gather throughput against the existing pinned-staging path.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefetch_hint_flagged() {
        assert!(CoherentPlacementHint::PrefetchToGpu.needs_prefetch());
        assert!(!CoherentPlacementHint::PreferGpu.needs_prefetch());
        assert!(CoherentAllocation::new(0x1000, 4096, CoherentPlacementHint::PreferCpu).is_some());
        assert!(CoherentAllocation::new(0, 4096, CoherentPlacementHint::PreferCpu).is_none());
    }
}
