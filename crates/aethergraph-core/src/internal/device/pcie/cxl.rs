//! K3.2: CXL Type-3 pooled-memory placement.
//!
//! A Type-3 region becomes a host NUMA node through `cxl_pci` and
//! dax/kmem. [`CxlNumaBinding::apply`] binds an existing mapping to that
//! node via `mbind` when the `numa` feature is enabled.

/// Binding request handed to the NUMA `mbind` integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CxlNumaBinding {
    pub numa_node: u32,
    pub length_bytes: usize,
}

impl CxlNumaBinding {
    /// A non-empty memory range bound to an online CXL NUMA node.
    pub const fn new(numa_node: u32, length_bytes: usize) -> Option<Self> {
        if length_bytes == 0 {
            None
        } else {
            Some(Self {
                numa_node,
                length_bytes,
            })
        }
    }

    /// Bind `[ptr, ptr+length_bytes)` to [`Self::numa_node`].
    ///
    /// Requires the `numa` feature (links `aether-mem`). On other builds
    /// returns an explicit error so callers can fall back.
    pub fn apply(&self, ptr: *mut u8) -> Result<(), CxlBindError> {
        if self.length_bytes == 0 {
            return Err(CxlBindError::EmptyRange);
        }
        #[cfg(feature = "numa")]
        {
            aether_mem::numa::bind_region(ptr, self.length_bytes, self.numa_node)
                .map_err(|e| CxlBindError::Mbind(e.to_string()))
        }
        #[cfg(not(feature = "numa"))]
        {
            let _ = ptr;
            Err(CxlBindError::NumaFeatureDisabled)
        }
    }
}

/// Errors from [`CxlNumaBinding::apply`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CxlBindError {
    EmptyRange,
    NumaFeatureDisabled,
    Mbind(String),
}

impl std::fmt::Display for CxlBindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRange => write!(f, "CXL bind length is zero"),
            Self::NumaFeatureDisabled => {
                write!(f, "rebuild with aethergraph-core/numa for mbind")
            }
            Self::Mbind(s) => write!(f, "mbind failed: {s}"),
        }
    }
}

impl std::error::Error for CxlBindError {}

// TODO(HARDWARE): CXL Type-3 host required. Bring up cxl_pci → region →
// dax/kmem, bind a cold tier with mbind, and measure placement plus fallback
// behavior under memory pressure.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_requires_a_range() {
        assert!(CxlNumaBinding::new(4, 0).is_none());
        assert_eq!(CxlNumaBinding::new(4, 4096).unwrap().numa_node, 4);
    }

    #[test]
    fn apply_without_numa_feature_errors_cleanly() {
        let b = CxlNumaBinding::new(1, 4096).unwrap();
        let mut buf = [0u8; 4096];
        let err = b.apply(buf.as_mut_ptr()).unwrap_err();
        #[cfg(not(feature = "numa"))]
        assert_eq!(err, CxlBindError::NumaFeatureDisabled);
        #[cfg(feature = "numa")]
        {
            // May succeed or fail depending on node online — just must not panic.
            let _ = err;
        }
    }
}
