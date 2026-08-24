//! K4.2: DAMON access-frequency demotion policy.
//!
//! These schemes replace a degree-weighted userfaultfd heuristic with measured
//! page access frequency. A future adapter maps them to DAMON regions and
//! schemes after the kernel-side monitor is configured.

/// One access-frequency threshold and demotion target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccessFrequencyScheme {
    /// Demote when sampled accesses are at or below this count.
    pub max_accesses: u32,
    /// Minimum age before a region is eligible for demotion.
    pub min_age_ms: u64,
    /// NUMA node or memory tier receiving demoted pages.
    pub target_node: u32,
}

/// DAMON sampling and demotion configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DamonConfig {
    pub sample_interval_ms: u64,
    pub aggregation_interval_ms: u64,
    pub scheme: AccessFrequencyScheme,
}

impl DamonConfig {
    /// Build a cadence where aggregation spans at least one sample.
    pub const fn new(
        sample_interval_ms: u64,
        aggregation_interval_ms: u64,
        scheme: AccessFrequencyScheme,
    ) -> Option<Self> {
        if sample_interval_ms == 0 || aggregation_interval_ms < sample_interval_ms {
            None
        } else {
            Some(Self {
                sample_interval_ms,
                aggregation_interval_ms,
                scheme,
            })
        }
    }
}

// Adapter: [`super::damon_sysfs::DamonSysfs`] writes attrs/schemes under a
// sysfs root (tests use a temp dir; production uses `/sys/kernel/mm/damon`).
// TODO(HARDWARE): On a rooted Linux VM with DAMON enabled, apply these schemes
// to the feature mapping and compare measured cold-page demotion, refaults,
// and latency against the degree-weighted userfaultfd heuristic.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregation_must_cover_a_sampling_window() {
        let scheme = AccessFrequencyScheme {
            max_accesses: 1,
            min_age_ms: 100,
            target_node: 2,
        };
        assert!(DamonConfig::new(0, 100, scheme).is_none());
        assert!(DamonConfig::new(100, 99, scheme).is_none());
        assert_eq!(
            DamonConfig::new(100, 500, scheme)
                .unwrap()
                .scheme
                .target_node,
            2
        );
    }
}
