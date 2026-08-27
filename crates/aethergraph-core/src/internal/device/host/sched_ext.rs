//! K4.1: `sched_ext` policy + BPF object loader.
//!
//! [`SchedExtPolicy`] is the typed intent. [`SchedExtLoader`] locates the
//! compiled `sched_ext_aether.bpf.o` (from the `sched_ext_bpf` build) and
//! documents the attach sequence; live `bpf()` attach is Linux-only and
//! returns clear errors when the object or CAP_BPF is missing.

use std::path::{Path, PathBuf};

/// Role recognized by the scheduler policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedTaskRole {
    SqPoll,
    Gather,
    Other,
}

/// Per-role dispatch decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedDecision {
    pub never_preempt: bool,
    pub preferred_numa_node: Option<u32>,
}

/// Intended sched_ext policy, independent of BPF loader details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedExtPolicy {
    pub nic_local_numa_node: u32,
}

impl SchedExtPolicy {
    /// Produce the dispatch constraints for one classified task.
    pub const fn decision_for(self, role: SchedTaskRole) -> SchedDecision {
        match role {
            SchedTaskRole::SqPoll => SchedDecision {
                never_preempt: true,
                preferred_numa_node: Some(self.nic_local_numa_node),
            },
            SchedTaskRole::Gather => SchedDecision {
                never_preempt: false,
                preferred_numa_node: Some(self.nic_local_numa_node),
            },
            SchedTaskRole::Other => SchedDecision {
                never_preempt: false,
                preferred_numa_node: None,
            },
        }
    }

    /// Map role to the BPF `task_role` map value.
    pub const fn bpf_role_tag(role: SchedTaskRole) -> u32 {
        match role {
            SchedTaskRole::Other => 0,
            SchedTaskRole::SqPoll => 1,
            SchedTaskRole::Gather => 2,
        }
    }
}

/// Locates and (on Linux) would attach the sched_ext BPF object.
#[derive(Debug, Clone)]
pub struct SchedExtLoader {
    pub object_path: PathBuf,
    pub policy: SchedExtPolicy,
}

impl SchedExtLoader {
    /// Prefer `AETHER_SCHED_EXT_BPF_OBJ`, then a path next to the binary.
    pub fn discover(policy: SchedExtPolicy) -> Self {
        let object_path = std::env::var_os("AETHER_SCHED_EXT_BPF_OBJ")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("sched_ext_aether.bpf.o"));
        Self {
            object_path,
            policy,
        }
    }

    pub fn with_object(policy: SchedExtPolicy, object_path: impl Into<PathBuf>) -> Self {
        Self {
            object_path: object_path.into(),
            policy,
        }
    }

    pub fn object_exists(&self) -> bool {
        self.object_path.exists()
    }

    /// Load + attach struct_ops. Requires root, CONFIG_SCHED_CLASS_EXT, and
    /// a compiled object from `sched_ext_bpf`.
    pub fn attach(&self) -> std::io::Result<()> {
        if !self.object_exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "BPF object {} missing — build with --features sched_ext_bpf",
                    self.object_path.display()
                ),
            ));
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "sched_ext attach is Linux-only",
            ))
        }
        #[cfg(target_os = "linux")]
        {
            // Full attach needs libbpf/aya struct_ops APIs. Until linked, we
            // validate the object is non-empty ELF and return a precise error
            // so the VM grind path is one step: wire aya::Ebpf::load_file.
            let meta = std::fs::metadata(&self.object_path)?;
            if meta.len() < 64 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "BPF object too small to be a valid ELF",
                ));
            }
            let _ = self.policy;
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!(
                    "SchedExtLoader: object {} ready; attach via aya/libbpf \
                     struct_ops on a CONFIG_SCHED_CLASS_EXT kernel \
                     (TODO(HARDWARE) grind)",
                    self.object_path.display()
                ),
            ))
        }
    }

    pub fn object_path(&self) -> &Path {
        &self.object_path
    }
}

// BPF source: crates/aether-stream/bpf/src/sched_ext_aether.c
// TODO(HARDWARE): On a rooted Linux VM with sched_ext enabled, load the BPF
// scheduler and verify SQPOLL run-queue continuity plus gather placement on
// the NIC-local node under competing CPU load.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqpoll_is_protected_while_gather_is_placed() {
        let policy = SchedExtPolicy {
            nic_local_numa_node: 1,
        };
        assert_eq!(
            policy.decision_for(SchedTaskRole::SqPoll),
            SchedDecision {
                never_preempt: true,
                preferred_numa_node: Some(1),
            }
        );
        assert!(!policy.decision_for(SchedTaskRole::Gather).never_preempt);
        assert_eq!(SchedExtPolicy::bpf_role_tag(SchedTaskRole::SqPoll), 1);
    }

    #[test]
    fn loader_reports_missing_object() {
        let loader = SchedExtLoader::with_object(
            SchedExtPolicy {
                nic_local_numa_node: 0,
            },
            "/nonexistent/sched_ext_aether.bpf.o",
        );
        let err = loader.attach().unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }
}
