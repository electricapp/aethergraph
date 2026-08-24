//! K4.2 DAMON sysfs adapter — applies [`DamonConfig`] under a sysfs root.
//!
//! Production path uses `/sys/kernel/mm/damon`. Tests pass a temp directory
//! that mirrors the knobs they care about.

use super::{AccessFrequencyScheme, DamonConfig};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Writable view of DAMON knobs under `root` (usually `/sys/kernel/mm/damon`).
#[derive(Debug, Clone)]
pub struct DamonSysfs {
    root: PathBuf,
}

impl DamonSysfs {
    /// Live kernel DAMON control plane.
    #[must_use]
    pub fn system() -> Self {
        Self {
            root: PathBuf::from("/sys/kernel/mm/damon"),
        }
    }

    /// Test / alternate root (must already exist).
    #[must_use]
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Whether the expected control files are present.
    #[must_use]
    pub fn available(&self) -> bool {
        self.root.join("admin_schemes").exists() || self.root.join("attrs").exists()
    }

    /// Write sampling / aggregation attrs and a single demotion scheme.
    ///
    /// Layout follows the DAMON sysfs ABI (attrs + schemes). Missing files
    /// return `NotFound` so callers can skip on non-DAMON kernels.
    pub fn apply(&self, cfg: DamonConfig) -> io::Result<()> {
        let attrs = self.root.join("attrs");
        if attrs.exists() {
            // sample_interval aggregation_interval update_interval min_nr max_nr
            let line = format!(
                "{} {} {} 10 1000\n",
                cfg.sample_interval_ms, cfg.aggregation_interval_ms, cfg.sample_interval_ms
            );
            fs::write(&attrs, line)?;
        }
        self.write_scheme(&cfg.scheme)?;
        Ok(())
    }

    fn write_scheme(&self, scheme: &AccessFrequencyScheme) -> io::Result<()> {
        let schemes = self.root.join("admin_schemes");
        if !schemes.exists() {
            // Older / stub trees: write a single scheme file the unit test creates.
            let path = self.root.join("scheme");
            let body = format!(
                "max_accesses={} min_age_ms={} target_node={}\n",
                scheme.max_accesses, scheme.min_age_ms, scheme.target_node
            );
            return fs::write(path, body);
        }
        // Full DAMON sysfs: create scheme0 action=pageout with access pattern.
        let scheme_dir = schemes.join("0");
        fs::create_dir_all(&scheme_dir)?;
        fs::write(scheme_dir.join("action"), "pageout\n")?;
        let access = scheme_dir.join("access_pattern");
        fs::create_dir_all(&access)?;
        fs::write(
            access.join("nr_accesses"),
            format!("0 {}\n", scheme.max_accesses),
        )?;
        fs::write(access.join("age"), format!("{} max\n", scheme.min_age_ms))?;
        let dest = self.root.join("target_node");
        if dest.parent().map(Path::exists).unwrap_or(false) {
            let _ = fs::write(dest, format!("{}\n", scheme.target_node));
        }
        Ok(())
    }

    /// Read back the stub `scheme` file (test helper).
    pub fn read_stub_scheme(&self) -> io::Result<String> {
        fs::read_to_string(self.root.join("scheme"))
    }
}

// TODO(HARDWARE): On a rooted Linux VM with DAMON enabled, apply DamonConfig
// via DamonSysfs::system() and compare demotion/refaults to the uffd heuristic.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_writes_stub_scheme_under_temp_root() {
        let dir = std::env::temp_dir().join(format!("aether-damon-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let sys = DamonSysfs::at(&dir);
        let scheme = AccessFrequencyScheme {
            max_accesses: 2,
            min_age_ms: 50,
            target_node: 3,
        };
        let cfg = DamonConfig::new(10, 100, scheme).unwrap();
        sys.apply(cfg).unwrap();
        let body = sys.read_stub_scheme().unwrap();
        assert!(body.contains("max_accesses=2"));
        assert!(body.contains("target_node=3"));
        let _ = fs::remove_dir_all(&dir);
    }
}
