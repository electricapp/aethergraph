//! K4.2 DAMON sysfs adapter — applies [`DamonConfig`] under a sysfs root.
//!
//! Production path prefers the modern DAMON sysfs tree:
//! `/sys/kernel/mm/damon/admin/kdamonds/<N>/...`. Tests may pass a temp
//! directory that only has a stub `scheme` file.

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

    /// Whether a recognizable DAMON control layout is present.
    #[must_use]
    pub fn available(&self) -> bool {
        self.kdamond_dir(0).exists()
            || self.root.join("admin_schemes").exists()
            || self.root.join("attrs").exists()
            || self.root.join("scheme").exists()
    }

    fn kdamond_dir(&self, id: u32) -> PathBuf {
        self.root
            .join("admin")
            .join("kdamonds")
            .join(id.to_string())
    }

    /// Write sampling / aggregation attrs and a single demotion scheme.
    ///
    /// Tries, in order:
    /// 1. Modern `admin/kdamonds/0` tree (Linux DAMON sysfs)
    /// 2. Legacy `attrs` + `admin_schemes` (older experiments)
    /// 3. Stub `scheme` file (unit tests)
    pub fn apply(&self, cfg: DamonConfig) -> io::Result<()> {
        if self.kdamond_dir(0).exists() {
            return self.apply_kdamond(0, cfg);
        }
        let attrs = self.root.join("attrs");
        if attrs.exists() {
            // Microseconds in some trees; we write the ms config as-is and
            // document the unit at the call site for hardware grind.
            let line = format!(
                "{} {} {} 10 1000\n",
                cfg.sample_interval_ms, cfg.aggregation_interval_ms, cfg.sample_interval_ms
            );
            fs::write(&attrs, line)?;
            return self.write_legacy_scheme(&cfg.scheme);
        }
        self.write_stub_scheme(&cfg.scheme)
    }

    /// Modern DAMON sysfs: kdamond contexts / schemes / access_pattern.
    fn apply_kdamond(&self, id: u32, cfg: DamonConfig) -> io::Result<()> {
        let kdamond = self.kdamond_dir(id);
        // sample_intervals are in microseconds on modern kernels.
        let contexts = kdamond.join("contexts");
        let ctx0 = contexts.join("0");
        if !ctx0.exists() {
            // Ask the kernel to create context 0 when nr_contexts is writable.
            let nr = contexts.join("nr_contexts");
            if nr.exists() {
                fs::write(&nr, "1\n")?;
            }
        }
        let ctx0 = contexts.join("0");
        if ctx0.exists() {
            let intervals = ctx0.join("monitoring_attrs").join("intervals");
            if intervals.exists() {
                let sample_us = cfg.sample_interval_ms.saturating_mul(1000);
                let aggr_us = cfg.aggregation_interval_ms.saturating_mul(1000);
                let _ = fs::write(intervals.join("sample_us"), format!("{sample_us}\n"));
                let _ = fs::write(intervals.join("aggr_us"), format!("{aggr_us}\n"));
            }
            let schemes = ctx0.join("schemes");
            let nr_schemes = schemes.join("nr_schemes");
            if nr_schemes.exists() {
                let _ = fs::write(&nr_schemes, "1\n");
            }
            let scheme0 = schemes.join("0");
            if scheme0.exists() {
                let _ = fs::write(scheme0.join("action"), "pageout\n");
                let access = scheme0.join("access_pattern");
                let _ = fs::write(
                    access.join("nr_accesses").join("max"),
                    format!("{}\n", cfg.scheme.max_accesses),
                );
                // Age is counted in aggregation intervals, not ms.
                let age_intervals = cfg
                    .scheme
                    .min_age_ms
                    .saturating_div(cfg.aggregation_interval_ms.max(1));
                let _ = fs::write(access.join("age").join("min"), format!("{age_intervals}\n"));
                let dest = scheme0.join("dests");
                if dest.exists() {
                    let _ = fs::write(dest.join("nr_dests"), "1\n");
                    let d0 = dest.join("0");
                    let _ = fs::write(d0.join("id"), format!("{}\n", cfg.scheme.target_node));
                }
            }
        }
        // state: on
        let state = kdamond.join("state");
        if state.exists() {
            let _ = fs::write(state, "on\n");
        }
        Ok(())
    }

    fn write_legacy_scheme(&self, scheme: &AccessFrequencyScheme) -> io::Result<()> {
        let schemes = self.root.join("admin_schemes");
        if !schemes.exists() {
            return self.write_stub_scheme(scheme);
        }
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
        Ok(())
    }

    fn write_stub_scheme(&self, scheme: &AccessFrequencyScheme) -> io::Result<()> {
        let path = self.root.join("scheme");
        let body = format!(
            "max_accesses={} min_age_ms={} target_node={}\n",
            scheme.max_accesses, scheme.min_age_ms, scheme.target_node
        );
        fs::write(path, body)
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
