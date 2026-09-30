//! K4.2 DAMON sysfs adapter — applies [`DamonConfig`] under a sysfs root.
//!
//! Drives the DAMON sysfs interface (`admin/kdamonds/<N>/...`, Linux 5.18+):
//! one kdamond monitoring physical memory, one `migrate_cold` scheme that
//! demotes cold regions to the configured node. Tests point it at a temp
//! directory laid out the way the kernel lays out that tree.

use super::{AccessFrequencyScheme, DamonConfig};
use std::fmt::Display;
use std::fs;
use std::io;
use std::ops::Range;
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

    fn kdamonds(&self) -> PathBuf {
        self.root.join("admin").join("kdamonds")
    }

    /// Whether the DAMON sysfs interface is present.
    #[must_use]
    pub fn available(&self) -> bool {
        self.kdamonds().join("nr_kdamonds").exists()
    }

    /// Monitor the largest `System RAM` range (from `/proc/iomem`, which
    /// needs root to show addresses) and demote cold regions per `cfg`.
    pub fn apply(&self, cfg: DamonConfig) -> io::Result<()> {
        let iomem = fs::read_to_string("/proc/iomem")?;
        let ram = biggest_system_ram(&iomem).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "no System RAM range readable in /proc/iomem (requires root)",
            )
        })?;
        self.apply_to_range(cfg, ram)
    }

    /// Configure kdamond 0 to monitor physical range `phys` and demote
    /// cold regions per `cfg`, then start it.
    ///
    /// `migrate_cold` is a physical-address (`paddr`) scheme, so the
    /// monitoring target is a range of physical memory rather than a
    /// process. Every write is checked: a kernel that lacks a knob this
    /// needs fails the call instead of starting a monitor that does
    /// something else.
    pub fn apply_to_range(&self, cfg: DamonConfig, phys: Range<u64>) -> io::Result<()> {
        if !self.available() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "DAMON sysfs interface not found under {}",
                    self.root.display()
                ),
            ));
        }
        if phys.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "empty physical monitoring range",
            ));
        }

        let kdamonds = self.kdamonds();
        ensure_count(&kdamonds, "nr_kdamonds", 1)?;
        let kdamond = kdamonds.join("0");
        let contexts = kdamond.join("contexts");
        ensure_count(&contexts, "nr_contexts", 1)?;
        let ctx = contexts.join("0");

        let avail = ctx.join("avail_operations");
        if avail.exists()
            && !fs::read_to_string(&avail)?
                .split_whitespace()
                .any(|op| op == "paddr")
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "DAMON physical address monitoring (paddr) is not available",
            ));
        }
        put(&ctx.join("operations"), "paddr")?;

        let intervals = ctx.join("monitoring_attrs").join("intervals");
        put(
            &intervals.join("sample_us"),
            cfg.sample_interval_ms.saturating_mul(1000),
        )?;
        put(
            &intervals.join("aggr_us"),
            cfg.aggregation_interval_ms.saturating_mul(1000),
        )?;

        let targets = ctx.join("targets");
        ensure_count(&targets, "nr_targets", 1)?;
        let regions = targets.join("0").join("regions");
        ensure_count(&regions, "nr_regions", 1)?;
        let region = regions.join("0");
        put(&region.join("start"), phys.start)?;
        put(&region.join("end"), phys.end)?;

        let schemes = ctx.join("schemes");
        ensure_count(&schemes, "nr_schemes", 1)?;
        write_scheme(&schemes.join("0"), &cfg)?;

        put(&kdamond.join("state"), "on")
    }
}

/// One `migrate_cold` scheme: regions accessed at most `max_accesses`
/// times per aggregation and at least `min_age_ms` old move to
/// `target_node`.
fn write_scheme(scheme: &Path, cfg: &DamonConfig) -> io::Result<()> {
    let s: &AccessFrequencyScheme = &cfg.scheme;
    put(&scheme.join("action"), "migrate_cold")?;

    // The destination is `target_nid` (6.11+) or, on kernels with
    // multi-destination migration, the `dests` directory.
    let target_nid = scheme.join("target_nid");
    let dests = scheme.join("dests");
    if target_nid.exists() {
        put(&target_nid, s.target_node)?;
    } else if dests.exists() {
        ensure_count(&dests, "nr_dests", 1)?;
        put(&dests.join("0").join("id"), s.target_node)?;
    } else {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "this kernel's DAMON cannot choose a migration target node",
        ));
    }

    let access = scheme.join("access_pattern");
    put(&access.join("nr_accesses").join("min"), 0)?;
    put(&access.join("nr_accesses").join("max"), s.max_accesses)?;
    // Age is counted in aggregation intervals, not ms.
    let age_intervals = s
        .min_age_ms
        .saturating_div(cfg.aggregation_interval_ms.max(1));
    put(&access.join("age").join("min"), age_intervals)
}

/// Write one sysfs value, naming the file in the error.
fn put(path: &Path, value: impl Display) -> io::Result<()> {
    fs::write(path, format!("{value}\n"))
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))
}

/// Set a `nr_*` count file to `n` unless it already reads `n`; the kernel
/// creates or removes the numbered children on write.
fn ensure_count(dir: &Path, file: &str, n: u32) -> io::Result<()> {
    let path = dir.join(file);
    let current = fs::read_to_string(&path)
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    if current.trim().parse::<u32>().ok() == Some(n) {
        return Ok(());
    }
    put(&path, n)
}

/// The largest top-level `System RAM` range in `/proc/iomem` text, as a
/// half-open byte range. Unprivileged readers see all-zero addresses,
/// which yield `None`.
fn biggest_system_ram(iomem: &str) -> Option<Range<u64>> {
    iomem
        .lines()
        // Nested resources are indented; only top-level ranges are RAM
        // banks.
        .filter(|line| !line.starts_with(' '))
        .filter_map(|line| {
            let (span, name) = line.split_once(" : ")?;
            if name.trim() != "System RAM" {
                return None;
            }
            let (start, end) = span.split_once('-')?;
            let start = u64::from_str_radix(start.trim(), 16).ok()?;
            let end = u64::from_str_radix(end.trim(), 16).ok()?.checked_add(1)?;
            (end > start && end > 1).then_some(start..end)
        })
        .max_by_key(|r| r.end - r.start)
}

// TODO(HARDWARE): On a rooted Linux VM with DAMON enabled, apply DamonConfig
// via DamonSysfs::system() and compare demotion/refaults to the uffd heuristic.

#[cfg(test)]
mod tests {
    use super::*;

    /// The directories the kernel creates as the `nr_*` counts are set,
    /// with the count files already reading 1.
    fn kernel_tree(root: &Path, dest_knob: &str) {
        let kd = root.join("admin/kdamonds");
        let ctx = kd.join("0/contexts/0");
        let scheme = ctx.join("schemes/0");
        for dir in [
            ctx.join("monitoring_attrs/intervals"),
            ctx.join("targets/0/regions/0"),
            scheme.join("access_pattern/nr_accesses"),
            scheme.join("access_pattern/age"),
        ] {
            fs::create_dir_all(dir).unwrap();
        }
        for count in [
            kd.join("nr_kdamonds"),
            kd.join("0/contexts/nr_contexts"),
            ctx.join("targets/nr_targets"),
            ctx.join("targets/0/regions/nr_regions"),
            ctx.join("schemes/nr_schemes"),
        ] {
            fs::write(count, "1\n").unwrap();
        }
        fs::write(ctx.join("avail_operations"), "vaddr fvaddr paddr\n").unwrap();
        match dest_knob {
            "target_nid" => fs::write(scheme.join("target_nid"), "0\n").unwrap(),
            "dests" => {
                fs::create_dir_all(scheme.join("dests/0")).unwrap();
                fs::write(scheme.join("dests/nr_dests"), "1\n").unwrap();
            }
            _ => {}
        }
    }

    fn read(root: &Path, rel: &str) -> String {
        fs::read_to_string(root.join(rel))
            .unwrap()
            .trim()
            .to_owned()
    }

    fn config() -> DamonConfig {
        let scheme = AccessFrequencyScheme {
            max_accesses: 2,
            min_age_ms: 500,
            target_node: 3,
        };
        DamonConfig::new(10, 100, scheme).unwrap()
    }

    #[test]
    fn applies_a_migrate_cold_scheme_over_physical_memory() {
        let dir = tempfile::tempdir().unwrap();
        kernel_tree(dir.path(), "target_nid");
        DamonSysfs::at(dir.path())
            .apply_to_range(config(), 0x10_0000..0x4000_0000)
            .unwrap();

        let ctx = "admin/kdamonds/0/contexts/0";
        assert_eq!(read(dir.path(), &format!("{ctx}/operations")), "paddr");
        assert_eq!(
            read(
                dir.path(),
                &format!("{ctx}/monitoring_attrs/intervals/sample_us")
            ),
            "10000"
        );
        assert_eq!(
            read(dir.path(), &format!("{ctx}/targets/0/regions/0/start")),
            "1048576"
        );
        assert_eq!(
            read(dir.path(), &format!("{ctx}/targets/0/regions/0/end")),
            "1073741824"
        );
        let scheme = format!("{ctx}/schemes/0");
        assert_eq!(
            read(dir.path(), &format!("{scheme}/action")),
            "migrate_cold"
        );
        assert_eq!(read(dir.path(), &format!("{scheme}/target_nid")), "3");
        assert_eq!(
            read(
                dir.path(),
                &format!("{scheme}/access_pattern/nr_accesses/max")
            ),
            "2"
        );
        // 500 ms at a 100 ms aggregation interval.
        assert_eq!(
            read(dir.path(), &format!("{scheme}/access_pattern/age/min")),
            "5"
        );
        assert_eq!(read(dir.path(), "admin/kdamonds/0/state"), "on");
    }

    #[test]
    fn multi_destination_kernels_get_the_target_through_dests() {
        let dir = tempfile::tempdir().unwrap();
        kernel_tree(dir.path(), "dests");
        DamonSysfs::at(dir.path())
            .apply_to_range(config(), 0..1 << 30)
            .unwrap();
        assert_eq!(
            read(
                dir.path(),
                "admin/kdamonds/0/contexts/0/schemes/0/dests/0/id"
            ),
            "3"
        );
    }

    #[test]
    fn missing_knobs_fail_instead_of_starting_a_different_monitor() {
        // No DAMON tree at all.
        let dir = tempfile::tempdir().unwrap();
        let err = DamonSysfs::at(dir.path())
            .apply_to_range(config(), 0..1 << 30)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);

        // A kernel that cannot pick a migration target.
        let dir = tempfile::tempdir().unwrap();
        kernel_tree(dir.path(), "none");
        let err = DamonSysfs::at(dir.path())
            .apply_to_range(config(), 0..1 << 30)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
        assert!(
            !dir.path().join("admin/kdamonds/0/state").exists(),
            "the monitor must not be started"
        );
    }

    #[test]
    fn system_ram_is_the_largest_top_level_bank() {
        let iomem = "\
00000000-00000fff : Reserved
00001000-0009efff : System RAM
00100000-bfffffff : System RAM
  01000000-01ffffff : Kernel code
c0000000-febfffff : PCI Bus 0000:00
100000000-43fffffff : System RAM
";
        assert_eq!(
            biggest_system_ram(iomem),
            Some(0x1_0000_0000..0x4_4000_0000)
        );
        // What an unprivileged reader sees.
        let masked = "00000000-00000000 : System RAM\n";
        assert_eq!(biggest_system_ram(masked), None);
    }
}
