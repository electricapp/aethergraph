//! Cold tier: single-file NVMe spill store.

use super::FeatureVector;
use crate::graph::NodeId;
use anyhow::{Context, Result};
use parking_lot::RwLock;
use rustc_hash::FxHashSet;
use std::os::unix::fs::FileExt;

/// Single-file NVMe spill tier.
///
/// One sparse `features.dat` holds every spilled node at byte offset
/// `node * record_bytes`, so a spill or a reload is one positional
/// read/write on a single always-open file — no per-node opens, closes,
/// inodes, or filesystem-block roundups. Presence is tracked in memory:
/// the tier is a process-lifetime spill, recreated empty at construction
/// (a cache is rebuildable by definition, so nothing is lost across
/// restarts — a cold start just misses).
///
/// Records are written at most with one value per node (features are
/// immutable per node in a training run), so a concurrent same-slot
/// read/write can only race identical bytes.
pub(super) struct NvmeTier {
    file: std::fs::File,
    record_bytes: u64,
    dim: usize,
    present: RwLock<FxHashSet<NodeId>>,
}

impl NvmeTier {
    pub(super) fn open(dir: &std::path::Path, dim: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create NVMe cache directory: {}", dir.display()))?;
        let path = dir.join("features.dat");
        // Truncate: presence lives in memory, so stale on-disk slots from
        // a previous process must not survive into this one.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .with_context(|| format!("failed to open NVMe slot file: {}", path.display()))?;
        Ok(Self {
            file,
            record_bytes: (dim * 4) as u64,
            dim,
            present: RwLock::new(FxHashSet::default()),
        })
    }

    /// Read one record, or `None` when the node was never spilled.
    pub(super) fn load_blocking(&self, node: NodeId) -> Result<Option<FeatureVector>> {
        if !self.present.read().contains(&node) {
            return Ok(None);
        }
        let mut features = vec![0f32; self.dim];
        let offset = node as u64 * self.record_bytes;
        self.file
            .read_exact_at(bytemuck::cast_slice_mut(&mut features), offset)
            .with_context(|| format!("failed to read NVMe slot for node {node}"))?;
        Ok(Some(features))
    }

    /// Write one record and mark it present. No fsync: this is a
    /// rebuildable cache, and an fsync per eviction serializes the write
    /// path on device flushes — a crash at worst loses cache entries.
    pub(super) fn save_blocking(&self, node: NodeId, features: &[f32]) -> Result<()> {
        debug_assert_eq!(features.len(), self.dim);
        let offset = node as u64 * self.record_bytes;
        self.file
            .write_all_at(bytemuck::cast_slice(features), offset)
            .with_context(|| format!("failed to write NVMe slot for node {node}"))?;
        self.present.write().insert(node);
        Ok(())
    }
}
