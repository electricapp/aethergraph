//! Cold tier: single-file NVMe spill store.
//!
//! On Linux, batch loads go through an io_uring lane (O_DIRECT when the
//! padded slot stride is device-aligned); elsewhere, and on uring setup
//! failure, loads fall back to positional `pread`.

use super::FeatureVector;
use crate::graph::NodeId;
use anyhow::{Context, Result};
#[cfg(target_os = "linux")]
use parking_lot::Mutex;
use parking_lot::RwLock;
use rustc_hash::FxHashSet;
use std::os::unix::fs::FileExt;

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;

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
/// `record_bytes` is the on-disk stride: the `dim * 4` payload, rounded up
/// to the O_DIRECT alignment on Linux so a batch gather can use the
/// aligned landing pool. Padding bytes are unused.
///
/// Records are written at most with one value per node (features are
/// immutable per node in a training run), so a concurrent same-slot
/// read/write can only race identical bytes.
pub(super) struct NvmeTier {
    file: std::fs::File,
    /// On-disk bytes per node (payload + O_DIRECT padding).
    record_bytes: u64,
    /// Feature payload bytes (`dim * size_of::<f32>()`).
    payload_bytes: usize,
    dim: usize,
    present: RwLock<FxHashSet<NodeId>>,
    /// Linux: persistent ring + landing buffers for batch gathers.
    /// Created lazily on first `load_batch` so construction stays cheap
    /// when the spill tier is never hit.
    #[cfg(target_os = "linux")]
    uring: Mutex<Option<crate::internal::uring::UringLane>>,
    #[cfg(target_os = "linux")]
    direct_io: bool,
}

impl NvmeTier {
    pub(super) fn open(dir: &std::path::Path, dim: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create NVMe cache directory: {}", dir.display()))?;
        let path = dir.join("features.dat");
        let payload_bytes = dim
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("feature dim {dim} overflows record size"))?;

        #[cfg(target_os = "linux")]
        let (file, record_bytes, direct_io) = {
            use crate::internal::uring::{
                DIRECT_IO_OFFSET_ALIGNMENT, direct_io_offset_alignment, open_direct_rw_or_fallback,
            };

            // Truncate via a buffered open first so we can probe alignment,
            // then reopen O_DIRECT when the padded stride clears the device
            // requirement. Cache files are always recreated empty — padding
            // the stride never breaks a prior layout.
            let probe = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
                .with_context(|| format!("failed to open NVMe slot file: {}", path.display()))?;
            let align = direct_io_offset_alignment(&probe).unwrap_or(DIRECT_IO_OFFSET_ALIGNMENT);
            let record_bytes = (payload_bytes.div_ceil(align) * align) as u64;
            drop(probe);

            // Spill tiers write; must use the rw O_DIRECT helper — a
            // read-only O_DIRECT fd returns EBADF on pwrite (CI failure mode).
            let (file, direct) = open_direct_rw_or_fallback(&path)?;
            // O_DIRECT only sticks when the padded stride matches the device
            // alignment the open path expects; otherwise fall back to the
            // buffered fd (still usable with io_uring scratch gathers).
            let direct = direct && (record_bytes as usize).is_multiple_of(align);
            let file = if direct {
                file
            } else {
                std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&path)
                    .with_context(|| {
                        format!(
                            "failed to reopen NVMe slot file buffered: {}",
                            path.display()
                        )
                    })?
            };
            (file, record_bytes, direct)
        };

        #[cfg(not(target_os = "linux"))]
        let (file, record_bytes) = {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
                .with_context(|| format!("failed to open NVMe slot file: {}", path.display()))?;
            (file, payload_bytes as u64)
        };

        Ok(Self {
            file,
            record_bytes,
            payload_bytes,
            dim,
            present: RwLock::new(FxHashSet::default()),
            #[cfg(target_os = "linux")]
            uring: Mutex::new(None),
            #[cfg(target_os = "linux")]
            direct_io,
        })
    }

    /// Read one record, or `None` when the node was never spilled.
    pub(super) fn load_blocking(&self, node: NodeId) -> Result<Option<FeatureVector>> {
        if !self.present.read().contains(&node) {
            return Ok(None);
        }
        let mut features = vec![0f32; self.dim];
        self.read_record(node, bytemuck::cast_slice_mut(&mut features))?;
        Ok(Some(features))
    }

    /// Load many nodes in one pass. Present nodes are gathered via a single
    /// io_uring submission on Linux (O_DIRECT when the slot stride allows);
    /// absent nodes return `None` without I/O. Order matches `nodes`.
    pub(super) fn load_batch(
        &self,
        nodes: &[NodeId],
    ) -> Result<Vec<(NodeId, Option<FeatureVector>)>> {
        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        let present = self.present.read();
        let mut out: Vec<(NodeId, Option<FeatureVector>)> = Vec::with_capacity(nodes.len());
        let mut hits: Vec<(usize, NodeId)> = Vec::new();
        for (i, &node) in nodes.iter().enumerate() {
            if present.contains(&node) {
                hits.push((i, node));
                out.push((node, None)); // placeholder; filled below
            } else {
                out.push((node, None));
            }
        }
        drop(present);

        if hits.is_empty() {
            return Ok(out);
        }

        #[cfg(target_os = "linux")]
        {
            match self.load_batch_uring(&hits) {
                Ok(rows) => {
                    for ((idx, _), row) in hits.iter().zip(rows) {
                        out[*idx].1 = Some(row);
                    }
                    return Ok(out);
                }
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        "NVMe spill io_uring gather failed; falling back to pread"
                    );
                }
            }
        }

        for (idx, node) in hits {
            let mut features = vec![0f32; self.dim];
            self.read_record(node, bytemuck::cast_slice_mut(&mut features))?;
            out[idx].1 = Some(features);
        }
        Ok(out)
    }

    fn read_record(&self, node: NodeId, dest: &mut [u8]) -> Result<()> {
        debug_assert_eq!(dest.len(), self.payload_bytes);
        let offset = u64::from(node) * self.record_bytes;
        #[cfg(target_os = "linux")]
        if self.direct_io {
            let mut slot = crate::internal::aligned::AlignedBuffer::try_new_default(
                self.record_bytes as usize,
            )
            .context("aligned O_DIRECT spill read buffer")?;
            let n = self.record_bytes as usize;
            self.file
                .read_exact_at(&mut slot.as_mut_slice()[..n], offset)
                .with_context(|| format!("failed to read NVMe slot for node {node}"))?;
            dest.copy_from_slice(&slot.as_slice()[..self.payload_bytes]);
            return Ok(());
        }
        self.file
            .read_exact_at(dest, offset)
            .with_context(|| format!("failed to read NVMe slot for node {node}"))
    }

    #[cfg(target_os = "linux")]
    fn load_batch_uring(&self, hits: &[(usize, NodeId)]) -> Result<Vec<FeatureVector>> {
        use crate::internal::uring::{UringLane, batch_read, create_feature_uring};

        let mut guard = self.uring.lock();
        if guard.is_none() {
            let handle = create_feature_uring(self.direct_io)
                .ok_or_else(|| anyhow::anyhow!("io_uring unavailable for NVMe spill tier"))?;
            // Register the spill fd so batch_read can use Fixed ops.
            let mut lane = UringLane::new(handle);
            let _ = lane.handle.register_fd(&self.file);
            *guard = Some(lane);
        }
        let lane = guard.as_mut().expect("populated above");
        let fd = self.file.as_raw_fd();
        let n = hits.len();
        let slot = self.record_bytes as usize;

        let mut reads: Vec<(u64, *mut u8, usize)> = Vec::with_capacity(n);
        if self.direct_io {
            let pool = lane.direct_pool(n, slot)?;
            for (i, &(_, node)) in hits.iter().enumerate() {
                let offset = u64::from(node) * self.record_bytes;
                reads.push((offset, pool.slot_ptr(i), slot));
            }
        } else {
            let total = n
                .checked_mul(slot)
                .ok_or_else(|| anyhow::anyhow!("NVMe batch buffer size overflow"))?;
            let scratch = lane.scratch(total);
            let base = scratch.as_mut_ptr();
            for (i, &(_, node)) in hits.iter().enumerate() {
                let offset = u64::from(node) * self.record_bytes;
                // SAFETY: scratch spans `n * slot` bytes; `i < n`.
                let ptr = unsafe { base.add(i * slot) };
                reads.push((offset, ptr, slot));
            }
        }

        // SAFETY: pointers land in the lane's pool/scratch, kept alive by
        // the exclusive `lane` borrow; batch_read reaps every CQE before
        // returning on both success and error.
        batch_read(&mut lane.handle, fd, &reads)?;

        let mut rows = Vec::with_capacity(n);
        if self.direct_io {
            let pool = lane.direct_pool(n, slot)?;
            for i in 0..n {
                let bytes = &pool.slot_slice(i, slot)[..self.payload_bytes];
                let mut features = vec![0f32; self.dim];
                features.copy_from_slice(bytemuck::cast_slice(bytes));
                rows.push(features);
            }
        } else {
            let scratch = lane.scratch(n * slot);
            for i in 0..n {
                let start = i * slot;
                let bytes = &scratch[start..start + self.payload_bytes];
                let mut features = vec![0f32; self.dim];
                features.copy_from_slice(bytemuck::cast_slice(bytes));
                rows.push(features);
            }
        }
        Ok(rows)
    }

    /// Write one record and mark it present. No fsync: this is a
    /// rebuildable cache, and an fsync per eviction serializes the write
    /// path on device flushes — a crash at worst loses cache entries.
    pub(super) fn save_blocking(&self, node: NodeId, features: &[f32]) -> Result<()> {
        debug_assert_eq!(features.len(), self.dim);
        let offset = u64::from(node) * self.record_bytes;
        // O_DIRECT writes need a full aligned slot; pad with zeros past the
        // payload so the kernel accepts the transfer length.
        #[cfg(target_os = "linux")]
        if self.direct_io {
            // O_DIRECT needs an address-aligned buffer of aligned length.
            let mut slot = crate::internal::aligned::AlignedBuffer::try_new_default(
                self.record_bytes as usize,
            )
            .context("aligned O_DIRECT spill write buffer")?;
            slot.as_mut_slice()[..self.payload_bytes]
                .copy_from_slice(bytemuck::cast_slice(features));
            // Write exactly one stride — AlignedBuffer may round the
            // allocation up past `record_bytes`.
            let n = self.record_bytes as usize;
            self.file
                .write_all_at(&slot.as_slice()[..n], offset)
                .with_context(|| format!("failed to write NVMe slot for node {node}"))?;
        } else {
            self.file
                .write_all_at(bytemuck::cast_slice(features), offset)
                .with_context(|| format!("failed to write NVMe slot for node {node}"))?;
        }
        #[cfg(not(target_os = "linux"))]
        {
            self.file
                .write_all_at(bytemuck::cast_slice(features), offset)
                .with_context(|| format!("failed to write NVMe slot for node {node}"))?;
        }
        self.present.write().insert(node);
        Ok(())
    }
}
