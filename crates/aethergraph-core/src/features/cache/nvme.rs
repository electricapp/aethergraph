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
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;

/// Single-file NVMe spill tier.
///
/// One sparse file holds every spilled node at byte offset
/// `node * record_bytes`, so a spill or a reload is one positional
/// read/write on a single always-open file — no per-node opens, closes,
/// inodes, or filesystem-block roundups. The file is anonymous (created
/// already unlinked) and private to this tier: two caches given the same
/// directory never see each other's records, and the space returns to the
/// filesystem when the tier drops. Presence is tracked in memory — a cache
/// is rebuildable by definition, so a cold start just misses.
///
/// `record_bytes` is the on-disk stride. Under O_DIRECT it is the `dim * 4`
/// payload rounded up to the device alignment, so a batch gather can use
/// the aligned landing pool; buffered records are packed at the payload
/// size. Records are written at most with one value per node (features are
/// immutable per node in a training run), so a concurrent same-slot
/// read/write can only race identical bytes.
pub(super) struct NvmeTier {
    file: File,
    /// On-disk bytes per node (payload, plus O_DIRECT padding).
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
    /// Whether the file is open O_DIRECT. Cleared, for good, the first time
    /// the device rejects a direct transfer's alignment; the padded stride
    /// stays valid for buffered I/O.
    #[cfg(target_os = "linux")]
    direct_io: AtomicBool,
}

impl NvmeTier {
    pub(super) fn open(dir: &Path, dim: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create NVMe cache directory: {}", dir.display()))?;
        let payload_bytes = dim
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("feature dim {dim} overflows record size"))?;
        let file = anonymous_spill_file(dir)?;

        #[cfg(target_os = "linux")]
        let (record_bytes, direct_io) = {
            let align = crate::internal::uring::DirectIoAlignment::probe(&file).bytes();
            match crate::features::set_direct_io(&file, true) {
                Ok(()) => ((payload_bytes.div_ceil(align) * align) as u64, true),
                Err(e) => {
                    tracing::debug!(
                        "O_DIRECT unavailable for the spill file in {} ({e}); buffered",
                        dir.display()
                    );
                    (payload_bytes as u64, false)
                }
            }
        };
        #[cfg(not(target_os = "linux"))]
        let record_bytes = payload_bytes as u64;

        Ok(Self {
            file,
            record_bytes,
            payload_bytes,
            dim,
            present: RwLock::new(FxHashSet::default()),
            #[cfg(target_os = "linux")]
            uring: Mutex::new(None),
            #[cfg(target_os = "linux")]
            direct_io: AtomicBool::new(direct_io),
        })
    }

    /// Whether `node` has a record in the tier.
    pub(super) fn contains(&self, node: NodeId) -> bool {
        self.present.read().contains(&node)
    }

    /// Read one record, or `None` when the node was never spilled.
    pub(super) fn load_blocking(&self, node: NodeId) -> Result<Option<FeatureVector>> {
        if !self.contains(node) {
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
            }
            out.push((node, None));
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
        if self.direct_io.load(Ordering::Relaxed) {
            let n = self.record_bytes as usize;
            let mut slot = crate::internal::aligned::AlignedBuffer::try_new_default(n)
                .context("aligned O_DIRECT spill read buffer")?;
            match self
                .file
                .read_exact_at(&mut slot.as_mut_slice()[..n], offset)
            {
                Ok(()) => {
                    dest.copy_from_slice(&slot.as_slice()[..self.payload_bytes]);
                    return Ok(());
                }
                Err(e) if !self.demote_on_einval(&e) => {
                    return Err(e)
                        .with_context(|| format!("failed to read NVMe slot for node {node}"));
                }
                // Demoted: retry below through the now-buffered file.
                Err(_) => {}
            }
        }
        self.file
            .read_exact_at(dest, offset)
            .with_context(|| format!("failed to read NVMe slot for node {node}"))
    }

    /// If `e` is the device rejecting an O_DIRECT transfer's alignment,
    /// drop O_DIRECT for good and report that the caller should retry
    /// buffered. The padded stride stays valid for buffered I/O, and the
    /// ring (built for the direct file) is rebuilt on next use.
    #[cfg(target_os = "linux")]
    fn demote_on_einval(&self, e: &std::io::Error) -> bool {
        if e.raw_os_error() != Some(libc::EINVAL) {
            return false;
        }
        let mut lane = self.uring.lock();
        if self.direct_io.swap(false, Ordering::Relaxed) {
            if let Err(err) = crate::features::set_direct_io(&self.file, false) {
                tracing::warn!("failed to drop O_DIRECT from the spill file: {err}");
            }
            *lane = None;
            tracing::warn!("NVMe spill tier: device rejected O_DIRECT alignment; buffered");
        }
        true
    }

    #[cfg(target_os = "linux")]
    fn load_batch_uring(&self, hits: &[(usize, NodeId)]) -> Result<Vec<FeatureVector>> {
        use crate::internal::uring::{UringLane, batch_read, create_feature_uring};

        let mut guard = self.uring.lock();
        let direct_io = self.direct_io.load(Ordering::Relaxed);
        if guard.is_none() {
            let handle = create_feature_uring(&self.file)
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
        if direct_io {
            // O_DIRECT reads whole aligned records into aligned slots.
            let pool = lane.direct_pool(n, slot)?;
            for (i, &(_, node)) in hits.iter().enumerate() {
                let offset = u64::from(node) * self.record_bytes;
                reads.push((offset, pool.slot_ptr(i), slot));
            }
        } else {
            // Buffered reads fetch just the payload, which is all a
            // buffered write stored.
            let total = n
                .checked_mul(self.payload_bytes)
                .ok_or_else(|| anyhow::anyhow!("NVMe batch buffer size overflow"))?;
            let scratch = lane.scratch(total);
            let base = scratch.as_mut_ptr();
            for (i, &(_, node)) in hits.iter().enumerate() {
                let offset = u64::from(node) * self.record_bytes;
                // SAFETY: scratch spans `n * payload_bytes` bytes; `i < n`.
                let ptr = unsafe { base.add(i * self.payload_bytes) };
                reads.push((offset, ptr, self.payload_bytes));
            }
        }

        // SAFETY: pointers land in the lane's pool/scratch, kept alive by
        // the exclusive `lane` borrow; batch_read reaps every CQE before
        // returning on both success and error.
        unsafe { batch_read(&mut lane.handle, fd, &reads)? };

        let rows = if direct_io {
            let pool = lane.direct_pool(n, slot)?;
            (0..n)
                .map(|i| pool.slot_slice_f32(i, self.dim).to_vec())
                .collect()
        } else {
            lane.scratch_f32(n * self.dim)
                .chunks_exact(self.dim)
                .map(<[f32]>::to_vec)
                .collect()
        };
        Ok(rows)
    }

    /// Write one record and mark it present. No fsync: this is a
    /// rebuildable cache, and an fsync per eviction serializes the write
    /// path on device flushes — a crash at worst loses cache entries.
    pub(super) fn save_blocking(&self, node: NodeId, features: &[f32]) -> Result<()> {
        debug_assert_eq!(features.len(), self.dim);
        let offset = u64::from(node) * self.record_bytes;
        #[cfg(target_os = "linux")]
        if self.direct_io.load(Ordering::Relaxed) {
            // O_DIRECT needs an address-aligned buffer of aligned length:
            // one full stride, the payload followed by zeroed padding.
            let n = self.record_bytes as usize;
            let mut slot = crate::internal::aligned::AlignedBuffer::try_new_default(n)
                .context("aligned O_DIRECT spill write buffer")?;
            let buf = &mut slot.as_mut_slice()[..n];
            buf[..self.payload_bytes].copy_from_slice(bytemuck::cast_slice(features));
            buf[self.payload_bytes..].fill(0);
            match self.file.write_all_at(buf, offset) {
                Ok(()) => {
                    self.present.write().insert(node);
                    return Ok(());
                }
                Err(e) if !self.demote_on_einval(&e) => {
                    return Err(e)
                        .with_context(|| format!("failed to write NVMe slot for node {node}"));
                }
                // Demoted: retry below through the now-buffered file.
                Err(_) => {}
            }
        }
        self.file
            .write_all_at(bytemuck::cast_slice(features), offset)
            .with_context(|| format!("failed to write NVMe slot for node {node}"))?;
        self.present.write().insert(node);
        Ok(())
    }
}

/// Create a spill file unique to one tier and already unlinked.
///
/// `O_TMPFILE` makes it anonymous from the start; where the filesystem
/// lacks it, a fresh `O_EXCL` name is created and unlinked at once.
fn anonymous_spill_file(dir: &Path) -> Result<File> {
    use std::fs::OpenOptions;
    use std::os::unix::fs::OpenOptionsExt;

    #[cfg(target_os = "linux")]
    match OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_TMPFILE)
        .mode(0o600)
        .open(dir)
    {
        Ok(f) => return Ok(f),
        Err(e) => tracing::debug!("O_TMPFILE unavailable in {} ({e})", dir.display()),
    }

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let pid = std::process::id();
    loop {
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = dir.join(format!(".aethergraph-spill.{pid}.{n}"));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
        {
            Ok(f) => {
                std::fs::remove_file(&path)
                    .with_context(|| format!("failed to unlink spill file {}", path.display()))?;
                return Ok(f);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(e).with_context(|| {
                    format!("failed to create NVMe spill file in {}", dir.display())
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records round-trip whatever I/O mode the host filesystem allows,
    /// both singly and through the batch gather — including the record at
    /// the highest offset, whose read ends at end of file.
    #[test]
    fn records_round_trip_through_every_read_path() {
        let dir = tempfile::tempdir().unwrap();
        // 100 lanes: a 400-byte payload, not a multiple of any alignment.
        let tier = NvmeTier::open(dir.path(), 100).unwrap();
        for node in [0u32, 7, 3, 12] {
            tier.save_blocking(node, &vec![node as f32 + 0.5; 100])
                .unwrap();
        }
        assert!(tier.contains(12) && !tier.contains(4));
        assert_eq!(tier.load_blocking(12).unwrap(), Some(vec![12.5; 100]));
        assert_eq!(tier.load_blocking(4).unwrap(), None);

        let got = tier.load_batch(&[12, 4, 0, 7, 12]).unwrap();
        let want: Vec<Option<Vec<f32>>> = [Some(12.5), None, Some(0.5), Some(7.5), Some(12.5)]
            .into_iter()
            .map(|v| v.map(|x| vec![x; 100]))
            .collect();
        assert_eq!(got.into_iter().map(|(_, r)| r).collect::<Vec<_>>(), want);
    }

    /// The spill file is anonymous: nothing is left in the directory, and a
    /// second tier over the same directory starts empty and stays
    /// independent of the first.
    #[test]
    fn tiers_sharing_a_directory_are_independent() {
        let dir = tempfile::tempdir().unwrap();
        let a = NvmeTier::open(dir.path(), 4).unwrap();
        a.save_blocking(3, &[1.0; 4]).unwrap();
        let b = NvmeTier::open(dir.path(), 4).unwrap();
        b.save_blocking(9, &[2.0; 4]).unwrap();

        assert_eq!(a.load_blocking(3).unwrap(), Some(vec![1.0; 4]));
        assert_eq!(b.load_blocking(9).unwrap(), Some(vec![2.0; 4]));
        assert_eq!(b.load_blocking(3).unwrap(), None);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}
