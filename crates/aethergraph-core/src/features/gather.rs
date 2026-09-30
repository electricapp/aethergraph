//! Shared io_uring gather for feature rows.
//!
//! `AsyncFeatureStore` and `SyncFeatureStore` funnel their io_uring batch
//! reads through [`uring_gather_rows`], so the O_DIRECT/buffered branching,
//! buffer-lifetime rules, and dtype decode live in one place.

#![cfg(target_os = "linux")]

use super::header::FeatureDtype;
use crate::graph::NodeId;
use crate::internal::aligned::DIRECT_IO_ALIGNMENT;
use crate::internal::uring::{UringLane, batch_read};
use anyhow::Result;
use std::os::unix::io::RawFd;
use tracing::trace;

#[cfg(feature = "nvme-passthru")]
use crate::internal::nvme::{ExtentMap, NamespaceTarget};

/// NVMe passthrough gather backend: the store file's namespace geometry
/// and extent map, shared by every lane. Each lane opens its own reader
/// (char-device handle plus ring) on first use, so lanes never contend.
///
/// Built once when the feature store opens; used only when the whole
/// request is LBA-resolvable, otherwise the caller stays on the io_uring
/// O_DIRECT path. The first device failure turns the backend off for good:
/// the caller re-reads that batch and every later one through io_uring.
#[cfg(feature = "nvme-passthru")]
pub(crate) struct NvmeGather {
    target: NamespaceTarget,
    extents: ExtentMap,
    disabled: std::sync::atomic::AtomicBool,
}

#[cfg(feature = "nvme-passthru")]
impl NvmeGather {
    /// Probe the backend for `store_file`. `None` when the platform can't
    /// support passthrough (not an NVMe namespace, CoW or unmappable
    /// filesystem, no `/dev/ng*` access) — the caller then relies on the
    /// io_uring path.
    pub(crate) fn build(store_file: &std::fs::File) -> Option<Self> {
        let target = NamespaceTarget::for_file(store_file)?;
        let extents = match ExtentMap::build(store_file, &target) {
            Ok(m) => m,
            Err(e) => {
                trace!("NVMe passthrough disabled: {e}");
                return None;
            }
        };
        // Open one reader now so a node the process cannot use is found
        // at open time, not on the first batch.
        if let Err(e) = crate::internal::nvme::NvmeReader::open(&target) {
            trace!("NVMe passthrough disabled: {e:#}");
            return None;
        }
        Some(Self {
            target,
            extents,
            disabled: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Gather rows via NVMe passthrough into one contiguous decoded
    /// buffer, landing them in `lane`'s aligned pool.
    ///
    /// `Ok(None)` sends the caller to the io_uring path for this batch: a
    /// row that is not LBA-resolvable, a lane that cannot open its reader,
    /// or a failed command. The last two also disable the backend.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gather(
        &self,
        lane: &mut UringLane,
        nodes: &[NodeId],
        features_start_offset: u64,
        feature_size: usize,
        dtype: FeatureDtype,
        feature_dim: usize,
    ) -> Result<Option<Vec<f32>>> {
        use std::sync::atomic::Ordering;

        if self.disabled.load(Ordering::Relaxed) || nodes.is_empty() {
            return Ok(None);
        }
        let lba = u64::from(self.target.lba_bytes());
        // The store file's own offset must be LBA-aligned for device reads
        // to line up, and a row must fit one command: splitting it would
        // give up the batch's single-round-trip property.
        if !features_start_offset.is_multiple_of(lba)
            || !(feature_size as u64).is_multiple_of(lba)
            || feature_size as u64 > u64::from(self.target.max_transfer_bytes())
        {
            return Ok(None);
        }

        // Resolve every row first; one that crosses an extent boundary
        // sends the whole batch to the caller's path.
        let mut reqs: Vec<(u64, *mut u8, u32)> = Vec::with_capacity(nodes.len());
        {
            let pool = lane.direct_pool(nodes.len(), feature_size)?;
            for (i, &node) in nodes.iter().enumerate() {
                let logical = features_start_offset + u64::from(node) * feature_size as u64;
                let Some(device_off) = self.extents.resolve(logical, feature_size as u64) else {
                    return Ok(None);
                };
                reqs.push((device_off, pool.slot_ptr(i), feature_size as u32));
            }
        }

        let Some(reader) = lane.nvme_reader(&self.target) else {
            self.disable("a lane could not open its passthrough reader");
            return Ok(None);
        };
        // SAFETY: each ptr is slot `i` of the lane's pool, which the
        // exclusive `lane` borrow keeps alive and unmoved until this call
        // returns; `read_batch` reaps every submitted command before
        // returning, on success and on error.
        if let Err(e) = unsafe { reader.read_batch(&reqs) } {
            self.disable(&format!("{e:#}"));
            return Ok(None);
        }

        let decoder = dtype.row_decoder();
        let pool = lane.direct_pool(nodes.len(), feature_size)?;
        let mut features = Vec::with_capacity(nodes.len() * feature_dim);
        if decoder.is_f32_passthrough() {
            for i in 0..nodes.len() {
                features.extend_from_slice(pool.slot_slice_f32(i, feature_dim));
            }
        } else {
            features.resize(nodes.len() * feature_dim, 0.0);
            for (i, out) in features.chunks_exact_mut(feature_dim).enumerate() {
                decoder.decode_row(pool.slot_slice(i, feature_size), out);
            }
        }
        Ok(Some(features))
    }

    fn disable(&self, why: &str) {
        if !self
            .disabled
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            tracing::warn!("NVMe passthrough gather disabled, using io_uring: {why}");
        }
    }
}

/// Largest single read a coalesced run may issue. Adjacent rows merge up
/// to this, so one request replaces many while each stays a size the
/// device queue splits cheaply.
const MAX_RUN_BYTES: usize = 1 << 20;

/// Where each requested row lands, and the reads that land them.
///
/// Distinct nodes are laid out back to back in ascending order, so rows
/// with consecutive IDs sit next to each other in the file *and* in the
/// landing buffer and merge into one read. A repeated node is read once.
struct LandingPlan {
    /// Landing position (in rows) of each requested node, in `nodes` order.
    slot_of: Vec<u32>,
    /// `(first node, landing position, rows)` per coalesced read.
    runs: Vec<(NodeId, u32, u32)>,
    /// Distinct rows landed.
    rows: usize,
}

impl LandingPlan {
    fn new(nodes: &[NodeId], feature_size: usize) -> Self {
        let max_run_rows = (MAX_RUN_BYTES / feature_size).max(1) as u32;
        // Positions packed with the node in the high half: one integer sort.
        let mut order: Vec<u64> = nodes
            .iter()
            .enumerate()
            .map(|(i, &n)| (u64::from(n) << 32) | i as u64)
            .collect();
        order.sort_unstable();

        let mut slot_of = vec![0u32; nodes.len()];
        let mut runs: Vec<(NodeId, u32, u32)> = Vec::new();
        let mut rows = 0u32;
        let mut prev: Option<NodeId> = None;
        for packed in order {
            let node = (packed >> 32) as NodeId;
            let idx = packed as u32 as usize;
            if prev != Some(node) {
                // Distinct rows land consecutively, so a run extends exactly
                // when the file rows are consecutive too.
                match runs.last_mut() {
                    Some((first, _, len))
                        if *len < max_run_rows
                            && u64::from(*first) + u64::from(*len) == u64::from(node) =>
                    {
                        *len += 1;
                    }
                    _ => runs.push((node, rows, 1)),
                }
                rows += 1;
                prev = Some(node);
            }
            slot_of[idx] = rows - 1;
        }
        Self {
            slot_of,
            runs,
            rows: rows as usize,
        }
    }

    /// Whether the rows land exactly in request order (ascending, no
    /// repeats), so the landing buffer already is the output layout.
    fn is_identity(&self) -> bool {
        self.rows == self.slot_of.len()
            && self
                .slot_of
                .iter()
                .enumerate()
                .all(|(i, &s)| s as usize == i)
    }
}

/// Read the feature rows for `nodes` from `fd` through `lane`'s ring and
/// decode them into one contiguous `f32` buffer in `nodes` order.
///
/// Distinct rows land packed at `feature_size` stride — in the lane's
/// aligned pool under `direct_io`, in its scratch otherwise — and runs of
/// consecutive node IDs merge into single reads (see [`LandingPlan`]). The
/// O_DIRECT layout makes `feature_size` a multiple of the device alignment,
/// so packed rows stay aligned without padding each to a page. The whole
/// batch goes through one pipelined [`batch_read`] submission.
///
/// Callers bounds-check `nodes` before calling; `feature_size` is the
/// caller's cached `feature_dim * dtype.element_size()`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn uring_gather_rows(
    lane: &mut UringLane,
    fd: RawFd,
    nodes: &[NodeId],
    features_start_offset: u64,
    feature_size: usize,
    direct_io: bool,
    dtype: FeatureDtype,
    feature_dim: usize,
) -> Result<Vec<f32>> {
    debug_assert_eq!(feature_size, feature_dim * dtype.element_size());

    if nodes.is_empty() {
        return Ok(Vec::new());
    }

    let plan = LandingPlan::new(nodes, feature_size);
    let landing_bytes = plan.rows.checked_mul(feature_size).ok_or_else(|| {
        anyhow::anyhow!(
            "buffer size overflow: {} rows * {} bytes",
            plan.rows,
            feature_size
        )
    })?;

    // Landing base: an aligned pool region (whole rounded slots covering
    // the packed rows) or the byte scratch. Neither moves until the next
    // call on this lane.
    let base: *mut u8 = if direct_io {
        let slot = feature_size.next_multiple_of(DIRECT_IO_ALIGNMENT);
        let pool = lane.direct_pool(landing_bytes.div_ceil(slot), feature_size)?;
        debug_assert!(pool.region_len() >= landing_bytes);
        pool.region_ptr()
    } else {
        lane.scratch(landing_bytes).as_mut_ptr()
    };

    let reads: Vec<(u64, *mut u8, usize)> = plan
        .runs
        .iter()
        .map(|&(first, at, len)| {
            let file_offset = features_start_offset + u64::from(first) * feature_size as u64;
            // SAFETY: `at + len <= plan.rows`, so the run lies inside the
            // `landing_bytes` the base was sized for.
            let dest = unsafe { base.add(at as usize * feature_size) };
            (file_offset, dest, len as usize * feature_size)
        })
        .collect();
    // SAFETY: every destination lies in the lane's pool or scratch, which
    // nothing touches until this call returns; batch_read reaps every
    // submitted completion before returning — on success AND on error.
    unsafe { batch_read(&mut lane.handle, fd, &reads)? };
    trace!(
        "Completed {} rows in {} reads via io_uring",
        plan.rows,
        reads.len()
    );

    let decoder = dtype.row_decoder();
    if !direct_io && plan.is_identity() {
        // Landed in request order: decode the scratch as one run.
        let total_elems = nodes.len() * feature_dim;
        if decoder.is_f32_passthrough() {
            return Ok(lane.scratch_f32(total_elems).to_vec());
        }
        let mut features = vec![0f32; total_elems];
        decoder.decode_row(lane.scratch(landing_bytes), &mut features);
        return Ok(features);
    }

    // SAFETY: `base..base + landing_bytes` is inside the pool region or
    // scratch (sized above), and batch_read returned Ok, so every run —
    // together covering exactly that span — was read in full.
    let landed = unsafe { std::slice::from_raw_parts(base.cast_const(), landing_bytes) };
    let mut features: Vec<f32> = Vec::with_capacity(nodes.len() * feature_dim);
    if decoder.is_f32_passthrough() {
        // Both landing buffers are f32-aligned and the stride is a whole
        // number of lanes, so every row views as `[f32]` directly.
        for &slot in &plan.slot_of {
            let start = slot as usize * feature_size;
            features.extend_from_slice(bytemuck::cast_slice(&landed[start..start + feature_size]));
        }
    } else {
        features.resize(nodes.len() * feature_dim, 0.0);
        for (out, &slot) in features.chunks_exact_mut(feature_dim).zip(&plan.slot_of) {
            let start = slot as usize * feature_size;
            decoder.decode_row(&landed[start..start + feature_size], out);
        }
    }
    Ok(features)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_merges_consecutive_ids_and_reads_repeats_once() {
        let plan = LandingPlan::new(&[7, 3, 4, 5, 9, 4, 10, 3], 16);
        assert_eq!(plan.rows, 6); // 3 4 5 7 9 10
        assert_eq!(plan.runs, vec![(3, 0, 3), (7, 3, 1), (9, 4, 2)]);
        assert_eq!(plan.slot_of, vec![3, 0, 1, 2, 4, 1, 5, 0]);
        assert!(!plan.is_identity());
        assert!(LandingPlan::new(&[1, 2, 5], 16).is_identity());
        assert!(!LandingPlan::new(&[1, 1], 16).is_identity());
    }

    #[test]
    fn plan_caps_a_run_at_the_read_limit() {
        let rows = MAX_RUN_BYTES / 4096;
        let nodes: Vec<NodeId> = (0..rows as u32 * 2 + 1).collect();
        let plan = LandingPlan::new(&nodes, 4096);
        let lens: Vec<u32> = plan.runs.iter().map(|r| r.2).collect();
        assert_eq!(lens, vec![rows as u32, rows as u32, 1]);
        assert!(plan.is_identity());
    }
}
