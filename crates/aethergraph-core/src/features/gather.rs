//! Shared io_uring gather for feature rows.
//!
//! `AsyncFeatureStore` and `SyncFeatureStore` funnel their io_uring batch
//! reads through [`uring_gather_rows`], so the O_DIRECT/buffered branching,
//! buffer-lifetime rules, and dtype decode live in one place.

#![cfg(target_os = "linux")]

use super::header::FeatureDtype;
use crate::graph::NodeId;
use crate::internal::uring::{UringLane, batch_read};
use anyhow::Result;
use std::os::unix::io::RawFd;
use tracing::trace;

#[cfg(feature = "nvme-passthru")]
use crate::internal::nvme::{ExtentMap, NvmeReader};

/// NVMe passthrough gather backend: a char-device reader plus the store
/// file's extent map. Built once when the feature store opens; used only
/// when the whole request is LBA-resolvable, otherwise the caller stays
/// on the io_uring O_DIRECT path.
#[cfg(feature = "nvme-passthru")]
pub(crate) struct NvmeGather {
    reader: NvmeReader,
    extents: ExtentMap,
}

#[cfg(feature = "nvme-passthru")]
impl NvmeGather {
    /// Try to build the backend for `store_file`. `Ok(None)` when the
    /// platform can't support passthrough (no `/dev/ng*`, CoW filesystem,
    /// missing privilege) — the caller then relies on the io_uring path.
    pub(crate) fn build(store_file: &std::fs::File) -> Result<Option<Self>> {
        let Some(reader) = NvmeReader::open_for(store_file)? else {
            return Ok(None);
        };
        let extents = match ExtentMap::build(store_file) {
            Ok(m) => m,
            // A CoW / unmappable filesystem is a fallback case, not fatal.
            Err(e) => {
                trace!("NVMe passthrough disabled: {e}");
                return Ok(None);
            }
        };
        Ok(Some(Self { reader, extents }))
    }

    /// Gather rows via NVMe passthrough into one contiguous decoded
    /// buffer. Every read must resolve to a single device extent and be
    /// LBA-aligned; a row that doesn't returns `Ok(None)` so the caller
    /// falls back for the whole batch (keeping the fast path branch-free).
    ///
    /// # Safety
    /// `landing` must be an aligned buffer pool large enough for
    /// `nodes.len()` slots of `feature_size` bytes; the caller owns it.
    pub(crate) fn gather(
        &mut self,
        landing: &mut crate::internal::aligned::AlignedBufferPool,
        nodes: &[NodeId],
        features_start_offset: u64,
        feature_size: usize,
        dtype: FeatureDtype,
        feature_dim: usize,
    ) -> Result<Option<Vec<f32>>> {
        let lba = self.reader.lba_bytes() as u64;
        // The store file's own offset must be LBA-aligned for device
        // reads to line up; feature_size likewise (O_DIRECT layout
        // already guarantees 512-alignment for the direct path).
        if !features_start_offset.is_multiple_of(lba) || !(feature_size as u64).is_multiple_of(lba)
        {
            return Ok(None);
        }

        // Resolve every row to a device offset first; bail to the caller
        // on the first row that crosses an extent boundary.
        let mut device_offsets = Vec::with_capacity(nodes.len());
        for &node in nodes {
            let logical = features_start_offset + (node as u64) * (feature_size as u64);
            match self.extents.resolve(logical, feature_size as u64) {
                Some((device_off, _)) => device_offsets.push(device_off),
                None => return Ok(None),
            }
        }

        // A row larger than one NVMe command has to go back to the file
        // path; splitting it here would submit two commands per row and
        // give up the batch's single-round-trip property.
        if feature_size as u64 > u64::from(self.reader.max_transfer_bytes()) {
            return Ok(None);
        }

        let mut features = vec![0f32; nodes.len() * feature_dim];
        let reqs: Vec<(u64, *mut u8, u32)> = device_offsets
            .iter()
            .enumerate()
            .map(|(i, &device_off)| (device_off, landing.slot_ptr(i), feature_size as u32))
            .collect();
        // SAFETY: each ptr is slot `i` of `landing`, valid for
        // `feature_size` bytes and kept alive by the exclusive borrow
        // until this call returns; `read_batch` reaps every submitted
        // command before returning, on success and on error.
        unsafe { self.reader.read_batch(&reqs)? };
        for i in 0..nodes.len() {
            let row = landing.slot_slice(i, feature_size);
            dtype.decode_row(row, &mut features[i * feature_dim..(i + 1) * feature_dim]);
        }
        Ok(Some(features))
    }
}

/// Read the feature rows for `nodes` from `fd` through `lane`'s ring and
/// decode them into one contiguous `f32` buffer in `nodes` order.
///
/// With `direct_io`, each row lands in its own slot of the lane's
/// persistent aligned pool (O_DIRECT requires aligned landing addresses);
/// otherwise rows land back-to-back in the lane's scratch bytes and decode
/// as one contiguous run. Either way the whole batch goes through a single
/// pipelined [`batch_read`] submission with no per-row temporaries.
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

    // An F32 payload is already `f32` lanes on disk, so rows are appended
    // straight into reserved-but-unwritten capacity and the buffer is
    // never zeroed on its way to being fully overwritten. Both landing
    // buffers are `f32`-aligned by construction — the O_DIRECT pool to a
    // page, the buffered scratch through its `Vec<f32>` backing — so
    // those views are infallible rather than alignment-dependent. The
    // half-width payloads still need somewhere to upcast into, and keep a
    // zeroed buffer.
    let decoder = dtype.row_decoder();
    let total_elems = nodes.len() * feature_dim;
    let mut features: Vec<f32> = if decoder.is_f32_passthrough() {
        Vec::with_capacity(total_elems)
    } else {
        vec![0f32; total_elems]
    };

    if direct_io {
        let total_slots = nodes.len();
        // Build reads against the lane's persistent aligned pool.
        let mut reads: Vec<(u64, *mut u8, usize)> = Vec::with_capacity(total_slots);
        {
            let pool = lane.direct_pool(total_slots, feature_size)?;
            for (i, &node) in nodes.iter().enumerate() {
                let file_offset = features_start_offset + (node as u64) * (feature_size as u64);
                reads.push((file_offset, pool.slot_ptr(i), feature_size));
            }
        }
        // SAFETY: each ptr points into the lane's pool, which the exclusive
        // `lane` borrow keeps alive until after this call; batch_read reaps
        // every submitted completion before returning — on success AND on
        // error.
        batch_read(&mut lane.handle, fd, &reads)?;
        trace!("Completed {} feature reads via io_uring", reads.len());

        let pool = lane.direct_pool(total_slots, feature_size)?;
        for i in 0..total_slots {
            if decoder.is_f32_passthrough() {
                features.extend_from_slice(pool.slot_slice_f32(i, feature_dim));
            } else {
                decoder.decode_row(
                    pool.slot_slice(i, feature_size),
                    &mut features[i * feature_dim..(i + 1) * feature_dim],
                );
            }
        }
    } else {
        let total_size = nodes.len().checked_mul(feature_size).ok_or_else(|| {
            anyhow::anyhow!(
                "buffer size overflow: {} nodes * {} bytes",
                nodes.len(),
                feature_size
            )
        })?;
        let mut reads: Vec<(u64, *mut u8, usize)> = Vec::with_capacity(nodes.len());
        {
            let scratch = lane.scratch(total_size);
            let base = scratch.as_mut_ptr();
            for (i, &node) in nodes.iter().enumerate() {
                let file_offset = features_start_offset + (node as u64) * (feature_size as u64);
                // SAFETY: `i < nodes.len()` and the scratch spans
                // `nodes.len() * feature_size` bytes.
                let buf_ptr = unsafe { base.add(i * feature_size) };
                reads.push((file_offset, buf_ptr, feature_size));
            }
        }
        // SAFETY: every ptr points into the lane's scratch, which the
        // exclusive `lane` borrow keeps alive until after this call;
        // batch_read reaps every submitted completion before returning.
        batch_read(&mut lane.handle, fd, &reads)?;
        trace!("Completed {} feature reads via io_uring", reads.len());

        if decoder.is_f32_passthrough() {
            features.extend_from_slice(lane.scratch_f32(total_elems));
        } else {
            decoder.decode_row(lane.scratch(total_size), &mut features);
        }
    }

    Ok(features)
}
