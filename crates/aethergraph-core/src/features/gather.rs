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

    let mut features = vec![0f32; nodes.len() * feature_dim];

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
            let row = pool.slot_slice(i, feature_size);
            dtype.decode_row(row, &mut features[i * feature_dim..(i + 1) * feature_dim]);
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

        let scratch = lane.scratch(total_size);
        dtype.decode_row(scratch, &mut features);
    }

    Ok(features)
}
