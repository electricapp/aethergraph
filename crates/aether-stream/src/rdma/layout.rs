//! Feature-table geometry parsed once from an advertised [`FeatureSchema`].
//!
//! The schema arrives off the network. [`SlotGeometry`] and [`RemoteTable`]
//! are its validated forms: every offset a READ, a staging region, or the
//! device validator derives from them is in bounds by construction, so the
//! gather paths carry these types inward instead of re-checking raw fields.

use crate::feature_table::FeatureSchema;
use aethergraph_core::{FEATURE_SLOT_HEAD_BYTES, feature_slot_stride};
use std::io;

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Slot stride alignment the device gather's `ld.*.v4` loads need: with the
/// 16-byte head, every slot's payload base stays 16-aligned.
const STRIDE_ALIGN: usize = 16;

/// Per-slot layout: `[head u64][pad][f32 × feature_dim][pad][tail u64]` at
/// `stride` bytes per slot, the payload at [`FEATURE_SLOT_HEAD_BYTES`].
///
/// Construction proves the tail sits inside the stride, the stride keeps
/// payloads 16-aligned, and every offset fits the `i32` the device kernels
/// index with and the `u32` an RDMA READ length carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotGeometry {
    feature_dim: usize,
    stride: usize,
    tail_offset: usize,
}

impl SlotGeometry {
    /// Validate `feature_dim` features at `stride` bytes per slot.
    pub fn new(feature_dim: usize, stride: usize) -> io::Result<Self> {
        if feature_dim == 0 {
            return Err(invalid("feature_dim must be > 0".into()));
        }
        // `feature_slot_tail_offset`, checked: the payload end rounded up to 8.
        let tail_offset = feature_dim
            .checked_mul(4)
            .and_then(|b| b.checked_add(FEATURE_SLOT_HEAD_BYTES + 7))
            .map(|end| end & !7)
            .ok_or_else(|| invalid(format!("feature_dim {feature_dim} overflows the slot")))?;
        let live = tail_offset + 8;
        if stride < live {
            return Err(invalid(format!(
                "slot stride {stride} is shorter than the {live}-byte slot for dim {feature_dim}"
            )));
        }
        if !stride.is_multiple_of(STRIDE_ALIGN) {
            return Err(invalid(format!(
                "slot stride {stride} is not a multiple of {STRIDE_ALIGN}"
            )));
        }
        if i32::try_from(stride).is_err() {
            return Err(invalid(format!("slot stride {stride} exceeds i32::MAX")));
        }
        Ok(Self {
            feature_dim,
            stride,
            tail_offset,
        })
    }

    /// The table's own stride for `feature_dim`
    /// ([`aethergraph_core::feature_slot_stride`]).
    pub fn packed(feature_dim: usize) -> io::Result<Self> {
        if feature_dim == 0 || feature_dim > (i32::MAX as usize) / 4 {
            return Err(invalid(format!("feature_dim {feature_dim} out of range")));
        }
        Self::new(feature_dim, feature_slot_stride(feature_dim))
    }

    /// Parse an advertised schema's slot fields. The node count is the
    /// table's business; see [`RemoteTable::parse`].
    pub fn from_schema(schema: &FeatureSchema) -> io::Result<Self> {
        if schema.feature_offset_in_slot != FEATURE_SLOT_HEAD_BYTES {
            return Err(invalid(format!(
                "advertised feature offset {} != {FEATURE_SLOT_HEAD_BYTES}: peer built with a \
                 different slot layout",
                schema.feature_offset_in_slot
            )));
        }
        let geometry = Self::new(schema.feature_dim, schema.slot_size)?;
        if schema.tail_offset_in_slot != geometry.tail_offset {
            return Err(invalid(format!(
                "advertised tail offset {} != {} for dim {}: peer built with a different slot \
                 layout",
                schema.tail_offset_in_slot, geometry.tail_offset, schema.feature_dim
            )));
        }
        Ok(geometry)
    }

    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Bytes between consecutive slots.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Byte offset of the tail version.
    pub fn tail_offset(&self) -> usize {
        self.tail_offset
    }

    /// Byte offset of the payload.
    pub fn feature_offset(&self) -> usize {
        FEATURE_SLOT_HEAD_BYTES
    }

    /// Bytes from the slot start through the tail version: what a READ must
    /// fetch. The rest of the stride is padding.
    pub fn live_len(&self) -> u32 {
        // In range: `stride >= live`, and `stride` fits i32.
        (self.tail_offset + 8) as u32
    }
}

/// A remote feature table: MR base and key plus validated geometry.
///
/// [`Self::slot_addr`] is the one bounds check a gather needs — the table
/// span `[base, base + node_count * stride)` was proven not to wrap here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteTable {
    base_addr: u64,
    rkey: u32,
    node_count: u64,
    geometry: SlotGeometry,
}

impl RemoteTable {
    /// Validate an advertisement's table fields.
    pub fn parse(base_addr: u64, rkey: u32, schema: &FeatureSchema) -> io::Result<Self> {
        let geometry = SlotGeometry::from_schema(schema)?;
        let node_count = schema.node_count as u64;
        if node_count == 0 {
            return Err(invalid("advertised table has no nodes".into()));
        }
        node_count
            .checked_mul(geometry.stride as u64)
            .and_then(|len| base_addr.checked_add(len))
            .ok_or_else(|| {
                invalid(format!(
                    "advertised table ({node_count} slots of {} bytes at {base_addr:#x}) wraps \
                     the address space",
                    geometry.stride
                ))
            })?;
        Ok(Self {
            base_addr,
            rkey,
            node_count,
            geometry,
        })
    }

    pub fn geometry(&self) -> &SlotGeometry {
        &self.geometry
    }

    pub fn rkey(&self) -> u32 {
        self.rkey
    }

    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }

    pub fn node_count(&self) -> u64 {
        self.node_count
    }

    /// Remote address of `node`'s slot, or an error naming the bound.
    pub fn slot_addr(&self, node: u64) -> io::Result<u64> {
        if node >= self.node_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("node {node} out of range (node_count {})", self.node_count),
            ));
        }
        // No overflow: node < node_count and parse proved the span fits.
        Ok(self.base_addr + node * self.geometry.stride as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aethergraph_core::feature_slot_tail_offset;

    fn schema(node_count: usize, dim: usize) -> FeatureSchema {
        FeatureSchema {
            node_count,
            feature_dim: dim,
            slot_size: feature_slot_stride(dim),
            feature_offset_in_slot: FEATURE_SLOT_HEAD_BYTES,
            tail_offset_in_slot: feature_slot_tail_offset(dim),
        }
    }

    #[test]
    fn accepts_the_tables_own_layout() {
        for dim in [1usize, 3, 4, 5, 128, 768] {
            let g = SlotGeometry::from_schema(&schema(8, dim)).unwrap();
            assert_eq!(g.stride(), feature_slot_stride(dim));
            assert_eq!(g.tail_offset(), feature_slot_tail_offset(dim));
            assert_eq!(g.live_len() as usize, feature_slot_tail_offset(dim) + 8);
            assert_eq!(g, SlotGeometry::packed(dim).unwrap());
        }
    }

    /// A peer from before the 16-byte head: its tail sits 8 bytes earlier
    /// than this build's, so a READ sized from its schema would miss the
    /// tail this build's validator reads.
    #[test]
    fn rejects_a_peer_with_a_different_head() {
        let mut s = schema(8, 128);
        s.feature_offset_in_slot = 8;
        s.tail_offset_in_slot -= 8;
        assert!(SlotGeometry::from_schema(&s).is_err());

        let mut s = schema(8, 128);
        s.tail_offset_in_slot += 8;
        assert!(SlotGeometry::from_schema(&s).is_err());
    }

    #[test]
    fn rejects_strides_that_break_reads_or_vector_loads() {
        let dim = 128;
        let live = feature_slot_tail_offset(dim) + 8;
        // Zero, short, and misaligned strides.
        assert!(SlotGeometry::new(dim, 0).is_err());
        assert!(SlotGeometry::new(dim, live - 8).is_err());
        assert!(SlotGeometry::new(dim, live.next_multiple_of(16) + 8).is_err());
        // Past what the kernels index with.
        assert!(SlotGeometry::new(dim, (i32::MAX as usize + 1).next_multiple_of(16)).is_err());
        assert!(SlotGeometry::new(0, 64).is_err());
        assert!(SlotGeometry::new(usize::MAX / 2, 64).is_err());
        assert!(SlotGeometry::packed(usize::MAX / 4).is_err());
    }

    #[test]
    fn table_bounds_are_checked_once() {
        let t = RemoteTable::parse(0x1000, 7, &schema(4, 4)).unwrap();
        let stride = feature_slot_stride(4) as u64;
        assert_eq!(t.slot_addr(0).unwrap(), 0x1000);
        assert_eq!(t.slot_addr(3).unwrap(), 0x1000 + 3 * stride);
        assert!(t.slot_addr(4).is_err());
        assert!(t.slot_addr(u64::MAX).is_err());

        assert!(RemoteTable::parse(0x1000, 7, &schema(0, 4)).is_err());
        // A span that wraps u64 is rejected before any address is formed.
        assert!(RemoteTable::parse(u64::MAX - 64, 7, &schema(4, 4)).is_err());
        assert!(RemoteTable::parse(0, 7, &schema(usize::MAX / 32, 4)).is_err());
    }
}
