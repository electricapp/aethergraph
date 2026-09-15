//! Portable seqlock acceptance predicate and slot geometry, shared by the
//! host table, the RDMA wire format, and the device readers.
//!
//! The geometry lives here so hosts without CUDA or `libibverbs` still
//! compile against one definition. `common.cuh` mirrors
//! [`FEATURE_SLOT_HEAD_BYTES`] as `AETHER_FEATURE_OFFSET`; drift fails
//! `aether-stream`'s `common_cuh_matches_core_layout`.

/// Returns whether a head/tail pair describes a published, stable row.
#[must_use]
pub const fn cpu_seqlock_accept(head: u64, tail: u64) -> bool {
    head == tail && head != 0 && head & 1 == 0
}

/// Bytes from slot start to the feature payload: `u64` head plus 8 of pad.
///
/// The pad puts the payload on a 16-byte boundary inside a slot whose stride
/// is a multiple of 64, which is what licenses `ld.global.*.v4` in the device
/// gather — whole sectors, a quarter of the instructions.
pub const FEATURE_SLOT_HEAD_BYTES: usize = 16;

/// Byte offset from slot start to the `u64` tail version.
#[must_use]
pub const fn feature_slot_tail_offset(feature_dim: usize) -> usize {
    let after_features = FEATURE_SLOT_HEAD_BYTES + feature_dim * size_of::<f32>();
    after_features.next_multiple_of(8)
}

/// Compact slot size, before the table rounds the stride to a cache line.
#[must_use]
pub const fn feature_slot_size(feature_dim: usize) -> usize {
    feature_slot_tail_offset(feature_dim) + 8
}

/// Slot stride alignment. One cache line, not one page: the table registers
/// a single MR and addresses slots by offset, so page-strided slots would
/// only spend DRAM on dead padding.
pub const FEATURE_SLOT_STRIDE_ALIGN: usize = 64;

/// Distance between consecutive slots in a table or staging region. Anything
/// staging for the device gather uses this: a multiple of 64, so every slot's
/// payload base stays 16-aligned.
#[must_use]
pub const fn feature_slot_stride(feature_dim: usize) -> usize {
    feature_slot_size(feature_dim).next_multiple_of(FEATURE_SLOT_STRIDE_ALIGN)
}

#[cfg(test)]
mod tests {
    use super::{
        FEATURE_SLOT_HEAD_BYTES, FEATURE_SLOT_STRIDE_ALIGN, cpu_seqlock_accept, feature_slot_size,
        feature_slot_stride, feature_slot_tail_offset,
    };

    #[test]
    fn accepts_only_published_even_versions() {
        assert!(cpu_seqlock_accept(2, 2));
        assert!(!cpu_seqlock_accept(0, 0));
        assert!(!cpu_seqlock_accept(3, 3));
        assert!(!cpu_seqlock_accept(2, 4));
    }

    #[test]
    fn payload_is_sixteen_byte_aligned_within_the_slot_stride() {
        // What `ld.global.cs.v4.u32` needs: a stride keeping every slot base
        // 16-aligned, and a head keeping every payload base 16-aligned in it.
        assert_eq!(FEATURE_SLOT_STRIDE_ALIGN % 16, 0);
        assert_eq!(FEATURE_SLOT_HEAD_BYTES % 16, 0);
        for dim in [1usize, 3, 4, 5, 128, 256, 768] {
            let stride = feature_slot_stride(dim);
            assert_eq!(stride % 16, 0, "stride for dim {dim}");
            assert!(stride >= feature_slot_size(dim));
        }
    }

    #[test]
    fn tail_offset_clears_the_payload_and_stays_eight_aligned() {
        for dim in [1usize, 3, 4, 5, 128, 256, 768] {
            let tail = feature_slot_tail_offset(dim);
            assert!(tail >= FEATURE_SLOT_HEAD_BYTES + dim * 4);
            assert_eq!(tail % 8, 0);
            assert_eq!(feature_slot_size(dim), tail + 8);
        }
        assert_eq!(feature_slot_tail_offset(768), 16 + 768 * 4);
        assert_eq!(feature_slot_tail_offset(3), 32);
    }
}
