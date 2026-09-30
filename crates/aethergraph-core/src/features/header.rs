//! Feature-file header format + parsing/validation.
//!
//! The binary feature file layout:
//!
//! ```text
//!   [0..8)     magic: b"AETHFEAT"
//!   [8..16)    num_nodes: u64 le
//!   [16..24)   feature_dim: u64 le
//!   [24..32)   features_start_offset: u64 le, > 32 (writers pick an
//!              O_DIRECT-aligned offset, currently 512)
//!   [32]       dtype tag (0 = F32, 1 = F16, 2 = BF16)
//!   [offset..) feature payload: num_nodes × feature_dim × elements (little-endian)
//! ```

use anyhow::{Context, Result};
use std::fs::File;
use std::os::unix::fs::FileExt;

/// Data type for stored features.
///
/// `BF16` trades mantissa bits for f32's exponent range: it truncates to
/// the top 16 bits of an f32, so upcasting is a shift rather than a
/// format conversion and no value ever overflows to infinity. That makes
/// it the natural half-width choice for embeddings trained in bf16, where
/// F16's narrower exponent would flush small magnitudes to zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FeatureDtype {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
}

impl FeatureDtype {
    /// Bytes per element.
    pub const fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }

    /// Decode one row, resolving the dispatch inline.
    ///
    /// F32 payloads are a straight byte copy into the `f32` destination
    /// (no source-alignment requirement); the half-width payloads upcast
    /// through their SIMD-dispatched converters. `src.len()` must equal
    /// `dst.len() * self.element_size()` — every branch panics otherwise.
    ///
    /// For loops over many rows, hoist [`FeatureDtype::row_decoder`] out of
    /// the loop instead.
    #[inline]
    pub(crate) fn decode_row(self, src: &[u8], dst: &mut [f32]) {
        self.row_decoder().decode_row(src, dst);
    }

    /// Resolve the row decoder once, for loops that decode many rows.
    ///
    /// The F16 branch carries a runtime CPU dispatch; hoisting it to the
    /// top of a batch keeps it off the per-row path.
    #[inline]
    pub(crate) fn row_decoder(self) -> RowDecoder {
        match self {
            Self::F32 => RowDecoder::F32,
            Self::F16 => RowDecoder::F16(crate::internal::simd::F16Decoder::resolve()),
            Self::BF16 => RowDecoder::BF16,
        }
    }

    pub(crate) fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::BF16),
            other => anyhow::bail!("unknown feature dtype tag: {other}"),
        }
    }
}

/// A [`FeatureDtype`] with its SIMD dispatch already resolved.
///
/// Resolve one per batch via [`FeatureDtype::row_decoder`] and call
/// [`RowDecoder::decode_row`] per row.
#[derive(Clone, Copy, Debug)]
pub(crate) enum RowDecoder {
    F32,
    F16(crate::internal::simd::F16Decoder),
    /// Carries no resolved state: the bf16 upcast is a shift-and-widen with
    /// a single cached AVX2 query, so there is nothing to hoist.
    BF16,
}

impl RowDecoder {
    /// Whether the payload is already `f32` lanes on disk, so rows can be
    /// appended into reserved capacity instead of upcast into a zeroed buffer.
    #[inline]
    pub(crate) fn is_f32_passthrough(self) -> bool {
        matches!(self, Self::F32)
    }

    /// Decode a little-endian feature row (or any contiguous run of rows)
    /// from `src` into `dst`.
    ///
    /// F32 payloads are a straight byte copy into the `f32` destination (no
    /// source-alignment requirement); the half-width payloads upcast, F16
    /// through the resolved converter and BF16 through its own dispatch.
    /// `src.len()` must equal `dst.len()` times the element size — every
    /// branch panics otherwise.
    #[inline(always)]
    pub(crate) fn decode_row(self, src: &[u8], dst: &mut [f32]) {
        match self {
            Self::F32 => bytemuck::cast_slice_mut::<f32, u8>(dst).copy_from_slice(src),
            Self::F16(conv) => conv.convert(src, dst),
            Self::BF16 => crate::internal::simd::bf16_le_to_f32(src, dst),
        }
    }
}

pub const HEADER_SIZE: u64 = 32;
pub const FEATURE_MAGIC: &[u8; 8] = b"AETHFEAT";
pub const MAX_FEATURE_NODES: u64 = 10_000_000_000;
pub const MAX_FEATURE_DIM: u64 = 100_000;

/// Parsed, validated feature-file header.
#[derive(Debug, Clone, Copy)]
pub struct FeatureHeader {
    pub num_nodes: usize,
    pub feature_dim: usize,
    /// Byte offset into the file where the feature payload starts.
    pub features_start_offset: u64,
    /// `feature_dim * dtype.element_size()` -- bytes per node's feature row.
    /// Only used by Linux-gated O_DIRECT alignment checks today, but validated
    /// for overflow on every load.
    pub feature_size: usize,
    /// `num_nodes * feature_size` -- bytes of payload, known to fit the file.
    pub payload_bytes: usize,
    /// Element data type (F32, F16 or BF16).
    pub dtype: FeatureDtype,
}

/// Bytes of file prefix [`parse_feature_header_bytes`] reads: the fixed
/// header plus the dtype tag at byte 32.
pub(crate) const HEADER_PREFIX_LEN: usize = HEADER_SIZE as usize + 1;

/// Read + validate the header. Also confirms the file is at least as large
/// as the payload the header claims, so later reads can't tear at EOF.
pub fn parse_feature_header(file: &File) -> Result<FeatureHeader> {
    let file_size = file
        .metadata()
        .context("failed to stat feature file")?
        .len();
    let mut prefix = [0u8; HEADER_PREFIX_LEN];
    let len = HEADER_PREFIX_LEN.min(usize::try_from(file_size).unwrap_or(HEADER_PREFIX_LEN));
    file.read_exact_at(&mut prefix[..len], 0)
        .context("failed to read header")?;
    parse_feature_header_bytes(&prefix[..len], file_size)
}

/// Validate a header from the file's leading bytes and its total size.
///
/// The single header parser: every store, mapped or read through a
/// descriptor, turns file bytes into a [`FeatureHeader`] here. `prefix`
/// holds the file's first bytes (at least [`HEADER_PREFIX_LEN`] of them
/// when the file has that many); `file_size` bounds every offset.
pub(crate) fn parse_feature_header_bytes(prefix: &[u8], file_size: u64) -> Result<FeatureHeader> {
    anyhow::ensure!(
        prefix.len() >= HEADER_SIZE as usize,
        "feature file too small: {file_size} bytes"
    );
    anyhow::ensure!(
        &prefix[0..8] == FEATURE_MAGIC,
        "invalid feature file format (bad magic)"
    );

    let num_nodes_u64 = u64::from_le_bytes(prefix[8..16].try_into()?);
    let feature_dim_u64 = u64::from_le_bytes(prefix[16..24].try_into()?);
    anyhow::ensure!(
        num_nodes_u64 <= MAX_FEATURE_NODES,
        "num_nodes {num_nodes_u64} exceeds maximum {MAX_FEATURE_NODES}"
    );
    anyhow::ensure!(
        feature_dim_u64 <= MAX_FEATURE_DIM,
        "feature_dim {feature_dim_u64} exceeds maximum {MAX_FEATURE_DIM}"
    );

    let features_start_offset = u64::from_le_bytes(prefix[24..32].try_into()?);
    // The dtype tag lives at byte 32, so the payload must start past it.
    anyhow::ensure!(
        features_start_offset > HEADER_SIZE,
        "invalid data_offset {features_start_offset} (must be > {HEADER_SIZE})"
    );
    // The f32 fast path casts the payload to &[f32], which requires a
    // 4-byte-aligned start.
    anyhow::ensure!(
        features_start_offset.is_multiple_of(std::mem::align_of::<f32>() as u64),
        "invalid data_offset {} (must be {}-byte aligned)",
        features_start_offset,
        std::mem::align_of::<f32>()
    );
    anyhow::ensure!(
        features_start_offset <= file_size,
        "invalid data_offset {features_start_offset} for file size {file_size}"
    );

    // In bounds: data_offset > HEADER_SIZE and data_offset <= file_size.
    let tag = *prefix
        .get(HEADER_SIZE as usize)
        .context("failed to read dtype tag")?;
    let dtype = FeatureDtype::from_u8(tag)?;

    // Do the byte-size and file-size validation entirely in u64 first, so a
    // 32-bit usize can't truncate the intermediate products before they're
    // checked. Cast to usize only after the file is known large enough.
    let feature_size_u64 = feature_dim_u64
        .checked_mul(dtype.element_size() as u64)
        .ok_or_else(|| anyhow::anyhow!("feature_size overflow"))?;
    let total_bytes_u64 = num_nodes_u64
        .checked_mul(feature_size_u64)
        .ok_or_else(|| anyhow::anyhow!("feature data size overflow"))?;
    let min_file_size = features_start_offset
        .checked_add(total_bytes_u64)
        .ok_or_else(|| anyhow::anyhow!("minimum feature file size overflow"))?;
    anyhow::ensure!(
        file_size >= min_file_size,
        "feature file truncated: expected at least {min_file_size} bytes, got {file_size}"
    );

    let num_nodes = usize::try_from(num_nodes_u64).context("num_nodes does not fit in usize")?;
    let feature_dim =
        usize::try_from(feature_dim_u64).context("feature_dim does not fit in usize")?;
    let feature_size =
        usize::try_from(feature_size_u64).context("feature_size does not fit in usize")?;
    let payload_bytes =
        usize::try_from(total_bytes_u64).context("feature payload does not fit in usize")?;

    Ok(FeatureHeader {
        num_nodes,
        feature_dim,
        features_start_offset,
        feature_size,
        payload_bytes,
        dtype,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(num_nodes: u64, dim: u64, offset: u64, tag: u8) -> Vec<u8> {
        let mut b = vec![0u8; HEADER_PREFIX_LEN];
        b[0..8].copy_from_slice(FEATURE_MAGIC);
        b[8..16].copy_from_slice(&num_nodes.to_le_bytes());
        b[16..24].copy_from_slice(&dim.to_le_bytes());
        b[24..32].copy_from_slice(&offset.to_le_bytes());
        b[32] = tag;
        b
    }

    #[test]
    fn parses_a_well_formed_prefix() {
        let h = parse_feature_header_bytes(&header(10, 4, 512, 1), 512 + 10 * 4 * 2).unwrap();
        assert_eq!(h.num_nodes, 10);
        assert_eq!(h.feature_dim, 4);
        assert_eq!(h.dtype, FeatureDtype::F16);
        assert_eq!(h.feature_size, 8);
        assert_eq!(h.payload_bytes, 80);
        assert_eq!(h.features_start_offset, 512);
    }

    #[test]
    fn rejects_every_malformed_field() {
        let ok = 512 + 10 * 16;
        for (bytes, size, needle) in [
            (header(10, 4, 512, 0)[..20].to_vec(), 20, "too small"),
            (header(10, 4, 32, 0), ok, "must be >"),
            (header(10, 4, 514, 0), ok + 2, "aligned"),
            (header(10, 4, 512, 9), ok, "dtype"),
            (header(10, 4, 512, 0), ok - 1, "truncated"),
            (header(10, 4, 4096, 0), ok, "for file size"),
            (header(u64::MAX, 4, 512, 0), ok, "exceeds maximum"),
        ] {
            let err = parse_feature_header_bytes(&bytes, size as u64).unwrap_err();
            assert!(err.to_string().contains(needle), "{needle}: got {err}");
        }
        let mut bad = header(10, 4, 512, 0);
        bad[0] = b'X';
        assert!(parse_feature_header_bytes(&bad, ok as u64).is_err());
    }
}
