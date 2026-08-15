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
//!   [32]       dtype tag (0 = F32, 1 = F16)
//!   [offset..) feature payload: num_nodes × feature_dim × elements (little-endian)
//! ```

use anyhow::{Context, Result};
use std::fs::File;
use std::os::unix::fs::FileExt;

/// Data type for stored features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FeatureDtype {
    F32 = 0,
    F16 = 1,
}

impl FeatureDtype {
    /// Bytes per element.
    pub const fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }

    /// Decode one row, resolving the dispatch inline.
    ///
    /// For loops over many rows, hoist [`FeatureDtype::row_decoder`] out of
    /// the loop instead.
    #[inline]
    pub(crate) fn decode_row(self, src: &[u8], dst: &mut [f32]) {
        self.row_decoder().decode_row(src, dst)
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
        }
    }

    pub(crate) fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            other => anyhow::bail!("unknown feature dtype tag: {}", other),
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
}

impl RowDecoder {
    /// Decode a little-endian feature row (or any contiguous run of rows)
    /// from `src` into `dst`.
    ///
    /// F32 payloads are a straight byte copy into the `f32` destination (no
    /// source-alignment requirement); F16 payloads upcast through the
    /// resolved converter. `src.len()` must equal `dst.len()` times the
    /// element size — both branches panic otherwise.
    #[inline(always)]
    pub(crate) fn decode_row(self, src: &[u8], dst: &mut [f32]) {
        match self {
            Self::F32 => bytemuck::cast_slice_mut::<f32, u8>(dst).copy_from_slice(src),
            Self::F16(conv) => conv.convert(src, dst),
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
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub feature_size: usize,
    /// Element data type (F32 or F16).
    pub dtype: FeatureDtype,
}

/// Read + validate the header. Also confirms the file is at least as large
/// as the payload the header claims, so later reads can't tear at EOF.
pub fn parse_feature_header(file: &File) -> Result<FeatureHeader> {
    let mut header = [0u8; HEADER_SIZE as usize];
    file.read_exact_at(&mut header, 0)
        .context("failed to read header")?;

    if &header[0..8] != FEATURE_MAGIC {
        anyhow::bail!("Invalid feature file format");
    }

    let num_nodes_u64 = u64::from_le_bytes(header[8..16].try_into()?);
    let feature_dim_u64 = u64::from_le_bytes(header[16..24].try_into()?);
    anyhow::ensure!(
        num_nodes_u64 <= MAX_FEATURE_NODES,
        "num_nodes {} exceeds maximum {}",
        num_nodes_u64,
        MAX_FEATURE_NODES
    );
    anyhow::ensure!(
        feature_dim_u64 <= MAX_FEATURE_DIM,
        "feature_dim {} exceeds maximum {}",
        feature_dim_u64,
        MAX_FEATURE_DIM
    );

    let features_start_offset = u64::from_le_bytes(header[24..32].try_into()?);
    // The dtype tag lives at byte 32, so the payload must start past it.
    anyhow::ensure!(
        features_start_offset > HEADER_SIZE,
        "invalid feature payload offset {} (must be > {})",
        features_start_offset,
        HEADER_SIZE
    );
    // Mirror FeatureStore::load: the f32 fast path casts the payload to
    // &[f32], which requires a 4-byte-aligned start. Written files (offset
    // 512) are always aligned; reject anything else here so every reader
    // validates it consistently.
    anyhow::ensure!(
        features_start_offset % std::mem::align_of::<f32>() as u64 == 0,
        "invalid feature payload offset {} (must be {}-byte aligned)",
        features_start_offset,
        std::mem::align_of::<f32>()
    );

    // Read the dtype tag at byte 32 (first byte of the padding region).
    let dtype = {
        let mut tag = [0u8; 1];
        file.read_exact_at(&mut tag, HEADER_SIZE)
            .context("failed to read dtype tag")?;
        FeatureDtype::from_u8(tag[0])?
    };

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
    let file_size = file
        .metadata()
        .context("failed to stat feature file")?
        .len();
    anyhow::ensure!(
        file_size >= min_file_size,
        "feature file truncated: expected at least {} bytes, got {}",
        min_file_size,
        file_size
    );

    let num_nodes = usize::try_from(num_nodes_u64).context("num_nodes does not fit in usize")?;
    let feature_dim =
        usize::try_from(feature_dim_u64).context("feature_dim does not fit in usize")?;
    let feature_size =
        usize::try_from(feature_size_u64).context("feature_size does not fit in usize")?;

    Ok(FeatureHeader {
        num_nodes,
        feature_dim,
        features_start_offset,
        feature_size,
        dtype,
    })
}
