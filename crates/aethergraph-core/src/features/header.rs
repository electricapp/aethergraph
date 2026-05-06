//! Feature-file header format + parsing/validation.
//!
//! The binary feature file layout:
//!
//! ```text
//!   [0..8)     magic: b"AETHFEAT"
//!   [8..16)    num_nodes: u64 le
//!   [16..24)   feature_dim: u64 le
//!   [24..32)   features_start_offset: u64 le
//!              (0 in legacy files means "starts right after the 32-byte header";
//!              newer writers pick an O_DIRECT-aligned offset)
//!   [offset..) feature payload: num_nodes × feature_dim × f32 (little-endian)
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

    pub(crate) fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            other => anyhow::bail!("unknown feature dtype tag: {}", other),
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
    let num_nodes =
        usize::try_from(num_nodes_u64).context("num_nodes does not fit in usize")?;
    let feature_dim =
        usize::try_from(feature_dim_u64).context("feature_dim does not fit in usize")?;

    let stored_offset = u64::from_le_bytes(header[24..32].try_into()?);
    let features_start_offset = if stored_offset == 0 {
        HEADER_SIZE
    } else {
        stored_offset
    };
    anyhow::ensure!(
        features_start_offset >= HEADER_SIZE,
        "invalid feature payload offset {}",
        features_start_offset
    );

    // Read dtype tag from byte 32 (first byte of padding region).
    // Legacy files with offset == HEADER_SIZE have no padding -- default to F32.
    // New files (offset >= 64) store the tag; byte 32 == 0 is F32 (backward compat).
    let dtype = if features_start_offset > HEADER_SIZE {
        let mut tag = [0u8; 1];
        file.read_exact_at(&mut tag, HEADER_SIZE)
            .context("failed to read dtype tag")?;
        FeatureDtype::from_u8(tag[0])?
    } else {
        FeatureDtype::F32
    };

    let feature_size = feature_dim
        .checked_mul(dtype.element_size())
        .ok_or_else(|| anyhow::anyhow!("feature_size overflow"))?;
    let total_bytes = num_nodes
        .checked_mul(feature_size)
        .ok_or_else(|| anyhow::anyhow!("feature data size overflow"))?;
    let min_file_size = features_start_offset
        .checked_add(total_bytes as u64)
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

    Ok(FeatureHeader {
        num_nodes,
        feature_dim,
        features_start_offset,
        feature_size,
        dtype,
    })
}
