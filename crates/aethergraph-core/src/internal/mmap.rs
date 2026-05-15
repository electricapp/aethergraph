//! Memory-mapped graph storage with scalable load-time validation.
//!
//! File format remains fixed-width and predictable for high-throughput I/O.

use crate::graph::{EdgeOffset, Graph, GraphValidationMode, NodeId};
use anyhow::{Context, Result};
use bytemuck::cast_slice;
use memmap2::MmapOptions;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, trace};

/// Magic number to identify AetherGraph files: "AETH" in ASCII
const MAGIC: u32 = 0x4145_5448;

/// Current file format version
const VERSION: u32 = 1;

/// Use full validation below this threshold; offsets-only above it.
const FULL_VALIDATION_THRESHOLD_BYTES: u64 = 512 * 1024 * 1024;

/// File header for graph storage.
///
/// Layout (32 bytes):
/// - magic: 4
/// - version: 4
/// - num_nodes: 8
/// - num_edges: 8
/// - has_weights: 4
/// - integrity_checksum32: 4 (0 => absent/legacy)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Header {
    magic: u32,
    version: u32,
    num_nodes: u64,
    num_edges: u64,
    has_weights: u32,
    integrity_checksum32: u32,
}

impl Header {
    pub const SIZE: usize = 32;

    fn new(
        num_nodes: usize,
        num_edges: usize,
        has_weights: bool,
        integrity_checksum32: Option<u32>,
    ) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            num_nodes: num_nodes as u64,
            num_edges: num_edges as u64,
            has_weights: if has_weights { 1 } else { 0 },
            integrity_checksum32: integrity_checksum32.unwrap_or(0),
        }
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.magic == MAGIC, "invalid magic number");
        anyhow::ensure!(self.version == VERSION, "unsupported version");
        Ok(())
    }

    fn checksum32(&self) -> Option<u32> {
        if self.integrity_checksum32 == 0 {
            None
        } else {
            Some(self.integrity_checksum32)
        }
    }

    #[inline(always)]
    fn to_bytes(self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..4].copy_from_slice(&self.magic.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.version.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.num_nodes.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.num_edges.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.has_weights.to_le_bytes());
        bytes[28..32].copy_from_slice(&self.integrity_checksum32.to_le_bytes());
        bytes
    }

    #[inline(always)]
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(
            bytes.len() >= Self::SIZE,
            "insufficient bytes for header: got {}, need {}",
            bytes.len(),
            Self::SIZE
        );

        fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
            let slice = bytes
                .get(offset..offset + 4)
                .ok_or_else(|| anyhow::anyhow!("header read out of bounds at offset {}", offset))?;
            let arr: [u8; 4] = slice
                .try_into()
                .map_err(|_| anyhow::anyhow!("failed to read u32 at offset {}", offset))?;
            Ok(u32::from_le_bytes(arr))
        }

        fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
            let slice = bytes
                .get(offset..offset + 8)
                .ok_or_else(|| anyhow::anyhow!("header read out of bounds at offset {}", offset))?;
            let arr: [u8; 8] = slice
                .try_into()
                .map_err(|_| anyhow::anyhow!("failed to read u64 at offset {}", offset))?;
            Ok(u64::from_le_bytes(arr))
        }

        Ok(Self {
            magic: read_u32(bytes, 0)?,
            version: read_u32(bytes, 4)?,
            num_nodes: read_u64(bytes, 8)?,
            num_edges: read_u64(bytes, 16)?,
            has_weights: read_u32(bytes, 24)?,
            integrity_checksum32: read_u32(bytes, 28)?,
        })
    }
}

#[derive(Debug, Clone)]
struct GraphFileLayout {
    num_nodes: usize,
    num_edges: usize,
    offsets_range: Range<usize>,
    edges_range: Range<usize>,
    weights_range: Option<Range<usize>>,
    checksum32: Option<u32>,
}

/// Saves a CSR graph to binary format.
#[inline]
pub fn save_graph(graph: &Graph, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    debug!(
        "Saving graph to {}: {} nodes, {} edges",
        path.display(),
        graph.num_nodes(),
        graph.num_edges()
    );

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .context("failed to create output file")?;

    let offsets_bytes = cast_slice::<EdgeOffset, u8>(graph.offsets());
    let edges_bytes = cast_slice::<NodeId, u8>(graph.edges());
    let checksum32 = Some(checksum32_parts(&[offsets_bytes, edges_bytes]));

    let header = Header::new(
        graph.num_nodes(),
        graph.num_edges(),
        graph.weights().is_some(),
        checksum32,
    );
    trace!(?checksum32, "Writing graph header");
    file.write_all(&header.to_bytes())
        .context("failed to write header")?;

    file.write_all(offsets_bytes)
        .context("failed to write offsets")?;
    file.write_all(edges_bytes)
        .context("failed to write edges")?;

    if let Some(weights) = graph.weights() {
        file.write_all(cast_slice::<f32, u8>(weights))
            .context("failed to write weights")?;
    }

    file.sync_all().context("failed to sync file")?;

    let file_size = std::fs::metadata(path)?.len();
    debug!("Graph saved ({:.2} MB)", file_size as f64 / 1_000_000.0);
    Ok(())
}

/// Loads a graph from file using automatic validation mode selection.
#[inline]
#[tracing::instrument(skip(path), fields(path = %path.as_ref().display()))]
pub fn load_graph(path: impl AsRef<Path>) -> Result<Graph> {
    let path = path.as_ref();
    let file_size = std::fs::metadata(path)
        .context("failed to stat graph file")?
        .len();
    let validation = default_validation_mode(file_size);
    load_graph_mmap(path, validation)
}

/// Loads a graph from file as mmap-backed storage with explicit validation.
#[inline]
pub fn load_graph_mmap(path: impl AsRef<Path>, validation: GraphValidationMode) -> Result<Graph> {
    let path = path.as_ref();
    debug!(
        "Loading mmap graph from {} with {:?} validation",
        path.display(),
        validation
    );

    let file = File::open(path).context("failed to open graph file")?;
    // SAFETY: the underlying file is owned through this scope and is treated as read-only;
    // we never mutate the mapped pages.
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .context("failed to mmap graph file")?
    };
    let mmap = Arc::new(mmap);

    let layout = parse_layout(&mmap)?;
    validate_checksum_if_present(&mmap, &layout, validation)?;

    let graph = Graph::from_mapped_parts(
        layout.num_nodes,
        layout.num_edges,
        Arc::clone(&mmap),
        layout.offsets_range.clone(),
        layout.edges_range.clone(),
        layout.weights_range.clone(),
    );
    graph.validate_with_mode(validation)?;
    Ok(graph)
}

/// Loads a graph from file as owned in-memory storage with explicit validation.
#[inline]
pub fn load_graph_owned(path: impl AsRef<Path>, validation: GraphValidationMode) -> Result<Graph> {
    let path = path.as_ref();
    let file = File::open(path).context("failed to open graph file")?;
    // SAFETY: file outlives the temporary mmap used for parsing; mapping is read-only
    // and the data is consumed (copied) by load_graph_from_mmap_with_validation.
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .context("failed to mmap graph file")?
    };
    load_graph_from_mmap_with_validation(&mmap, validation)
}

/// Alias to explicitly request validation mode on default mmap-backed load path.
#[inline]
pub fn load_graph_with_validation(
    path: impl AsRef<Path>,
    validation: GraphValidationMode,
) -> Result<Graph> {
    load_graph_mmap(path, validation)
}

/// Loads a CSR graph from existing bytes with full validation.
#[inline]
#[cfg(test)]
pub fn load_graph_from_mmap(mmap: &[u8]) -> Result<Graph> {
    load_graph_from_mmap_with_validation(mmap, GraphValidationMode::Full)
}

/// Loads a CSR graph from existing bytes with configurable validation.
#[inline]
pub fn load_graph_from_mmap_with_validation(
    mmap: &[u8],
    validation: GraphValidationMode,
) -> Result<Graph> {
    let layout = parse_layout(mmap)?;
    validate_checksum_if_present(mmap, &layout, validation)?;

    let offsets_bytes = &mmap[layout.offsets_range.start..layout.offsets_range.end];
    let edges_bytes = &mmap[layout.edges_range.start..layout.edges_range.end];

    let offsets: Vec<EdgeOffset> = offsets_bytes
        .chunks_exact(std::mem::size_of::<EdgeOffset>())
        .map(|chunk| {
            EdgeOffset::from_le_bytes(
                chunk
                    .try_into()
                    .expect("chunks_exact guarantees EdgeOffset-sized chunks"),
            )
        })
        .collect();

    let edges: Vec<NodeId> = edges_bytes
        .chunks_exact(std::mem::size_of::<NodeId>())
        .map(|chunk| {
            NodeId::from_le_bytes(
                chunk
                    .try_into()
                    .expect("chunks_exact guarantees NodeId-sized chunks"),
            )
        })
        .collect();

    let weights = if let Some(range) = &layout.weights_range {
        let weights_bytes = &mmap[range.start..range.end];
        Some(
            weights_bytes
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|chunk| {
                    f32::from_le_bytes(
                        chunk
                            .try_into()
                            .expect("chunks_exact guarantees f32-sized chunks"),
                    )
                })
                .collect(),
        )
    } else {
        None
    };

    let graph = Graph::from_csr_arrays(layout.num_nodes, offsets, edges, weights);
    graph.validate_with_mode(validation)?;
    Ok(graph)
}

fn parse_layout(bytes: &[u8]) -> Result<GraphFileLayout> {
    let header = Header::from_bytes(bytes)?;
    header.validate()?;

    const MAX_NODES: u64 = 10_000_000_000;
    const MAX_EDGES: u64 = 100_000_000_000;
    anyhow::ensure!(
        header.num_nodes <= MAX_NODES,
        "num_nodes {} exceeds maximum {}",
        header.num_nodes,
        MAX_NODES
    );
    anyhow::ensure!(
        header.num_edges <= MAX_EDGES,
        "num_edges {} exceeds maximum {}",
        header.num_edges,
        MAX_EDGES
    );

    let num_nodes = usize::try_from(header.num_nodes).context("num_nodes does not fit in usize")?;
    let num_edges = usize::try_from(header.num_edges).context("num_edges does not fit in usize")?;
    let has_weights = header.has_weights != 0;

    let offsets_size = num_nodes
        .checked_add(1)
        .and_then(|n| n.checked_mul(std::mem::size_of::<EdgeOffset>()))
        .ok_or_else(|| anyhow::anyhow!("offsets size overflow"))?;
    let edges_size = num_edges
        .checked_mul(std::mem::size_of::<NodeId>())
        .ok_or_else(|| anyhow::anyhow!("edges size overflow"))?;

    let offsets_start = Header::SIZE;
    let offsets_end = offsets_start
        .checked_add(offsets_size)
        .ok_or_else(|| anyhow::anyhow!("offsets range overflow"))?;
    let edges_start = offsets_end;
    let edges_end = edges_start
        .checked_add(edges_size)
        .ok_or_else(|| anyhow::anyhow!("edges range overflow"))?;

    let (weights_range, required_size) = if has_weights {
        let weights_size = num_edges
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("weights size overflow"))?;
        let weights_end = edges_end
            .checked_add(weights_size)
            .ok_or_else(|| anyhow::anyhow!("weights range overflow"))?;
        (Some(edges_end..weights_end), weights_end)
    } else {
        (None, edges_end)
    };

    anyhow::ensure!(
        bytes.len() >= required_size,
        "file too small: expected at least {} bytes, got {}",
        required_size,
        bytes.len()
    );

    Ok(GraphFileLayout {
        num_nodes,
        num_edges,
        offsets_range: offsets_start..offsets_end,
        edges_range: edges_start..edges_end,
        weights_range,
        checksum32: header.checksum32(),
    })
}

fn validate_checksum_if_present(
    bytes: &[u8],
    layout: &GraphFileLayout,
    validation: GraphValidationMode,
) -> Result<()> {
    if validation == GraphValidationMode::HeaderOnly {
        return Ok(());
    }

    let Some(expected) = layout.checksum32 else {
        // Older files may not include checksum metadata.
        return Ok(());
    };

    let offsets = &bytes[layout.offsets_range.start..layout.offsets_range.end];
    let edges = &bytes[layout.edges_range.start..layout.edges_range.end];
    let actual = checksum32_parts(&[offsets, edges]);

    anyhow::ensure!(
        actual == expected,
        "graph integrity checksum mismatch: expected {:#x}, got {:#x}",
        expected,
        actual
    );
    Ok(())
}

fn default_validation_mode(file_size: u64) -> GraphValidationMode {
    if file_size > FULL_VALIDATION_THRESHOLD_BYTES {
        GraphValidationMode::OffsetsOnly
    } else {
        GraphValidationMode::Full
    }
}

fn checksum32_parts(parts: &[&[u8]]) -> u32 {
    const FNV_OFFSET_BASIS: u32 = 0x811c_9dc5;
    const FNV_PRIME: u32 = 0x0100_0193;

    let mut hash = FNV_OFFSET_BASIS;
    for part in parts {
        for &b in *part {
            hash ^= b as u32;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_and_load() {
        let edges = vec![(0, 1), (0, 2), (1, 2), (2, 0)];
        let graph = Graph::from_edges(3, &edges, None).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        save_graph(&graph, temp_file.path()).unwrap();

        let loaded = load_graph(temp_file.path()).unwrap();

        assert_eq!(loaded.num_nodes(), graph.num_nodes());
        assert_eq!(loaded.num_edges(), graph.num_edges());
        assert_eq!(loaded.neighbors(0), graph.neighbors(0));
        assert_eq!(loaded.neighbors(1), graph.neighbors(1));
        assert_eq!(loaded.neighbors(2), graph.neighbors(2));
    }

    #[test]
    fn test_save_and_load_weighted() {
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let weights = vec![0.5, 1.0, 0.3];
        let graph = Graph::from_edges(3, &edges, Some(&weights)).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        save_graph(&graph, temp_file.path()).unwrap();

        let loaded = load_graph(temp_file.path()).unwrap();

        assert_eq!(loaded.num_nodes(), graph.num_nodes());
        assert_eq!(loaded.num_edges(), graph.num_edges());
        assert_eq!(loaded.neighbor_weights(0), Some(&[0.5, 1.0][..]));
        assert_eq!(loaded.neighbor_weights(1), Some(&[0.3][..]));
    }

    #[test]
    fn test_save_and_load_owned() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let graph = Graph::from_edges(3, &edges, None).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        save_graph(&graph, temp_file.path()).unwrap();

        let loaded = load_graph_owned(temp_file.path(), GraphValidationMode::Full).unwrap();
        assert_eq!(loaded.num_nodes(), 3);
        assert_eq!(loaded.num_edges(), 3);
        assert_eq!(loaded.neighbors(0), &[1]);
    }

    #[test]
    fn test_large_graph() {
        let mut edges = Vec::new();
        for i in 0..1000 {
            for j in 0..10 {
                edges.push((i, (i + j + 1) % 1000));
            }
        }

        let graph = Graph::from_edges(1000, &edges, None).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        save_graph(&graph, temp_file.path()).unwrap();

        let loaded = load_graph(temp_file.path()).unwrap();

        assert_eq!(loaded.num_nodes(), graph.num_nodes());
        assert_eq!(loaded.num_edges(), graph.num_edges());
    }

    #[test]
    fn test_invalid_magic() {
        let mut bytes = vec![0u8; 64];
        bytes[0..4].copy_from_slice(b"BAAD");
        let result = load_graph_from_mmap(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("magic"));
    }

    #[test]
    fn test_excessive_num_nodes() {
        let mut bytes = vec![0u8; 64];
        bytes[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&VERSION.to_le_bytes());
        bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        let result = load_graph_from_mmap(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds maximum"));
    }

    #[test]
    fn test_truncated_header() {
        let bytes = vec![0u8; 16];
        let result = load_graph_from_mmap(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_version() {
        let mut bytes = vec![0u8; 64];
        bytes[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&999u32.to_le_bytes());
        let result = load_graph_from_mmap(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("version"));
    }

    #[test]
    fn test_truncated_offsets() {
        let mut bytes = vec![0u8; 64];
        bytes[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&VERSION.to_le_bytes());
        bytes[8..16].copy_from_slice(&100u64.to_le_bytes());
        bytes[16..24].copy_from_slice(&50u64.to_le_bytes());
        bytes[24..28].copy_from_slice(&0u32.to_le_bytes());
        let result = load_graph_from_mmap(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_edges_count() {
        let edges = vec![(0, 1), (1, 2)];
        let graph = Graph::from_edges(3, &edges, None).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        save_graph(&graph, temp_file.path()).unwrap();

        let mut bytes = std::fs::read(temp_file.path()).unwrap();
        bytes[16..24].copy_from_slice(&999u64.to_le_bytes());

        let result = load_graph_from_mmap(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_bytes_file() {
        let bytes: Vec<u8> = vec![];
        let result = load_graph_from_mmap(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_garbage() {
        let garbage: Vec<u8> = (0..1000).map(|i| (i * 17 % 256) as u8).collect();
        let result = load_graph_from_mmap(&garbage);
        assert!(result.is_err());
    }

    #[test]
    fn test_overflow_edge_count() {
        let mut bytes = vec![0u8; 64];
        bytes[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&VERSION.to_le_bytes());
        bytes[8..16].copy_from_slice(&10u64.to_le_bytes());
        bytes[16..24].copy_from_slice(&(u64::MAX / 2).to_le_bytes());
        let result = load_graph_from_mmap(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_checksum_mismatch_detected() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let graph = Graph::from_edges(3, &edges, None).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        save_graph(&graph, temp_file.path()).unwrap();

        let mut bytes = std::fs::read(temp_file.path()).unwrap();
        let data_start = Header::SIZE + (graph.num_nodes() + 1) * std::mem::size_of::<EdgeOffset>();
        bytes[data_start] ^= 0xFF;

        let result = load_graph_from_mmap_with_validation(&bytes, GraphValidationMode::OffsetsOnly);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("checksum mismatch")
        );
    }
}
