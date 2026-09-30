//! Memory-mapped graph storage with scalable load-time validation.
//!
//! File format remains fixed-width and predictable for high-throughput I/O.

use crate::graph::csr::alloc_hinted;
use crate::graph::{EdgeOffset, Graph, GraphValidationMode, MAX_NODES, NodeId};
use anyhow::{Context, Result};
use bytemuck::cast_slice;
use memmap2::MmapOptions;
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, trace};

/// Magic number to identify AetherGraph files: "AETH" in ASCII
pub(crate) const MAGIC: u32 = 0x4145_5448;

/// Current file format version. The integrity checksum is CRC32 (IEEE) —
/// SIMD-accelerated with runtime dispatch, ~10-30 GB/s over the body.
const VERSION: u32 = 1;

/// Use full validation below this threshold; offsets-only above it.
const FULL_VALIDATION_THRESHOLD_BYTES: u64 = 512 * 1024 * 1024;

/// Structural upper bound on edges, shared by both file format versions.
/// Nodes are bounded by the node-id width, [`MAX_NODES`].
pub(crate) const MAX_EDGES: u64 = 100_000_000_000;

/// Write `path` atomically: the bytes go to a temporary sibling that is
/// fsynced and renamed over `path`, then the directory is fsynced.
///
/// Readers never see a partly written file, and a graph still mapped from
/// `path` keeps its now-unlinked inode instead of faulting on a truncation.
pub(crate) fn write_atomically(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
) -> Result<()> {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let name = path
        .file_name()
        .with_context(|| format!("output path {} has no file name", path.display()))?;
    let tmp = dir.join(format!(
        ".{}.{}.{}.tmp",
        name.to_string_lossy(),
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));

    let staged = (|| -> Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .with_context(|| format!("failed to create {}", tmp.display()))?;
        let mut out = BufWriter::with_capacity(1 << 20, file);
        write(&mut out)?;
        let file = out
            .into_inner()
            .map_err(std::io::IntoInnerError::into_error)
            .context("failed to flush output file")?;
        file.sync_all().context("failed to sync output file")?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("failed to move output into {}", path.display()))
    })();
    if staged.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    staged?;

    // Persist the rename. Filesystems that cannot sync a directory report
    // EINVAL; there is nothing further to persist on those.
    match File::open(dir).and_then(|d| d.sync_all()) {
        Err(e) if e.kind() != std::io::ErrorKind::InvalidInput => {
            Err(e).with_context(|| format!("failed to sync directory {}", dir.display()))
        }
        _ => Ok(()),
    }
}

/// Copy a little-endian on-disk section into a fresh typed array.
///
/// The destination is allocated zeroed and huge-page-hinted before the copy
/// touches it, and the copy runs in parallel chunks. Viewing the typed
/// destination as bytes keeps the source free of any alignment requirement.
fn copy_section<T: bytemuck::Pod>(bytes: &[u8]) -> Vec<T> {
    const CHUNK: usize = 8 << 20;
    let mut out: Vec<T> = alloc_hinted(bytes.len() / std::mem::size_of::<T>());
    bytemuck::cast_slice_mut::<T, u8>(&mut out)
        .par_chunks_mut(CHUNK)
        .zip(bytes.par_chunks(CHUNK))
        .for_each(|(dst, src)| dst.copy_from_slice(src));
    out
}

/// File header for graph storage.
///
/// Layout (32 bytes):
/// - magic: 4
/// - version: 4
/// - num_nodes: 8
/// - num_edges: 8
/// - has_weights: 4
/// - integrity_checksum32: 4 (0 => absent; checksums are optional by design)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Header {
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
        anyhow::ensure!(
            self.version == VERSION,
            "unsupported version: expected {}, got {}",
            VERSION,
            self.version
        );
        Ok(())
    }

    /// Header for a version-2 compressed file; the checksum covers the
    /// compressed payload rather than flat arrays.
    pub(crate) fn for_compressed(
        num_nodes: usize,
        num_edges: usize,
        has_weights: bool,
        payload_checksum32: u32,
    ) -> Self {
        Self {
            magic: MAGIC,
            version: super::compressed_graph::COMPRESSED_VERSION,
            num_nodes: num_nodes as u64,
            num_edges: num_edges as u64,
            has_weights: if has_weights { 1 } else { 0 },
            integrity_checksum32: payload_checksum32,
        }
    }

    /// Validate magic/version/bounds for the compressed format.
    pub(crate) fn validate_compressed(&self) -> Result<()> {
        anyhow::ensure!(self.magic == MAGIC, "invalid magic number");
        anyhow::ensure!(
            self.version == super::compressed_graph::COMPRESSED_VERSION,
            "not a compressed graph file: version {}",
            self.version
        );
        anyhow::ensure!(
            self.num_nodes <= MAX_NODES as u64 && self.num_edges <= MAX_EDGES,
            "graph dimensions out of bounds: {} nodes, {} edges",
            self.num_nodes,
            self.num_edges
        );
        Ok(())
    }

    pub(crate) fn version(&self) -> u32 {
        self.version
    }

    pub(crate) fn num_nodes(&self) -> u64 {
        self.num_nodes
    }

    pub(crate) fn num_edges(&self) -> u64 {
        self.num_edges
    }

    pub(crate) fn has_weights(&self) -> bool {
        self.has_weights != 0
    }

    pub(crate) fn checksum32(&self) -> Option<u32> {
        if self.integrity_checksum32 == 0 {
            None
        } else {
            Some(self.integrity_checksum32)
        }
    }

    #[inline(always)]
    pub(crate) fn to_bytes(self) -> [u8; Self::SIZE] {
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
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(
            bytes.len() >= Self::SIZE,
            "insufficient bytes for header: got {}, need {}",
            bytes.len(),
            Self::SIZE
        );

        fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
            let slice = bytes
                .get(offset..offset + 4)
                .ok_or_else(|| anyhow::anyhow!("header read out of bounds at offset {offset}"))?;
            let arr: [u8; 4] = slice
                .try_into()
                .map_err(|_| anyhow::anyhow!("failed to read u32 at offset {offset}"))?;
            Ok(u32::from_le_bytes(arr))
        }

        fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
            let slice = bytes
                .get(offset..offset + 8)
                .ok_or_else(|| anyhow::anyhow!("header read out of bounds at offset {offset}"))?;
            let arr: [u8; 8] = slice
                .try_into()
                .map_err(|_| anyhow::anyhow!("failed to read u64 at offset {offset}"))?;
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

/// Saves a homogeneous CSR graph to binary format, atomically replacing
/// any existing file at `path`.
#[inline]
pub fn save_graph(graph: &Graph, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    debug!(
        "Saving graph to {}: {} nodes, {} edges",
        path.display(),
        graph.num_nodes(),
        graph.num_edges()
    );
    anyhow::ensure!(
        graph.is_homogeneous(),
        "the graph file format has one node count; a bipartite CSR belongs in a hetero graph file"
    );

    let offsets_bytes = cast_slice::<EdgeOffset, u8>(graph.offsets());
    let edges_bytes = cast_slice::<NodeId, u8>(graph.edges());
    let checksum32 = Some(crc32_parts(&[offsets_bytes, edges_bytes]));

    let header = Header::new(
        graph.num_nodes(),
        graph.num_edges(),
        graph.weights().is_some(),
        checksum32,
    );
    trace!(?checksum32, "Writing graph header");
    write_atomically(path, |file| {
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
        Ok(())
    })?;

    let file_size = std::fs::metadata(path)?.len();
    debug!("Graph saved ({:.2} MB)", file_size as f64 / 1_000_000.0);
    Ok(())
}

/// Reads just the header to learn which format version a file carries.
fn sniff_version(path: &Path) -> Result<u32> {
    use std::io::Read;
    let mut file = File::open(path).context("failed to open graph file")?;
    let mut head = [0u8; Header::SIZE];
    file.read_exact(&mut head)
        .context("failed to read graph header")?;
    Ok(Header::from_bytes(&head)?.version())
}

/// Loads a graph from file using automatic validation mode selection.
///
/// Dispatches on the header's version field: flat version-1 files load
/// mmap-backed, compressed version-2 files decode into owned storage.
#[inline]
#[tracing::instrument(skip(path), fields(path = %path.as_ref().display()))]
pub fn load_graph(path: impl AsRef<Path>) -> Result<Graph> {
    let path = path.as_ref();
    let file_size = std::fs::metadata(path)
        .context("failed to stat graph file")?
        .len();
    let validation = default_validation_mode(file_size);
    if sniff_version(path)? == super::compressed_graph::COMPRESSED_VERSION {
        return super::compressed_graph::load_graph_compressed(path, validation);
    }
    load_graph_mmap(path, validation)
}

/// Loads a graph from file as mmap-backed storage with explicit validation.
///
/// Refuses compressed (version-2) files: they decode into owned arrays,
/// which is not the page-cache-backed storage this entry point promises.
#[inline]
pub fn load_graph_mmap(path: impl AsRef<Path>, validation: GraphValidationMode) -> Result<Graph> {
    let path = path.as_ref();
    if sniff_version(path)? == super::compressed_graph::COMPRESSED_VERSION {
        anyhow::bail!(
            "{} is a compressed graph file, which always decodes into owned \
             storage; load it with the auto or owned storage mode instead of mmap",
            path.display()
        );
    }
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

    // Hints go on before any page is brought in, so they shape the faults
    // rather than trail them. The offsets array gets MADV_HUGEPAGE
    // (best-effort): random per-node lookups over a multi-GB offsets array
    // are dTLB-bound at 4 KiB pages. Outside Full validation the edges
    // array is hinted MADV_RANDOM: sampling faults one page per useful
    // neighbor list, and default readahead would drag in 128 KiB per fault.
    let offsets_bytes = &mmap[layout.offsets_range.start..layout.offsets_range.end];
    let edges_bytes = &mmap[layout.edges_range.start..layout.edges_range.end];
    let body = &mmap[layout.offsets_range.start..layout.edges_range.end];
    crate::internal::hint::advise_hugepage(offsets_bytes.as_ptr(), offsets_bytes.len());
    crate::internal::hint::advise_hugepage(edges_bytes.as_ptr(), edges_bytes.len());
    if validation != GraphValidationMode::Full {
        crate::internal::hint::advise_mmap_random(edges_bytes.as_ptr(), edges_bytes.len());
    }

    // Bring in what the load will read, on this thread and under an
    // interleaved placement so the page cache spreads across NUMA nodes.
    // Full validation streams the whole body (checksum + destination
    // checks), so it reads ahead the body. OffsetsOnly/HeaderOnly never
    // touch the edge pages at load — a body-wide read would turn the
    // O(1)-startup mmap of a multi-GB graph into a full sequential read —
    // but the offsets array is on the critical path of every batch, so it
    // is populated synchronously; WILLNEED remains the fallback on kernels
    // without MADV_POPULATE_READ.
    let ((), interleaved) = crate::internal::hint::with_interleaved_placement(|| {
        if validation == GraphValidationMode::Full {
            crate::internal::hint::prefetch_mmap_range(body.as_ptr(), body.len());
        } else if !crate::internal::hint::populate_read(offsets_bytes.as_ptr(), offsets_bytes.len())
        {
            crate::internal::hint::prefetch_mmap_range(offsets_bytes.as_ptr(), offsets_bytes.len());
        }
    });
    if interleaved {
        debug!("graph pages brought in under interleaved NUMA placement");
    }

    validate_checksum_if_present(&mmap, &layout, validation)?;

    Graph::from_mapped_parts(
        layout.num_nodes,
        layout.num_nodes,
        layout.num_edges,
        Arc::clone(&mmap),
        layout.offsets_range.clone(),
        layout.edges_range.clone(),
        layout.weights_range,
        validation,
    )
}

/// Loads a graph from file as owned in-memory storage.
///
/// The copy into memory touches every byte, so the owned graph is always
/// proven `Full` (monotone offsets, destinations in range) whatever
/// `validation` asks for; `validation` selects whether the checksum is
/// verified (`Full` only).
#[inline]
pub fn load_graph_owned(path: impl AsRef<Path>, validation: GraphValidationMode) -> Result<Graph> {
    let path = path.as_ref();
    if sniff_version(path)? == super::compressed_graph::COMPRESSED_VERSION {
        return super::compressed_graph::load_graph_compressed(path, validation);
    }
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

/// Loads a CSR graph from existing bytes into owned storage, proven `Full`
/// (see [`load_graph_owned`]); `validation` selects checksum verification.
#[inline]
pub fn load_graph_from_mmap_with_validation(
    mmap: &[u8],
    validation: GraphValidationMode,
) -> Result<Graph> {
    const {
        assert!(
            cfg!(target_endian = "little"),
            "graph file is little-endian"
        );
    };
    let layout = parse_layout(mmap)?;
    validate_checksum_if_present(mmap, &layout, validation)?;

    let offsets = copy_section::<EdgeOffset>(&mmap[layout.offsets_range.clone()]);
    let edges = copy_section::<NodeId>(&mmap[layout.edges_range.clone()]);
    let weights = layout
        .weights_range
        .as_ref()
        .map(|range| copy_section::<f32>(&mmap[range.clone()]));

    Graph::from_csr_vecs(
        layout.num_nodes,
        layout.num_nodes,
        offsets,
        edges,
        weights,
        GraphValidationMode::Full,
    )
}

fn parse_layout(bytes: &[u8]) -> Result<GraphFileLayout> {
    let header = Header::from_bytes(bytes)?;
    header.validate()?;

    anyhow::ensure!(
        header.num_nodes <= MAX_NODES as u64,
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

    // Each section length must be an exact multiple of its element size for the
    // zero-copy `try_cast_slice` reads to succeed (the mmap base is page-aligned
    // and Header::SIZE keeps the offsets start 8-byte aligned, so alignment
    // already holds). Enforcing it here surfaces any future range-computation
    // change as a load-time `Err` rather than a panic on first access.
    let offsets_range = offsets_start..offsets_end;
    let edges_range = edges_start..edges_end;
    anyhow::ensure!(
        offsets_range
            .len()
            .is_multiple_of(std::mem::size_of::<EdgeOffset>()),
        "offsets section length {} is not a multiple of {}",
        offsets_range.len(),
        std::mem::size_of::<EdgeOffset>()
    );
    anyhow::ensure!(
        edges_range
            .len()
            .is_multiple_of(std::mem::size_of::<NodeId>()),
        "edges section length {} is not a multiple of {}",
        edges_range.len(),
        std::mem::size_of::<NodeId>()
    );
    if let Some(range) = &weights_range {
        anyhow::ensure!(
            range.len().is_multiple_of(std::mem::size_of::<f32>()),
            "weights section length {} is not a multiple of {}",
            range.len(),
            std::mem::size_of::<f32>()
        );
    }

    Ok(GraphFileLayout {
        num_nodes,
        num_edges,
        offsets_range,
        edges_range,
        weights_range,
        checksum32: header.checksum32(),
    })
}

fn validate_checksum_if_present(
    bytes: &[u8],
    layout: &GraphFileLayout,
    validation: GraphValidationMode,
) -> Result<()> {
    // Only Full validation pays for the checksum. Hashing covers the
    // entire offsets+edges body, which faults every page of the mapping
    // into memory — for the multi-GB files that default to OffsetsOnly,
    // that turns the O(1)-startup mmap load into a full sequential read
    // of the file. OffsetsOnly still gets structural protection from
    // `validate_with_mode` (monotonic offsets, edge-count consistency)
    // without touching the edge pages.
    if validation != GraphValidationMode::Full {
        return Ok(());
    }

    let Some(expected) = layout.checksum32 else {
        // Older files may not include checksum metadata.
        return Ok(());
    };

    let offsets = &bytes[layout.offsets_range.start..layout.offsets_range.end];
    let edges = &bytes[layout.edges_range.start..layout.edges_range.end];
    let actual = crc32_parts(&[offsets, edges]);

    anyhow::ensure!(
        actual == expected,
        "graph integrity checksum mismatch: expected {expected:#x}, got {actual:#x}"
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

/// CRC32 (IEEE) over the concatenated parts. crc32fast dispatches to the
/// SIMD carryless-multiply path at runtime, so this runs at memory speed
/// rather than the cycles-per-byte a serial byte hash costs over multi-GB
/// bodies.
pub(crate) fn crc32_parts(parts: &[&[u8]]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize()
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
    fn test_checksum_mismatch_detected_under_full_validation() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let graph = Graph::from_edges(3, &edges, None).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        save_graph(&graph, temp_file.path()).unwrap();

        let mut bytes = std::fs::read(temp_file.path()).unwrap();
        let data_start = Header::SIZE + (graph.num_nodes() + 1) * std::mem::size_of::<EdgeOffset>();
        bytes[data_start] ^= 0xFF;

        let result = load_graph_from_mmap_with_validation(&bytes, GraphValidationMode::Full);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("checksum mismatch")
        );
    }

    #[test]
    fn test_offsets_only_skips_body_checksum() {
        // OffsetsOnly is the default for large files precisely because it
        // must not fault the whole body in; hashing every byte would do
        // exactly that. A corrupt edge byte therefore goes undetected in
        // this mode (structural offsets checks still run) — that's the
        // documented trade-off, not an accident.
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let graph = Graph::from_edges(3, &edges, None).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        save_graph(&graph, temp_file.path()).unwrap();

        let mut bytes = std::fs::read(temp_file.path()).unwrap();
        let data_start = Header::SIZE + (graph.num_nodes() + 1) * std::mem::size_of::<EdgeOffset>();
        bytes[data_start] ^= 0x01;

        let result = load_graph_from_mmap_with_validation(&bytes, GraphValidationMode::OffsetsOnly);
        assert!(result.is_ok());
    }

    /// Saving over the file a live graph is mapped from replaces it by
    /// rename, so the mapping keeps reading the old contents instead of
    /// faulting on a truncated file, and no temporary is left behind.
    #[test]
    fn test_save_over_mapped_file_keeps_mapping_readable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.bin");
        let first = Graph::from_edges(3, &[(0, 1), (1, 2), (2, 0)], None).unwrap();
        save_graph(&first, &path).unwrap();
        let mapped = load_graph_mmap(&path, GraphValidationMode::Full).unwrap();

        let second = Graph::from_edges(2, &[(1, 0)], None).unwrap();
        save_graph(&second, &path).unwrap();

        assert_eq!(mapped.neighbors(2), &[0]);
        assert_eq!(mapped.num_edges(), 3);
        let reloaded = load_graph_mmap(&path, GraphValidationMode::Full).unwrap();
        assert_eq!(reloaded.num_nodes(), 2);
        let entries: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
        assert_eq!(entries.len(), 1, "temporary file left behind");
    }

    /// Owned loads copy every byte, so they prove destinations whatever the
    /// requested mode; mmap loads prove exactly what was asked.
    #[test]
    fn test_owned_load_is_full_and_mmap_load_matches_mode() {
        let graph = Graph::from_edges(3, &[(0, 1), (1, 2)], None).unwrap();
        let tmp = NamedTempFile::new().unwrap();
        save_graph(&graph, tmp.path()).unwrap();
        let owned = load_graph_owned(tmp.path(), GraphValidationMode::HeaderOnly).unwrap();
        assert_eq!(owned.validated(), GraphValidationMode::Full);
        let mapped = load_graph_mmap(tmp.path(), GraphValidationMode::OffsetsOnly).unwrap();
        assert_eq!(mapped.validated(), GraphValidationMode::OffsetsOnly);

        // An out-of-range destination fails the owned load in any mode.
        let mut bytes = std::fs::read(tmp.path()).unwrap();
        let edges_start = Header::SIZE + 4 * std::mem::size_of::<EdgeOffset>();
        bytes[edges_start..edges_start + 4].copy_from_slice(&9u32.to_le_bytes());
        let err = load_graph_from_mmap_with_validation(&bytes, GraphValidationMode::HeaderOnly)
            .unwrap_err()
            .to_string();
        assert!(err.contains("out of range"), "got: {err}");
    }
}
