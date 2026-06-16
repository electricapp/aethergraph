//! Heterogeneous graph binary format: save/load multi-relational CSR.
//!
//! File layout (version 1):
//! ```text
//! [Header: 64 bytes]
//!   magic:          8 bytes  "AETHHETG"
//!   version:        u32 (currently 1; readers reject newer versions)
//!   num_node_types: u32
//!   num_edge_types: u32
//!   reserved:       44 bytes (zero)
//!
//! [Node Type Table: num_node_types × 72 bytes each]
//!   name:       64 bytes (UTF-8, zero-padded)
//!   node_count: u64
//!
//! [Edge Type Table: num_edge_types × 216 bytes each]
//!   src_name:      64 bytes (UTF-8, zero-padded)
//!   rel_name:      64 bytes (UTF-8, zero-padded)
//!   dst_name:      64 bytes (UTF-8, zero-padded)
//!   num_src_nodes: u64
//!   num_edges:     u64
//!   has_weights:   u32
//!   _padding:      u32
//!
//! [Per-edge-type CSR sections, concatenated]
//!   For each edge type:
//!     offsets: [u64; num_src_nodes + 1]
//!     edges:   [u32; num_edges]   (zero-padded to an 8-byte boundary)
//!     weights: [f32; num_edges]   (if has_weights; zero-padded to 8 bytes)
//! ```
//!
//! The 8-byte padding after the u32 sections keeps every following
//! offsets array (u64) naturally aligned in the mapping — without it an
//! odd edge count would misalign every subsequent section and the
//! zero-copy u64 casts would fail.
//!
//! The loader validates everything it reads against the file length with
//! checked arithmetic before constructing any view: a corrupt or
//! truncated file must surface as `Err`, never as a panic or a SIGBUS on
//! first access.

use crate::graph::hetero::HeteroGraph;
use crate::graph::{EdgeOffset, Graph, NodeId};
use anyhow::{Context, Result, bail, ensure};
use memmap2::MmapOptions;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"AETHHETG";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;
const NODE_TYPE_ENTRY_SIZE: usize = 72; // 64 name + 8 count
const EDGE_TYPE_ENTRY_SIZE: usize = 216; // 64*3 names + 8+8+4+4

/// Type IDs are u8 throughout `HeteroGraph`.
const MAX_TYPES: usize = 256;
/// Sanity caps mirroring the homogeneous loader — a corrupt header must
/// not drive multi-GB allocations or offset arithmetic.
const MAX_NODES: u64 = 10_000_000_000;
const MAX_EDGES: u64 = 100_000_000_000;

/// Bytes of zero padding needed to align `len` up to 8.
fn pad8(len: usize) -> usize {
    len.wrapping_neg() & 7
}

fn write_padded_name(buf: &mut Vec<u8>, name: &str) {
    let bytes = name.as_bytes();
    assert!(bytes.len() < 64, "type name too long: {name}");
    buf.extend_from_slice(bytes);
    buf.resize(buf.len() + (64 - bytes.len()), 0);
}

fn read_padded_name(data: &[u8]) -> Result<String> {
    // Names are written NUL-terminated and zero-padded. A field with no NUL
    // is unterminated — treat it as corruption rather than silently using
    // all 64 bytes.
    let end = data.iter().position(|&b| b == 0).ok_or_else(|| {
        anyhow::anyhow!("type name field has no NUL terminator — file is corrupt")
    })?;
    String::from_utf8(data[..end].to_vec()).context("invalid UTF-8 in type name")
}

/// Save a heterogeneous graph to a binary file.
pub fn save_hetero_graph(graph: &HeteroGraph, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let mut file = File::create(path).context("create file")?;

    let num_nt = graph.node_type_count();
    let num_et = graph.edge_type_count();
    ensure!(
        num_nt <= MAX_TYPES && num_et <= MAX_TYPES,
        "type counts ({num_nt} node, {num_et} edge) exceed the format limit of {MAX_TYPES}"
    );

    // Header
    let mut header = vec![0u8; HEADER_SIZE];
    header[..8].copy_from_slice(MAGIC);
    header[8..12].copy_from_slice(&VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&(num_nt as u32).to_le_bytes());
    header[16..20].copy_from_slice(&(num_et as u32).to_le_bytes());
    file.write_all(&header)?;

    // Node type table
    for nt_id in 0..num_nt as u8 {
        let meta = graph.node_type_meta(nt_id);
        let mut entry = Vec::with_capacity(NODE_TYPE_ENTRY_SIZE);
        write_padded_name(&mut entry, &meta.name);
        entry.extend_from_slice(&(meta.count as u64).to_le_bytes());
        file.write_all(&entry)?;
    }

    // Edge type table
    for et_id in 0..num_et as u8 {
        let meta = graph.edge_type_meta(et_id);
        let src_name = &graph.node_type_meta(meta.src_type).name;
        let dst_name = &graph.node_type_meta(meta.dst_type).name;
        let csr = graph.csr(et_id);

        let mut entry = Vec::with_capacity(EDGE_TYPE_ENTRY_SIZE);
        write_padded_name(&mut entry, src_name);
        write_padded_name(&mut entry, &meta.relation);
        write_padded_name(&mut entry, dst_name);
        let has_weights: u32 = if csr.weights().is_some() { 1 } else { 0 };
        entry.extend_from_slice(&(csr.num_nodes() as u64).to_le_bytes());
        entry.extend_from_slice(&(csr.num_edges() as u64).to_le_bytes());
        entry.extend_from_slice(&has_weights.to_le_bytes());
        entry.extend_from_slice(&0u32.to_le_bytes()); // padding
        file.write_all(&entry)?;
    }

    // CSR sections. The u32 sections (edges, weights) are padded to an
    // 8-byte boundary so the next edge type's u64 offsets array stays
    // naturally aligned in the mapping.
    const ZERO_PAD: [u8; 8] = [0u8; 8];
    for et_id in 0..num_et as u8 {
        let csr = graph.csr(et_id);
        let offsets = csr.offsets_raw();
        let edges = csr.edges_raw();
        file.write_all(bytemuck::cast_slice(offsets))?;
        let edges_bytes: &[u8] = bytemuck::cast_slice(edges);
        file.write_all(edges_bytes)?;
        file.write_all(&ZERO_PAD[..pad8(edges_bytes.len())])?;
        if let Some(weights) = csr.weights() {
            let weight_bytes: &[u8] = bytemuck::cast_slice(weights);
            file.write_all(weight_bytes)?;
            file.write_all(&ZERO_PAD[..pad8(weight_bytes.len())])?;
        }
    }

    file.flush()?;
    Ok(())
}

/// Load a heterogeneous graph from a binary file via mmap.
pub fn load_hetero_graph(path: impl AsRef<Path>) -> Result<HeteroGraph> {
    let path = path.as_ref();
    let file = File::open(path).context("open file")?;
    // SAFETY: file is owned for the duration of the mapping; the Arc keeps the read-only
    // mmap alive and we never mutate its pages.
    let mmap = Arc::new(unsafe { MmapOptions::new().map(&file)? });
    let file_len = mmap.len();

    // Hint the kernel to fault in the mapping ahead of the section-by-section
    // structural validation that streams the offsets/edges pages below.
    crate::internal::hint::prefetch_mmap_range(mmap.as_ptr(), file_len);

    if file_len < HEADER_SIZE {
        bail!("file too small for header: {file_len} bytes");
    }

    // Parse header
    if &mmap[..8] != MAGIC {
        bail!("invalid magic (expected AETHHETG)");
    }
    let version = u32::from_le_bytes(mmap[8..12].try_into()?);
    ensure!(
        version == VERSION,
        "hetero graph file format version {version} is not supported by this build (max {VERSION})"
    );
    let num_nt = u32::from_le_bytes(mmap[12..16].try_into()?) as usize;
    let num_et = u32::from_le_bytes(mmap[16..20].try_into()?) as usize;
    ensure!(
        num_nt <= MAX_TYPES && num_et <= MAX_TYPES,
        "type counts ({num_nt} node, {num_et} edge) exceed the format limit of {MAX_TYPES} \
         — file is corrupt"
    );

    // Both tables must fit before anything is sliced.
    let tables_len = num_nt
        .checked_mul(NODE_TYPE_ENTRY_SIZE)
        .and_then(|n| num_et.checked_mul(EDGE_TYPE_ENTRY_SIZE).map(|e| (n, e)))
        .and_then(|(n, e)| n.checked_add(e))
        .and_then(|t| t.checked_add(HEADER_SIZE))
        .ok_or_else(|| anyhow::anyhow!("type table size overflows"))?;
    ensure!(
        tables_len <= file_len,
        "file truncated: type tables need {tables_len} bytes, file has {file_len}"
    );

    let mut offset = HEADER_SIZE;

    // Parse node types
    let mut node_types: Vec<(String, usize)> = Vec::with_capacity(num_nt);
    for _ in 0..num_nt {
        let entry = &mmap[offset..offset + NODE_TYPE_ENTRY_SIZE];
        let name = read_padded_name(&entry[..64])?;
        let count = u64::from_le_bytes(entry[64..72].try_into()?);
        ensure!(
            count <= MAX_NODES,
            "node type '{name}' count {count} exceeds maximum {MAX_NODES} — file is corrupt"
        );
        node_types.push((name, count as usize));
        offset += NODE_TYPE_ENTRY_SIZE;
    }

    // Parse edge type headers
    struct EdgeTypeHeader {
        src_name: String,
        rel_name: String,
        dst_name: String,
        num_src_nodes: usize,
        num_edges: usize,
        has_weights: bool,
    }
    let mut et_headers: Vec<EdgeTypeHeader> = Vec::with_capacity(num_et);
    for _ in 0..num_et {
        let entry = &mmap[offset..offset + EDGE_TYPE_ENTRY_SIZE];
        let src_name = read_padded_name(&entry[0..64])?;
        let rel_name = read_padded_name(&entry[64..128])?;
        let dst_name = read_padded_name(&entry[128..192])?;
        let num_src_nodes = u64::from_le_bytes(entry[192..200].try_into()?);
        let num_edges = u64::from_le_bytes(entry[200..208].try_into()?);
        ensure!(
            num_src_nodes <= MAX_NODES,
            "edge type '{rel_name}' num_src_nodes {num_src_nodes} exceeds maximum {MAX_NODES} \
             — file is corrupt"
        );
        ensure!(
            num_edges <= MAX_EDGES,
            "edge type '{rel_name}' num_edges {num_edges} exceeds maximum {MAX_EDGES} \
             — file is corrupt"
        );
        let has_weights = u32::from_le_bytes(entry[208..212].try_into()?) != 0;
        et_headers.push(EdgeTypeHeader {
            src_name,
            rel_name,
            dst_name,
            num_src_nodes: num_src_nodes as usize,
            num_edges: num_edges as usize,
            has_weights,
        });
        offset += EDGE_TYPE_ENTRY_SIZE;
    }

    // Validate and slice CSR sections (mmap-backed, zero-copy). Every
    // range is bounds- and overflow-checked against the file length
    // before a Graph view is built over it.
    let take =
        |offset: &mut usize, len: usize, what: &str, rel: &str| -> Result<std::ops::Range<usize>> {
            let start = *offset;
            let end = start
                .checked_add(len)
                .ok_or_else(|| anyhow::anyhow!("section size overflows for edge type '{rel}'"))?;
            ensure!(
                end <= file_len,
                "file truncated: {what} section of edge type '{rel}' needs bytes {start}..{end}, \
             file has {file_len}"
            );
            *offset = end;
            Ok(start..end)
        };

    let mut edge_types: Vec<(String, String, String, Graph)> = Vec::with_capacity(num_et);
    for hdr in et_headers {
        let rel = hdr.rel_name.as_str();
        let offsets_bytes = hdr
            .num_src_nodes
            .checked_add(1)
            .and_then(|n| n.checked_mul(std::mem::size_of::<EdgeOffset>()))
            .ok_or_else(|| anyhow::anyhow!("offsets size overflows for edge type '{rel}'"))?;
        let edges_bytes = hdr
            .num_edges
            .checked_mul(std::mem::size_of::<NodeId>())
            .ok_or_else(|| anyhow::anyhow!("edges size overflows for edge type '{rel}'"))?;

        let offsets_range = take(&mut offset, offsets_bytes, "offsets", rel)?;
        ensure!(
            offsets_range
                .start
                .is_multiple_of(std::mem::align_of::<EdgeOffset>()),
            "offsets section of edge type '{rel}' is misaligned — file is corrupt"
        );
        // O(1) structural check: a CSR offsets array starts at 0 and ends
        // at num_edges. Catches sections that landed on the wrong bytes
        // without faulting in the whole file.
        let first =
            u64::from_le_bytes(mmap[offsets_range.start..offsets_range.start + 8].try_into()?);
        let last = u64::from_le_bytes(mmap[offsets_range.end - 8..offsets_range.end].try_into()?);
        ensure!(
            first == 0 && last == hdr.num_edges as u64,
            "offsets of edge type '{rel}' are inconsistent (first={first}, last={last}, \
             num_edges={}) — file is corrupt",
            hdr.num_edges
        );
        let edges_range = take(&mut offset, edges_bytes, "edges", rel)?;
        offset = offset
            .checked_add(pad8(edges_bytes))
            .ok_or_else(|| anyhow::anyhow!("section size overflows for edge type '{rel}'"))?;
        let weights_range = if hdr.has_weights {
            let r = take(&mut offset, edges_bytes, "weights", rel)?;
            offset = offset
                .checked_add(pad8(edges_bytes))
                .ok_or_else(|| anyhow::anyhow!("section size overflows for edge type '{rel}'"))?;
            Some(r)
        } else {
            None
        };
        ensure!(
            offset <= file_len,
            "file truncated: section padding of edge type '{rel}' extends past EOF"
        );

        let graph = Graph::from_mapped_parts(
            hdr.num_src_nodes,
            hdr.num_edges,
            Arc::clone(&mmap),
            offsets_range,
            edges_range,
            weights_range,
        );

        // `et_headers` is consumed by value, so move the names instead of
        // cloning them into the edge-type list.
        edge_types.push((hdr.src_name, hdr.rel_name, hdr.dst_name, graph));
    }

    HeteroGraph::try_from_parts(node_types, edge_types)
        .context("hetero graph type tables are inconsistent — file is corrupt")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_save_load() {
        let mut votes = Vec::new();
        for u in 0u32..10 {
            for p in 0u32..3 {
                votes.push((u, p));
            }
        }
        let votes_csr = Graph::from_edges(20, &votes, None).unwrap();

        let mut belongs = Vec::new();
        for p in 0u32..20 {
            belongs.push((p, p % 5));
        }
        let belongs_csr = Graph::from_edges(20, &belongs, None).unwrap();

        let graph = HeteroGraph::from_parts(
            vec![
                ("user".into(), 10),
                ("post".into(), 20),
                ("subreddit".into(), 5),
            ],
            vec![
                ("user".into(), "votes".into(), "post".into(), votes_csr),
                (
                    "post".into(),
                    "belongs_to".into(),
                    "subreddit".into(),
                    belongs_csr,
                ),
            ],
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hetero.bin");

        save_hetero_graph(&graph, &path).unwrap();
        let loaded = load_hetero_graph(&path).unwrap();

        assert_eq!(loaded.node_type_count(), 3);
        assert_eq!(loaded.edge_type_count(), 2);
        assert_eq!(loaded.num_nodes(loaded.node_type_id("user").unwrap()), 10);
        assert_eq!(loaded.num_nodes(loaded.node_type_id("post").unwrap()), 20);
        assert_eq!(
            loaded.num_edges(loaded.edge_type_id("user", "votes", "post").unwrap()),
            30
        );
        assert_eq!(
            loaded.num_edges(
                loaded
                    .edge_type_id("post", "belongs_to", "subreddit")
                    .unwrap()
            ),
            20
        );
    }

    #[test]
    fn test_save_load_roundtrip() {
        // 2 node types, 2 edge types: verify full roundtrip fidelity.
        let mut author_edges = Vec::new();
        for a in 0u32..5 {
            for p in 0u32..4 {
                author_edges.push((a, p));
            }
        }
        let author_csr = Graph::from_edges(10, &author_edges, None).unwrap();

        let mut cite_edges = Vec::new();
        for p in 0u32..8 {
            cite_edges.push((p, (p + 1) % 15));
        }
        let cite_csr = Graph::from_edges(15, &cite_edges, None).unwrap();

        let graph = HeteroGraph::from_parts(
            vec![("author".into(), 10), ("paper".into(), 15)],
            vec![
                ("author".into(), "writes".into(), "paper".into(), author_csr),
                ("paper".into(), "cites".into(), "paper".into(), cite_csr),
            ],
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roundtrip.bin");

        save_hetero_graph(&graph, &path).unwrap();
        let loaded = load_hetero_graph(&path).unwrap();

        // Structural checks
        assert_eq!(loaded.node_type_count(), 2);
        assert_eq!(loaded.edge_type_count(), 2);
        assert_eq!(loaded.num_nodes(loaded.node_type_id("author").unwrap()), 10);
        assert_eq!(loaded.num_nodes(loaded.node_type_id("paper").unwrap()), 15);

        let writes_id = loaded.edge_type_id("author", "writes", "paper").unwrap();
        let cites_id = loaded.edge_type_id("paper", "cites", "paper").unwrap();
        assert_eq!(loaded.num_edges(writes_id), 20); // 5 authors * 4 papers
        assert_eq!(loaded.num_edges(cites_id), 8);

        // Verify CSR neighbor data survived
        let writes_csr = loaded.csr(writes_id);
        assert_eq!(writes_csr.neighbors(0), &[0u32, 1, 2, 3]);
        assert_eq!(writes_csr.neighbors(4), &[0u32, 1, 2, 3]);
        assert_eq!(writes_csr.degree(5), 0); // author 5 wrote nothing

        let cites_csr = loaded.csr(cites_id);
        assert_eq!(cites_csr.neighbors(0), &[1u32]);
        assert_eq!(cites_csr.neighbors(7), &[8u32]);
    }

    #[test]
    fn test_save_load_single_edge_type() {
        // Minimal: 1 node type, 1 edge type.
        let edges: Vec<(u32, u32)> = vec![(0, 1), (1, 2), (2, 0)];
        let csr = Graph::from_edges(3, &edges, None).unwrap();

        let graph = HeteroGraph::from_parts(
            vec![("node".into(), 3)],
            vec![("node".into(), "link".into(), "node".into(), csr)],
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("single_edge.bin");

        save_hetero_graph(&graph, &path).unwrap();
        let loaded = load_hetero_graph(&path).unwrap();

        assert_eq!(loaded.node_type_count(), 1);
        assert_eq!(loaded.edge_type_count(), 1);
        assert_eq!(loaded.num_nodes(0), 3);

        let et = loaded.edge_type_id("node", "link", "node").unwrap();
        assert_eq!(loaded.num_edges(et), 3);

        let csr = loaded.csr(et);
        assert_eq!(csr.neighbors(0), &[1u32]);
        assert_eq!(csr.neighbors(1), &[2u32]);
        assert_eq!(csr.neighbors(2), &[0u32]);
    }

    #[test]
    fn test_save_load_empty_edge_type() {
        // Edge type with 0 edges: verify structure is preserved.
        let empty_csr = Graph::from_edges(5, &[], None).unwrap();
        let nonempty_csr = Graph::from_edges(5, &[(0, 1), (1, 2)], None).unwrap();

        let graph = HeteroGraph::from_parts(
            vec![("a".into(), 5), ("b".into(), 3)],
            vec![
                ("a".into(), "empty_rel".into(), "b".into(), empty_csr),
                ("a".into(), "has_rel".into(), "a".into(), nonempty_csr),
            ],
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty_edge.bin");

        save_hetero_graph(&graph, &path).unwrap();
        let loaded = load_hetero_graph(&path).unwrap();

        assert_eq!(loaded.node_type_count(), 2);
        assert_eq!(loaded.edge_type_count(), 2);

        let empty_id = loaded.edge_type_id("a", "empty_rel", "b").unwrap();
        let has_id = loaded.edge_type_id("a", "has_rel", "a").unwrap();

        assert_eq!(loaded.num_edges(empty_id), 0);
        assert_eq!(loaded.num_edges(has_id), 2);

        // Empty edge type CSR should have correct num_nodes but no neighbors
        let empty_loaded = loaded.csr(empty_id);
        assert_eq!(empty_loaded.num_nodes(), 5);
        assert_eq!(empty_loaded.num_edges(), 0);
        for n in 0..5u32 {
            assert!(empty_loaded.neighbors(n).is_empty());
        }
    }

    #[test]
    fn test_save_load_with_weights() {
        let edges: Vec<(u32, u32)> = vec![(0, 1), (0, 2), (1, 2)];
        let weights: Vec<f32> = vec![0.5, 1.0, 0.25];
        let csr = Graph::from_edges(3, &edges, Some(&weights)).unwrap();

        assert!(csr.weights().is_some());
        assert_eq!(csr.neighbor_weights(0), Some(&[0.5f32, 1.0][..]));

        let graph = HeteroGraph::from_parts(
            vec![("v".into(), 3)],
            vec![("v".into(), "e".into(), "v".into(), csr)],
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weighted.bin");

        save_hetero_graph(&graph, &path).unwrap();
        let loaded = load_hetero_graph(&path).unwrap();

        let et = loaded.edge_type_id("v", "e", "v").unwrap();
        assert_eq!(loaded.num_edges(et), 3);

        let loaded_csr = loaded.csr(et);
        assert_eq!(loaded_csr.neighbors(0), &[1u32, 2]);
        assert_eq!(loaded_csr.neighbors(1), &[2u32]);

        // Weights survive roundtrip
        assert!(loaded_csr.weights().is_some());
        assert_eq!(loaded_csr.neighbor_weights(0), Some(&[0.5f32, 1.0][..]));
        assert_eq!(loaded_csr.neighbor_weights(1), Some(&[0.25f32][..]));
    }

    #[test]
    fn test_mixed_weighted_and_unweighted_edge_types() {
        // Edge type A has weights, edge type B does not.
        // Verify both survive roundtrip with correct data and no cross-contamination.
        let edges_a: Vec<(u32, u32)> = vec![(0, 1), (0, 2), (1, 2), (2, 0)];
        let weights_a: Vec<f32> = vec![1.5, 2.5, 3.5, 4.5];
        let csr_a = Graph::from_edges(3, &edges_a, Some(&weights_a)).unwrap();

        let edges_b: Vec<(u32, u32)> = vec![(0, 1), (1, 0)];
        let csr_b = Graph::from_edges(3, &edges_b, None).unwrap();

        let graph = HeteroGraph::from_parts(
            vec![("x".into(), 3), ("y".into(), 3)],
            vec![
                ("x".into(), "weighted_rel".into(), "y".into(), csr_a),
                ("y".into(), "plain_rel".into(), "x".into(), csr_b),
            ],
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.bin");
        save_hetero_graph(&graph, &path).unwrap();
        let loaded = load_hetero_graph(&path).unwrap();

        let w_id = loaded.edge_type_id("x", "weighted_rel", "y").unwrap();
        let p_id = loaded.edge_type_id("y", "plain_rel", "x").unwrap();

        // Weighted edge type: topology + weights preserved
        let w_csr = loaded.csr(w_id);
        assert_eq!(w_csr.num_edges(), 4);
        assert!(w_csr.weights().is_some());
        assert_eq!(w_csr.neighbor_weights(0), Some(&[1.5f32, 2.5][..]));
        assert_eq!(w_csr.neighbor_weights(1), Some(&[3.5f32][..]));
        assert_eq!(w_csr.neighbor_weights(2), Some(&[4.5f32][..]));

        // Unweighted edge type: topology preserved, no weights (no bleed from A)
        let p_csr = loaded.csr(p_id);
        assert_eq!(p_csr.num_edges(), 2);
        assert!(p_csr.weights().is_none());
        assert_eq!(p_csr.neighbors(0), &[1u32]);
        assert_eq!(p_csr.neighbors(1), &[0u32]);
    }

    #[test]
    fn test_weights_roundtrip_large_values() {
        // Adversarial weight values: zero, negative, subnormal, large, NaN, infinity.
        // Verify bit-exact roundtrip via mmap (bytemuck cast, no float math).
        let edges: Vec<(u32, u32)> = (0..6).map(|i| (0, i + 1)).collect();
        let weights: Vec<f32> = vec![
            0.0,
            -1.0,
            f32::MIN_POSITIVE, // subnormal boundary
            f32::MAX,
            f32::NAN,
            f32::INFINITY,
        ];
        let csr = Graph::from_edges(7, &edges, Some(&weights)).unwrap();

        let graph = HeteroGraph::from_parts(
            vec![("n".into(), 7)],
            vec![("n".into(), "e".into(), "n".into(), csr)],
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adversarial_weights.bin");
        save_hetero_graph(&graph, &path).unwrap();
        let loaded = load_hetero_graph(&path).unwrap();

        let et = loaded.edge_type_id("n", "e", "n").unwrap();
        let loaded_w = loaded.csr(et).neighbor_weights(0).unwrap();

        assert_eq!(loaded_w[0], 0.0f32);
        assert_eq!(loaded_w[1], -1.0f32);
        assert_eq!(loaded_w[2], f32::MIN_POSITIVE);
        assert_eq!(loaded_w[3], f32::MAX);
        assert!(loaded_w[4].is_nan()); // NaN != NaN, must use is_nan()
        assert_eq!(loaded_w[5], f32::INFINITY);
    }

    #[test]
    fn test_truncated_file_with_weights() {
        // Save a weighted graph, then truncate the file mid-weights section.
        // Load should fail (not silently return garbage).
        let edges: Vec<(u32, u32)> = vec![(0, 1), (0, 2), (1, 2)];
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0];
        let csr = Graph::from_edges(3, &edges, Some(&weights)).unwrap();

        let graph = HeteroGraph::from_parts(
            vec![("n".into(), 3)],
            vec![("n".into(), "e".into(), "n".into(), csr)],
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.bin");
        save_hetero_graph(&graph, &path).unwrap();

        // Truncate: remove last 8 bytes (cuts into weights section)
        let data = std::fs::read(&path).unwrap();
        std::fs::write(&path, &data[..data.len() - 8]).unwrap();

        let result = load_hetero_graph(&path);
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("truncated"),
            "truncated file must fail at load time, got: {msg}"
        );
    }

    #[test]
    fn test_odd_edge_count_keeps_next_section_aligned() {
        // An unweighted edge type with an odd edge count leaves the file
        // cursor 4 bytes past an 8-byte boundary; the writer must pad so
        // the NEXT edge type's u64 offsets stay aligned. Reading the
        // second type's neighbors exercises the zero-copy u64 cast.
        let odd_edges: Vec<(u32, u32)> = vec![(0, 1), (1, 2), (2, 0)]; // 3 edges (odd)
        let odd_csr = Graph::from_edges(3, &odd_edges, None).unwrap();
        let second_edges: Vec<(u32, u32)> = vec![(0, 2), (1, 0)];
        let second_csr = Graph::from_edges(3, &second_edges, None).unwrap();

        let graph = HeteroGraph::from_parts(
            vec![("n".into(), 3)],
            vec![
                ("n".into(), "odd".into(), "n".into(), odd_csr),
                ("n".into(), "second".into(), "n".into(), second_csr),
            ],
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("odd.bin");
        save_hetero_graph(&graph, &path).unwrap();
        let loaded = load_hetero_graph(&path).unwrap();

        let odd_id = loaded.edge_type_id("n", "odd", "n").unwrap();
        let second_id = loaded.edge_type_id("n", "second", "n").unwrap();
        assert_eq!(loaded.csr(odd_id).neighbors(0), &[1u32]);
        assert_eq!(loaded.csr(second_id).neighbors(0), &[2u32]);
        assert_eq!(loaded.csr(second_id).neighbors(1), &[0u32]);
    }

    #[test]
    fn test_corrupt_type_counts_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("counts.bin");

        // Valid magic + version, absurd type counts. Must error without
        // attempting giant allocations.
        let mut bytes = vec![0u8; HEADER_SIZE];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..12].copy_from_slice(&VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        bytes[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let msg = load_hetero_graph(&path).unwrap_err().to_string();
        assert!(msg.contains("exceed the format limit"), "got: {msg}");
    }

    #[test]
    fn test_truncated_type_table_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("table.bin");

        // Header promises one node type but the file ends after the header.
        let mut bytes = vec![0u8; HEADER_SIZE];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..12].copy_from_slice(&VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&1u32.to_le_bytes());
        bytes[16..20].copy_from_slice(&0u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let msg = load_hetero_graph(&path).unwrap_err().to_string();
        assert!(msg.contains("truncated"), "got: {msg}");
    }

    #[test]
    fn test_newer_version_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("future.bin");

        let mut bytes = vec![0u8; HEADER_SIZE];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..12].copy_from_slice(&(VERSION + 1).to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let msg = load_hetero_graph(&path).unwrap_err().to_string();
        assert!(msg.contains("not supported"), "got: {msg}");
    }

    #[test]
    fn test_corrupt_edge_count_rejected() {
        // Build a valid file, then corrupt one edge type's num_edges so
        // the section ranges extend past EOF.
        let csr = Graph::from_edges(3, &[(0, 1), (1, 2)], None).unwrap();
        let graph = HeteroGraph::from_parts(
            vec![("n".into(), 3)],
            vec![("n".into(), "e".into(), "n".into(), csr)],
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("badcount.bin");
        save_hetero_graph(&graph, &path).unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        // num_edges lives at header + node table + 200 bytes into the
        // (single) edge type entry.
        let pos = HEADER_SIZE + NODE_TYPE_ENTRY_SIZE + 200;
        bytes[pos..pos + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        assert!(load_hetero_graph(&path).is_err());
    }

    #[test]
    fn test_load_corrupted_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupted.bin");

        // Write garbage that is at least HEADER_SIZE (64) bytes but has wrong magic
        let garbage = vec![0xFFu8; 128];
        std::fs::write(&path, &garbage).unwrap();
        let result = load_hetero_graph(&path);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("invalid magic"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_save_load_large_graph() {
        // 3 node types, 4 edge types, >10K edges total.
        //   user(500), item(1000), tag(50)
        //   user->buys->item  : 500 users buy 8 items each  = 4000
        //   user->likes->item : 500 users like 6 items each = 3000
        //   item->tagged->tag : 1000 items each tagged once  = 1000
        //   user->follows->user: 500 users follow 5 each    = 2500
        //   total = 10500

        // CSR num_nodes must cover both source and destination IDs.
        // For cross-type edges, use max(src_count, dst_count).
        let mut buys = Vec::new();
        for u in 0u32..500 {
            for i in 0u32..8 {
                buys.push((u, (u * 3 + i) % 1000));
            }
        }
        let buys_csr = Graph::from_edges(1000, &buys, None).unwrap();

        let mut likes = Vec::new();
        for u in 0u32..500 {
            for i in 0u32..6 {
                likes.push((u, (u * 7 + i) % 1000));
            }
        }
        let likes_csr = Graph::from_edges(1000, &likes, None).unwrap();

        let mut tagged = Vec::new();
        for item in 0u32..1000 {
            tagged.push((item, item % 50));
        }
        let tagged_csr = Graph::from_edges(1000, &tagged, None).unwrap();

        let mut follows = Vec::new();
        for u in 0u32..500 {
            for f in 0u32..5 {
                follows.push((u, (u + f + 1) % 500));
            }
        }
        let follows_csr = Graph::from_edges(500, &follows, None).unwrap();

        let graph = HeteroGraph::from_parts(
            vec![
                ("user".into(), 500),
                ("item".into(), 1000),
                ("tag".into(), 50),
            ],
            vec![
                ("user".into(), "buys".into(), "item".into(), buys_csr),
                ("user".into(), "likes".into(), "item".into(), likes_csr),
                ("item".into(), "tagged".into(), "tag".into(), tagged_csr),
                ("user".into(), "follows".into(), "user".into(), follows_csr),
            ],
        );

        assert_eq!(graph.total_edges(), 10500);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large.bin");

        save_hetero_graph(&graph, &path).unwrap();
        let loaded = load_hetero_graph(&path).unwrap();

        // Verify type counts
        assert_eq!(loaded.node_type_count(), 3);
        assert_eq!(loaded.edge_type_count(), 4);

        // Verify node counts
        assert_eq!(loaded.num_nodes(loaded.node_type_id("user").unwrap()), 500);
        assert_eq!(loaded.num_nodes(loaded.node_type_id("item").unwrap()), 1000);
        assert_eq!(loaded.num_nodes(loaded.node_type_id("tag").unwrap()), 50);

        // Verify edge counts
        let buys_id = loaded.edge_type_id("user", "buys", "item").unwrap();
        let likes_id = loaded.edge_type_id("user", "likes", "item").unwrap();
        let tagged_id = loaded.edge_type_id("item", "tagged", "tag").unwrap();
        let follows_id = loaded.edge_type_id("user", "follows", "user").unwrap();

        assert_eq!(loaded.num_edges(buys_id), 4000);
        assert_eq!(loaded.num_edges(likes_id), 3000);
        assert_eq!(loaded.num_edges(tagged_id), 1000);
        assert_eq!(loaded.num_edges(follows_id), 2500);
        assert_eq!(loaded.total_edges(), 10500);

        // Spot-check CSR neighbor data
        let buys_loaded = loaded.csr(buys_id);
        assert_eq!(buys_loaded.degree(0), 8);
        // user 0 buys items: (0*3 + i) % 1000 for i in 0..8 = [0,1,2,3,4,5,6,7]
        assert_eq!(buys_loaded.neighbors(0), &[0u32, 1, 2, 3, 4, 5, 6, 7]);

        let tagged_loaded = loaded.csr(tagged_id);
        assert_eq!(tagged_loaded.degree(0), 1);
        assert_eq!(tagged_loaded.neighbors(0), &[0u32]); // item 0 -> tag 0
        assert_eq!(tagged_loaded.neighbors(49), &[49u32]); // item 49 -> tag 49
        assert_eq!(tagged_loaded.neighbors(50), &[0u32]); // item 50 -> tag 0

        let follows_loaded = loaded.csr(follows_id);
        assert_eq!(follows_loaded.degree(0), 5);
        // user 0 follows: (0+f+1)%500 for f in 0..5 = [1,2,3,4,5]
        assert_eq!(follows_loaded.neighbors(0), &[1u32, 2, 3, 4, 5]);
    }
}
