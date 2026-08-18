//! Version-2 graph file: succinct-coded CSR at rest.
//!
//! The version-1 format stores the CSR arrays flat (8 bytes per offset,
//! 4 per edge). This format stores the same graph through the codecs in
//! [`internal::succinct`](super::succinct): the offsets prefix-sum as
//! Elias-Fano (a few bits per node) and the edges array as StreamVByte
//! wrapping deltas (sorted neighbor lists delta down to ~1 byte per
//! edge). Weights, when present, stay raw `f32le` — they don't compress
//! through integer codecs.
//!
//! Compression is an at-rest property only: loading decodes into owned
//! flat arrays, so sampling runs at exactly the speed of an owned
//! version-1 load. That is why an explicit mmap-storage load of a
//! compressed file is refused rather than silently degraded — the caller
//! asked for page-cache-backed storage the format cannot provide.
//!
//! Both formats share the 32-byte header ([`super::mmap::Header`]); the
//! `version` field is the dispatcher, and the checksum field covers the
//! compressed payload here (v1 checksums the flat arrays instead).

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use bytemuck::cast_slice;
use tracing::debug;

use super::mmap::{Header, crc32_parts};
use super::succinct::{EliasFano, StreamVByte};
use crate::graph::{Graph, GraphValidationMode};

/// `version` header value for this format. Version 1 is the flat format
/// in [`super::mmap`].
pub(crate) const COMPRESSED_VERSION: u32 = 2;

/// Serialize `graph` to `path` in the compressed format.
pub fn save_graph_compressed(graph: &Graph, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();

    let ef = EliasFano::encode(graph.offsets());
    let svb = StreamVByte::encode_deltas(graph.edges());

    let mut payload = Vec::with_capacity(ef.heap_bytes() + svb.heap_bytes() + 64);
    ef.write_into(&mut payload);
    svb.write_into(&mut payload);
    if let Some(weights) = graph.weights() {
        payload.extend_from_slice(cast_slice::<f32, u8>(weights));
    }

    let header = Header::for_compressed(
        graph.num_nodes(),
        graph.num_edges(),
        graph.weights().is_some(),
        crc32_parts(&[&payload]),
    );

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .context("failed to create output file")?;
    file.write_all(&header.to_bytes())
        .context("failed to write header")?;
    file.write_all(&payload)
        .context("failed to write compressed payload")?;
    file.sync_all().context("failed to sync file")?;

    let flat = graph.num_nodes() * 8 + 8 + graph.num_edges() * 4;
    debug!(
        "Compressed graph saved to {}: {} payload bytes vs {} flat ({:.2}x)",
        path.display(),
        payload.len(),
        flat,
        flat as f64 / payload.len().max(1) as f64
    );
    Ok(())
}

/// Load a compressed graph from `path` into owned storage.
pub fn load_graph_compressed(
    path: impl AsRef<Path>,
    validation: GraphValidationMode,
) -> Result<Graph> {
    let bytes = std::fs::read(path.as_ref()).context("failed to read graph file")?;
    load_graph_compressed_from_bytes(&bytes, validation)
}

/// Decode a compressed graph from in-memory file bytes.
pub(crate) fn load_graph_compressed_from_bytes(
    bytes: &[u8],
    validation: GraphValidationMode,
) -> Result<Graph> {
    let header = Header::from_bytes(bytes)?;
    header.validate_compressed()?;
    let payload = &bytes[Header::SIZE..];

    // The checksum covers the whole compressed payload. Unlike v1 (where
    // Full validation alone pays for hashing multi-GB flat arrays), the
    // payload here is small and is being fully decoded anyway, so every
    // load with a checksum verifies it.
    if let Some(expected) = header.checksum32() {
        let actual = crc32_parts(&[payload]);
        anyhow::ensure!(
            actual == expected,
            "compressed graph checksum mismatch: expected {expected:#x}, got {actual:#x}"
        );
    }

    let num_nodes = usize::try_from(header.num_nodes()).context("num_nodes")?;
    let num_edges = usize::try_from(header.num_edges()).context("num_edges")?;

    let (ef, ef_len) = EliasFano::read_from(payload).context("offsets section")?;
    let (svb, svb_len) = StreamVByte::read_from(&payload[ef_len..]).context("edges section")?;

    anyhow::ensure!(
        ef.len() == num_nodes + 1,
        "offsets section has {} entries for {num_nodes} nodes",
        ef.len()
    );
    anyhow::ensure!(
        svb.len() == num_edges,
        "edges section has {} entries, header says {num_edges}",
        svb.len()
    );

    let offsets = ef.to_vec();
    let edges = svb.decode();

    let weights = if header.has_weights() {
        let raw = &payload[ef_len + svb_len..];
        anyhow::ensure!(
            raw.len() == num_edges * 4,
            "weights section is {} bytes for {num_edges} edges",
            raw.len()
        );
        let mut w: Vec<f32> = vec![0.0; num_edges];
        bytemuck::cast_slice_mut::<f32, u8>(&mut w).copy_from_slice(raw);
        Some(w)
    } else {
        anyhow::ensure!(
            ef_len + svb_len == payload.len(),
            "trailing bytes after edges section"
        );
        None
    };

    let graph = Graph::from_csr_arrays(num_nodes, offsets, edges, weights);
    graph.validate_with_mode(validation)?;
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::NodeId;
    use tempfile::NamedTempFile;

    /// A small scale-free-ish graph with sorted neighbor lists.
    fn test_graph(weighted: bool) -> Graph {
        let num_nodes = 500usize;
        let mut offsets = vec![0u64];
        let mut edges: Vec<NodeId> = Vec::new();
        for v in 0..num_nodes {
            let degree = 1 + (v * 7) % 23;
            let mut list: Vec<NodeId> = (0..degree)
                .map(|k| ((v * 31 + k * 97) % num_nodes) as NodeId)
                .collect();
            list.sort_unstable();
            list.dedup();
            edges.extend_from_slice(&list);
            offsets.push(edges.len() as u64);
        }
        let weights =
            weighted.then(|| (0..edges.len()).map(|i| i as f32 * 0.5).collect::<Vec<_>>());
        Graph::from_csr_arrays(num_nodes, offsets, edges, weights)
    }

    fn assert_same_graph(a: &Graph, b: &Graph) {
        assert_eq!(a.num_nodes(), b.num_nodes());
        assert_eq!(a.num_edges(), b.num_edges());
        assert_eq!(a.offsets(), b.offsets());
        assert_eq!(a.edges(), b.edges());
        assert_eq!(a.weights(), b.weights());
    }

    #[test]
    fn round_trips_unweighted() {
        let graph = test_graph(false);
        let tmp = NamedTempFile::new().unwrap();
        save_graph_compressed(&graph, tmp.path()).unwrap();
        let back = load_graph_compressed(tmp.path(), GraphValidationMode::Full).unwrap();
        assert_same_graph(&graph, &back);
    }

    #[test]
    fn round_trips_weighted() {
        let graph = test_graph(true);
        let tmp = NamedTempFile::new().unwrap();
        save_graph_compressed(&graph, tmp.path()).unwrap();
        let back = load_graph_compressed(tmp.path(), GraphValidationMode::Full).unwrap();
        assert_same_graph(&graph, &back);
    }

    #[test]
    fn compresses_below_flat() {
        let graph = test_graph(false);
        let tmp = NamedTempFile::new().unwrap();
        save_graph_compressed(&graph, tmp.path()).unwrap();
        let compressed = std::fs::metadata(tmp.path()).unwrap().len() as usize;
        let flat = (graph.num_nodes() + 1) * 8 + graph.num_edges() * 4;
        assert!(
            compressed < flat / 2,
            "compressed {compressed} bytes vs flat {flat}"
        );
    }

    #[test]
    fn detects_payload_corruption() {
        let graph = test_graph(false);
        let tmp = NamedTempFile::new().unwrap();
        save_graph_compressed(&graph, tmp.path()).unwrap();
        let mut bytes = std::fs::read(tmp.path()).unwrap();
        let mid = Header::SIZE + (bytes.len() - Header::SIZE) / 2;
        bytes[mid] ^= 0xFF;
        let err = load_graph_compressed_from_bytes(&bytes, GraphValidationMode::Full)
            .unwrap_err()
            .to_string();
        assert!(err.contains("checksum"), "unexpected error: {err}");
    }

    #[test]
    fn dispatches_through_load_graph() {
        // The public auto loader must route on the version field.
        let graph = test_graph(true);
        let tmp = NamedTempFile::new().unwrap();
        save_graph_compressed(&graph, tmp.path()).unwrap();
        let back = super::super::mmap::load_graph(tmp.path()).unwrap();
        assert_same_graph(&graph, &back);
    }

    #[test]
    fn explicit_mmap_load_refuses_compressed() {
        let graph = test_graph(false);
        let tmp = NamedTempFile::new().unwrap();
        save_graph_compressed(&graph, tmp.path()).unwrap();
        let err = super::super::mmap::load_graph_mmap(tmp.path(), GraphValidationMode::Full)
            .unwrap_err()
            .to_string();
        assert!(err.contains("compressed"), "unexpected error: {err}");
    }

    #[test]
    fn owned_load_accepts_compressed() {
        let graph = test_graph(false);
        let tmp = NamedTempFile::new().unwrap();
        save_graph_compressed(&graph, tmp.path()).unwrap();
        let back =
            super::super::mmap::load_graph_owned(tmp.path(), GraphValidationMode::Full).unwrap();
        assert_same_graph(&graph, &back);
    }

    #[test]
    fn sampler_runs_on_loaded_graph() {
        use crate::loader::{NeighborSampler, SamplingConfig};

        let graph = test_graph(false);
        let tmp = NamedTempFile::new().unwrap();
        save_graph_compressed(&graph, tmp.path()).unwrap();
        let back = super::super::mmap::load_graph(tmp.path()).unwrap();

        let config = SamplingConfig {
            fanout: vec![5, 3],
            ..Default::default()
        };
        let mut sampler = NeighborSampler::new(&back, config);
        let sub = sampler.sample_neighbors(&[0, 1, 2, 3]);
        assert!(sub.num_nodes() >= 4);
    }
}
