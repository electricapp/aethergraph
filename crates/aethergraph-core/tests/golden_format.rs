//! Golden-file regression tests for on-disk binary formats.
//!
//! Pins the bit-level layout of every format we read or write so a silent
//! drift (struct reorder, padding change, magic typo) fails CI on the
//! first affected commit. When a format is intentionally evolved, the
//! corresponding golden bytes in this file MUST be updated as part of the
//! same commit — anyone reviewing the PR sees the byte diff directly.
//!
//! See `ARCH.md > Binary Formats` for the schema-evolution policy.

use std::io::Write;

use aethergraph_core::{Graph, GraphValidationMode, load_graph_with_validation, save_graph};

/// Build a tiny 3-node / 3-edge unweighted graph deterministically.
fn small_graph() -> Graph {
    let edges = vec![(0u32, 1u32), (0, 2), (1, 2)];
    Graph::from_edges(3, &edges, None).expect("build graph")
}

#[test]
fn graph_v1_roundtrip_matches_golden_bytes() {
    // Step 1: build a small graph, write it to a temp file, read the file
    // bytes back. We don't hardcode the entire file because the
    // integrity_checksum32 field is allowed to be 0 OR a real checksum
    // depending on the writer's current policy. Instead we pin the
    // header layout PRECISELY and assert the body matches what we built.
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path().to_path_buf();
    let graph = small_graph();
    save_graph(&graph, &path).expect("save");

    let bytes = std::fs::read(&path).expect("read");
    assert!(bytes.len() >= 32, "file too short for header");

    // Magic + version are pinned forever (anything else is a v2).
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    assert_eq!(magic, 0x4145_5448, "magic must remain 'AETH' (0x41455448)");
    assert_eq!(version, 1, "current Graph format is v1; bump deliberately");

    // Counts.
    let num_nodes = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let num_edges = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let has_weights = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
    assert_eq!(num_nodes, 3);
    assert_eq!(num_edges, 3);
    assert_eq!(has_weights, 0);

    // Offsets array (4 entries × 8 bytes for 3 nodes + 1 sentinel).
    let body = &bytes[32..];
    assert!(
        body.len() >= 4 * 8 + 3 * 4,
        "body too short for offsets+edges"
    );
    let offsets: Vec<u64> = (0..4)
        .map(|i| {
            let s = i * 8;
            u64::from_le_bytes(body[s..s + 8].try_into().unwrap())
        })
        .collect();
    assert_eq!(offsets, vec![0, 2, 3, 3], "CSR offsets for this graph");

    // Edges array.
    let edge_base = 4 * 8;
    let edges: Vec<u32> = (0..3)
        .map(|i| {
            let s = edge_base + i * 4;
            u32::from_le_bytes(body[s..s + 4].try_into().unwrap())
        })
        .collect();
    assert_eq!(edges, vec![1, 2, 2], "CSR edges for this graph");

    // Round-trip: loader must produce a structurally identical graph.
    let loaded = load_graph_with_validation(&path, GraphValidationMode::Full).expect("load");
    assert_eq!(loaded.num_nodes(), 3);
    assert_eq!(loaded.num_edges(), 3);
    assert_eq!(loaded.neighbors(0), &[1, 2]);
    assert_eq!(loaded.neighbors(1), &[2]);
    assert_eq!(loaded.neighbors(2), &[]);
}

#[test]
fn graph_loader_rejects_newer_version() {
    // Write a synthetic file with version = 9999 and verify the loader
    // refuses it rather than mis-parsing. Forward-compat policy: a wheel
    // that knows version N must reject version N+1.
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let mut f = tmp.reopen().expect("reopen");

    // 32-byte header with magic = AETH, version = 9999, plausible counts.
    let mut hdr = [0u8; 32];
    hdr[0..4].copy_from_slice(&0x4145_5448u32.to_le_bytes());
    hdr[4..8].copy_from_slice(&9999u32.to_le_bytes());
    hdr[8..16].copy_from_slice(&0u64.to_le_bytes()); // num_nodes
    hdr[16..24].copy_from_slice(&0u64.to_le_bytes()); // num_edges
    hdr[24..28].copy_from_slice(&0u32.to_le_bytes()); // has_weights
    hdr[28..32].copy_from_slice(&0u32.to_le_bytes()); // checksum
    f.write_all(&hdr).expect("write");

    // A bogus offsets array (one u64 = 0) so the file isn't suspiciously short.
    f.write_all(&0u64.to_le_bytes()).expect("write offsets");
    drop(f);

    let result = load_graph_with_validation(tmp.path(), GraphValidationMode::HeaderOnly);
    assert!(
        result.is_err(),
        "loader must reject newer-than-known versions"
    );
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.to_ascii_lowercase().contains("version"),
        "error must mention version mismatch; got: {err}"
    );
}

#[test]
fn graph_loader_rejects_bad_magic() {
    // Sanity check: corrupt magic must trip the header validator.
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let mut f = tmp.reopen().expect("reopen");
    let mut hdr = [0u8; 32];
    hdr[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes()); // not AETH
    hdr[4..8].copy_from_slice(&1u32.to_le_bytes());
    f.write_all(&hdr).expect("write");
    drop(f);
    let result = load_graph_with_validation(tmp.path(), GraphValidationMode::HeaderOnly);
    assert!(result.is_err(), "loader must reject bad magic");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.to_ascii_lowercase().contains("magic"),
        "error must mention magic; got: {err}"
    );
}
