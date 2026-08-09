//! Exhaustive single-point fault injection for WAL recovery.
//!
//! Durability is a claim, so it is a test. Take a known-good WAL and apply
//! every single-point fault a torn flush or bit-rot can produce — truncate at
//! every byte offset, and flip a bit at every byte offset — then assert
//! recovery always lands on a *valid prefix* of the committed records, or
//! cleanly rejects the file. It must never panic, never replay a record that
//! was never written, and never reorder what it does replay.

#![cfg(feature = "wal")]

use aether_graph::{DynamicGraph, replay_wal};

// On-disk framing (the `aether_graph::wal` constants are crate-private). The
// length assertion in `build_wal` fails loudly if the real format drifts.
const HEADER_LEN: usize = 16;
const RECORD_LEN: usize = 12;

fn sample_edges() -> Vec<(u32, u32)> {
    (0..16u32).map(|i| (i, i + 1)).collect()
}

/// Build a fully-committed WAL and return its exact bytes.
fn build_wal(edges: &[(u32, u32)]) -> Vec<u8> {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    {
        let g = DynamicGraph::open_with_wal(tmp.path(), 64, 1 << 20).unwrap();
        let mut w = g.writer().unwrap();
        for &(s, d) in edges {
            w.insert_edge(s, d).unwrap();
        }
        // Dropping the writer fsyncs the records.
    }
    let bytes = std::fs::read(tmp.path()).unwrap();
    assert_eq!(
        bytes.len(),
        HEADER_LEN + edges.len() * RECORD_LEN,
        "on-disk WAL framing drifted from this test's assumptions"
    );
    bytes
}

/// Replay `path`, returning the recovered edges, or `Err` if recovery rejected
/// the file outright. Also cross-checks the callback count against the
/// reported `applied` total.
fn replay_edges(path: &std::path::Path) -> Result<Vec<(u32, u32)>, ()> {
    let mut got = Vec::new();
    match replay_wal(path, |rec| {
        got.push((rec.src, rec.dst));
        Ok(())
    }) {
        Ok(outcome) => {
            assert_eq!(
                got.len() as u64,
                outcome.applied,
                "callback fired a different number of times than `applied` reports"
            );
            Ok(got)
        }
        Err(_) => Err(()),
    }
}

/// Truncation at any offset must recover exactly the complete records that
/// survived the cut — no rejection, no partial record.
#[test]
fn truncation_at_every_offset_recovers_a_clean_prefix() {
    let edges = sample_edges();
    let bytes = build_wal(&edges);
    let tmp = tempfile::NamedTempFile::new().unwrap();

    for cut in 0..=bytes.len() {
        std::fs::write(tmp.path(), &bytes[..cut]).unwrap();
        let got = replay_edges(tmp.path())
            .expect("a truncated WAL must always recover a prefix, never reject");
        let complete = cut.saturating_sub(HEADER_LEN) / RECORD_LEN;
        assert_eq!(
            got,
            edges[..complete],
            "cut={cut}: unexpected recovered prefix"
        );
    }
}

/// Flipping any single byte must either reject the file (header damage) or
/// recover an exact prefix of the committed edges (a damaged record fails its
/// CRC and truncates the tail). It must never surface a fabricated edge.
#[test]
fn single_byte_corruption_recovers_a_prefix_or_rejects() {
    let edges = sample_edges();
    let bytes = build_wal(&edges);
    let tmp = tempfile::NamedTempFile::new().unwrap();

    for off in 0..bytes.len() {
        let mut corrupt = bytes.clone();
        corrupt[off] ^= 0xFF;
        std::fs::write(tmp.path(), &corrupt).unwrap();

        if let Ok(got) = replay_edges(tmp.path()) {
            assert!(
                got.len() <= edges.len(),
                "off={off}: replayed more records than were written"
            );
            assert_eq!(
                got.as_slice(),
                &edges[..got.len()],
                "off={off}: recovered records are not a prefix"
            );
        }
    }
}
