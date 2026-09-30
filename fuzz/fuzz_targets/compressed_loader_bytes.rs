#![no_main]
//! Fuzz target: feed arbitrary bytes to the compressed (version-2) graph
//! loader. It must return `Err` or a `Full`-validated graph — the Elias-Fano
//! and StreamVByte parsers prove what they decode, so no input may panic,
//! and every accessor on a returned graph must stay in bounds.
//!
//! Run:  cargo +nightly fuzz run compressed_loader_bytes

use libfuzzer_sys::fuzz_target;

use aethergraph_core::{GraphValidationMode, NodeId, load_graph_compressed};

/// Graphs above this size skip the O(V + E) reorder calls, keeping each
/// fuzz iteration cheap.
const REORDER_LIMIT: usize = 1 << 12;

fuzz_target!(|data: &[u8]| {
    let tmp = std::env::temp_dir().join("aether_fuzz_compressed.bin");
    if std::fs::write(&tmp, data).is_err() {
        return;
    }
    let Ok(graph) = load_graph_compressed(&tmp, GraphValidationMode::Full) else {
        return;
    };
    assert_eq!(graph.validated(), GraphValidationMode::Full);
    assert_eq!(graph.degrees().len(), graph.num_nodes());
    for v in 0..graph.num_nodes().min(64) as NodeId {
        assert_eq!(graph.degree(v), graph.neighbors(v).len());
        assert!(
            graph
                .neighbors(v)
                .iter()
                .all(|&d| (d as usize) < graph.num_nodes())
        );
    }
    if graph.num_nodes() <= REORDER_LIMIT && graph.num_edges() <= REORDER_LIMIT * 16 {
        let perm = graph.reorder_rabbit().expect("a Full graph reorders");
        let permuted = graph.permute(&perm).expect("a Full graph permutes");
        assert_eq!(permuted.num_edges(), graph.num_edges());
    }
});
