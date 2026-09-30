#![no_main]
//! Fuzz target: feed arbitrary bytes to the CSR graph loader and assert it
//! either returns a graph that upholds its invariants or an `Err`. The
//! loader must NEVER panic, and a graph loaded under a weaker validation
//! mode must stay memory-safe: guarded accessors agree with each other,
//! and whole-graph rebuilds (`permute`, Rabbit Order) either prove the
//! graph first or refuse it.
//!
//! Run:  cargo +nightly fuzz run csr_loader_bytes

use libfuzzer_sys::fuzz_target;

use aethergraph_core::{GraphValidationMode, NodeId, load_graph_owned, load_graph_with_validation};

/// Graphs above this size skip the O(V + E) rebuilds, keeping each fuzz
/// iteration cheap.
const REBUILD_LIMIT: usize = 1 << 12;

fuzz_target!(|data: &[u8]| {
    // Write the fuzz input to a temp file (the loader is path-based).
    // libfuzzer reuses a tmpfs-backed temp dir so this is cheap.
    let tmp = std::env::temp_dir().join("aether_fuzz_csr.bin");
    if std::fs::write(&tmp, data).is_err() {
        return;
    }

    // Drive every validation mode — each is a separate code path with its
    // own potential panic sites.
    for mode in [
        GraphValidationMode::HeaderOnly,
        GraphValidationMode::OffsetsOnly,
        GraphValidationMode::Full,
    ] {
        if let Ok(graph) = load_graph_with_validation(&tmp, mode) {
            assert!(graph.validated() >= mode);
            exercise(&graph);
        }
        if let Ok(graph) = load_graph_owned(&tmp, mode) {
            assert_eq!(graph.validated(), GraphValidationMode::Full);
            exercise(&graph);
        }
    }
});

fn exercise(graph: &aethergraph_core::Graph) {
    let degrees = graph.degrees();
    assert_eq!(degrees.len(), graph.num_nodes());
    for v in 0..graph.num_nodes().min(64) as NodeId {
        assert_eq!(graph.degree(v), graph.neighbors(v).len());
        assert_eq!(degrees[v as usize] as usize, graph.degree(v));
    }
    let _ = graph.stats();
    if graph.num_nodes() > REBUILD_LIMIT || graph.num_edges() > REBUILD_LIMIT * 16 {
        return;
    }
    let identity: Vec<NodeId> = (0..graph.num_nodes() as NodeId).collect();
    match graph.permute(&identity) {
        Ok(permuted) => {
            // A successful permute proved the source Full.
            assert_eq!(graph.validated(), GraphValidationMode::Full);
            assert_eq!(permuted.edges(), graph.edges());
            let perm = graph.reorder_rabbit().expect("a Full graph reorders");
            assert_eq!(perm.len(), graph.num_nodes());
        }
        Err(_) => assert!(graph.validate().is_err()),
    }
}
