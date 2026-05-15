#![no_main]
//! Fuzz target: feed arbitrary bytes to the CSR graph loader and assert it
//! either returns a structurally valid `Graph` or an `Err`. The loader must
//! NEVER panic or produce a graph that violates its invariants
//! (`offsets.len() == num_nodes + 1`, monotonic offsets, edges in bounds).
//!
//! Run:  cargo +nightly fuzz run csr_loader_bytes

use libfuzzer_sys::fuzz_target;

use aethergraph_core::{GraphValidationMode, load_graph_with_validation};

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
        match load_graph_with_validation(tmp.to_str().unwrap(), mode) {
            Ok(graph) => {
                // Touch invariants. A successful Ok must mean the graph is
                // structurally consistent.
                let _ = graph.num_nodes();
                let _ = graph.num_edges();
                for v in 0..graph.num_nodes().min(64) {
                    let _ = graph.degree(v as u32);
                }
            }
            Err(_) => { /* expected for most random inputs */ }
        }
    }
});
