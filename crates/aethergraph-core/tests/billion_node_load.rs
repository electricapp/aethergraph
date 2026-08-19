//! TEST_PLAN R2 — billion-node mmap load works at scale.
//!
//! Gated on `AETHER_BIG_GRAPH=<path>`; if the file isn't there, the test
//! streams a synthetic CSR graph to that path first. Defaults: 1B nodes,
//! avg-degree 15, seed 0 → ~68 GB on disk. Override via
//! `AETHER_BIG_GRAPH_NODES` / `AETHER_BIG_GRAPH_DEGREE` / `AETHER_BIG_GRAPH_SEED`.
//!
//! Run on a box with ≥70 GB of fast free NVMe at the target path, e.g.
//!
//!     AETHER_BIG_GRAPH=/nvme/graph_1B.bin \
//!         cargo test --release -p aethergraph-core --test billion_node_load \
//!         -- --nocapture --ignored

use aethergraph_core::{GraphValidationMode, load_graph, load_graph_mmap};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

// Kept local to this test; `Header` in internal/mmap.rs is module-private and
// exposing it for a single scale test isn't worth widening the library surface.
const GRAPH_MAGIC: u32 = 0x4145_5448;
const GRAPH_VERSION: u32 = 1;
const GRAPH_HEADER_SIZE: usize = 32;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn generate_synthetic(path: &Path, num_nodes: usize, avg_degree: u32, seed: u64) {
    let num_edges = (num_nodes as u64) * u64::from(avg_degree);
    eprintln!(
        "generating {num_nodes} nodes × {avg_degree} degree = {num_edges} edges → {}",
        path.display()
    );

    let start = Instant::now();
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .expect("create output file");
    let mut w = BufWriter::with_capacity(16 * 1024 * 1024, file);

    // Header: magic, version, num_nodes, num_edges, has_weights=0, checksum=0.
    // checksum=0 means "absent" — checksums are optional and load_graph
    // accepts files without one.
    let mut header = [0u8; GRAPH_HEADER_SIZE];
    header[0..4].copy_from_slice(&GRAPH_MAGIC.to_le_bytes());
    header[4..8].copy_from_slice(&GRAPH_VERSION.to_le_bytes());
    header[8..16].copy_from_slice(&(num_nodes as u64).to_le_bytes());
    header[16..24].copy_from_slice(&num_edges.to_le_bytes());
    w.write_all(&header).unwrap();

    // Offsets: uniform degree → offsets[i] = i * avg_degree. Length N+1.
    let chunk = 1_000_000usize;
    let mut buf = vec![0u8; chunk * 8];
    let mut i = 0usize;
    while i <= num_nodes {
        let n = (num_nodes + 1 - i).min(chunk);
        for j in 0..n {
            let off = ((i + j) as u64) * u64::from(avg_degree);
            buf[j * 8..j * 8 + 8].copy_from_slice(&off.to_le_bytes());
        }
        w.write_all(&buf[..n * 8]).unwrap();
        i += n;
    }

    // Edges: for each node, `avg_degree` random u32 targets in [0, N).
    let mut rng = SmallRng::seed_from_u64(seed);
    let batch = 256 * 1024usize;
    let mut ebuf = vec![0u8; batch * 4];
    let mut emitted: u64 = 0;
    while emitted < num_edges {
        let n = ((num_edges - emitted) as usize).min(batch);
        for j in 0..n {
            let v: u32 = rng.random_range(0..num_nodes as u32);
            ebuf[j * 4..j * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        w.write_all(&ebuf[..n * 4]).unwrap();
        emitted += n as u64;
    }

    w.flush().unwrap();
    let file = w.into_inner().unwrap();
    file.sync_all().unwrap();

    let size = std::fs::metadata(path).unwrap().len();
    let elapsed = start.elapsed();
    eprintln!(
        "  wrote {:.2} GB in {:.1}s ({:.0} MB/s)",
        size as f64 / 1e9,
        elapsed.as_secs_f64(),
        (size as f64 / 1e6) / elapsed.as_secs_f64()
    );
}

#[test]
#[ignore = "scale test: needs ~70 GB NVMe. Set AETHER_BIG_GRAPH=<path> and pass --ignored"]
fn billion_node_mmap_load_and_walk() {
    let Some(path) = std::env::var("AETHER_BIG_GRAPH").ok() else {
        eprintln!("AETHER_BIG_GRAPH not set; skipping");
        return;
    };
    let path = std::path::PathBuf::from(path);
    let num_nodes: usize = env_or("AETHER_BIG_GRAPH_NODES", 1_000_000_000);
    let avg_degree: u32 = env_or("AETHER_BIG_GRAPH_DEGREE", 15);
    let seed: u64 = env_or("AETHER_BIG_GRAPH_SEED", 0);

    if !path.exists() {
        generate_synthetic(&path, num_nodes, avg_degree, seed);
    } else {
        eprintln!("reusing existing {}", path.display());
    }

    // Two load paths worth distinguishing at this size:
    //  (1) default load → OffsetsOnly validation on >512 MB files, which walks
    //      the full offsets array for monotonicity. For 1B+1 u64s that's 8 GB
    //      of cold disk reads — the dominant cost, not mmap setup.
    //  (2) HeaderOnly load → mmap + header-consistency checks only; expected
    //      to be sub-second since the offsets array stays lazily mapped.
    let t0 = Instant::now();
    let g = load_graph(&path).expect("load_graph default");
    let default_secs = t0.elapsed().as_secs_f64();
    eprintln!("load_graph (OffsetsOnly): {default_secs:.3}s");

    assert_eq!(g.num_nodes(), num_nodes, "num_nodes mismatch");
    let expected_edges = (num_nodes as u64) * u64::from(avg_degree);
    assert_eq!(
        g.num_edges() as u64,
        expected_edges,
        "num_edges mismatch (expected {expected_edges}, got {})",
        g.num_edges()
    );

    drop(g);

    let t0 = Instant::now();
    let g = load_graph_mmap(&path, GraphValidationMode::HeaderOnly).expect("load_graph HeaderOnly");
    let header_only_secs = t0.elapsed().as_secs_f64();
    eprintln!("load_graph (HeaderOnly):  {header_only_secs:.3}s");

    // Cold walk: touch a 0.1% sample of the offsets array. First access per
    // page faults in the mmap; timing this gives the working-set warm-up cost.
    let step = (num_nodes / 1_000).max(1);
    let mut checksum: u64 = 0;
    let t1 = Instant::now();
    for i in (0..num_nodes).step_by(step) {
        checksum = checksum.wrapping_add(g.degree(i as u32) as u64);
    }
    let walk_secs = t1.elapsed().as_secs_f64();
    eprintln!(
        "cold walk over {} sampled nodes: {:.3}s (degree checksum = {checksum})",
        num_nodes / step,
        walk_secs
    );

    // HeaderOnly is the genuine mmap-only path — it must stay fast regardless
    // of graph size. Default (OffsetsOnly) scales with offsets bytes at disk
    // read speed, so its timing is recorded but not gated here.
    assert!(
        header_only_secs < 2.0,
        "HeaderOnly load took {header_only_secs:.3}s — expected <2s for pure mmap path"
    );
}
