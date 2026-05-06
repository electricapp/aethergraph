//! TEST_PLAN R3 — NeighborLoader throughput at billion-node scale.
//!
//! Reuses the graph file produced by R2. Measures `batch_neighbors` latency on
//! the io_uring path (`AsyncCsrGraph`) at batch_size 1024, over many iterations,
//! and reports p50/p99 + batches/sec.
//!
//! Not a criterion bench: criterion preloads every registered benchmark's
//! setup code before honoring filters, which blows 15 GB RAM on g4dn.xlarge
//! with this graph (each `GraphFileView` allocates 8 GB of offsets).
//!
//! Run:
//!     AETHER_BIG_GRAPH=/nvme/graph_1B.bin \
//!         cargo test --release -p aethergraph-core \
//!         --test billion_node_throughput -- --nocapture --ignored

use aethergraph_core::{AsyncCsrGraph, NodeId};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::time::{Duration, Instant};

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn p_nanos(sorted: &[Duration], q: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx].as_nanos()
}

#[test]
#[ignore = "scale test: needs ~70 GB /nvme graph + ~8 GB RAM. Set AETHER_BIG_GRAPH=<path> and pass --ignored"]
fn billion_node_async_throughput() {
    let Ok(path) = std::env::var("AETHER_BIG_GRAPH") else {
        eprintln!("AETHER_BIG_GRAPH not set; skipping");
        return;
    };
    let batch_size: usize = env_or("AETHER_BENCH_BATCH_SIZE", 1024);
    let num_batches: usize = env_or("AETHER_BENCH_NUM_BATCHES", 200);
    let warmup: usize = env_or("AETHER_BENCH_WARMUP", 20);
    let seed: u64 = env_or("AETHER_BENCH_SEED", 42);

    let rt = tokio::runtime::Runtime::new().unwrap();

    let t_load = Instant::now();
    let graph = rt
        .block_on(AsyncCsrGraph::load(&path))
        .expect("AsyncCsrGraph::load");
    eprintln!(
        "AsyncCsrGraph::load: {:.3}s ({} nodes, {} edges, io_uring_active={})",
        t_load.elapsed().as_secs_f64(),
        graph.num_nodes(),
        graph.num_edges(),
        graph.io_uring_active()
    );
    assert!(
        graph.io_uring_active(),
        "expected io_uring fast path on Linux NVMe box; fell back to tokio"
    );

    // Pre-generate batches so batch sampling isn't in the hot loop.
    let mut rng = SmallRng::seed_from_u64(seed);
    let num_nodes = graph.num_nodes() as u32;
    let batches: Vec<Vec<NodeId>> = (0..(warmup + num_batches))
        .map(|_| {
            (0..batch_size)
                .map(|_| rng.random_range(0..num_nodes))
                .collect()
        })
        .collect();

    // Warmup — faults in the offsets array and sets up io_uring state.
    rt.block_on(async {
        for b in &batches[..warmup] {
            let _ = graph.batch_neighbors(b).await.unwrap();
        }
    });

    // Measure.
    let mut per_batch = Vec::with_capacity(num_batches);
    let t_total = Instant::now();
    rt.block_on(async {
        for b in &batches[warmup..] {
            let t = Instant::now();
            let _ = graph.batch_neighbors(b).await.unwrap();
            per_batch.push(t.elapsed());
        }
    });
    let total = t_total.elapsed();

    per_batch.sort();
    let mean_ns: u128 =
        per_batch.iter().map(|d| d.as_nanos()).sum::<u128>() / per_batch.len() as u128;
    let batches_per_sec = num_batches as f64 / total.as_secs_f64();
    let seeds_per_sec = batches_per_sec * batch_size as f64;

    eprintln!();
    eprintln!("Throughput:");
    eprintln!("  batch_size      = {}", batch_size);
    eprintln!("  warmup batches  = {}", warmup);
    eprintln!("  measure batches = {}", num_batches);
    eprintln!("  total           = {:.3}s", total.as_secs_f64());
    eprintln!("  batches/sec     = {:.1}", batches_per_sec);
    eprintln!("  seeds/sec       = {:.0}", seeds_per_sec);
    eprintln!();
    eprintln!("Per-batch latency:");
    eprintln!("  mean = {:>8.1} µs", mean_ns as f64 / 1_000.0);
    eprintln!(
        "  p50  = {:>8.1} µs",
        p_nanos(&per_batch, 0.50) as f64 / 1_000.0
    );
    eprintln!(
        "  p90  = {:>8.1} µs",
        p_nanos(&per_batch, 0.90) as f64 / 1_000.0
    );
    eprintln!(
        "  p99  = {:>8.1} µs",
        p_nanos(&per_batch, 0.99) as f64 / 1_000.0
    );
    eprintln!(
        "  max  = {:>8.1} µs",
        per_batch.last().unwrap().as_nanos() as f64 / 1_000.0
    );

    // TEST_PLAN R3 target: >500 batches/sec (>500K seeds/sec) at bs=1024.
    // Keep as a soft warn, not an assert — g4dn NVMe is slower than an i4i
    // (the target instance class), so failing here is a data point, not a bug.
    if batch_size == 1024 && batches_per_sec < 500.0 {
        eprintln!(
            "WARN: {batches_per_sec:.1} batches/sec < 500 target (likely NVMe-bound on g4dn)"
        );
    }
}
