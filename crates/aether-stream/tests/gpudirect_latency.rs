//! GPUDirect gather latency profile.
//!
//! Times 10 000 gathers of 1 024 random node IDs and reports p50 / p90 /
//! p99 / max in microseconds. Marked `#[ignore]` so a regular `cargo test`
//! doesn't pay for it — run explicitly with `-- --ignored --nocapture`
//! when profiling.
//!
//! Targets for a healthy host: RDMA READ < 10 µs, CUDA validation < 5 µs,
//! total < 20 µs. The assertion fires only when p50 exceeds 50 µs — that's
//! the "something's wrong" threshold (PCIe ACS off, cross-NUMA placement,
//! MTU too small). Tighter targets are signal for tuning, not pass/fail.

#![cfg(all(target_os = "linux", feature = "rdma", feature = "gpudirect"))]

use aether_stream::feature_table::FeatureTable;
use aether_stream::rdma::client::RdmaFeatureClient;
use aether_stream::rdma::context::RdmaContext;
use aether_stream::rdma::control::{RdmaAdvertisement, serve_control_plane_with_qp};
use aether_stream::rdma::ffi::{IBV_ACCESS_LOCAL_WRITE, IBV_ACCESS_REMOTE_READ};
use cudarc::driver::CudaContext;
use std::net::TcpListener;
use std::thread;
use std::time::{Duration, Instant};

const ROCE_V2_GID_INDEX: u8 = 1;

fn pick_free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let port = l.local_addr().unwrap().port();
    drop(l);
    port
}

fn wait_for_listen(addr: &str, deadline: Duration) -> bool {
    let target = addr.parse().expect("parse addr");
    let start = Instant::now();
    while start.elapsed() < deadline {
        if std::net::TcpStream::connect_timeout(&target, Duration::from_millis(100)).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

fn percentile(sorted_us: &[u64], pct: f64) -> u64 {
    // Linear interpolation between the two nearest ranks. `sorted_us` is
    // ascending µs and non-empty by precondition.
    let n = sorted_us.len();
    let rank = (pct / 100.0) * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted_us[lo]
    } else {
        let frac = rank - lo as f64;
        ((sorted_us[lo] as f64) * (1.0 - frac) + (sorted_us[hi] as f64) * frac) as u64
    }
}

#[test]
#[ignore]
fn gpudirect_latency_profile() {
    if std::env::var("AETHER_SKIP_RDMA").is_ok() {
        eprintln!("skipping: AETHER_SKIP_RDMA set");
        return;
    }
    if RdmaContext::open(16, ROCE_V2_GID_INDEX).is_err() {
        eprintln!("skipping: no RDMA device");
        return;
    }
    if CudaContext::new(0).is_err() {
        eprintln!("skipping: no CUDA device");
        return;
    }

    const NODE_COUNT: usize = 8192;
    const FEATURE_DIM: usize = 128;
    const BATCH: usize = 1024;
    const ITERS: usize = 10_000;
    const WARMUP: usize = 200;

    // --- Server side ---------------------------------------------------------
    let server_ctx = RdmaContext::open(2048, ROCE_V2_GID_INDEX).expect("server open");
    let server_table = FeatureTable::new(NODE_COUNT, FEATURE_DIM, vec![]).expect("table alloc");
    for node in 0..NODE_COUNT {
        let feats: Vec<f32> = (0..FEATURE_DIM).map(|i| (node * 7 + i) as f32).collect();
        server_table.write_node(node, &feats);
    }
    // SAFETY: `server_table` owns the registered range and outlives
    // `server_mr` (dropped explicitly at end of test, before the table).
    let server_mr = unsafe {
        server_ctx.reg_mr(
            server_table.base_addr() as *mut u8,
            server_table.total_size(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
    }
    .expect("server reg_mr");
    let adv = RdmaAdvertisement {
        base_addr: server_table.base_addr(),
        rkey: server_mr.rkey(),
        schema: server_table.schema(),
    };

    let port = pick_free_port();
    let bind_addr = format!("127.0.0.1:{port}");
    let bind_for_server = bind_addr.clone();
    let adv_for_server = adv.clone();
    let server_ctx_addr = &server_ctx as *const RdmaContext as usize;
    let _control_thread = thread::spawn(move || {
        // SAFETY: server_ctx outlives this thread.
        let ctx = unsafe { &*(server_ctx_addr as *const RdmaContext) };
        let _ = serve_control_plane_with_qp(&bind_for_server, &adv_for_server, ctx);
    });
    assert!(
        wait_for_listen(&bind_addr, Duration::from_secs(5)),
        "server failed to bind {bind_addr}"
    );

    // --- Client side ---------------------------------------------------------
    let mut client = RdmaFeatureClient::connect(&bind_addr, 0, BATCH, ROCE_V2_GID_INDEX)
        .expect("client connect");

    // Build a stable per-iter shuffled node-id batch so each gather sees a
    // mix that defeats prefetcher / cache patterns.
    let mut batches: Vec<Vec<u32>> = Vec::with_capacity(ITERS + WARMUP);
    let mut rng_state = 0x243F6A8885A308D3u64;
    for _ in 0..(ITERS + WARMUP) {
        let mut node_ids = Vec::with_capacity(BATCH);
        while node_ids.len() < BATCH {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let candidate = (rng_state >> 33) as u32 % NODE_COUNT as u32;
            if !node_ids.contains(&candidate) {
                node_ids.push(candidate);
            }
        }
        batches.push(node_ids);
    }

    // Warmup: lets the validator's CUDA module fully resident, MR cache primed.
    for nids in batches.iter().take(WARMUP) {
        client.gather(nids).expect("warmup gather");
    }

    let mut samples = Vec::with_capacity(ITERS);
    for nids in batches.iter().skip(WARMUP).take(ITERS) {
        let t0 = Instant::now();
        client.gather(nids).expect("gather");
        let elapsed = t0.elapsed();
        samples.push(elapsed.as_nanos() as u64 / 1_000); // µs
    }

    samples.sort_unstable();
    let p50 = percentile(&samples, 50.0);
    let p90 = percentile(&samples, 90.0);
    let p99 = percentile(&samples, 99.0);
    let max = *samples.last().unwrap();
    let mean = samples.iter().sum::<u64>() / samples.len() as u64;

    // Default threshold tolerates SoftRoCE (kernel-stack overhead pushes p50
    // into the hundreds-of-microseconds range). Real GPUDirect should clear
    // 50 µs comfortably; set `AETHER_TIGHT_LATENCY=1` to apply the tight
    // threshold when benchmarking actual GPUDirect hardware.
    let tight = std::env::var("AETHER_TIGHT_LATENCY").is_ok();
    let p50_ceiling: u64 = if tight { 50 } else { 2_000 };
    let target_note = if tight {
        "GPUDirect target"
    } else {
        "SoftRoCE-tolerant; export AETHER_TIGHT_LATENCY=1 for the tight 50 µs check"
    };

    eprintln!(
        "\n=== gather latency profile: batch={BATCH} feature_dim={FEATURE_DIM} iters={ITERS} ==="
    );
    eprintln!("  mean   {mean:>5} µs");
    eprintln!("  p50    {p50:>5} µs   ({target_note})");
    eprintln!("  p90    {p90:>5} µs");
    eprintln!("  p99    {p99:>5} µs");
    eprintln!("  max    {max:>5} µs");

    assert!(
        p50 <= p50_ceiling,
        "p50 latency {p50} µs > {p50_ceiling} µs threshold — \
         check PCIe ACS, NUMA, MTU, or substrate (SoftRoCE vs real GPUDirect)"
    );

    drop(client);
    drop(server_mr);
}
