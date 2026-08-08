//! Torn-read detection under writer contention.
//!
//! The server overwrites the FeatureTable in a tight loop while the client
//! gathers the same slots over RDMA. The seqlock kernel must reject any
//! torn slot and the gather retry loop must resolve it before returning,
//! so every successful output row should be internally consistent — no row
//! assembled from mixed-generation feature bytes.
//!
//! Invariant exploited:
//!     features[i] = base + i,   where `base` changes on every write.
//!
//! `features[i] - i` is therefore constant across `i` within a consistent
//! row, and varies across `i` in a torn row, which is what the assertion
//! checks per slot.

#![cfg(all(target_os = "linux", feature = "rdma", feature = "gpudirect"))]

use aether_stream::feature_table::FeatureTable;
use aether_stream::rdma::client::RdmaFeatureClient;
use aether_stream::rdma::context::RdmaContext;
use aether_stream::rdma::control::{RdmaAdvertisement, serve_control_plane_with_qp};
use aether_stream::rdma::ffi::{IBV_ACCESS_LOCAL_WRITE, IBV_ACCESS_REMOTE_READ};
use cudarc::driver::CudaContext;
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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

/// Decode `features[i]` into the row's base value (the generation tag).
/// A consistent row yields the same `base` for every `i`.
fn row_base_set(row: &[f32]) -> Vec<f32> {
    row.iter().enumerate().map(|(i, &v)| v - i as f32).collect()
}

#[test]
fn no_torn_features_under_writer_contention() {
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

    const NODE_COUNT: usize = 128;
    const FEATURE_DIM: usize = 16;
    const BATCH: usize = 16;
    // Bumped to make it statistically near-certain that at least one slot
    // exhausts the gather retry budget under writer contention. Without
    // that signal the test could pass trivially on a slow writer.
    const GATHER_ITERS: usize = 500;

    // Base values stay inside f32's exact-integer range (≤ 2^24) so the
    // `features[i] - i = base` invariant holds with no rounding. The 100×
    // gap between generations keeps successive writes observably different
    // without ever spilling past the exact-integer range.
    //
    // --- Server side ---------------------------------------------------------
    let server_ctx = RdmaContext::open(256, ROCE_V2_GID_INDEX).expect("server open");
    let server_table =
        Arc::new(FeatureTable::new(NODE_COUNT, FEATURE_DIM, vec![]).expect("table alloc"));
    // Prime each slot with generation 0 so the very first gather doesn't race
    // the writer at slot init.
    for node in 0..NODE_COUNT {
        let base = ((node as u64 + 1) * 1000) as f32;
        let feats: Vec<f32> = (0..FEATURE_DIM).map(|i| base + i as f32).collect();
        server_table.write_node(node, &feats);
    }

    // SAFETY: `server_table` is Arc-held past `server_mr`'s explicit drop at
    // end of test, so the registered range stays valid for the MR's lifetime.
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

    // Control-plane server thread.
    let bind_for_server = bind_addr.clone();
    let adv_for_server = adv.clone();
    let server_ctx_addr = &server_ctx as *const RdmaContext as usize;
    let _control_thread = thread::spawn(move || {
        // SAFETY: server_ctx outlives this thread (test main flow joins below).
        let ctx = unsafe { &*(server_ctx_addr as *const RdmaContext) };
        let _ = serve_control_plane_with_qp(&bind_for_server, &adv_for_server, ctx);
    });
    assert!(
        wait_for_listen(&bind_addr, Duration::from_secs(5)),
        "server failed to bind {bind_addr}"
    );

    // Writer thread: hammer every node in a round-robin, incrementing the
    // generation tag so the seqlock head/tail flips between values.
    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_writer = Arc::clone(&stop);
    let table_for_writer = Arc::clone(&server_table);
    let writer = thread::spawn(move || {
        let mut generation: u64 = 1;
        while !stop_for_writer.load(Ordering::Relaxed) {
            for node in 0..NODE_COUNT {
                let base = ((node as u64 + 1) * 1000 + generation * 100) as f32;
                // Build the row directly into a stack buffer to keep the
                // write loop tight.
                let mut feats = [0.0f32; FEATURE_DIM];
                for i in 0..FEATURE_DIM {
                    feats[i] = base + i as f32;
                }
                table_for_writer.write_node(node, &feats);
            }
            generation = generation.wrapping_add(1);
            // Breathe between passes: a writer that saturates the table
            // full-time can keep the double-snapshot reader from ever seeing
            // two matching reads, so no gather would succeed. Alternating
            // busy/quiet windows exercises both outcomes.
            std::thread::sleep(Duration::from_micros(200));
        }
    });

    // --- Client side ---------------------------------------------------------
    let mut client = RdmaFeatureClient::connect(&bind_addr, 0, BATCH, ROCE_V2_GID_INDEX)
        .expect("client connect");

    let mut rng_state = 0xDEAD_BEEF_CAFE_F00Du64;
    let mut violations = 0usize;
    let mut total_rows = 0usize;
    // Rows the validator left untouched after gather's retry budget ran
    // out: direct evidence that real torn-read contention occurred (and
    // that the kernel correctly refused to surface stale bytes).
    let mut retry_budget_exhausted = 0usize;
    let mut host_out = vec![0.0f32; BATCH * FEATURE_DIM];

    for _ in 0..GATHER_ITERS {
        let mut node_ids = Vec::with_capacity(BATCH);
        while node_ids.len() < BATCH {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let candidate = (rng_state >> 33) as u32 % NODE_COUNT as u32;
            if !node_ids.contains(&candidate) {
                node_ids.push(candidate);
            }
        }

        match client.gather(&node_ids) {
            Ok(()) => {}
            Err(e) if e.to_string().contains("did not converge") => {
                // Retry budget exhausted under contention: the client
                // correctly refused to surface unvalidated rows. Count it
                // and move on — this is a legitimate outcome, not a failure.
                retry_budget_exhausted += 1;
                continue;
            }
            Err(e) => panic!("gather failed: {e}"),
        }

        let validator = client.validator();
        validator
            .stream()
            .memcpy_dtoh(validator.output(), &mut host_out)
            .expect("D2H");

        for i in 0..BATCH {
            let row = &host_out[i * FEATURE_DIM..(i + 1) * FEATURE_DIM];
            // Skip rows that the kernel never wrote (still all zeros after a
            // gather where every retry attempt found torn slots). Validator
            // leaves those untouched; that's a "no successful read" outcome,
            // not a torn feature, but it IS evidence of irreducible
            // contention which we count as a positive signal that the
            // test is actually exercising the torn-read path.
            if row.iter().all(|&v| v == 0.0) {
                retry_budget_exhausted += 1;
                continue;
            }
            total_rows += 1;

            let bases = row_base_set(row);
            // With bases in [1000, ~256_000] and FEATURE_DIM=16, every
            // intermediate computation stays inside f32's exact-integer range
            // (≤ 2^24), so a consistent row should have *exactly* equal bases
            // across `i`. Anything else is a torn read.
            let first = bases[0];
            let consistent = bases.iter().all(|&b| b == first);
            if !consistent {
                violations += 1;
                eprintln!(
                    "torn row at iter slot {i} (node {}): bases={bases:?}",
                    node_ids[i]
                );
            }
        }
    }

    stop.store(true, Ordering::Relaxed);
    let _ = writer.join();

    assert!(
        total_rows > 0,
        "no successful rows observed across {GATHER_ITERS} gathers — \
         test environment likely broken"
    );
    assert_eq!(
        violations, 0,
        "{violations} torn rows leaked through validator across {total_rows} successful rows"
    );
    // Pre-condition for the test to actually mean something: the seqlock
    // validator must have rejected at least one torn slot along the way.
    // Zero detections would mean the writer never interleaved with an RDMA
    // read and the retry path went unexercised — a trivial pass.
    let torn_detected = client.torn_slots_detected();
    assert!(
        torn_detected > 0,
        "no torn slots detected across {GATHER_ITERS} gathers — writer \
         never interleaved with gathers; increase FEATURE_DIM, NODE_COUNT \
         pressure, or iters"
    );
    eprintln!(
        "torn-read stress: {total_rows} successful rows, {torn_detected} \
         torn-slot detections, {retry_budget_exhausted} retry-budget \
         exhaustions"
    );

    drop(client);
    drop(server_mr);
}
