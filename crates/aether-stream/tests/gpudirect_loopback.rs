//! GPUDirect RDMA gather, byte-for-byte loopback.
//!
//! Server + client in the same process: the server publishes a FeatureTable
//! via the control plane, the client connects through `RdmaFeatureClient`
//! and gathers a batch of nodes directly into VRAM. The compacted output is
//! copied back to host memory and compared byte-for-byte against what the
//! server wrote.
//!
//! Skips when RDMA, CUDA, or nvidia-peermem aren't available.

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

#[test]
fn gpudirect_rdma_read_byte_for_byte_match() {
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

    const NODE_COUNT: usize = 256;
    const FEATURE_DIM: usize = 16;
    const BATCH: usize = 32;

    // --- Server: FeatureTable + reg_mr + control plane in a thread ------------
    let server_ctx = RdmaContext::open(256, ROCE_V2_GID_INDEX).expect("server open");
    let server_table = FeatureTable::new(NODE_COUNT, FEATURE_DIM, vec![]).expect("table alloc");

    let mut expected: Vec<Vec<f32>> = Vec::with_capacity(NODE_COUNT);
    for node in 0..NODE_COUNT {
        let feats: Vec<f32> = (0..FEATURE_DIM)
            .map(|i| (node * 10 + i) as f32 + 0.125)
            .collect();
        server_table.write_node(node, &feats);
        expected.push(feats);
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
    let _server_thread = thread::spawn(move || {
        // SAFETY: server_ctx outlives the thread — the test joins before
        // returning, by way of dropping the client/server scope below.
        let ctx = unsafe { &*(server_ctx_addr as *const RdmaContext) };
        let _ = serve_control_plane_with_qp(&bind_for_server, &adv_for_server, ctx);
    });
    assert!(
        wait_for_listen(&bind_addr, Duration::from_secs(5)),
        "server failed to bind {bind_addr}"
    );

    // --- Client: GPUDirect RDMA feature client --------------------------------
    let mut client = RdmaFeatureClient::connect(&bind_addr, 0, BATCH, ROCE_V2_GID_INDEX)
        .expect("RdmaFeatureClient::connect");

    // Sample BATCH random-ish nodes (deterministic LCG so failures reproduce).
    let mut rng_state = 0x9E3779B97F4A7C15u64;
    let mut node_ids = Vec::with_capacity(BATCH);
    while node_ids.len() < BATCH {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let candidate = (rng_state >> 33) as u32 % NODE_COUNT as u32;
        if !node_ids.contains(&candidate) {
            node_ids.push(candidate);
        }
    }

    client.gather(&node_ids).expect("gather");

    // Copy the compacted output back to host and compare byte-for-byte.
    let validator = client.validator();
    let output = validator.output();
    let mut host_out = vec![0.0f32; BATCH * FEATURE_DIM];
    validator
        .stream()
        .memcpy_dtoh(output, &mut host_out)
        .expect("D2H output readback");

    for (i, &n) in node_ids.iter().enumerate() {
        let got = &host_out[i * FEATURE_DIM..(i + 1) * FEATURE_DIM];
        assert_eq!(
            got,
            expected[n as usize].as_slice(),
            "node {n} (batch index {i}): features mismatch"
        );
    }

    drop(client);
    drop(server_mr);
}
