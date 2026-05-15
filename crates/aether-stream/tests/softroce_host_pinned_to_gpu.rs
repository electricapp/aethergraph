//! RDMA READ into a host buffer, host→VRAM memcpy, CUDA seqlock validation.
//!
//! Exercises the feature-gather pipeline on a box that has CUDA but no
//! peer-memory module — the NIC writes into a regular host buffer, the CPU
//! stages the bytes into VRAM, and the seqlock kernel verifies each slot.
//! Requires a loaded RDMA device and a CUDA-capable GPU; otherwise the test
//! skips cleanly.

#![cfg(all(target_os = "linux", feature = "rdma", feature = "gpudirect"))]

use aether_stream::feature_table::FeatureTable;
use aether_stream::gpu::kernel::SeqlockValidator;
use aether_stream::rdma::context::RdmaContext;
use aether_stream::rdma::ffi::*;
use aether_stream::rdma::qp::{DEFAULT_QP_CAP, RdmaQp, RdmaRead};
use cudarc::driver::{CudaContext, CudaSlice, DevicePtrMut};
use std::time::{Duration, Instant};

const ROCE_V2_GID_INDEX: u8 = 1;

fn rdma_available() -> bool {
    if std::env::var("AETHER_SKIP_RDMA").is_ok() {
        return false;
    }
    RdmaContext::open(16, ROCE_V2_GID_INDEX).is_ok()
}

fn cuda_available() -> bool {
    CudaContext::new(0).is_ok()
}

/// Drain the client CQ until a single completion shows up, mirroring
/// `softroce_e2e::drain_one_completion`. Duplicated here to keep the test
/// file self-contained.
fn drain_one_completion(
    qp: &RdmaQp,
    ctx: &RdmaContext,
    deadline: Duration,
) -> Result<IbvWc, String> {
    let start = Instant::now();
    let mut wcs = [unsafe { std::mem::zeroed::<IbvWc>() }; 32];
    loop {
        let n = qp
            .poll_cq(ctx, &mut wcs)
            .map_err(|e| format!("poll_cq: {e}"))?;
        if n > 0 {
            return Ok(wcs[0]);
        }
        if start.elapsed() > deadline {
            return Err(format!("no completion after {deadline:?}"));
        }
        std::hint::spin_loop();
    }
}

#[test]
fn rdma_into_host_then_memcpy_into_vram_and_validate() {
    if !rdma_available() {
        eprintln!("skipping: no RDMA device (set AETHER_SKIP_RDMA to silence)");
        return;
    }
    if !cuda_available() {
        eprintln!("skipping: no CUDA device");
        return;
    }

    const NODE_COUNT: usize = 16;
    const FEATURE_DIM: usize = 4;
    const NODES_READ: usize = 8;

    // --- Server side: FeatureTable + MR registration --------------------------
    let server_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("server open");
    let server_table =
        FeatureTable::new(NODE_COUNT, FEATURE_DIM, vec![]).expect("feature table alloc");

    let mut expected: Vec<Vec<f32>> = Vec::with_capacity(NODES_READ);
    for node in 0..NODES_READ {
        let feats: Vec<f32> = (0..FEATURE_DIM)
            .map(|i| (node * 1000 + i) as f32 + 0.25)
            .collect();
        server_table.write_node(node, &feats);
        expected.push(feats);
    }

    let schema = server_table.schema();
    let server_mr = server_ctx
        .reg_mr(
            server_table.base_addr() as *mut u8,
            server_table.total_size(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
        .expect("server reg_mr");
    let server_rkey = server_mr.rkey();

    // --- Client side: a pageable host buffer registered for local writes -----
    //
    // Use `Vec<u8>` rather than `cudaMallocHost` pinned memory: `memcpy_htod`
    // tolerates both; this test cares about correctness, not throughput.
    let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("client open");
    let server_qp = RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).expect("server qp");
    let client_qp = RdmaQp::create(&client_ctx, &DEFAULT_QP_CAP).expect("client qp");
    server_qp
        .connect(&server_ctx, &client_qp.endpoint(&client_ctx))
        .unwrap();
    client_qp
        .connect(&client_ctx, &server_qp.endpoint(&server_ctx))
        .unwrap();

    let host_len = NODES_READ * schema.slot_size;
    let mut host_buf = vec![0u8; host_len];
    let client_mr = client_ctx
        .reg_mr(host_buf.as_mut_ptr(), host_len, IBV_ACCESS_LOCAL_WRITE)
        .expect("client reg_mr");
    let client_lkey = client_mr.lkey();

    // --- RDMA READ NODES_READ slots into the host buffer ---------------------
    let server_base = server_table.base_addr();
    let host_base = host_buf.as_mut_ptr() as u64;
    let reads: Vec<RdmaRead> = (0..NODES_READ)
        .map(|i| RdmaRead {
            local_addr: host_base + (i * schema.slot_size) as u64,
            local_lkey: client_lkey,
            remote_addr: server_base + (i * schema.slot_size) as u64,
            remote_rkey: server_rkey,
            length: schema.slot_size as u32,
        })
        .collect();

    client_qp.post_reads(&reads).expect("post_reads");
    let wc =
        drain_one_completion(&client_qp, &client_ctx, Duration::from_secs(5)).expect("CQ drain");
    assert_eq!(wc.status, IBV_WC_SUCCESS, "RDMA READ must succeed");

    // --- Stage the host buffer in VRAM and run the validator -----------------
    let cuda_ctx = CudaContext::new(0).expect("CUDA init");
    let stream = cuda_ctx.default_stream();

    let mut staging: CudaSlice<u8> = stream.alloc_zeros(host_len).expect("alloc VRAM staging");
    stream
        .memcpy_htod(&host_buf, &mut staging)
        .expect("H2D memcpy");
    let staging_ptr = {
        let (p, _g) = staging.device_ptr_mut(&stream);
        p as u64
    };

    let mut validator = SeqlockValidator::new(&cuda_ctx, &stream, NODES_READ, FEATURE_DIM)
        .expect("SeqlockValidator nvrtc compile");

    // epoch_version=0 → accept any consistent read.
    let retries = validator
        .validate(staging_ptr, schema.slot_size, NODES_READ, 0)
        .expect("kernel launch");
    assert_eq!(
        retries, 0,
        "no torn slots expected — the server wrote each node exactly once"
    );

    // --- Read compacted output back and assert byte-for-byte match -----------
    let output = validator.output();
    let mut host_out = vec![0.0f32; NODES_READ * FEATURE_DIM];
    stream
        .memcpy_dtoh(output, &mut host_out)
        .expect("D2H output readback");

    for (i, expected_feats) in expected.iter().enumerate() {
        let got = &host_out[i * FEATURE_DIM..(i + 1) * FEATURE_DIM];
        assert_eq!(
            got,
            expected_feats.as_slice(),
            "node {i}: features mismatch"
        );
    }

    drop(client_mr);
    drop(server_mr);
}
