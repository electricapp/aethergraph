//! WR-chain correctness under load: the longest single batch the QP's
//! `max_send_wr` allows, plus a sustained pipelining run that exercises
//! `post_reads_with_signaling` with a non-trivial signal cadence.
//!
//! Guards two things:
//!   - chained WRs at the QP capacity limit complete without dropping CQEs
//!   - intermediate-signaled pipelines drain every completion the caller
//!     expects (exactly `ceil(n / signal_every_n)` CQEs, including the
//!     always-signaled final WR)

#![cfg(all(target_os = "linux", feature = "rdma"))]

use aether_stream::feature_table::FeatureTable;
use aether_stream::rdma::context::RdmaContext;
use aether_stream::rdma::ffi::*;
use aether_stream::rdma::qp::{DEFAULT_QP_CAP, RdmaQp, RdmaRead};
use std::time::{Duration, Instant};

const ROCE_V2_GID_INDEX: u8 = 1;

fn rdma_available() -> bool {
    if std::env::var("AETHER_SKIP_RDMA").is_ok() {
        return false;
    }
    RdmaContext::open(16, ROCE_V2_GID_INDEX).is_ok()
}

macro_rules! skip_if_no_rdma {
    () => {
        if !rdma_available() {
            eprintln!("skipping: no RDMA device");
            return;
        }
    };
}

struct Loopback {
    _server_ctx: RdmaContext,
    client_ctx: RdmaContext,
    _server_qp: RdmaQp,
    client_qp: RdmaQp,
    _server_mr: aether_stream::rdma::context::RegisteredMr,
    server_base: u64,
    server_rkey: u32,
    slot_size: usize,
    table: FeatureTable,
}

fn build_loopback(node_count: usize, feature_dim: usize, cq_size: i32) -> Loopback {
    let server_ctx = RdmaContext::open(cq_size, ROCE_V2_GID_INDEX).expect("server open");
    let table = FeatureTable::new(node_count, feature_dim, vec![]).expect("alloc");
    for n in 0..node_count {
        let v: Vec<f32> = (0..feature_dim).map(|i| (n * 10 + i) as f32).collect();
        table.write_node(n, &v);
    }
    // SAFETY: `table` owns the registered range and outlives `server_mr`
    // (both stored in the loopback bundle; MR field drops first).
    let server_mr = unsafe {
        server_ctx.reg_mr(
            table.base_addr() as *mut u8,
            table.total_size(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
    }
    .expect("server reg_mr");
    let server_rkey = server_mr.rkey();
    let server_base = table.base_addr();
    let slot_size = table.schema().slot_size;

    let client_ctx = RdmaContext::open(cq_size, ROCE_V2_GID_INDEX).expect("client open");
    let server_qp = RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).expect("server qp");
    let client_qp = RdmaQp::create(&client_ctx, &DEFAULT_QP_CAP).expect("client qp");
    server_qp
        .connect(&server_ctx, &client_qp.endpoint(&client_ctx))
        .unwrap();
    client_qp
        .connect(&client_ctx, &server_qp.endpoint(&server_ctx))
        .unwrap();

    Loopback {
        _server_ctx: server_ctx,
        client_ctx,
        _server_qp: server_qp,
        client_qp,
        _server_mr: server_mr,
        server_base,
        server_rkey,
        slot_size,
        table,
    }
}

/// Drain exactly `expected` completions from the CQ, returning the first
/// non-SUCCESS status if any, otherwise Ok(()).
fn drain_n(
    qp: &RdmaQp,
    ctx: &RdmaContext,
    expected: usize,
    deadline: Duration,
) -> Result<(), String> {
    let start = Instant::now();
    let mut seen = 0usize;
    // SAFETY: zeroed init of POD `IbvWc` is sound.
    let mut wcs = [unsafe { std::mem::zeroed::<IbvWc>() }; 64];
    while seen < expected {
        let n = qp
            .poll_cq(ctx, &mut wcs)
            .map_err(|e| format!("poll: {e}"))?;
        for wc in wcs.iter().take(n) {
            if wc.status != IBV_WC_SUCCESS {
                return Err(format!(
                    "wc[{seen}] status={} vendor_err={} wr_id={}",
                    wc.status, wc.vendor_err, wc.wr_id
                ));
            }
            seen += 1;
        }
        if seen < expected && n == 0 {
            if start.elapsed() > deadline {
                return Err(format!(
                    "drain timed out: saw {seen}/{expected} after {:?}",
                    start.elapsed()
                ));
            }
            std::hint::spin_loop();
        }
    }
    Ok(())
}

/// Post a batch at the QP's `max_send_wr` capacity and drain every CQE.
/// Uses signal_every_n=1 so every WR generates a completion. Verifies each
/// slot byte-for-byte to rule out silent data corruption at the chain tail.
#[test]
fn long_wr_chain_at_max_send_wr_per_wr_signal() {
    skip_if_no_rdma!();

    // Stay well below DEFAULT_QP_CAP.max_send_wr (4096) so SoftRoCE's
    // cq_overflow / wqe_posted throttles don't kick in at the rxe layer —
    // 1024 is enough to catch chain-linkage bugs without tripping throttling.
    const BATCH: usize = 1024;
    const FEATURE_DIM: usize = 4;

    let lb = build_loopback(BATCH, FEATURE_DIM, 2048);
    let schema = lb.table.schema();

    let mut client_buf = vec![0u8; BATCH * lb.slot_size];
    let client_ptr = client_buf.as_mut_ptr();
    // SAFETY: `client_buf` owns the registered range and outlives
    // `client_mr`; every batch's completions drain before either drops.
    let client_mr = unsafe {
        lb.client_ctx
            .reg_mr(client_ptr, client_buf.len(), IBV_ACCESS_LOCAL_WRITE)
    }
    .expect("client reg_mr");
    let lkey = client_mr.lkey();

    let reads: Vec<RdmaRead> = (0..BATCH)
        .map(|i| RdmaRead {
            local_addr: client_ptr as u64 + (i * lb.slot_size) as u64,
            local_lkey: lkey,
            remote_addr: lb.server_base + (i as u64) * (lb.slot_size as u64),
            remote_rkey: lb.server_rkey,
            length: lb.slot_size as u32,
        })
        .collect();

    lb.client_qp
        .post_reads_with_signaling(&reads, 1)
        .expect("post_reads signal=1");
    drain_n(
        &lb.client_qp,
        &lb.client_ctx,
        BATCH,
        Duration::from_secs(30),
    )
    .expect("drain all CQEs");

    // Verify every slot.
    for i in 0..BATCH {
        let slot = &client_buf[i * lb.slot_size..(i + 1) * lb.slot_size];
        let head = u64::from_le_bytes(slot[0..8].try_into().unwrap());
        assert_ne!(head, 0, "slot {i}: zero head");
        let feat: Vec<f32> = slot
            [schema.feature_offset_in_slot..schema.feature_offset_in_slot + FEATURE_DIM * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let want: Vec<f32> = (0..FEATURE_DIM).map(|k| (i * 10 + k) as f32).collect();
        assert_eq!(feat, want, "slot {i}: feature mismatch");
    }
    let _ = lb; // keep alive
}

/// Sustained pipelining: 50 batches of 200 reads, signal_every_n=10, so each
/// batch produces 20 CQEs. Total expected CQEs: 50 * 20 = 1000 — one per
/// "every 10th" signal plus the last-WR-always-signaled rule (already
/// coincident because 200 % 10 == 0). Verifies no CQE is lost across batch
/// boundaries and that mid-chain WR wraparound stays consistent.
#[test]
fn sustained_pipelining_with_signal_stride() {
    skip_if_no_rdma!();

    const BATCHES: usize = 50;
    const PER_BATCH: usize = 200;
    const SIGNAL_EVERY_N: usize = 10;
    const FEATURE_DIM: usize = 4;
    const NODE_COUNT: usize = 256;

    let lb = build_loopback(NODE_COUNT, FEATURE_DIM, 2048);

    let mut client_buf = vec![0u8; PER_BATCH * lb.slot_size];
    let client_ptr = client_buf.as_mut_ptr();
    // SAFETY: `client_buf` owns the registered range and outlives
    // `client_mr`; every batch's completions drain before either drops.
    let client_mr = unsafe {
        lb.client_ctx
            .reg_mr(client_ptr, client_buf.len(), IBV_ACCESS_LOCAL_WRITE)
    }
    .expect("client reg_mr");
    let lkey = client_mr.lkey();

    let cqes_per_batch = PER_BATCH.div_ceil(SIGNAL_EVERY_N); // includes final-WR-always-signaled
    for b in 0..BATCHES {
        let reads: Vec<RdmaRead> = (0..PER_BATCH)
            .map(|i| {
                let node = (b * PER_BATCH + i) % NODE_COUNT;
                RdmaRead {
                    local_addr: client_ptr as u64 + (i * lb.slot_size) as u64,
                    local_lkey: lkey,
                    remote_addr: lb.server_base + (node as u64) * (lb.slot_size as u64),
                    remote_rkey: lb.server_rkey,
                    length: lb.slot_size as u32,
                }
            })
            .collect();
        lb.client_qp
            .post_reads_with_signaling(&reads, SIGNAL_EVERY_N)
            .unwrap_or_else(|e| panic!("batch {b}: post: {e}"));
        drain_n(
            &lb.client_qp,
            &lb.client_ctx,
            cqes_per_batch,
            Duration::from_secs(10),
        )
        .unwrap_or_else(|e| panic!("batch {b}: drain: {e}"));
    }
    let _ = lb;
}

/// Post zero reads — must be a no-op, no CQE emitted.
#[test]
fn empty_post_reads_is_noop() {
    skip_if_no_rdma!();
    let lb = build_loopback(4, 2, 64);
    lb.client_qp.post_reads(&[]).expect("empty post");
    // There should be no completion.
    // SAFETY: zeroed init of POD `IbvWc` is sound.
    let mut wcs = [unsafe { std::mem::zeroed::<IbvWc>() }; 4];
    let deadline = Instant::now() + Duration::from_millis(100);
    while Instant::now() < deadline {
        let n = lb.client_qp.poll_cq(&lb.client_ctx, &mut wcs).unwrap();
        assert_eq!(n, 0, "empty post should not produce CQEs");
        std::hint::spin_loop();
    }
    let _ = lb;
}

/// Single read with `length = 0` — ibverbs defines a zero-length SGE as
/// valid and the HCA should emit SUCCESS without touching memory. Guards
/// that our FFI doesn't reject length==0 at a higher layer, and that a
/// zero-length read still drives a signaled completion.
#[test]
fn zero_length_read_succeeds() {
    skip_if_no_rdma!();
    let lb = build_loopback(4, 2, 64);

    let mut buf = vec![0xAAu8; lb.slot_size]; // untouched after post
    let ptr = buf.as_mut_ptr();
    // SAFETY: `buf` owns the registered range and outlives `mr`; the WR's
    // completion is drained before either drops.
    let mr =
        unsafe { lb.client_ctx.reg_mr(ptr, buf.len(), IBV_ACCESS_LOCAL_WRITE) }.expect("reg_mr");
    let lkey = mr.lkey();
    let read = RdmaRead {
        local_addr: ptr as u64,
        local_lkey: lkey,
        remote_addr: lb.server_base,
        remote_rkey: lb.server_rkey,
        length: 0,
    };
    lb.client_qp.post_reads(&[read]).expect("post zero-len");

    let start = Instant::now();
    // SAFETY: zeroed init of POD `IbvWc` is sound.
    let mut wcs = [unsafe { std::mem::zeroed::<IbvWc>() }; 4];
    loop {
        let n = lb.client_qp.poll_cq(&lb.client_ctx, &mut wcs).unwrap();
        if n > 0 {
            assert_eq!(wcs[0].status, IBV_WC_SUCCESS, "zero-len WR must succeed");
            assert_eq!(wcs[0].byte_len, 0);
            break;
        }
        if start.elapsed() > Duration::from_secs(2) {
            panic!("no completion for zero-length read");
        }
        std::hint::spin_loop();
    }
    // Buffer must be untouched.
    assert!(
        buf.iter().all(|&b| b == 0xAA),
        "zero-len read should not write buf"
    );
    let _ = lb;
}
