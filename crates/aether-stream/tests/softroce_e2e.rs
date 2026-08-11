//! End-to-end SoftRoCE integration tests.
//!
//! Exercises every layer of the RDMA stack:
//!   - `RdmaContext::open` — device discovery, PD + CQ allocation
//!   - `RdmaContext::reg_mr` — registering a FeatureTable
//!   - `RdmaQp::create` + `connect` — RESET → INIT → RTR → RTS transitions
//!   - `serve_control_plane_with_qp` / `connect_with_qp` — TCP advertisement + QP exchange
//!   - `RdmaQp::post_reads` — chained RDMA READ batch
//!   - `RdmaQp::poll_cq` — completion polling, byte-for-byte verification
//!   - Error path: dereg MR → post → error CQE drained → reregister → success
//!
//! Prerequisites: a loaded RDMA device (e.g. `rdma link add rxe0 type rxe netdev ens5`).
//! Without one, `RdmaContext::open` returns `NotFound` and the tests will
//! panic; skip the whole file via env var for environments without one.

#![cfg(all(target_os = "linux", feature = "rdma"))]

use aether_stream::feature_table::FeatureTable;
use aether_stream::rdma::context::RdmaContext;
use aether_stream::rdma::control::{
    RdmaAdvertisement, connect_with_qp, serve_control_plane_with_qp,
};
use aether_stream::rdma::event::CompletionChannel;
use aether_stream::rdma::ffi::*;
use aether_stream::rdma::qp::{DEFAULT_QP_ACCESS, DEFAULT_QP_CAP, RdmaQp, RdmaRead, RdmaWrite};
use aether_stream::rdma::srq::Srq;
use std::net::TcpListener;
use std::thread;
use std::time::{Duration, Instant};

/// GID index for the routable RoCEv2 IPv4-mapped GID on this test box.
/// Verified via `show_gids` / `ibv_devinfo -v` — index 0 is link-local IPv6,
/// index 1 is `::ffff:<ipv4>` (the routable one over Ethernet).
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
            eprintln!("skipping: no RDMA device (set AETHER_SKIP_RDMA to silence)");
            return;
        }
    };
}

/// Reserve an ephemeral TCP port. Race-safe enough for tests.
fn pick_free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let port = l.local_addr().unwrap().port();
    drop(l);
    port
}

/// Poll the CQ until at least one completion lands or deadline expires.
/// Uses `spin_loop` (PAUSE on x86, YIELD on aarch64) to match the production
/// `RdmaFeatureClient::post_and_wait` hot path — `thread::yield_now` is ~1µs
/// of pure overhead which would dominate any RDMA latency we're measuring.
fn drain_one_completion(
    qp: &RdmaQp,
    ctx: &RdmaContext,
    deadline: Duration,
) -> Result<IbvWc, String> {
    let start = Instant::now();
    let mut wcs = [IbvWc::default(); 32];
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

// ---------------------------------------------------------------------------
// T1.3 — RdmaContext::open + QP handshake loopback (two QPs on two contexts)
// ---------------------------------------------------------------------------

#[test]
fn t1_3_context_open_and_qp_handshake() {
    skip_if_no_rdma!();

    // Two independent contexts — simulates two processes on one box.
    let server_ctx = RdmaContext::open(256, ROCE_V2_GID_INDEX).expect("server open");
    let client_ctx = RdmaContext::open(256, ROCE_V2_GID_INDEX).expect("client open");

    let server_qp = RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).expect("server qp create");
    let client_qp = RdmaQp::create(&client_ctx, &DEFAULT_QP_CAP).expect("client qp create");

    let server_ep = server_qp.endpoint(&server_ctx);
    let client_ep = client_qp.endpoint(&client_ctx);
    assert_ne!(server_ep.qpn, 0, "server QPN must be nonzero");
    assert_ne!(client_ep.qpn, 0, "client QPN must be nonzero");
    assert_ne!(server_ep.qpn, client_ep.qpn, "QPNs must differ");

    // Cross-connect.
    server_qp
        .connect(&server_ctx, &client_ep)
        .expect("server connect");
    client_qp
        .connect(&client_ctx, &server_ep)
        .expect("client connect");
    // If we got here both QPs reached RTS without ibv_modify_qp errors.
}

// ---------------------------------------------------------------------------
// T1.4 — RDMA READ of a FeatureTable, byte-for-byte verified
// ---------------------------------------------------------------------------

#[test]
fn t1_4_rdma_read_feature_table_bytes_match() {
    skip_if_no_rdma!();

    const NODE_COUNT: usize = 64;
    const FEATURE_DIM: usize = 16;
    const NODES_READ: usize = 32;

    // --- Server side -------------------------------------------------------
    let server_ctx = RdmaContext::open(256, ROCE_V2_GID_INDEX).expect("server open");
    let server_table =
        FeatureTable::new(NODE_COUNT, FEATURE_DIM, vec![]).expect("feature table alloc");

    // Write known features to nodes 0..NODES_READ.
    let mut expected: Vec<Vec<f32>> = Vec::with_capacity(NODES_READ);
    for node in 0..NODES_READ {
        let feats: Vec<f32> = (0..FEATURE_DIM)
            .map(|i| (node * 1000 + i) as f32 + 0.5)
            .collect();
        server_table.write_node(node, &feats);
        expected.push(feats);
    }

    let schema = server_table.schema();
    let server_base = server_table.base_addr();
    let server_total = server_table.total_size();

    // Register the full table for remote read.
    // SAFETY: `server_table` owns the registered range and outlives
    // `server_mr` (RAII drop at end of scope, before the table).
    let server_mr = unsafe {
        server_ctx.reg_mr(
            server_base as *mut u8,
            server_total,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
    }
    .expect("server reg_mr");
    let server_rkey = server_mr.rkey();
    assert_ne!(server_rkey, 0, "rkey must be nonzero");

    // --- Client side -------------------------------------------------------
    let client_ctx = RdmaContext::open(256, ROCE_V2_GID_INDEX).expect("client open");
    let server_qp = RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).expect("server qp");
    let client_qp = RdmaQp::create(&client_ctx, &DEFAULT_QP_CAP).expect("client qp");

    let server_ep = server_qp.endpoint(&server_ctx);
    let client_ep = client_qp.endpoint(&client_ctx);
    server_qp.connect(&server_ctx, &client_ep).unwrap();
    client_qp.connect(&client_ctx, &server_ep).unwrap();

    // Client buffer: one slot per node we'll read, back-to-back.
    let client_len = NODES_READ * schema.slot_size;
    let mut client_buf = vec![0u8; client_len];
    // SAFETY: `client_buf` owns the registered range and outlives
    // `client_mr`; the READ completions below are drained before either drops.
    let client_mr =
        unsafe { client_ctx.reg_mr(client_buf.as_mut_ptr(), client_len, IBV_ACCESS_LOCAL_WRITE) }
            .expect("client reg_mr");
    let client_lkey = client_mr.lkey();

    // Build one RDMA READ per node.
    let client_base = client_buf.as_mut_ptr() as u64;
    let reads: Vec<RdmaRead> = (0..NODES_READ)
        .map(|i| RdmaRead {
            local_addr: client_base + (i * schema.slot_size) as u64,
            local_lkey: client_lkey,
            remote_addr: server_base + (i * schema.slot_size) as u64,
            remote_rkey: server_rkey,
            length: schema.slot_size as u32,
        })
        .collect();

    client_qp.post_reads(&reads).expect("post_reads");

    // Wait for the single signaled completion (last WR).
    let wc =
        drain_one_completion(&client_qp, &client_ctx, Duration::from_secs(5)).expect("CQ drain");
    assert_eq!(
        wc.status, IBV_WC_SUCCESS,
        "WC status must be SUCCESS, got {}",
        wc.status
    );
    assert_eq!(
        wc.wr_id,
        (NODES_READ - 1) as u64,
        "wr_id must be the last WR id"
    );

    // Byte-for-byte verification: parse each slot, assert head==tail==2 and
    // features match what the server wrote.
    for (i, expected_feats) in expected.iter().enumerate() {
        let slot = &client_buf[i * schema.slot_size..(i + 1) * schema.slot_size];

        let head = u64::from_le_bytes(slot[0..8].try_into().unwrap());
        let tail = u64::from_le_bytes(
            slot[schema.tail_offset_in_slot..schema.tail_offset_in_slot + 8]
                .try_into()
                .unwrap(),
        );
        assert_eq!(
            head, tail,
            "node {i}: torn read — head {head} != tail {tail}"
        );
        assert_eq!(head, 2, "node {i}: head must be 2 after one write");

        let feat_bytes =
            &slot[schema.feature_offset_in_slot..schema.feature_offset_in_slot + FEATURE_DIM * 4];
        let got: Vec<f32> = feat_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, *expected_feats, "node {i}: features mismatch");
    }

    // RegisteredMr Drop handles ibv_dereg_mr automatically.
}

// ---------------------------------------------------------------------------
// T1.5 — Control plane: TCP advertisement + QP exchange handshake
// ---------------------------------------------------------------------------

#[test]
fn t1_5_control_plane_qp_exchange() {
    skip_if_no_rdma!();

    const NODE_COUNT: usize = 16;
    const FEATURE_DIM: usize = 4;

    let server_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("server open");
    let table = FeatureTable::new(NODE_COUNT, FEATURE_DIM, vec![]).unwrap();
    // SAFETY: `table` owns the registered range and outlives `server_mr`
    // (dropped explicitly before the table at end of test).
    let server_mr = unsafe {
        server_ctx.reg_mr(
            table.base_addr() as *mut u8,
            table.total_size(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
    }
    .expect("reg_mr");
    let adv = RdmaAdvertisement {
        base_addr: table.base_addr(),
        rkey: server_mr.rkey(),
        schema: table.schema(),
    };

    let port = pick_free_port();
    let bind_addr = format!("127.0.0.1:{port}");
    let bind_for_server = bind_addr.clone();
    let adv_for_server = adv.clone();

    // Server thread serves handshakes then the test ends; the thread is
    // detached and the OS reaps it when the test process exits. Pass the
    // context as usize so the captured value is Send.
    let server_ctx_addr = &server_ctx as *const RdmaContext as usize;
    let _server_thread = thread::spawn(move || {
        // SAFETY: server_ctx outlives this thread — the test's main flow
        // completes the handshake before server_ctx drops.
        let ctx = unsafe { &*(server_ctx_addr as *const RdmaContext) };
        let _ = serve_control_plane_with_qp(&bind_for_server, &adv_for_server, ctx);
    });

    // Give the listener a moment to bind.
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if std::net::TcpStream::connect_timeout(
            &bind_addr.parse().unwrap(),
            Duration::from_millis(100),
        )
        .is_ok()
        {
            break;
        }
        if Instant::now() > deadline {
            panic!("server never bound {bind_addr}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("client open");
    let (got_adv, _client_qp) = connect_with_qp(&bind_addr, &client_ctx).expect("connect_with_qp");

    assert_eq!(got_adv.base_addr, adv.base_addr, "advertised base_addr");
    assert_eq!(got_adv.rkey, adv.rkey, "advertised rkey");
    assert_eq!(got_adv.schema.node_count, NODE_COUNT);
    assert_eq!(got_adv.schema.feature_dim, FEATURE_DIM);
    assert_eq!(got_adv.schema.slot_size, adv.schema.slot_size);

    drop(server_mr); // explicit dereg before context goes out of scope
    // Server thread is left running (listener loop). It dies with the process.
}

// ---------------------------------------------------------------------------
// T1.6 — CQ error recovery: invalid MR → error WC → drained → reg+succeed
// ---------------------------------------------------------------------------

#[test]
fn t1_6_cq_error_recovery() {
    skip_if_no_rdma!();

    const NODE_COUNT: usize = 8;
    const FEATURE_DIM: usize = 4;

    let server_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("server open");
    let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("client open");
    let table = FeatureTable::new(NODE_COUNT, FEATURE_DIM, vec![]).unwrap();
    let feats: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0];
    table.write_node(0, &feats);

    let schema = table.schema();
    // SAFETY: `table` owns the registered range and outlives this MR — the
    // test drops the MR deliberately (to provoke error CQEs) while the
    // table stays alive.
    let server_mr_initial = unsafe {
        server_ctx.reg_mr(
            table.base_addr() as *mut u8,
            table.total_size(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
    }
    .expect("initial reg_mr");
    let initial_rkey = server_mr_initial.rkey();

    let server_qp = RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).unwrap();
    let client_qp = RdmaQp::create(&client_ctx, &DEFAULT_QP_CAP).unwrap();
    server_qp
        .connect(&server_ctx, &client_qp.endpoint(&client_ctx))
        .unwrap();
    client_qp
        .connect(&client_ctx, &server_qp.endpoint(&server_ctx))
        .unwrap();

    let mut client_buf = vec![0u8; schema.slot_size];
    // SAFETY: `client_buf` owns the registered range and outlives
    // `client_mr` (explicit drop at end of test); all READs targeting it
    // are drained before then.
    let client_mr = unsafe {
        client_ctx.reg_mr(
            client_buf.as_mut_ptr(),
            client_buf.len(),
            IBV_ACCESS_LOCAL_WRITE,
        )
    }
    .expect("client reg_mr");
    let client_lkey = client_mr.lkey();

    // Step 1: deregister the server MR — any subsequent READ must fail.
    drop(server_mr_initial);

    // Step 2: post a READ using the now-stale rkey.
    let bad_read = [RdmaRead {
        local_addr: client_buf.as_mut_ptr() as u64,
        local_lkey: client_lkey,
        remote_addr: table.base_addr(),
        remote_rkey: initial_rkey,
        length: schema.slot_size as u32,
    }];
    client_qp.post_reads(&bad_read).expect("post_reads (stale)");

    // Step 3: CQ must drain with a non-SUCCESS status (typically REM_ACCESS_ERR).
    let wc = drain_one_completion(&client_qp, &client_ctx, Duration::from_secs(5))
        .expect("CQ must drain even on failure");
    assert_ne!(
        wc.status, IBV_WC_SUCCESS,
        "expected error WC after remote MR dereg, got SUCCESS"
    );

    // After one failed WR the RC QP enters the error state; fresh READs will
    // not progress until the QP is rebuilt. Validate that the failure was
    // observed and drained (no stale CQEs linger) by polling a few more times.
    let mut wcs = [IbvWc::default(); 32];
    for _ in 0..5 {
        let n = client_qp.poll_cq(&client_ctx, &mut wcs).unwrap();
        assert_eq!(n, 0, "no stale CQEs should linger after drain");
    }

    // Step 4: re-register and confirm a fresh QP pair + MR lets READs succeed.
    // SAFETY: `table` owns the registered range and outlives
    // `server_mr_fresh` (dropped explicitly before the table).
    let server_mr_fresh = unsafe {
        server_ctx.reg_mr(
            table.base_addr() as *mut u8,
            table.total_size(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
    }
    .expect("re-reg_mr");
    let new_rkey = server_mr_fresh.rkey();

    let server_qp2 = RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).unwrap();
    let client_qp2 = RdmaQp::create(&client_ctx, &DEFAULT_QP_CAP).unwrap();
    server_qp2
        .connect(&server_ctx, &client_qp2.endpoint(&client_ctx))
        .unwrap();
    client_qp2
        .connect(&client_ctx, &server_qp2.endpoint(&server_ctx))
        .unwrap();

    let good_read = [RdmaRead {
        local_addr: client_buf.as_mut_ptr() as u64,
        local_lkey: client_lkey,
        remote_addr: table.base_addr(),
        remote_rkey: new_rkey,
        length: schema.slot_size as u32,
    }];
    client_qp2
        .post_reads(&good_read)
        .expect("post_reads (fresh)");
    let wc_ok = drain_one_completion(&client_qp2, &client_ctx, Duration::from_secs(5))
        .expect("fresh CQ drain");
    assert_eq!(
        wc_ok.status, IBV_WC_SUCCESS,
        "fresh READ must succeed after re-register"
    );

    // Verify the read produced the right features.
    let head = u64::from_le_bytes(client_buf[0..8].try_into().unwrap());
    let tail = u64::from_le_bytes(
        client_buf[schema.tail_offset_in_slot..schema.tail_offset_in_slot + 8]
            .try_into()
            .unwrap(),
    );
    assert_eq!(head, tail);
    assert_eq!(head, 2);
    let got: Vec<f32> = client_buf
        [schema.feature_offset_in_slot..schema.feature_offset_in_slot + FEATURE_DIM * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got, feats);

    drop(server_mr_fresh);
    drop(client_mr);
    // RegisteredMr Drop handles ibv_dereg_mr automatically.
}

// ---------------------------------------------------------------------------
// SoftRoCE RDMA READ micro-bench — gated #[ignore].
//   cargo test --release -p aether-stream --features rdma --test softroce_e2e \
//       -- --ignored --nocapture bench_rdma_read_latency
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn bench_rdma_read_latency() {
    skip_if_no_rdma!();

    const NODE_COUNT: usize = 4096;
    const FEATURE_DIM: usize = 768;
    const ITERS: usize = 1000;

    let server_ctx = RdmaContext::open(1024, ROCE_V2_GID_INDEX).expect("server open");
    let server_table = FeatureTable::new(NODE_COUNT, FEATURE_DIM, vec![]).expect("alloc table");
    for n in 0..NODE_COUNT {
        let v: Vec<f32> = (0..FEATURE_DIM).map(|i| (n + i) as f32).collect();
        server_table.write_node(n, &v);
    }
    let schema = server_table.schema();
    let server_base = server_table.base_addr();
    let server_total = server_table.total_size();
    // SAFETY: `server_table` owns the registered range and outlives
    // `server_mr` (dropped explicitly at end of bench, before the table).
    let server_mr = unsafe {
        server_ctx.reg_mr(
            server_base as *mut u8,
            server_total,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
    }
    .expect("server reg_mr");
    let server_rkey = server_mr.rkey();

    let client_ctx = RdmaContext::open(1024, ROCE_V2_GID_INDEX).expect("client open");
    let server_qp = RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).unwrap();
    let client_qp = RdmaQp::create(&client_ctx, &DEFAULT_QP_CAP).unwrap();
    server_qp
        .connect(&server_ctx, &client_qp.endpoint(&client_ctx))
        .unwrap();
    client_qp
        .connect(&client_ctx, &server_qp.endpoint(&server_ctx))
        .unwrap();

    eprintln!(
        "\n=== SoftRoCE RDMA READ latency ({} nodes, dim={}, slot={}B) ===",
        NODE_COUNT, FEATURE_DIM, schema.slot_size
    );

    for &batch in &[1usize, 8, 64, 256, 1024] {
        if batch > NODE_COUNT {
            continue;
        }
        let client_len = batch * schema.slot_size;
        let mut client_buf = vec![0u8; client_len];
        // SAFETY: `client_buf` owns the registered range and outlives
        // `client_mr` (explicit drop at end of this batch iteration); all
        // READ completions are drained before that.
        let client_mr = unsafe {
            client_ctx.reg_mr(client_buf.as_mut_ptr(), client_len, IBV_ACCESS_LOCAL_WRITE)
        }
        .expect("client reg_mr");
        let client_lkey = client_mr.lkey();
        let client_base = client_buf.as_mut_ptr() as u64;

        let reads: Vec<RdmaRead> = (0..batch)
            .map(|i| RdmaRead {
                local_addr: client_base + (i * schema.slot_size) as u64,
                local_lkey: client_lkey,
                remote_addr: server_base + (i * schema.slot_size) as u64,
                remote_rkey: server_rkey,
                length: schema.slot_size as u32,
            })
            .collect();

        // Warm-up
        for _ in 0..16 {
            client_qp.post_reads(&reads).unwrap();
            drain_one_completion(&client_qp, &client_ctx, Duration::from_secs(2)).unwrap();
        }

        let start = Instant::now();
        for _ in 0..ITERS {
            client_qp.post_reads(&reads).unwrap();
            drain_one_completion(&client_qp, &client_ctx, Duration::from_secs(2)).unwrap();
        }
        let elapsed = start.elapsed();

        let per_iter_us = elapsed.as_secs_f64() * 1e6 / ITERS as f64;
        let total_bytes = (batch * schema.slot_size * ITERS) as f64;
        let mb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "  batch={:>4}  {:>6.1} µs/iter   {:>6.0} MB/s   ({:.2} µs/read)",
            batch,
            per_iter_us,
            mb_per_sec,
            per_iter_us / batch as f64,
        );

        drop(client_mr);
    }

    drop(server_mr);
}

// ---------------------------------------------------------------------------
// One-sided atomics: FETCH_ADD cursor reservation + CMP_SWP claim
// ---------------------------------------------------------------------------

#[test]
fn atomic_fetch_add_and_compare_swap_loopback() {
    skip_if_no_rdma!();

    let server_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("server open");
    match server_ctx.device_atomic_caps() {
        Ok(caps) if caps.atomic_cap != IBV_ATOMIC_NONE => {}
        Ok(_) => {
            eprintln!("skipping: device reports IBV_ATOMIC_NONE");
            return;
        }
        Err(e) => panic!("device_atomic_caps failed: {e}"),
    }

    // Server-side 8-byte cursor, 8-aligned by Box<u64>.
    let mut cursor = Box::new(0u64);
    // SAFETY: `cursor` outlives `server_mr` (dropped at end of scope).
    let server_mr = unsafe {
        server_ctx.reg_mr(
            (&mut *cursor as *mut u64) as *mut u8,
            8,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC,
        )
    }
    .expect("server reg_mr");
    let remote = (&*cursor as *const u64 as u64, server_mr.rkey());

    let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("client open");
    // Client landing slot for returned originals.
    let mut landing = Box::new(0u64);
    // SAFETY: `landing` outlives `client_mr`.
    let client_mr = unsafe {
        client_ctx.reg_mr(
            (&mut *landing as *mut u64) as *mut u8,
            8,
            IBV_ACCESS_LOCAL_WRITE,
        )
    }
    .expect("client reg_mr");
    let local = (&*landing as *const u64 as u64, client_mr.lkey());

    let server_qp = RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).expect("server qp");
    let client_qp = RdmaQp::create(&client_ctx, &DEFAULT_QP_CAP).expect("client qp");
    let atomic_access = DEFAULT_QP_ACCESS | IBV_ACCESS_REMOTE_ATOMIC as u32;
    server_qp
        .connect_with_access(&server_ctx, &client_qp.endpoint(&client_ctx), atomic_access)
        .unwrap();
    client_qp
        .connect(&client_ctx, &server_qp.endpoint(&server_ctx))
        .unwrap();

    // Two FETCH_ADDs: originals must come back 0 then 5, cursor ends at 12.
    for (i, (add, expect_original)) in [(5u64, 0u64), (7, 5)].iter().enumerate() {
        client_qp
            .post_fetch_add(100 + i as u64, local, remote, *add)
            .expect("post_fetch_add");
        let wc = drain_one_completion(&client_qp, &client_ctx, Duration::from_secs(2)).unwrap();
        assert_eq!(wc.status, 0, "FETCH_ADD completion status");
        assert_eq!(wc.opcode, IBV_WC_FETCH_ADD, "completion opcode");
        assert_eq!(*landing, *expect_original, "returned original (op {i})");
    }
    assert_eq!(*cursor, 12, "server cursor after two adds");

    // Winning CAS: 12 -> 13 (an even->odd seqlock-style claim).
    client_qp
        .post_compare_swap(200, local, remote, 12, 13)
        .expect("post_compare_swap");
    let wc = drain_one_completion(&client_qp, &client_ctx, Duration::from_secs(2)).unwrap();
    assert_eq!(wc.status, 0);
    assert_eq!(wc.opcode, IBV_WC_COMP_SWAP);
    assert_eq!(*landing, 12, "winning CAS returns the matched value");
    assert_eq!(*cursor, 13, "winning CAS installed the new value");

    // Losing CAS: expects 12 but the value is now 13 — must not modify.
    client_qp
        .post_compare_swap(201, local, remote, 12, 99)
        .expect("post_compare_swap (losing)");
    let wc = drain_one_completion(&client_qp, &client_ctx, Duration::from_secs(2)).unwrap();
    assert_eq!(wc.status, 0);
    assert_eq!(*landing, 13, "losing CAS returns the actual value");
    assert_eq!(*cursor, 13, "losing CAS must leave the target untouched");

    drop(client_mr);
    drop(server_mr);
}

// ---------------------------------------------------------------------------
// WRITE + WRITE_WITH_IMM: push-mode publication, imm off the recv CQ
// ---------------------------------------------------------------------------

#[test]
fn write_with_imm_push_publication_loopback() {
    skip_if_no_rdma!();

    const SLOT_BYTES: usize = 256;

    let server_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("server open");
    let mut server_buf = vec![0u8; SLOT_BYTES * 2];
    // SAFETY: `server_buf` outlives `server_mr`.
    let server_mr = unsafe {
        server_ctx.reg_mr(
            server_buf.as_mut_ptr(),
            server_buf.len(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
        )
    }
    .expect("server reg_mr");

    let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("client open");
    let payload: Vec<u8> = (0..SLOT_BYTES * 2).map(|i| (i % 251) as u8).collect();
    let mut client_buf = payload.clone();
    // SAFETY: `client_buf` outlives `client_mr`.
    let client_mr = unsafe {
        client_ctx.reg_mr(
            client_buf.as_mut_ptr(),
            client_buf.len(),
            IBV_ACCESS_LOCAL_WRITE,
        )
    }
    .expect("client reg_mr");

    // Receiver needs recv-queue depth for the imm sentinels.
    let mut recv_cap = DEFAULT_QP_CAP;
    recv_cap.max_recv_wr = 16;
    let server_qp = RdmaQp::create(&server_ctx, &recv_cap).expect("server qp");
    let client_qp = RdmaQp::create(&client_ctx, &DEFAULT_QP_CAP).expect("client qp");
    let write_access = DEFAULT_QP_ACCESS | IBV_ACCESS_REMOTE_WRITE as u32;
    server_qp
        .connect_with_access(&server_ctx, &client_qp.endpoint(&client_ctx), write_access)
        .unwrap();
    client_qp
        .connect(&client_ctx, &server_qp.endpoint(&server_ctx))
        .unwrap();

    server_qp.post_recv_sentinels(9000, 4).expect("post recvs");

    // Slot 0: silent WRITE. Slot 1: WRITE_WITH_IMM carrying the slot index.
    let writes = [
        RdmaWrite {
            local_addr: client_buf.as_ptr() as u64,
            local_lkey: client_mr.lkey(),
            remote_addr: server_buf.as_ptr() as u64,
            remote_rkey: server_mr.rkey(),
            length: SLOT_BYTES as u32,
            imm: None,
        },
        RdmaWrite {
            local_addr: client_buf.as_ptr() as u64 + SLOT_BYTES as u64,
            local_lkey: client_mr.lkey(),
            remote_addr: server_buf.as_ptr() as u64 + SLOT_BYTES as u64,
            remote_rkey: server_mr.rkey(),
            length: SLOT_BYTES as u32,
            imm: Some(1),
        },
    ];
    client_qp.post_writes(&writes).expect("post_writes");

    // Sender side: one signaled completion for the chain.
    let wc = drain_one_completion(&client_qp, &client_ctx, Duration::from_secs(2)).unwrap();
    assert_eq!(wc.status, 0, "sender completion status");

    // Receiver side: the imm arrives on the recv CQ with the slot index.
    let wc = drain_one_completion(&server_qp, &server_ctx, Duration::from_secs(2)).unwrap();
    assert_eq!(wc.status, 0, "receiver completion status");
    assert_eq!(wc.opcode, IBV_WC_RECV_RDMA_WITH_IMM, "receiver opcode");
    assert_eq!(wc.imm_data, 1, "immediate carries the published slot index");
    assert_eq!(wc.wr_id, 9000, "first sentinel consumed");

    // Both slots landed byte-for-byte, imm'd or not.
    assert_eq!(&server_buf[..], &payload[..], "payload after push writes");

    drop(client_mr);
    drop(server_mr);
}

// ---------------------------------------------------------------------------
// SRQ: one shared sentinel pool feeding two QPs
// ---------------------------------------------------------------------------

#[test]
fn srq_shared_sentinel_pool_across_qps() {
    skip_if_no_rdma!();

    const SLOT_BYTES: usize = 128;

    let server_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("server open");
    let mut server_buf = vec![0u8; SLOT_BYTES * 2];
    // SAFETY: `server_buf` outlives `server_mr`.
    let server_mr = unsafe {
        server_ctx.reg_mr(
            server_buf.as_mut_ptr(),
            server_buf.len(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
        )
    }
    .expect("server reg_mr");

    let srq = Srq::create(&server_ctx, 32, 1).expect("srq create");
    srq.post_recv_sentinels(5000, 8).expect("srq sentinels");

    let server_qp_a = RdmaQp::create_with_cqs_srq(
        &server_ctx,
        &DEFAULT_QP_CAP,
        server_ctx.cq,
        server_ctx.cq,
        &srq,
    )
    .expect("server qp a");
    let server_qp_b = RdmaQp::create_with_cqs_srq(
        &server_ctx,
        &DEFAULT_QP_CAP,
        server_ctx.cq,
        server_ctx.cq,
        &srq,
    )
    .expect("server qp b");

    let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("client open");
    let payload: Vec<u8> = (0..SLOT_BYTES * 2).map(|i| (i % 249) as u8).collect();
    let mut client_buf = payload.clone();
    // SAFETY: `client_buf` outlives `client_mr`.
    let client_mr = unsafe {
        client_ctx.reg_mr(
            client_buf.as_mut_ptr(),
            client_buf.len(),
            IBV_ACCESS_LOCAL_WRITE,
        )
    }
    .expect("client reg_mr");

    let client_qp_a = RdmaQp::create(&client_ctx, &DEFAULT_QP_CAP).expect("client qp a");
    let client_qp_b = RdmaQp::create(&client_ctx, &DEFAULT_QP_CAP).expect("client qp b");

    let write_access = DEFAULT_QP_ACCESS | IBV_ACCESS_REMOTE_WRITE as u32;
    server_qp_a
        .connect_with_access(
            &server_ctx,
            &client_qp_a.endpoint(&client_ctx),
            write_access,
        )
        .unwrap();
    client_qp_a
        .connect(&client_ctx, &server_qp_a.endpoint(&server_ctx))
        .unwrap();
    server_qp_b
        .connect_with_access(
            &server_ctx,
            &client_qp_b.endpoint(&client_ctx),
            write_access,
        )
        .unwrap();
    client_qp_b
        .connect(&client_ctx, &server_qp_b.endpoint(&server_ctx))
        .unwrap();

    // One imm'd write through each QP; both consume from the shared pool.
    for (slot, (client_qp, imm)) in [(&client_qp_a, 0u32), (&client_qp_b, 1u32)]
        .into_iter()
        .enumerate()
    {
        let write = RdmaWrite {
            local_addr: client_buf.as_ptr() as u64 + (slot * SLOT_BYTES) as u64,
            local_lkey: client_mr.lkey(),
            remote_addr: server_buf.as_ptr() as u64 + (slot * SLOT_BYTES) as u64,
            remote_rkey: server_mr.rkey(),
            length: SLOT_BYTES as u32,
            imm: Some(imm),
        };
        client_qp.post_writes(std::slice::from_ref(&write)).unwrap();
        let wc = drain_one_completion(client_qp, &client_ctx, Duration::from_secs(2)).unwrap();
        assert_eq!(wc.status, 0, "sender completion (slot {slot})");
    }

    // Server: two recv completions off the shared pool, imms 0 and 1,
    // wr_ids distinct sentinels from the 5000 range.
    let mut imms = Vec::new();
    let mut wr_ids = Vec::new();
    for _ in 0..2 {
        let wc = drain_one_completion(&server_qp_a, &server_ctx, Duration::from_secs(2)).unwrap();
        assert_eq!(wc.status, 0, "receiver completion");
        assert_eq!(wc.opcode, IBV_WC_RECV_RDMA_WITH_IMM);
        assert!(
            (5000..5008).contains(&wc.wr_id),
            "wr_id {} from the shared sentinel pool",
            wc.wr_id
        );
        imms.push(wc.imm_data);
        wr_ids.push(wc.wr_id);
    }
    imms.sort_unstable();
    assert_eq!(imms, vec![0, 1], "both immediates observed");
    assert_ne!(wr_ids[0], wr_ids[1], "distinct sentinels consumed");
    assert_eq!(&server_buf[..], &payload[..], "payloads landed");

    drop(server_qp_a);
    drop(server_qp_b);
    drop(client_qp_a);
    drop(client_qp_b);
    drop(srq);
    drop(client_mr);
    drop(server_mr);
}

// ---------------------------------------------------------------------------
// CQ event channel: block on the channel fd instead of spinning
// ---------------------------------------------------------------------------

#[test]
fn cq_event_channel_wakeup() {
    skip_if_no_rdma!();

    let server_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("server open");
    let mut server_buf: Vec<u8> = (0..4096u32).map(|i| (i % 253) as u8).collect();
    // SAFETY: `server_buf` outlives `server_mr`.
    let server_mr = unsafe {
        server_ctx.reg_mr(
            server_buf.as_mut_ptr(),
            server_buf.len(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
    }
    .expect("server reg_mr");

    let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("client open");
    let channel = CompletionChannel::create(&client_ctx).expect("channel");
    let ev_cq = channel.create_cq(&client_ctx, 64).expect("event cq");

    let mut client_buf = vec![0u8; 4096];
    // SAFETY: `client_buf` outlives `client_mr`.
    let client_mr = unsafe {
        client_ctx.reg_mr(
            client_buf.as_mut_ptr(),
            client_buf.len(),
            IBV_ACCESS_LOCAL_WRITE,
        )
    }
    .expect("client reg_mr");

    let server_qp = RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).expect("server qp");
    let client_qp =
        RdmaQp::create_with_cqs(&client_ctx, &DEFAULT_QP_CAP, ev_cq.as_ptr(), ev_cq.as_ptr())
            .expect("client qp");
    server_qp
        .connect(&server_ctx, &client_qp.endpoint(&client_ctx))
        .unwrap();
    client_qp
        .connect(&client_ctx, &server_qp.endpoint(&server_ctx))
        .unwrap();

    // Timeout path first: armed, nothing posted, must report false.
    channel.arm(&ev_cq, false).expect("arm");
    assert!(
        !channel
            .wait(Some(Duration::from_millis(100)))
            .expect("wait"),
        "wait must time out with no completion pending"
    );

    // Arm (still armed from above — re-arm is idempotent), post, wait.
    channel.arm(&ev_cq, false).expect("re-arm");
    let reads = [RdmaRead {
        local_addr: client_buf.as_ptr() as u64,
        local_lkey: client_mr.lkey(),
        remote_addr: server_buf.as_ptr() as u64,
        remote_rkey: server_mr.rkey(),
        length: 4096,
    }];
    client_qp.post_reads(&reads).expect("post_reads");

    assert!(
        channel.wait(Some(Duration::from_secs(2))).expect("wait"),
        "completion must raise a channel event"
    );

    // The completion is sitting in the event CQ.
    let mut wcs = [IbvWc::default(); 4];
    // SAFETY: `ev_cq` is alive; `wcs` is a valid out-array.
    let n = unsafe { RdmaQp::poll_cq_on(ev_cq.as_ptr(), &mut wcs) }.expect("poll");
    assert_eq!(n, 1, "exactly one completion");
    assert_eq!(wcs[0].status, 0);
    assert_eq!(&client_buf[..], &server_buf[..], "read payload verified");

    drop(client_qp);
    drop(server_qp);
    drop(ev_cq);
    drop(channel);
    drop(client_mr);
    drop(server_mr);
}

// ---------------------------------------------------------------------------
// ODP: capability probe + on-demand registration
// ---------------------------------------------------------------------------

#[test]
fn odp_caps_probe_and_on_demand_read() {
    skip_if_no_rdma!();

    let server_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("server open");
    let caps = server_ctx.odp_caps().expect("odp caps query");
    eprintln!(
        "ODP: general_caps={:#x} rc_odp_caps={:#x} (supported={} implicit={} rc_read={})",
        caps.general_caps,
        caps.rc_odp_caps,
        caps.supported(),
        caps.implicit(),
        caps.rc_read()
    );
    if !caps.supported() || !caps.rc_read() {
        eprintln!("skipping: device lacks ODP RC READ support");
        return;
    }

    // Register the source buffer on-demand — nothing pinned up front.
    let mut server_buf: Vec<u8> = (0..4096u32).map(|i| (i % 247) as u8).collect();
    // SAFETY: `server_buf` outlives `server_mr`.
    let server_mr = unsafe {
        server_ctx.reg_mr(
            server_buf.as_mut_ptr(),
            server_buf.len(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_ON_DEMAND,
        )
    }
    .expect("ODP reg_mr");

    let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("client open");
    let mut client_buf = vec![0u8; 4096];
    // SAFETY: `client_buf` outlives `client_mr`.
    let client_mr = unsafe {
        client_ctx.reg_mr(
            client_buf.as_mut_ptr(),
            client_buf.len(),
            IBV_ACCESS_LOCAL_WRITE,
        )
    }
    .expect("client reg_mr");

    let server_qp = RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).expect("server qp");
    let client_qp = RdmaQp::create(&client_ctx, &DEFAULT_QP_CAP).expect("client qp");
    server_qp
        .connect(&server_ctx, &client_qp.endpoint(&client_ctx))
        .unwrap();
    client_qp
        .connect(&client_ctx, &server_qp.endpoint(&server_ctx))
        .unwrap();

    // The READ faults the pages into the MR on first touch.
    let reads = [RdmaRead {
        local_addr: client_buf.as_ptr() as u64,
        local_lkey: client_mr.lkey(),
        remote_addr: server_buf.as_ptr() as u64,
        remote_rkey: server_mr.rkey(),
        length: 4096,
    }];
    client_qp.post_reads(&reads).expect("post_reads");
    let wc = drain_one_completion(&client_qp, &client_ctx, Duration::from_secs(5)).unwrap();
    assert_eq!(wc.status, 0, "ODP-backed READ completion");
    assert_eq!(&client_buf[..], &server_buf[..], "bytes through ODP MR");

    // Implicit ODP where offered: whole-address-space registration.
    if caps.implicit() {
        let implicit_mr = server_ctx
            .reg_mr_implicit_odp(IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ)
            .expect("implicit ODP reg_mr");
        assert_ne!(implicit_mr.rkey(), 0);
        drop(implicit_mr);
    }

    drop(client_mr);
    drop(server_mr);
}
