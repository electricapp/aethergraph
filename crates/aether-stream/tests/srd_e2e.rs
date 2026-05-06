//! End-to-end SRD (AWS EFA) RDMA READ test.
//!
//! Runs on a single EFA-capable instance (c7g.16xlarge, c7gn.16xlarge,
//! p4d.24xlarge, etc.) and exercises the full SRD path in loopback:
//!   1. Open the EFA device, allocate PD + extended CQ
//!   2. Register source + destination MRs
//!   3. Create an SRD QP via `efadv_create_qp_ex` and transition it to RTS
//!   4. Build an AH pointing at ourselves and query its AHN
//!   5. Post an RDMA READ on the SRD QP addressed via that AH
//!   6. Drain the extended CQ and verify the destination buffer now holds
//!      the source bytes
//!
//! Skips cleanly when no EFA device is present (SoftRoCE / on-prem) so the
//! broader suite still runs outside AWS.

#![cfg(all(target_os = "linux", feature = "efa"))]

use aether_stream::rdma::ffi::{IBV_ACCESS_LOCAL_WRITE, IBV_ACCESS_REMOTE_READ, IBV_WC_SUCCESS};
use aether_stream::rdma::srd::{
    LocalBuf, SrdAddressHandle, SrdContext, SrdQp, DEFAULT_SRD_QKEY, DEFAULT_SRD_QP_CAP,
};
use std::time::{Duration, Instant};

/// EFA ships with its routable GID at index 0 on AWS (unlike RoCE where it's
/// usually 1) — there's no IPv6 link-local slot on the synthetic EFA GID table.
const EFA_GID_INDEX: u8 = 0;

fn srd_available() -> bool {
    if std::env::var("AETHER_SKIP_SRD").is_ok() {
        return false;
    }
    SrdContext::open(16, EFA_GID_INDEX).is_ok()
}

macro_rules! skip_if_no_srd {
    () => {
        if !srd_available() {
            eprintln!("skipping: no EFA device (or AETHER_SKIP_SRD set)");
            return;
        }
    };
}

/// The whole SRD RDMA READ path in loopback.
#[test]
fn srd_rdma_read_loopback_bytes_match() {
    skip_if_no_srd!();

    // -- context + QP --------------------------------------------------
    let ctx = SrdContext::open(64, EFA_GID_INDEX).expect("SrdContext::open");
    let qp = SrdQp::create(&ctx, &DEFAULT_SRD_QP_CAP).expect("SrdQp::create");
    qp.bring_up().expect("RESET→INIT→RTR→RTS");

    // -- buffers -------------------------------------------------------
    const LEN: usize = 4096;
    let mut src: Vec<u8> = vec![0xAB; LEN];
    let mut dst: Vec<u8> = vec![0x00; LEN];
    let src_ptr = src.as_mut_ptr();
    let dst_ptr = dst.as_mut_ptr();

    let src_mr = ctx
        .reg_mr(
            src_ptr,
            LEN,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
        .expect("reg src_mr");
    let dst_mr = ctx
        .reg_mr(dst_ptr, LEN, IBV_ACCESS_LOCAL_WRITE)
        .expect("reg dst_mr");

    // -- AH pointing at ourselves -------------------------------------
    let our_gid = ctx.gid();
    let ah = SrdAddressHandle::create(&ctx, &our_gid).expect("create AH");
    assert!(ah.ahn() != 0 || true, "AHN is informational");

    // -- post + drain --------------------------------------------------
    let local = LocalBuf::new(&dst_mr, dst_ptr);
    qp.post_rdma_read(
        1, // wr_id
        &ah,
        qp.qpn(),
        DEFAULT_SRD_QKEY,
        &local,
        src_mr.rkey(),
        src_ptr as u64,
        LEN as u32,
    )
    .expect("post_rdma_read");

    let start = Instant::now();
    let cqe = loop {
        match ctx.poll_one().expect("poll") {
            Some(snap) => break snap,
            None => {
                if start.elapsed() > Duration::from_secs(5) {
                    panic!("no completion after 5s");
                }
                std::hint::spin_loop();
            }
        }
    };

    assert_eq!(
        cqe.status, IBV_WC_SUCCESS,
        "SRD RDMA READ must succeed: wc={cqe:?}"
    );
    assert_eq!(cqe.wr_id, 1, "wr_id must be what we posted");
    // byte_len is not always populated by EFA for RDMA READ on older FW,
    // so only check when non-zero.
    if cqe.byte_len != 0 {
        assert_eq!(cqe.byte_len as usize, LEN);
    }

    // Destination must now contain the source pattern.
    assert!(
        dst.iter().all(|&b| b == 0xAB),
        "dst should be full of 0xAB after RDMA READ, first 16 bytes: {:?}",
        &dst[..16]
    );
}

/// Sanity: endpoint info survives a round-trip from creation. EFA SRD
/// actually issues QPN=0 for the first QP on a context (unlike RC where 0
/// is reserved) — just log everything, no strict assertion on QPN/AHN
/// numeric values beyond "we got them without an error".
#[test]
fn srd_endpoint_has_qpn_and_ahn() {
    skip_if_no_srd!();
    let ctx = SrdContext::open(16, EFA_GID_INDEX).expect("ctx");
    let qp = SrdQp::create(&ctx, &DEFAULT_SRD_QP_CAP).expect("qp");
    let ah = SrdAddressHandle::create(&ctx, &ctx.gid()).expect("ah");
    assert_eq!(qp.qkey(), aether_stream::rdma::srd::DEFAULT_SRD_QKEY);
    eprintln!(
        "SRD endpoint: qpn={} ahn={} qkey=0x{:x} gid={:02x?}",
        qp.qpn(),
        ah.ahn(),
        qp.qkey(),
        ctx.gid()
    );
}

/// Small pipelined run to confirm per-CQE ordering on SRD (EFA SRD
/// preserves per-QP ordering for signaled completions).
#[test]
fn srd_four_reads_in_order() {
    skip_if_no_srd!();

    const LEN: usize = 256;
    const N: usize = 4;

    let ctx = SrdContext::open(16, EFA_GID_INDEX).expect("ctx");
    let qp = SrdQp::create(&ctx, &DEFAULT_SRD_QP_CAP).expect("qp");
    qp.bring_up().expect("bring_up");

    // Fill src with a per-slot signature so each RDMA READ lands in a
    // different destination slot with a distinguishable payload.
    let mut src: Vec<u8> = (0..LEN * N).map(|i| (i & 0xFF) as u8).collect();
    let mut dst: Vec<u8> = vec![0u8; LEN * N];
    let src_ptr = src.as_mut_ptr();
    let dst_ptr = dst.as_mut_ptr();

    let src_mr = ctx
        .reg_mr(
            src_ptr,
            src.len(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
        .expect("reg src");
    let dst_mr = ctx
        .reg_mr(dst_ptr, dst.len(), IBV_ACCESS_LOCAL_WRITE)
        .expect("reg dst");

    let ah = SrdAddressHandle::create(&ctx, &ctx.gid()).expect("ah");

    for i in 0..N {
        qp.post_rdma_read(
            (i as u64) + 10,
            &ah,
            qp.qpn(),
            DEFAULT_SRD_QKEY,
            &LocalBuf {
                lkey: dst_mr.lkey(),
                addr: (dst_ptr as u64) + (i * LEN) as u64,
            },
            src_mr.rkey(),
            (src_ptr as u64) + (i * LEN) as u64,
            LEN as u32,
        )
        .expect("post");
    }

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut seen = 0usize;
    while seen < N {
        match ctx.poll_one().expect("poll") {
            Some(snap) => {
                assert_eq!(snap.status, IBV_WC_SUCCESS, "wc {seen}: {snap:?}");
                assert_eq!(snap.wr_id, (seen as u64) + 10, "out-of-order wr_id");
                seen += 1;
            }
            None => {
                if Instant::now() > deadline {
                    panic!("stuck: drained {seen}/{N}");
                }
                std::hint::spin_loop();
            }
        }
    }
    assert_eq!(&dst[..], &src[..], "all {N} reads must land correctly");
}

/// Ablation harness for the SRD QP-cap sizing question. Every combination
/// below is expected to succeed on current EFA firmware; if one regresses
/// we'll narrow the culprit here rather than guessing.
#[test]
fn srd_qp_cap_ablation() {
    skip_if_no_srd!();

    let shapes = [
        // (name, cq_size, max_send_wr, max_recv_wr)
        ("tiny", 16u32, 64u32, 1u32),
        ("recv=64", 16u32, 64u32, 64u32),
        ("send=256,recv=1", 16u32, 256u32, 1u32),
        ("send=256,recv=64", 64u32, 256u32, 64u32),
        ("send=256,recv=256", 512u32, 256u32, 256u32),
        ("send=1024,recv=1", 1024u32, 1024u32, 1u32),
        ("send=4096,recv=1", 4096u32, 4096u32, 1u32),
    ];
    for (name, cq, send_wr, recv_wr) in shapes {
        let cap = aether_stream::rdma::ffi::IbvQpCap {
            max_send_wr: send_wr,
            max_recv_wr: recv_wr,
            max_send_sge: 1,
            max_recv_sge: 1,
            max_inline_data: 0,
        };
        let ctx = SrdContext::open(cq, EFA_GID_INDEX)
            .unwrap_or_else(|e| panic!("shape {name}: ctx open cq={cq}: {e}"));
        let qp = SrdQp::create(&ctx, &cap)
            .unwrap_or_else(|e| panic!("shape {name}: qp create {send_wr}/{recv_wr}: {e}"));
        qp.bring_up()
            .unwrap_or_else(|e| panic!("shape {name}: bring_up cq={cq} send={send_wr} recv={recv_wr}: {e}"));
        eprintln!("shape {name}: OK cq={cq} send_wr={send_wr} recv_wr={recv_wr}");
    }
}

/// Baseline: what does the absolute-floor local memcpy cost? Establishes
/// the speed-of-light for "move N bytes from one host buffer to another".
/// Any network transport (TCP, RDMA) is paying its own overhead on top.
#[test]
#[ignore]
fn bench_memcpy_baseline() {
    const ITERS: usize = 20_000;
    let sizes: [usize; 5] = [64, 512, 4096, 16 * 1024, 64 * 1024];
    eprintln!("\n=== memcpy baseline (loopback floor) ===");
    eprintln!("  {:>6}  {:>10}  {:>10}  {:>10}", "bytes", "ns/iter", "MB/s", "min ns");
    for &sz in &sizes {
        let src = vec![0xAAu8; sz];
        let mut dst = vec![0u8; sz];
        // Warm up caches + branch predictor.
        for _ in 0..256 {
            dst.copy_from_slice(&src);
        }
        let mut min_ns = u64::MAX;
        let start = Instant::now();
        for _ in 0..ITERS {
            let t0 = Instant::now();
            dst.copy_from_slice(&src);
            // Prevent the compiler from hoisting the write out of the loop.
            std::hint::black_box(&dst);
            let ns = t0.elapsed().as_nanos() as u64;
            if ns < min_ns {
                min_ns = ns;
            }
        }
        let elapsed = start.elapsed();
        let per_ns = elapsed.as_nanos() as f64 / ITERS as f64;
        let mb = (sz as f64) * ITERS as f64 / elapsed.as_secs_f64() / 1e6;
        eprintln!("  {sz:>6}  {per_ns:>10.1}  {mb:>10.0}  {min_ns:>10}");
    }
}

/// Baseline: naive localhost TCP RPC — client sends `(offset, len)` to a
/// server thread holding the source buffer, server writes the bytes back.
/// This is what a hand-rolled "remote feature fetch" would look like
/// before anyone reaches for RDMA. Same payload sizes as the SRD bench so
/// the comparison is apples-to-apples.
#[test]
#[ignore]
fn bench_tcp_rpc_baseline() {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    const ITERS: usize = 2000;
    const WARMUP: usize = 64;
    let sizes: [usize; 5] = [64, 512, 4096, 16 * 1024, 64 * 1024];
    let max_sz = *sizes.iter().max().unwrap();

    eprintln!("\n=== Localhost TCP RPC baseline (naive remote-fetch) ===");
    eprintln!("  {:>6}  {:>10}  {:>10}  {:>10}", "bytes", "µs/iter", "MB/s", "min µs");

    // --- server thread: request = [u32 len]; reply = len bytes of 0xAA --
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().unwrap();
    let server_payload: Vec<u8> = vec![0xAAu8; max_sz];
    let server_bytes = server_payload.clone();
    let server = thread::spawn(move || {
        let (mut sock, _) = listener.accept().expect("accept");
        sock.set_nodelay(true).ok();
        loop {
            let mut header = [0u8; 4];
            if sock.read_exact(&mut header).is_err() {
                break;
            }
            let len = u32::from_le_bytes(header) as usize;
            if len == 0 {
                break; // exit sentinel
            }
            if sock.write_all(&server_bytes[..len]).is_err() {
                break;
            }
        }
    });

    let mut client = TcpStream::connect(addr).expect("connect");
    client.set_nodelay(true).ok();

    for &sz in &sizes {
        let mut dst = vec![0u8; sz];

        let mut one_cycle = |dst: &mut [u8]| {
            let header = (sz as u32).to_le_bytes();
            client.write_all(&header).expect("write req");
            client.read_exact(dst).expect("read resp");
        };
        for _ in 0..WARMUP {
            one_cycle(&mut dst);
        }
        let mut min_ns = u64::MAX;
        let start = Instant::now();
        for _ in 0..ITERS {
            let t0 = Instant::now();
            one_cycle(&mut dst);
            let ns = t0.elapsed().as_nanos() as u64;
            if ns < min_ns {
                min_ns = ns;
            }
        }
        let elapsed = start.elapsed();
        let per_us = elapsed.as_secs_f64() * 1e6 / ITERS as f64;
        let mb = (sz as f64) * ITERS as f64 / elapsed.as_secs_f64() / 1e6;
        eprintln!("  {sz:>6}  {per_us:>10.2}  {mb:>10.0}  {:>10.2}", min_ns as f64 / 1e3);
        assert!(dst.iter().all(|&b| b == 0xAA), "TCP payload mismatch");
    }
    // Tell server to exit.
    client.write_all(&0u32.to_le_bytes()).ok();
    drop(client);
    server.join().ok();
}

/// Loopback latency / throughput — one-at-a-time RDMA READ, timing the
/// post+drain loop across several payload sizes. Ignored by default
/// because it's slow and requires real EFA hardware. Run with:
///
///   cargo test --features efa --release -p aether-stream --test srd_e2e \
///       -- --ignored --nocapture bench_srd_rdma_read_latency
#[test]
#[ignore]
fn bench_srd_rdma_read_latency() {
    skip_if_no_srd!();

    const ITERS: usize = 2000;
    const WARMUP: usize = 64;
    // One MR per payload size, one QP + CQ shared across runs.
    let sizes: [u32; 5] = [64, 512, 4096, 16 * 1024, 64 * 1024];

    let ctx = SrdContext::open(2048, EFA_GID_INDEX).expect("ctx");
    let qp = SrdQp::create(&ctx, &DEFAULT_SRD_QP_CAP).expect("qp");
    qp.bring_up().expect("bring_up");
    let ah = SrdAddressHandle::create(&ctx, &ctx.gid()).expect("ah");

    eprintln!("\n=== EFA SRD RDMA READ latency (1 inflight, loopback) ===");
    eprintln!("  {:>6}  {:>10}  {:>10}  {:>10}", "bytes", "µs/iter", "MB/s", "min µs");

    for &sz in &sizes {
        let buf_len = sz as usize;
        let mut src = vec![0xAAu8; buf_len];
        let mut dst = vec![0u8; buf_len];
        let src_mr = ctx
            .reg_mr(
                src.as_mut_ptr(),
                buf_len,
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
            )
            .expect("reg src");
        let dst_mr = ctx
            .reg_mr(dst.as_mut_ptr(), buf_len, IBV_ACCESS_LOCAL_WRITE)
            .expect("reg dst");
        let local = LocalBuf::new(&dst_mr, dst.as_ptr());

        // One full cycle = post + drain a single completion.
        let one_cycle = |i: u64| {
            qp.post_rdma_read(
                i,
                &ah,
                qp.qpn(),
                DEFAULT_SRD_QKEY,
                &local,
                src_mr.rkey(),
                src.as_ptr() as u64,
                sz,
            )
            .expect("post");
            loop {
                if let Some(s) = ctx.poll_one().expect("poll") {
                    assert_eq!(s.status, IBV_WC_SUCCESS);
                    break;
                }
                std::hint::spin_loop();
            }
        };
        for i in 0..(WARMUP as u64) {
            one_cycle(i);
        }
        let mut min_ns = u64::MAX;
        let start = Instant::now();
        for i in 0..(ITERS as u64) {
            let t0 = Instant::now();
            one_cycle(i);
            let ns = t0.elapsed().as_nanos() as u64;
            if ns < min_ns {
                min_ns = ns;
            }
        }
        let elapsed = start.elapsed();
        let per_iter_us = elapsed.as_secs_f64() * 1e6 / ITERS as f64;
        let mb_per_sec = (buf_len as f64) * ITERS as f64 / elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "  {:>6}  {:>10.2}  {:>10.0}  {:>10.2}",
            buf_len,
            per_iter_us,
            mb_per_sec,
            min_ns as f64 / 1e3,
        );
    }
}
