//! Control-plane robustness: the server accepts connections forever and
//! must not be knocked over by a peer that sends a malformed length prefix,
//! a short-read, or a close mid-handshake. Guards that `MAX_MSG_SIZE` /
//! `TCP_TIMEOUT` enforcement keeps the listener alive and subsequent valid
//! clients can still complete the QP exchange.

#![cfg(all(target_os = "linux", feature = "rdma"))]

use aether_stream::feature_table::FeatureTable;
use aether_stream::rdma::context::RdmaContext;
use aether_stream::rdma::control::{connect_with_qp, serve_control_plane_with_qp, RdmaAdvertisement};
use aether_stream::rdma::ffi::*;
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::thread;
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

fn pick_free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = l.local_addr().unwrap().port();
    drop(l);
    port
}

/// Wait for the server listener to accept connections.
fn wait_for_bind(addr: &str, deadline: Duration) {
    let start = Instant::now();
    loop {
        if TcpStream::connect_timeout(&addr.parse().unwrap(), Duration::from_millis(100)).is_ok() {
            return;
        }
        if start.elapsed() > deadline {
            panic!("server never bound {addr}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Spin up a control-plane server on a fresh port. Returns the bind address
/// and a leaked RdmaContext (server needs to outlive the listener thread).
fn start_server() -> (String, RdmaAdvertisement) {
    let server_ctx = Box::leak(Box::new(
        RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("server open"),
    ));
    let table = Box::leak(Box::new(
        FeatureTable::new(16, 4, vec![]).expect("table alloc"),
    ));
    let mr = Box::leak(Box::new(
        server_ctx
            .reg_mr(
                table.base_addr() as *mut u8,
                table.total_size(),
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
            )
            .expect("reg_mr"),
    ));
    let adv = RdmaAdvertisement {
        base_addr: table.base_addr(),
        rkey: mr.rkey(),
        schema: table.schema(),
    };
    let port = pick_free_port();
    let bind_addr = format!("127.0.0.1:{port}");
    let bind_for_server = bind_addr.clone();
    let adv_for_server = adv.clone();
    let ctx_ptr: &'static RdmaContext = server_ctx;
    thread::spawn(move || {
        let _ = serve_control_plane_with_qp(&bind_for_server, &adv_for_server, ctx_ptr);
    });
    wait_for_bind(&bind_addr, Duration::from_secs(2));
    (bind_addr, adv)
}

/// Client: connect, skip sending anything, close immediately. Server must
/// not panic and must keep accepting.
#[test]
fn server_survives_client_immediate_close() {
    skip_if_no_rdma!();
    let (addr, adv) = start_server();

    // Bad peer: connect + close (no read, no write).
    for _ in 0..5 {
        let s = TcpStream::connect(&addr).expect("connect");
        drop(s);
    }

    // Well-formed client should still succeed.
    let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("client open");
    let (got_adv, _qp) = connect_with_qp(&addr, &client_ctx).expect("good client");
    assert_eq!(got_adv.base_addr, adv.base_addr);
    assert_eq!(got_adv.rkey, adv.rkey);
}

/// Client: send an over-sized length prefix. `recv_msg` must reject it
/// (server never allocates 4 GiB) and continue serving subsequent peers.
#[test]
fn server_rejects_oversized_length_prefix() {
    skip_if_no_rdma!();
    let (addr, adv) = start_server();

    for _ in 0..3 {
        let mut s = TcpStream::connect(&addr).expect("connect");
        s.set_write_timeout(Some(Duration::from_secs(2))).unwrap();
        // Let the server send its ServerHello first (we just drain it).
        let mut drain = [0u8; 4096];
        let _ = std::io::Read::read(&mut s, &mut drain);
        // Now send a length prefix of 1 GiB — well above the 64 KiB cap.
        let bad_len: u32 = 1024 * 1024 * 1024;
        let _ = s.write_all(&bad_len.to_le_bytes());
        let _ = s.write_all(&[0u8; 32]); // a few bytes of "body"
        drop(s);
    }

    let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("client open");
    let (got_adv, _qp) = connect_with_qp(&addr, &client_ctx).expect("good client after bad");
    assert_eq!(got_adv.base_addr, adv.base_addr);
}

/// Client: send a length prefix then never the body. Server's read_exact
/// must time out via TCP_TIMEOUT, log, and loop to the next peer.
#[test]
fn server_survives_truncated_body() {
    skip_if_no_rdma!();
    let (addr, adv) = start_server();

    let start_bad = Instant::now();
    for _ in 0..2 {
        let mut s = TcpStream::connect(&addr).expect("connect");
        let mut drain = [0u8; 4096];
        let _ = std::io::Read::read(&mut s, &mut drain);
        let len: u32 = 128;
        let _ = s.write_all(&len.to_le_bytes());
        // Send only 10 of 128 bytes then close.
        let _ = s.write_all(&[0xAA; 10]);
        drop(s);
    }
    // The server's TCP_TIMEOUT is 10s; we don't wait for it in this test
    // since the server's accept loop is per-connection. We just move on and
    // attempt a good handshake — it should succeed even while the bad
    // connections are still timing out in the server's handler for them.
    let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("client open");
    let (got_adv, _qp) = connect_with_qp(&addr, &client_ctx).expect("good client");
    assert_eq!(got_adv.base_addr, adv.base_addr);
    let elapsed = start_bad.elapsed();
    // Accept loop should not be held hostage by bad peers beyond a few
    // seconds total; if this exceeds TCP_TIMEOUT + buffer (12s) the accept
    // loop isn't resilient.
    assert!(
        elapsed < Duration::from_secs(12),
        "handshake blocked {elapsed:?} — accept loop held up by bad peers"
    );
}

/// Client: send valid-length JSON containing bytes that don't parse as a
/// ClientHello. Server must log, close, and keep serving.
#[test]
fn server_rejects_malformed_client_hello_json() {
    skip_if_no_rdma!();
    let (addr, adv) = start_server();

    for _ in 0..3 {
        let mut s = TcpStream::connect(&addr).expect("connect");
        let mut drain = [0u8; 4096];
        let _ = std::io::Read::read(&mut s, &mut drain);
        let body = b"{this is not valid JSON"; // 24 bytes
        let len: u32 = body.len() as u32;
        let _ = s.write_all(&len.to_le_bytes());
        let _ = s.write_all(body);
        drop(s);
    }

    let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("client open");
    let (got_adv, _qp) = connect_with_qp(&addr, &client_ctx).expect("good client");
    assert_eq!(got_adv.base_addr, adv.base_addr);
}

/// Happy path: 10 sequential good clients all complete a QP exchange.
/// Guards that the server's `active_qps` Vec growth doesn't stall or skip
/// hellos.
#[test]
fn server_handles_many_sequential_clients() {
    skip_if_no_rdma!();
    let (addr, adv) = start_server();

    for i in 0..10 {
        let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("client open");
        let (got_adv, _qp) = connect_with_qp(&addr, &client_ctx)
            .unwrap_or_else(|e| panic!("client {i}: {e}"));
        assert_eq!(got_adv.base_addr, adv.base_addr);
        assert_eq!(got_adv.rkey, adv.rkey);
    }
}
