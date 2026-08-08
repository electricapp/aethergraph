//! Cross-thread MR correctness — four alloc/reg/post thread-layout
//! variants. Guards the contract that `RegisteredMr` is freely `Send`:
//! the thread that calls `reg_mr` does not have to be the thread that
//! posts the WR. A subtly broken FFI layout in `ibv_post_send` would
//! surface here as `IBV_WC_LOC_PROT_ERR` on at least the cross-thread
//! scenarios.
//!
//! Run with:
//!   cargo test --features rdma --test mr_xthread_repro -- --nocapture \
//!     --test-threads=1

#![cfg(all(target_os = "linux", feature = "rdma"))]

use aether_stream::feature_table::FeatureTable;
use aether_stream::rdma::context::{RdmaContext, RegisteredMr};
use aether_stream::rdma::ffi::*;
use aether_stream::rdma::qp::{DEFAULT_QP_CAP, RdmaQp, RdmaRead};
use std::sync::Arc;
use std::thread;

const ROCE_V2_GID_INDEX: u8 = 1;

fn rdma_available() -> bool {
    if std::env::var("AETHER_SKIP_RDMA").is_ok() {
        return false;
    }
    RdmaContext::open(16, ROCE_V2_GID_INDEX).is_ok()
}

/// Build a minimal loopback (server QP + client QP on the same process) and
/// return everything the worker needs to issue one RDMA READ.
struct Loopback {
    _server_ctx: Arc<RdmaContext>,
    client_ctx: Arc<RdmaContext>,
    client_qp: Arc<RdmaQp>,
    _server_qp: RdmaQp,
    _server_mr: RegisteredMr,
    remote_addr: u64,
    remote_rkey: u32,
    slot_size: usize,
    table_base: u64,
    _table: FeatureTable,
}

fn build_loopback() -> Loopback {
    let server_ctx = Arc::new(RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("server open"));
    let table = FeatureTable::new(16, 8, vec![]).expect("feature table");
    let feats: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
    table.write_node(3, &feats);
    let schema = table.schema();
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
    let server_qp = RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).expect("server qp");

    let client_ctx = Arc::new(RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("client open"));
    let client_qp = Arc::new(RdmaQp::create(&client_ctx, &DEFAULT_QP_CAP).expect("client qp"));
    server_qp
        .connect(&server_ctx, &client_qp.endpoint(&client_ctx))
        .expect("server connect");
    client_qp
        .connect(&client_ctx, &server_qp.endpoint(&server_ctx))
        .expect("client connect");

    Loopback {
        _server_ctx: server_ctx,
        client_ctx,
        client_qp,
        _server_qp: server_qp,
        _server_mr: server_mr,
        remote_addr: table.base_addr() + 3 * (schema.slot_size as u64),
        remote_rkey: server_rkey,
        slot_size: schema.slot_size,
        table_base: table.base_addr(),
        _table: table,
    }
}

fn post_and_wait(qp: &RdmaQp, reads: &[RdmaRead], cq: *mut IbvCq) -> Result<(), String> {
    qp.post_reads(reads)
        .map_err(|e| format!("post_reads: {e}"))?;
    let mut wcs = [IbvWc::default(); 8];
    let wc = loop {
        // SAFETY: `cq` is the live CQ for this test harness.
        let n = unsafe { RdmaQp::poll_cq_on(cq, &mut wcs) }.map_err(|e| format!("poll_cq: {e}"))?;
        if n > 0 {
            break wcs[0];
        }
        std::hint::spin_loop();
    };
    if wc.status != IBV_WC_SUCCESS {
        return Err(format!(
            "wc status={} vendor_err={} wr_id={} opcode={}",
            wc.status, wc.vendor_err, wc.wr_id, wc.opcode
        ));
    }
    Ok(())
}

/// Scenario A: MR allocated + registered on the main thread, buffer + lkey
/// moved via struct into a worker thread, worker posts the RDMA READ.
#[test]
fn mr_on_main_post_on_worker() {
    if !rdma_available() {
        eprintln!("skipping: no RDMA device");
        return;
    }
    let lb = build_loopback();

    // Allocate & register MR on main thread.
    let mut buf = vec![0u8; lb.slot_size];
    let buf_ptr = buf.as_mut_ptr();
    let buf_len = buf.len();
    // SAFETY: `buf` and `mr` move into the worker together; the READ is
    // drained inside the closure before either drops.
    let mr = unsafe {
        lb.client_ctx
            .reg_mr(buf_ptr, buf_len, IBV_ACCESS_LOCAL_WRITE)
    }
    .expect("client reg_mr on main");
    let lkey = mr.lkey();
    eprintln!("[A] main-registered: buf_ptr={buf_ptr:p} lkey={lkey} len={buf_len}");

    // Hand everything to a worker thread via move.
    let client_ctx = Arc::clone(&lb.client_ctx);
    let client_qp = Arc::clone(&lb.client_qp);
    let remote_addr = lb.remote_addr;
    let remote_rkey = lb.remote_rkey;
    let slot_size = lb.slot_size;
    let table_base = lb.table_base;
    let local_addr = buf_ptr as u64;
    let result = thread::spawn(move || -> Result<Vec<u8>, String> {
        // Keep mr + buf alive on THIS thread.
        let _ = &mr;
        let _ = &buf;
        let read = RdmaRead {
            local_addr,
            local_lkey: lkey,
            remote_addr,
            remote_rkey,
            length: slot_size as u32,
        };
        eprintln!(
            "[A] worker posting: local_addr=0x{local_addr:x} lkey={lkey} remote=0x{remote_addr:x} rkey={remote_rkey} len={slot_size} table_base=0x{table_base:x}"
        );
        post_and_wait(&client_qp, &[read], client_ctx.cq).map_err(|e| format!("[A] {e}"))?;
        Ok(buf.clone())
    })
    .join()
    .expect("thread panic");

    match result {
        Ok(buf) => {
            eprintln!(
                "[A] SUCCESS, first 16 bytes = {:?}",
                &buf[..16.min(buf.len())]
            );
            // Sanity: head field should have advanced past 0.
            let head = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            assert_ne!(head, 0, "[A] read returned zeros");
        }
        Err(e) => {
            eprintln!("[A] FAILED: {e}");
            panic!("scenario A failed: {e}");
        }
    }
}

/// Scenario B: same loopback, but everything runs inside the worker thread —
/// the alloc + reg_mr + post all happen on the same worker. This is the
/// documented-safe pattern.
#[test]
fn mr_on_worker_post_on_worker() {
    if !rdma_available() {
        eprintln!("skipping: no RDMA device");
        return;
    }
    let lb = build_loopback();

    let client_ctx = Arc::clone(&lb.client_ctx);
    let client_qp = Arc::clone(&lb.client_qp);
    let remote_addr = lb.remote_addr;
    let remote_rkey = lb.remote_rkey;
    let slot_size = lb.slot_size;
    let result = thread::spawn(move || -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; slot_size];
        let buf_ptr = buf.as_mut_ptr();
        // SAFETY: `buf` outlives `mr` in this closure and the READ is
        // drained before either drops.
        let mr = unsafe { client_ctx.reg_mr(buf_ptr, buf.len(), IBV_ACCESS_LOCAL_WRITE) }
            .map_err(|e| format!("reg_mr: {e}"))?;
        let lkey = mr.lkey();
        let read = RdmaRead {
            local_addr: buf_ptr as u64,
            local_lkey: lkey,
            remote_addr,
            remote_rkey,
            length: slot_size as u32,
        };
        eprintln!("[B] worker-registered: buf_ptr={buf_ptr:p} lkey={lkey} len={slot_size}");
        post_and_wait(&client_qp, &[read], client_ctx.cq)?;
        let _ = mr;
        Ok(buf)
    })
    .join()
    .expect("thread panic");

    match result {
        Ok(buf) => {
            eprintln!(
                "[B] SUCCESS, first 16 bytes = {:?}",
                &buf[..16.min(buf.len())]
            );
            let head = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            assert_ne!(head, 0, "[B] read returned zeros");
        }
        Err(e) => {
            panic!("[B] scenario B failed (was supposed to succeed): {e}");
        }
    }
}

/// Scenario C: alloc on main, reg_mr on main, post on main — pure same-thread
/// control. If this fails, the loopback itself is broken (not a threading
/// issue). Useful as a sanity baseline.
#[test]
fn mr_on_main_post_on_main() {
    if !rdma_available() {
        eprintln!("skipping: no RDMA device");
        return;
    }
    let lb = build_loopback();
    let mut buf = vec![0u8; lb.slot_size];
    let buf_ptr = buf.as_mut_ptr();
    // SAFETY: `buf` owns the registered range and outlives `mr`; the READ
    // is drained by `post_and_wait` below before either drops.
    let mr = unsafe {
        lb.client_ctx
            .reg_mr(buf_ptr, buf.len(), IBV_ACCESS_LOCAL_WRITE)
    }
    .expect("reg_mr");
    let lkey = mr.lkey();
    let read = RdmaRead {
        local_addr: buf_ptr as u64,
        local_lkey: lkey,
        remote_addr: lb.remote_addr,
        remote_rkey: lb.remote_rkey,
        length: lb.slot_size as u32,
    };
    eprintln!(
        "[C] all-main: buf_ptr={buf_ptr:p} lkey={lkey} len={}",
        lb.slot_size
    );
    post_and_wait(&lb.client_qp, &[read], lb.client_ctx.cq).expect("[C] post_and_wait");
    let head = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    assert_ne!(head, 0);
}

/// Scenario D: alloc on main, reg_mr on main, then TOUCH buf from worker
/// (write a byte) before posting. Isolates the "thread-local arena page-fault"
/// hypothesis — if touching the buffer on another thread migrates pages out
/// from under the MR, this should fail even with same-thread reg+post.
#[test]
fn mr_on_main_touched_by_worker_then_post_on_main() {
    if !rdma_available() {
        eprintln!("skipping: no RDMA device");
        return;
    }
    let lb = build_loopback();
    let mut buf = vec![0u8; lb.slot_size];
    let buf_ptr = buf.as_mut_ptr();
    // SAFETY: `buf` owns the registered range and outlives `mr`; the READ
    // is drained by `post_and_wait` below before either drops.
    let mr = unsafe {
        lb.client_ctx
            .reg_mr(buf_ptr, buf.len(), IBV_ACCESS_LOCAL_WRITE)
    }
    .expect("reg_mr");
    let lkey = mr.lkey();

    // Touch from worker thread.
    let buf_addr = buf_ptr as usize;
    let buf_len = buf.len();
    thread::spawn(move || {
        let p = buf_addr as *mut u8;
        for i in 0..buf_len {
            // SAFETY: `buf_addr` points to `buf_len` bytes owned by the test.
            let slot = unsafe { p.add(i) };
            // SAFETY: `slot` is in-bounds; write is exclusive within this thread.
            unsafe { slot.write(0xA5) };
        }
    })
    .join()
    .expect("worker panic");

    let read = RdmaRead {
        local_addr: buf_ptr as u64,
        local_lkey: lkey,
        remote_addr: lb.remote_addr,
        remote_rkey: lb.remote_rkey,
        length: lb.slot_size as u32,
    };
    eprintln!("[D] touch-by-worker-then-post-on-main");
    post_and_wait(&lb.client_qp, &[read], lb.client_ctx.cq).expect("[D] post_and_wait");
    let head = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    assert_ne!(head, 0);
}
