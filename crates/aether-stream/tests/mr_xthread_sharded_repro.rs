//! Sharded pool with main-thread-preregistered MRs: main thread creates N
//! MRs in a Vec and moves them to N caller threads, which call
//! `pool.gather()` that ships the work to a shard worker thread that posts
//! the WR. Guards that the `ShardedQpPool` contract holds when MRs are
//! registered on one thread and used on another.

#![cfg(all(target_os = "linux", feature = "rdma"))]

use aether_stream::feature_table::FeatureTable;
use aether_stream::rdma::context::{RdmaContext, RegisteredMr};
use aether_stream::rdma::ffi::*;
use aether_stream::rdma::qp::{DEFAULT_QP_CAP, RdmaQp, RdmaRead};
use aether_stream::rdma::sharded::{ShardedConfig, ShardedQpPool};
use std::sync::Arc;
use std::thread;

const ROCE_V2_GID_INDEX: u8 = 1;

fn rdma_available() -> bool {
    if std::env::var("AETHER_SKIP_RDMA").is_ok() {
        return false;
    }
    RdmaContext::open(16, ROCE_V2_GID_INDEX).is_ok()
}

/// Packaged struct, the "move N MRs via a Vec into N worker threads" pattern.
struct CallerBundle {
    buf: Vec<u8>,
    mr: RegisteredMr,
    lkey: u32,
}

#[test]
fn main_preregisters_mrs_ships_to_caller_threads() {
    if !rdma_available() {
        eprintln!("skipping: no RDMA device");
        return;
    }

    const NODE_COUNT: usize = 256;
    const FEATURE_DIM: usize = 16;
    const NUM_SHARDS: usize = 4;
    const NUM_CALLER_THREADS: usize = 16;
    const ITERS_PER_THREAD: usize = 1000;
    const READS_PER_BATCH: usize = 8;

    // -------- server side --------
    let server_ctx = RdmaContext::open(256, ROCE_V2_GID_INDEX).expect("server open");
    let server_table = FeatureTable::new(NODE_COUNT, FEATURE_DIM, vec![]).expect("alloc");
    let mut expected = Vec::with_capacity(NODE_COUNT);
    for n in 0..NODE_COUNT {
        let v: Vec<f32> = (0..FEATURE_DIM).map(|i| (n * 100 + i) as f32).collect();
        server_table.write_node(n, &v);
        expected.push(v);
    }
    let schema = server_table.schema();
    let server_base = server_table.base_addr();
    let server_total = server_table.total_size();
    let server_mr = server_ctx
        .reg_mr(
            server_base as *mut u8,
            server_total,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
        .expect("server reg_mr");
    let server_rkey = server_mr.rkey();
    let server_qps: Vec<RdmaQp> = (0..NUM_SHARDS)
        .map(|_| RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).expect("server qp"))
        .collect();

    // -------- client side --------
    let client_ctx = Arc::new(RdmaContext::open(256, ROCE_V2_GID_INDEX).expect("client open"));
    let pool = Arc::new(
        ShardedQpPool::new(
            &client_ctx,
            ShardedConfig {
                num_shards: NUM_SHARDS,
                cq_size: 256,
                qp_cap: DEFAULT_QP_CAP,
                worker_cores: vec![],
            },
        )
        .expect("pool"),
    );
    let client_eps = pool.endpoints(&client_ctx);
    let server_eps: Vec<_> = server_qps.iter().map(|q| q.endpoint(&server_ctx)).collect();
    for (sqp, cep) in server_qps.iter().zip(&client_eps) {
        sqp.connect(&server_ctx, cep).unwrap();
    }
    pool.connect_all(&client_ctx, &server_eps).unwrap();

    // Pre-register N MRs in the main thread and keep them in a Vec,
    // one MR per caller thread. Move each via the thread closure.
    let mut bundles: Vec<CallerBundle> = Vec::with_capacity(NUM_CALLER_THREADS);
    for _ in 0..NUM_CALLER_THREADS {
        let mut buf = vec![0u8; READS_PER_BATCH * schema.slot_size];
        let buf_ptr = buf.as_mut_ptr();
        let mr = client_ctx
            .reg_mr(buf_ptr, buf.len(), IBV_ACCESS_LOCAL_WRITE)
            .expect("reg_mr on main");
        let lkey = mr.lkey();
        bundles.push(CallerBundle { buf, mr, lkey });
    }
    eprintln!("main-thread registered {} MRs", bundles.len());

    let expected = Arc::new(expected);
    let mut handles = Vec::with_capacity(NUM_CALLER_THREADS);
    for (tidx, bundle) in bundles.into_iter().enumerate() {
        let pool = Arc::clone(&pool);
        let expected = Arc::clone(&expected);
        let schema = schema.clone();
        handles.push(thread::spawn(move || -> Vec<String> {
            let CallerBundle { mut buf, mr, lkey } = bundle;
            let mut errors = Vec::new();
            let mut rng: u64 = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul((tidx as u64) + 1);
            for iter in 0..ITERS_PER_THREAD {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let nodes: Vec<usize> = (0..READS_PER_BATCH)
                    .map(|i| (rng.wrapping_add(i as u64) as usize) % NODE_COUNT)
                    .collect();
                let local_base = buf.as_mut_ptr() as u64;
                let reads: Vec<RdmaRead> = nodes
                    .iter()
                    .enumerate()
                    .map(|(i, &node)| RdmaRead {
                        local_addr: local_base + (i * schema.slot_size) as u64,
                        local_lkey: lkey,
                        remote_addr: server_base + (node as u64) * (schema.slot_size as u64),
                        remote_rkey: server_rkey,
                        length: schema.slot_size as u32,
                    })
                    .collect();
                if let Err(e) = pool.gather(reads) {
                    errors.push(format!("thread {tidx} iter {iter}: {e}"));
                    continue;
                }
                for (i, &node) in nodes.iter().enumerate() {
                    let slot = &buf[i * schema.slot_size..(i + 1) * schema.slot_size];
                    let head = u64::from_le_bytes(slot[0..8].try_into().unwrap());
                    if head == 0 {
                        errors.push(format!(
                            "thread {tidx} iter {iter} slot {i} node {node}: zero head"
                        ));
                        continue;
                    }
                    let feat_bytes = &slot[schema.feature_offset_in_slot
                        ..schema.feature_offset_in_slot + FEATURE_DIM * 4];
                    let got: Vec<f32> = feat_bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    if got != expected[node] {
                        errors.push(format!(
                            "thread {tidx} iter {iter} slot {i} node {node}: feature mismatch"
                        ));
                    }
                }
            }
            let _ = mr; // keep alive
            errors
        }));
    }

    let mut all_errors: Vec<String> = Vec::new();
    for h in handles {
        all_errors.extend(h.join().expect("thread panicked"));
    }
    if !all_errors.is_empty() {
        for e in &all_errors {
            eprintln!("ERR: {e}");
        }
    }
    assert!(
        all_errors.is_empty(),
        "{} errors observed with main-preregistered MRs",
        all_errors.len()
    );
}
