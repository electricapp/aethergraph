//! Sharded multi-threaded RDMA gather — concurrent gather() correctness
//! under contention against real (SoftRoCE) RDMA hardware.

#![cfg(all(target_os = "linux", feature = "rdma"))]

use aether_stream::feature_table::FeatureTable;
use aether_stream::rdma::context::RdmaContext;
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

macro_rules! skip_if_no_rdma {
    () => {
        if !rdma_available() {
            eprintln!("skipping: no RDMA device");
            return;
        }
    };
}

#[test]
fn sharded_concurrent_gather_correctness() {
    skip_if_no_rdma!();

    const NODE_COUNT: usize = 256;
    const FEATURE_DIM: usize = 16;
    const NUM_SHARDS: usize = 4;
    const READS_PER_BATCH: usize = 8;
    const ITERS_PER_THREAD: usize = 200;
    const NUM_CALLER_THREADS: usize = 8;

    // ---- Server side ----------------------------------------------------
    let server_ctx = RdmaContext::open(1024, ROCE_V2_GID_INDEX).expect("server open");
    let server_table = FeatureTable::new(NODE_COUNT, FEATURE_DIM, vec![]).expect("alloc");
    let mut expected: Vec<Vec<f32>> = Vec::with_capacity(NODE_COUNT);
    for n in 0..NODE_COUNT {
        let v: Vec<f32> = (0..FEATURE_DIM).map(|i| (n * 100 + i) as f32).collect();
        server_table.write_node(n, &v);
        expected.push(v);
    }
    let schema = server_table.schema();
    let server_base = server_table.base_addr();
    let server_total = server_table.total_size();
    // SAFETY: `server_table` owns the registered range and outlives
    // `server_mr` (both live to end of test; MR drops first in scope order).
    let server_mr = unsafe {
        server_ctx.reg_mr(
            server_base as *mut u8,
            server_total,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
    }
    .expect("server reg_mr");
    let server_rkey = server_mr.rkey();

    // Server: NUM_SHARDS standalone QPs (one per client shard).
    let server_qps: Vec<RdmaQp> = (0..NUM_SHARDS)
        .map(|_| RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).expect("server qp"))
        .collect();

    // ---- Client side ----------------------------------------------------
    let client_ctx = Arc::new(RdmaContext::open(1024, ROCE_V2_GID_INDEX).expect("client open"));
    let pool = Arc::new(
        ShardedQpPool::new(
            &client_ctx,
            ShardedConfig {
                num_shards: NUM_SHARDS,
                cq_size: 1024,
                qp_cap: DEFAULT_QP_CAP,
                worker_cores: vec![],
                ..Default::default()
            },
        )
        .expect("pool"),
    );

    // Endpoint exchange: pair client shard i with server QP i.
    let client_eps = pool.endpoints(&client_ctx);
    let server_eps: Vec<_> = server_qps.iter().map(|q| q.endpoint(&server_ctx)).collect();
    for (sqp, cep) in server_qps.iter().zip(&client_eps) {
        sqp.connect(&server_ctx, cep).expect("server connect");
    }
    pool.connect_all(&client_ctx, &server_eps)
        .expect("client connect_all");

    struct ThreadState {
        buf: Vec<u8>,
        mr_lkey: u32,
        _mr: aether_stream::rdma::context::RegisteredMr,
    }
    // MRs can be pre-registered on any thread and moved into the caller
    // threads — covered by `mr_xthread_sharded_repro.rs`. This test keeps
    // the per-thread-alloc pattern since each caller reuses its own slab
    // for all iterations anyway.
    let expected = Arc::new(expected);
    let pool_for_threads = Arc::clone(&pool);
    let schema_for_threads = schema.clone();
    let mut handles = Vec::with_capacity(NUM_CALLER_THREADS);
    for tidx in 0..NUM_CALLER_THREADS {
        let pool = Arc::clone(&pool_for_threads);
        let expected = Arc::clone(&expected);
        let schema = schema_for_threads.clone();
        let client_ctx_for_thread = Arc::clone(&client_ctx);
        handles.push(thread::spawn(move || {
            let mut buf = vec![0u8; READS_PER_BATCH * schema.slot_size];
            let buf_ptr = buf.as_mut_ptr();
            let buf_len = buf.len();
            // SAFETY: `buf` moves into `ThreadState` alongside `mr` below, so
            // the registered range outlives the MR; gathers drain before drop.
            let mr = unsafe {
                client_ctx_for_thread.reg_mr(buf_ptr, buf_len, IBV_ACCESS_LOCAL_WRITE)
            }
            .expect("client mr");
            let mr_lkey = mr.lkey();
            let mut state = ThreadState { buf, mr_lkey, _mr: mr };
            let mut rng: u64 = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul((tidx as u64) + 1);
            let mut errors = Vec::new();
            for iter in 0..ITERS_PER_THREAD {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;

                let nodes: Vec<usize> = (0..READS_PER_BATCH)
                    .map(|i| (rng.wrapping_add(i as u64) as usize) % NODE_COUNT)
                    .collect();
                let local_base = state.buf.as_mut_ptr() as u64;
                let reads: Vec<RdmaRead> = nodes
                    .iter()
                    .enumerate()
                    .map(|(i, &node)| RdmaRead {
                        local_addr: local_base + (i * schema.slot_size) as u64,
                        local_lkey: state.mr_lkey,
                        remote_addr: server_base + (node as u64) * (schema.slot_size as u64),
                        remote_rkey: server_rkey,
                        length: schema.slot_size as u32,
                    })
                    .collect();

                if let Err(e) = pool.gather(reads) {
                    errors.push(format!("thread {tidx} iter {iter}: {e}"));
                    continue;
                }

                // Verify each slot byte-for-byte against expected.
                for (i, &node) in nodes.iter().enumerate() {
                    let slot = &state.buf[i * schema.slot_size..(i + 1) * schema.slot_size];
                    let head = u64::from_le_bytes(slot[0..8].try_into().unwrap());
                    let tail = u64::from_le_bytes(
                        slot[schema.tail_offset_in_slot..schema.tail_offset_in_slot + 8]
                            .try_into()
                            .unwrap(),
                    );
                    if head != tail || head == 0 {
                        errors.push(format!(
                            "thread {tidx} iter {iter} slot {i} node {node}: torn (h={head} t={tail})"
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
                            "thread {tidx} iter {iter} slot {i} node {node}: features mismatch"
                        ));
                    }
                }
            }
            errors
        }));
    }

    // Collect errors from all threads.
    let mut all_errors: Vec<String> = Vec::new();
    for h in handles {
        all_errors.extend(h.join().expect("thread panicked"));
    }

    assert!(
        all_errors.is_empty(),
        "{} errors observed across {} threads:\n  {}",
        all_errors.len(),
        NUM_CALLER_THREADS,
        all_errors
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

#[test]
fn sharded_4shard_sequential_gather() {
    // 4 shards but only the main thread submits — sequential gather across
    // round-robin shards. Tests that connecting 4 QP pairs and dispatching
    // to all 4 shards works without concurrent callers.
    skip_if_no_rdma!();

    const NUM_SHARDS: usize = 4;
    let server_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).unwrap();
    let table = FeatureTable::new(16, 8, vec![]).unwrap();
    let feats: Vec<f32> = (0..8).map(|i| i as f32 + 7.0).collect();
    table.write_node(5, &feats);
    let schema = table.schema();
    // SAFETY: `table` owns the registered range and outlives `server_mr`
    // (both live to end of test; MR drops first in scope order).
    let server_mr = unsafe {
        server_ctx.reg_mr(
            table.base_addr() as *mut u8,
            table.total_size(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
    }
    .unwrap();
    let server_rkey = server_mr.rkey();

    let server_qps: Vec<RdmaQp> = (0..NUM_SHARDS)
        .map(|_| RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).unwrap())
        .collect();

    let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).unwrap();
    let pool = ShardedQpPool::new(
        &client_ctx,
        ShardedConfig {
            num_shards: NUM_SHARDS,
            cq_size: 64,
            qp_cap: DEFAULT_QP_CAP,
            worker_cores: vec![],
            ..Default::default()
        },
    )
    .unwrap();

    let client_eps = pool.endpoints(&client_ctx);
    for (sqp, cep) in server_qps.iter().zip(&client_eps) {
        sqp.connect(&server_ctx, cep).unwrap();
    }
    let server_eps: Vec<_> = server_qps.iter().map(|q| q.endpoint(&server_ctx)).collect();
    pool.connect_all(&client_ctx, &server_eps).unwrap();

    let mut buf = vec![0u8; schema.slot_size];
    let buf_ptr = buf.as_mut_ptr();
    // SAFETY: `buf` owns the registered range and outlives `client_mr`
    // (both live to end of test); each gather drains before the next.
    let client_mr =
        unsafe { client_ctx.reg_mr(buf_ptr, buf.len(), IBV_ACCESS_LOCAL_WRITE) }.unwrap();
    let client_lkey = client_mr.lkey();

    // Submit a gather one at a time, exercising each shard via round-robin.
    for round in 0..(NUM_SHARDS * 2) {
        let read = RdmaRead {
            local_addr: buf_ptr as u64,
            local_lkey: client_lkey,
            remote_addr: table.base_addr() + 5 * (schema.slot_size as u64),
            remote_rkey: server_rkey,
            length: schema.slot_size as u32,
        };
        pool.gather(vec![read])
            .unwrap_or_else(|e| panic!("round {round}: {e}"));
        let head = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let tail = u64::from_le_bytes(
            buf[schema.tail_offset_in_slot..schema.tail_offset_in_slot + 8]
                .try_into()
                .unwrap(),
        );
        assert_eq!(head, tail, "round {round}: torn (h={head} t={tail})");
        assert_eq!(head, 2, "round {round}: head should be 2");
    }
}

#[test]
fn sharded_single_shard_single_gather() {
    // Minimal "does the basic path work at all" — 1 shard, 1 caller thread,
    // 1 iteration. If this fails, the 8-thread test will too.
    skip_if_no_rdma!();

    let server_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).unwrap();
    let table = FeatureTable::new(16, 8, vec![]).unwrap();
    let feats: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
    table.write_node(3, &feats);
    let schema = table.schema();
    // SAFETY: `table` owns the registered range and outlives `server_mr`
    // (both live to end of test; MR drops first in scope order).
    let server_mr = unsafe {
        server_ctx.reg_mr(
            table.base_addr() as *mut u8,
            table.total_size(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
    }
    .unwrap();
    let server_rkey = server_mr.rkey();
    let server_qp = RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).unwrap();

    let client_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).unwrap();
    let pool = ShardedQpPool::new(
        &client_ctx,
        ShardedConfig {
            num_shards: 1,
            cq_size: 64,
            qp_cap: DEFAULT_QP_CAP,
            worker_cores: vec![],
            ..Default::default()
        },
    )
    .unwrap();

    let client_eps = pool.endpoints(&client_ctx);
    server_qp.connect(&server_ctx, &client_eps[0]).unwrap();
    pool.connect_all(&client_ctx, &[server_qp.endpoint(&server_ctx)])
        .unwrap();

    let mut buf = vec![0u8; schema.slot_size];
    let buf_ptr = buf.as_mut_ptr();
    let buf_len = buf.len();
    // SAFETY: `buf` owns the registered range and outlives `client_mr`
    // (both live to end of test); the gather drains before either drops.
    let client_mr = unsafe { client_ctx.reg_mr(buf_ptr, buf_len, IBV_ACCESS_LOCAL_WRITE) }.unwrap();
    let client_lkey = client_mr.lkey();

    let read = RdmaRead {
        local_addr: buf_ptr as u64,
        local_lkey: client_lkey,
        remote_addr: table.base_addr() + 3 * (schema.slot_size as u64),
        remote_rkey: server_rkey,
        length: schema.slot_size as u32,
    };
    pool.gather(vec![read]).expect("gather");

    let head = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    let tail = u64::from_le_bytes(
        buf[schema.tail_offset_in_slot..schema.tail_offset_in_slot + 8]
            .try_into()
            .unwrap(),
    );
    assert_eq!(head, tail, "torn (h={head} t={tail})");
    assert_eq!(head, 2);
    let got: Vec<f32> = buf[schema.feature_offset_in_slot..schema.feature_offset_in_slot + 8 * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got, feats);
}

#[test]
fn sharded_two_threads_minimal() {
    // 2 caller threads, 1 shard, 1 gather each — minimal multi-thread repro.
    skip_if_no_rdma!();

    let server_ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).unwrap();
    let table = FeatureTable::new(16, 8, vec![]).unwrap();
    let feats: Vec<f32> = (0..8).map(|i| i as f32 + 7.0).collect();
    table.write_node(2, &feats);
    let schema = table.schema();
    // SAFETY: `table` owns the registered range and outlives `server_mr`
    // (both live to end of test; MR drops first in scope order).
    let server_mr = unsafe {
        server_ctx.reg_mr(
            table.base_addr() as *mut u8,
            table.total_size(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
    }
    .unwrap();
    let server_rkey = server_mr.rkey();
    let server_qp = RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).unwrap();

    let client_ctx = Arc::new(RdmaContext::open(64, ROCE_V2_GID_INDEX).unwrap());
    let pool = Arc::new(
        ShardedQpPool::new(
            &client_ctx,
            ShardedConfig {
                num_shards: 1,
                cq_size: 64,
                qp_cap: DEFAULT_QP_CAP,
                worker_cores: vec![],
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let cep = pool.endpoints(&client_ctx).remove(0);
    server_qp.connect(&server_ctx, &cep).unwrap();
    pool.connect_all(&client_ctx, &[server_qp.endpoint(&server_ctx)])
        .unwrap();

    let server_base = table.base_addr();
    let mut handles = Vec::new();
    for tidx in 0..2 {
        let pool = Arc::clone(&pool);
        let client_ctx = Arc::clone(&client_ctx);
        let schema = schema.clone();
        handles.push(std::thread::spawn(move || -> Result<u64, String> {
            let mut buf = vec![0u8; schema.slot_size];
            let buf_ptr = buf.as_mut_ptr();
            // SAFETY: `buf` owns the registered range and outlives `mr` in
            // this closure; each gather drains before the iteration ends.
            let mr = unsafe { client_ctx.reg_mr(buf_ptr, buf.len(), IBV_ACCESS_LOCAL_WRITE) }
                .map_err(|e| format!("t{tidx} reg_mr: {e}"))?;
            let lkey = mr.lkey();
            let read = RdmaRead {
                local_addr: buf_ptr as u64,
                local_lkey: lkey,
                remote_addr: server_base + 2 * (schema.slot_size as u64),
                remote_rkey: server_rkey,
                length: schema.slot_size as u32,
            };
            pool.gather(vec![read])
                .map_err(|e| format!("t{tidx} gather: {e}"))?;
            let head = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            let _ = mr; // keep alive
            Ok(head)
        }));
    }
    for (i, h) in handles.into_iter().enumerate() {
        let r = h.join().expect("panic");
        let head = r.unwrap_or_else(|e| panic!("thread {i}: {e}"));
        assert_eq!(head, 2, "thread {i}: head should be 2");
    }
}

#[test]
fn sharded_pool_drops_cleanly() {
    skip_if_no_rdma!();
    // Just create + drop. Verifies workers exit on Shutdown without leaking
    // file descriptors or panicking on context teardown.
    let ctx = RdmaContext::open(64, ROCE_V2_GID_INDEX).expect("open");
    let pool = ShardedQpPool::new(
        &ctx,
        ShardedConfig {
            num_shards: 2,
            cq_size: 64,
            qp_cap: DEFAULT_QP_CAP,
            worker_cores: vec![],
            ..Default::default()
        },
    )
    .expect("pool");
    assert_eq!(pool.num_shards(), 2);
    drop(pool);
    drop(ctx);
}

/// The event-driven completion path must produce identical results to the
/// busy-poll path.
///
/// `spin_before_block: Some(ZERO)` sends every wait through arm-and-block
/// immediately, so this exercises the completion channel rather than
/// hoping a gather happens to be slow enough to fall out of the spin.
#[test]
fn sharded_gather_blocks_on_completion_channel() {
    skip_if_no_rdma!();

    const NODE_COUNT: usize = 64;
    const FEATURE_DIM: usize = 8;
    const NUM_SHARDS: usize = 2;

    let server_ctx = RdmaContext::open(256, ROCE_V2_GID_INDEX).expect("server open");
    let server_table = FeatureTable::new(NODE_COUNT, FEATURE_DIM, vec![]).expect("alloc");
    let mut expected: Vec<Vec<f32>> = Vec::with_capacity(NODE_COUNT);
    for n in 0..NODE_COUNT {
        let v: Vec<f32> = (0..FEATURE_DIM).map(|i| (n * 10 + i) as f32).collect();
        server_table.write_node(n, &v);
        expected.push(v);
    }
    let schema = server_table.schema();
    let server_base = server_table.base_addr();
    let server_total = server_table.total_size();
    // SAFETY: `server_table` owns the registered range and outlives the MR.
    let server_mr = unsafe {
        server_ctx.reg_mr(
            server_base as *mut u8,
            server_total,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
        )
    }
    .expect("server reg_mr");
    let server_rkey = server_mr.rkey();

    let server_qps: Vec<RdmaQp> = (0..NUM_SHARDS)
        .map(|_| RdmaQp::create(&server_ctx, &DEFAULT_QP_CAP).expect("server qp"))
        .collect();

    let client_ctx = RdmaContext::open(256, ROCE_V2_GID_INDEX).expect("client open");
    let pool = ShardedQpPool::new(
        &client_ctx,
        ShardedConfig {
            num_shards: NUM_SHARDS,
            cq_size: 256,
            // Zero spin budget: every completion goes through arm + block.
            spin_before_block: Some(std::time::Duration::ZERO),
            ..Default::default()
        },
    )
    .expect("pool");

    let client_eps = pool.endpoints(&client_ctx);
    let server_eps: Vec<_> = server_qps.iter().map(|q| q.endpoint(&server_ctx)).collect();
    for (sqp, cep) in server_qps.iter().zip(&client_eps) {
        sqp.connect(&server_ctx, cep).expect("server connect");
    }
    pool.connect_all(&client_ctx, &server_eps)
        .expect("client connect_all");

    let slot_size = schema.slot_size;
    let mut buf = vec![0u8; slot_size];
    // SAFETY: `buf` outlives the MR and every gather posted against it.
    let mr = unsafe { client_ctx.reg_mr(buf.as_mut_ptr(), buf.len(), IBV_ACCESS_LOCAL_WRITE) }
        .expect("client reg_mr");

    for node in [0usize, 7, 63] {
        let reads = vec![RdmaRead {
            local_addr: buf.as_mut_ptr() as u64,
            local_lkey: mr.lkey(),
            remote_addr: server_base + (node * slot_size) as u64,
            remote_rkey: server_rkey,
            length: slot_size as u32,
        }];
        pool.gather(reads).expect("gather via completion channel");

        let feat_bytes =
            &buf[schema.feature_offset_in_slot..schema.feature_offset_in_slot + FEATURE_DIM * 4];
        let got: Vec<f32> = feat_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, expected[node], "node {node} through the blocking path");
    }
}
