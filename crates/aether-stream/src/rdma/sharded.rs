//! Sharded multi-threaded RDMA gather pool.
//!
//! `RdmaQp::post_reads` + `RdmaFeatureClient::post_and_wait` are sequential.
//! At >1 GB/s sustained — or for clients that issue many independent gather
//! batches concurrently — the single-thread post + CQ-poll loop is the
//! bottleneck. This module provides a fixed pool of N (QP, CQ, worker thread)
//! shards. Each worker spins on its OWN CQ, so:
//!
//! - No two threads contend on a single CQ ring.
//! - One slow `gather()` call never head-of-line-blocks others.
//! - Pin worker threads to distinct cores via the `core_id` argument so the
//!   poll loops live in their own L1.
//!
//! GPU integration lives in `gpu/buffer.rs` + `client.rs`; this module is the
//! pure-RDMA half so it's testable on SoftRoCE without CUDA.
//!
//! ## MR cross-thread usage
//!
//! `RegisteredMr` is `Send` — allocate + `reg_mr` on any thread, ship to any
//! other thread, post from any thread. Correctness depends on the FFI
//! layouts in `ffi.rs` matching the kernel ABI exactly (an undersized
//! `IbvSendWr` makes `ibv_post_send` read trailing stack/heap bytes and
//! manifests as `IBV_WC_LOC_PROT_ERR` on completions). The regression guards
//! are `tests/mr_xthread_repro.rs` (four alloc/post thread-layout variants)
//! and `tests/mr_xthread_sharded_repro.rs` (16 caller threads × 4 shards ×
//! 1k iters × 8 reads with MRs pre-registered on the main thread).

use super::context::{RdmaContext, RegisteredCq, create_cq};
use super::qp::{QpEndpoint, RdmaQp, RdmaRead};
use crate::rdma::ffi::{IBV_WC_SUCCESS, IbvCq, IbvQpCap, IbvWc};
use crossbeam_channel::{Sender, bounded};
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// Configuration for a `ShardedQpPool`.
pub struct ShardedConfig {
    /// Number of shards (= QPs = worker threads).
    pub num_shards: usize,
    /// CQ depth per shard. Must be ≥ the largest single-batch read count.
    pub cq_size: i32,
    /// QP capabilities. Use `super::qp::DEFAULT_QP_CAP` for the typical
    /// gather workload; for sustained pipelining bump `max_send_wr` higher.
    pub qp_cap: IbvQpCap,
    /// Optional core IDs to pin each worker to. Length must equal
    /// `num_shards`, or be empty (no pinning). Cross-NUMA pinning hurts.
    pub worker_cores: Vec<usize>,
}

/// One unit of gather work — a batch of RDMA READs + a one-shot reply
/// channel for the result.
struct GatherJob {
    reads: Vec<RdmaRead>,
    reply: Sender<io::Result<()>>,
}

/// Sentinel: tells the worker thread to exit cleanly when the pool drops.
enum WorkerMsg {
    Job(GatherJob),
    Shutdown,
}

/// Handle to one shard's worker.
struct ShardHandle {
    work_tx: Sender<WorkerMsg>,
    join: Option<thread::JoinHandle<()>>,
}

/// Pool of N (QP, CQ, worker thread) shards.
///
/// Use `endpoints()` to pull each QP's endpoint for the control-plane
/// exchange, then `connect_all()` once you have the matching remote
/// endpoints. From then on `gather(reads)` round-robins across shards.
pub struct ShardedQpPool {
    /// QPs are kept here so `endpoints()` and `connect_all()` can see them
    /// without going through the worker. Each QP is owned by exactly one
    /// worker for posting; lifetime-wise we hand the QP to the worker via
    /// the shared Arc, and use it from this struct only for connect.
    qps: Vec<Arc<RdmaQp>>,
    /// CQs held in the pool to keep them alive for the worker threads.
    /// Drop order: workers join first (handles), then CQs drop.
    _cqs: Vec<Arc<RegisteredCq>>,
    /// Worker handles. `Option` so we can `take()` them in `Drop` to join.
    handles: Vec<ShardHandle>,
    /// Round-robin selector for `gather()`.
    next_shard: AtomicUsize,
}

impl ShardedQpPool {
    /// Build the pool: N CQs + N QPs + N worker threads, each pinned if
    /// `worker_cores` is provided.
    pub fn new(ctx: &RdmaContext, cfg: ShardedConfig) -> io::Result<Self> {
        if cfg.num_shards == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "num_shards must be > 0",
            ));
        }
        if !cfg.worker_cores.is_empty() && cfg.worker_cores.len() != cfg.num_shards {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "worker_cores must be empty or have length == num_shards",
            ));
        }

        let mut qps = Vec::with_capacity(cfg.num_shards);
        let mut cqs = Vec::with_capacity(cfg.num_shards);
        let mut handles = Vec::with_capacity(cfg.num_shards);

        for shard_idx in 0..cfg.num_shards {
            let cq = Arc::new(create_cq(ctx, cfg.cq_size)?);
            let qp = Arc::new(RdmaQp::create_with_cqs(
                ctx,
                &cfg.qp_cap,
                cq.as_ptr(),
                cq.as_ptr(),
            )?);

            // Bounded channel keeps backpressure visible to the caller —
            // an overloaded shard slows submitters before queue grows.
            let (work_tx, work_rx) = bounded::<WorkerMsg>(64);

            let qp_for_worker = Arc::clone(&qp);
            let cq_for_worker = Arc::clone(&cq);
            let core_id = cfg.worker_cores.get(shard_idx).copied();

            let join = thread::Builder::new()
                .name(format!("aether-shard-{shard_idx}"))
                .spawn(move || {
                    if let Some(id) = core_id {
                        let _ = core_affinity::set_for_current(core_affinity::CoreId { id });
                    }
                    worker_loop(qp_for_worker, cq_for_worker, work_rx);
                })
                .map_err(|e| io::Error::other(format!("spawn shard: {e}")))?;

            qps.push(qp);
            cqs.push(cq);
            handles.push(ShardHandle {
                work_tx,
                join: Some(join),
            });
        }

        Ok(Self {
            qps,
            _cqs: cqs,
            handles,
            next_shard: AtomicUsize::new(0),
        })
    }

    /// Local endpoints for the control-plane exchange. Pass each one to the
    /// peer; the peer creates its own pool and sends back its endpoints.
    pub fn endpoints(&self, ctx: &RdmaContext) -> Vec<QpEndpoint> {
        self.qps.iter().map(|qp| qp.endpoint(ctx)).collect()
    }

    /// Connect each local QP to its corresponding remote endpoint by index.
    /// Slice length must equal `num_shards`.
    pub fn connect_all(
        &self,
        ctx: &RdmaContext,
        remote_endpoints: &[QpEndpoint],
    ) -> io::Result<()> {
        if remote_endpoints.len() != self.qps.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "remote_endpoints len {} != num_shards {}",
                    remote_endpoints.len(),
                    self.qps.len()
                ),
            ));
        }
        for (qp, remote) in self.qps.iter().zip(remote_endpoints) {
            qp.connect(ctx, remote)?;
        }
        Ok(())
    }

    /// Submit a gather batch — picks the next shard round-robin, sends the
    /// work over the shard's channel, and blocks on the reply.
    pub fn gather(&self, reads: Vec<RdmaRead>) -> io::Result<()> {
        let idx = self.next_shard.fetch_add(1, Ordering::Relaxed) % self.handles.len();
        self.gather_on_shard(idx, reads)
    }

    /// Submit to a specific shard — useful when the caller already knows
    /// which shard's MR / staging buffer this batch should land in.
    pub fn gather_on_shard(&self, shard: usize, reads: Vec<RdmaRead>) -> io::Result<()> {
        let (tx, rx) = bounded::<io::Result<()>>(1);
        self.handles[shard]
            .work_tx
            .send(WorkerMsg::Job(GatherJob { reads, reply: tx }))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "shard worker died"))?;
        rx.recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "shard worker dropped reply"))?
    }

    /// Number of shards in this pool.
    pub fn num_shards(&self) -> usize {
        self.handles.len()
    }
}

impl Drop for ShardedQpPool {
    fn drop(&mut self) {
        // Tell every worker to exit its loop, then join. Order matters:
        // workers must release their QP/CQ refs before the Arcs in `qps` /
        // `_cqs` drop, otherwise the destructors race against an in-flight
        // poll.
        for handle in &mut self.handles {
            let _ = handle.work_tx.send(WorkerMsg::Shutdown);
        }
        for handle in &mut self.handles {
            if let Some(join) = handle.join.take() {
                let _ = join.join();
            }
        }
    }
}

fn worker_loop(qp: Arc<RdmaQp>, cq: Arc<RegisteredCq>, rx: crossbeam_channel::Receiver<WorkerMsg>) {
    while let Ok(msg) = rx.recv() {
        let job = match msg {
            WorkerMsg::Shutdown => return,
            WorkerMsg::Job(j) => j,
        };
        let result = post_and_drain(&qp, cq.as_ptr(), &job.reads);
        let _ = job.reply.send(result);
    }
}

/// Post the batch (chained WRs, only the last signaled) and busy-poll the
/// shard's CQ until the signaled completion lands. Mirrors the production
/// `RdmaFeatureClient::post_and_wait` loop in client.rs.
fn post_and_drain(qp: &RdmaQp, cq: *mut IbvCq, reads: &[RdmaRead]) -> io::Result<()> {
    if reads.is_empty() {
        return Ok(());
    }
    qp.post_reads(reads)?;
    let signaled_wr_id = (reads.len() - 1) as u64;
    let mut wcs = [IbvWc::default(); 32];
    let mut first_error: Option<(u32, u32)> = None;
    loop {
        // SAFETY: `cq` is the live CQ borrowed for this gather batch.
        let n = unsafe { RdmaQp::poll_cq_on(cq, &mut wcs) }?;
        for wc in wcs.iter().take(n) {
            if wc.status != IBV_WC_SUCCESS && first_error.is_none() {
                first_error = Some((wc.status, wc.vendor_err));
            }
            if wc.wr_id == signaled_wr_id {
                if let Some((status, vendor_err)) = first_error {
                    return Err(io::Error::other(format!(
                        "RDMA READ failed: status={status}, vendor_err={vendor_err}"
                    )));
                }
                return Ok(());
            }
        }
        if n == 0 {
            std::hint::spin_loop();
        }
    }
}
