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
//! A shard whose QP fails (an error completion, a post failure, or a stall
//! past a ten-second deadline) is quiesced — every WR it was given has finished
//! or been flushed before its caller sees the error — and marked dead; the
//! pool stops routing to it. An RC QP in the error state cannot rejoin
//! without a fresh endpoint exchange with the peer.
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

use super::context::{RdmaContext, RegisteredCq};
use super::event::CompletionChannel;
use super::qp::{QpEndpoint, RdmaQp, RdmaRead, next_wr_generation, required_cq_depth};
use crate::rdma::ffi::{IBV_WC_SUCCESS, IbvQpCap, IbvWc};
use crossbeam_channel::{Sender, bounded};
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Wall-clock bound on one window's signaled completion. RC retries give up
/// well inside it (timeout 14 × retry_cnt 7 ≈ 0.5 s) and flush the QP, so
/// only a hung device reaches it.
const POLL_DEADLINE: Duration = Duration::from_secs(10);

/// Bound on draining a stopped QP; see [`RdmaQp::quiesce`].
const QUIESCE_DEADLINE: Duration = Duration::from_secs(30);

/// Configuration for a `ShardedQpPool`.
///
/// Each shard's CQ is sized from `qp_cap` ([`required_cq_depth`]) — a CQ
/// too shallow for the QP's worst-case flush would drop the completion a
/// drain waits on.
pub struct ShardedConfig {
    /// Number of shards (= QPs = worker threads).
    pub num_shards: usize,
    /// QP capabilities. Use `super::qp::DEFAULT_QP_CAP` for the typical
    /// gather workload; for sustained pipelining bump `max_send_wr` higher.
    /// A gather larger than `max_send_wr` streams through in windows.
    pub qp_cap: IbvQpCap,
    /// Optional core IDs to pin each worker to. Length must equal
    /// `num_shards`, or be empty (no pinning). Cross-NUMA pinning hurts.
    pub worker_cores: Vec<usize>,

    /// Spin for this long waiting for a completion before blocking on the
    /// shard's completion channel. `None` spins indefinitely.
    ///
    /// Busy-polling wins on latency and is right when batches arrive
    /// back-to-back, but N shards spinning through an idle period burn N
    /// cores for nothing — which matters when the sampler and the trainer
    /// want those cores. A budget gets both: the common case still
    /// completes inside the spin, and a genuinely idle shard sleeps until
    /// the NIC wakes it.
    pub spin_before_block: Option<Duration>,
}

impl Default for ShardedConfig {
    /// One shard, default QP caps, no pinning, and a 200µs spin before
    /// blocking — long enough to absorb a local-fabric round trip inline,
    /// short enough that an idle shard yields its core promptly.
    fn default() -> Self {
        Self {
            num_shards: 1,
            qp_cap: super::qp::DEFAULT_QP_CAP,
            worker_cores: Vec::new(),
            spin_before_block: Some(Duration::from_micros(200)),
        }
    }
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
    dead: Arc<AtomicBool>,
}

/// Pool of N (QP, CQ, worker thread) shards.
///
/// Use `endpoints()` to pull each QP's endpoint for the control-plane
/// exchange, then `connect_all()` once you have the matching remote
/// endpoints. From then on `gather(reads)` round-robins across live shards.
///
/// Every QP holds its CQ, channel, and device, so dropping the pool (or a
/// failed `new`) tears them down only after the workers holding them exit.
pub struct ShardedQpPool {
    /// QPs are kept here so `endpoints()` and `connect_all()` can see them
    /// without going through the worker. Each QP is owned by exactly one
    /// worker for posting.
    qps: Vec<Arc<RdmaQp>>,
    /// Worker handles. `Option` join handles so `Drop` can take and join.
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
        let cq_depth = required_cq_depth(&cfg.qp_cap);

        // Built in place so a failure part-way drops through `Drop`, which
        // shuts down and joins every worker already spawned.
        let mut pool = Self {
            qps: Vec::with_capacity(cfg.num_shards),
            handles: Vec::with_capacity(cfg.num_shards),
            next_shard: AtomicUsize::new(0),
        };

        // Workers prefer the NIC's NUMA node for their allocations so
        // per-shard scratch lands where the device DMAs. Preference, not
        // bind: an overfull node spills instead of failing.
        let nic_node = ctx.device_numa_node().and_then(|n| u32::try_from(n).ok());

        for shard_idx in 0..cfg.num_shards {
            // Every shard gets its own completion channel, so a blocking
            // wait wakes only the shard whose completion arrived.
            let channel = CompletionChannel::create(ctx)?;
            let cq = channel.create_cq(ctx, cq_depth)?;
            let qp = Arc::new(RdmaQp::create_with_cqs(ctx, &cfg.qp_cap, &cq, &cq)?);

            // Bounded channel keeps backpressure visible to the caller —
            // an overloaded shard slows submitters before queue grows.
            let (work_tx, work_rx) = bounded::<WorkerMsg>(64);
            let dead = Arc::new(AtomicBool::new(false));

            let shard = Shard {
                qp: Arc::clone(&qp),
                cq,
                channel,
                spin_budget: cfg.spin_before_block,
                dead: Arc::clone(&dead),
                generation: 0,
            };
            let core_id = cfg.worker_cores.get(shard_idx).copied();

            let join = thread::Builder::new()
                .name(format!("aether-shard-{shard_idx}"))
                .spawn(move || {
                    if let Some(id) = core_id {
                        let _ = core_affinity::set_for_current(core_affinity::CoreId { id });
                    }
                    if let Some(node) = nic_node {
                        let _ = aether_mem::numa::prefer_current_thread(node);
                    }
                    shard.run(work_rx);
                })
                .map_err(|e| io::Error::other(format!("spawn shard: {e}")))?;

            pool.qps.push(qp);
            pool.handles.push(ShardHandle {
                work_tx,
                join: Some(join),
                dead,
            });
        }

        Ok(pool)
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

    /// Submit a gather batch to the next live shard, round-robin, and block
    /// on the reply. Errors when every shard is dead.
    pub fn gather(&self, reads: Vec<RdmaRead>) -> io::Result<()> {
        let n = self.handles.len();
        let start = self.next_shard.fetch_add(1, Ordering::Relaxed);
        let live = (0..n)
            .map(|k| (start + k) % n)
            .find(|&i| !self.handles[i].dead.load(Ordering::Acquire))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::BrokenPipe, "every shard's QP has failed")
            })?;
        self.gather_on_shard(live, reads)
    }

    /// Submit to a specific shard — useful when the caller already knows
    /// which shard's MR / staging buffer this batch should land in.
    pub fn gather_on_shard(&self, shard: usize, reads: Vec<RdmaRead>) -> io::Result<()> {
        let handle = self.handles.get(shard).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("shard {shard} out of range ({})", self.handles.len()),
            )
        })?;
        let (tx, rx) = bounded::<io::Result<()>>(1);
        handle
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

    /// Shards still accepting work.
    pub fn live_shards(&self) -> usize {
        self.handles
            .iter()
            .filter(|h| !h.dead.load(Ordering::Acquire))
            .count()
    }
}

impl Drop for ShardedQpPool {
    fn drop(&mut self) {
        // Tell every worker to exit its loop, then join, so no poll is in
        // flight when the QPs, CQs, and channels release.
        for handle in &self.handles {
            let _ = handle.work_tx.send(WorkerMsg::Shutdown);
        }
        for handle in &mut self.handles {
            if let Some(join) = handle.join.take() {
                let _ = join.join();
            }
        }
    }
}

/// One shard's worker state. The QP and CQ are exclusive to it.
struct Shard {
    qp: Arc<RdmaQp>,
    cq: RegisteredCq,
    channel: CompletionChannel,
    spin_budget: Option<Duration>,
    dead: Arc<AtomicBool>,
    /// Tags each window's `wr_id`s so a completion is matched to the window
    /// that posted it.
    generation: u32,
}

impl Shard {
    fn run(mut self, rx: crossbeam_channel::Receiver<WorkerMsg>) {
        while let Ok(msg) = rx.recv() {
            let job = match msg {
                WorkerMsg::Shutdown => return,
                WorkerMsg::Job(j) => j,
            };
            let result = if self.dead.load(Ordering::Acquire) {
                Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "shard QP has failed",
                ))
            } else {
                self.gather(&job.reads)
            };
            let _ = job.reply.send(result);
        }
    }

    /// Stream `reads` through the QP in send-queue-sized windows. On any
    /// failure the QP is quiesced before the error returns, so the caller's
    /// buffers are no longer targeted, and the shard is retired.
    fn gather(&mut self, reads: &[RdmaRead]) -> io::Result<()> {
        let window = self.qp.max_send_wr() as usize;
        for chunk in reads.chunks(window) {
            if let Err(e) = self.post_and_drain(chunk) {
                self.qp.quiesce(&self.cq, QUIESCE_DEADLINE);
                self.dead.store(true, Ordering::Release);
                return Err(e);
            }
        }
        Ok(())
    }

    /// Post one window (chained WRs, only the last signaled) and wait for
    /// its signaled completion.
    ///
    /// The wait spins first, then — once `spin_budget` is spent — arms the
    /// CQ and blocks on the completion channel, so an idle shard stops
    /// consuming its core. `spin_budget: None` keeps a pure busy-poll.
    fn post_and_drain(&mut self, reads: &[RdmaRead]) -> io::Result<()> {
        if reads.is_empty() {
            return Ok(());
        }
        self.generation = next_wr_generation(self.generation);
        let base = u64::from(self.generation) << 32;
        self.qp.post_reads_tagged(reads, reads.len(), base)?;
        let signaled_wr_id = base + (reads.len() - 1) as u64;

        let mut wcs = [IbvWc::default(); 32];
        let started = Instant::now();
        let deadline = started + POLL_DEADLINE;
        let mut armed = false;

        loop {
            let n = self.cq.poll(&mut wcs)?;
            for wc in &wcs[..n] {
                if wc.status != IBV_WC_SUCCESS {
                    // The QP is in the error state now; the caller
                    // quiesces it, which reaps the flushed remainder.
                    return Err(io::Error::other(format!(
                        "RDMA READ failed: status={}, vendor_err={}, wr_id={:#x}",
                        wc.status, wc.vendor_err, wc.wr_id
                    )));
                }
                if wc.wr_id == signaled_wr_id {
                    return Ok(());
                }
            }
            if n > 0 {
                continue;
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("RDMA READ completion timed out after {POLL_DEADLINE:?}"),
                ));
            }

            match self.spin_budget {
                None => std::hint::spin_loop(),
                Some(budget) if started.elapsed() < budget && !armed => std::hint::spin_loop(),
                Some(_) => {
                    if !armed {
                        // Arm before the next poll, never after: arming only
                        // covers completions that arrive from here on, so a
                        // completion landing between the last poll and the arm
                        // would otherwise be missed and the wait would hang.
                        self.channel.arm(&self.cq, false)?;
                        armed = true;
                        // Re-poll immediately — the arm closes the race by
                        // making this poll authoritative for everything before
                        // it, and the block below only for what comes after.
                        continue;
                    }
                    // A timeout is not an error: loop back and re-poll. The
                    // NIC may have raced the block, and a spurious wakeup or a
                    // quiet period both just mean "look again".
                    self.channel.wait(Some(Duration::from_millis(100)))?;
                    armed = false;
                }
            }
        }
    }
}
