//! RDMA queue pair lifecycle and batch RDMA READ posting.
//!
//! Manages RC (Reliable Connected) QP transitions through
//! RESET → INIT → RTR → RTS, and provides batch RDMA READ
//! with chained work requests (single doorbell).

use super::context::RdmaContext;
use super::ffi::*;
use serde::{Deserialize, Serialize};
use std::io;
use std::ptr;

/// Endpoint info exchanged over TCP for QP connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QpEndpoint {
    /// Queue pair number.
    pub qpn: u32,
    /// Local ID (IB fabrics).
    pub lid: u16,
    /// Global ID (RoCE).
    pub gid: [u8; 16],
    /// Packet sequence number.
    pub psn: u32,
}

/// A single RDMA READ descriptor for batch posting.
pub struct RdmaRead {
    /// Local VRAM/host address to read into.
    pub local_addr: u64,
    /// Local key from ibv_reg_mr.
    pub local_lkey: u32,
    /// Remote address on the server.
    pub remote_addr: u64,
    /// Remote key from the server's advertisement.
    pub remote_rkey: u32,
    /// Number of bytes to read.
    pub length: u32,
}

/// Reusable SGE/WR arrays for batch posting. An `IbvSendWr` is ~128 bytes,
/// so building a 1024-read chain allocated ~132 KB per post; the scratch
/// amortizes that to steady state. Behind a Mutex only for interior
/// mutability — each QP is posted from one worker at a time, so the lock
/// is uncontended.
struct PostScratch {
    sges: Vec<IbvSge>,
    wrs: Vec<IbvSendWr>,
}

/// Owns an RC queue pair. Created via `RdmaQp::create`, connected
/// via `connect`, then used for RDMA READ operations.
pub struct RdmaQp {
    qp: *mut IbvQp,
    /// Send-queue depth this QP was created with. Posts are bounded against it
    /// so an over-long chain fails fast in Rust instead of overflowing the
    /// queue inside `ibv_post_send`.
    max_send_wr: u32,
    /// Reusable posting arrays (see [`PostScratch`]).
    post_scratch: std::sync::Mutex<PostScratch>,
}

// SAFETY: QP is thread-safe after creation (single-threaded posting assumed).
unsafe impl Send for RdmaQp {}
// SAFETY: see Send impl above.
unsafe impl Sync for RdmaQp {}

/// Default starting packet sequence number for new QPs.
///
/// Held at 0 rather than randomized per QP: this crate pulls in no randomness
/// source, and the QP endpoints are exchanged over an out-of-band TCP control
/// plane on a trusted fabric, so the off-path packet-injection guard a random
/// PSN provides is not relied on here. Both ends use this same value so the
/// RTR `rq_psn` and RTS `sq_psn` agree.
const DEFAULT_PSN: u32 = 0;

/// Default QP capabilities — sized for batched RDMA READ workloads typical of
/// the GNN feature gather use case. Callers with different needs (deeper
/// pipelining, larger SGL fanout, inline data) should construct their own
/// `IbvQpCap` and pass it to `RdmaQp::create`. The values here are NOT
/// auto-tuned against device caps — that's deliberate, peak performance code
/// must know its hardware.
pub const DEFAULT_QP_CAP: IbvQpCap = IbvQpCap {
    max_send_wr: 4096,
    max_recv_wr: 1, // RDMA READ doesn't need recv WRs
    max_send_sge: 1,
    max_recv_sge: 1,
    max_inline_data: 0,
};

impl RdmaQp {
    /// Create an RC queue pair on the given context with caller-supplied caps.
    ///
    /// Uses the context's PD and shared CQ for both send and receive. Use
    /// `DEFAULT_QP_CAP` for the sane GNN-gather defaults. For sharded
    /// many-worker designs use `create_with_cqs` to pin each QP to its own CQ.
    pub fn create(ctx: &RdmaContext, cap: &IbvQpCap) -> io::Result<Self> {
        Self::create_with_cqs(ctx, cap, ctx.cq, ctx.cq)
    }

    /// Create an RC queue pair on the given context with explicit send/recv
    /// CQs. Use `super::context::create_cq` to allocate per-shard CQs so
    /// worker threads don't contend on a single CQ in the poll loop.
    pub fn create_with_cqs(
        ctx: &RdmaContext,
        cap: &IbvQpCap,
        send_cq: *mut IbvCq,
        recv_cq: *mut IbvCq,
    ) -> io::Result<Self> {
        // SAFETY: zeroed init of a POD ibverbs struct is sound.
        let mut init_attr: IbvQpInitAttr = unsafe { std::mem::zeroed() };
        init_attr.send_cq = send_cq;
        init_attr.recv_cq = recv_cq;
        init_attr.qp_type = IBV_QPT_RC;
        init_attr.cap = *cap;

        // SAFETY: `ctx.pd` is alive; `init_attr` is a valid out-param.
        let qp = unsafe { ibv_create_qp(ctx.pd, &mut init_attr) };
        if qp.is_null() {
            return Err(io::Error::other("ibv_create_qp failed"));
        }

        Ok(Self {
            qp,
            max_send_wr: cap.max_send_wr,
            post_scratch: std::sync::Mutex::new(PostScratch {
                sges: Vec::new(),
                wrs: Vec::new(),
            }),
        })
    }

    /// Send-queue depth this QP was created with (`IbvQpCap::max_send_wr`).
    /// Callers size posting windows against it so pipelining follows the
    /// QP's actual capability.
    pub fn max_send_wr(&self) -> u32 {
        self.max_send_wr
    }

    /// Get local endpoint for exchange over TCP control plane.
    pub fn endpoint(&self, ctx: &RdmaContext) -> QpEndpoint {
        // SAFETY: `self.qp` is alive for this RdmaQp's lifetime.
        let qpn = unsafe { (*self.qp).qp_num };
        QpEndpoint {
            qpn,
            lid: ctx.port_lid,
            gid: ctx.port_gid.raw,
            psn: DEFAULT_PSN,
        }
    }

    /// Transition RESET → INIT → RTR → RTS to connect to a remote QP.
    ///
    /// The RDMA-atomic depths (`max_dest_rd_atomic` / `max_rd_atomic`) are
    /// requested as 16. This crate links only the subset of ibverbs in
    /// `ffi.rs`, which has no `ibv_query_device`, so the value is not clamped to
    /// the device's `max_qp_rd_atom`. On an HCA that reports a smaller cap,
    /// `ibv_modify_qp` rejects the RTR/RTS transition and `connect` returns the
    /// underlying error — feed a smaller value here if you target such hardware.
    pub fn connect(&self, ctx: &RdmaContext, remote: &QpEndpoint) -> io::Result<()> {
        self.to_init(ctx)?;
        self.to_rtr(ctx, remote)?;
        self.to_rts()?;
        Ok(())
    }

    /// Post a batch of RDMA READs as a chained WR linked list (single doorbell).
    ///
    /// The last WR is signaled; caller should poll CQ for exactly one completion.
    /// Equivalent to `post_reads_with_signaling(reads, reads.len())` — the
    /// "drain once at end" pattern that's right for a request/response gather.
    pub fn post_reads(&self, reads: &[RdmaRead]) -> io::Result<()> {
        let stride = reads.len().max(1);
        self.post_reads_with_signaling(reads, stride)
    }

    /// Post a batch with caller-controlled signaling cadence.
    ///
    /// Every `signal_every_n`-th WR is signaled (0-indexed: WRs at positions
    /// `n-1, 2n-1, 3n-1, ...` and the final WR get `IBV_SEND_SIGNALED`). This
    /// lets a sustained-pipeline caller throttle posts: keep N inflight, drain
    /// every Nth completion to reclaim slots in the QP send queue. For one-shot
    /// "post then drain" use the simpler `post_reads`.
    ///
    /// `signal_every_n` must be ≥ 1; pass `reads.len()` to signal only the last.
    pub fn post_reads_with_signaling(
        &self,
        reads: &[RdmaRead],
        signal_every_n: usize,
    ) -> io::Result<()> {
        if reads.is_empty() {
            return Ok(());
        }
        if signal_every_n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "signal_every_n must be >= 1",
            ));
        }
        // A chain longer than the QP's send-queue depth would overflow inside
        // `ibv_post_send` (ENOMEM) or, worse, be partially accepted. Reject up
        // front so the failure is observable here.
        if reads.len() > self.max_send_wr as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "reads.len() {} exceeds QP max_send_wr {}",
                    reads.len(),
                    self.max_send_wr
                ),
            ));
        }

        // Build SGE and WR arrays in the QP's reusable scratch.
        let mut scratch = self.post_scratch.lock().unwrap_or_else(|e| e.into_inner());
        let PostScratch { sges, wrs } = &mut *scratch;
        sges.clear();
        wrs.clear();
        sges.reserve(reads.len());
        wrs.reserve(reads.len());

        for read in reads {
            sges.push(IbvSge {
                addr: read.local_addr,
                length: read.length,
                lkey: read.local_lkey,
            });
        }

        let last = reads.len() - 1;
        for (i, read) in reads.iter().enumerate() {
            // SAFETY: zeroed init of a POD ibverbs struct is sound.
            let mut wr: IbvSendWr = unsafe { std::mem::zeroed() };
            wr.wr_id = i as u64;
            wr.sg_list = &mut sges[i] as *mut IbvSge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_RDMA_READ;
            // Signal at the cadence requested AND always on the final WR so
            // a "drain to completion" caller has a guaranteed sentinel.
            let on_stride = (i + 1) % signal_every_n == 0;
            wr.send_flags = if on_stride || i == last {
                IBV_SEND_SIGNALED
            } else {
                0
            };
            wr.wr = IbvSendWrUnion {
                rdma: IbvSendWrRdma {
                    remote_addr: read.remote_addr,
                    rkey: read.remote_rkey,
                },
            };
            wrs.push(wr);
        }

        // Chain WRs via next pointers
        for i in 0..wrs.len() - 1 {
            let next_ptr = &mut wrs[i + 1] as *mut IbvSendWr;
            wrs[i].next = next_ptr;
        }
        // Last WR has next = null (already zeroed)

        let mut bad_wr: *mut IbvSendWr = ptr::null_mut();
        // SAFETY: `self.qp` is alive; `wrs[0]` is the head of a valid WR chain.
        let ret = unsafe { ibv_post_send(self.qp, &mut wrs[0], &mut bad_wr) };
        if ret != 0 {
            // `ibv_post_send` sets `bad_wr` to the first WR it did NOT process;
            // every WR before it in the chain was accepted onto the send queue.
            // Surface that count so the caller's completion accounting stays
            // correct (those accepted WRs will still generate CQEs).
            let accepted = if bad_wr.is_null() {
                // Whole chain rejected with no specific offender reported.
                0
            } else {
                // SAFETY: `bad_wr`, when non-null, points at one of the WRs in
                // `wrs` (the chain we just handed in); the offset is therefore
                // a valid in-bounds element index.
                let off = unsafe { bad_wr.offset_from(wrs.as_ptr()) };
                if (0..wrs.len() as isize).contains(&off) {
                    off as usize
                } else {
                    0
                }
            };
            return Err(io::Error::other(format!(
                "ibv_post_send failed: {ret} ({accepted} of {} WRs accepted before bad_wr)",
                wrs.len()
            )));
        }

        Ok(())
    }

    /// Poll the context's shared CQ for completions. Use `poll_cq_on` if this
    /// QP was created with a dedicated CQ via `create_with_cqs`.
    pub fn poll_cq(&self, ctx: &RdmaContext, wcs: &mut [IbvWc]) -> io::Result<usize> {
        // SAFETY: `ctx.cq` is alive for as long as `ctx` is borrowed.
        unsafe { Self::poll_cq_on(ctx.cq, wcs) }
    }

    /// Poll an explicit CQ — for sharded designs where each worker thread
    /// owns its own CQ and must not see other workers' completions.
    ///
    /// # Safety
    /// `cq` must be a live `*mut IbvCq` that hasn't been destroyed.
    pub unsafe fn poll_cq_on(cq: *mut IbvCq, wcs: &mut [IbvWc]) -> io::Result<usize> {
        // SAFETY: `cq` validity is the caller's contract above.
        let ret = unsafe { ibv_poll_cq(cq, wcs.len() as i32, wcs.as_mut_ptr()) };
        if ret < 0 {
            return Err(io::Error::other(format!("ibv_poll_cq failed: {ret}")));
        }
        Ok(ret as usize)
    }

    // -----------------------------------------------------------------------
    // QP state transitions
    // -----------------------------------------------------------------------

    /// RESET → INIT
    fn to_init(&self, ctx: &RdmaContext) -> io::Result<()> {
        // SAFETY: zeroed init of a POD ibverbs struct is sound.
        let mut attr: IbvQpAttr = unsafe { std::mem::zeroed() };
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = 1;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_READ as u32 | IBV_ACCESS_LOCAL_WRITE as u32;

        let mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
        // SAFETY: `self.qp` is alive; `attr` is a valid in-param.
        let ret = unsafe { ibv_modify_qp(self.qp, &mut attr, mask) };
        if ret != 0 {
            return Err(io::Error::other(format!("QP RESET→INIT failed: {ret}")));
        }
        let _ = ctx; // port_num is hardcoded to 1
        Ok(())
    }

    /// INIT → RTR (Ready to Receive)
    fn to_rtr(&self, ctx: &RdmaContext, remote: &QpEndpoint) -> io::Result<()> {
        // SAFETY: zeroed init of a POD ibverbs struct is sound.
        let mut attr: IbvQpAttr = unsafe { std::mem::zeroed() };
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_4096;
        attr.dest_qp_num = remote.qpn;
        attr.rq_psn = remote.psn;
        attr.max_dest_rd_atomic = 16;
        attr.min_rnr_timer = 12;

        // Address handle
        attr.ah_attr.port_num = 1;
        attr.ah_attr.dlid = remote.lid;
        attr.ah_attr.sl = 0;

        // Use GRH for RoCE or if GID is non-zero
        let gid_nonzero = remote.gid.iter().any(|&b| b != 0);
        if gid_nonzero {
            attr.ah_attr.is_global = 1;
            attr.ah_attr.grh.dgid = IbvGid { raw: remote.gid };
            // sgid_index must point at the local GID we advertise — using
            // index 0 unconditionally would mismatch when the routable GID
            // lives at a different index (e.g. RoCEv2 IPv4-mapped at 1).
            attr.ah_attr.grh.sgid_index = ctx.gid_index;
            attr.ah_attr.grh.hop_limit = 64;
            attr.ah_attr.grh.traffic_class = 0;
        }

        let mask = IBV_QP_STATE
            | IBV_QP_AV
            | IBV_QP_PATH_MTU
            | IBV_QP_DEST_QPN
            | IBV_QP_RQ_PSN
            | IBV_QP_MAX_DEST_RD_ATOMIC
            | IBV_QP_MIN_RNR_TIMER;
        // SAFETY: `self.qp` is alive; `attr` is a valid in-param.
        let ret = unsafe { ibv_modify_qp(self.qp, &mut attr, mask) };
        if ret != 0 {
            return Err(io::Error::other(format!("QP INIT→RTR failed: {ret}")));
        }
        let _ = ctx;
        Ok(())
    }

    /// RTR → RTS (Ready to Send)
    fn to_rts(&self) -> io::Result<()> {
        // SAFETY: zeroed init of a POD ibverbs struct is sound.
        let mut attr: IbvQpAttr = unsafe { std::mem::zeroed() };
        attr.qp_state = IBV_QPS_RTS;
        attr.sq_psn = DEFAULT_PSN;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.max_rd_atomic = 16;

        let mask = IBV_QP_STATE
            | IBV_QP_SQ_PSN
            | IBV_QP_TIMEOUT
            | IBV_QP_RETRY_CNT
            | IBV_QP_RNR_RETRY
            | IBV_QP_MAX_QP_RD_ATOMIC;
        // SAFETY: `self.qp` is alive; `attr` is a valid in-param.
        let ret = unsafe { ibv_modify_qp(self.qp, &mut attr, mask) };
        if ret != 0 {
            return Err(io::Error::other(format!("QP RTR→RTS failed: {ret}")));
        }
        Ok(())
    }
}

impl Drop for RdmaQp {
    fn drop(&mut self) {
        // SAFETY: `self.qp` was returned by `ibv_create_qp` and is live until now.
        unsafe { ibv_destroy_qp(self.qp) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_wr_chain_addresses() {
        // Verify WR chain construction for 4 nodes
        let reads: Vec<RdmaRead> = (0..4)
            .map(|i| RdmaRead {
                local_addr: 0x1000 + i * 4096,
                local_lkey: 42,
                remote_addr: 0xDEAD_0000 + i * 4096,
                remote_rkey: 99,
                length: 3088,
            })
            .collect();

        // Build SGE + WR arrays (same logic as post_reads but without ibv_post_send)
        let mut sges: Vec<IbvSge> = Vec::with_capacity(reads.len());
        let mut wrs: Vec<IbvSendWr> = Vec::with_capacity(reads.len());

        for read in &reads {
            sges.push(IbvSge {
                addr: read.local_addr,
                length: read.length,
                lkey: read.local_lkey,
            });
        }

        for (i, read) in reads.iter().enumerate() {
            // SAFETY: zeroed init of a POD ibverbs struct is sound.
            let mut wr: IbvSendWr = unsafe { std::mem::zeroed() };
            wr.wr_id = i as u64;
            wr.sg_list = &mut sges[i] as *mut IbvSge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_RDMA_READ;
            wr.send_flags = if i == reads.len() - 1 {
                IBV_SEND_SIGNALED
            } else {
                0
            };
            wr.wr = IbvSendWrUnion {
                rdma: IbvSendWrRdma {
                    remote_addr: read.remote_addr,
                    rkey: read.remote_rkey,
                },
            };
            wrs.push(wr);
        }

        // Chain
        for i in 0..wrs.len() - 1 {
            let next_ptr = &mut wrs[i + 1] as *mut IbvSendWr;
            wrs[i].next = next_ptr;
        }

        // Verify chain
        assert_eq!(wrs.len(), 4);
        assert_eq!(wrs[0].wr_id, 0);
        assert_eq!(wrs[3].wr_id, 3);
        assert!(!wrs[0].next.is_null());
        assert!(!wrs[1].next.is_null());
        assert!(!wrs[2].next.is_null());
        assert!(wrs[3].next.is_null());

        // Only last is signaled
        assert_eq!(wrs[0].send_flags, 0);
        assert_eq!(wrs[3].send_flags, IBV_SEND_SIGNALED);

        // Verify addresses
        assert_eq!(sges[0].addr, 0x1000);
        assert_eq!(sges[1].addr, 0x1000 + 4096);
        // SAFETY: we just wrote the rdma variant above.
        let r0 = unsafe { wrs[0].wr.rdma.remote_addr };
        // SAFETY: same as above.
        let r1 = unsafe { wrs[1].wr.rdma.remote_addr };
        assert_eq!(r0, 0xDEAD_0000);
        assert_eq!(r1, 0xDEAD_0000 + 4096);
    }
}
