//! Shared receive queue — one recv-sentinel pool feeding many QPs.
//!
//! A server with N shard QPs would otherwise provision N private receive
//! queues, each sized for the worst-case WRITE_WITH_IMM burst; an SRQ
//! lets every QP draw from a single pool sized for the *aggregate*
//! burst. Completions still land on each QP's own recv CQ — only the WR
//! pool is shared.

use super::context::RdmaContext;
use super::ffi::*;
use std::io;
use std::ptr;

/// Owns an `ibv_srq`. Attach QPs at creation via
/// [`super::qp::RdmaQp::create_with_cqs_srq`]; drop those QPs before the
/// SRQ, and the SRQ before its context.
pub struct Srq {
    srq: *mut IbvSrq,
}

// SAFETY: ibverbs SRQs are thread-safe after creation.
unsafe impl Send for Srq {}
// SAFETY: see Send impl above.
unsafe impl Sync for Srq {}

impl Srq {
    pub fn create(ctx: &RdmaContext, max_wr: u32, max_sge: u32) -> io::Result<Self> {
        let mut init = IbvSrqInitAttr {
            srq_context: ptr::null_mut(),
            attr: IbvSrqAttr {
                max_wr,
                max_sge,
                srq_limit: 0,
            },
        };
        // SAFETY: `ctx.pd` is alive; `init` is a valid in-param.
        let srq = unsafe { ibv_create_srq(ctx.pd, &mut init) };
        if srq.is_null() {
            return Err(io::Error::other("ibv_create_srq failed"));
        }
        Ok(Self { srq })
    }

    /// Raw pointer for QP creation.
    pub fn as_ptr(&self) -> *mut IbvSrq {
        self.srq
    }

    /// Post `count` zero-length recv WRs into the shared pool; WR `i`
    /// gets `wr_id = base_wr_id + i`. Same sentinel shape as
    /// [`super::qp::RdmaQp::post_recv_sentinels`], consumed by
    /// WRITE_WITH_IMM arrivals on any attached QP.
    pub fn post_recv_sentinels(&self, base_wr_id: u64, count: u32) -> io::Result<()> {
        if count == 0 {
            return Ok(());
        }
        let mut wrs: Vec<IbvRecvWr> = (0..count)
            .map(|i| IbvRecvWr {
                wr_id: base_wr_id + u64::from(i),
                next: ptr::null_mut(),
                sg_list: ptr::null_mut(),
                num_sge: 0,
            })
            .collect();
        for i in 0..wrs.len() - 1 {
            let next_ptr = &mut wrs[i + 1] as *mut IbvRecvWr;
            wrs[i].next = next_ptr;
        }

        let mut bad_wr: *mut IbvRecvWr = ptr::null_mut();
        // SAFETY: `self.srq` is alive; `wrs[0]` heads a valid chain and
        // the zero-SGE WRs reference no memory the kernel could write.
        let ret = unsafe { ibv_post_srq_recv(self.srq, &mut wrs[0], &mut bad_wr) };
        if ret != 0 {
            return Err(io::Error::other(format!("ibv_post_srq_recv failed: {ret}")));
        }
        Ok(())
    }
}

impl Drop for Srq {
    fn drop(&mut self) {
        // SAFETY: `self.srq` was created by ibv_create_srq and every
        // attached QP has been dropped per the struct contract.
        let ret = unsafe { ibv_destroy_srq(self.srq) };
        if ret != 0 {
            tracing::warn!(ret, "ibv_destroy_srq failed");
        }
    }
}
