//! Shared receive queue — one recv-sentinel pool feeding many QPs.
//!
//! A server with N shard QPs would otherwise provision N private receive
//! queues, each sized for the worst-case WRITE_WITH_IMM burst; an SRQ
//! lets every QP draw from a single pool sized for the *aggregate*
//! burst. Completions still land on each QP's own recv CQ — only the WR
//! pool is shared.

use super::context::{Device, RdmaContext};
use super::ffi::*;
use std::io;
use std::ptr;
use std::sync::Arc;

struct SrqInner {
    srq: *mut IbvSrq,
    // Released after `Drop::drop` destroys the SRQ.
    _dev: Arc<Device>,
}

// SAFETY: ibverbs SRQs are thread-safe after creation.
unsafe impl Send for SrqInner {}
// SAFETY: see Send impl above.
unsafe impl Sync for SrqInner {}

impl Drop for SrqInner {
    fn drop(&mut self) {
        // SAFETY: created by ibv_create_srq; every attached QP holds an
        // `Srq` clone, so none is left.
        let ret = unsafe { ibv_destroy_srq(self.srq) };
        if ret != 0 {
            tracing::warn!(ret, "ibv_destroy_srq failed");
        }
    }
}

/// Owns an `ibv_srq`. Attach QPs at creation via
/// [`super::qp::RdmaQp::create_with_cqs_srq`]; each attached QP holds a
/// clone, so the SRQ outlives them.
///
/// TODO(deferred): the whole module has no product caller yet — it is
/// exercised only by `tests/softroce_e2e.rs`. An SRQ earns its keep on
/// the server side, where one receive queue backs many client QPs so
/// buffer memory scales with concurrent arrivals instead of with
/// connection count. The in-tree feature server is examples-only, so
/// there is nothing to attach this to; revisit when it becomes a product
/// component, together with the WRITE_WITH_IMM push path it pairs with.
#[derive(Clone)]
pub struct Srq {
    inner: Arc<SrqInner>,
}

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
        // SAFETY: the context's PD is alive; `init` is a valid in-param.
        let srq = unsafe { ibv_create_srq(ctx.pd_ptr(), &mut init) };
        if srq.is_null() {
            return Err(io::Error::other("ibv_create_srq failed"));
        }
        Ok(Self {
            inner: Arc::new(SrqInner {
                srq,
                _dev: ctx.device().clone(),
            }),
        })
    }

    /// Raw pointer. Using it is `unsafe`; it lives as long as any clone.
    pub fn as_ptr(&self) -> *mut IbvSrq {
        self.inner.srq
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
        // SAFETY: the SRQ is alive; `wrs[0]` heads a valid chain and the
        // zero-SGE WRs reference no memory the kernel could write.
        let ret = unsafe { ibv_post_srq_recv(self.inner.srq, &mut wrs[0], &mut bad_wr) };
        if ret != 0 {
            return Err(io::Error::other(format!("ibv_post_srq_recv failed: {ret}")));
        }
        Ok(())
    }
}
