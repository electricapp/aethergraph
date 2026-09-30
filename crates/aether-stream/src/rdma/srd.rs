//! AWS EFA SRD (Scalable Reliable Datagram) RDMA READ path.
//!
//! EFA on AWS does not expose classic RC QPs — `ibv_create_qp` with
//! `IBV_QPT_RC` returns failure against an EFA device. The supported
//! transport is SRD: a reliable-delivery, datagram-style protocol that
//! carries RDMA READ (and WRITE, ATOMIC on modern firmware) on top of a
//! driver QP (`IBV_QPT_DRIVER` + `efadv_create_qp_ex` with
//! `driver_qp_type = SRD`). Completions come off an `ibv_cq_ex` via the
//! builder API.
//!
//! Callers pick up an `SrdContext`, create an `SrdQp`, transition it
//! RESET → INIT → RTR → RTS, exchange endpoints (GID + QPN + QKEY)
//! with the peer, and post RDMA READs via `SrdQp::post_rdma_read`. The
//! actual post-and-complete dance is wrapped to a single FFI crossing by
//! the C shim in `csrc/ibv_shim.c` (`aether_ibv_post_rdma_read_srd` +
//! `aether_ibv_poll_cq_ex_one`) so Rust never chases `ibv_qp_ex`'s
//! function-pointer builder at the FFI boundary.
//!
//! Every verbs object holds an `Arc` to what it was created on (QP → CQ →
//! device), so teardown runs child-before-parent however handles drop.

#![cfg(all(target_os = "linux", feature = "efa"))]

use super::context::RegisteredMr;
use super::control::{handshake_deadline, recv_msg, send_msg};
use super::efa_ffi::*;
use super::ffi::{
    IBV_ACCESS_LOCAL_WRITE, IBV_QP_PKEY_INDEX, IBV_QP_PORT, IBV_QP_QKEY, IBV_QP_SQ_PSN,
    IBV_QP_STATE, IBV_QPS_ERR, IBV_QPS_INIT, IBV_QPS_RTR, IBV_QPS_RTS, IBV_SEND_SIGNALED,
    IBV_WC_SUCCESS, IbvAhAttr, IbvContext, IbvGid, IbvGlobalRoute, IbvPd, IbvQp, IbvQpAttr,
    IbvQpCap,
};
use super::layout::RemoteTable;
use super::qp::next_wr_generation;
use crate::feature_table::FeatureSchema;
use serde::{Deserialize, Serialize};
use std::alloc::Layout;
use std::ffi::CStr;
use std::io;
use std::net::{TcpListener, TcpStream};
use std::ptr::{self, NonNull};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Default SRD QP capabilities. `max_send_sge` is capped at 2 on current
/// EFA firmware — callers wanting longer per-WR SGLs need to chain WRs.
/// `max_recv_wr` must be ≥ 1 on SRD; the EFA driver rejects INIT transitions
/// on QPs whose recv queue is zero-sized even when no receives will be
/// posted. Upper bounds against the device's `max_sq_wr`/`max_rq_wr`
/// (currently 4096/1 on c6gn.16xlarge) are the caller's responsibility —
/// `srd_qp_cap_ablation` in `tests/srd_e2e.rs` sweeps the supported space.
pub const DEFAULT_SRD_QP_CAP: IbvQpCap = IbvQpCap {
    max_send_wr: 4096,
    max_recv_wr: 1,
    max_send_sge: 1,
    max_recv_sge: 1,
    max_inline_data: 0,
};

/// Q-key used on the SRD path. EFA does not enforce any particular value,
/// but both ends must agree — we hardcode a single constant so the control
/// plane doesn't need to carry it.
pub const DEFAULT_SRD_QKEY: u32 = 0x1111_1111;

/// Wall-clock bound on draining one posted batch before the QP is forced
/// into the error state to flush it.
const DRAIN_DEADLINE: Duration = Duration::from_secs(30);

/// Rounds of torn-row re-reads before a gather gives up.
const MAX_RETRIES: usize = 8;

/// Everything a peer needs to address us: GID + QPN + QKEY. Exchange these
/// out-of-band (TCP control plane) before posting reads. The peer creates
/// their OWN `ibv_ah` pointing at our GID — AHN is a local identifier the
/// kernel assigns to that handle, not something the peer needs from us.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SrdEndpoint {
    /// Local GID that identifies the EFA port.
    pub gid: [u8; 16],
    /// Our SRD QP number.
    pub qpn: u32,
    /// Shared QKEY. Mismatch between sender and receiver produces
    /// `IBV_WC_GENERAL_ERR` on completion.
    pub qkey: u32,
}

/// SRD advertisement: everything a client needs to read from our feature
/// table over SRD — MR rkey, base address, schema, plus our SRD endpoint
/// so the client can address us.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SrdAdvertisement {
    /// Base virtual address of the feature table on the server.
    pub base_addr: u64,
    /// Remote key from the server's `ibv_reg_mr`.
    pub rkey: u32,
    /// Feature table layout.
    pub schema: FeatureSchema,
    /// Server's SRD endpoint (GID + QPN + QKEY).
    pub endpoint: SrdEndpoint,
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// An open EFA device and its protection domain, kept alive by every object
/// created on it.
struct SrdDevice {
    context: *mut IbvContext,
    pd: *mut IbvPd,
}

// SAFETY: ibverbs contexts and PDs are thread-safe after creation.
unsafe impl Send for SrdDevice {}
// SAFETY: see Send impl above.
unsafe impl Sync for SrdDevice {}

impl Drop for SrdDevice {
    fn drop(&mut self) {
        // SAFETY: every object on this PD holds an `Arc<SrdDevice>`, so
        // none is left.
        let rc = unsafe { super::ffi::ibv_dealloc_pd(self.pd) };
        if rc != 0 {
            tracing::warn!(rc, "ibv_dealloc_pd failed");
        }
        // SAFETY: opened in `SrdContext::open`; nothing created on it remains.
        let rc = unsafe { super::ffi::ibv_close_device(self.context) };
        if rc != 0 {
            tracing::warn!(rc, "ibv_close_device failed");
        }
    }
}

/// The extended CQ, kept alive by every QP created on it.
struct SrdCq {
    cq_ex: *mut IbvCqEx,
    // Released after `Drop::drop` destroys the CQ.
    _dev: Arc<SrdDevice>,
}

// SAFETY: ibverbs CQs are thread-safe after creation.
unsafe impl Send for SrdCq {}
// SAFETY: see Send impl above.
unsafe impl Sync for SrdCq {}

impl Drop for SrdCq {
    fn drop(&mut self) {
        // SAFETY: `cq_ex` was created in `SrdContext::open` and is live.
        let cq = unsafe { aether_ibv_cq_ex_to_cq(self.cq_ex) };
        // SAFETY: every QP on the CQ holds an `Arc<SrdCq>`, so none is left.
        let rc = unsafe { super::ffi::ibv_destroy_cq(cq) };
        if rc != 0 {
            tracing::warn!(rc, "ibv_destroy_cq failed");
        }
    }
}

/// EFA device context. Opens the first visible EFA device (`rdmap*` or
/// `efa_*` depending on kernel naming) and allocates a PD + extended CQ.
pub struct SrdContext {
    dev: Arc<SrdDevice>,
    cq: Arc<SrdCq>,
    port_gid: IbvGid,
    gid_index: u8,
}

impl SrdContext {
    /// Open the first EFA device (by index 0) and allocate a PD + extended CQ.
    /// The CQ is sized to `cq_size` CQEs, WITH `IBV_WC_EX_WITH_BYTE_LEN` so
    /// completions carry a byte count.
    pub fn open(cq_size: u32, gid_index: u8) -> io::Result<Self> {
        let mut n: i32 = 0;
        // SAFETY: ibverbs FFI; `n` is a valid out-param.
        let list = unsafe { super::ffi::ibv_get_device_list(&mut n) };
        if list.is_null() || n <= 0 {
            if !list.is_null() {
                // SAFETY: `list` is non-null and not yet freed.
                unsafe { super::ffi::ibv_free_device_list(list) };
            }
            return Err(io::Error::new(io::ErrorKind::NotFound, "no RDMA devices"));
        }
        // Pick the first EFA-capable device. `efadv_query_device` returns
        // non-zero on non-EFA HCAs, so we probe and skip.
        let mut context: *mut IbvContext = ptr::null_mut();
        let mut picked_name = String::new();
        for i in 0..n as isize {
            // SAFETY: `i < n`; `list` is non-null per the check above.
            let dev_slot = unsafe { list.offset(i) };
            // SAFETY: `dev_slot` is in-bounds; the list is non-null.
            let dev = unsafe { *dev_slot };
            // SAFETY: `dev` is a valid device pointer from the list.
            let ctx = unsafe { super::ffi::ibv_open_device(dev) };
            if ctx.is_null() {
                continue;
            }
            let mut efa_attr = EfadvDeviceAttr::default();
            // SAFETY: `ctx` is open; `efa_attr` is a valid out-param of the
            // size passed.
            let rc = unsafe {
                efadv_query_device(
                    ctx,
                    &mut efa_attr,
                    std::mem::size_of::<EfadvDeviceAttr>() as u32,
                )
            };
            if rc == 0 && (efa_attr.device_caps & EFADV_DEVICE_ATTR_CAPS_RDMA_READ) != 0 {
                context = ctx;
                // SAFETY: `dev` is a valid device pointer from the list.
                let name_ptr = unsafe { super::ffi::ibv_get_device_name(dev) };
                if !name_ptr.is_null() {
                    // SAFETY: `name_ptr` is a NUL-terminated string owned by
                    // ibverbs, valid for the list's lifetime.
                    picked_name = unsafe { CStr::from_ptr(name_ptr) }
                        .to_string_lossy()
                        .into_owned();
                }
                break;
            }
            // SAFETY: `ctx` was opened above and is not the picked device.
            unsafe { super::ffi::ibv_close_device(ctx) };
        }
        // SAFETY: `list` is non-null and not yet freed.
        unsafe { super::ffi::ibv_free_device_list(list) };
        if context.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "no EFA device with RDMA_READ capability",
            ));
        }
        tracing::debug!(device = %picked_name, "opened EFA device");

        // SAFETY: `context` is the just-opened device context.
        let pd = unsafe { super::ffi::ibv_alloc_pd(context) };
        if pd.is_null() {
            // SAFETY: `context` is non-null and not yet closed.
            unsafe { super::ffi::ibv_close_device(context) };
            return Err(io::Error::other("ibv_alloc_pd"));
        }
        // From here every early return unwinds through the RAII owners.
        let dev = Arc::new(SrdDevice { context, pd });

        let mut cq_attr = IbvCqInitAttrEx::zeroed();
        cq_attr.cqe = cq_size;
        cq_attr.wc_flags = IBV_WC_EX_WITH_BYTE_LEN;
        // SAFETY: `context` is open; `cq_attr` is initialized above.
        let cq_ex = unsafe { ibv_create_cq_ex(context, &mut cq_attr) };
        if cq_ex.is_null() {
            return Err(io::Error::other("ibv_create_cq_ex"));
        }
        let cq = Arc::new(SrdCq {
            cq_ex,
            _dev: dev.clone(),
        });

        // SAFETY: zeroed init of POD struct is sound.
        let mut port_gid: IbvGid = unsafe { std::mem::zeroed() };
        // SAFETY: `context` is open; `port_gid` is a valid out-param.
        let rc = unsafe { super::ffi::ibv_query_gid(context, 1, gid_index as i32, &mut port_gid) };
        if rc != 0 {
            return Err(io::Error::other(format!(
                "ibv_query_gid({gid_index}) rc={rc}"
            )));
        }

        Ok(Self {
            dev,
            cq,
            port_gid,
            gid_index,
        })
    }

    /// Register a memory region for local write (SGE destination) or remote
    /// read (peer source). The MR keeps the PD alive until it deregisters.
    ///
    /// # Safety
    /// Same contract as [`super::context::RdmaContext::reg_mr`]:
    /// `[addr, addr + len)` must stay valid until the MR drops and every WR
    /// referencing it has completed, and no Rust reference to bytes a WR
    /// targets may be live while that WR is in flight.
    pub unsafe fn reg_mr(
        &self,
        addr: *mut u8,
        len: usize,
        access: i32,
    ) -> io::Result<RegisteredMr> {
        // SAFETY: the PD is alive; `addr/len/access` are the caller's contract.
        let mr =
            unsafe { super::ffi::ibv_reg_mr(self.dev.pd, addr as *mut libc::c_void, len, access) };
        if mr.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `mr` is a fresh registration on this context's PD.
        Ok(unsafe { RegisteredMr::from_raw(mr, self.dev.clone()) })
    }

    /// Raw PD pointer. Using it is `unsafe`; it lives as long as this
    /// context or anything created on it.
    pub fn pd(&self) -> *mut IbvPd {
        self.dev.pd
    }

    /// Raw extended-CQ pointer; same lifetime rule as [`Self::pd`].
    pub fn cq_ex(&self) -> *mut IbvCqEx {
        self.cq.cq_ex
    }

    /// Raw context pointer; same lifetime rule as [`Self::pd`].
    pub fn context_ptr(&self) -> *mut IbvContext {
        self.dev.context
    }

    /// Local GID. Peers need this to create an AH pointing at us.
    pub fn gid(&self) -> [u8; 16] {
        self.port_gid.raw
    }

    /// GID index we queried.
    pub fn gid_index(&self) -> u8 {
        self.gid_index
    }

    /// Drain one CQE through the extended-CQ builder API. Returns:
    ///   Ok(Some(snapshot)) on a real completion,
    ///   Ok(None) if the CQ was empty,
    ///   Err(_) on hard failure.
    pub fn poll_one(&self) -> io::Result<Option<AetherCqeSnapshot>> {
        let mut out = AetherCqeSnapshot::default();
        // SAFETY: the CQ is alive; `out` is a valid out-param.
        let rc = unsafe { aether_ibv_poll_cq_ex_one(self.cq.cq_ex, &mut out) };
        match rc {
            0 => Ok(Some(out)),
            libc::ENOENT => Ok(None),
            other => Err(io::Error::other(format!("poll_cq_ex rc={other}"))),
        }
    }

    /// Drain up to `out.len()` CQEs in one FFI crossing.
    pub fn poll_many(&self, out: &mut [AetherCqeSnapshot]) -> io::Result<usize> {
        let want = u32::try_from(out.len()).unwrap_or(u32::MAX);
        // SAFETY: the CQ is alive; `out` has room for `want` snapshots.
        let got = unsafe { aether_ibv_poll_cq_ex_many(self.cq.cq_ex, out.as_mut_ptr(), want) };
        if got < 0 {
            return Err(io::Error::from_raw_os_error(-got));
        }
        Ok(got as usize)
    }
}

// ---------------------------------------------------------------------------
// Address handle
// ---------------------------------------------------------------------------

/// Wraps an `ibv_ah` with its AHN. Drop calls `ibv_destroy_ah`.
pub struct SrdAddressHandle {
    ah: *mut IbvAh,
    ahn: u16,
    // Released after `Drop::drop` destroys the AH.
    _dev: Arc<SrdDevice>,
}

// SAFETY: ibverbs AH handles are thread-safe to share.
unsafe impl Send for SrdAddressHandle {}
// SAFETY: see Send impl above.
unsafe impl Sync for SrdAddressHandle {}

impl SrdAddressHandle {
    /// Create an AH in the context's PD pointing at `remote_gid`, then
    /// extract the AHN via `efadv_query_ah`.
    pub fn create(ctx: &SrdContext, remote_gid: &[u8; 16]) -> io::Result<Self> {
        let mut ah_attr = IbvAhAttr {
            grh: IbvGlobalRoute {
                dgid: IbvGid { raw: *remote_gid },
                flow_label: 0,
                sgid_index: ctx.gid_index,
                hop_limit: 64,
                traffic_class: 0,
            },
            is_global: 1,
            port_num: 1,
            ..Default::default()
        };
        // SAFETY: the PD is alive; `ah_attr` is fully initialized above.
        let ah = unsafe { ibv_create_ah(ctx.dev.pd, &mut ah_attr) };
        if ah.is_null() {
            return Err(io::Error::last_os_error());
        }
        let mut efa = EfadvAhAttr::default();
        // SAFETY: `ah` is the just-created AH; `efa` is a valid out-param of
        // the size passed.
        let rc = unsafe { efadv_query_ah(ah, &mut efa, std::mem::size_of::<EfadvAhAttr>() as u32) };
        if rc != 0 {
            // SAFETY: `ah` is live and being abandoned on this error path.
            unsafe { ibv_destroy_ah(ah) };
            return Err(io::Error::other(format!("efadv_query_ah rc={rc}")));
        }
        Ok(Self {
            ah,
            ahn: efa.ahn,
            _dev: ctx.dev.clone(),
        })
    }

    pub fn ahn(&self) -> u16 {
        self.ahn
    }
    pub fn as_ptr(&self) -> *mut IbvAh {
        self.ah
    }
}

impl Drop for SrdAddressHandle {
    fn drop(&mut self) {
        // SAFETY: `self.ah` was created in `create` and is live until now.
        let rc = unsafe { ibv_destroy_ah(self.ah) };
        if rc != 0 {
            tracing::warn!(rc, "ibv_destroy_ah failed");
        }
    }
}

// ---------------------------------------------------------------------------
// SRD queue pair
// ---------------------------------------------------------------------------

/// SRD QP. Transitions `to_init` → `to_rtr` → `to_rts` before use.
pub struct SrdQp {
    qp: *mut IbvQp,
    qp_ex: *mut IbvQpEx,
    qkey: u32,
    // Released after `Drop::drop` destroys the QP.
    _cq: Arc<SrdCq>,
    _dev: Arc<SrdDevice>,
}

// SAFETY: ibverbs QP/QpEx handles are thread-safe after creation;
// posting from a single worker thread is the established invariant.
unsafe impl Send for SrdQp {}
// SAFETY: see Send impl above.
unsafe impl Sync for SrdQp {}

impl SrdQp {
    /// Create an SRD QP on the given context, using the extended-QP init path
    /// so we can post RDMA READ via the builder API.
    pub fn create(ctx: &SrdContext, cap: &IbvQpCap) -> io::Result<Self> {
        Self::create_with_qkey(ctx, cap, DEFAULT_SRD_QKEY)
    }

    pub fn create_with_qkey(ctx: &SrdContext, cap: &IbvQpCap, qkey: u32) -> io::Result<Self> {
        // SAFETY: the extended CQ is alive.
        let cq_plain = unsafe { aether_ibv_cq_ex_to_cq(ctx.cq.cq_ex) };
        let mut attr = IbvQpInitAttrEx::zeroed();
        attr.send_cq = cq_plain;
        attr.recv_cq = cq_plain;
        attr.cap = *cap;
        attr.qp_type = IBV_QPT_DRIVER;
        attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
        attr.pd = ctx.dev.pd;
        attr.send_ops_flags = IBV_QP_EX_WITH_RDMA_READ;
        let mut efa = EfadvQpInitAttr {
            comp_mask: 0,
            driver_qp_type: EFADV_QP_DRIVER_TYPE_SRD,
            flags: 0,
            sl: 0,
            reserved: 0,
        };
        // SAFETY: the context is open; `attr` and `efa` are fully
        // initialized with the sizes passed.
        let qp = unsafe {
            efadv_create_qp_ex(
                ctx.dev.context,
                &mut attr,
                &mut efa,
                std::mem::size_of::<EfadvQpInitAttr>() as u32,
            )
        };
        if qp.is_null() {
            return Err(io::Error::other(format!(
                "efadv_create_qp_ex failed: {}",
                io::Error::last_os_error()
            )));
        }
        // SAFETY: `qp` is the just-created QP.
        let qp_ex = unsafe { aether_ibv_qp_to_qp_ex(qp) };
        if qp_ex.is_null() {
            // SAFETY: `qp` is live and being abandoned on this error path.
            unsafe { super::ffi::ibv_destroy_qp(qp) };
            return Err(io::Error::other("ibv_qp_to_qp_ex failed"));
        }
        Ok(Self {
            qp,
            qp_ex,
            qkey,
            _cq: ctx.cq.clone(),
            _dev: ctx.dev.clone(),
        })
    }

    /// QP number — ship to the peer so they can target us.
    pub fn qpn(&self) -> u32 {
        // SAFETY: `self.qp` is live for this wrapper's lifetime; `qp_num` is
        // a plain field of the ibverbs-owned struct.
        unsafe { (*self.qp).qp_num }
    }

    /// Raw `ibv_qp_ex *` for callers that need to drive the builder API
    /// directly (e.g. the batched-read shim). Lifetime-tied to this `SrdQp`.
    pub fn qp_ex_ptr(&self) -> *mut IbvQpEx {
        self.qp_ex
    }

    /// Agreed-upon Q-key — ship to the peer with the endpoint.
    pub fn qkey(&self) -> u32 {
        self.qkey
    }

    /// Everything a peer needs to post RDMA READs targeting this QP:
    /// our GID + QPN + QKEY. The peer builds their own `ibv_ah` from our GID.
    pub fn endpoint(&self, ctx: &SrdContext) -> SrdEndpoint {
        SrdEndpoint {
            gid: ctx.gid(),
            qpn: self.qpn(),
            qkey: self.qkey,
        }
    }

    /// Run the full RESET → INIT → RTR → RTS state machine. SRD's RTS needs
    /// only the send-queue PSN; no timeout/retry_cnt knobs exist on SRD.
    pub fn bring_up(&self) -> io::Result<()> {
        // INIT — needs PKEY_INDEX + PORT + QKEY.
        // SAFETY: zeroed init of POD struct is sound.
        let mut attr: IbvQpAttr = unsafe { std::mem::zeroed() };
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = 1;
        attr.qkey = self.qkey;
        let mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY;
        // SAFETY: `self.qp` is live; `attr` is initialized for `mask`.
        let rc = unsafe { super::ffi::ibv_modify_qp(self.qp, &mut attr, mask) };
        if rc != 0 {
            return Err(io::Error::other(format!("SRD INIT rc={rc}")));
        }
        // RTR — bare state transition.
        // SAFETY: zeroed init of POD struct is sound.
        let mut attr: IbvQpAttr = unsafe { std::mem::zeroed() };
        attr.qp_state = IBV_QPS_RTR;
        // SAFETY: `self.qp` is live; `attr` is initialized for the mask.
        let rc = unsafe { super::ffi::ibv_modify_qp(self.qp, &mut attr, IBV_QP_STATE) };
        if rc != 0 {
            return Err(io::Error::other(format!("SRD RTR rc={rc}")));
        }
        // RTS — sq_psn only.
        // SAFETY: zeroed init of POD struct is sound.
        let mut attr: IbvQpAttr = unsafe { std::mem::zeroed() };
        attr.qp_state = IBV_QPS_RTS;
        attr.sq_psn = 0;
        // SAFETY: `self.qp` is live; `attr` is initialized for the mask.
        let rc =
            unsafe { super::ffi::ibv_modify_qp(self.qp, &mut attr, IBV_QP_STATE | IBV_QP_SQ_PSN) };
        if rc != 0 {
            return Err(io::Error::other(format!("SRD RTS rc={rc}")));
        }
        Ok(())
    }

    /// Move the QP to the error state, flushing every outstanding WR with
    /// an error completion. The QP is unusable afterwards.
    pub fn to_error(&self) -> io::Result<()> {
        // SAFETY: zeroed init of POD struct is sound.
        let mut attr: IbvQpAttr = unsafe { std::mem::zeroed() };
        attr.qp_state = IBV_QPS_ERR;
        // SAFETY: `self.qp` is live; `attr` is initialized for the mask.
        let rc = unsafe { super::ffi::ibv_modify_qp(self.qp, &mut attr, IBV_QP_STATE) };
        if rc != 0 {
            return Err(io::Error::other(format!("SRD → ERR rc={rc}")));
        }
        Ok(())
    }

    /// Post a single signaled RDMA READ on this SRD QP, addressed by the
    /// peer's AH + QPN + QKEY. Returns after `ibv_wr_complete` succeeds —
    /// the caller must still drain the CQ to observe the result.
    pub fn post_rdma_read(
        &self,
        wr_id: u64,
        ah: &SrdAddressHandle,
        remote_qpn: u32,
        remote_qkey: u32,
        local: &LocalBuf,
        remote: RemoteBuf,
    ) -> io::Result<()> {
        // SAFETY: `self.qp_ex` and `ah` are live; the NIC bounds the local
        // write by the lkey's MR, whose memory the (unsafe) registration
        // vouched for.
        let rc = unsafe {
            aether_ibv_post_rdma_read_srd(
                self.qp_ex,
                wr_id,
                IBV_SEND_SIGNALED,
                remote.rkey,
                remote.addr,
                ah.as_ptr(),
                remote_qpn,
                remote_qkey,
                local.lkey,
                local.addr,
                remote.len,
            )
        };
        if rc != 0 {
            return Err(io::Error::other(format!("post_rdma_read_srd rc={rc}")));
        }
        Ok(())
    }
}

impl Drop for SrdQp {
    fn drop(&mut self) {
        // SAFETY: `self.qp` was created in `create_with_qkey` and is live until now.
        let rc = unsafe { super::ffi::ibv_destroy_qp(self.qp) };
        if rc != 0 {
            tracing::warn!(rc, "ibv_destroy_qp failed");
        }
    }
}

// ---------------------------------------------------------------------------
// LocalBuf — small carrier for (lkey, addr). Keeps the post-site tidy.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct LocalBuf {
    pub lkey: u32,
    pub addr: u64,
}

impl LocalBuf {
    pub fn new(mr: &RegisteredMr, addr: *const u8) -> Self {
        Self {
            lkey: mr.lkey(),
            addr: addr as u64,
        }
    }
}

/// Remote-side counterpart of [`LocalBuf`]: the peer MR's rkey plus the
/// start address and byte length of the range to READ.
#[derive(Debug, Clone, Copy)]
pub struct RemoteBuf {
    pub rkey: u32,
    pub addr: u64,
    pub len: u32,
}

// ---------------------------------------------------------------------------
// Control plane (TCP) — carries the SRD advertisement + endpoint exchange
// ---------------------------------------------------------------------------

/// Serve the SRD advertisement over TCP. For each connecting client:
///   1. Receive the client's `SrdEndpoint` (GID + QPN + QKEY).
///   2. Create a local `ibv_ah` pointing at the client's GID — this
///      populates the server's EFA hardware peer table so that when the
///      server processes an incoming RDMA READ request from that client,
///      it can route the response back. Without this step, EFA rejects
///      incoming one-sided reads from an unknown peer (vendor_err=14,
///      `REMOTE_ERROR_UNKNOWN_PEER` on the client's completion).
///   3. Reply with the advertisement.
///
/// Each handshake is bounded as a whole, so a stalled client holds the
/// single-threaded accept loop for at most that long. Blocks forever; run
/// in a dedicated thread. Address handles accumulate per connected client
/// for the server's lifetime — they're small and the EFA device supports
/// 64k+ peers.
pub fn serve_srd_control_plane(
    bind_addr: &str,
    adv: &SrdAdvertisement,
    ctx: &SrdContext,
) -> io::Result<()> {
    let listener = TcpListener::bind(bind_addr)?;
    tracing::info!(addr = bind_addr, "SRD control plane listening");
    let adv_payload =
        serde_json::to_vec(adv).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    // Hold peer AHs so the underlying EFA peer entries stay valid across
    // many reads. Grows monotonically with connected clients.
    let mut peer_ahs: Vec<SrdAddressHandle> = Vec::new();
    for conn in listener.incoming() {
        let mut conn = match conn {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "accept failed");
                continue;
            }
        };
        let peer = conn.peer_addr().ok();
        let deadline = handshake_deadline();
        // 1. Receive client endpoint.
        let client_ep: SrdEndpoint = match recv_msg(&mut conn, deadline).and_then(|buf| {
            serde_json::from_slice(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        }) {
            Ok(ep) => ep,
            Err(e) => {
                tracing::warn!(?peer, error = %e, "recv client endpoint");
                continue;
            }
        };
        // 2. Create AH for the client — this populates the EFA peer table.
        match SrdAddressHandle::create(ctx, &client_ep.gid) {
            Ok(ah) => peer_ahs.push(ah),
            Err(e) => {
                tracing::warn!(?peer, error = %e, "server-side AH create");
                continue;
            }
        }
        // 3. Send advertisement.
        if let Err(e) = send_msg(&mut conn, &adv_payload, deadline) {
            tracing::warn!(?peer, error = %e, "send advertisement");
        }
    }
    Ok(())
}

/// Connect to a SRD server: send our local endpoint, then receive the
/// advertisement. Caller is expected to have an `SrdQp` already brought up
/// so the endpoint it ships is valid. The exchange is bounded as a whole.
pub fn exchange_srd_endpoints(addr: &str, local: &SrdEndpoint) -> io::Result<SrdAdvertisement> {
    let mut conn = TcpStream::connect(addr)?;
    let deadline = handshake_deadline();
    let local_buf =
        serde_json::to_vec(local).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    send_msg(&mut conn, &local_buf, deadline)?;
    let buf = recv_msg(&mut conn, deadline)?;
    serde_json::from_slice(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// ---------------------------------------------------------------------------
// SrdFeatureClient — cross-node RDMA READ gather over SRD
// ---------------------------------------------------------------------------

/// Page-aligned, zeroed heap memory the NIC writes into. Held as a raw
/// pointer, never a `Box` or slice, so no Rust reference asserts exclusive
/// access while a READ lands in it.
struct DmaBuffer {
    ptr: NonNull<u8>,
    layout: Layout,
}

// SAFETY: the buffer is uniquely owned; access is gated by `&`/`&mut`
// borrows of the client that holds it.
unsafe impl Send for DmaBuffer {}
// SAFETY: see Send impl above; shared access is read-only.
unsafe impl Sync for DmaBuffer {}

impl DmaBuffer {
    fn zeroed(len: usize) -> io::Result<Self> {
        let layout = Layout::from_size_align(len.max(1), 4096)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        // SAFETY: `layout` has nonzero size.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw)
            .ok_or_else(|| io::Error::new(io::ErrorKind::OutOfMemory, "DMA buffer allocation"))?;
        Ok(Self { ptr, layout })
    }

    fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for DmaBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated in `zeroed` with this layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// Which of the two staging regions a READ lands in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Snapshot {
    First,
    Second,
}

/// Gather features from a remote SRD server into a locally-registered MR.
///
/// The client holds its own `SrdContext` + `SrdQp` (brought to RTS), a
/// locally-allocated + registered destination buffer, and an AH aimed at
/// the server's GID. Every row is READ twice into two staging regions — the
/// second only after the first has fully completed — and accepted only when
/// both snapshots agree on an even version and on every payload byte, the
/// RDMA reader contract in `feature_table.rs`; torn rows are re-read.
/// `gather` takes `&mut self`, so no read of [`Self::dst_slice`] can overlap
/// the NIC writing it.
///
/// If a drain cannot account for every posted READ, the QP is forced into
/// the error state and the client refuses further work; should even that
/// fail, the destination buffer is leaked rather than freed under the NIC.
pub struct SrdFeatureClient {
    // Drop order: the QP stops DMA, the MR deregisters, then the buffer it
    // covered is freed.
    qp: SrdQp,
    dst_mr: RegisteredMr,
    dst: Option<DmaBuffer>,
    ah: SrdAddressHandle,
    ctx: SrdContext,
    adv: SrdAdvertisement,
    table: RemoteTable,
    /// Rows per staging region.
    max_inflight: usize,
    /// Tags each post's `wr_id`s so its completions are told from any other's.
    generation: u32,
    /// Set once the QP has been forced into the error state.
    failed: bool,
    /// Set when a drain could not prove the NIC finished; the buffer leaks.
    poisoned: bool,
}

impl Drop for SrdFeatureClient {
    fn drop(&mut self) {
        if self.poisoned
            && let Some(buf) = self.dst.take()
        {
            std::mem::forget(buf);
        }
    }
}

impl SrdFeatureClient {
    /// Connect to `addr`: open a local SRD context + QP, ship our endpoint
    /// so the server can register us as a known EFA peer (required —
    /// otherwise the server's incoming RDMA READ handler rejects us with
    /// `REMOTE_ERROR_UNKNOWN_PEER`), then receive the advertisement and
    /// build two destination regions of `max_inflight` slots each.
    ///
    /// `gid_index`: which GID slot to use locally (typically 0 on EFA).
    pub fn connect(addr: &str, gid_index: u8, max_inflight: usize) -> io::Result<Self> {
        let depth = u32::try_from(max_inflight)
            .ok()
            .filter(|&d| d > 0)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("max_inflight {max_inflight} must be in 1..=u32::MAX"),
                )
            })?;
        // Every READ is signaled and at most `max_inflight` are outstanding.
        let ctx = SrdContext::open(depth.max(16), gid_index)?;
        let cap = IbvQpCap {
            max_send_wr: depth.max(64),
            max_recv_wr: 1,
            max_send_sge: 1,
            max_recv_sge: 1,
            max_inline_data: 0,
        };
        let qp = SrdQp::create(&ctx, &cap)?;
        qp.bring_up()?;
        let local_ep = qp.endpoint(&ctx);
        let adv = exchange_srd_endpoints(addr, &local_ep)?;
        // Sanity: server advertises a qkey we can match. If not, the QP we
        // already brought up has the wrong qkey and we can't talk to this
        // server — caller must rebuild with a matching qkey.
        if adv.endpoint.qkey != local_ep.qkey {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "qkey mismatch: local {:#x} vs server {:#x}",
                    local_ep.qkey, adv.endpoint.qkey
                ),
            ));
        }
        Self::finish_connect(ctx, qp, adv, max_inflight)
    }

    /// Build a client against a pre-fetched advertisement — useful for tests
    /// or applications that own the TCP exchange themselves. Expected usage:
    /// create ctx + qp + bring_up + exchange endpoints manually, then hand
    /// the pieces here to finalize.
    pub fn from_components(
        ctx: SrdContext,
        qp: SrdQp,
        adv: SrdAdvertisement,
        max_inflight: usize,
    ) -> io::Result<Self> {
        Self::finish_connect(ctx, qp, adv, max_inflight)
    }

    fn finish_connect(
        ctx: SrdContext,
        qp: SrdQp,
        adv: SrdAdvertisement,
        max_inflight: usize,
    ) -> io::Result<Self> {
        let table = RemoteTable::parse(adv.base_addr, adv.rkey, &adv.schema)?;
        if max_inflight == 0 || u32::try_from(max_inflight).is_err() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("max_inflight {max_inflight} must be in 1..=u32::MAX"),
            ));
        }
        let ah = SrdAddressHandle::create(&ctx, &adv.endpoint.gid)?;

        // Two regions of `max_inflight` slots at the table's stride — one
        // per snapshot — in one allocation under one MR.
        let dst_len = max_inflight
            .checked_mul(table.geometry().stride())
            .and_then(|b| b.checked_mul(2))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "destination buffer size overflows usize",
                )
            })?;
        let dst = DmaBuffer::zeroed(dst_len)?;
        // SAFETY: `dst` is owned by the client, freed only after the MR
        // deregisters (field order) and never while poisoned; the NIC writes
        // it only inside `gather`, which holds `&mut self`.
        let dst_mr = unsafe { ctx.reg_mr(dst.as_ptr(), dst_len, IBV_ACCESS_LOCAL_WRITE) }?;

        Ok(Self {
            qp,
            dst_mr,
            dst: Some(dst),
            ah,
            ctx,
            adv,
            table,
            max_inflight,
            generation: 0,
            failed: false,
            poisoned: false,
        })
    }

    /// Remote schema — caller needs it to parse the returned slot bytes.
    pub fn schema(&self) -> &FeatureSchema {
        &self.adv.schema
    }

    /// The remote table, as parsed from the advertisement.
    pub fn table(&self) -> &RemoteTable {
        &self.table
    }

    fn region_ptr(&self, snap: Snapshot) -> *mut u8 {
        let base = self.dst.as_ref().expect("present until drop").as_ptr();
        let region = self.max_inflight * self.table.geometry().stride();
        match snap {
            Snapshot::First => base,
            // In bounds: the buffer holds two regions.
            Snapshot::Second => base.wrapping_add(region),
        }
    }

    /// The validated rows of the last successful `gather`, back to back at
    /// the table's slot stride: row `i` of that call at `i * slot_size`.
    pub fn dst_slice(&self) -> &[u8] {
        let len = self.max_inflight * self.table.geometry().stride();
        // SAFETY: the first region is `len` bytes inside the owned buffer;
        // no READ is in flight outside `gather`, which `&self` excludes.
        unsafe { std::slice::from_raw_parts(self.region_ptr(Snapshot::First), len) }
    }

    /// Remote slot addresses for `nodes`, each checked against the table.
    fn remote_addrs(&self, nodes: &[usize]) -> io::Result<Vec<u64>> {
        if nodes.len() > self.max_inflight {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{} nodes exceed max_inflight {}",
                    nodes.len(),
                    self.max_inflight
                ),
            ));
        }
        nodes
            .iter()
            .map(|&n| self.table.slot_addr(n as u64))
            .collect()
    }

    /// Post one READ per row in `rows` into `snap`'s region. Returns how
    /// many were accepted — each yields one completion to drain — and the
    /// provider's refusal, if it stopped short.
    fn post_snapshot(
        &mut self,
        remote: &[u64],
        rows: &[usize],
        snap: Snapshot,
    ) -> (usize, Option<io::Error>) {
        if rows.is_empty() {
            return (0, None);
        }
        let stride = self.table.geometry().stride();
        let region = self.region_ptr(snap) as u64;
        let reads: Vec<AetherSrdRead> = rows
            .iter()
            .map(|&i| AetherSrdRead {
                remote_addr: remote[i],
                local_addr: region + (i * stride) as u64,
                length: self.table.geometry().live_len(),
                _pad: 0,
            })
            .collect();
        self.generation = next_wr_generation(self.generation);
        let mut posted = 0u32;
        // SAFETY: the QP and AH are live; `reads` holds `reads.len()`
        // entries (≤ max_inflight, which fits u32); every local range is a
        // slot inside the registered buffer and every remote one a slot the
        // table vouched for.
        let rc = unsafe {
            aether_ibv_post_rdma_reads_srd_batch(
                self.qp.qp_ex_ptr(),
                self.ah.as_ptr(),
                self.adv.endpoint.qpn,
                self.adv.endpoint.qkey,
                self.table.rkey(),
                self.dst_mr.lkey(),
                u64::from(self.generation) << 32,
                reads.as_ptr(),
                reads.len() as u32,
                &mut posted,
            )
        };
        let err = (rc != 0).then(|| io::Error::other(format!("post_rdma_reads_srd_batch rc={rc}")));
        (posted as usize, err)
    }

    /// Reap exactly `posted` completions of the last post. Every READ is
    /// signaled, so the count is exact; completions tagged with another
    /// generation are strays and are not counted. Past the deadline the QP
    /// is forced into the error state, which flushes what is left; if even
    /// that leaves READs unaccounted for, the client is poisoned.
    fn drain(&mut self, posted: usize) -> io::Result<()> {
        if posted == 0 {
            return Ok(());
        }
        let tag = u64::from(self.generation);
        let mut batch = vec![AetherCqeSnapshot::default(); posted.min(256)];
        let mut reaped = 0usize;
        let mut first_err: Option<io::Error> = None;
        let mut deadline = Instant::now() + DRAIN_DEADLINE;
        let mut forced = false;
        while reaped < posted {
            let got = match self.ctx.poll_many(&mut batch) {
                Ok(n) => n,
                Err(e) => {
                    self.poisoned = true;
                    self.failed = true;
                    return Err(io::Error::other(format!("SRD CQ poll failed: {e}")));
                }
            };
            for snap in &batch[..got] {
                if snap.wr_id >> 32 != tag {
                    tracing::debug!(wr_id = snap.wr_id, "stray SRD completion");
                    continue;
                }
                reaped += 1;
                if snap.status != IBV_WC_SUCCESS && first_err.is_none() {
                    first_err = Some(io::Error::other(format!(
                        "SRD WC failed at wr_id {:#x}: status={} vendor_err={}",
                        snap.wr_id, snap.status, snap.vendor_err
                    )));
                }
            }
            if got == 0 {
                if Instant::now() >= deadline {
                    if forced {
                        self.poisoned = true;
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            format!("SRD gather reaped {reaped}/{posted} even after flushing"),
                        ));
                    }
                    // Flush the rest; each flushed READ still completes.
                    let _ = self.qp.to_error();
                    self.failed = true;
                    forced = true;
                    first_err.get_or_insert_with(|| {
                        io::Error::new(
                            io::ErrorKind::TimedOut,
                            format!(
                                "SRD gather stalled at {reaped}/{posted} after {DRAIN_DEADLINE:?}"
                            ),
                        )
                    });
                    deadline = Instant::now() + DRAIN_DEADLINE;
                }
                std::hint::spin_loop();
            }
        }
        // Completions observed; order the payload reads after them.
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        first_err.map_or(Ok(()), Err)
    }

    /// Post then drain one snapshot of `rows`, draining whatever was
    /// accepted even when the post stopped short.
    fn read_snapshot(&mut self, remote: &[u64], rows: &[usize], snap: Snapshot) -> io::Result<()> {
        let (posted, post_err) = self.post_snapshot(remote, rows, snap);
        let drained = self.drain(posted);
        match post_err {
            Some(e) => Err(e),
            None => drained,
        }
    }

    /// Rows among `rows` whose two snapshots disagree or show no stable
    /// published version.
    fn torn_rows(&self, rows: &[usize]) -> Vec<usize> {
        let g = self.table.geometry();
        let stride = g.stride();
        let region = self.max_inflight * stride;
        let payload = g.feature_offset()..g.feature_offset() + g.feature_dim() * 4;
        // SAFETY: the first region lies in the owned buffer, and no READ is
        // in flight (every post was drained before this is called).
        let s1 = unsafe { std::slice::from_raw_parts(self.region_ptr(Snapshot::First), region) };
        // SAFETY: as above, for the second region.
        let s2 = unsafe { std::slice::from_raw_parts(self.region_ptr(Snapshot::Second), region) };
        let version = |slot: &[u8], at: usize| {
            u64::from_le_bytes(slot[at..at + 8].try_into().expect("8 bytes"))
        };
        rows.iter()
            .copied()
            .filter(|&i| {
                let a = &s1[i * stride..(i + 1) * stride];
                let b = &s2[i * stride..(i + 1) * stride];
                let v = version(a, 0);
                let consistent = v == version(a, g.tail_offset())
                    && v == version(b, 0)
                    && v == version(b, g.tail_offset())
                    && aethergraph_core::cpu_seqlock_accept(v, v);
                !(consistent && a[payload.clone()] == b[payload.clone()])
            })
            .collect()
    }

    /// Read the `nodes` slots from the server, landing them back-to-back in
    /// the local destination buffer ([`Self::dst_slice`]). Returns once every
    /// row has two agreeing snapshots, or errors after `MAX_RETRIES` rounds
    /// of re-reading torn rows. Every node id is checked before anything is
    /// posted.
    pub fn gather(&mut self, nodes: &[usize]) -> io::Result<()> {
        if self.failed {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "SRD QP failed on an earlier gather; reconnect",
            ));
        }
        let remote = self.remote_addrs(nodes)?;
        let mut rows: Vec<usize> = (0..nodes.len()).collect();
        for _ in 0..=MAX_RETRIES {
            // Snapshot 2 is posted only after snapshot 1 fully completed.
            self.read_snapshot(&remote, &rows, Snapshot::First)?;
            self.read_snapshot(&remote, &rows, Snapshot::Second)?;
            rows = self.torn_rows(&rows);
            if rows.is_empty() {
                return Ok(());
            }
        }
        Err(io::Error::other(format!(
            "{} rows still torn after {MAX_RETRIES} re-reads",
            rows.len()
        )))
    }
}

// ---------------------------------------------------------------------------
// SrdShardedFeatureClient — N independent (context, QP, AH, MR) shards
// driven in parallel so N doorbells can be in flight simultaneously. Single
// QP hits ~4.3 GB/s on c6gn.16xlarge; 4-shard pool reaches ~7 GB/s (adapter's
// shared-resource ceiling; further shards plateau).
// ---------------------------------------------------------------------------

/// Multi-QP SRD pool. Each shard is an independent `SrdFeatureClient`
/// (its own context, QP, AH, MR) assembled via its own TCP + bidirectional
/// endpoint exchange with the server. `gather(nodes)` splits `nodes` evenly
/// across shards, then runs each snapshot phase as post-everywhere then
/// drain-everywhere on the calling thread, so all shards' DMA is on the wire
/// concurrently.
pub struct SrdShardedFeatureClient {
    shards: Vec<SrdFeatureClient>,
    max_inflight_per_shard: usize,
}

impl SrdShardedFeatureClient {
    /// Connect `num_shards` independent shards to `addr`. Each shard has
    /// `max_inflight_per_shard` destination slots.
    pub fn connect(
        addr: &str,
        gid_index: u8,
        num_shards: usize,
        max_inflight_per_shard: usize,
    ) -> io::Result<Self> {
        if num_shards == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "num_shards must be > 0",
            ));
        }
        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            shards.push(SrdFeatureClient::connect(
                addr,
                gid_index,
                max_inflight_per_shard,
            )?);
        }
        Ok(Self {
            shards,
            max_inflight_per_shard,
        })
    }

    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    pub fn schema(&self) -> &FeatureSchema {
        self.shards[0].schema()
    }

    /// Read `shard_idx`'s piece of the last gather. The logical result of a
    /// gather is the concatenation of `shard_dst(0)..shard_dst(N-1)` in
    /// shard order, matching how `gather` sliced the node list.
    pub fn shard_dst(&self, shard_idx: usize) -> &[u8] {
        self.shards[shard_idx].dst_slice()
    }

    /// Post `snap` for each shard's pending rows, then drain every shard
    /// that posted anything — even after a post error, the READs other
    /// shards (and a short-stopped shard) put on the wire must be reaped.
    fn read_snapshot_all(
        &mut self,
        remote: &[Vec<u64>],
        rows: &[Vec<usize>],
        snap: Snapshot,
    ) -> io::Result<()> {
        let mut first_err: Option<io::Error> = None;
        let mut posted = vec![0usize; self.shards.len()];
        for (s, shard) in self.shards.iter_mut().enumerate() {
            let (n, err) = shard.post_snapshot(&remote[s], &rows[s], snap);
            posted[s] = n;
            if let Some(e) = err {
                first_err.get_or_insert(e);
            }
        }
        for (shard, &n) in self.shards.iter_mut().zip(&posted) {
            if let Err(e) = shard.drain(n) {
                first_err.get_or_insert(e);
            }
        }
        first_err.map_or(Ok(()), Err)
    }

    /// Split `nodes` evenly across shards and read every shard's slice with
    /// the two-snapshot protocol, re-reading torn rows. Posting is
    /// non-blocking, so all shards' DMA runs concurrently from the calling
    /// thread — no per-gather OS thread spawn/join, which costs tens of
    /// microseconds and is on the order of the gather itself.
    pub fn gather(&mut self, nodes: &[usize]) -> io::Result<()> {
        if nodes.is_empty() {
            return Ok(());
        }
        if let Some(i) = self.shards.iter().position(|s| s.failed) {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                format!("SRD shard {i} failed on an earlier gather; reconnect"),
            ));
        }
        let n = self.shards.len();
        let chunk = nodes.len().div_ceil(n);
        if chunk > self.max_inflight_per_shard {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "shard slice {chunk} > max_inflight_per_shard {}",
                    self.max_inflight_per_shard
                ),
            ));
        }
        let slices: Vec<&[usize]> = (0..n)
            .map(|i| {
                let start = (i * chunk).min(nodes.len());
                &nodes[start..(start + chunk).min(nodes.len())]
            })
            .collect();
        let remote: Vec<Vec<u64>> = self
            .shards
            .iter()
            .zip(&slices)
            .map(|(shard, slice)| shard.remote_addrs(slice))
            .collect::<io::Result<_>>()?;

        let mut rows: Vec<Vec<usize>> = slices.iter().map(|s| (0..s.len()).collect()).collect();
        for _ in 0..=MAX_RETRIES {
            self.read_snapshot_all(&remote, &rows, Snapshot::First)?;
            self.read_snapshot_all(&remote, &rows, Snapshot::Second)?;
            for (shard, pending) in self.shards.iter().zip(rows.iter_mut()) {
                *pending = shard.torn_rows(pending);
            }
            if rows.iter().all(Vec::is_empty) {
                return Ok(());
            }
        }
        let torn: usize = rows.iter().map(Vec::len).sum();
        Err(io::Error::other(format!(
            "{torn} rows still torn after {MAX_RETRIES} re-reads"
        )))
    }
}
