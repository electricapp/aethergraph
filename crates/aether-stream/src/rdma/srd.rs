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
//! RESET → INIT → RTR → RTS, exchange endpoints (GID + QPN + AHN + QKEY)
//! with the peer, and post RDMA READs via `SrdQp::post_rdma_read`. The
//! actual post-and-complete dance is wrapped to a single FFI crossing by
//! the C shim in `csrc/ibv_shim.c` (`aether_ibv_post_rdma_read_srd` +
//! `aether_ibv_poll_cq_ex_one`) so Rust never chases `ibv_qp_ex`'s
//! function-pointer builder at the FFI boundary.

#![cfg(all(target_os = "linux", feature = "efa"))]

use super::context::RegisteredMr;
use super::efa_ffi::*;
use super::ffi::{
    IBV_QP_PKEY_INDEX, IBV_QP_PORT, IBV_QP_QKEY, IBV_QP_SQ_PSN, IBV_QP_STATE, IBV_QPS_INIT,
    IBV_QPS_RTR, IBV_QPS_RTS, IBV_SEND_SIGNALED, IbvAhAttr, IbvContext, IbvGid, IbvGlobalRoute,
    IbvPd, IbvQp, IbvQpAttr, IbvQpCap,
};
use crate::feature_table::FeatureSchema;
use serde::{Deserialize, Serialize};
use std::ffi::CStr;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::ptr;
use std::sync::Arc;
use std::time::Duration;

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

/// EFA device context. Opens the first visible EFA device (`rdmap*` or
/// `efa_*` depending on kernel naming) and allocates a PD + extended CQ.
pub struct SrdContext {
    context: *mut IbvContext,
    pd: *mut IbvPd,
    cq_ex: *mut IbvCqEx,
    port_gid: IbvGid,
    gid_index: u8,
}

// SAFETY: SrdContext owns ibverbs PD/CQ handles that are thread-safe to share.
unsafe impl Send for SrdContext {}
// SAFETY: see Send impl above.
unsafe impl Sync for SrdContext {}

impl SrdContext {
    /// Open the first EFA device (by index 0) and allocate a PD + extended CQ.
    /// The CQ is sized to `cq_size` CQEs, WITH `IBV_WC_EX_WITH_BYTE_LEN` so
    /// completions carry a byte count.
    pub fn open(cq_size: u32, gid_index: u8) -> io::Result<Self> {
        let mut n: i32 = 0;
        // SAFETY: ibverbs FFI; `n` is a valid out-param.
        let list = unsafe { super::ffi::ibv_get_device_list(&mut n) };
        if list.is_null() || n == 0 {
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

        let mut cq_attr = IbvCqInitAttrEx::zeroed();
        cq_attr.cqe = cq_size;
        cq_attr.wc_flags = IBV_WC_EX_WITH_BYTE_LEN;
        // SAFETY: `context` is open; `cq_attr` is initialized above.
        let cq_ex = unsafe { ibv_create_cq_ex(context, &mut cq_attr) };
        if cq_ex.is_null() {
            // SAFETY: `pd` and `context` are live, not yet freed.
            unsafe { super::ffi::ibv_dealloc_pd(pd) };
            // SAFETY: see above.
            unsafe { super::ffi::ibv_close_device(context) };
            return Err(io::Error::other("ibv_create_cq_ex"));
        }

        // SAFETY: zeroed init of POD struct is sound.
        let mut port_gid: IbvGid = unsafe { std::mem::zeroed() };
        // SAFETY: `context` is open; `port_gid` is a valid out-param.
        let rc = unsafe { super::ffi::ibv_query_gid(context, 1, gid_index as i32, &mut port_gid) };
        if rc != 0 {
            // Drop CQ + PD before bailing.
            // SAFETY: `cq_ex` is the live extended CQ created above.
            let cq_plain = unsafe { aether_ibv_cq_ex_to_cq(cq_ex) };
            // SAFETY: all handles are live, not yet freed.
            unsafe { super::ffi::ibv_destroy_cq(cq_plain) };
            // SAFETY: see above.
            unsafe { super::ffi::ibv_dealloc_pd(pd) };
            // SAFETY: see above.
            unsafe { super::ffi::ibv_close_device(context) };
            return Err(io::Error::other(format!(
                "ibv_query_gid({gid_index}) rc={rc}"
            )));
        }

        Ok(Self {
            context,
            pd,
            cq_ex,
            port_gid,
            gid_index,
        })
    }

    /// Register a memory region for local write (SGE destination) or remote
    /// read (peer source). Returns an RAII wrapper; drop before the context.
    pub fn reg_mr(&self, addr: *mut u8, len: usize, access: i32) -> io::Result<RegisteredMr> {
        // SAFETY: `self.pd` is alive; `addr/len/access` are the caller's contract.
        let mr = unsafe { super::ffi::ibv_reg_mr(self.pd, addr as *mut libc::c_void, len, access) };
        if mr.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `mr` is non-null, fresh from `ibv_reg_mr`; the wrapper
        // takes ownership and deregisters on drop.
        Ok(unsafe { RegisteredMr::__from_raw_mr(mr) })
    }

    /// Raw PD pointer — needed by callers that want to register the MR
    /// themselves or create auxiliary AHs.
    pub fn pd(&self) -> *mut IbvPd {
        self.pd
    }

    /// Raw extended-CQ pointer.
    pub fn cq_ex(&self) -> *mut IbvCqEx {
        self.cq_ex
    }

    /// Raw context pointer — used by QP creation.
    pub fn context_ptr(&self) -> *mut IbvContext {
        self.context
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
        // SAFETY: `self.cq_ex` is the live extended CQ; `out` is a valid out-param.
        let rc = unsafe { aether_ibv_poll_cq_ex_one(self.cq_ex, &mut out) };
        match rc {
            0 => Ok(Some(out)),
            2 => Ok(None), // ENOENT — empty CQ
            other => Err(io::Error::other(format!("poll_cq_ex rc={other}"))),
        }
    }
}

impl Drop for SrdContext {
    fn drop(&mut self) {
        // SAFETY: `self.cq_ex` was created in `open` and is live until now.
        let cq_plain = unsafe { aether_ibv_cq_ex_to_cq(self.cq_ex) };
        // SAFETY: handles were created in `open` and are live until now.
        unsafe { super::ffi::ibv_destroy_cq(cq_plain) };
        // SAFETY: see above.
        unsafe { super::ffi::ibv_dealloc_pd(self.pd) };
        // SAFETY: see above.
        unsafe { super::ffi::ibv_close_device(self.context) };
    }
}

// ---------------------------------------------------------------------------
// Address handle
// ---------------------------------------------------------------------------

/// Wraps an `ibv_ah` with its AHN. Drop calls `ibv_destroy_ah`.
pub struct SrdAddressHandle {
    ah: *mut IbvAh,
    ahn: u16,
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
        // SAFETY: `ctx.pd` is alive; `ah_attr` is fully initialized above.
        let ah = unsafe { ibv_create_ah(ctx.pd, &mut ah_attr) };
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
        Ok(Self { ah, ahn: efa.ahn })
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
        unsafe { ibv_destroy_ah(self.ah) };
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
        // SAFETY: `ctx.cq_ex` is the live extended CQ.
        let cq_plain = unsafe { aether_ibv_cq_ex_to_cq(ctx.cq_ex) };
        let mut attr = IbvQpInitAttrEx::zeroed();
        attr.send_cq = cq_plain;
        attr.recv_cq = cq_plain;
        attr.cap = *cap;
        attr.qp_type = IBV_QPT_DRIVER;
        attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
        attr.pd = ctx.pd;
        attr.send_ops_flags = IBV_QP_EX_WITH_RDMA_READ;
        let mut efa = EfadvQpInitAttr {
            comp_mask: 0,
            driver_qp_type: EFADV_QP_DRIVER_TYPE_SRD,
            flags: 0,
            sl: 0,
            reserved: 0,
        };
        // SAFETY: `ctx.context` is open; `attr` and `efa` are fully
        // initialized with the sizes passed.
        let qp = unsafe {
            efadv_create_qp_ex(
                ctx.context,
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
        Ok(Self { qp, qp_ex, qkey })
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
        // SAFETY: `self.qp_ex` and `ah` are live; the lkey/rkey and address
        // ranges are the caller's contract with the registered MRs.
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
        unsafe { super::ffi::ibv_destroy_qp(self.qp) };
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

/// Max serialized control-plane message we'll accept — guards against a
/// malformed length prefix triggering a multi-GB allocation.
const SRD_CONTROL_MAX_MSG: usize = 64 * 1024;
const SRD_CONTROL_TIMEOUT: Duration = Duration::from_secs(10);

fn send_len_prefixed(conn: &mut TcpStream, payload: &[u8]) -> io::Result<()> {
    // The wire prefix is a u32; a payload ≥ 4 GiB cannot be framed. Error
    // rather than truncate the length (which would desync the stream).
    let len = u32::try_from(payload.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "control message too large to frame: {} bytes",
                payload.len()
            ),
        )
    })?;
    conn.write_all(&len.to_le_bytes())?;
    conn.write_all(payload)?;
    Ok(())
}

fn recv_len_prefixed(conn: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    conn.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > SRD_CONTROL_MAX_MSG {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("message too large: {len} (max {SRD_CONTROL_MAX_MSG})"),
        ));
    }
    let mut buf = vec![0u8; len];
    conn.read_exact(&mut buf)?;
    Ok(buf)
}

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
/// Blocks forever; run in a dedicated thread. Address handles accumulate
/// per connected client for the server's lifetime — they're small and
/// the EFA device supports 64k+ peers.
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
        let _ = conn.set_read_timeout(Some(SRD_CONTROL_TIMEOUT));
        let _ = conn.set_write_timeout(Some(SRD_CONTROL_TIMEOUT));
        // 1. Receive client endpoint.
        let client_ep_buf = match recv_len_prefixed(&mut conn) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(?peer, error = %e, "recv client endpoint");
                continue;
            }
        };
        let client_ep: SrdEndpoint = match serde_json::from_slice(&client_ep_buf) {
            Ok(ep) => ep,
            Err(e) => {
                tracing::warn!(?peer, error = %e, "parse client endpoint");
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
        if let Err(e) = send_len_prefixed(&mut conn, &adv_payload) {
            tracing::warn!(?peer, error = %e, "send advertisement");
        }
    }
    Ok(())
}

/// Connect to a SRD server: send our local endpoint, then receive the
/// advertisement. Caller is expected to have an `SrdQp` already brought up
/// so the endpoint it ships is valid.
pub fn exchange_srd_endpoints(addr: &str, local: &SrdEndpoint) -> io::Result<SrdAdvertisement> {
    let mut conn = TcpStream::connect(addr)?;
    conn.set_read_timeout(Some(SRD_CONTROL_TIMEOUT))?;
    conn.set_write_timeout(Some(SRD_CONTROL_TIMEOUT))?;
    let local_buf =
        serde_json::to_vec(local).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    send_len_prefixed(&mut conn, &local_buf)?;
    let buf = recv_len_prefixed(&mut conn)?;
    serde_json::from_slice(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// ---------------------------------------------------------------------------
// SrdFeatureClient — cross-node RDMA READ gather over SRD
// ---------------------------------------------------------------------------

/// Gather features from a remote SRD server into a locally-registered MR.
///
/// The client holds its own `SrdContext` + `SrdQp` (brought to RTS), a
/// locally-allocated + registered destination buffer, and an AH aimed at
/// the server's GID. `gather` issues one RDMA READ per node — each read
/// carries the local_lkey + local_addr + server rkey + server base_addr +
/// node_offset + slot_size — and drains all completions before returning.
pub struct SrdFeatureClient {
    ctx: SrdContext,
    qp: SrdQp,
    ah: SrdAddressHandle,
    adv: SrdAdvertisement,
    /// Destination MR for read landing; kept alive for the client's lifetime.
    _dst_mr: RegisteredMr,
    dst_lkey: u32,
    dst_ptr: *mut u8,
    dst_len: usize,
}

// SAFETY: all ibverbs handles inside are thread-safe to share read-only;
// mutation on the hot path happens inside the worker thread that owns the
// `Arc<SrdFeatureClient>` dispatched by `SrdShardedFeatureClient`.
unsafe impl Send for SrdFeatureClient {}
// SAFETY: see Send impl above.
unsafe impl Sync for SrdFeatureClient {}

impl SrdFeatureClient {
    /// Connect to `addr`: open a local SRD context + QP, ship our endpoint
    /// so the server can register us as a known EFA peer (required —
    /// otherwise the server's incoming RDMA READ handler rejects us with
    /// `REMOTE_ERROR_UNKNOWN_PEER`), then receive the advertisement and
    /// build the destination MR sized for `max_inflight * slot_size`.
    ///
    /// `gid_index`: which GID slot to use locally (typically 0 on EFA).
    pub fn connect(addr: &str, gid_index: u8, max_inflight: usize) -> io::Result<Self> {
        let cq_size = (max_inflight as u32).max(16);
        let ctx = SrdContext::open(cq_size, gid_index)?;
        let cap = IbvQpCap {
            max_send_wr: (max_inflight as u32).max(64),
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
        let ah = SrdAddressHandle::create(&ctx, &adv.endpoint.gid)?;

        // Per-slot destination buffer — one contiguous region so each post's
        // SGE points at a distinct offset within a single MR.
        let dst_len = max_inflight * adv.schema.slot_size;
        let mut dst = vec![0u8; dst_len].into_boxed_slice();
        let dst_ptr = dst.as_mut_ptr();
        // Register against the still-owned allocation; its pages are valid for
        // the duration of this call. Only leak the box AFTER registration
        // succeeds so a `reg_mr` failure frees the buffer instead of leaking
        // it. On success the leaked allocation stays pinned for the client's
        // lifetime and `_dst_mr` owns the dereg via Drop.
        let dst_mr = ctx.reg_mr(dst_ptr, dst_len, super::ffi::IBV_ACCESS_LOCAL_WRITE)?;
        std::mem::forget(dst);
        let dst_lkey = dst_mr.lkey();

        Ok(Self {
            ctx,
            qp,
            ah,
            adv,
            _dst_mr: dst_mr,
            dst_lkey,
            dst_ptr,
            dst_len,
        })
    }

    /// Remote schema — caller needs it to parse the returned slot bytes.
    pub fn schema(&self) -> &FeatureSchema {
        &self.adv.schema
    }

    /// Immutable view of the destination buffer (after `gather`).
    pub fn dst_slice(&self) -> &[u8] {
        // SAFETY: dst_ptr / dst_len come from the `vec![0u8; dst_len]` we
        // built in `from_advertisement`; the allocation was leaked + pinned
        // by `reg_mr` and is freed only when the MR's Drop runs.
        unsafe { std::slice::from_raw_parts(self.dst_ptr, self.dst_len) }
    }

    /// Translate a node id into the remote address to READ from, validating it
    /// against the advertised table bounds.
    ///
    /// Two independent guards, both required:
    ///   1. `node < schema.node_count` — the table holds exactly `node_count`
    ///      logical slots.
    ///   2. `(node + 1) * slot_size <= node_count * slot_size` via checked
    ///      arithmetic — the slot's last byte must stay inside the region the
    ///      server registered. The server registers the whole feature table
    ///      (`node_count` rounded up to a power of two, each slot page-rounded),
    ///      so `node_count * slot_size` is a conservative lower bound on the
    ///      registered MR length; staying within it guarantees the one-sided
    ///      READ never targets memory outside the MR even for an out-of-range id.
    fn remote_addr_for(&self, node: usize) -> io::Result<u64> {
        let node_count = self.adv.schema.node_count as u64;
        let slot_size = self.adv.schema.slot_size as u64;
        let node = node as u64;
        if node >= node_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("node {node} out of range (node_count {node_count})"),
            ));
        }
        let end_offset = node
            .checked_add(1)
            .and_then(|n| n.checked_mul(slot_size))
            .ok_or_else(|| io::Error::other("remote offset overflow"))?;
        let region_len = node_count
            .checked_mul(slot_size)
            .ok_or_else(|| io::Error::other("region length overflow"))?;
        if end_offset > region_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("node {node} slot end {end_offset} exceeds region length {region_len}"),
            ));
        }
        let start_offset = node
            .checked_mul(slot_size)
            .ok_or_else(|| io::Error::other("remote offset overflow"))?;
        self.adv
            .base_addr
            .checked_add(start_offset)
            .ok_or_else(|| io::Error::other("remote address overflow"))
    }

    /// Post one RDMA READ per node without draining. Pair with
    /// [`drain_gather`](Self::drain_gather); [`gather`](Self::gather)
    /// composes the two. Split out so a sharded pool can post every
    /// shard's batch before draining any of them — the posts are
    /// non-blocking, so the shards' DMA runs concurrently from one thread.
    ///
    /// EFA SRD requires each WR to sit in its own `wr_start`/`wr_complete`
    /// bracket and be individually signaled, so N reads produce N CQEs.
    /// The batched shim (`aether_ibv_post_rdma_reads_srd_batch`) collapses
    /// the post loop to one FFI crossing regardless of batch size.
    pub fn post_gather(&self, nodes: &[usize]) -> io::Result<usize> {
        if nodes.is_empty() {
            return Ok(0);
        }
        let slot = self.adv.schema.slot_size as u32;
        assert!(
            nodes.len() * slot as usize <= self.dst_len,
            "caller exceeded advertised max_inflight"
        );
        // Validate every node id against the advertised table bounds before
        // turning it into a one-sided READ. An id past the table must never be
        // translated into a read outside the server's registered region.
        let reads: Vec<AetherSrdRead> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| {
                Ok(AetherSrdRead {
                    remote_addr: self.remote_addr_for(node)?,
                    local_addr: self.dst_ptr as u64 + (i * slot as usize) as u64,
                    length: slot,
                    _pad: 0,
                })
            })
            .collect::<io::Result<Vec<_>>>()?;
        // SAFETY: the QP and AH are live for `self`'s lifetime; `reads` holds
        // `reads.len()` initialized entries; lkey/rkey and every address range
        // come from registered MRs (bounds-checked via `remote_addr_for`).
        let rc = unsafe {
            aether_ibv_post_rdma_reads_srd_batch(
                self.qp.qp_ex_ptr(),
                self.ah.as_ptr(),
                self.adv.endpoint.qpn,
                self.adv.endpoint.qkey,
                self.adv.rkey,
                self.dst_lkey,
                0,
                reads.as_ptr(),
                reads.len() as u32,
            )
        };
        if rc != 0 {
            return Err(io::Error::other(format!(
                "post_rdma_reads_srd_batch rc={rc}"
            )));
        }
        Ok(nodes.len())
    }

    /// Drain `posted` completions from this shard's CQ. Every WR posted by
    /// [`post_gather`](Self::post_gather) is signaled, so the count must
    /// match — leftover CQEs would be misattributed to the next gather.
    pub fn drain_gather(&self, posted: usize) -> io::Result<()> {
        if posted == 0 {
            return Ok(());
        }
        // Drain via the batched poll shim (one FFI crossing per drain
        // burst instead of one per CQE).
        let mut batch: Vec<AetherCqeSnapshot> = vec![AetherCqeSnapshot::default(); posted];
        let mut drained = 0usize;
        let start = std::time::Instant::now();
        while drained < posted {
            let want = posted - drained;
            // SAFETY: the CQ is live for `self`'s lifetime; `batch` has room
            // for `want` snapshots (`want <= batch.len()`).
            let got = unsafe {
                aether_ibv_poll_cq_ex_many(self.ctx.cq_ex(), batch.as_mut_ptr(), want as u32)
            };
            if got < 0 {
                return Err(io::Error::other(format!("poll_cq_ex_many rc={got}")));
            }
            for snap in batch.iter().take(got as usize) {
                if snap.status != super::ffi::IBV_WC_SUCCESS {
                    return Err(io::Error::other(format!(
                        "SRD WC failed at wr_id {}: status={} vendor_err={}",
                        snap.wr_id, snap.status, snap.vendor_err
                    )));
                }
            }
            drained += got as usize;
            if got == 0 {
                if start.elapsed() > Duration::from_secs(30) {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        format!("SRD gather drained {drained}/{posted} after 30s"),
                    ));
                }
                std::hint::spin_loop();
            }
        }
        Ok(())
    }

    /// Read the `nodes` slots from the server, landing them back-to-back in
    /// the local destination buffer. Returns once every completion has
    /// drained.
    pub fn gather(&self, nodes: &[usize]) -> io::Result<()> {
        let posted = self.post_gather(nodes)?;
        self.drain_gather(posted)
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
/// across shards and runs each shard's post+drain on its own OS thread.
pub struct SrdShardedFeatureClient {
    shards: Vec<Arc<SrdFeatureClient>>,
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
        assert!(num_shards > 0);
        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            let client = SrdFeatureClient::connect(addr, gid_index, max_inflight_per_shard)?;
            shards.push(Arc::new(client));
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

    /// Split `nodes` evenly across shards, post every shard's batch, then
    /// drain each shard's CQ. Posting is non-blocking, so all shards' DMA
    /// runs concurrently — the same parallel wire time the old
    /// thread-per-shard version bought, without paying an OS thread
    /// spawn/join (tens of microseconds) per gather call that rivaled the
    /// gather itself. Panics if any single shard's slice exceeds
    /// `max_inflight_per_shard`.
    pub fn gather(&self, nodes: &[usize]) -> io::Result<()> {
        if nodes.is_empty() {
            return Ok(());
        }
        let n = self.shards.len();
        let chunk = nodes.len().div_ceil(n);
        assert!(
            chunk <= self.max_inflight_per_shard,
            "shard slice {chunk} > max_inflight_per_shard {}",
            self.max_inflight_per_shard
        );

        // Phase 1: post on every shard.
        let mut posted: Vec<usize> = Vec::with_capacity(n);
        let mut post_err: Option<io::Error> = None;
        for (i, shard) in self.shards.iter().enumerate() {
            let start = (i * chunk).min(nodes.len());
            let end = (start + chunk).min(nodes.len());
            match shard.post_gather(&nodes[start..end]) {
                Ok(count) => posted.push(count),
                Err(e) => {
                    post_err = Some(e);
                    break;
                }
            }
        }

        // Phase 2: drain every shard that posted — even on a post error,
        // earlier shards' in-flight WRs must be reaped or their CQEs would
        // corrupt the next gather's accounting.
        let mut first_err: Option<io::Error> = post_err;
        for (shard, &count) in self.shards.iter().zip(posted.iter()) {
            if let Err(e) = shard.drain_gather(count)
                && first_err.is_none()
            {
                first_err = Some(e);
            }
        }

        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
