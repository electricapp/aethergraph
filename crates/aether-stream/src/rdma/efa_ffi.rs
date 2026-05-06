//! EFA (AWS Elastic Fabric Adapter) FFI bindings for the SRD path.
//!
//! Covers the subset of `<infiniband/verbs.h>` extended verbs (qp_ex, cq_ex,
//! builder API) and `<infiniband/efadv.h>` needed to:
//!   - Query EFA device caps (confirms `RDMA_READ` is supported)
//!   - Create an SRD QP via `efadv_create_qp_ex`
//!   - Drive the builder-API RDMA READ post through the C shim
//!   - Drain one extended completion at a time through the C shim
//!
//! Links against `libibverbs` and `libefa` (the EFA provider).
//!
//! Struct sizes + offsets verified against rdma-core / efa-installer on
//! Ubuntu 24.04 arm64, kernel 6.17, libefa 1.4.61 (c7g.16xlarge, 2026-04-14).

#![allow(non_camel_case_types)]

use super::ffi::{IbvCq, IbvContext, IbvMr, IbvPd, IbvQp, IbvQpCap};

// ---------------------------------------------------------------------------
// Opaque extended-verbs handles
// ---------------------------------------------------------------------------

/// Extended completion queue (`struct ibv_cq_ex`). Opaque in Rust — the
/// builder-API accessors are reached through the C shim.
#[repr(C)]
pub struct IbvCqEx {
    _opaque: [u8; 0],
}

/// Extended queue pair (`struct ibv_qp_ex`). Opaque.
#[repr(C)]
pub struct IbvQpEx {
    _opaque: [u8; 0],
}

/// Address handle (`struct ibv_ah`). Opaque.
#[repr(C)]
pub struct IbvAh {
    _opaque: [u8; 0],
}

// ---------------------------------------------------------------------------
// Extended QP init attributes (136 bytes total, verified).
// ---------------------------------------------------------------------------

/// `struct ibv_qp_init_attr_ex` — full layout required because the driver
/// reads past the comp-mask-selected fields based on the total struct size.
/// Use `IbvQpInitAttrEx::zeroed()` to construct; pad fields must stay zero.
#[repr(C)]
pub struct IbvQpInitAttrEx {
    pub qp_context: *mut libc::c_void,
    pub send_cq: *mut IbvCq,
    pub recv_cq: *mut IbvCq,
    pub srq: *mut libc::c_void,
    pub cap: IbvQpCap,
    pub qp_type: u32,
    pub sq_sig_all: i32,
    pub comp_mask: u32,
    pub pd: *mut IbvPd,
    pub xrcd: *mut libc::c_void,
    pub create_flags: u32,
    pub max_tso_header: u16,
    _tso_pad: u16,
    pub rwq_ind_tbl: *mut libc::c_void,
    /// `struct ibv_rx_hash_conf` — 24 bytes, all zero for SRD.
    pub rx_hash_conf: [u8; 24],
    pub source_qpn: u32,
    _source_qpn_pad: u32,
    pub send_ops_flags: u64,
}

const _: [(); 136] = [(); std::mem::size_of::<IbvQpInitAttrEx>()];

impl IbvQpInitAttrEx {
    /// Zeroed-out instance — safe because every field is POD (`*mut T` pointers
    /// zero to NULL, integers zero to 0, `IbvQpCap` is `#[derive(Default)]`).
    #[inline]
    pub fn zeroed() -> Self {
        // SAFETY: all fields are plain-old-data with valid zero bit patterns.
        unsafe { std::mem::zeroed() }
    }
}

/// `struct ibv_cq_init_attr_ex` — 56 bytes.
#[repr(C)]
pub struct IbvCqInitAttrEx {
    pub cqe: u32,
    _cqe_pad: u32,
    pub cq_context: *mut libc::c_void,
    pub channel: *mut libc::c_void,
    pub comp_vector: u32,
    _comp_vector_pad: u32,
    pub wc_flags: u64,
    pub comp_mask: u32,
    pub flags: u32,
    pub parent_domain: *mut libc::c_void,
}

const _: [(); 56] = [(); std::mem::size_of::<IbvCqInitAttrEx>()];

impl IbvCqInitAttrEx {
    #[inline]
    pub fn zeroed() -> Self {
        // SAFETY: same as IbvQpInitAttrEx — all fields are POD.
        unsafe { std::mem::zeroed() }
    }
}

// ---------------------------------------------------------------------------
// EFA-specific attributes (from <infiniband/efadv.h>).
// ---------------------------------------------------------------------------

/// `struct efadv_qp_init_attr` — 16 bytes.
#[repr(C)]
#[derive(Default)]
pub struct EfadvQpInitAttr {
    pub comp_mask: u64,
    pub driver_qp_type: u32,
    pub flags: u16,
    pub sl: u8,
    pub reserved: u8,
}

const _: [(); 16] = [(); std::mem::size_of::<EfadvQpInitAttr>()];

/// `struct efadv_device_attr` — 32 bytes. `max_rdma_size` is the largest
/// payload a single RDMA READ can carry on SRD (EFA caps it below RC's
/// 2 GiB on most HCAs).
#[repr(C)]
#[derive(Default)]
pub struct EfadvDeviceAttr {
    pub comp_mask: u64,
    pub max_sq_wr: u32,
    pub max_rq_wr: u32,
    pub max_sq_sge: u16,
    pub max_rq_sge: u16,
    pub inline_buf_size: u16,
    pub reserved: [u8; 2],
    pub device_caps: u32,
    pub max_rdma_size: u32,
}

const _: [(); 32] = [(); std::mem::size_of::<EfadvDeviceAttr>()];

/// `struct efadv_ah_attr` — 16 bytes. `ahn` (Address Handle Number) is the
/// 16-bit identifier the peer must include in its post to address us.
#[repr(C)]
#[derive(Default)]
pub struct EfadvAhAttr {
    pub comp_mask: u64,
    pub ahn: u16,
    pub reserved: [u8; 6],
}

const _: [(); 16] = [(); std::mem::size_of::<EfadvAhAttr>()];

/// Snapshot of one CQE produced by the C shim, so Rust never touches the
/// extended-CQ start/end_poll state directly.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct AetherCqeSnapshot {
    pub status: u32,
    pub opcode: u32,
    pub wr_id: u64,
    pub byte_len: u32,
    pub vendor_err: u32,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Driver QP type passed to `efadv_create_qp_ex` — SRD is the only one.
pub const EFADV_QP_DRIVER_TYPE_SRD: u32 = 0;

/// QP type marker that says "this is a driver-specific QP" (SRD here).
pub const IBV_QPT_DRIVER: u32 = 255;

/// `comp_mask` bits for `ibv_qp_init_attr_ex`.
pub const IBV_QP_INIT_ATTR_PD: u32 = 1 << 0;
pub const IBV_QP_INIT_ATTR_SEND_OPS_FLAGS: u32 = 1 << 6;

/// `send_ops_flags` bits for `ibv_qp_init_attr_ex`.
pub const IBV_QP_EX_WITH_RDMA_READ: u64 = 1 << 4;

/// `wc_flags` bits for `ibv_cq_init_attr_ex`.
pub const IBV_WC_EX_WITH_BYTE_LEN: u64 = 1 << 0;

/// EFA device capability bits (`efadv_device_attr.device_caps`).
pub const EFADV_DEVICE_ATTR_CAPS_RDMA_READ: u32 = 1 << 0;

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

#[link(name = "ibverbs")]
extern "C" {
    /// Create an address handle from an `ibv_ah_attr`.
    pub fn ibv_create_ah(
        pd: *mut IbvPd,
        attr: *mut super::ffi::IbvAhAttr,
    ) -> *mut IbvAh;
    pub fn ibv_destroy_ah(ah: *mut IbvAh) -> i32;
}

// `ibv_create_cq_ex` is `static inline` in <infiniband/verbs.h> (dispatches
// through `context->ops.create_cq_ex`) so it has no ABI symbol — linked via
// the C shim instead.
extern "C" {
    #[link_name = "aether_ibv_create_cq_ex"]
    pub fn ibv_create_cq_ex(
        context: *mut IbvContext,
        attr: *mut IbvCqInitAttrEx,
    ) -> *mut IbvCqEx;
}

#[link(name = "efa")]
extern "C" {
    /// Query EFA-specific caps.
    pub fn efadv_query_device(
        ctx: *mut IbvContext,
        attr: *mut EfadvDeviceAttr,
        inlen: u32,
    ) -> i32;

    /// Query the EFA-specific attrs of an address handle — produces the AHN
    /// we ship over the control plane.
    pub fn efadv_query_ah(
        ah: *mut IbvAh,
        attr: *mut EfadvAhAttr,
        inlen: u32,
    ) -> i32;

    /// Create an SRD QP via the extended API. Unlike `ibv_create_qp_ex`, this
    /// one additionally takes the EFA-specific attrs (driver_qp_type = SRD).
    pub fn efadv_create_qp_ex(
        ctx: *mut IbvContext,
        attr_ex: *mut IbvQpInitAttrEx,
        efa_attr: *mut EfadvQpInitAttr,
        inlen: u32,
    ) -> *mut IbvQp;
}

// Linked via csrc/ibv_shim.c when built with AETHER_EFA_SHIM.
extern "C" {
    pub fn aether_ibv_cq_ex_to_cq(cqx: *mut IbvCqEx) -> *mut IbvCq;
    pub fn aether_ibv_qp_to_qp_ex(qp: *mut IbvQp) -> *mut IbvQpEx;

    /// One-shot RDMA READ on an SRD QP: wraps the
    /// `wr_start`/`wr_rdma_read`/`wr_set_ud_addr`/`wr_set_sge`/`wr_complete`
    /// sequence so Rust doesn't need to chase the builder's function
    /// pointers itself. Returns 0 on success, errno-style non-zero on failure.
    pub fn aether_ibv_post_rdma_read_srd(
        qpx: *mut IbvQpEx,
        wr_id: u64,
        wr_flags: u32,
        remote_rkey: u32,
        remote_addr: u64,
        ah: *mut IbvAh,
        remote_qpn: u32,
        remote_qkey: u32,
        local_lkey: u32,
        local_addr: u64,
        length: u32,
    ) -> i32;

    /// Post N chained RDMA READs inside ONE builder-API transaction. All
    /// reads share the same AH + (remote_qpn, remote_qkey, remote_rkey,
    /// local_lkey); per-read offset + length come from `reads`. Only the
    /// final WR is signaled, so the caller drains exactly one CQE after
    /// this call. Zero Rust ↔ C boundary crossings per WR — the whole
    /// batch posts in one FFI call. Returns 0 on success.
    pub fn aether_ibv_post_rdma_reads_srd_batch(
        qpx: *mut IbvQpEx,
        ah: *mut IbvAh,
        remote_qpn: u32,
        remote_qkey: u32,
        remote_rkey: u32,
        local_lkey: u32,
        base_wr_id: u64,
        reads: *const AetherSrdRead,
        n: u32,
    ) -> i32;

    /// Drain at most one CQE into the snapshot. Returns:
    ///   0 on success (out populated)
    ///   ENOENT (2) when CQ is empty
    ///   other errno on hard error.
    pub fn aether_ibv_poll_cq_ex_one(
        cqx: *mut IbvCqEx,
        out: *mut AetherCqeSnapshot,
    ) -> i32;

    /// Drain up to `max_out` CQEs into the array in a single FFI call.
    /// Returns the number drained (0 if CQ was empty). Uses `ibv_next_poll`
    /// inside one `start_poll`/`end_poll` bracket, so N drains cost 1 FFI
    /// crossing instead of N.
    pub fn aether_ibv_poll_cq_ex_many(
        cqx: *mut IbvCqEx,
        out: *mut AetherCqeSnapshot,
        max_out: u32,
    ) -> i32;
}

/// Per-WR descriptor for the batched RDMA READ path. Layout matches
/// `struct aether_srd_read` in `csrc/ibv_shim.c` byte-for-byte (24B).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AetherSrdRead {
    pub remote_addr: u64,
    pub local_addr: u64,
    pub length: u32,
    pub _pad: u32,
}

const _: [(); 24] = [(); std::mem::size_of::<AetherSrdRead>()];

// Keep aether_mem's IbvMr reachable for SRD memory registration.
pub use super::ffi::{IbvAhAttr, IbvGid, IbvQpAttr};
pub type IbvMrOpaque = IbvMr;
