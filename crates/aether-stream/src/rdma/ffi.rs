//! ibverbs FFI bindings — inline `extern "C"` matching project style.
//!
//! No `rdma-sys` dependency. These bindings cover the subset needed for:
//! - Device enumeration and context
//! - Protection domains
//! - Completion queues
//! - Reliable Connected (RC) queue pairs
//! - Memory registration
//! - RDMA READ posting and completion polling
//!
//! Links against `libibverbs` at link time.

#![allow(non_camel_case_types)]

// Re-export IbvMr and IbvPd from aether-mem so there's one canonical definition.
pub use aether_mem::hooks::rdma::{IbvMr, IbvPd};

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// ibverbs device (from device list).
#[repr(C)]
pub struct IbvDevice {
    _opaque: [u8; 0],
}

/// ibverbs context (opened device).
#[repr(C)]
pub struct IbvContext {
    _opaque: [u8; 0],
}

/// ibverbs completion queue.
#[repr(C)]
pub struct IbvCq {
    _opaque: [u8; 0],
}

/// ibverbs shared receive queue.
#[repr(C)]
pub struct IbvSrq {
    _opaque: [u8; 0],
}

/// ibverbs completion event channel.
///
/// Non-opaque: `struct ibv_comp_channel` is public ABI in verbs.h, and
/// the `fd` field is the whole point — it's what event loops poll.
#[repr(C)]
pub struct IbvCompChannel {
    pub context: *mut IbvContext,
    pub fd: i32,
    pub refcnt: i32,
}

/// SRQ attributes (matches `struct ibv_srq_attr`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IbvSrqAttr {
    pub max_wr: u32,
    pub max_sge: u32,
    pub srq_limit: u32,
}

/// SRQ creation attributes (matches `struct ibv_srq_init_attr`).
#[repr(C)]
pub struct IbvSrqInitAttr {
    pub srq_context: *mut libc::c_void,
    pub attr: IbvSrqAttr,
}

/// ibverbs queue pair — non-opaque layout through `qp_num`.
///
/// The fields before `qp_num` must match `struct ibv_qp` in rdma-core
/// (libibverbs/verbs.h) so that the `qp_num` offset is correct.
#[repr(C)]
pub struct IbvQp {
    pub context: *mut IbvContext,
    pub qp_context: *mut libc::c_void,
    pub pd: *mut IbvPd,
    pub send_cq: *mut IbvCq,
    pub recv_cq: *mut IbvCq,
    pub srq: *mut libc::c_void,
    pub handle: u32,
    pub qp_num: u32,
    _opaque: [u8; 0],
}

// ABI drift guard: `qp_num` is read by offset from the pointer `ibv_create_qp`
// returns. Six 8-byte pointers + `handle` (u32) place it at byte 52.
const _: () = assert!(core::mem::offset_of!(IbvQp, qp_num) == 52);

// ABI drift guards for the re-exported `IbvMr` — `lkey()`/`rkey()` read these
// by offset from the `ibv_reg_mr` pointer. `context`+`pd`+`addr`+`length`
// (4×8) then `handle` (u32) place `lkey` at 36 and `rkey` at 40.
const _: () = assert!(core::mem::offset_of!(IbvMr, lkey) == 36);
const _: () = assert!(core::mem::offset_of!(IbvMr, rkey) == 40);

// ---------------------------------------------------------------------------
// QP init attributes
// ---------------------------------------------------------------------------

/// Queue pair capabilities.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IbvQpCap {
    pub max_send_wr: u32,
    pub max_recv_wr: u32,
    pub max_send_sge: u32,
    pub max_recv_sge: u32,
    pub max_inline_data: u32,
}

/// Queue pair init attributes.
#[repr(C)]
pub struct IbvQpInitAttr {
    pub qp_context: *mut libc::c_void,
    pub send_cq: *mut IbvCq,
    pub recv_cq: *mut IbvCq,
    pub srq: *mut libc::c_void, // unused, null
    pub cap: IbvQpCap,
    pub qp_type: u32, // IBV_QPT_RC
    pub sq_sig_all: i32,
}

// ---------------------------------------------------------------------------
// QP modify attributes
// ---------------------------------------------------------------------------

/// Global routing header for RoCE/IB.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IbvGlobalRoute {
    pub dgid: IbvGid,
    pub flow_label: u32,
    pub sgid_index: u8,
    pub hop_limit: u8,
    pub traffic_class: u8,
}

/// Address handle attributes (routing info for the remote QP).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IbvAhAttr {
    pub grh: IbvGlobalRoute,
    pub dlid: u16,
    pub sl: u8,
    pub src_path_bits: u8,
    pub static_rate: u8,
    pub is_global: u8,
    pub port_num: u8,
}

/// QP attributes for `ibv_modify_qp`. Layout matches `struct ibv_qp_attr` in
/// rdma-core — total size 144 bytes (asserted at compile time). The kernel
/// reads the full struct, so a short Rust struct = read past allocation
/// (UB). Zero-init and set only the fields named in the attr_mask.
#[repr(C)]
pub struct IbvQpAttr {
    pub qp_state: u32,
    pub cur_qp_state: u32,
    pub path_mtu: u32,
    pub path_mig_state: u32,
    pub qkey: u32,
    pub rq_psn: u32,
    pub sq_psn: u32,
    pub dest_qp_num: u32,
    pub qp_access_flags: u32,
    pub cap: IbvQpCap,
    pub ah_attr: IbvAhAttr,
    pub alt_ah_attr: IbvAhAttr,
    pub pkey_index: u16,
    pub alt_pkey_index: u16,
    pub en_sqd_async_notify: u8,
    pub sq_draining: u8,
    pub max_rd_atomic: u8,
    pub max_dest_rd_atomic: u8,
    pub min_rnr_timer: u8,
    pub port_num: u8,
    pub timeout: u8,
    pub retry_cnt: u8,
    pub rnr_retry: u8,
    pub alt_port_num: u8,
    pub alt_timeout: u8,
    pub rate_limit: u32,
    /// Trailing pad to match C struct size (144 bytes). The C struct ends
    /// with rate_limit but pads to 8-byte alignment; without this pad the
    /// kernel reads 4 bytes past our allocation on every `ibv_modify_qp`.
    _trailing_pad: u32,
}

const _: [(); 144] = [(); std::mem::size_of::<IbvQpAttr>()];
const _: [(); 32] = [(); std::mem::size_of::<IbvAhAttr>()];
const _: [(); 24] = [(); std::mem::size_of::<IbvGlobalRoute>()];

// ---------------------------------------------------------------------------
// Send work request
// ---------------------------------------------------------------------------

/// Scatter/gather entry for RDMA operations.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IbvSge {
    pub addr: u64,
    pub length: u32,
    pub lkey: u32,
}

/// RDMA-specific work request fields (union member of `wr`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IbvSendWrRdma {
    pub remote_addr: u64,
    pub rkey: u32,
}

/// Atomic-specific work request fields (union member of `wr`).
///
/// For FETCH_ADD, `compare_add` is the addend and `swap` is ignored; for
/// CMP_AND_SWP, `compare_add` is the expected value and `swap` replaces
/// it on match. Both operate on exactly 8 bytes at an 8-byte-aligned
/// `remote_addr`, and the original remote value lands in the WR's SGE.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IbvSendWrAtomic {
    pub remote_addr: u64,
    pub compare_add: u64,
    pub swap: u64,
    pub rkey: u32,
}

/// Send work request.
///
/// Layout matches `struct ibv_send_wr` in <infiniband/verbs.h> exactly. The
/// trailing 56 bytes after `wr` cover the C struct's `qp_type` (xrc) and the
/// unnamed bind_mw/tso union — unused for RDMA READ but the rxe/mlx5 drivers
/// memcpy / read past `wr` for some opcodes, so total size MUST be 128 bytes
/// or the kernel reads past our allocation (UB).
///
/// Verified via `sizeof(struct ibv_send_wr) == 128` on rdma-core / Ubuntu 24.04.
#[repr(C)]
pub struct IbvSendWr {
    pub wr_id: u64,
    pub next: *mut IbvSendWr,
    pub sg_list: *mut IbvSge,
    pub num_sge: i32,
    pub opcode: u32,
    pub send_flags: u32,
    /// imm_data / invalidate_rkey union — unused for RDMA READ.
    pub imm_data: u32,
    pub wr: IbvSendWrUnion,
    /// Padding to match the C struct's full size (128 bytes total). Covers
    /// qp_type (xrc) + bind_mw/tso unions we never set; MUST stay zeroed.
    _trailing_pad: [u8; 56],
}

/// Union for work request type-specific fields.
///
/// The atomic variant is the largest at 28 bytes (padded to 32).
/// Using a struct instead of a union would make IbvSendWr too small,
/// corrupting chained WR traversal in ibv_post_send.
#[repr(C)]
#[derive(Clone, Copy)]
pub union IbvSendWrUnion {
    pub rdma: IbvSendWrRdma,
    pub atomic: IbvSendWrAtomic,
    /// Pad to the size of the largest variant (atomic: remote_addr + rkey + compare_add + swap).
    _atomic_pad: [u8; 32],
}

// ABI drift guard: `struct ibv_send_wr` is 128 bytes on rdma-core (LP64). A
// short Rust mirror makes `ibv_post_send` read past our allocation for opcodes
// that touch the trailing unions. Pin the size so any layout change is a build
// error, not silent memory corruption.
const _: () = assert!(core::mem::size_of::<IbvSendWr>() == 128);
// The wr union must cover the largest variant (atomic = 28 bytes, padded to 32).
const _: () = assert!(core::mem::size_of::<IbvSendWrUnion>() == 32);

/// Receive work request.
///
/// Layout matches `struct ibv_recv_wr` in <infiniband/verbs.h>. A
/// zero-SGE recv (`num_sge = 0`, null `sg_list`) is the sentinel shape
/// used to consume WRITE_WITH_IMM notifications — the payload arrives
/// via RDMA, only the immediate needs a queued recv.
#[repr(C)]
pub struct IbvRecvWr {
    pub wr_id: u64,
    pub next: *mut IbvRecvWr,
    pub sg_list: *mut IbvSge,
    pub num_sge: i32,
}

// ABI drift guard: `struct ibv_recv_wr` is 32 bytes on rdma-core (LP64) —
// 28 bytes of fields padded to pointer alignment.
const _: () = assert!(core::mem::size_of::<IbvRecvWr>() == 32);

// ---------------------------------------------------------------------------
// Work completion
// ---------------------------------------------------------------------------

/// Work completion entry from `ibv_poll_cq`. All fields are plain integers,
/// so the derived `Default` (all-zero) is a valid pre-poll placeholder.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct IbvWc {
    pub wr_id: u64,
    pub status: u32,
    pub opcode: u32,
    pub vendor_err: u32,
    pub byte_len: u32,
    pub imm_data: u32, // or invalidated_rkey
    pub qp_num: u32,
    pub src_qp: u32,
    pub wc_flags: u32,
    pub pkey_index: u16,
    pub slid: u16,
    pub sl: u8,
    pub dlid_path_bits: u8,
}

// ABI drift guard: `struct ibv_wc` is u64 + 8×u32 + 2×u16 + 2×u8 = 46 bytes,
// rounded to 48 by the u64's 8-byte alignment. `ibv_poll_cq` writes one full
// `ibv_wc` per completion into our array, so an undersized mirror = the kernel
// writing past our buffer.
const _: () = assert!(core::mem::size_of::<IbvWc>() == 48);

// ---------------------------------------------------------------------------
// Port attributes (partial — only fields we need)
// ---------------------------------------------------------------------------

/// Port attributes from `ibv_query_port`. Layout matches rdma-core's
/// `struct ibv_port_attr` exactly — current size is 56 bytes (verified via
/// `sizeof` on Ubuntu 24.04 rdma-core). The `active_speed_ex` trailing field
/// was added in newer rdma-core; without it the kernel writes past our
/// allocation on every `ibv_query_port` call (4-byte stack overrun, UB).
#[repr(C)]
pub struct IbvPortAttr {
    pub state: u32,
    pub max_mtu: u32,
    pub active_mtu: u32,
    pub gid_tbl_len: i32,
    pub port_cap_flags: u32,
    pub max_msg_sz: u32,
    pub bad_pkey_cntr: u32,
    pub qkey_viol_cntr: u32,
    pub pkey_tbl_len: u16,
    pub lid: u16,
    pub sm_lid: u16,
    pub lmc: u8,
    pub max_vl_num: u8,
    pub sm_sl: u8,
    pub subnet_timeout: u8,
    pub init_type_reply: u8,
    pub active_width: u8,
    pub active_speed: u8,
    pub phys_state: u8,
    pub link_layer: u8,
    pub flags: u8,
    pub port_cap_flags2: u16,
    pub active_speed_ex: u32,
}

// ABI drift guard: `struct ibv_port_attr` is 56 bytes on current rdma-core
// (the trailing `active_speed_ex` sits at offset 52 after 2 bytes of padding
// behind `port_cap_flags2`). `ibv_query_port` writes the full struct, so a
// short mirror = a stack overrun on every query.
const _: () = assert!(core::mem::size_of::<IbvPortAttr>() == 56);

// ---------------------------------------------------------------------------
// GID
// ---------------------------------------------------------------------------

/// 16-byte GID (Global Identifier) for RoCE / IB routing.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IbvGid {
    pub raw: [u8; 16],
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// QP states
pub const IBV_QPS_RESET: u32 = 0;
pub const IBV_QPS_INIT: u32 = 1;
pub const IBV_QPS_RTR: u32 = 2;
pub const IBV_QPS_RTS: u32 = 3;

// QP type
pub const IBV_QPT_RC: u32 = 2;

// Work request opcodes (matches enum ibv_wr_opcode in <infiniband/verbs.h>).
// Verified against verbs.h on rdma-core (Ubuntu 24.04). RDMA_READ is the
// 5th variant, value 4 — not 1.
pub const IBV_WR_RDMA_WRITE: u32 = 0;
pub const IBV_WR_RDMA_WRITE_WITH_IMM: u32 = 1;
pub const IBV_WR_SEND: u32 = 2;
pub const IBV_WR_SEND_WITH_IMM: u32 = 3;
pub const IBV_WR_RDMA_READ: u32 = 4;
pub const IBV_WR_ATOMIC_CMP_AND_SWP: u32 = 5;
pub const IBV_WR_ATOMIC_FETCH_AND_ADD: u32 = 6;

// Work completion opcodes (matches enum ibv_wc_opcode). Send-side
// completions mirror the WR kind; receive-side completions have the
// IBV_WC_RECV bit (1 << 7) set.
pub const IBV_WC_RDMA_WRITE: u32 = 1;
pub const IBV_WC_RDMA_READ: u32 = 2;
pub const IBV_WC_COMP_SWAP: u32 = 3;
pub const IBV_WC_FETCH_ADD: u32 = 4;
pub const IBV_WC_RECV: u32 = 1 << 7;
pub const IBV_WC_RECV_RDMA_WITH_IMM: u32 = (1 << 7) | 1;

// Device atomic capability levels (matches enum ibv_atomic_cap).
// HCA = atomic w.r.t. other HCA operations only; GLOB = also atomic
// w.r.t. CPU accesses on the responder.
pub const IBV_ATOMIC_NONE: i32 = 0;
pub const IBV_ATOMIC_HCA: i32 = 1;
pub const IBV_ATOMIC_GLOB: i32 = 2;

// Send flags (matches enum ibv_send_flags). SIGNALED must be bit 1 (=2);
// using IBV_SEND_SOLICITED (bit 2) here instead suppresses completion
// generation and masks RDMA bugs as missing CQEs.
pub const IBV_SEND_FENCE: u32 = 1 << 0;
pub const IBV_SEND_SIGNALED: u32 = 1 << 1;
pub const IBV_SEND_SOLICITED: u32 = 1 << 2;
pub const IBV_SEND_INLINE: u32 = 1 << 3;

// Access flags (matches enum ibv_access_flags). Order matters — swapping
// REMOTE_WRITE and REMOTE_READ silently grants writers the ability to
// mutate the region when the caller asked for read-only.
pub const IBV_ACCESS_LOCAL_WRITE: i32 = 1 << 0;
pub const IBV_ACCESS_REMOTE_WRITE: i32 = 1 << 1;
pub const IBV_ACCESS_REMOTE_READ: i32 = 1 << 2;
pub const IBV_ACCESS_REMOTE_ATOMIC: i32 = 1 << 3;
/// On-demand paging: the MR faults pages as the HCA touches them instead
/// of pinning the whole range at registration.
pub const IBV_ACCESS_ON_DEMAND: i32 = 1 << 6;

// ODP capability masks from `ibv_query_device_ex` (enum
// ibv_odp_general_caps and enum ibv_odp_transport_cap_bits).
pub const IBV_ODP_SUPPORT: u64 = 1 << 0;
pub const IBV_ODP_SUPPORT_IMPLICIT: u64 = 1 << 1;
pub const IBV_ODP_SUPPORT_SEND: u32 = 1 << 0;
pub const IBV_ODP_SUPPORT_RECV: u32 = 1 << 1;
pub const IBV_ODP_SUPPORT_WRITE: u32 = 1 << 2;
pub const IBV_ODP_SUPPORT_READ: u32 = 1 << 3;
pub const IBV_ODP_SUPPORT_ATOMIC: u32 = 1 << 4;

// QP attribute masks for ibv_modify_qp
pub const IBV_QP_STATE: i32 = 1 << 0;
pub const IBV_QP_CUR_STATE: i32 = 1 << 1;
pub const IBV_QP_EN_SQD_ASYNC_NOTIFY: i32 = 1 << 2;
pub const IBV_QP_ACCESS_FLAGS: i32 = 1 << 3;
pub const IBV_QP_PKEY_INDEX: i32 = 1 << 4;
pub const IBV_QP_PORT: i32 = 1 << 5;
pub const IBV_QP_QKEY: i32 = 1 << 6;
pub const IBV_QP_AV: i32 = 1 << 7;
pub const IBV_QP_PATH_MTU: i32 = 1 << 8;
pub const IBV_QP_TIMEOUT: i32 = 1 << 9;
pub const IBV_QP_RETRY_CNT: i32 = 1 << 10;
pub const IBV_QP_RNR_RETRY: i32 = 1 << 11;
pub const IBV_QP_RQ_PSN: i32 = 1 << 12;
pub const IBV_QP_MAX_QP_RD_ATOMIC: i32 = 1 << 13;
pub const IBV_QP_ALT_PATH: i32 = 1 << 14;
pub const IBV_QP_MIN_RNR_TIMER: i32 = 1 << 15;
pub const IBV_QP_SQ_PSN: i32 = 1 << 16;
pub const IBV_QP_MAX_DEST_RD_ATOMIC: i32 = 1 << 17;
pub const IBV_QP_DEST_QPN: i32 = 1 << 20;

// Path MTU
pub const IBV_MTU_4096: u32 = 5;

// Work completion status
pub const IBV_WC_SUCCESS: u32 = 0;

// ---------------------------------------------------------------------------
// FFI functions
// ---------------------------------------------------------------------------

#[link(name = "ibverbs")]
unsafe extern "C" {
    // Device enumeration
    pub fn ibv_get_device_list(num_devices: *mut i32) -> *mut *mut IbvDevice;
    pub fn ibv_free_device_list(list: *mut *mut IbvDevice);
    pub fn ibv_get_device_name(device: *mut IbvDevice) -> *const libc::c_char;
    pub fn ibv_open_device(device: *mut IbvDevice) -> *mut IbvContext;
    pub fn ibv_close_device(context: *mut IbvContext) -> i32;

    // Protection domain
    pub fn ibv_alloc_pd(context: *mut IbvContext) -> *mut IbvPd;
    pub fn ibv_dealloc_pd(pd: *mut IbvPd) -> i32;

    // Completion queue
    pub fn ibv_create_cq(
        context: *mut IbvContext,
        cqe: i32,
        cq_context: *mut libc::c_void,
        channel: *mut libc::c_void,
        comp_vector: i32,
    ) -> *mut IbvCq;
    pub fn ibv_destroy_cq(cq: *mut IbvCq) -> i32;

    // Queue pair
    pub fn ibv_create_qp(pd: *mut IbvPd, qp_init_attr: *mut IbvQpInitAttr) -> *mut IbvQp;
    pub fn ibv_destroy_qp(qp: *mut IbvQp) -> i32;
    pub fn ibv_modify_qp(qp: *mut IbvQp, attr: *mut IbvQpAttr, attr_mask: i32) -> i32;

    // Memory registration
    pub fn ibv_reg_mr(
        pd: *mut IbvPd,
        addr: *mut libc::c_void,
        length: usize,
        access: i32,
    ) -> *mut IbvMr;
    // dma-buf registration — a real exported symbol since rdma-core v34
    // (Ubuntu 22.04 ships v39). `offset` is the byte offset of the region
    // inside the dma-buf; `iova` is the virtual address the MR's range
    // starts at from the HCA's point of view.
    pub fn ibv_reg_dmabuf_mr(
        pd: *mut IbvPd,
        offset: u64,
        length: usize,
        iova: u64,
        fd: i32,
        access: i32,
    ) -> *mut IbvMr;
    pub fn ibv_dereg_mr(mr: *mut IbvMr) -> i32;

    // Data path: linked via the C shim in build.rs because the upstream
    // ibv_post_send / ibv_poll_cq are static inline in <infiniband/verbs.h>
    // and are NOT exported as ABI symbols from libibverbs.so.
    #[link_name = "aether_ibv_post_send"]
    pub fn ibv_post_send(qp: *mut IbvQp, wr: *mut IbvSendWr, bad_wr: *mut *mut IbvSendWr) -> i32;
    #[link_name = "aether_ibv_poll_cq"]
    pub fn ibv_poll_cq(cq: *mut IbvCq, num_entries: i32, wc: *mut IbvWc) -> i32;
    #[link_name = "aether_ibv_post_recv"]
    pub fn ibv_post_recv(qp: *mut IbvQp, wr: *mut IbvRecvWr, bad_wr: *mut *mut IbvRecvWr) -> i32;
    // Shimmed so Rust never mirrors the large `struct ibv_device_attr`.
    pub fn aether_ibv_query_atomic_caps(
        context: *mut IbvContext,
        atomic_cap: *mut i32,
        max_qp_rd_atom: *mut i32,
    ) -> i32;
    // Shimmed so Rust never mirrors `struct ibv_device_attr_ex`.
    pub fn aether_ibv_query_odp_caps(
        context: *mut IbvContext,
        general_caps: *mut u64,
        rc_odp_caps: *mut u32,
    ) -> i32;

    // Shared receive queue (real ABI symbols apart from the inline post).
    pub fn ibv_create_srq(pd: *mut IbvPd, attr: *mut IbvSrqInitAttr) -> *mut IbvSrq;
    pub fn ibv_destroy_srq(srq: *mut IbvSrq) -> i32;
    #[link_name = "aether_ibv_post_srq_recv"]
    pub fn ibv_post_srq_recv(
        srq: *mut IbvSrq,
        wr: *mut IbvRecvWr,
        bad_wr: *mut *mut IbvRecvWr,
    ) -> i32;

    // Completion event channel (real ABI symbols apart from req_notify).
    pub fn ibv_create_comp_channel(context: *mut IbvContext) -> *mut IbvCompChannel;
    pub fn ibv_destroy_comp_channel(channel: *mut IbvCompChannel) -> i32;
    pub fn ibv_get_cq_event(
        channel: *mut IbvCompChannel,
        cq: *mut *mut IbvCq,
        cq_context: *mut *mut libc::c_void,
    ) -> i32;
    pub fn ibv_ack_cq_events(cq: *mut IbvCq, nevents: u32);
    #[link_name = "aether_ibv_req_notify_cq"]
    pub fn ibv_req_notify_cq(cq: *mut IbvCq, solicited_only: i32) -> i32;

    // Port / GID queries
    pub fn ibv_query_port(
        context: *mut IbvContext,
        port_num: u8,
        port_attr: *mut IbvPortAttr,
    ) -> i32;
    pub fn ibv_query_gid(
        context: *mut IbvContext,
        port_num: u8,
        index: i32,
        gid: *mut IbvGid,
    ) -> i32;
}

#[cfg(feature = "mlx5dv")]
unsafe extern "C" {
    // mlx5 direct-verbs capability probe (csrc/mlx5dv_shim.c). Returns
    // ENOTSUP on non-mlx5 devices.
    pub fn aether_mlx5dv_query(
        context: *mut IbvContext,
        max_dynamic_bfregs: *mut u32,
        flags: *mut u64,
    ) -> i32;
}
