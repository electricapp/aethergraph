//! RDMA device context: device + protection domain + completion queue.
//!
//! `RdmaContext` opens the device and provides memory registration; QPs are
//! created from it. Every verbs object created on a device holds an `Arc` to
//! the device that owns the context and PD, and a QP holds its CQs, so
//! teardown runs child-before-parent regardless of which handle drops first.

use super::ffi::*;
use std::any::Any;
use std::ffi::CStr;
use std::io;
use std::ptr;
use std::sync::Arc;
use tracing::debug;

/// How a port reaches its peers, from `ibv_port_attr.link_layer`.
///
/// This decides how an address handle is built, and the two fabrics want
/// opposite things: InfiniBand routes on LIDs assigned by a subnet manager
/// and adds a Global Route Header only to leave the subnet, while RoCE has
/// no LIDs at all and requires a GRH on every packet.
///
/// Reading it from the port rather than inferring it matters because the
/// obvious inference is wrong. "The peer sent a non-zero GID, so this must
/// be RoCE" holds for RoCE and EFA, but IB ports also have GIDs — a subnet
/// prefix plus the port GUID — so that guess forces a 40-byte GRH onto
/// every packet of an intra-subnet IB link that should route on its LID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkLayer {
    /// Native InfiniBand: LID-routed within a subnet, GRH across subnets.
    InfiniBand,
    /// RoCE (and SoftRoCE): GID-routed, GRH always, LID is meaningless.
    Ethernet,
}

impl LinkLayer {
    /// Parse `ibv_port_attr.link_layer`.
    ///
    /// `IBV_LINK_LAYER_UNSPECIFIED` is reported by some drivers (and by
    /// EFA) that are nonetheless GID-routed, so anything that is not
    /// explicitly InfiniBand is treated as Ethernet — the addressing that
    /// carries a GRH, which is the safe direction to guess wrong in.
    pub fn from_port_attr(link_layer: u8) -> Self {
        // From <infiniband/verbs.h>: UNSPECIFIED = 0, INFINIBAND = 1,
        // ETHERNET = 2.
        const IBV_LINK_LAYER_INFINIBAND: u8 = 1;
        if link_layer == IBV_LINK_LAYER_INFINIBAND {
            Self::InfiniBand
        } else {
            Self::Ethernet
        }
    }

    /// Whether an address handle on this fabric needs a Global Route
    /// Header to reach `remote_gid` from `local_gid`.
    ///
    /// RoCE always needs one. InfiniBand needs one only to cross a subnet,
    /// which the top 8 bytes of the GID — the subnet prefix — identify; a
    /// peer on the same subnet is reached by LID alone.
    pub fn needs_grh(self, local_gid: &[u8; 16], remote_gid: &[u8; 16]) -> bool {
        match self {
            Self::Ethernet => true,
            Self::InfiniBand => local_gid[..8] != remote_gid[..8],
        }
    }
}

/// An open verbs device and its protection domain.
///
/// Held behind an `Arc` by every object created on it (MRs, CQs, QPs, SRQs,
/// completion channels), so the PD is deallocated and the device closed only
/// after the last of them is destroyed.
pub(crate) struct Device {
    context: *mut IbvContext,
    pd: *mut IbvPd,
}

// SAFETY: ibverbs contexts and PDs are thread-safe after creation; the
// pointers are only dereferenced by libibverbs.
unsafe impl Send for Device {}
// SAFETY: see Send impl above.
unsafe impl Sync for Device {}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: every object created on this PD holds an `Arc<Device>`, so
        // none is left; the PD was allocated in `open_on_device`.
        let rc = unsafe { ibv_dealloc_pd(self.pd) };
        if rc != 0 {
            tracing::warn!(rc, "ibv_dealloc_pd failed");
        }
        // SAFETY: the context was opened in `open_on_device` and nothing
        // created on it survives.
        let rc = unsafe { ibv_close_device(self.context) };
        if rc != 0 {
            tracing::warn!(rc, "ibv_close_device failed");
        }
    }
}

/// An opened RDMA port: device, protection domain, default completion queue,
/// and the port's addressing.
pub struct RdmaContext {
    dev: Arc<Device>,
    cq: RegisteredCq,
    /// How this port addresses peers, read from the port at open.
    pub link_layer: LinkLayer,
    /// Local port LID (for IB fabrics; 0 for RoCE).
    pub port_lid: u16,
    /// Local port GID (for RoCE routing).
    pub port_gid: IbvGid,
    /// Index of `port_gid` in the device's GID table — must be passed to the
    /// remote QP's address handle (sgid_index) so packets carry the right SGID.
    pub gid_index: u8,
    /// `ibv_port_attr.active_mtu` (an `IBV_MTU_*` code), the largest path
    /// MTU the link carries.
    active_mtu: u32,
    /// NUMA node of the opened device, read from sysfs at open time.
    /// `None` when sysfs doesn't expose it (VMs, SoftRoCE).
    numa_node: Option<i32>,
}

/// Device atomic capabilities, from `ibv_query_device`.
#[derive(Debug, Clone, Copy)]
pub struct AtomicCaps {
    /// One of `IBV_ATOMIC_NONE` / `IBV_ATOMIC_HCA` / `IBV_ATOMIC_GLOB`.
    pub atomic_cap: i32,
    /// Max outstanding READ/atomic operations per QP as initiator.
    pub max_qp_rd_atom: i32,
}

/// Device on-demand-paging capabilities, from `ibv_query_device_ex`.
#[derive(Debug, Clone, Copy)]
pub struct OdpCaps {
    /// `IBV_ODP_SUPPORT` / `IBV_ODP_SUPPORT_IMPLICIT` mask.
    pub general_caps: u64,
    /// Per-verb `IBV_ODP_SUPPORT_*` mask for RC transport.
    pub rc_odp_caps: u32,
}

impl OdpCaps {
    /// Whether ODP registration works at all.
    pub fn supported(&self) -> bool {
        self.general_caps & IBV_ODP_SUPPORT != 0
    }

    /// Whether whole-address-space registration works
    /// ([`RdmaContext::reg_mr_implicit_odp`]).
    pub fn implicit(&self) -> bool {
        self.general_caps & IBV_ODP_SUPPORT_IMPLICIT != 0
    }

    /// Whether RC RDMA READ may target an ODP MR — the bit the gather
    /// path needs. [`RdmaContext::reg_feature_mr`] uses this under
    /// [`FeatureMrPolicy::Auto`].
    pub fn rc_read(&self) -> bool {
        self.rc_odp_caps & IBV_ODP_SUPPORT_READ != 0
    }
}

/// How [`RdmaContext::reg_feature_mr`] registers a feature-table (or other
/// long-lived host) region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeatureMrPolicy {
    /// Prefer range ODP when [`OdpCaps::rc_read`] is set (fast register,
    /// HCA faults pages on first touch). Otherwise pin and
    /// [`touch_mr_pages`] so bring-up pays the pin cost, not the first gather.
    #[default]
    Auto,
    /// Always `ibv_reg_mr` pin + touch every page.
    Pinned,
    /// Require range ODP; error if the device cannot RC-READ an ODP MR.
    Odp,
}

/// Which registration path [`RdmaContext::reg_feature_mr`] took.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureMrKind {
    /// Fully pinned MR; pages were touched at register time.
    Pinned,
    /// On-demand MR (`IBV_ACCESS_ON_DEMAND`); first HCA touch faults pages.
    Odp,
}

/// Result of [`RdmaContext::reg_feature_mr`].
pub struct FeatureMr {
    pub mr: RegisteredMr,
    pub kind: FeatureMrKind,
}

/// Touch every page in `[addr, addr + len)` so a freshly pinned MR pays its
/// soft-fault / pin cost at register time instead of on the first RDMA READ.
///
/// # Safety
/// `[addr, addr + len)` must be a readable mapping owned by the caller for
/// the duration of this call.
pub unsafe fn touch_mr_pages(addr: *mut u8, len: usize) {
    if addr.is_null() || len == 0 {
        return;
    }
    const PAGE: usize = 4096;
    let mut off = 0usize;
    while off < len {
        // `off < len`, so this stays inside the caller's mapping.
        let page = addr.wrapping_add(off);
        // SAFETY: `page` is in-bounds of the caller's live mapping.
        unsafe {
            std::ptr::read_volatile(page);
        }
        off = off.saturating_add(PAGE);
    }
}

/// Current `RLIMIT_MEMLOCK` soft limit, or `None` if unlimited / unreadable.
pub fn memlock_soft_limit_bytes() -> Option<usize> {
    let mut rlim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `rlim` is a valid out-pointer.
    let rc = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rlim) };
    if rc != 0 {
        return None;
    }
    if rlim.rlim_cur == libc::RLIM_INFINITY {
        return None;
    }
    Some(rlim.rlim_cur as usize)
}

/// What the kernel's `ib_umem_get` charges a pinned registration against:
/// nothing when the process holds `CAP_IPC_LOCK`, otherwise the pages it has
/// already pinned (`VmPin`) plus the new ones, under `RLIMIT_MEMLOCK`.
#[derive(Debug, Clone, Copy, Default)]
struct PinState {
    cap_ipc_lock: bool,
    pinned_bytes: usize,
}

impl PinState {
    /// Parse `CapEff` and `VmPin` from `/proc/self/status` text.
    fn parse(status: &str) -> Self {
        const CAP_IPC_LOCK: u32 = 14;
        let mut state = Self::default();
        for line in status.lines() {
            if let Some(v) = line.strip_prefix("CapEff:") {
                if let Ok(bits) = u64::from_str_radix(v.trim(), 16) {
                    state.cap_ipc_lock = (bits >> CAP_IPC_LOCK) & 1 == 1;
                }
            } else if let Some(v) = line.strip_prefix("VmPin:")
                && let Some(kb) = v
                    .split_whitespace()
                    .next()
                    .and_then(|n| n.parse::<usize>().ok())
            {
                state.pinned_bytes = kb.saturating_mul(1024);
            }
        }
        state
    }

    fn current() -> Self {
        std::fs::read_to_string("/proc/self/status")
            .map(|s| Self::parse(&s))
            .unwrap_or_default()
    }
}

fn memlock_verdict(len: usize, limit: Option<usize>, state: PinState) -> io::Result<()> {
    let Some(limit) = limit else {
        return Ok(());
    };
    if state.cap_ipc_lock || state.pinned_bytes.saturating_add(len) <= limit {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::PermissionDenied,
        format!(
            "pinned RDMA registration needs {len} bytes on top of {} already pinned, but \
             RLIMIT_MEMLOCK soft limit is {limit}; run `ulimit -l unlimited` (or raise memlock \
             in limits.conf, or grant CAP_IPC_LOCK) before registering large tables",
            state.pinned_bytes
        ),
    ))
}

/// Fail early when a pinned registration of `len` bytes cannot fit under
/// the process memlock limit — the usual silent `ibv_reg_mr` death on HPC
/// nodes that still have the default `ulimit -l`. Mirrors the kernel's own
/// charge: exempt under `CAP_IPC_LOCK`, cumulative with pages already pinned.
pub fn check_memlock_for(len: usize) -> io::Result<()> {
    memlock_verdict(len, memlock_soft_limit_bytes(), PinState::current())
}

fn enrich_reg_mr_error(err: io::Error, len: usize) -> io::Error {
    let kind = err.kind();
    let base = err.to_string();
    let memlock = match memlock_soft_limit_bytes() {
        None => "RLIMIT_MEMLOCK=unlimited".to_string(),
        Some(n) => format!("RLIMIT_MEMLOCK soft={n} bytes"),
    };
    io::Error::new(
        kind,
        format!(
            "ibv_reg_mr({len} bytes) failed: {base} ({memlock}); \
             if this is EPERM/ENOMEM, raise memlock (`ulimit -l unlimited`) and retry"
        ),
    )
}

/// mlx5 direct-verbs capabilities, from `mlx5dv_query_device`.
#[cfg(feature = "mlx5dv")]
#[derive(Debug, Clone, Copy)]
pub struct Mlx5Caps {
    /// Dynamic BlueFlame doorbell registers available; 0 when the device
    /// (or provider build) has none.
    pub max_dynamic_bfregs: u32,
    /// `MLX5DV_CONTEXT_FLAGS_*` word.
    pub flags: u64,
}

impl RdmaContext {
    /// Open the first available RDMA device.
    ///
    /// Convenience wrapper for `open_on_device(cq_size, 0, gid_index)`. On a
    /// multi-NIC box (typical NUMA HPC node), use `open_on_device` instead so
    /// you can pick the NIC on the same NUMA node as your GPU and worker
    /// threads — cross-NUMA RDMA pays a 2–3× latency tax on real hardware.
    pub fn open(cq_size: i32, gid_index: u8) -> io::Result<Self> {
        Self::open_on_device(cq_size, 0, gid_index)
    }

    /// Open a specific RDMA device by index in `ibv_get_device_list`.
    ///
    /// `device_index`: position in the list returned by `enumerate_devices()`.
    /// Caller is responsible for picking the device co-located with their
    /// memory + worker threads — see `device_numa_node` to query placement.
    /// No auto-balancing or fallback in the hot path.
    ///
    /// Queries port 1 for LID and the supplied `gid_index` for the GID. The
    /// caller MUST pick a routable GID — on RoCEv2 over Ethernet that is
    /// typically the IPv4-mapped GID (commonly index 1 on Linux; verify via
    /// `show_gids` or `ibv_devinfo -v`). Picking a link-local IPv6 GID
    /// (often index 0) silently produces undeliverable packets; we do NOT
    /// fall back or probe in the data path.
    ///
    /// Creates a protection domain and completion queue with `cq_size` entries.
    pub fn open_on_device(cq_size: i32, device_index: usize, gid_index: u8) -> io::Result<Self> {
        // Enumerate devices.
        let mut num_devices: i32 = 0;
        // SAFETY: ibverbs FFI; `num_devices` is a valid out-param.
        let device_list = unsafe { ibv_get_device_list(&mut num_devices) };
        // `num_devices <= 0` covers both "no devices" and a defensively-handled
        // negative count: casting a negative `i32` to `usize` below would
        // produce a huge bound and let `device_index` pass the range check.
        if device_list.is_null() || num_devices <= 0 {
            if !device_list.is_null() {
                // SAFETY: `device_list` is non-null here.
                unsafe { ibv_free_device_list(device_list) };
            }
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "no RDMA devices found",
            ));
        }
        if device_index >= num_devices as usize {
            // SAFETY: `device_list` is non-null per the check above.
            unsafe { ibv_free_device_list(device_list) };
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "device_index {} out of range ({} devices found)",
                    device_index, num_devices
                ),
            ));
        }

        // Open the requested device.
        // SAFETY: `device_index < num_devices` checked above.
        let device_slot = unsafe { device_list.add(device_index) };
        // SAFETY: `device_slot` is in-bounds; the list is non-null.
        let device = unsafe { *device_slot };
        // SAFETY: `device` is a valid device pointer from the list.
        let name_ptr = unsafe { ibv_get_device_name(device) };
        let numa_node = if name_ptr.is_null() {
            None
        } else {
            // SAFETY: `name_ptr` is a NUL-terminated string owned by
            // ibverbs, valid until the list is freed below.
            let name = unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy();
            read_numa_node(&name)
        };
        // SAFETY: `device` is a valid device pointer from the list.
        let context = unsafe { ibv_open_device(device) };
        // SAFETY: `device_list` is non-null and not yet freed.
        unsafe { ibv_free_device_list(device_list) };

        if context.is_null() {
            return Err(io::Error::other("ibv_open_device failed"));
        }

        // SAFETY: `context` is the just-opened device context.
        let pd = unsafe { ibv_alloc_pd(context) };
        if pd.is_null() {
            // SAFETY: `context` is non-null and not yet closed.
            unsafe { ibv_close_device(context) };
            return Err(io::Error::other("ibv_alloc_pd failed"));
        }
        // From here every early return unwinds through `Device`'s Drop.
        let dev = Arc::new(Device { context, pd });
        let cq = RegisteredCq::create(&dev, cq_size, None)?;

        // SAFETY: zeroed init of POD struct is sound.
        let mut port_attr: IbvPortAttr = unsafe { std::mem::zeroed() };
        // SAFETY: `context` is open; `port_attr` is a valid out-param.
        let ret = unsafe { ibv_query_port(context, 1, &mut port_attr) };
        if ret != 0 {
            return Err(io::Error::other("ibv_query_port failed"));
        }

        // Query the caller-specified GID index. No fallback or probing —
        // the caller is responsible for picking a routable GID.
        // SAFETY: zeroed init of POD struct is sound.
        let mut gid: IbvGid = unsafe { std::mem::zeroed() };
        // SAFETY: `context` is open; `gid` is a valid out-param.
        let ret = unsafe { ibv_query_gid(context, 1, gid_index as i32, &mut gid) };
        if ret != 0 {
            return Err(io::Error::other(format!(
                "ibv_query_gid({gid_index}) failed"
            )));
        }

        let link_layer = LinkLayer::from_port_attr(port_attr.link_layer);
        let ctx = Self {
            dev,
            cq,
            link_layer,
            port_lid: port_attr.lid,
            port_gid: gid,
            gid_index,
            active_mtu: port_attr.active_mtu,
            numa_node,
        };
        debug!(
            ?link_layer,
            lid = ctx.port_lid,
            gid_index,
            active_mtu = ctx.active_mtu,
            "RDMA port addressing"
        );

        // Report the device's fast-path capabilities once, at open. Which
        // of these a fabric offers decides whether the atomic, ODP, and
        // BlueFlame paths are usable at all, and a wrong assumption
        // otherwise only surfaces as a failed work request much later.
        if let Ok(caps) = ctx.device_atomic_caps() {
            debug!(
                "RDMA device atomics: atomic_cap={}, max_qp_rd_atom={}",
                caps.atomic_cap, caps.max_qp_rd_atom
            );
        }
        #[cfg(feature = "mlx5dv")]
        match ctx.mlx5_caps() {
            Ok(caps) => debug!(
                "mlx5 direct verbs: max_dynamic_bfregs={} (BlueFlame {}), flags={:#x}",
                caps.max_dynamic_bfregs,
                if caps.max_dynamic_bfregs > 0 {
                    "available"
                } else {
                    "unavailable"
                },
                caps.flags
            ),
            Err(e) => debug!("mlx5 direct verbs unavailable (non-mlx5 device?): {e}"),
        }

        Ok(ctx)
    }

    /// NUMA node of this device, as sysfs reported it at open time.
    ///
    /// Use it to co-locate served memory (`NumaBindHook`) and worker
    /// threads with the NIC; `None` when the platform doesn't expose
    /// placement (VMs, SoftRoCE).
    pub fn device_numa_node(&self) -> Option<i32> {
        self.numa_node
    }

    /// The port's active path MTU as an `IBV_MTU_*` code.
    pub fn active_mtu(&self) -> u32 {
        self.active_mtu
    }

    /// The context's default completion queue.
    pub fn cq(&self) -> &RegisteredCq {
        &self.cq
    }

    /// Raw protection-domain pointer. Using it is `unsafe`; the PD lives as
    /// long as this context or anything created on it.
    pub fn pd_ptr(&self) -> *mut IbvPd {
        self.dev.pd
    }

    /// Owner handle every child object keeps alive.
    pub(crate) fn device(&self) -> &Arc<Device> {
        &self.dev
    }

    /// On-demand-paging capabilities of the opened device.
    ///
    /// ODP registration (`IBV_ACCESS_ON_DEMAND`) skips pinning: the HCA
    /// faults pages as it touches them, trading first-touch latency for
    /// instant registration of arbitrarily large ranges. Check the
    /// per-verb bits before relying on a path — a device may fault READs
    /// but not atomics.
    ///
    /// [`RdmaContext::reg_feature_mr`] with [`FeatureMrPolicy::Auto`] is the
    /// product consumer of [`OdpCaps::rc_read`].
    pub fn odp_caps(&self) -> io::Result<OdpCaps> {
        let mut general_caps: u64 = 0;
        let mut rc_odp_caps: u32 = 0;
        // SAFETY: the context is open; both out-pointers are valid.
        let ret = unsafe {
            aether_ibv_query_odp_caps(self.dev.context, &mut general_caps, &mut rc_odp_caps)
        };
        if ret != 0 {
            return Err(io::Error::other(format!(
                "ibv_query_device_ex failed: {ret}"
            )));
        }
        Ok(OdpCaps {
            general_caps,
            rc_odp_caps,
        })
    }

    /// Register the process's entire address space as one on-demand MR.
    ///
    /// Nothing is pinned; the HCA faults pages in as work requests touch
    /// them, so one registration covers every buffer the process will
    /// ever expose — no per-buffer `reg_mr` calls, no registration cache.
    /// Requires [`OdpCaps::implicit`]; the device rejects the call
    /// otherwise.
    ///
    /// TODO(deferred): no product caller yet — exercised only by
    /// `tests/softroce_e2e.rs`. Range ODP is already the Auto path in
    /// [`Self::reg_feature_mr`]. Switching the feature server to *implicit*
    /// (whole-address-space) ODP would drop any remaining registration
    /// cache entirely, but it trades pinned-memory cost for HCA page
    /// faults on first touch, so it needs measurement on real ConnectX
    /// (rxe reports the capability without the fault behaviour that makes
    /// the trade real) before becoming the default.
    ///
    /// # Safety
    /// The MR's lkey (and rkey, if `access` grants remote rights) reaches
    /// every mapped byte of the process. Every work request posted against
    /// it must target memory that is live and not concurrently borrowed by
    /// Rust for the whole DMA, and a remote key must only be handed to
    /// peers trusted with the entire address space.
    pub unsafe fn reg_mr_implicit_odp(&self, access: i32) -> io::Result<RegisteredMr> {
        // SAFETY: null addr + SIZE_MAX length is the documented implicit-
        // ODP registration form; no memory is pinned by it, and what WRs
        // may reach through it is the caller's contract above.
        let mr = unsafe {
            ibv_reg_mr(
                self.dev.pd,
                ptr::null_mut(),
                usize::MAX,
                access | IBV_ACCESS_ON_DEMAND,
            )
        };
        if mr.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `mr` is a fresh registration on this context's PD.
        Ok(unsafe { RegisteredMr::from_raw(mr, self.dev.clone()) })
    }

    /// mlx5 direct-verbs capabilities, `Err` on non-mlx5 devices.
    ///
    /// A nonzero `max_dynamic_bfregs` means the device exposes BlueFlame
    /// doorbell registers — the MMIO fast-post path small WQEs can take
    /// on ConnectX hardware (mapped per-QP via `mlx5dv_init_obj`).
    #[cfg(feature = "mlx5dv")]
    pub fn mlx5_caps(&self) -> io::Result<Mlx5Caps> {
        let mut max_dynamic_bfregs: u32 = 0;
        let mut flags: u64 = 0;
        // SAFETY: the context is open; both out-pointers are valid.
        let ret =
            unsafe { aether_mlx5dv_query(self.dev.context, &mut max_dynamic_bfregs, &mut flags) };
        if ret != 0 {
            return Err(io::Error::other(format!("mlx5dv_query_device: {ret}")));
        }
        Ok(Mlx5Caps {
            max_dynamic_bfregs,
            flags,
        })
    }

    /// Atomic capabilities of the opened device.
    ///
    /// Consult before using `post_fetch_add` / `post_compare_swap`:
    /// `IBV_ATOMIC_NONE` devices (EFA among them) reject atomic WRs, and
    /// `IBV_ATOMIC_HCA` guarantees atomicity only against other HCA
    /// operations — CPU stores on the responder can still tear.
    pub fn device_atomic_caps(&self) -> io::Result<AtomicCaps> {
        let mut atomic_cap: i32 = 0;
        let mut max_qp_rd_atom: i32 = 0;
        // SAFETY: the context is open; both out-pointers are valid.
        let ret = unsafe {
            aether_ibv_query_atomic_caps(self.dev.context, &mut atomic_cap, &mut max_qp_rd_atom)
        };
        if ret != 0 {
            return Err(io::Error::other(format!("ibv_query_device failed: {ret}")));
        }
        Ok(AtomicCaps {
            atomic_cap,
            max_qp_rd_atom,
        })
    }

    /// Register a long-lived feature-table (or similar) host region.
    ///
    /// This is the product registration entry point for feature servers:
    /// - [`FeatureMrPolicy::Auto`]: range ODP when [`OdpCaps::rc_read`], else
    ///   pin + [`touch_mr_pages`] so huge-table bring-up does not stall the
    ///   first gather.
    /// - [`FeatureMrPolicy::Pinned`] / [`FeatureMrPolicy::Odp`]: force a path.
    ///
    /// Checks `RLIMIT_MEMLOCK` before pinning. Prefer this over raw
    /// [`Self::reg_mr`] for table advertisement.
    ///
    /// # Safety
    /// Same contract as [`Self::reg_mr`]: the range must outlive the MR and
    /// every in-flight WR that references it.
    pub unsafe fn reg_feature_mr(
        &self,
        addr: *mut u8,
        len: usize,
        access: i32,
        policy: FeatureMrPolicy,
    ) -> io::Result<FeatureMr> {
        let want_odp = match policy {
            FeatureMrPolicy::Odp => true,
            FeatureMrPolicy::Pinned => false,
            FeatureMrPolicy::Auto => self
                .odp_caps()
                .map(|c| c.supported() && c.rc_read())
                .unwrap_or(false),
        };

        if want_odp {
            // SAFETY: caller owns `[addr, addr+len)` for the MR lifetime.
            match unsafe { self.reg_mr(addr, len, access | IBV_ACCESS_ON_DEMAND) } {
                Ok(mr) => {
                    debug!(len, "feature MR registered via range ODP");
                    return Ok(FeatureMr {
                        mr,
                        kind: FeatureMrKind::Odp,
                    });
                }
                Err(e) if matches!(policy, FeatureMrPolicy::Odp) => {
                    return Err(io::Error::other(format!(
                        "ODP feature MR required but ibv_reg_mr(ON_DEMAND) failed: {e}"
                    )));
                }
                Err(e) => {
                    debug!(error = %e, "ODP feature MR failed; falling back to pinned");
                }
            }
        }

        check_memlock_for(len)?;
        // SAFETY: caller owns the range for the MR lifetime.
        let mr = unsafe { self.reg_mr(addr, len, access) }?;
        // SAFETY: same live mapping we just registered.
        unsafe { touch_mr_pages(addr, len) };
        debug!(len, "feature MR registered pinned + pages touched");
        Ok(FeatureMr {
            mr,
            kind: FeatureMrKind::Pinned,
        })
    }

    /// Register memory for RDMA access (host or GPU via nvidia-peermem).
    ///
    /// Returns an RAII `RegisteredMr` that calls `ibv_dereg_mr` on drop and
    /// keeps the PD alive until then. The caller must keep it alive while any
    /// QP holds references to its lkey/rkey.
    ///
    /// For feature-table advertisement prefer [`Self::reg_feature_mr`], which
    /// applies ODP/pin policy and memlock preflight.
    ///
    /// # Safety
    /// `[addr, addr + len)` must be a valid, registerable memory range, and it
    /// must remain valid — not freed, unmapped, or repurposed — until the
    /// returned `RegisteredMr` is dropped AND every in-flight work request
    /// referencing its lkey/rkey has completed. The MR holds no lifetime tie
    /// to the buffer; once the region is advertised, remote peers can DMA
    /// into/out of it, so violating this contract is remote reads/writes of
    /// freed memory. Work requests posted against the lkey write into the
    /// range from the NIC, so no Rust reference to the bytes a WR targets may
    /// be live while that WR is in flight.
    pub unsafe fn reg_mr(
        &self,
        addr: *mut u8,
        len: usize,
        access: i32,
    ) -> io::Result<RegisteredMr> {
        // SAFETY: the PD is alive; `addr/len/access` are the caller's contract.
        let mr = unsafe { ibv_reg_mr(self.dev.pd, addr as *mut libc::c_void, len, access) };
        if mr.is_null() {
            return Err(enrich_reg_mr_error(io::Error::last_os_error(), len));
        }
        // SAFETY: `mr` is a fresh registration on this context's PD.
        Ok(unsafe { RegisteredMr::from_raw(mr, self.dev.clone()) })
    }

    /// Register a dma-buf region for RDMA access.
    ///
    /// `offset`/`len` locate the region inside the dma-buf identified by
    /// `fd`; `iova` is the address the MR's range starts at for work
    /// requests (pass the CUDA device VA so existing address arithmetic
    /// keeps working). The fd may be closed after this returns — the MR
    /// holds its own reference to the dma-buf.
    pub fn reg_mr_dmabuf(
        &self,
        fd: i32,
        offset: u64,
        len: usize,
        iova: u64,
        access: i32,
    ) -> io::Result<RegisteredMr> {
        // SAFETY: the PD is alive; the kernel bounds `offset/len` by the
        // dma-buf, whose pages the MR pins by reference.
        let mr =
            unsafe { super::ffi::ibv_reg_dmabuf_mr(self.dev.pd, offset, len, iova, fd, access) };
        if mr.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `mr` is a fresh registration on this context's PD.
        Ok(unsafe { RegisteredMr::from_raw(mr, self.dev.clone()) })
    }

    /// Raw ibverbs context pointer. Using it is `unsafe`; the context lives
    /// as long as this `RdmaContext` or anything created on it.
    pub fn context_ptr(&self) -> *mut IbvContext {
        self.dev.context
    }
}

/// Pick the RDMA device nearest a PCI function (e.g. a GPU's
/// `0000:3b:00.0`) in the PCIe hierarchy: the one sharing the longest
/// prefix of sysfs device path — the same switch before the same root
/// port before the same socket. Ties go to the lower index. `None` when
/// sysfs cannot place the PCI function or any device.
pub fn closest_device_to_pci(pci_bus_id: &str) -> Option<usize> {
    let target = std::fs::canonicalize(format!("/sys/bus/pci/devices/{pci_bus_id}")).ok()?;
    let devices = enumerate_devices().ok()?;
    devices
        .iter()
        .filter_map(|d| {
            let path =
                std::fs::canonicalize(format!("/sys/class/infiniband/{}/device", d.name)).ok()?;
            Some((pci_path_affinity(&target, &path), d.index))
        })
        .max_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)))
        .map(|(_, index)| index)
}

/// Shared leading components of two sysfs device paths.
fn pci_path_affinity(a: &std::path::Path, b: &std::path::Path) -> usize {
    a.components()
        .zip(b.components())
        .take_while(|(x, y)| x == y)
        .count()
}

/// One discovered RDMA device — name + NUMA node.
///
/// Use this to pick `device_index` for `RdmaContext::open_on_device` so it
/// matches your worker thread / GPU NUMA placement.
#[derive(Debug, Clone)]
pub struct RdmaDeviceInfo {
    /// Index into the kernel's device list — pass directly to `open_on_device`.
    pub index: usize,
    /// Device name (e.g. `"mlx5_0"`, `"rxe0"`, `"efa_0"`).
    pub name: String,
    /// NUMA node from `/sys/class/infiniband/<name>/device/numa_node`.
    /// `None` if the sysfs file is missing/unreadable; `Some(-1)` on
    /// non-NUMA systems (kernel reports -1).
    pub numa_node: Option<i32>,
}

/// Enumerate all RDMA devices visible to the process.
///
/// Returns a Vec of `RdmaDeviceInfo` with index, name, and NUMA node — enough
/// for the caller to pick the device on the same NUMA node as their pinned
/// threads + memory. Cross-NUMA RDMA costs 2–3× latency on real hardware,
/// so the placement decision is the caller's job.
pub fn enumerate_devices() -> io::Result<Vec<RdmaDeviceInfo>> {
    let mut num_devices: i32 = 0;
    // SAFETY: ibverbs FFI; `num_devices` is a valid out-param.
    let list = unsafe { ibv_get_device_list(&mut num_devices) };
    if list.is_null() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "ibv_get_device_list returned null",
        ));
    }
    let mut out = Vec::with_capacity(num_devices as usize);
    for i in 0..num_devices as usize {
        // SAFETY: `i < num_devices`; `list` is non-null.
        let dev_slot = unsafe { list.add(i) };
        // SAFETY: `dev_slot` is in-bounds.
        let dev = unsafe { *dev_slot };
        // ibv_get_device_name is typically a real ABI symbol (not inline);
        // returns a `const char *` valid for the device list's lifetime.
        // SAFETY: `dev` is a valid device pointer from the list.
        let name_ptr = unsafe { ibv_get_device_name(dev) };
        let name = if name_ptr.is_null() {
            String::new()
        } else {
            // SAFETY: `name_ptr` is a NUL-terminated string owned by ibverbs,
            // valid for the list's lifetime.
            unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned()
        };
        let numa_node = read_numa_node(&name);
        out.push(RdmaDeviceInfo {
            index: i,
            name,
            numa_node,
        });
    }
    // SAFETY: `list` is non-null and not yet freed.
    unsafe { ibv_free_device_list(list) };
    Ok(out)
}

fn read_numa_node(dev_name: &str) -> Option<i32> {
    let path = format!("/sys/class/infiniband/{dev_name}/device/numa_node");
    std::fs::read_to_string(&path).ok()?.trim().parse().ok()
}

/// Create an additional completion queue on this context. Use this when you
/// want per-thread CQs (one CQ per shard / worker thread) so polling threads
/// don't contend on a single CQ. The CQ keeps the device alive and is
/// destroyed when its last handle (including any QP created on it) drops.
pub fn create_cq(ctx: &RdmaContext, cq_size: i32) -> io::Result<RegisteredCq> {
    RegisteredCq::create(&ctx.dev, cq_size, None)
}

/// A completion channel a CQ raises events on: the raw channel plus the
/// owner that keeps it alive at least as long as the CQ.
pub(crate) struct ChannelBinding {
    pub(crate) channel: *mut IbvCompChannel,
    pub(crate) owner: Arc<dyn Any + Send + Sync>,
}

struct CqInner {
    cq: *mut IbvCq,
    capacity: u32,
    // Drop order: the CQ is destroyed in `Drop::drop`, before the channel it
    // raises events on and the device it lives on.
    _channel: Option<Arc<dyn Any + Send + Sync>>,
    _dev: Arc<Device>,
}

// SAFETY: ibverbs CQs are thread-safe after creation.
unsafe impl Send for CqInner {}
// SAFETY: see Send impl above.
unsafe impl Sync for CqInner {}

impl Drop for CqInner {
    fn drop(&mut self) {
        // SAFETY: the CQ was created in `RegisteredCq::create`; every QP on
        // it holds a `RegisteredCq` clone, so none is left.
        let rc = unsafe { ibv_destroy_cq(self.cq) };
        if rc != 0 {
            tracing::warn!(rc, "ibv_destroy_cq failed");
        }
    }
}

/// A completion queue. Cloning shares it; QPs created on it hold a clone,
/// so it is destroyed only after them.
#[derive(Clone)]
pub struct RegisteredCq {
    inner: Arc<CqInner>,
}

impl RegisteredCq {
    pub(crate) fn create(
        dev: &Arc<Device>,
        cq_size: i32,
        channel: Option<ChannelBinding>,
    ) -> io::Result<Self> {
        let capacity = u32::try_from(cq_size)
            .ok()
            .filter(|&c| c > 0)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "cq_size must be > 0"))?;
        let raw_channel = channel
            .as_ref()
            .map_or(ptr::null_mut(), |c| c.channel as *mut libc::c_void);
        // SAFETY: the context is open for as long as `dev` lives; the channel,
        // when given, belongs to it and is kept alive by the binding's owner.
        let cq = unsafe { ibv_create_cq(dev.context, cq_size, ptr::null_mut(), raw_channel, 0) };
        if cq.is_null() {
            return Err(io::Error::other("ibv_create_cq failed"));
        }
        Ok(Self {
            inner: Arc::new(CqInner {
                cq,
                capacity,
                _channel: channel.map(|c| c.owner),
                _dev: dev.clone(),
            }),
        })
    }

    /// Raw `*mut IbvCq`. Using it is `unsafe`; it lives as long as any clone.
    #[inline]
    pub fn as_ptr(&self) -> *mut IbvCq {
        self.inner.cq
    }

    /// The depth requested at creation; the provider may round it up.
    pub fn capacity(&self) -> u32 {
        self.inner.capacity
    }

    /// Poll up to `wcs.len()` completions.
    pub fn poll(&self, wcs: &mut [IbvWc]) -> io::Result<usize> {
        let n = i32::try_from(wcs.len()).unwrap_or(i32::MAX);
        // SAFETY: the CQ is alive for `self`'s lifetime; `wcs` has room for `n`.
        let ret = unsafe { ibv_poll_cq(self.inner.cq, n, wcs.as_mut_ptr()) };
        if ret < 0 {
            return Err(io::Error::other(format!("ibv_poll_cq failed: {ret}")));
        }
        Ok(ret as usize)
    }
}

/// RAII wrapper around `*mut IbvMr` — calls `ibv_dereg_mr` on drop, then
/// releases the PD owner it holds, so the PD always outlives the MR. The
/// accessor methods (`lkey`, `rkey`, `as_ptr`) are zero-cost — the
/// underlying `IbvMr` struct is non-opaque from `aether-mem`.
pub struct RegisteredMr {
    mr: *mut IbvMr,
    // Released after `Drop::drop` deregisters the MR.
    _owner: Arc<dyn Any + Send + Sync>,
}

// SAFETY: ibverbs MR is thread-safe after creation.
unsafe impl Send for RegisteredMr {}
// SAFETY: see Send impl above.
unsafe impl Sync for RegisteredMr {}

impl RegisteredMr {
    /// Take ownership of a fresh registration.
    ///
    /// # Safety
    /// `mr` must be non-null, owned by the caller, not yet deregistered, and
    /// registered on the PD that `owner` keeps alive.
    #[inline]
    pub(crate) unsafe fn from_raw(mr: *mut IbvMr, owner: Arc<dyn Any + Send + Sync>) -> Self {
        debug_assert!(
            !mr.is_null(),
            "RegisteredMr::from_raw called with null pointer"
        );
        Self { mr, _owner: owner }
    }

    /// Local key for use as `local_lkey` in send/RDMA work requests.
    #[inline]
    pub fn lkey(&self) -> u32 {
        // SAFETY: mr is non-null for the lifetime of this wrapper and the
        // IbvMr layout has been stable since libibverbs 1.0.
        unsafe { (*self.mr).lkey }
    }

    /// Remote key — pass to peers via the control plane so they can READ/WRITE
    /// to this region.
    #[inline]
    pub fn rkey(&self) -> u32 {
        // SAFETY: see lkey().
        unsafe { (*self.mr).rkey }
    }

    /// Raw `*mut IbvMr` for callers that need the bare handle.
    #[inline]
    pub fn as_ptr(&self) -> *mut IbvMr {
        self.mr
    }
}

impl Drop for RegisteredMr {
    fn drop(&mut self) {
        // SAFETY: mr was returned by ibv_reg_mr and is owned here; its PD is
        // still alive through `_owner`.
        let rc = unsafe { ibv_dereg_mr(self.mr) };
        if rc != 0 {
            tracing::warn!(rc, "ibv_dereg_mr failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_memlock_accepts_when_under_limit_or_unlimited() {
        // Unlimited → Ok. Finite limit larger than a tiny request → Ok.
        // We only assert the helper does not spuriously fail for 1 byte.
        check_memlock_for(1).expect("1-byte registration must clear memlock check");
    }

    #[test]
    fn pin_state_reads_cap_ipc_lock_and_vm_pin() {
        let root = PinState::parse("Name:\tx\nVmPin:\t    2048 kB\nCapEff:\t000001ffffffffff\n");
        assert!(root.cap_ipc_lock);
        assert_eq!(root.pinned_bytes, 2048 * 1024);

        // Bit 14 clear: an unprivileged process.
        let user = PinState::parse("VmPin:\t0 kB\nCapEff:\t0000000000000000\n");
        assert!(!user.cap_ipc_lock);
        assert_eq!(user.pinned_bytes, 0);

        // Only CAP_IPC_LOCK set.
        assert!(PinState::parse("CapEff:\t0000000000004000\n").cap_ipc_lock);
    }

    #[test]
    fn memlock_verdict_mirrors_the_kernel_charge() {
        let unprivileged = |pinned| PinState {
            cap_ipc_lock: false,
            pinned_bytes: pinned,
        };
        // Unlimited never fails.
        assert!(memlock_verdict(usize::MAX, None, unprivileged(0)).is_ok());
        // Fits on its own.
        assert!(memlock_verdict(100, Some(100), unprivileged(0)).is_ok());
        // Fits alone but not on top of what is already pinned.
        let err = memlock_verdict(100, Some(150), unprivileged(60)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
        // CAP_IPC_LOCK is exempt from the limit entirely.
        let privileged = PinState {
            cap_ipc_lock: true,
            pinned_bytes: 1 << 40,
        };
        assert!(memlock_verdict(1 << 40, Some(64 << 10), privileged).is_ok());
    }

    #[test]
    fn pci_affinity_prefers_the_deepest_shared_bridge() {
        use std::path::Path;
        let gpu = Path::new("/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/0000:02:00.0");
        let same_switch =
            Path::new("/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/0000:03:00.0");
        let same_root = Path::new("/sys/devices/pci0000:00/0000:00:02.0/0000:04:00.0");
        let other_socket = Path::new("/sys/devices/pci0000:80/0000:80:01.0/0000:81:00.0");
        let s = pci_path_affinity(gpu, same_switch);
        let r = pci_path_affinity(gpu, same_root);
        let o = pci_path_affinity(gpu, other_socket);
        assert!(s > r && r > o, "switch {s} > root {r} > socket {o}");
    }

    #[test]
    fn enrich_reg_mr_error_mentions_ulimit() {
        let e = enrich_reg_mr_error(io::Error::from_raw_os_error(libc::EPERM), 1 << 30);
        let msg = e.to_string();
        assert!(msg.contains("ulimit") || msg.contains("MEMLOCK"), "{msg}");
    }
}
