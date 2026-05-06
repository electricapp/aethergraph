//! RDMA device context: device + protection domain + completion queue.
//!
//! `RdmaContext` owns the lifetime of the ibverbs resources and provides
//! memory registration. One context per process; QPs are created from it.

use super::ffi::*;
use std::ffi::CStr;
use std::io;
use std::ptr;

/// Owns an ibverbs device context, protection domain, and completion queue.
///
/// All RDMA resources (QPs, MRs) are created through this context.
/// Dropped in reverse order: CQ → PD → device context.
pub struct RdmaContext {
    context: *mut IbvContext,
    pub pd: *mut IbvPd,
    pub cq: *mut IbvCq,
    /// Local port LID (for IB fabrics; 0 for RoCE).
    pub port_lid: u16,
    /// Local port GID (for RoCE routing).
    pub port_gid: IbvGid,
    /// Index of `port_gid` in the device's GID table — must be passed to the
    /// remote QP's address handle (sgid_index) so packets carry the right SGID.
    pub gid_index: u8,
}

// SAFETY: ibverbs resources are thread-safe after creation.
unsafe impl Send for RdmaContext {}
unsafe impl Sync for RdmaContext {}

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
        unsafe {
            // Enumerate devices
            let mut num_devices: i32 = 0;
            let device_list = ibv_get_device_list(&mut num_devices);
            if device_list.is_null() || num_devices == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "no RDMA devices found",
                ));
            }
            if device_index >= num_devices as usize {
                ibv_free_device_list(device_list);
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "device_index {} out of range ({} devices found)",
                        device_index, num_devices
                    ),
                ));
            }

            // Open the requested device
            let device = *device_list.add(device_index);
            let context = ibv_open_device(device);
            ibv_free_device_list(device_list);

            if context.is_null() {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "ibv_open_device failed",
                ));
            }

            // Allocate protection domain
            let pd = ibv_alloc_pd(context);
            if pd.is_null() {
                ibv_close_device(context);
                return Err(io::Error::new(io::ErrorKind::Other, "ibv_alloc_pd failed"));
            }

            // Create completion queue
            let cq = ibv_create_cq(context, cq_size, ptr::null_mut(), ptr::null_mut(), 0);
            if cq.is_null() {
                ibv_dealloc_pd(pd);
                ibv_close_device(context);
                return Err(io::Error::new(io::ErrorKind::Other, "ibv_create_cq failed"));
            }

            // Query port 1 for LID
            let mut port_attr: IbvPortAttr = std::mem::zeroed();
            let ret = ibv_query_port(context, 1, &mut port_attr);
            if ret != 0 {
                ibv_destroy_cq(cq);
                ibv_dealloc_pd(pd);
                ibv_close_device(context);
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "ibv_query_port failed",
                ));
            }

            // Query the caller-specified GID index. No fallback or probing —
            // the caller is responsible for picking a routable GID.
            let mut gid: IbvGid = std::mem::zeroed();
            let ret = ibv_query_gid(context, 1, gid_index as i32, &mut gid);
            if ret != 0 {
                ibv_destroy_cq(cq);
                ibv_dealloc_pd(pd);
                ibv_close_device(context);
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("ibv_query_gid({gid_index}) failed"),
                ));
            }

            Ok(Self {
                context,
                pd,
                cq,
                port_lid: port_attr.lid,
                port_gid: gid,
                gid_index,
            })
        }
    }

    /// Register memory for RDMA access (host or GPU via nvidia-peermem).
    ///
    /// Returns an RAII `RegisteredMr` that calls `ibv_dereg_mr` on drop. The
    /// caller must keep it alive while any QP holds references to its lkey/rkey,
    /// and must drop it BEFORE this `RdmaContext` (otherwise `ibv_dealloc_pd`
    /// fails with EBUSY at context drop). Storing it in the same struct as the
    /// context, declared *before* the context field, gives the right Drop order
    /// (Rust drops fields in declaration order).
    pub fn reg_mr(&self, addr: *mut u8, len: usize, access: i32) -> io::Result<RegisteredMr> {
        unsafe {
            let mr = ibv_reg_mr(self.pd, addr as *mut libc::c_void, len, access);
            if mr.is_null() {
                return Err(io::Error::last_os_error());
            }
            Ok(RegisteredMr { mr })
        }
    }

    /// Raw ibverbs context pointer (for creating QPs on this device).
    pub fn context_ptr(&self) -> *mut IbvContext {
        self.context
    }
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
    unsafe {
        let mut num_devices: i32 = 0;
        let list = ibv_get_device_list(&mut num_devices);
        if list.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "ibv_get_device_list returned null",
            ));
        }
        let mut out = Vec::with_capacity(num_devices as usize);
        for i in 0..num_devices as usize {
            let dev = *list.add(i);
            // ibv_get_device_name is typically a real ABI symbol (not inline);
            // returns a `const char *` valid for the device list's lifetime.
            let name_ptr = ibv_get_device_name(dev);
            let name = if name_ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
            };
            let numa_node = read_numa_node(&name);
            out.push(RdmaDeviceInfo {
                index: i,
                name,
                numa_node,
            });
        }
        ibv_free_device_list(list);
        Ok(out)
    }
}

fn read_numa_node(dev_name: &str) -> Option<i32> {
    let path = format!("/sys/class/infiniband/{dev_name}/device/numa_node");
    std::fs::read_to_string(&path).ok()?.trim().parse().ok()
}

impl Drop for RdmaContext {
    fn drop(&mut self) {
        unsafe {
            ibv_destroy_cq(self.cq);
            ibv_dealloc_pd(self.pd);
            ibv_close_device(self.context);
        }
    }
}

/// Create an additional completion queue on this context. Use this when you
/// want per-thread CQs (one CQ per shard / worker thread) so polling threads
/// don't contend on a single CQ. Returns an RAII `RegisteredCq` that destroys
/// the CQ on drop. Callers MUST drop the CQ before this context.
pub fn create_cq(ctx: &RdmaContext, cq_size: i32) -> io::Result<RegisteredCq> {
    unsafe {
        let cq = ibv_create_cq(ctx.context, cq_size, ptr::null_mut(), ptr::null_mut(), 0);
        if cq.is_null() {
            return Err(io::Error::new(io::ErrorKind::Other, "ibv_create_cq failed"));
        }
        Ok(RegisteredCq { cq })
    }
}

/// RAII wrapper around `*mut IbvCq`. Drop calls `ibv_destroy_cq`.
///
/// Tied by contract (not lifetime) to the `RdmaContext` that created it —
/// must be dropped before the context, otherwise `ibv_destroy_cq` would run
/// against a freed device handle.
pub struct RegisteredCq {
    cq: *mut IbvCq,
}

unsafe impl Send for RegisteredCq {}
unsafe impl Sync for RegisteredCq {}

impl RegisteredCq {
    /// Raw `*mut IbvCq` for passing to QP creation / `ibv_poll_cq`.
    #[inline]
    pub fn as_ptr(&self) -> *mut IbvCq {
        self.cq
    }
}

impl Drop for RegisteredCq {
    fn drop(&mut self) {
        unsafe {
            ibv_destroy_cq(self.cq);
        }
    }
}

/// RAII wrapper around `*mut IbvMr` — calls `ibv_dereg_mr` on drop.
///
/// Holds no lifetime to the owning `RdmaContext` (would be viral). Caller must
/// drop this before the context. The accessor methods (`lkey`, `rkey`, `as_ptr`)
/// are zero-cost — the underlying `IbvMr` struct is non-opaque from `aether-mem`.
pub struct RegisteredMr {
    mr: *mut IbvMr,
}

unsafe impl Send for RegisteredMr {}
unsafe impl Sync for RegisteredMr {}

impl RegisteredMr {
    /// Wrap an existing `ibv_reg_mr`-produced MR handle. Caller must ensure
    /// the handle is non-null, owned, and has not been deregistered. Used by
    /// alternate transport paths (SRD) that call `ibv_reg_mr` directly.
    #[inline]
    pub fn __from_raw_mr(mr: *mut IbvMr) -> Self {
        Self { mr }
    }

    /// Local key for use as `local_lkey` in send/RDMA work requests.
    #[inline]
    pub fn lkey(&self) -> u32 {
        // SAFETY: mr is non-null for the lifetime of this wrapper (set in reg_mr).
        unsafe { (*self.mr).lkey }
    }

    /// Remote key — pass to peers via the control plane so they can READ/WRITE
    /// to this region.
    #[inline]
    pub fn rkey(&self) -> u32 {
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
        // SAFETY: mr was returned by ibv_reg_mr in reg_mr; we own it.
        unsafe {
            ibv_dereg_mr(self.mr);
        }
    }
}
