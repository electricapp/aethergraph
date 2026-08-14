//! NUMA memory-policy helpers built directly on the kernel syscall ABI.
//!
//! No libnuma dependency: the nodemask handling is a few lines and the
//! syscall interface is stable. Every call here is fail-soft in the same
//! way `mlock` is — a refused policy call (seccomp, old kernel) leaves
//! the allocation working with default placement.
//!
//! Placement calls issue the syscall even on single-node machines, where
//! it trivially succeeds: the cost is one syscall per allocation and it
//! keeps the code path exercised everywhere.

use crate::HookError;

/// Words in a nodemask. 16 × 64 bits = 1024 nodes, comfortably above any
/// machine this code will meet, and passing the full fixed-size mask
/// sidesteps the kernel's maxnode rounding quirks.
#[cfg(target_os = "linux")]
const NODE_MASK_WORDS: usize = 16;

// Memory-policy modes and flags, from the kernel UAPI
// (include/uapi/linux/mempolicy.h). Defined locally: the values are
// stable kernel ABI.
#[cfg(target_os = "linux")]
const MPOL_PREFERRED: libc::c_int = 1;
#[cfg(target_os = "linux")]
const MPOL_BIND: libc::c_int = 2;
#[cfg(target_os = "linux")]
const MPOL_INTERLEAVE: libc::c_int = 3;
#[cfg(target_os = "linux")]
const MPOL_F_NODE: libc::c_ulong = 1 << 0;
#[cfg(target_os = "linux")]
const MPOL_F_ADDR: libc::c_ulong = 1 << 1;
#[cfg(target_os = "linux")]
const MPOL_MF_MOVE: libc::c_uint = 1 << 1;

/// Number of NUMA nodes with memory, cached after the first call.
/// Returns 1 when the topology cannot be read (including non-Linux).
pub fn num_nodes() -> usize {
    #[cfg(target_os = "linux")]
    {
        static NODES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *NODES.get_or_init(|| nodes_online().len().max(1))
    }
    #[cfg(not(target_os = "linux"))]
    1
}

/// IDs of online NUMA nodes, `[0]` when the topology cannot be read.
pub fn nodes_online() -> Vec<u32> {
    #[cfg(target_os = "linux")]
    {
        let listed = std::fs::read_to_string("/sys/devices/system/node/online")
            .ok()
            .map(|s| parse_id_list(s.trim()))
            .unwrap_or_default();
        if listed.is_empty() { vec![0] } else { listed }
    }
    #[cfg(not(target_os = "linux"))]
    vec![0]
}

/// NUMA node owning `cpu`, if the topology exposes it.
pub fn node_of_cpu(cpu: usize) -> Option<u32> {
    #[cfg(target_os = "linux")]
    {
        for node in nodes_online() {
            let path = format!("/sys/devices/system/node/node{node}/cpulist");
            let Ok(list) = std::fs::read_to_string(path) else {
                continue;
            };
            if parse_id_list(list.trim()).contains(&(cpu as u32)) {
                return Some(node);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpu;
        None
    }
}

/// CPU IDs on `node`, empty when the topology cannot be read.
pub fn cores_on_node(node: u32) -> Vec<usize> {
    #[cfg(target_os = "linux")]
    {
        let path = format!("/sys/devices/system/node/node{node}/cpulist");
        std::fs::read_to_string(path)
            .map(|s| {
                parse_id_list(s.trim())
                    .into_iter()
                    .map(|c| c as usize)
                    .collect()
            })
            .unwrap_or_default()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = node;
        Vec::new()
    }
}

/// Parse a sysfs ID list ("0-3,8-11,16") into the expanded IDs.
///
/// Malformed segments are skipped rather than failing the whole list —
/// a partially readable topology beats none.
#[cfg(any(target_os = "linux", test))]
fn parse_id_list(s: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((lo, hi)) => {
                if let (Ok(lo), Ok(hi)) = (lo.parse::<u32>(), hi.parse::<u32>())
                    && lo <= hi
                {
                    out.extend(lo..=hi);
                }
            }
            None => {
                if let Ok(v) = part.parse::<u32>() {
                    out.push(v);
                }
            }
        }
    }
    out
}

/// Bind `len` bytes at `ptr` to `node`, migrating already-faulted pages.
///
/// Ring allocations pre-fault before hooks run, so `MPOL_MF_MOVE` is what
/// makes the call meaningful — without it the policy would only govern
/// pages faulted later.
pub fn bind_region(ptr: *mut u8, len: usize, node: u32) -> Result<(), HookError> {
    #[cfg(target_os = "linux")]
    {
        mbind_region(ptr, len, MPOL_BIND, &[node])
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptr, len, node);
        Err(HookError::new("NUMA policy not supported on this platform"))
    }
}

/// Interleave `len` bytes at `ptr` page-round-robin across `nodes`,
/// migrating already-faulted pages.
pub fn interleave_region(ptr: *mut u8, len: usize, nodes: &[u32]) -> Result<(), HookError> {
    #[cfg(target_os = "linux")]
    {
        mbind_region(ptr, len, MPOL_INTERLEAVE, nodes)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptr, len, nodes);
        Err(HookError::new("NUMA policy not supported on this platform"))
    }
}

/// Prefer `node` for this thread's future page faults.
///
/// `MPOL_PREFERRED` rather than `MPOL_BIND`: a worker that outgrows its
/// node's free memory should spill to a remote node, not OOM.
pub fn prefer_current_thread(node: u32) -> Result<(), HookError> {
    #[cfg(target_os = "linux")]
    {
        let mask = nodemask(&[node]);
        // SAFETY: the mask array outlives the call; set_mempolicy reads
        // `maxnode` bits from it and touches nothing else.
        let ret = unsafe {
            libc::syscall(
                libc::SYS_set_mempolicy,
                MPOL_PREFERRED,
                mask.as_ptr(),
                (NODE_MASK_WORDS * 64) as libc::c_ulong,
            )
        };
        syscall_result(ret, "set_mempolicy")
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = node;
        Err(HookError::new("NUMA policy not supported on this platform"))
    }
}

/// Confine this thread to `node`'s cores and prefer `node` for its pages.
///
/// Memory policy alone only decides where a thread's pages land; without
/// pinning, the scheduler is free to migrate the thread to another socket
/// and every access it makes turns into a remote one. The two calls belong
/// together: affinity keeps the thread on the node, `MPOL_PREFERRED` keeps
/// its allocations there.
///
/// Fail-soft like the rest of this module. A node with no listed cores, or
/// a restricted affinity mask (cpuset, taskset), leaves the thread
/// scheduled wherever it already was.
pub fn pin_current_thread(node: u32) -> Result<(), HookError> {
    #[cfg(target_os = "linux")]
    {
        let cores = cores_on_node(node);
        if cores.is_empty() {
            return Err(HookError::new("node has no online cores"));
        }
        // SAFETY: an all-zero cpu_set_t is the documented empty mask; the
        // CPU_SET calls below only set bits within it.
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        for cpu in &cores {
            if *cpu < libc::CPU_SETSIZE as usize {
                // SAFETY: `set` is a live cpu_set_t and `cpu` is in range.
                unsafe { libc::CPU_SET(*cpu, &mut set) };
            }
        }
        // SAFETY: pid 0 is the calling thread; `set` outlives the call and
        // the kernel reads exactly `size_of::<cpu_set_t>()` bytes from it.
        let ret =
            unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
        if ret != 0 {
            return Err(HookError::new("sched_setaffinity failed"));
        }
        prefer_current_thread(node)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = node;
        Err(HookError::new("NUMA policy not supported on this platform"))
    }
}

/// Node currently backing the page at `ptr`, if the kernel exposes it.
/// The page must have been touched; unfaulted addresses report nothing
/// useful.
pub fn region_node(ptr: *const u8) -> Option<u32> {
    #[cfg(target_os = "linux")]
    {
        let mut node: libc::c_int = -1;
        // SAFETY: `node` is a valid out-pointer; MPOL_F_NODE | MPOL_F_ADDR
        // asks for the node of the page at `ptr` and writes only `node`.
        let ret = unsafe {
            libc::syscall(
                libc::SYS_get_mempolicy,
                &mut node as *mut libc::c_int,
                std::ptr::null_mut::<libc::c_ulong>(),
                0 as libc::c_ulong,
                ptr as *mut libc::c_void,
                MPOL_F_NODE | MPOL_F_ADDR,
            )
        };
        if ret == 0 && node >= 0 {
            Some(node as u32)
        } else {
            None
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = ptr;
        None
    }
}

#[cfg(target_os = "linux")]
fn nodemask(nodes: &[u32]) -> [libc::c_ulong; NODE_MASK_WORDS] {
    let mut mask = [0 as libc::c_ulong; NODE_MASK_WORDS];
    for &node in nodes {
        let idx = node as usize / 64;
        if idx < NODE_MASK_WORDS {
            mask[idx] |= 1 << (node as usize % 64);
        }
    }
    mask
}

#[cfg(target_os = "linux")]
fn mbind_region(
    ptr: *mut u8,
    len: usize,
    mode: libc::c_int,
    nodes: &[u32],
) -> Result<(), HookError> {
    let mask = nodemask(nodes);
    // SAFETY: `ptr..ptr+len` is a mapping owned by the caller; the mask
    // array outlives the call. mbind changes only the region's memory
    // policy (plus page placement under MPOL_MF_MOVE) — it never
    // invalidates the mapping.
    let ret = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            ptr as *mut libc::c_void,
            len as libc::c_ulong,
            mode,
            mask.as_ptr(),
            (NODE_MASK_WORDS * 64) as libc::c_ulong,
            MPOL_MF_MOVE,
        )
    };
    syscall_result(ret, "mbind")
}

#[cfg(target_os = "linux")]
fn syscall_result(ret: libc::c_long, name: &str) -> Result<(), HookError> {
    if ret == 0 {
        Ok(())
    } else {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        #[cfg(feature = "tracing")]
        tracing::warn!(errno, "{name} failed (non-fatal)");
        Err(HookError::with_code(format!("{name} failed"), errno))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_list_parsing() {
        assert_eq!(parse_id_list("0"), vec![0]);
        assert_eq!(parse_id_list("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_id_list("0-2,8,10-11"), vec![0, 1, 2, 8, 10, 11]);
        assert_eq!(parse_id_list(""), Vec::<u32>::new());
        // Malformed segments are skipped, valid ones kept.
        assert_eq!(parse_id_list("x,3-1,5"), vec![5]);
    }

    #[test]
    fn topology_queries_never_panic() {
        assert!(num_nodes() >= 1);
        assert!(!nodes_online().is_empty());
        let _ = node_of_cpu(0);
        let _ = cores_on_node(0);
    }

    /// Exercises the affinity + policy syscalls on node 0, which is online
    /// everywhere. A single-node CI machine still runs the whole path —
    /// the only thing it cannot show is a *choice* between nodes — and the
    /// thread must be left running on that node's cores afterwards.
    #[cfg(target_os = "linux")]
    #[test]
    fn pin_current_thread_confines_to_the_node_cores() {
        let cores = cores_on_node(0);
        if cores.is_empty() {
            println!("no cpulist for node 0; skipping");
            return;
        }
        match pin_current_thread(0) {
            Ok(()) => {
                // SAFETY: an all-zero cpu_set_t is a valid empty mask that
                // sched_getaffinity overwrites.
                let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
                // SAFETY: pid 0 is this thread; the kernel writes at most
                // `size_of::<cpu_set_t>()` bytes into `set`.
                let ret = unsafe {
                    libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set)
                };
                assert_eq!(ret, 0, "sched_getaffinity failed after a successful pin");
                // Every CPU left in the mask must belong to node 0.
                for cpu in 0..libc::CPU_SETSIZE as usize {
                    // SAFETY: `set` is live and `cpu` is within CPU_SETSIZE.
                    if unsafe { libc::CPU_ISSET(cpu, &set) } {
                        assert!(
                            cores.contains(&cpu),
                            "cpu {cpu} is set but is not on node 0"
                        );
                    }
                }
            }
            Err(e) => println!("pin unavailable here ({e}); skipping"),
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn bind_and_query_own_page() {
        // Bind a touched page to node 0 (always online) and read its
        // placement back. On a single-node machine this exercises the
        // full syscall path with a trivially satisfiable policy.
        let mut page = vec![0u8; 4096];
        page[0] = 1;
        match bind_region(page.as_mut_ptr(), page.len(), 0) {
            Ok(()) => {
                // Placement query is best-effort; when it answers, the
                // page must be where the bind put it.
                if let Some(node) = region_node(page.as_ptr()) {
                    assert_eq!(node, 0);
                }
            }
            Err(e) => println!("mbind unavailable here ({e}); skipping"),
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn thread_preference_accepts_node_zero() {
        match prefer_current_thread(0) {
            Ok(()) => {}
            Err(e) => println!("set_mempolicy unavailable here ({e}); skipping"),
        }
    }
}
