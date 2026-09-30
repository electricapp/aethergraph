//! NUMA placement policy for the core's worker pools.
//!
//! The graph body is interleaved across nodes at load
//! ([`hint::with_interleaved_placement`](super::hint::with_interleaved_placement)),
//! which spreads bandwidth but leaves every access a coin flip between
//! local and remote. Pinning the pool completes the other half: a worker
//! that stays on one node keeps its own allocations — landing buffers,
//! frontier vectors, the subgraph it builds — local to the socket reading
//! them, and stops the scheduler from migrating it mid-batch and stranding
//! every one of those buffers across the interconnect.
//!
//! Placement stays inside the CPUs the process was already given. A worker
//! inherits its spawner's affinity (a `taskset`, `numactl`, or launcher
//! binding), and only nodes with CPUs in that set are candidates; a
//! process confined to one node is left alone. Workers are spread
//! round-robin over the candidates starting from the spawner's own node,
//! so concurrent unbound ranks don't all put worker 0 on node 0.
//!
//! Everything here is fail-soft. A single-node machine, a kernel without
//! the policy syscalls, or an affinity mask spanning one node all leave the
//! pool scheduled exactly as it was.

/// NUMA node the calling thread is running on, when it can be read.
#[cfg(all(target_os = "linux", feature = "numa"))]
pub fn current_node() -> Option<u32> {
    // SAFETY: sched_getcpu has no preconditions.
    let cpu = unsafe { libc::sched_getcpu() };
    usize::try_from(cpu)
        .ok()
        .and_then(aether_mem::numa::node_of_cpu)
}

#[cfg(not(all(target_os = "linux", feature = "numa")))]
pub fn current_node() -> Option<u32> {
    None
}

/// Pin worker `index` of a pool to one NUMA node's share of the thread's
/// inherited CPUs and prefer that node for its allocations. `home` is the
/// spawning thread's node (from [`current_node`]), where round-robin starts.
/// Returns the node it was pinned to, or `None` when placement did not
/// apply.
#[cfg(all(target_os = "linux", feature = "numa"))]
pub fn pin_worker(index: usize, home: Option<u32>) -> Option<u32> {
    // SAFETY: an all-zero cpu_set_t is the documented empty mask.
    let mut allowed: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: pid 0 is the calling thread; `allowed` is a valid out-pointer
    // of exactly the size passed.
    let rc =
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut allowed) };
    if rc != 0 {
        return None;
    }
    let is_allowed = |cpu: usize| {
        if cpu >= libc::CPU_SETSIZE as usize {
            return false;
        }
        // SAFETY: `allowed` is a live cpu_set_t and `cpu` is in range.
        unsafe { libc::CPU_ISSET(cpu, &allowed) }
    };
    let mut candidates: Vec<(u32, Vec<usize>)> = aether_mem::numa::nodes_online()
        .into_iter()
        .filter_map(|node| {
            let cores: Vec<usize> = aether_mem::numa::cores_on_node(node)
                .into_iter()
                .filter(|&cpu| is_allowed(cpu))
                .collect();
            (!cores.is_empty()).then_some((node, cores))
        })
        .collect();
    // One reachable node leaves nothing to decide, and pinning there would
    // only narrow the scheduler's choices for no locality gain.
    if candidates.len() < 2 {
        return None;
    }
    if let Some(start) = home.and_then(|h| candidates.iter().position(|(node, _)| *node == h)) {
        candidates.rotate_left(start);
    }
    let (node, cores) = &candidates[index % candidates.len()];

    // SAFETY: an all-zero cpu_set_t is the documented empty mask.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &cpu in cores {
        // SAFETY: `set` is a live cpu_set_t and `cpu` passed `is_allowed`,
        // which bounds it by CPU_SETSIZE.
        unsafe { libc::CPU_SET(cpu, &mut set) };
    }
    // SAFETY: pid 0 is the calling thread; `set` outlives the call and the
    // kernel reads exactly `size_of::<cpu_set_t>()` bytes from it.
    let rc = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    if rc != 0 {
        tracing::debug!(node, "NUMA pin refused; worker stays unpinned");
        return None;
    }
    if let Err(e) = aether_mem::numa::prefer_current_thread(*node) {
        tracing::debug!(node, error = %e, "NUMA memory preference refused");
    }
    Some(*node)
}

#[cfg(not(all(target_os = "linux", feature = "numa")))]
pub fn pin_worker(_index: usize, _home: Option<u32>) -> Option<u32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Placement is advisory: on a single-node machine, a non-Linux host,
    /// or a build without the feature it reports `None` and the caller
    /// carries on. The call must never fail the worker that makes it.
    #[test]
    fn pin_worker_is_infallible_for_any_index() {
        let home = current_node();
        for index in [0usize, 1, 7, 64, usize::MAX] {
            let placed = std::thread::spawn(move || {
                let placed = pin_worker(index, home);
                #[cfg(all(target_os = "linux", feature = "numa"))]
                if let Some(node) = placed {
                    // The worker now runs only on CPUs of the node it
                    // reports, all of which it was allowed before.
                    let now = current_node();
                    assert!(now.is_none() || now == Some(node));
                }
                placed
            })
            .join()
            .unwrap();

            // Where placement can happen, it must name a node that exists:
            // the round-robin indexes into the candidate list, so an index
            // far past the node count must still wrap into it.
            #[cfg(all(target_os = "linux", feature = "numa"))]
            if let Some(node) = placed {
                assert!(
                    aether_mem::numa::nodes_online().contains(&node),
                    "index {index} pinned to node {node}, which is not online"
                );
            }

            // Where it cannot, the call is a no-op that reports so rather
            // than erroring — callers spawn workers regardless.
            #[cfg(not(all(target_os = "linux", feature = "numa")))]
            assert!(
                placed.is_none(),
                "index {index} reported placement without the numa feature"
            );
        }
    }
}
