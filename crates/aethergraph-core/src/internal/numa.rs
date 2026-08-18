//! NUMA placement policy for the core's worker pools.
//!
//! The graph body is interleaved across nodes at load
//! ([`hint::interleave_mmap_range`](super::hint::interleave_mmap_range)),
//! which spreads bandwidth but leaves every access a coin flip between
//! local and remote. Pinning the pool completes the other half: a worker
//! that stays on one node keeps its own allocations — landing buffers,
//! frontier vectors, the subgraph it builds — local to the socket reading
//! them, and stops the scheduler from migrating it mid-batch and stranding
//! every one of those buffers across the interconnect.
//!
//! Workers are spread round-robin over the online nodes rather than packed
//! onto one: a pool sized to the machine should use every memory
//! controller, and consecutive worker indices landing on different nodes is
//! the simplest assignment that does.
//!
//! Everything here is fail-soft. A single-node machine, a kernel without
//! the policy syscalls, or a cpuset that already constrains the process all
//! leave the pool scheduled exactly as it was.

/// Pin worker `index` of a pool to one NUMA node and prefer that node for
/// its allocations. Returns the node it was pinned to, or `None` when
/// placement did not apply.
#[cfg(all(target_os = "linux", feature = "numa"))]
pub fn pin_worker(index: usize) -> Option<u32> {
    let nodes = aether_mem::numa::nodes_online();
    // Nothing to decide on a single-node machine, and pinning there would
    // only narrow the scheduler's choices for no locality gain.
    if nodes.len() < 2 {
        return None;
    }
    let node = nodes[index % nodes.len()];
    match aether_mem::numa::pin_current_thread(node) {
        Ok(()) => Some(node),
        Err(e) => {
            tracing::debug!(node, error = %e, "NUMA pin refused; worker stays unpinned");
            None
        }
    }
}

#[cfg(not(all(target_os = "linux", feature = "numa")))]
pub fn pin_worker(_index: usize) -> Option<u32> {
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
        for index in [0usize, 1, 7, 64, usize::MAX] {
            let placed = pin_worker(index);

            // Where placement can happen, it must name a node that exists:
            // the round-robin indexes into the online list, so an index far
            // past the node count must still wrap into it.
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
