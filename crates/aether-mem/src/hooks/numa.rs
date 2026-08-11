//! NUMA placement hooks — steer ring allocations onto specific nodes.
//!
//! Ring allocations pre-fault before hooks run, so both hooks rely on
//! `MPOL_MF_MOVE` migration inside [`crate::numa`]; the one-time move at
//! allocation is what buys every later access its local-node latency.

use crate::{HookError, MemoryHook, numa};

/// Binds the region to one NUMA node.
///
/// The canonical use is DMA locality: memory served by a NIC (RDMA MRs,
/// AF_XDP UMEM) belongs on the NIC's node so device reads never cross
/// the socket interconnect.
pub struct NumaBindHook {
    node: u32,
}

impl NumaBindHook {
    /// Bind to `node` (as reported by e.g. a device's sysfs `numa_node`).
    pub const fn new(node: u32) -> Self {
        Self { node }
    }
}

impl MemoryHook for NumaBindHook {
    fn on_alloc(&self, ptr: *mut u8, size: usize) -> Result<(), HookError> {
        numa::bind_region(ptr, size, self.node)
    }

    fn on_dealloc(&self, _ptr: *mut u8, _size: usize) {
        // Policy dies with the mapping; nothing to undo.
    }
}

/// Interleaves the region page-round-robin across NUMA nodes so no single
/// memory controller carries every reader of a shared table.
pub struct NumaInterleaveHook {
    nodes: Vec<u32>,
}

impl NumaInterleaveHook {
    /// Interleave across every online node. On single-node machines the
    /// hook is a no-op.
    pub fn all_nodes() -> Self {
        Self {
            nodes: numa::nodes_online(),
        }
    }

    /// Interleave across a caller-chosen node set.
    pub fn new(nodes: Vec<u32>) -> Self {
        Self { nodes }
    }
}

impl MemoryHook for NumaInterleaveHook {
    fn on_alloc(&self, ptr: *mut u8, size: usize) -> Result<(), HookError> {
        // One node means interleaving degenerates to default placement;
        // skip the syscall rather than report a meaningless success.
        if self.nodes.len() <= 1 {
            return Ok(());
        }
        numa::interleave_region(ptr, size, &self.nodes)
    }

    fn on_dealloc(&self, _ptr: *mut u8, _size: usize) {
        // Policy dies with the mapping; nothing to undo.
    }
}
