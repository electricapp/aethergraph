//! Lock-free slab allocator with pre-allocated, aligned slots.
//!
//! Provides a [`SharedMemoryRing`] backed by an ABA-tagged Treiber stack free
//! list, optional HugePages, and pluggable [`MemoryHook`]s for registering
//! memory with external systems (mlock, CUDA, RDMA).

mod fence;
mod ring;

pub mod hooks;
pub mod numa;

pub use fence::dma_release_fence;
pub use ring::{
    DetachedSlot, FreeList, HookError, HookSpan, MemoryHook, RingBuilderError, RingSlot,
    SharedMemoryRing, SlotOverflow, SlotRegion,
};
