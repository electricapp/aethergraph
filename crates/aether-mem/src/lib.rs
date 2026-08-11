//! Lock-free slab allocator with pre-allocated, aligned slots.
//!
//! Provides a [`SharedMemoryRing`] backed by an ABA-tagged Treiber stack free
//! list, optional HugePages, and pluggable [`MemoryHook`]s for registering
//! memory with external systems (mlock, CUDA, RDMA).

mod ring;

pub mod hooks;
pub mod numa;

pub use ring::{
    DetachedSlot, FreeList, HookError, MemoryHook, RingBuilderError, RingSlot, SharedMemoryRing,
    SlotOverflow,
};
