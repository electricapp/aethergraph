//! Lock-free slab allocator with pre-allocated, page-aligned slots.
//!
//! Extracted from amira-asr-server-rs for use in high-performance streaming
//! systems. Provides a [`SharedMemoryRing`] backed by an ABA-tagged Treiber
//! stack free list, optional HugePages, and pluggable [`MemoryHook`]s for
//! registering memory with external systems (mlock, CUDA, RDMA).

mod ring;

pub mod hooks;

pub use ring::{MemoryHook, RingSlot, SharedMemoryRing};
