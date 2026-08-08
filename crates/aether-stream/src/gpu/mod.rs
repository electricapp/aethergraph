//! GPU-side infrastructure for GPUDirect RDMA feature gathering.
//!
//! Provides VRAM staging buffers registered with the InfiniBand NIC
//! (nvidia-peermem, with a dma-buf fallback) and a CUDA kernel for
//! seqlock validation + feature compaction.

pub mod buffer;
pub mod kernel;
