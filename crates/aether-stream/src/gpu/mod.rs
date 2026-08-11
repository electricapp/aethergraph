//! GPU-side infrastructure for GPUDirect RDMA feature gathering.
//!
//! Provides VRAM staging buffers registered with the InfiniBand NIC
//! (nvidia-peermem, with a dma-buf fallback) and a CUDA kernel for
//! seqlock validation + feature compaction.
//!
//! Orchestration primitives that drive the GPU without writing kernels:
//! [`uvm`] (sampler-driven UVM prefetch), [`vmm`] (a growable VRAM cache
//! over the virtual-memory-management API), [`ipc`] (multi-process sharing
//! of that cache via exported handles), and [`gdrcopy`] (sub-µs CPU stores
//! into VRAM through a BAR1 mapping).

pub mod buffer;
#[cfg(feature = "gdrcopy")]
pub mod gdrcopy;
pub mod ipc;
pub mod kernel;
pub mod uvm;
pub mod vmm;
