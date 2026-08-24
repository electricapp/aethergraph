//! K5.1 persistent work-ring kernel and host producer.
//!
//! One long-lived CTA drains a device-visible SPSC ring of
//! [`PersistentWork`] entries until the host sets a stop flag. Three warps
//! specialize into fetch / transform / compute roles over shared queues so
//! the next ring claim overlaps in-flight local work.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

const KERNEL_SRC: &str = include_str!("persistent.cu");
const KERNEL_NAME: &str = "persistent_work_drain";

/// Work classes the persistent drain kernel recognizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PersistentWorkKind {
    /// Validate a FeatureTable slot snapshot.
    Validate = 1,
    /// Gather a requested neighbor frontier.
    Gather = 2,
    /// Release a completed response entry.
    Complete = 3,
}

/// Fixed-size descriptor posted into the device ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct PersistentWork {
    /// Discriminator for the work payload ([`PersistentWorkKind`] as u32).
    pub kind: u32,
    /// Device pointer or opaque work identifier.
    pub payload: u64,
    /// Number of rows/items addressed by `payload`.
    pub len: u32,
}

// SAFETY: plain POD matching the CUDA struct layout.
unsafe impl DeviceRepr for PersistentWork {}

impl PersistentWork {
    /// Build a work item from a typed kind.
    #[must_use]
    pub const fn new(kind: PersistentWorkKind, payload: u64, len: u32) -> Self {
        Self {
            kind: kind as u32,
            payload,
            len,
        }
    }
}

/// Host-side controller for a draining persistent kernel.
pub struct PersistentWorker {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    ring: CudaSlice<PersistentWork>,
    head: CudaSlice<u32>,
    tail: CudaSlice<u32>,
    stop: CudaSlice<i32>,
    completed: CudaSlice<u64>,
    capacity: u32,
    host_tail: u32,
}

impl PersistentWorker {
    /// Compile the drain kernel and allocate a power-of-two ring.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        capacity: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if capacity == 0 || !capacity.is_power_of_two() {
            return Err("persistent ring capacity must be a non-zero power of two".into());
        }
        let module = ctx.load_module(cudarc::nvrtc::compile_ptx(KERNEL_SRC)?)?;
        Ok(Self {
            stream: stream.clone(),
            func: module.load_function(KERNEL_NAME)?,
            ring: stream.alloc_zeros(capacity as usize)?,
            head: stream.alloc_zeros(1)?,
            tail: stream.alloc_zeros(1)?,
            stop: stream.alloc_zeros(1)?,
            completed: stream.alloc_zeros(1)?,
            capacity,
            host_tail: 0,
        })
    }

    /// Launch the persistent drain on `stream` (returns immediately).
    pub fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        let capacity = self.capacity as i32;
        // SAFETY: arguments match persistent.cu; allocations outlive the kernel.
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(&self.ring)
                .arg(&self.head)
                .arg(&self.tail)
                .arg(&self.stop)
                .arg(&self.completed)
                .arg(&capacity)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    // Three specialized warps: fetch / transform / compute.
                    block_dim: (96, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Post one work item. Returns false if the ring is full.
    pub fn post(&mut self, work: PersistentWork) -> Result<bool, Box<dyn std::error::Error>> {
        let mut head = [0u32];
        self.stream.memcpy_dtoh(&self.head, &mut head)?;
        if self.host_tail.wrapping_sub(head[0]) >= self.capacity {
            return Ok(false);
        }
        let slot = (self.host_tail & (self.capacity - 1)) as usize;
        {
            let mut view = self.ring.slice_mut(slot..slot + 1);
            self.stream.memcpy_htod(&[work], &mut view)?;
        }
        self.host_tail = self.host_tail.wrapping_add(1);
        self.stream.memcpy_htod(&[self.host_tail], &mut self.tail)?;
        Ok(true)
    }

    /// Ask the kernel to exit its spin loop, then synchronize the stream.
    pub fn stop_and_join(&mut self) -> Result<u64, Box<dyn std::error::Error>> {
        self.stream.memcpy_htod(&[1i32], &mut self.stop)?;
        self.stream.synchronize()?;
        let mut completed = [0u64];
        self.stream.memcpy_dtoh(&self.completed, &mut completed)?;
        Ok(completed[0])
    }
}

// TODO(HARDWARE): prove forward progress under concurrent RDMA producers,
// SM preemption/MPS, and multi-warp specialization on a real GPU.
