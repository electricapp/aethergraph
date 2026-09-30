//! K5.1 persistent work-ring kernel and host producer.
//!
//! One long-lived CTA drains a ring of [`PersistentWork`] entries until the
//! host stops it. Three warps specialize into fetch / transform / compute
//! roles over shared queues so the next ring claim overlaps in-flight local
//! work.
//!
//! The ring and its control words live in mapped pinned host memory. The
//! host posts with plain stores and reads progress with atomic loads, so no
//! CUDA call is ever queued behind the kernel, and none waits on the
//! per-buffer events cudarc attaches to device memory a kernel holds for its
//! whole run. The kernel runs on a stream of its own for the same reason: it
//! never finishes until stopped, so nothing else may queue behind it.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg, result, sys,
};
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub(super) const KERNEL_SRC: &str = include_str!("persistent.cu");
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

/// Fixed-size descriptor posted into the ring. Mirrors the CUDA struct.
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

/// Control words shared with the kernel. Mirrors `PersistentControl` in
/// persistent.cu.
#[repr(C)]
struct PersistentControl {
    /// Host → device: items posted.
    tail: AtomicU32,
    /// Host → device: nonzero once posting has ended.
    stop: AtomicU32,
    /// Device → host: items claimed; slots below it are free to reuse.
    head: AtomicU32,
    _pad: u32,
    /// Device → host: items finished.
    completed: AtomicU64,
}

/// Bytes ahead of the ring, keeping the control words off the ring's cache
/// lines.
const CONTROL_BYTES: usize = 128;
const _: () = assert!(size_of::<PersistentControl>() <= CONTROL_BYTES);
const _: () = assert!(CONTROL_BYTES.is_multiple_of(align_of::<PersistentWork>()));

/// Pinned, device-mapped host memory: the control block, then the ring.
struct MappedRing {
    ctx: Arc<CudaContext>,
    host: NonNull<u8>,
    device: sys::CUdeviceptr,
    capacity: u32,
}

impl MappedRing {
    fn new(ctx: &Arc<CudaContext>, capacity: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = (capacity as usize)
            .checked_mul(size_of::<PersistentWork>())
            .and_then(|ring| ring.checked_add(CONTROL_BYTES))
            .ok_or("persistent ring size overflows")?;
        ctx.bind_to_thread()?;
        // SAFETY: a fresh allocation of `bytes`; it is zeroed below before
        // anything reads it. DEVICEMAP makes it addressable from the GPU.
        let raw = unsafe {
            result::malloc_host(
                bytes,
                sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE,
            )?
        };
        let host = NonNull::new(raw.cast::<u8>()).ok_or("cuMemHostAlloc returned null")?;
        // SAFETY: `host` is valid for `bytes` writes; all-zero is a valid
        // control block (nothing posted, not stopped) and ring.
        unsafe { host.as_ptr().write_bytes(0, bytes) };
        let mut device: sys::CUdeviceptr = 0;
        // SAFETY: `host` came from cuMemHostAlloc with DEVICEMAP on this
        // context, which is current; `device` is a valid out-pointer.
        let res = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device, raw, 0) };
        if res != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: `raw` came from malloc_host and is freed once here.
            let _ = unsafe { result::free_host(raw) };
            return Err(format!("cuMemHostGetDevicePointer failed: {res:?}").into());
        }
        Ok(Self {
            ctx: ctx.clone(),
            host,
            device,
            capacity,
        })
    }

    fn control(&self) -> &PersistentControl {
        // SAFETY: the allocation starts with CONTROL_BYTES >= the control
        // block, page-aligned and zero-initialized; its fields are atomics
        // (or never read), so sharing it with the device is sound.
        unsafe { &*self.host.as_ptr().cast::<PersistentControl>() }
    }

    fn slot(&self, index: u32) -> *mut PersistentWork {
        let offset =
            CONTROL_BYTES + (index & (self.capacity - 1)) as usize * size_of::<PersistentWork>();
        // SAFETY: the masked index is below `capacity`, so the entry lies in
        // the ring; CONTROL_BYTES keeps entries aligned.
        unsafe { self.host.as_ptr().add(offset).cast::<PersistentWork>() }
    }

    fn device_control(&self) -> sys::CUdeviceptr {
        self.device
    }

    fn device_ring(&self) -> sys::CUdeviceptr {
        self.device + CONTROL_BYTES as u64
    }
}

impl Drop for MappedRing {
    fn drop(&mut self) {
        if let Err(e) = self.ctx.bind_to_thread() {
            tracing::warn!(error = %e, "persistent ring: binding context for free failed");
        }
        // SAFETY: `host` came from malloc_host; the owning worker has joined
        // the kernel before its ring drops, so nothing still reads it.
        if let Err(e) = unsafe { result::free_host(self.host.as_ptr().cast()) } {
            tracing::warn!(error = %e, "persistent ring: cuMemFreeHost failed");
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Running,
    Stopped,
}

/// Host-side controller for a draining persistent kernel. Single-shot:
/// [`Self::start`] once, [`Self::post`] any number of times, then
/// [`Self::stop_and_join`].
pub struct PersistentWorker {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    phase: Phase,
    host_tail: u32,
    /// Dropped by hand once `Drop::drop` has joined the kernel.
    ring: ManuallyDrop<MappedRing>,
}

// SAFETY: `MappedRing`'s raw pointer is to pinned host memory this worker
// owns; posting goes through `&mut self`, and the shared control words are
// atomics.
unsafe impl Send for PersistentWorker {}

impl PersistentWorker {
    /// Compile the drain kernel, create its stream, and allocate a
    /// power-of-two ring of `capacity` entries.
    pub fn new(ctx: &Arc<CudaContext>, capacity: u32) -> Result<Self, Box<dyn std::error::Error>> {
        if capacity == 0 || !capacity.is_power_of_two() || i32::try_from(capacity).is_err() {
            return Err("persistent ring capacity must be a power of two below 2^31".into());
        }
        let module = ctx.load_module(super::compile_for_device(ctx, KERNEL_SRC)?)?;
        Ok(Self {
            stream: ctx.new_stream()?,
            func: module.load_function(KERNEL_NAME)?,
            phase: Phase::Idle,
            host_tail: 0,
            ring: ManuallyDrop::new(MappedRing::new(ctx, capacity)?),
        })
    }

    /// Launch the persistent drain (returns immediately).
    pub fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.phase != Phase::Idle {
            return Err("persistent worker already started".into());
        }
        let control = self.ring.device_control();
        let ring = self.ring.device_ring();
        let capacity = self.ring.capacity as i32;
        // SAFETY: arguments match persistent.cu; the mapped allocation
        // outlives the kernel (joined before the ring drops).
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(&control)
                .arg(&ring)
                .arg(&capacity)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    // Three specialized warps: fetch / transform / compute.
                    block_dim: (96, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        self.phase = Phase::Running;
        Ok(())
    }

    /// Post one work item. Returns `false` if the ring is full.
    pub fn post(&mut self, work: PersistentWork) -> Result<bool, Box<dyn std::error::Error>> {
        if self.phase == Phase::Stopped {
            return Err("persistent worker already stopped".into());
        }
        let control = self.ring.control();
        // Acquire pairs with the kernel's release after copying an entry
        // out, so a slot below `head` is safe to overwrite.
        let claimed = control.head.load(Ordering::Acquire);
        if self.host_tail.wrapping_sub(claimed) >= self.ring.capacity {
            return Ok(false);
        }
        // SAFETY: the slot is inside the ring and not yet published (or
        // already claimed), so the kernel is not reading it.
        unsafe { self.ring.slot(self.host_tail).write_volatile(work) };
        self.host_tail = self.host_tail.wrapping_add(1);
        aether_mem::dma_release_fence();
        control.tail.store(self.host_tail, Ordering::Release);
        Ok(true)
    }

    /// Items the kernel has finished so far.
    #[must_use]
    pub fn completed(&self) -> u64 {
        self.ring.control().completed.load(Ordering::Acquire)
    }

    /// Stop posting, let the kernel drain everything already posted, and
    /// wait for it to exit. Returns the number of items completed.
    pub fn stop_and_join(&mut self) -> Result<u64, Box<dyn std::error::Error>> {
        if self.phase == Phase::Running {
            aether_mem::dma_release_fence();
            self.ring.control().stop.store(1, Ordering::Release);
            self.stream.synchronize()?;
        }
        self.phase = Phase::Stopped;
        Ok(self.completed())
    }
}

impl Drop for PersistentWorker {
    fn drop(&mut self) {
        if self.phase == Phase::Running {
            self.ring.control().stop.store(1, Ordering::Release);
            if let Err(e) = self.stream.synchronize() {
                // The kernel may still read the ring; freeing it would hand
                // the device freed memory.
                tracing::error!(error = %e, "persistent worker join failed; leaking its ring");
                return;
            }
        }
        // SAFETY: the kernel never started or has been joined, so nothing
        // reads the ring; it is dropped exactly once, here.
        unsafe { ManuallyDrop::drop(&mut self.ring) };
    }
}

// TODO(HARDWARE): prove forward progress under concurrent RDMA producers,
// SM preemption/MPS, and multi-warp specialization on a real GPU.
