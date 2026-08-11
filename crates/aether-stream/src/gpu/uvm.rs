//! Unified-memory (UVM) feature staging with sampler-driven prefetch.
//!
//! When the feature tensor is oversubscribed — larger than the VRAM
//! budget — CUDA managed memory lets the GPU touch any row and the driver
//! migrates the backing page on demand. Left to the driver's own
//! heuristics that migration stalls the access. But the sampler already
//! knows the *next* batch's node IDs, so we can migrate those rows ahead
//! of the compute with [`ManagedFeatures::prefetch_rows`]: the page
//! movement hides behind the current batch's kernels using exact
//! knowledge the driver's fault handler can't have.
//!
//! Built on `cuMemAllocManaged` + `cuMemAdvise` (read-mostly / preferred
//! location) + `cuMemPrefetchAsync`. Compile-gated behind `cuda`; the live
//! path needs a GPU and is validated on hardware, not in CI.

use std::io;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, sys};

/// A managed (unified-memory) feature matrix. Rows are `row_bytes` wide;
/// a node ID maps to a byte range arithmetically. The whole matrix has a
/// single virtual address valid on both host and device; the driver (and
/// our explicit prefetch) moves pages between them.
pub struct ManagedFeatures {
    ctx: Arc<CudaContext>,
    ptr: sys::CUdeviceptr,
    len: usize,
    row_bytes: usize,
    device: sys::CUdevice,
}

// SAFETY: the managed allocation is owned for the struct's lifetime and
// the CUDA context is thread-safe after creation.
unsafe impl Send for ManagedFeatures {}
// SAFETY: see the Send impl above.
unsafe impl Sync for ManagedFeatures {}

impl ManagedFeatures {
    /// Allocate a managed matrix of `num_rows * row_bytes` bytes and advise
    /// the driver it is read-mostly with its preferred location on
    /// `device` — the access pattern of a feature table read every batch
    /// but rarely written.
    pub fn new(
        ctx: &Arc<CudaContext>,
        device: sys::CUdevice,
        num_rows: usize,
        row_bytes: usize,
    ) -> io::Result<Self> {
        let len = num_rows
            .checked_mul(row_bytes)
            .ok_or_else(|| io::Error::other("managed feature size overflow"))?;
        if len == 0 {
            return Err(io::Error::other("managed features must be non-empty"));
        }

        let mut ptr: sys::CUdeviceptr = 0;
        // CU_MEM_ATTACH_GLOBAL = 1: the allocation is accessible from any
        // stream on any device.
        // SAFETY: `ptr` is a valid out-pointer; `len` is non-zero.
        let res = unsafe { sys::cuMemAllocManaged(&mut ptr, len, 1) };
        cuda_ok(res, "cuMemAllocManaged")?;

        let this = Self {
            ctx: ctx.clone(),
            ptr,
            len,
            row_bytes,
            device,
        };

        // Read-mostly: the driver keeps read-only replicas near every
        // reader instead of bouncing a single copy.
        this.advise(sys::CUmem_advise::CU_MEM_ADVISE_SET_READ_MOSTLY, device)?;
        // Preferred location device: cold pages migrate back here, not to
        // host, when evicted from a peer.
        this.advise(
            sys::CUmem_advise::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
            device,
        )?;
        Ok(this)
    }

    /// Raw device pointer to row `row`'s first byte, for kernel arguments.
    pub fn row_ptr(&self, row: usize) -> sys::CUdeviceptr {
        self.ptr + (row * self.row_bytes) as u64
    }

    /// The base device pointer.
    pub fn device_ptr(&self) -> sys::CUdeviceptr {
        self.ptr
    }

    /// Bytes per row.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Prefetch the pages backing `rows` onto the device on `stream`, so
    /// they are resident before a kernel on the same stream reads them.
    ///
    /// Each row's byte range is prefetched individually; adjacent rows the
    /// driver coalesces at page granularity. Call this from the loader
    /// with the *next* batch's node IDs while the current batch computes.
    pub fn prefetch_rows(&self, rows: &[u32], stream: &Arc<CudaStream>) -> io::Result<()> {
        let cu_stream = stream.cu_stream();
        for &row in rows {
            let row = row as usize;
            let start = row * self.row_bytes;
            if start + self.row_bytes > self.len {
                return Err(io::Error::other(format!(
                    "prefetch row {row} out of range (len {})",
                    self.len
                )));
            }
            // SAFETY: `[ptr+start, ptr+start+row_bytes)` is inside the
            // managed allocation; `cu_stream` belongs to this context.
            let res = unsafe {
                sys::cuMemPrefetchAsync(
                    self.ptr + start as u64,
                    self.row_bytes,
                    self.device,
                    cu_stream,
                )
            };
            cuda_ok(res, "cuMemPrefetchAsync")?;
        }
        Ok(())
    }

    fn advise(&self, advice: sys::CUmem_advise, device: sys::CUdevice) -> io::Result<()> {
        // SAFETY: the range is the full managed allocation; `advice` and
        // `device` are valid enum / device values.
        let res = unsafe { sys::cuMemAdvise(self.ptr, self.len, advice, device) };
        cuda_ok(res, "cuMemAdvise")
    }
}

impl Drop for ManagedFeatures {
    fn drop(&mut self) {
        // Keep the context alive until after the free.
        let _ctx = &self.ctx;
        // SAFETY: `ptr` came from cuMemAllocManaged and is freed once.
        let res = unsafe { sys::cuMemFree_v2(self.ptr) };
        if res != sys::CUresult::CUDA_SUCCESS {
            tracing::warn!(?res, "cuMemFree on managed features failed");
        }
    }
}

fn cuda_ok(res: sys::CUresult, what: &str) -> io::Result<()> {
    if res == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(io::Error::other(format!("{what} failed: {res:?}")))
    }
}
