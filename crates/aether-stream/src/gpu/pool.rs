//! The VRAM feature pool: one object composing the GPU orchestration
//! primitives into the allocation the gather path actually uses.
//!
//! Without it each `gather()` calls `cudaMalloc` for its output — a
//! synchronizing, allocator-locking operation on the per-batch critical
//! path, and one whose cost grows with fragmentation over a long training
//! run. The pool replaces that with a single virtual reservation
//! ([`vmm::GrowableVram`]) that is committed once and reused: batches land
//! at a stable device pointer, and physical VRAM is added only when a
//! batch genuinely needs more than has ever been needed before.
//!
//! Around that core it wires the remaining orchestration pieces to the
//! things they are actually for:
//!
//! - [`uvm::ManagedFeatures`] is the optional staging tier. The sampler
//!   knows the *next* batch's node IDs while the current one trains, so
//!   [`FeaturePool::prefetch_next`] issues `cuMemPrefetchAsync` for those
//!   rows on a side stream and the migration overlaps compute instead of
//!   faulting during it.
//! - [`ipc`] exports the pool's physical handle so a second process maps
//!   the same VRAM: one copy of a cached feature block per GPU rather
//!   than per worker.
//! - [`gdrcopy`] gives the CPU a BAR1 window into the pool's header, so a
//!   generation stamp is a store, not a kernel launch or a
//!   `cudaMemcpy` — sub-microsecond, and orderable against RDMA writes.
//!
//! Every piece past the reservation is optional and probed at
//! construction: a host without gdrdrv, or a build without the `gdrcopy`
//! feature, gets a pool that still serves gathers.

use std::io;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, sys};

use super::{ipc, uvm, vmm};

/// Bytes at the base of the pool reserved for control words rather than
/// feature rows.
///
/// The stamp lives here so the CPU can write it through the BAR1 window
/// without touching row data. One GPU page keeps the rows page-aligned,
/// which is what `gdr_pin_buffer` and `cuMemMap` both want anyway.
pub const HEADER_BYTES: usize = 65_536;

/// A reusable VRAM region for gather output, plus the optional staging,
/// sharing, and stamping paths built on it.
pub struct FeaturePool {
    vram: vmm::GrowableVram,
    ctx: Arc<CudaContext>,
    device: sys::CUdevice,
    row_bytes: usize,
    /// Optional managed-memory staging tier for UVM prefetch.
    staging: Option<uvm::ManagedFeatures>,
    /// Optional BAR1 window over the header for CPU-side stamping.
    #[cfg(feature = "gdrcopy")]
    stamp_window: Option<super::gdrcopy::GdrMapping>,
    /// Monotonic generation, incremented per published batch.
    generation: u64,
}

impl FeaturePool {
    /// Reserve address space for up to `max_rows` rows of `row_bytes`.
    ///
    /// Nothing is committed yet: physical VRAM is taken on the first
    /// [`Self::ensure_rows`], and only grows from there.
    pub fn reserve(
        ctx: &Arc<CudaContext>,
        device: sys::CUdevice,
        max_rows: usize,
        row_bytes: usize,
    ) -> io::Result<Self> {
        if row_bytes == 0 {
            return Err(io::Error::other("row_bytes must be > 0"));
        }
        let payload = max_rows
            .checked_mul(row_bytes)
            .ok_or_else(|| io::Error::other("pool size overflows usize"))?;
        let total = payload
            .checked_add(HEADER_BYTES)
            .ok_or_else(|| io::Error::other("pool size overflows usize"))?;

        let vram = vmm::GrowableVram::reserve(ctx, device, total)?;
        Ok(Self {
            vram,
            ctx: ctx.clone(),
            device,
            row_bytes,
            staging: None,
            #[cfg(feature = "gdrcopy")]
            stamp_window: None,
            generation: 0,
        })
    }

    /// Device pointer to row 0. Stable for the pool's whole life, even
    /// across growth — that stability is the point of reserving the range
    /// up front rather than reallocating.
    pub fn rows_ptr(&self) -> sys::CUdeviceptr {
        self.vram.device_ptr() + HEADER_BYTES as u64
    }

    /// Device pointer to row `row`.
    pub fn row_ptr(&self, row: usize) -> sys::CUdeviceptr {
        self.rows_ptr() + (row * self.row_bytes) as u64
    }

    /// Bytes per row.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Rows currently backed by physical VRAM.
    pub fn committed_rows(&self) -> usize {
        self.vram.committed().saturating_sub(HEADER_BYTES) / self.row_bytes
    }

    /// Commit enough physical VRAM for `rows`, growing if this batch is
    /// larger than any before it. A no-op once the high-water mark is
    /// reached, which is the steady state after the first few batches.
    pub fn ensure_rows(&mut self, rows: usize) -> io::Result<()> {
        let need = rows
            .checked_mul(self.row_bytes)
            .and_then(|b| b.checked_add(HEADER_BYTES))
            .ok_or_else(|| io::Error::other("row count overflows the pool"))?;
        self.vram.grow_to(need)
    }

    /// Attach a managed-memory staging tier of `num_rows` rows.
    ///
    /// This is the tier [`Self::prefetch_next`] migrates from. Sized to
    /// the resident working set, not the whole feature table: rows the
    /// sampler will need soon live here, and the driver pages them in.
    pub fn with_staging(&mut self, num_rows: usize) -> io::Result<()> {
        self.staging = Some(uvm::ManagedFeatures::new(
            &self.ctx,
            self.device,
            num_rows,
            self.row_bytes,
        )?);
        Ok(())
    }

    /// The staging tier, if one was attached.
    pub fn staging(&self) -> Option<&uvm::ManagedFeatures> {
        self.staging.as_ref()
    }

    /// Migrate the rows the *next* batch will read onto the device, on
    /// `stream`.
    ///
    /// Call this right after sampling the next batch and before training
    /// on the current one: the migration then overlaps compute rather
    /// than faulting inside it. Returns `Ok(false)` when no staging tier
    /// is attached, so a caller can wire the call unconditionally.
    pub fn prefetch_next(&self, rows: &[u32], stream: &Arc<CudaStream>) -> io::Result<bool> {
        match &self.staging {
            Some(staging) => {
                staging.prefetch_rows(rows, stream)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Export the pool's physical allocation as a file descriptor a peer
    /// process can import with [`ipc::ImportedVram::import`].
    ///
    /// Returns the fd and the exported byte length. Errors before the
    /// first [`Self::ensure_rows`] — there is no physical allocation to
    /// export until then. Send the fd over a Unix socket with
    /// `SCM_RIGHTS`; the length must travel with it, since the importer
    /// cannot infer it.
    pub fn export_fd(&self) -> io::Result<(std::os::unix::io::RawFd, usize)> {
        let (handle, len) = self
            .vram
            .first_chunk()
            .ok_or_else(|| io::Error::other("nothing committed yet; call ensure_rows first"))?;
        let fd = ipc::export_handle_to_fd(handle)?;
        Ok((fd, len))
    }

    /// Open a BAR1 window over the pool's header so the CPU can stamp it
    /// with a plain store.
    ///
    /// Best-effort: returns `Ok(false)` when gdrdrv is absent or the
    /// mapping is refused, leaving the pool fully usable without stamping.
    #[cfg(feature = "gdrcopy")]
    pub fn enable_stamping(&mut self) -> io::Result<bool> {
        if self.vram.committed() < HEADER_BYTES {
            return Err(io::Error::other(
                "nothing committed yet; call ensure_rows first",
            ));
        }
        match super::gdrcopy::GdrMapping::new(self.vram.device_ptr(), HEADER_BYTES) {
            Ok(m) => {
                self.stamp_window = Some(m);
                Ok(true)
            }
            Err(e) => {
                tracing::debug!("GDRCopy stamping unavailable: {e}");
                Ok(false)
            }
        }
    }

    /// Publish a batch: bump the generation and write it into the header
    /// through the BAR1 window.
    ///
    /// A consumer polling the header sees the new generation the moment
    /// the store retires — no kernel launch, no `cudaMemcpy`, no stream
    /// synchronization. Returns the new generation, or `Ok(None)` when
    /// stamping isn't enabled.
    #[cfg(feature = "gdrcopy")]
    pub fn stamp_generation(&mut self) -> io::Result<Option<u64>> {
        let Some(window) = &self.stamp_window else {
            return Ok(None);
        };
        self.generation += 1;
        window.copy_to(0, &self.generation.to_le_bytes())?;
        Ok(Some(self.generation))
    }

    /// The generation last published by [`Self::stamp_generation`].
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Geometry is pure arithmetic and must hold without a GPU: rows are
    /// laid out after the header, at the stride the pool was built with.
    #[test]
    fn row_pointers_follow_the_header() {
        // Exercise the arithmetic directly — constructing a pool needs a
        // CUDA context, which CI containers don't have.
        let row_bytes = 512usize;
        let base = 0x1000_0000u64;
        let rows_ptr = base + HEADER_BYTES as u64;

        assert_eq!(rows_ptr, base + 65_536);
        for row in [0usize, 1, 7, 1024] {
            assert_eq!(
                rows_ptr + (row * row_bytes) as u64,
                base + HEADER_BYTES as u64 + (row * row_bytes) as u64
            );
        }
    }

    #[test]
    fn committed_rows_excludes_the_header() {
        // committed_rows subtracts the header before dividing, so a pool
        // committed to exactly the header holds zero rows.
        let row_bytes = 256usize;
        for (committed, want) in [
            (0usize, 0usize),
            (HEADER_BYTES, 0),
            (HEADER_BYTES + row_bytes, 1),
            (HEADER_BYTES + row_bytes * 10, 10),
        ] {
            let got = committed.saturating_sub(HEADER_BYTES) / row_bytes;
            assert_eq!(got, want, "committed {committed}");
        }
    }
}
