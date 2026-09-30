//! RDMA-backed feature gather that integrates with aethergraph-core's
//! sampling pipeline. Wraps RdmaFeatureClient to produce GPU feature
//! tensors from sampled node IDs.
//!
//! This is the bridge between graph sampling (aethergraph-core) and
//! feature serving (aether-stream). The prefetch thread in Python's
//! NeighborLoader calls `gather()` inline after sampling — takes ~20μs.

use crate::rdma::client::RdmaFeatureClient;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::sync::Arc;

/// VRAM holding one gathered batch.
///
/// Complete when handed out: the copy into it has finished, so a consumer
/// on any stream may read it. Freed stream-ordered behind everything
/// enqueued on the CUDA legacy default stream when it drops — the stream a
/// framework consumer such as PyTorch computes on by default. A consumer on
/// a non-blocking stream must synchronize that stream before releasing the
/// tensor, as PyTorch's own `record_stream` contract requires for memory
/// used across streams.
#[derive(Debug)]
pub struct GatheredRows {
    slice: Option<CudaSlice<f32>>,
}

impl GatheredRows {
    /// Raw device pointer (`CUdeviceptr`) to the rows.
    pub fn device_ptr(&self) -> u64 {
        let slice = self.slice.as_ref().expect("present until drop");
        let (ptr, _record) = slice.device_ptr(slice.stream());
        ptr
    }
}

impl Drop for GatheredRows {
    fn drop(&mut self) {
        let Some(slice) = self.slice.take() else {
            return;
        };
        // The allocation's stream may reuse it as soon as the free is
        // ordered; order the free after work consumers queued on the legacy
        // stream up to this moment. On failure, leak rather than free early.
        let stream = slice.stream().clone();
        let legacy = stream.context().default_stream();
        let ordered = legacy
            .record_event(None)
            .and_then(|event| stream.wait(&event));
        match ordered {
            Ok(()) => drop(slice),
            Err(e) => {
                tracing::warn!(error = %e, "could not order a gather buffer's free; leaking it");
                std::mem::forget(slice);
            }
        }
    }
}

/// Feature tensor in VRAM, owned by this handle.
///
/// The buffer is a fresh per-call allocation that the compacted validator
/// output is copied into, so the tensor stays valid across subsequent
/// `gather()` calls and after the [`RdmaFeatureGather`] drops — the VRAM is
/// freed when the last clone of `buf` is dropped (see [`GatheredRows`] for
/// the ordering).
#[derive(Debug, Clone)]
pub struct OwnedGpuFeatures {
    /// Owned device buffer holding the tensor.
    /// Layout: contiguous `[num_nodes, feature_dim]` row-major f32.
    pub buf: Arc<GatheredRows>,
    /// Number of nodes (rows) in the tensor.
    pub num_nodes: usize,
    /// Feature dimension (columns).
    pub feature_dim: usize,
    /// CUDA device ordinal.
    pub gpu_id: i32,
}

impl OwnedGpuFeatures {
    /// Raw CUDA device pointer (`CUdeviceptr`) to the tensor data.
    pub fn device_ptr(&self) -> u64 {
        self.buf.device_ptr()
    }
}

/// A batch of features living in the gather's [`crate::gpu::FeaturePool`].
///
/// Unlike [`OwnedGpuFeatures`], this **borrows** the pool's VRAM: the next
/// [`RdmaFeatureGather::gather_pooled`] overwrites it. That is the whole
/// point — no per-batch `cudaMalloc` on the critical path — but it means a
/// consumer must finish with the tensor (or copy it) before asking for the
/// next batch. The lifetime ties it to the `&mut` borrow of the gather, so
/// the compiler enforces exactly that. The rows are complete when handed
/// out.
#[derive(Debug)]
pub struct PooledGpuFeatures<'a> {
    /// Device pointer to the batch's first row, inside the pool.
    pub device_ptr: u64,
    /// Number of nodes (rows).
    pub num_nodes: usize,
    /// Feature dimension (columns).
    pub feature_dim: usize,
    /// CUDA device ordinal.
    pub gpu_id: i32,
    /// Generation stamped for this batch, when stamping is enabled.
    pub generation: u64,
    _pool: std::marker::PhantomData<&'a ()>,
}

/// RDMA feature gather — connects to a remote FeatureTable and pulls
/// features directly into VRAM for a batch of node IDs.
pub struct RdmaFeatureGather {
    client: RdmaFeatureClient,
    gpu_id: i32,
    /// Optional reusable output pool. When present,
    /// [`Self::gather_pooled`] writes batches here instead of allocating.
    pool: Option<crate::gpu::FeaturePool>,
    /// Side stream the staging tier's prefetches run on, so migrations
    /// overlap the gather instead of queueing behind it.
    prefetch_stream: Option<Arc<CudaStream>>,
}

impl RdmaFeatureGather {
    /// Connect to a feature server and set up GPUDirect RDMA.
    ///
    /// `server_addr` is `"host:port"` of the RDMA control plane.
    /// `gpu_id` selects which GPU to allocate VRAM on.
    /// `max_batch_nodes` is the upper bound on nodes per gather call.
    /// `gid_index` is the local GID table index — pass the routable RoCEv2
    /// IPv4-mapped GID (typically 1 on Linux; verify with `show_gids`). No
    /// auto-probing; callers must know their fabric.
    pub fn connect(
        server_addr: &str,
        gpu_id: usize,
        max_batch_nodes: usize,
        gid_index: u8,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let client = RdmaFeatureClient::connect(server_addr, gpu_id, max_batch_nodes, gid_index)?;
        Ok(Self {
            client,
            gpu_id: gpu_id as i32,
            pool: None,
            prefetch_stream: None,
        })
    }

    /// Switch gathers onto a reusable [`crate::gpu::FeaturePool`] sized for
    /// `max_batch_nodes`.
    ///
    /// After this, [`Self::gather_pooled`] writes each batch into the same
    /// committed VRAM instead of allocating a fresh buffer per call — the
    /// allocator leaves the per-batch critical path entirely. `staging_rows`,
    /// when non-zero, additionally attaches a managed-memory tier that
    /// [`Self::prefetch_next`] migrates into ahead of the batch.
    ///
    /// The plain [`Self::gather`] keeps its owned-allocation behaviour, so
    /// enabling the pool never changes existing callers.
    pub fn enable_pool(
        &mut self,
        max_batch_nodes: usize,
        staging_rows: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ctx = self.client.cuda_context().clone();
        let device = self.gpu_id;
        let row_bytes = self.client.feature_dim() * std::mem::size_of::<f32>();

        let mut pool = crate::gpu::FeaturePool::reserve(&ctx, device, max_batch_nodes, row_bytes)?;
        // Commit the full batch width up front: the first gather would
        // otherwise grow mid-flight, which is exactly the allocator work
        // the pool exists to remove.
        pool.ensure_rows(max_batch_nodes)?;
        if staging_rows > 0 {
            pool.with_staging(staging_rows)?;
            self.prefetch_stream = Some(ctx.new_stream()?);
        }
        #[cfg(feature = "gdrcopy")]
        {
            // Best-effort: a host without gdrdrv still gets a working pool,
            // just without CPU-side generation stamps.
            pool.enable_stamping()?;
        }
        self.pool = Some(pool);
        Ok(())
    }

    /// Migrate the rows the *next* batch will read onto the device.
    ///
    /// Call after sampling batch N+1 and before training on batch N, so
    /// the migration overlaps compute. Returns `false` when no pool or no
    /// staging tier is attached.
    pub fn prefetch_next(&self, rows: &[u32]) -> Result<bool, Box<dyn std::error::Error>> {
        match (&self.pool, &self.prefetch_stream) {
            (Some(pool), Some(stream)) => Ok(pool.prefetch_next(rows, stream)?),
            _ => Ok(false),
        }
    }

    /// Export the pool's VRAM as a file descriptor a peer process can map,
    /// returning `(fd, bytes)`. Errors when no pool is attached.
    ///
    /// Pair with `SCM_RIGHTS` to hand a second worker the same physical
    /// feature block — one copy per GPU rather than per process.
    pub fn export_pool_fd(
        &self,
    ) -> Result<(std::os::unix::io::RawFd, usize), Box<dyn std::error::Error>> {
        let pool = self
            .pool
            .as_ref()
            .ok_or("no feature pool attached; call enable_pool first")?;
        Ok(pool.export_fd()?)
    }

    /// Gather a batch into the pool, reusing its VRAM.
    ///
    /// The returned tensor borrows the pool and is invalidated by the next
    /// call — copy it if it must outlive the batch. Errors when no pool is
    /// attached; [`Self::gather`] is the allocating alternative.
    pub fn gather_pooled(
        &mut self,
        node_ids: &[u32],
    ) -> Result<PooledGpuFeatures<'_>, Box<dyn std::error::Error>> {
        if self.pool.is_none() {
            return Err("no feature pool attached; call enable_pool first".into());
        }
        self.client.gather(node_ids)?;

        let num_nodes = node_ids.len();
        let feature_dim = self.client.feature_dim();
        let gpu_id = self.gpu_id;

        let pool = self.pool.as_mut().expect("checked above");
        pool.ensure_rows(num_nodes)?;

        // Copy the validator's compacted output into the pool. Same
        // device-to-device copy the owned path makes, minus the allocation
        // that preceded it.
        let stream = self.client.validator().stream();
        let output = self.client.validator().output();
        let src = output.slice(0..num_nodes * feature_dim);
        let (src_ptr, _sync) = src.device_ptr(stream);
        let bytes = num_nodes * feature_dim * std::mem::size_of::<f32>();
        // A raw driver call acts on the thread's current context.
        stream.context().bind_to_thread()?;
        // SAFETY: `src_ptr` is `bytes` of valid device memory on this
        // stream, and the pool has just been grown to hold `num_nodes`
        // rows, so its destination range is mapped and writable.
        let res = unsafe {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                pool.rows_ptr(),
                src_ptr,
                bytes,
                stream.cu_stream() as _,
            )
        };
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuMemcpyDtoDAsync failed: {res:?}").into());
        }
        // The rows must be in place before the batch is handed out — and
        // before a generation stamp announces them.
        stream.synchronize()?;

        #[cfg(feature = "gdrcopy")]
        let generation = pool.stamp_generation(stream)?.unwrap_or(0);
        #[cfg(not(feature = "gdrcopy"))]
        let generation = 0;

        Ok(PooledGpuFeatures {
            device_ptr: pool.rows_ptr(),
            num_nodes,
            feature_dim,
            gpu_id,
            generation,
            _pool: std::marker::PhantomData,
        })
    }

    /// Gather features for a batch of node IDs into VRAM.
    ///
    /// After seqlock validation, the compacted rows are copied
    /// device-to-device out of the validator's reused output buffer into a
    /// fresh per-call allocation, so the returned [`OwnedGpuFeatures`] owns
    /// its VRAM outright. The copy has completed when this returns.
    pub fn gather(
        &mut self,
        node_ids: &[u32],
    ) -> Result<OwnedGpuFeatures, Box<dyn std::error::Error>> {
        self.client.gather(node_ids)?;

        let num_nodes = node_ids.len();
        let feature_dim = self.client.feature_dim();
        let stream = self.client.validator().stream();
        let output = self.client.validator().output();
        let buf = if num_nodes == 0 {
            // A zero-length allocation is an error to the driver.
            stream.alloc_zeros::<f32>(1)?
        } else {
            stream.clone_dtod(&output.slice(0..num_nodes * feature_dim))?
        };
        // Consumers read through a bare pointer on streams we cannot see;
        // finish the copy before any of them can.
        stream.synchronize()?;

        Ok(OwnedGpuFeatures {
            buf: Arc::new(GatheredRows { slice: Some(buf) }),
            num_nodes,
            feature_dim,
            gpu_id: self.gpu_id,
        })
    }

    /// Feature dimension from the remote schema.
    pub fn feature_dim(&self) -> usize {
        self.client.feature_dim()
    }
}
