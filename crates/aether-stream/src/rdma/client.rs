//! High-level RDMA feature gather client.
//!
//! Ties together: RdmaContext → QP → GpuGatherBuffer → SeqlockValidator
//! into a single `gather(node_ids)` call that returns a device pointer
//! to a contiguous `[batch_size, feature_dim]` f32 tensor in VRAM.

use crate::gpu::buffer::GpuGatherBuffer;
use crate::gpu::kernel::SeqlockValidator;
use crate::rdma::context::{RdmaContext, closest_device_to_pci};
use crate::rdma::control;
use crate::rdma::ffi::{IBV_WC_SUCCESS, IbvWc};
use crate::rdma::layout::RemoteTable;
use crate::rdma::qp::{DEFAULT_QP_CAP, RdmaQp, RdmaRead, next_wr_generation, required_cq_depth};
use cudarc::driver::{CudaContext, sys};
use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Maximum retries for torn reads before giving up.
const MAX_RETRIES: usize = 8;

/// Wall-clock deadline for one window's signaled completion. A healthy
/// RDMA READ of a feature batch completes in microseconds, and RC retries
/// give up and flush the QP well inside this; only a hung device reaches it.
const POLL_DEADLINE: Duration = Duration::from_secs(10);

/// Bound on draining a stopped QP; see [`RdmaQp::quiesce`].
const QUIESCE_DEADLINE: Duration = Duration::from_secs(30);

/// GPUDirect RDMA feature client.
///
/// Connects to a feature server's control plane, exchanges QP endpoints,
/// then gathers features for batches of node IDs directly into VRAM.
///
/// A READ that fails, a post that is refused, or a completion that never
/// arrives stops the QP — every READ it was given has finished or been
/// flushed before the error returns — and the client refuses further work.
/// An RC QP in the error state needs a fresh connection.
pub struct RdmaFeatureClient {
    // The QP drops before the staging it targets.
    qp: RdmaQp,
    buffer: GpuGatherBuffer,
    validator: SeqlockValidator,
    ctx: RdmaContext,
    table: RemoteTable,
    cuda_ctx: Arc<CudaContext>,
    /// Cumulative torn-slot detections across all gathers — every slot the
    /// seqlock validator rejected (first pass and retries). Nonzero means
    /// writer contention actually interleaved with RDMA reads.
    torn_slots_detected: u64,
    /// Tags each window's `wr_id`s so a completion is matched to the window
    /// that posted it.
    generation: u32,
    /// Set once the QP has been stopped after a failure.
    failed: bool,
}

/// PCI address (`dddd:bb:dd.0`) of the GPU behind `ctx`, from the driver.
fn gpu_pci_bus_id(ctx: &CudaContext) -> Option<String> {
    use sys::CUdevice_attribute as A;
    let domain = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID).ok()?;
    let bus = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID).ok()?;
    let device = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID).ok()?;
    Some(format!("{domain:04x}:{bus:02x}:{device:02x}.0"))
}

impl RdmaFeatureClient {
    /// Connect to a feature server's control plane and set up GPUDirect
    /// RDMA through the NIC nearest `gpu_id` in the PCIe topology.
    ///
    /// `gid_index` is the local GID table index — typically 1 for the RoCEv2
    /// IPv4-mapped GID; 0 is link-local IPv6 and not routable over Ethernet.
    /// See [`Self::connect_on_device`] to name the NIC explicitly.
    pub fn connect(
        server_addr: &str,
        gpu_id: usize,
        max_batch_size: usize,
        gid_index: u8,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::connect_on_device(server_addr, gpu_id, max_batch_size, gid_index, None)
    }

    /// [`Self::connect`] on RDMA device `device_index` (see
    /// [`crate::rdma::context::enumerate_devices`]); `None` picks the device
    /// sharing the deepest PCIe bridge with the GPU, else device 0.
    ///
    /// 1. Opens the RDMA device, sized so a flushed QP cannot overrun its CQ
    /// 2. Connects to the server's TCP control plane, parses its table, and
    ///    exchanges QP endpoints
    /// 3. Allocates the VRAM staging buffer and registers it with the NIC
    /// 4. Compiles the GPU validation kernel on its own stream
    pub fn connect_on_device(
        server_addr: &str,
        gpu_id: usize,
        max_batch_size: usize,
        gid_index: u8,
        device_index: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cuda_ctx = CudaContext::new(gpu_id)
            .map_err(|e| io::Error::other(format!("CUDA init failed: {e}")))?;
        let device_index = device_index
            .or_else(|| gpu_pci_bus_id(&cuda_ctx).and_then(|bdf| closest_device_to_pci(&bdf)))
            .unwrap_or(0);

        // The client QP is the only one on this context's CQ.
        let rdma_ctx = RdmaContext::open_on_device(
            required_cq_depth(&DEFAULT_QP_CAP),
            device_index,
            gid_index,
        )?;
        let (table, qp) = control::connect_with_qp(server_addr, &rdma_ctx)?;

        // A stream of our own: capturable for the validate graph, and not
        // serialized behind whatever else runs on the legacy stream.
        let stream = cuda_ctx.new_stream()?;
        let buffer = GpuGatherBuffer::new(
            &rdma_ctx,
            &cuda_ctx,
            &stream,
            max_batch_size,
            table.geometry(),
        )?;
        let validator =
            SeqlockValidator::new(&cuda_ctx, &stream, max_batch_size, table.geometry())?;

        Ok(Self {
            qp,
            buffer,
            validator,
            ctx: rdma_ctx,
            table,
            cuda_ctx,
            torn_slots_detected: 0,
            generation: 0,
            failed: false,
        })
    }

    /// Cumulative number of torn-slot detections across all gathers on this
    /// client. Observability hook for writer/reader contention: each unit is
    /// one slot validation the seqlock kernel rejected and the gather loop
    /// re-read.
    pub fn torn_slots_detected(&self) -> u64 {
        self.torn_slots_detected
    }

    /// Gather features for a batch of node IDs into VRAM.
    ///
    /// After this call, the validated output tensor is available via
    /// `self.validator().output()`, complete (the validator's stream has
    /// been synchronized).
    ///
    /// Each row is READ twice, sequentially, into two staging regions; the
    /// GPU kernel accepts a row only when both snapshots agree on version
    /// and payload (see the RDMA reader contract in `feature_table.rs`).
    /// Handles seqlock validation and automatic retry for torn reads. Every
    /// node id is checked against the table before anything is posted.
    pub fn gather(&mut self, node_ids: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
        if self.failed {
            return Err("RDMA QP failed on an earlier gather; reconnect".into());
        }
        let remote: Vec<u64> = node_ids
            .iter()
            .map(|&id| self.table.slot_addr(u64::from(id)))
            .collect::<io::Result<_>>()?;

        if node_ids.len() > self.buffer.max_batch_size() {
            // One amortized re-reg + VRAM realloc — not per tensor, only when
            // the connect-time ceiling is actually exceeded.
            let stream = self.validator.stream().clone();
            self.buffer.ensure_capacity(
                &self.ctx,
                &stream,
                node_ids.len(),
                self.table.geometry(),
            )?;
            self.validator.ensure_capacity(node_ids.len())?;
        }

        let batch_size = node_ids.len();

        // READ both snapshots for every row, then cross-validate on the GPU.
        let all_indices: Vec<usize> = (0..batch_size).collect();
        self.read_snapshots(&remote, &all_indices)?;
        let retry_count = self
            .validator
            .validate(&self.buffer.staging_regions()?, batch_size)?;

        if retry_count == 0 {
            return Ok(());
        }
        self.torn_slots_detected += retry_count as u64;

        // Retry loop for torn reads — both snapshots are re-read for every
        // failed row.
        for _ in 0..MAX_RETRIES {
            let retry_indices = self.validator.retry_indices(batch_size)?;
            if retry_indices.is_empty() {
                break;
            }

            self.read_snapshots(&remote, &retry_indices)?;
            let remaining = self
                .validator
                .validate(&self.buffer.staging_regions()?, batch_size)?;
            self.torn_slots_detected += remaining as u64;
            if remaining == 0 {
                return Ok(());
            }
        }

        // Exhausted MAX_RETRIES with slots still torn. Returning Ok here would
        // hand the caller a buffer containing inconsistent (partially-written)
        // feature rows, which the GPU consumes as if valid. Surface the failure
        // instead so the caller can back off or re-issue.
        Err(format!("seqlock validation did not converge after {MAX_RETRIES} retries").into())
    }

    /// READ the two sequential snapshots of each row in `indices` into the
    /// buffer's two staging regions. `remote[i]` is row `i`'s slot address,
    /// already checked against the table.
    ///
    /// A row's snapshot-2 READ is posted only after its snapshot-1
    /// completion has been observed AND the GPUDirect write flush has run,
    /// so per-slot visibility in VRAM is monotone between the snapshots —
    /// the ordering the two-snapshot contract in `feature_table.rs`
    /// requires. That constraint is per ROW, not per batch: the batch is
    /// split into windows and window k's snapshot-2 posts in the same
    /// chain as window k+1's snapshot-1, keeping the NIC busy through the
    /// protocol's serialization point instead of draining to idle between
    /// two full-batch rounds.
    fn read_snapshots(
        &mut self,
        remote: &[u64],
        indices: &[usize],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if indices.is_empty() {
            return Ok(());
        }
        let lkey = self.buffer.lkey();
        let rkey = self.table.rkey();
        // Live bytes end at tail_version's last byte; the rest of the slot
        // stride is padding not worth the PCIe traffic.
        let read_len = self.table.geometry().live_len();

        let build = |buffer: &GpuGatherBuffer, idx_window: &[usize], snapshot: usize| {
            idx_window
                .iter()
                .map(|&i| RdmaRead {
                    local_addr: if snapshot == 0 {
                        buffer.slot_addr(i)
                    } else {
                        buffer.slot_addr2(i)
                    },
                    local_lkey: lkey,
                    remote_addr: remote[i],
                    remote_rkey: rkey,
                    length: read_len,
                })
                .collect::<Vec<_>>()
        };

        // Window size: combined S2(k) + S1(k+1) chains must fit the send
        // queue, and even small batches split in two so the pipeline has
        // an overlap step.
        let cap = (self.qp.max_send_wr() as usize / 2).max(1);
        let window = indices.len().div_ceil(2).clamp(1, cap);
        let windows: Vec<&[usize]> = indices.chunks(window).collect();

        // Prologue: snapshot 1 of the first window.
        let first_s1 = build(&self.buffer, windows[0], 0);
        self.post_and_wait(&first_s1)?;
        self.flush_gpudirect_writes()?;

        for k in 0..windows.len() {
            // Snapshot 2 of window k (its snapshot 1 completed and was
            // flushed in the previous iteration / prologue), chained with
            // snapshot 1 of window k+1.
            let mut combined = build(&self.buffer, windows[k], 1);
            if k + 1 < windows.len() {
                combined.extend(build(&self.buffer, windows[k + 1], 0));
            }
            self.post_and_wait(&combined)?;
            self.flush_gpudirect_writes()?;
        }
        Ok(())
    }

    /// Make third-party DMA writes (the NIC's RDMA READ completions) visible
    /// to subsequently launched device work. A CPU-observed CQE orders the
    /// data for the CPU, not for the GPU; this flush closes that gap. Cheap
    /// no-op on platforms with native GPUDirect write ordering.
    fn flush_gpudirect_writes(&self) -> Result<(), Box<dyn std::error::Error>> {
        // The flush acts on the calling thread's current context, and a
        // gather may run on any thread: make it ours.
        self.cuda_ctx.bind_to_thread()?;
        // SAFETY: no pointers involved; both enum arguments are valid, and
        // this client's context is current on the calling thread.
        let res = unsafe {
            sys::cuFlushGPUDirectRDMAWrites(
                sys::CUflushGPUDirectRDMAWritesTarget::CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                sys::CUflushGPUDirectRDMAWritesScope::CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER,
            )
        };
        // CUDA_ERROR_NOT_SUPPORTED means the platform does not expose the
        // flush (CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST absent) —
        // remote-write visibility is then governed by the device's native
        // ordering, so there is nothing to flush.
        if res != sys::CUresult::CUDA_SUCCESS && res != sys::CUresult::CUDA_ERROR_NOT_SUPPORTED {
            return Err(format!("cuFlushGPUDirectRDMAWrites failed: {res:?}").into());
        }
        Ok(())
    }

    /// Access the validator (for stream-ordered access to the output tensor).
    pub fn validator(&self) -> &SeqlockValidator {
        &self.validator
    }

    /// Post RDMA READs and busy-poll the CQ until every completion arrives.
    ///
    /// Batches larger than the QP send-queue depth stream through in
    /// windows: post one window (one WR per read, only the last signaled),
    /// drain its signaled completion, post the next. The window is the QP's
    /// own `max_send_wr`, so any batch size works without over-posting
    /// `ENOMEM`.
    fn post_and_wait(&mut self, reads: &[RdmaRead]) -> Result<(), Box<dyn std::error::Error>> {
        let window_size = self.qp.max_send_wr() as usize;
        for window in reads.chunks(window_size) {
            if let Err(e) = self.post_and_wait_window(window) {
                // Nothing this QP was given may still be in flight when the
                // error reaches a caller that could reuse the staging.
                self.qp.quiesce(self.ctx.cq(), QUIESCE_DEADLINE);
                self.failed = true;
                return Err(e);
            }
        }
        Ok(())
    }

    /// Post one send-queue-sized window of READs and wait for its signaled
    /// completion. RC completes in order, so that completion covers every
    /// unsignaled READ before it; any error completion ends the wait (the
    /// caller quiesces the QP, reaping the flushed rest).
    fn post_and_wait_window(
        &mut self,
        reads: &[RdmaRead],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if reads.is_empty() {
            return Ok(());
        }
        self.generation = next_wr_generation(self.generation);
        let base = u64::from(self.generation) << 32;
        self.qp.post_reads_tagged(reads, reads.len(), base)?;
        let signaled_wr_id = base + (reads.len() - 1) as u64;

        let mut wc_buf = [IbvWc::default(); 16];
        let deadline = Instant::now() + POLL_DEADLINE;
        loop {
            let n = self.ctx.cq().poll(&mut wc_buf)?;
            for wc in &wc_buf[..n] {
                if wc.status != IBV_WC_SUCCESS {
                    return Err(format!(
                        "RDMA READ failed: status={}, vendor_err={}, wr_id={:#x}",
                        wc.status, wc.vendor_err, wc.wr_id
                    )
                    .into());
                }
                if wc.wr_id == signaled_wr_id {
                    return Ok(());
                }
            }
            if n == 0 {
                if Instant::now() >= deadline {
                    return Err(format!(
                        "RDMA READ completion timed out after {POLL_DEADLINE:?} \
                         (signaled wr_id {signaled_wr_id:#x} never landed)"
                    )
                    .into());
                }
                std::hint::spin_loop();
            }
        }
    }

    /// Feature dimension.
    pub fn feature_dim(&self) -> usize {
        self.table.geometry().feature_dim()
    }

    /// The remote feature table, as parsed from the server's advertisement.
    pub fn table(&self) -> &RemoteTable {
        &self.table
    }

    /// The CUDA context the gathers land in.
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_ctx
    }
}
