//! High-level RDMA feature gather client.
//!
//! Ties together: RdmaContext → QP → GpuGatherBuffer → SeqlockValidator
//! into a single `gather(node_ids)` call that returns a device pointer
//! to a contiguous `[batch_size, feature_dim]` f32 tensor in VRAM.

use crate::feature_table::FeatureSchema;
use crate::gpu::buffer::GpuGatherBuffer;
use crate::gpu::kernel::SeqlockValidator;
use crate::rdma::context::RdmaContext;
use crate::rdma::control;
use crate::rdma::ffi::IbvWc;
use crate::rdma::qp::{RdmaQp, RdmaRead};
use cudarc::driver::{CudaContext, sys};
use std::io;
use std::time::{Duration, Instant};

/// Maximum retries for torn reads before giving up.
const MAX_RETRIES: usize = 8;

/// Wall-clock deadline for a single `post_and_wait` completion drain. A
/// healthy RDMA READ of a feature batch completes in microseconds; if no
/// signaled completion lands within this window the QP has stalled (link
/// down, remote gone) and we error out rather than spin a core forever.
const POLL_DEADLINE: Duration = Duration::from_secs(10);

/// GPUDirect RDMA feature client.
///
/// Connects to a feature server's control plane, exchanges QP endpoints,
/// then gathers features for batches of node IDs directly into VRAM.
pub struct RdmaFeatureClient {
    ctx: RdmaContext,
    qp: RdmaQp,
    buffer: GpuGatherBuffer,
    validator: SeqlockValidator,
    schema: FeatureSchema,
    remote_rkey: u32,
    remote_base: u64,
    /// Cumulative torn-slot detections across all gathers — every slot the
    /// seqlock validator rejected (first pass and retries). Nonzero means
    /// writer contention actually interleaved with RDMA reads.
    torn_slots_detected: u64,
}

impl RdmaFeatureClient {
    /// Connect to a feature server's control plane and set up GPUDirect RDMA.
    ///
    /// 1. Opens the first RDMA device using the caller-specified `gid_index`
    ///    (typically 1 for RoCEv2 IPv4-mapped GID; 0 is link-local IPv6 and
    ///    not routable over Ethernet)
    /// 2. Connects to the server's TCP control plane
    /// 3. Exchanges QP endpoints
    /// 4. Allocates VRAM staging buffer + registers with NIC
    /// 5. Compiles the GPU validation kernel
    pub fn connect(
        server_addr: &str,
        gpu_id: usize,
        max_batch_size: usize,
        gid_index: u8,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Open RDMA device
        let rdma_ctx = RdmaContext::open(max_batch_size as i32 * 2, gid_index)?;

        // Connect to server and exchange QP endpoints
        let (advertisement, qp) = control::connect_with_qp(server_addr, &rdma_ctx)?;

        let schema = advertisement.schema.clone();
        let remote_rkey = advertisement.rkey;
        let remote_base = advertisement.base_addr;

        // Initialize CUDA context + stream
        let cuda_ctx = CudaContext::new(gpu_id)
            .map_err(|e| io::Error::other(format!("CUDA init failed: {e}")))?;
        let stream = cuda_ctx.default_stream();

        // Allocate VRAM staging buffer
        let buffer = GpuGatherBuffer::new(&rdma_ctx, &cuda_ctx, &stream, max_batch_size, &schema)?;

        // Compile GPU validation kernel
        let validator =
            SeqlockValidator::new(&cuda_ctx, &stream, max_batch_size, schema.feature_dim)?;

        Ok(Self {
            ctx: rdma_ctx,
            qp,
            buffer,
            validator,
            schema,
            remote_rkey,
            remote_base,
            torn_slots_detected: 0,
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
    /// `self.validator().output()`. Use `validator().stream()` to get
    /// a properly stream-ordered device pointer.
    ///
    /// Each row is READ twice, sequentially, into two staging regions; the
    /// GPU kernel accepts a row only when both snapshots agree on version
    /// and payload (see the RDMA reader contract in `feature_table.rs`).
    /// Handles seqlock validation and automatic retry for torn reads.
    pub fn gather(&mut self, node_ids: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
        if node_ids.len() > self.buffer.max_batch_size() {
            return Err(format!(
                "node_ids.len() {} exceeds buffer.max_batch_size() {}",
                node_ids.len(),
                self.buffer.max_batch_size()
            )
            .into());
        }

        let batch_size = node_ids.len();

        // READ both snapshots for every row, then cross-validate on the GPU.
        let all_indices: Vec<usize> = (0..batch_size).collect();
        self.read_snapshots(node_ids, &all_indices)?;
        let retry_count = self.validator.validate(
            self.buffer.staging_ptr(),
            self.buffer.staging2_ptr(),
            self.buffer.slot_size(),
            batch_size,
        )?;

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

            self.read_snapshots(node_ids, &retry_indices)?;
            let remaining = self.validator.validate(
                self.buffer.staging_ptr(),
                self.buffer.staging2_ptr(),
                self.buffer.slot_size(),
                batch_size,
            )?;
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

    /// Translate a node id into the remote VRAM/host address to READ from,
    /// validating it against the advertised table bounds.
    ///
    /// Two independent guards, both required:
    ///   1. `node_id < schema.node_count` — the table holds exactly
    ///      `node_count` logical slots.
    ///   2. `(node_id + 1) * slot_size <= node_count * slot_size` via checked
    ///      arithmetic — the offset of the slot's last byte must stay inside
    ///      the region the server registered. The server registers the whole
    ///      feature table (`node_count` rounded up to a power of two, each slot
    ///      page-rounded), so `node_count * slot_size` is a conservative lower
    ///      bound on the registered MR length; staying within it guarantees the
    ///      one-sided READ never targets memory outside the MR even if a peer
    ///      supplies an out-of-range id.
    fn remote_addr_for(&self, node_id: u32) -> Result<u64, Box<dyn std::error::Error>> {
        let node_count = self.schema.node_count as u64;
        let slot_size = self.schema.slot_size as u64;
        if (node_id as u64) >= node_count {
            return Err(format!("node_id {node_id} out of range (node_count {node_count})").into());
        }
        // end_offset = (node_id + 1) * slot_size, checked.
        let end_offset = (node_id as u64)
            .checked_add(1)
            .and_then(|n| n.checked_mul(slot_size))
            .ok_or("remote offset overflow")?;
        let region_len = node_count
            .checked_mul(slot_size)
            .ok_or("region length overflow")?;
        if end_offset > region_len {
            return Err(format!(
                "node_id {node_id} slot end {end_offset} exceeds region length {region_len}"
            )
            .into());
        }
        // start = base + node_id * slot_size, checked against u64 wrap.
        let start_offset = (node_id as u64)
            .checked_mul(slot_size)
            .ok_or("remote offset overflow")?;
        self.remote_base
            .checked_add(start_offset)
            .ok_or_else(|| "remote address overflow".into())
    }

    /// READ the two sequential snapshots of each row in `indices` into the
    /// buffer's two staging regions.
    ///
    /// A row's snapshot-2 READ is posted only after its snapshot-1
    /// completion has been observed AND the GPUDirect write flush has run,
    /// so per-slot visibility in VRAM is monotone between the snapshots —
    /// the ordering the two-snapshot contract in `feature_table.rs`
    /// requires. That constraint is per ROW, not per batch: the batch is
    /// split into windows and window k's snapshot-2 posts in the same
    /// chain as window k+1's snapshot-1, keeping the NIC busy through the
    /// protocol's serialization point instead of draining to idle between
    /// two full-batch rounds. Each remote address is validated against the
    /// advertised table bounds (see `remote_addr_for`) before it's turned
    /// into a one-sided READ — a node_id past the table must never be
    /// translated into a read outside the server's registered region.
    fn read_snapshots(
        &self,
        node_ids: &[u32],
        indices: &[usize],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let lkey = self.buffer.lkey();
        // Live bytes end at tail_version's last byte; the page-rounded
        // remainder of the slot is dead padding not worth the PCIe traffic.
        let read_len = (self.schema.tail_offset_in_slot + 8) as u32;

        let build = |idx_window: &[usize],
                     snapshot: usize|
         -> Result<Vec<RdmaRead>, Box<dyn std::error::Error>> {
            idx_window
                .iter()
                .map(|&i| {
                    Ok(RdmaRead {
                        local_addr: if snapshot == 0 {
                            self.buffer.slot_addr(i)
                        } else {
                            self.buffer.slot_addr2(i)
                        },
                        local_lkey: lkey,
                        remote_addr: self.remote_addr_for(node_ids[i])?,
                        remote_rkey: self.remote_rkey,
                        length: read_len,
                    })
                })
                .collect()
        };

        if indices.is_empty() {
            return Ok(());
        }

        // Window size: combined S2(k) + S1(k+1) chains must fit the send
        // queue, and even small batches split in two so the pipeline has
        // an overlap step.
        let cap = (self.qp.max_send_wr() as usize) / 2;
        let window = indices.len().div_ceil(2).clamp(1, cap);
        let windows: Vec<&[usize]> = indices.chunks(window).collect();

        // Prologue: snapshot 1 of the first window.
        let first_s1 = build(windows[0], 0)?;
        self.post_and_wait(&first_s1)?;
        self.flush_gpudirect_writes()?;

        for k in 0..windows.len() {
            // Snapshot 2 of window k (its snapshot 1 completed and was
            // flushed in the previous iteration / prologue), chained with
            // snapshot 1 of window k+1.
            let mut combined = build(windows[k], 1)?;
            if k + 1 < windows.len() {
                combined.extend(build(windows[k + 1], 0)?);
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
        // SAFETY: no pointers involved; both enum arguments are valid, and a
        // live CUDA context exists for this process (created in `connect`).
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

    /// Post RDMA READs and busy-poll CQ until every completion arrives.
    ///
    /// Batches larger than the QP send-queue depth stream through in
    /// windows: post one window (one WR per read, only the last signaled),
    /// drain its signaled completion, post the next. The window is the QP's
    /// own `max_send_wr`, so any batch size works without over-posting
    /// `ENOMEM`.
    fn post_and_wait(&self, reads: &[RdmaRead]) -> Result<(), Box<dyn std::error::Error>> {
        let window_size = self.qp.max_send_wr() as usize;
        for window in reads.chunks(window_size) {
            self.post_and_wait_window(window)?;
        }
        Ok(())
    }

    /// Post one send-queue-sized window of READs and drain its completion.
    ///
    /// On error from an unsignaled WR, continues polling until the signaled
    /// WR's CQE is also consumed. This prevents stale CQEs from corrupting
    /// subsequent gather calls.
    fn post_and_wait_window(&self, reads: &[RdmaRead]) -> Result<(), Box<dyn std::error::Error>> {
        if reads.is_empty() {
            return Ok(());
        }

        self.qp.post_reads(reads)?;

        let signaled_wr_id = (reads.len() - 1) as u64;
        let mut first_error: Option<(u32, u32)> = None;
        let mut wc_buf = [IbvWc::default(); 16];

        // Poll until we see the signaled WR's completion (success or error).
        // Error CQEs from unsignaled WRs are consumed along the way. A deadline
        // bounds the spin so a stalled QP can't pin a core indefinitely.
        let deadline = Instant::now() + POLL_DEADLINE;
        loop {
            let n = self.qp.poll_cq(&self.ctx, &mut wc_buf)?;
            for wc in wc_buf.iter().take(n) {
                if wc.status != crate::rdma::ffi::IBV_WC_SUCCESS && first_error.is_none() {
                    first_error = Some((wc.status, wc.vendor_err));
                }
                if wc.wr_id == signaled_wr_id {
                    // Signaled WR completed — CQ is now fully drained for this batch.
                    if let Some((status, vendor_err)) = first_error {
                        return Err(format!(
                            "RDMA READ failed: status={status}, vendor_err={vendor_err}"
                        )
                        .into());
                    }
                    return Ok(());
                }
            }
            if n == 0 {
                if Instant::now() >= deadline {
                    return Err(format!(
                        "RDMA READ completion timed out after {POLL_DEADLINE:?} \
                         (signaled wr_id {signaled_wr_id} never landed)"
                    )
                    .into());
                }
                std::hint::spin_loop();
            }
        }
    }

    /// Feature dimension.
    pub fn feature_dim(&self) -> usize {
        self.schema.feature_dim
    }

    /// Schema of the remote feature table.
    pub fn schema(&self) -> &FeatureSchema {
        &self.schema
    }
}
