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
use cudarc::driver::CudaContext;
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
        })
    }

    /// Gather features for a batch of node IDs into VRAM.
    ///
    /// After this call, the validated output tensor is available via
    /// `self.validator().output()`. Use `validator().stream()` to get
    /// a properly stream-ordered device pointer.
    ///
    /// `epoch_version`: MVCC epoch pin. Pass 0 to accept any consistent read
    /// (latest features). Pass a non-zero version to reject features written
    /// after that version — ensures training sees a stable feature snapshot
    /// for the entire epoch even while writers update the FeatureTable.
    ///
    /// Handles seqlock validation and automatic retry for torn reads.
    pub fn gather(
        &mut self,
        node_ids: &[u32],
        epoch_version: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if node_ids.len() > self.buffer.max_batch_size() {
            return Err(format!(
                "node_ids.len() {} exceeds buffer.max_batch_size() {}",
                node_ids.len(),
                self.buffer.max_batch_size()
            )
            .into());
        }
        // QP send-queue rate limit: a single `post_reads` chain posts one
        // WR per node and signals only the last one. The QP's
        // `max_send_wr` is `DEFAULT_QP_CAP.max_send_wr` (4096); posting
        // more would overflow with `ENOMEM`. Reject upfront so the
        // failure mode is observable, not silent. For batches larger
        // than this cap, the caller should chunk OR construct the
        // client with a deeper QP cap via `RdmaQp::create_with_cqs` and
        // a custom `IbvQpCap`.
        //
        // TODO(perf): support streaming posts — post N, drain N
        // completions via `signal_every_n`, post next N. Requires
        // bookkeeping that splits `post_and_wait` into post + drain
        // primitives and tracks the inflight wr_id window.
        const MAX_INFLIGHT_WR: usize = crate::rdma::qp::DEFAULT_QP_CAP.max_send_wr as usize;
        if node_ids.len() > MAX_INFLIGHT_WR {
            return Err(format!(
                "node_ids.len() {} exceeds QP max_send_wr {}; chunk the batch or \
                 build the QP with a deeper send queue",
                node_ids.len(),
                MAX_INFLIGHT_WR,
            )
            .into());
        }

        let batch_size = node_ids.len();
        let lkey = self.buffer.lkey();

        // Build initial RDMA READ list for all nodes. Each remote address is
        // validated against the advertised table bounds (see `remote_addr_for`)
        // before it's turned into a one-sided READ — a node_id past the table
        // must never be translated into a read outside the server's registered
        // region.
        let reads: Vec<RdmaRead> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &node_id)| {
                Ok(RdmaRead {
                    local_addr: self.buffer.slot_addr(i),
                    local_lkey: lkey,
                    remote_addr: self.remote_addr_for(node_id)?,
                    remote_rkey: self.remote_rkey,
                    length: self.schema.slot_size as u32,
                })
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

        // Post RDMA READs and wait for completion
        self.post_and_wait(&reads)?;

        // Validate on GPU
        let retry_count = self.validator.validate(
            self.buffer.staging_ptr(),
            self.buffer.slot_size(),
            batch_size,
            epoch_version,
        )?;

        if retry_count == 0 {
            return Ok(());
        }

        // Retry loop for torn reads
        for _ in 0..MAX_RETRIES {
            let retry_indices = self.validator.retry_indices(batch_size)?;
            if retry_indices.is_empty() {
                break;
            }

            // Re-post reads only for failed nodes. node_ids were already
            // bounds-checked when the initial batch was built, but re-validate
            // here so this loop is correct independent of the caller above.
            let retry_reads: Vec<RdmaRead> = retry_indices
                .iter()
                .map(|&i| {
                    Ok(RdmaRead {
                        local_addr: self.buffer.slot_addr(i),
                        local_lkey: lkey,
                        remote_addr: self.remote_addr_for(node_ids[i])?,
                        remote_rkey: self.remote_rkey,
                        length: self.schema.slot_size as u32,
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

            self.post_and_wait(&retry_reads)?;
            let remaining = self.validator.validate(
                self.buffer.staging_ptr(),
                self.buffer.slot_size(),
                batch_size,
                epoch_version,
            )?;
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

    /// Access the validator (for stream-ordered access to the output tensor).
    pub fn validator(&self) -> &SeqlockValidator {
        &self.validator
    }

    /// Post RDMA READs and busy-poll CQ until the signaled completion arrives.
    ///
    /// On error from an unsignaled WR, continues polling until the signaled
    /// WR's CQE is also consumed. This prevents stale CQEs from corrupting
    /// subsequent gather calls.
    fn post_and_wait(&self, reads: &[RdmaRead]) -> Result<(), Box<dyn std::error::Error>> {
        if reads.is_empty() {
            return Ok(());
        }

        self.qp.post_reads(reads)?;

        let signaled_wr_id = (reads.len() - 1) as u64;
        let mut first_error: Option<(u32, u32)> = None;
        let mut wc_buf = [unsafe { std::mem::zeroed::<IbvWc>() }; 16];

        // Poll until we see the signaled WR's completion (success or error).
        // Error CQEs from unsignaled WRs are consumed along the way. A deadline
        // bounds the spin so a stalled QP can't pin a core indefinitely.
        let deadline = Instant::now() + POLL_DEADLINE;
        loop {
            let n = self.qp.poll_cq(&self.ctx, &mut wc_buf)?;
            for i in 0..n {
                let wc = &wc_buf[i];
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
