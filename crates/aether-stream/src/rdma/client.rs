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

/// Maximum retries for torn reads before giving up.
const MAX_RETRIES: usize = 8;

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
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("CUDA init failed: {e}")))?;
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
        assert!(node_ids.len() <= self.buffer.max_batch_size());

        let batch_size = node_ids.len();
        let lkey = self.buffer.lkey();

        // Build initial RDMA READ list for all nodes
        let reads: Vec<RdmaRead> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &node_id)| RdmaRead {
                local_addr: self.buffer.slot_addr(i),
                local_lkey: lkey,
                remote_addr: self.remote_base + (node_id as u64) * (self.schema.slot_size as u64),
                remote_rkey: self.remote_rkey,
                length: self.schema.slot_size as u32,
            })
            .collect();

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

            // Re-post reads only for failed nodes
            let retry_reads: Vec<RdmaRead> = retry_indices
                .iter()
                .map(|&i| RdmaRead {
                    local_addr: self.buffer.slot_addr(i),
                    local_lkey: lkey,
                    remote_addr: self.remote_base
                        + (node_ids[i] as u64) * (self.schema.slot_size as u64),
                    remote_rkey: self.remote_rkey,
                    length: self.schema.slot_size as u32,
                })
                .collect();

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

        Ok(())
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
        // Error CQEs from unsignaled WRs are consumed along the way.
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
