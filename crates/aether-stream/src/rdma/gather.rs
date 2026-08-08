//! RDMA-backed feature gather that integrates with aethergraph-core's
//! sampling pipeline. Wraps RdmaFeatureClient to produce GPU feature
//! tensors from sampled node IDs.
//!
//! This is the bridge between graph sampling (aethergraph-core) and
//! feature serving (aether-stream). The prefetch thread in Python's
//! NeighborLoader calls `gather()` inline after sampling — takes ~20μs.

use crate::rdma::client::RdmaFeatureClient;
use cudarc::driver::{CudaSlice, DevicePtr};
use std::sync::Arc;

/// Feature tensor in VRAM, owned by this handle.
///
/// The buffer is a fresh per-call allocation that the compacted validator
/// output is copied into, so the tensor stays valid across subsequent
/// `gather()` calls and after the [`RdmaFeatureGather`] drops — the VRAM is
/// freed when the last clone of `buf` is dropped.
#[derive(Debug, Clone)]
pub struct OwnedGpuFeatures {
    /// Owned device buffer holding the tensor.
    /// Layout: contiguous `[num_nodes, feature_dim]` row-major f32.
    pub buf: Arc<CudaSlice<f32>>,
    /// Number of nodes (rows) in the tensor.
    pub num_nodes: usize,
    /// Feature dimension (columns).
    pub feature_dim: usize,
    /// CUDA device ordinal.
    pub gpu_id: i32,
}

impl OwnedGpuFeatures {
    /// Raw CUDA device pointer (`CUdeviceptr`) to the tensor data, ordered
    /// on the buffer's own allocation stream.
    pub fn device_ptr(&self) -> u64 {
        let (ptr, _sync) = self.buf.device_ptr(self.buf.stream());
        ptr as u64
    }
}

/// RDMA feature gather — connects to a remote FeatureTable and pulls
/// features directly into VRAM for a batch of node IDs.
pub struct RdmaFeatureGather {
    client: RdmaFeatureClient,
    gpu_id: i32,
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
        })
    }

    /// Gather features for a batch of node IDs into VRAM.
    ///
    /// After seqlock validation, the compacted rows are copied
    /// device-to-device out of the validator's reused output buffer into a
    /// fresh per-call allocation on the same stream, so the returned
    /// [`OwnedGpuFeatures`] owns its VRAM outright.
    pub fn gather(
        &mut self,
        node_ids: &[u32],
    ) -> Result<OwnedGpuFeatures, Box<dyn std::error::Error>> {
        self.client.gather(node_ids)?;

        let num_nodes = node_ids.len();
        let feature_dim = self.client.feature_dim();
        let stream = self.client.validator().stream();
        let output = self.client.validator().output();
        let buf = stream.clone_dtod(&output.slice(0..num_nodes * feature_dim))?;

        Ok(OwnedGpuFeatures {
            buf: Arc::new(buf),
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
