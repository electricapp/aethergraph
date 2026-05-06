//! RDMA-backed feature gather that integrates with aethergraph-core's
//! sampling pipeline. Wraps RdmaFeatureClient to produce GPU feature
//! tensors from sampled node IDs.
//!
//! This is the bridge between graph sampling (aethergraph-core) and
//! feature serving (aether-stream). The prefetch thread in Python's
//! NeighborLoader calls `gather()` inline after sampling — takes ~20μs.

use crate::rdma::client::RdmaFeatureClient;

/// Handle to a feature tensor sitting in VRAM.
///
/// The pointer is valid until the next call to `RdmaFeatureGather::gather()`,
/// which reuses the validator's output buffer. Callers must consume or copy
/// the tensor before the next gather.
#[derive(Debug, Clone)]
pub struct GpuFeatures {
    /// Raw CUDA device pointer (`CUdeviceptr`) to the output tensor.
    /// Layout: contiguous `[num_nodes, feature_dim]` row-major f32.
    pub ptr: u64,
    /// Number of nodes (rows) in the tensor.
    pub num_nodes: usize,
    /// Feature dimension (columns).
    pub feature_dim: usize,
    /// CUDA device ordinal.
    pub gpu_id: i32,
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
    /// Returns a `GpuFeatures` handle. The underlying VRAM is reused across
    /// calls — the caller must consume (e.g., DLPack export) before the next
    /// `gather()`.
    ///
    /// `epoch_version`: pass 0 for latest features, or a version number
    /// to pin features to a specific epoch (MVCC snapshot isolation).
    pub fn gather(
        &mut self,
        node_ids: &[u32],
        epoch_version: u64,
    ) -> Result<GpuFeatures, Box<dyn std::error::Error>> {
        self.client.gather(node_ids, epoch_version)?;

        let stream = self.client.validator().stream();
        let output = self.client.validator().output();
        let (ptr, _guard) = cudarc::driver::DevicePtr::device_ptr(output, stream);

        Ok(GpuFeatures {
            ptr: ptr as u64,
            num_nodes: node_ids.len(),
            feature_dim: self.client.feature_dim(),
            gpu_id: self.gpu_id,
        })
    }

    /// Feature dimension from the remote schema.
    pub fn feature_dim(&self) -> usize {
        self.client.feature_dim()
    }
}
