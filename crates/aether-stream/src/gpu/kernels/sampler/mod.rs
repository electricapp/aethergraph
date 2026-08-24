//! K5.2 warp-cooperative neighbor sampling.
//!
//! One warp owns one seed. Algorithm R (`fanout <= 32`) streams neighbors
//! through a shared 32-wide `ld.global.cs` window; reservoir state stays in
//! registers with `__ballot_sync` replacement. Larger fanouts use
//! with-replacement Philox draws. Bit-parity vs [`aethergraph_core::reservoir_sample`].
//!
//! TODO(HARDWARE): C-tree arena walker + NeighborSampler parity on-box.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

const KERNEL_SRC: &str = concat!(
    include_str!("../common.cuh"),
    "\n",
    include_str!("sampler.cu")
);
const KERNEL_NAME: &str = "warp_sample_neighbors";

/// CPU reference RNG used by the sampler device code.
pub use aethergraph_core::philox4x32_10 as philox;

/// Compiled warp sampler (opt-in; not the production C-tree path yet).
pub struct WarpSampler {
    stream: Arc<CudaStream>,
    func: CudaFunction,
}

impl WarpSampler {
    /// Compile the CSR sampler kernel once for this CUDA context.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let module = ctx.load_module(cudarc::nvrtc::compile_ptx(KERNEL_SRC)?)?;
        Ok(Self {
            stream: stream.clone(),
            func: module.load_function(KERNEL_NAME)?,
        })
    }

    /// Enqueue a warp-per-row sample.
    ///
    /// `offsets` is a u64 CSR offset table, `nodes` identifies source rows,
    /// and `output` must contain `node_count * fanout` u32 elements.
    pub fn sample(
        &self,
        offsets: &CudaSlice<u64>,
        neighbors: &CudaSlice<u32>,
        nodes: &CudaSlice<u64>,
        output: &mut CudaSlice<u32>,
        node_count: usize,
        fanout: usize,
        seed: u64,
        layer: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if node_count == 0 || fanout == 0 {
            return Ok(());
        }
        let node_count_i32 = i32::try_from(node_count)?;
        let fanout_i32 = i32::try_from(fanout)?;
        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((node_count as u32 * 32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: argument order matches the kernel; callers own checked buffers.
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(offsets)
                .arg(neighbors)
                .arg(nodes)
                .arg(output)
                .arg(&node_count_i32)
                .arg(&fanout_i32)
                .arg(&seed)
                .arg(&layer)
                .launch(cfg)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::philox;

    #[test]
    fn sampler_rng_is_deterministic() {
        assert_eq!(philox(9, 2, 7), philox(9, 2, 7));
    }
}
