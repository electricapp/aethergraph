//! K2.1 IBGDA GPU WQE poster (NVRTC).

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

const KERNEL_SRC: &str = include_str!("ibgda_post.cu");
const KERNEL_NAME: &str = "ibgda_post_rdma_read";

/// Compiled IBGDA post kernel.
pub struct IbgdaPoster {
    stream: Arc<CudaStream>,
    func: CudaFunction,
}

impl IbgdaPoster {
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

    /// Post `n` RDMA READ WQEs from device arrays (see kernel signature).
    #[allow(clippy::too_many_arguments)]
    pub fn post(
        &self,
        ring: &mut CudaSlice<u8>,
        send_db: &mut CudaSlice<u32>,
        wqe_counter: &mut CudaSlice<u32>,
        qpn: u32,
        depth: u32,
        local_addrs: &CudaSlice<u64>,
        lkeys: &CudaSlice<u32>,
        byte_counts: &CudaSlice<u32>,
        remote_addrs: &CudaSlice<u64>,
        rkeys: &CudaSlice<u32>,
        n: i32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if n <= 0 {
            return Ok(());
        }
        if !depth.is_power_of_two() {
            return Err("IBGDA depth must be power of two".into());
        }
        let depth_mask = depth - 1;
        let threads = 256u32;
        // SAFETY: matches ibgda_post_rdma_read; caller sizes buffers.
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(ring)
                .arg(send_db)
                .arg(&qpn)
                .arg(&depth_mask)
                .arg(wqe_counter)
                .arg(local_addrs)
                .arg(lkeys)
                .arg(byte_counts)
                .arg(remote_addrs)
                .arg(rkeys)
                .arg(&n)
                .launch(LaunchConfig {
                    grid_dim: ((n as u32).div_ceil(threads), 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }
}
