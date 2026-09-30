//! K2.1 IBGDA GPU WQE poster (NVRTC).

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

pub(super) const KERNEL_SRC: &str = include_str!("ibgda_post.cu");
const KERNEL_NAME: &str = "ibgda_post_rdma_read";

/// mlx5 WQE basic-block size in bytes; one RDMA READ WQE per block.
const WQE_BB: usize = 64;

/// One batch of RDMA READs as device arrays, all the same length.
pub struct IbgdaReadBatch<'a> {
    local_addrs: &'a CudaSlice<u64>,
    lkeys: &'a CudaSlice<u32>,
    byte_counts: &'a CudaSlice<u32>,
    remote_addrs: &'a CudaSlice<u64>,
    rkeys: &'a CudaSlice<u32>,
    len: usize,
}

impl<'a> IbgdaReadBatch<'a> {
    /// Pair up the per-READ arrays; errors unless every one holds the same
    /// number of READs.
    pub fn new(
        local_addrs: &'a CudaSlice<u64>,
        lkeys: &'a CudaSlice<u32>,
        byte_counts: &'a CudaSlice<u32>,
        remote_addrs: &'a CudaSlice<u64>,
        rkeys: &'a CudaSlice<u32>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let len = local_addrs.len();
        let lens = [
            lkeys.len(),
            byte_counts.len(),
            remote_addrs.len(),
            rkeys.len(),
        ];
        if lens.iter().any(|&l| l != len) {
            return Err(format!(
                "IBGDA batch arrays differ in length: {len} local addresses vs {lens:?}"
            )
            .into());
        }
        Ok(Self {
            local_addrs,
            lkeys,
            byte_counts,
            remote_addrs,
            rkeys,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Compiled IBGDA post kernel plus the queue state it posts against.
///
/// The poster owns the claim and publish counters and tracks, on the host,
/// how many WQEs are outstanding: a batch that would overwrite a slot the
/// HCA has not consumed is refused before launch. Retire completed WQEs
/// with [`Self::retire`] as their CQEs are polled.
pub struct IbgdaPoster {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    qpn: u32,
    depth: u32,
    /// Next WQE index to claim (device-side counter).
    claimed: CudaSlice<u32>,
    /// WQEs published through the doorbell record (device-side counter).
    ready: CudaSlice<u32>,
    posted: u64,
    retired: u64,
}

impl IbgdaPoster {
    /// Compile the kernel for a send queue of `depth` WQEs on QP `qpn`.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        qpn: u32,
        depth: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if depth == 0 || !depth.is_power_of_two() || depth > 1 << 16 {
            return Err("IBGDA depth must be a power of two in 1..=65536".into());
        }
        if qpn > 0x00ff_ffff {
            return Err("mlx5 QPN exceeds 24 bits".into());
        }
        let module = ctx.load_module(super::compile_for_device(ctx, KERNEL_SRC)?)?;
        Ok(Self {
            stream: stream.clone(),
            func: module.load_function(KERNEL_NAME)?,
            qpn,
            depth,
            claimed: stream.alloc_zeros(1)?,
            ready: stream.alloc_zeros(1)?,
            posted: 0,
            retired: 0,
        })
    }

    /// WQEs posted and not yet retired.
    pub fn in_flight(&self) -> u64 {
        self.posted - self.retired
    }

    /// Mark `n` WQEs complete (their CQEs were polled), freeing their slots.
    pub fn retire(&mut self, n: u64) -> Result<(), Box<dyn std::error::Error>> {
        if n > self.in_flight() {
            return Err(
                format!("retiring {n} WQEs but only {} in flight", self.in_flight()).into(),
            );
        }
        self.retired += n;
        Ok(())
    }

    /// Post one RDMA READ WQE per entry of `batch` into `ring` (the send
    /// queue, `depth` 64-byte blocks) and publish them through `dbr` (the
    /// two-word doorbell record).
    pub fn post(
        &mut self,
        ring: &mut CudaSlice<u8>,
        dbr: &mut CudaSlice<u32>,
        batch: &IbgdaReadBatch<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if batch.is_empty() {
            return Ok(());
        }
        if ring.len() < self.depth as usize * WQE_BB {
            return Err(format!(
                "ring holds {} bytes, needs {} for {} WQEs",
                ring.len(),
                self.depth as usize * WQE_BB,
                self.depth
            )
            .into());
        }
        if dbr.len() < 2 {
            return Err("doorbell record needs two words".into());
        }
        let free = u64::from(self.depth) - self.in_flight();
        if batch.len() as u64 > free {
            return Err(format!(
                "{} WQEs exceed the {free} free send-queue slots; retire completions first",
                batch.len()
            )
            .into());
        }
        let n = i32::try_from(batch.len())?;
        let depth_mask = self.depth - 1;
        let threads = 256u32;
        // SAFETY: matches ibgda_post_rdma_read. Every array holds `n`
        // entries (checked by `IbgdaReadBatch`), the ring holds `depth`
        // blocks, the record two words, and at most `free` indices are
        // claimed, so no slot the HCA still owns is overwritten.
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(ring)
                .arg(dbr)
                .arg(&self.qpn)
                .arg(&depth_mask)
                .arg(&mut self.claimed)
                .arg(&mut self.ready)
                .arg(batch.local_addrs)
                .arg(batch.lkeys)
                .arg(batch.byte_counts)
                .arg(batch.remote_addrs)
                .arg(batch.rkeys)
                .arg(&n)
                .launch(LaunchConfig {
                    grid_dim: ((n as u32).div_ceil(threads), 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        self.posted += batch.len() as u64;
        Ok(())
    }
}
