//! K5.2 warp-cooperative neighbor sampling.
//!
//! One warp owns one seed. Algorithm R (`fanout <= 32`) streams neighbors
//! through a shared 32-wide `ld.global.cs` window; reservoir state stays in
//! registers with `__ballot_sync` replacement. Bit-parity vs
//! [`aethergraph_core::reservoir_sample`] holds for that path.
//!
//! Larger fanouts use with-replacement Philox draws (not Algorithm R) — do
//! not diff those outputs against `reservoir_sample`.
//!
//! The graph and the output are typed: [`DeviceCsr`] is only built from a
//! CSR checked on the host, and [`SampleBatch`] carries the per-row counts
//! alongside the samples, so a launch cannot read past the neighbor array
//! or leave a short row indistinguishable from edges to node 0.
//!
//! TODO(HARDWARE): C-tree arena walker + NeighborSampler parity on-box.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

pub(super) const KERNEL_SRC: &str = concat!(
    include_str!("../common.cuh"),
    "\n",
    include_str!("sampler.cu")
);
const KERNEL_NAME: &str = "warp_sample_neighbors";

/// Ceiling on the sampler grid; the kernel grid-strides past it.
const MAX_SAMPLE_BLOCKS: u32 = 65_535;

/// Value in every output slot past a row's count. Mirrors
/// `AETHER_SAMPLE_PAD` in sampler.cu.
pub const SAMPLE_PAD: u32 = u32::MAX;

/// CPU reference RNG used by the sampler device code.
pub use aethergraph_core::philox4x32_10 as philox;

/// A CSR adjacency resident in VRAM, checked on upload: offsets start at 0,
/// never decrease, and end at the neighbor count, so every row's range lies
/// inside the neighbor array.
pub struct DeviceCsr {
    offsets: CudaSlice<u64>,
    neighbors: CudaSlice<u32>,
    num_nodes: u64,
}

impl DeviceCsr {
    /// Validate `offsets`/`neighbors` and upload them onto `stream`.
    pub fn upload(
        stream: &Arc<CudaStream>,
        offsets: &[u64],
        neighbors: &[u32],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let Some((&first, _)) = offsets.split_first() else {
            return Err("CSR offsets must hold num_nodes + 1 entries".into());
        };
        if first != 0 {
            return Err(format!("CSR offsets must start at 0, not {first}").into());
        }
        if let Some(w) = offsets.windows(2).position(|w| w[1] < w[0]) {
            return Err(format!("CSR offsets decrease at row {w}").into());
        }
        let last = offsets[offsets.len() - 1];
        if last != neighbors.len() as u64 {
            return Err(format!(
                "CSR offsets end at {last}, but there are {} neighbors",
                neighbors.len()
            )
            .into());
        }
        let mut d_offsets = stream.alloc_zeros::<u64>(offsets.len())?;
        stream.memcpy_htod(offsets, &mut d_offsets)?;
        let mut d_neighbors = stream.alloc_zeros::<u32>(neighbors.len().max(1))?;
        if !neighbors.is_empty() {
            stream.memcpy_htod(neighbors, &mut d_neighbors)?;
        }
        Ok(Self {
            offsets: d_offsets,
            neighbors: d_neighbors,
            num_nodes: (offsets.len() - 1) as u64,
        })
    }

    /// Rows in the graph.
    #[must_use]
    pub fn num_nodes(&self) -> u64 {
        self.num_nodes
    }
}

/// Output of one sample: `rows * fanout` neighbor IDs, row-major, plus one
/// count per row. Slots past a row's count hold [`SAMPLE_PAD`].
pub struct SampleBatch {
    neighbors: CudaSlice<u32>,
    counts: CudaSlice<u32>,
    rows: usize,
    fanout: usize,
}

impl SampleBatch {
    /// Allocate room for up to `rows` seeds at `fanout` samples each.
    pub fn new(
        stream: &Arc<CudaStream>,
        rows: usize,
        fanout: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if rows == 0 || fanout == 0 {
            return Err("a sample batch needs at least one row and one sample".into());
        }
        if i32::try_from(rows).is_err() || i32::try_from(fanout).is_err() {
            return Err(format!("batch {rows}x{fanout} exceeds the kernel's i32 extents").into());
        }
        let slots = rows.checked_mul(fanout).ok_or("rows * fanout overflows")?;
        Ok(Self {
            neighbors: stream.alloc_zeros(slots)?,
            counts: stream.alloc_zeros(rows)?,
            rows,
            fanout,
        })
    }

    /// Sampled neighbor IDs, `rows * fanout`, row-major.
    pub fn neighbors(&self) -> &CudaSlice<u32> {
        &self.neighbors
    }

    /// Samples written per row; the rest of the row is [`SAMPLE_PAD`].
    pub fn counts(&self) -> &CudaSlice<u32> {
        &self.counts
    }

    /// Seed capacity.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Samples per row.
    #[must_use]
    pub fn fanout(&self) -> usize {
        self.fanout
    }
}

/// One sample request's scalars. Grouped so the `u64`/`u32` values cannot
/// be transposed at a call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleRequest {
    /// Philox key, with `layer`, making the draw reproducible on the CPU.
    pub seed: u64,
    /// Sampling layer, part of the Philox key.
    pub layer: u32,
}

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
        let module = ctx.load_module(super::compile_for_device(ctx, KERNEL_SRC)?)?;
        Ok(Self {
            stream: stream.clone(),
            func: module.load_function(KERNEL_NAME)?,
        })
    }

    /// Enqueue a warp-per-seed sample of every seed in `seeds` into `out`,
    /// `out.fanout()` neighbors each.
    ///
    /// A seed outside the graph samples nothing (count 0) rather than
    /// reading past the offsets.
    pub fn sample(
        &self,
        csr: &DeviceCsr,
        seeds: &CudaSlice<u64>,
        out: &mut SampleBatch,
        req: SampleRequest,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = seeds.len();
        if node_count == 0 {
            return Ok(());
        }
        if node_count > out.rows {
            return Err(format!("{node_count} seeds exceed the batch's {} rows", out.rows).into());
        }
        let node_count_i32 = i32::try_from(node_count)?;
        let fanout_i32 = i32::try_from(out.fanout)?;
        let (seed, layer) = (req.seed, req.layer);
        // One warp per seed; the kernel grid-strides past the cap.
        let threads = 256u32;
        let blocks = u32::try_from(node_count.div_ceil((threads / 32) as usize))
            .unwrap_or(u32::MAX)
            .clamp(1, MAX_SAMPLE_BLOCKS);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: argument order matches the kernel. `csr` is a validated
        // CSR, so every in-graph row's range lies inside its neighbors; the
        // kernel skips seeds >= num_nodes; `out` holds `node_count` rows of
        // `fanout` slots and counts.
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(&csr.offsets)
                .arg(&csr.neighbors)
                .arg(&csr.num_nodes)
                .arg(seeds)
                .arg(&mut out.neighbors)
                .arg(&mut out.counts)
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
