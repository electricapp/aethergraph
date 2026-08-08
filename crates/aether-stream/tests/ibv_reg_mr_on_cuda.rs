//! Registering a CUDA device pointer with the HCA.
//!
//! Registers 1 MiB of VRAM via `reg_mr_cuda` — nvidia-peermem
//! (`ibv_reg_mr` on the device VA) with a dma-buf fallback
//! (`ibv_reg_dmabuf_mr`) — and asserts the returned MR has non-zero
//! `lkey` / `rkey`. This is the cheapest signal that GPUDirect is
//! actually wired up on a host that has CUDA and ibverbs individually
//! working.
//!
//! Skipped automatically if either RDMA or CUDA is missing.

#![cfg(all(target_os = "linux", feature = "rdma", feature = "gpudirect"))]

use aether_stream::gpu::buffer::reg_mr_cuda;
use aether_stream::rdma::context::RdmaContext;
use aether_stream::rdma::ffi::IBV_ACCESS_LOCAL_WRITE;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtrMut};

const ROCE_V2_GID_INDEX: u8 = 1;

#[test]
fn ibv_reg_mr_on_cuda_device_pointer() {
    if std::env::var("AETHER_SKIP_RDMA").is_ok() {
        eprintln!("skipping: AETHER_SKIP_RDMA set");
        return;
    }
    let Ok(rdma_ctx) = RdmaContext::open(16, ROCE_V2_GID_INDEX) else {
        eprintln!("skipping: no RDMA device");
        return;
    };
    let Ok(cuda_ctx) = CudaContext::new(0) else {
        eprintln!("skipping: no CUDA device");
        return;
    };

    let stream = cuda_ctx.default_stream();
    let size = 1usize << 20; // 1 MiB
    let mut vram: CudaSlice<u8> = stream.alloc_zeros(size).expect("alloc VRAM");
    let vram_ptr = {
        let (p, _g) = vram.device_ptr_mut(&stream);
        p as u64
    };

    // The critical call: register VRAM with the HCA. Succeeds via
    // nvidia-peermem on bare metal, or via the dma-buf fallback under
    // IOMMU-mediated virtualization (driver ≥ 515, rdma-core ≥ v34).
    // SAFETY: `vram` is a live CUDA allocation held for the whole test; the
    // MR is dropped explicitly before `vram` goes out of scope.
    let mr = match unsafe { reg_mr_cuda(&rdma_ctx, vram_ptr, size, IBV_ACCESS_LOCAL_WRITE) } {
        Ok(mr) => mr,
        Err(e) => {
            panic!(
                "CUDA MR registration failed on both peermem and dmabuf paths: {e}\n\
                 hint: `sudo modprobe nvidia-peermem`; for the dmabuf path the \
                 open-flavor NVIDIA kernel modules and rdma-core ≥ v34 are required"
            );
        }
    };

    assert_ne!(
        mr.lkey(),
        0,
        "lkey must be non-zero after successful reg_mr"
    );
    assert_ne!(
        mr.rkey(),
        0,
        "rkey must be non-zero after successful reg_mr"
    );

    // Explicit drop so dereg runs before the CUDA context falls.
    drop(mr);
}
