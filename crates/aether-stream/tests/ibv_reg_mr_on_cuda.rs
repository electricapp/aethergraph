//! `ibv_reg_mr` on a CUDA device pointer.
//!
//! Registers 1 MiB of VRAM with the HCA and asserts the returned MR has
//! non-zero `lkey` / `rkey`. This call only succeeds when `nvidia-peermem`
//! (or an equivalent peer-memory module) is loaded and the HCA supports
//! GPUDirect — it's the cheapest signal that GPUDirect is actually wired
//! up on a host that has CUDA and ibverbs individually working.
//!
//! Skipped automatically if either RDMA or CUDA is missing.

#![cfg(all(target_os = "linux", feature = "rdma", feature = "gpudirect"))]

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

    // The critical call: register VRAM with the HCA. This only succeeds when
    // `nvidia-peermem` (or equivalent peer-memory module) is loaded and the
    // device/HCA pair supports GPUDirect.
    let mr = match rdma_ctx.reg_mr(vram_ptr as *mut u8, size, IBV_ACCESS_LOCAL_WRITE) {
        Ok(mr) => mr,
        Err(e) => {
            // The single most likely cause on a misconfigured box. Surface it
            // explicitly so the failure tells the operator what to fix.
            panic!(
                "ibv_reg_mr on CUDA pointer failed: {e}\n\
                 hint: `sudo modprobe nvidia-peermem` and check `dmesg | grep peermem`"
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
