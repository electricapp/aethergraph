//! KERNELS.md Tier A integration tests (require `--features gpudirect`).
//!
//! Skip cleanly when no CUDA device is present.

#![cfg(all(target_os = "linux", feature = "gpudirect"))]

mod decompress;
mod persistent;
mod sampler;
mod tma;
mod validate;

use aether_stream::gpu::kernels::harness::cuda_or_skip;

#[test]
fn cuda_device_probe() {
    match cuda_or_skip() {
        Some(_) => eprintln!("CUDA device 0 available"),
        None => eprintln!("skipping: no CUDA device"),
    }
}
