//! Skip-friendly CUDA helpers for kernel tests and benches.

use cudarc::driver::{CudaContext, CudaStream};
use std::sync::Arc;
use std::time::Instant;

/// Open device 0, or `None` when CUDA is unavailable.
#[must_use]
pub fn cuda_or_skip() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = CudaContext::new(0).ok()?;
    let stream = ctx.default_stream();
    Some((ctx, stream))
}

/// True when the device's compute capability major*10+minor/10 style
/// `sm` (e.g. 90 for Hopper, 100 for Blackwell) is at least `min_sm`.
#[must_use]
pub fn require_sm(ctx: &CudaContext, min_sm: u32) -> bool {
    let Ok(major) = ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    ) else {
        return false;
    };
    let Ok(minor) = ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    ) else {
        return false;
    };
    let sm = (major as u32) * 10 + (minor as u32);
    sm >= min_sm
}

/// Elapsed microseconds for a closure (host wall clock).
#[must_use]
pub fn timed_us<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let start = Instant::now();
    let out = f();
    (out, start.elapsed().as_micros() as u64)
}
