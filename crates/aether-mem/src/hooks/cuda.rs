//! CUDA host memory registration hook.
//!
//! Registers ring buffer memory with CUDA for zero-copy DMA transfers
//! via `cuMemHostRegister_v2`. This avoids pageable-to-pinned staging
//! copies, cutting memcpy latency roughly in half.

use crate::{HookError, MemoryHook};

/// `CU_MEMHOSTREGISTER_DEVICEMAP` — enables GPU DMA access to the host pages.
/// Defined here as a const because cudarc does not expose a named constant.
const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 0x02;

/// Registers memory with CUDA for zero-copy DMA transfers.
///
/// Uses `cuMemHostRegister_v2` with `CU_MEMHOSTREGISTER_DEVICEMAP`
/// to enable direct GPU access to the ring buffer memory.
pub struct CudaRegHook;

impl MemoryHook for CudaRegHook {
    fn on_alloc(&self, ptr: *mut u8, size: usize) -> Result<(), HookError> {
        // SAFETY: ptr is valid for `size` bytes (caller invariant); cuMemHostRegister_v2
        // tolerates any page-aligned region and reports failure via CUresult.
        unsafe {
            use cudarc::driver::sys::{CUresult, cuMemHostRegister_v2};
            let result = cuMemHostRegister_v2(ptr as *mut _, size, CU_MEMHOSTREGISTER_DEVICEMAP);
            if result != CUresult::CUDA_SUCCESS {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    error_code = result as i32,
                    size,
                    "CUDA host registration failed — falling back to pageable transfers (~2x memcpy latency)"
                );
                return Err(HookError::with_code(
                    "cuMemHostRegister_v2 failed",
                    result as i32,
                ));
            }
            Ok(())
        }
    }

    fn on_dealloc(&self, ptr: *mut u8, _size: usize) {
        // SAFETY: ptr was registered via cuMemHostRegister_v2 in on_alloc;
        // unregister of a not-registered region is detected and ignored below.
        unsafe {
            use cudarc::driver::sys::{CUresult, cuMemHostUnregister};
            let result = cuMemHostUnregister(ptr as *mut _);
            if result != CUresult::CUDA_SUCCESS
                && result != CUresult::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED
            {
                #[cfg(feature = "tracing")]
                tracing::debug!(
                    error_code = result as i32,
                    "CUDA host memory unregister returned non-success"
                );
            }
        }
    }
}
