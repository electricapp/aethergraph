//! CUDA host memory registration hook.
//!
//! Registers ring buffer memory with CUDA for zero-copy DMA transfers
//! via `cuMemHostRegister_v2`. This avoids pageable-to-pinned staging
//! copies, cutting memcpy latency roughly in half.

use crate::MemoryHook;

/// Registers memory with CUDA for zero-copy DMA transfers.
///
/// Uses `cuMemHostRegister_v2` with `CU_MEMHOSTREGISTER_DEVICEMAP`
/// to enable direct GPU access to the ring buffer memory.
pub struct CudaRegHook;

impl MemoryHook for CudaRegHook {
    fn on_alloc(&self, ptr: *mut u8, size: usize) -> bool {
        unsafe {
            use cudarc::driver::sys::{CUresult, cuMemHostRegister_v2};
            // 0x02 = CU_MEMHOSTREGISTER_DEVICEMAP (enable DMA)
            let result = cuMemHostRegister_v2(ptr as *mut _, size, 0x02);
            if result != CUresult::CUDA_SUCCESS {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    error_code = result as i32,
                    size,
                    "CUDA host registration failed — falling back to pageable transfers (~2x memcpy latency)"
                );
                false
            } else {
                true
            }
        }
    }

    fn on_dealloc(&self, ptr: *mut u8, _size: usize) {
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
