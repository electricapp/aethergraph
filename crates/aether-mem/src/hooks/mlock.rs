//! mlock hook — prevents the OS from swapping ring buffer memory to disk.

use crate::MemoryHook;

/// Prevents the OS from swapping ring buffer memory to disk.
///
/// Calls `mlock` on allocation and `munlock` on deallocation (Unix only).
/// Failure is non-fatal — the allocation proceeds even if mlock fails.
pub struct MlockHook;

impl MemoryHook for MlockHook {
    fn on_alloc(&self, ptr: *mut u8, size: usize) -> bool {
        #[cfg(unix)]
        {
            // SAFETY: ptr is valid for size bytes, mlock is safe on any valid region.
            let ret = unsafe { libc::mlock(ptr as *const libc::c_void, size) };
            if ret != 0 {
                #[cfg(feature = "tracing")]
                tracing::warn!(size, "mlock failed (non-fatal)");
            }
            ret == 0
        }
        #[cfg(not(unix))]
        {
            let _ = (ptr, size);
            false
        }
    }

    fn on_dealloc(&self, ptr: *mut u8, size: usize) {
        #[cfg(unix)]
        // SAFETY: ptr/size match a region previously passed to mlock; munlock tolerates non-locked regions.
        unsafe {
            let _ = libc::munlock(ptr as *const libc::c_void, size);
        }
        #[cfg(not(unix))]
        {
            let _ = (ptr, size);
        }
    }
}
