//! mlock hook — prevents the OS from swapping ring buffer memory to disk.

use crate::{HookError, MemoryHook};
use std::sync::atomic::{AtomicBool, Ordering};

/// Prevents the OS from swapping ring buffer memory to disk.
///
/// Calls `mlock` on allocation and `munlock` on deallocation (Unix only).
/// Failure is non-fatal — the allocation proceeds even if mlock fails. The
/// matching `munlock` runs only when `mlock` actually succeeded, so a failed
/// lock never issues an unbalanced unlock.
#[derive(Default)]
pub struct MlockHook {
    /// Set once `mlock` succeeds; gates the `munlock` in `on_dealloc`.
    locked: AtomicBool,
}

impl MlockHook {
    /// Create a hook that has not yet locked any region.
    pub const fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
        }
    }
}

impl MemoryHook for MlockHook {
    fn on_alloc(&self, ptr: *mut u8, size: usize) -> Result<(), HookError> {
        #[cfg(unix)]
        {
            // SAFETY: ptr is valid for size bytes, mlock is safe on any valid region.
            let ret = unsafe { libc::mlock(ptr as *const libc::c_void, size) };
            if ret != 0 {
                let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                #[cfg(feature = "tracing")]
                tracing::warn!(size, errno, "mlock failed (non-fatal)");
                return Err(HookError::with_code("mlock failed", errno));
            }
            self.locked.store(true, Ordering::Relaxed);
            Ok(())
        }
        #[cfg(not(unix))]
        {
            let _ = (ptr, size);
            Err(HookError::new("mlock not supported on this platform"))
        }
    }

    fn on_dealloc(&self, ptr: *mut u8, size: usize) {
        #[cfg(unix)]
        {
            // Only unlock what was locked; a failed mlock leaves nothing to undo.
            if self.locked.swap(false, Ordering::Relaxed) {
                // SAFETY: ptr/size match the region this hook successfully passed to mlock.
                unsafe {
                    let _ = libc::munlock(ptr as *const libc::c_void, size);
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = (ptr, size);
        }
    }
}
