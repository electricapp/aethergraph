//! Cache-line isolation for fields whose writers must not evict
//! read-mostly neighbors from other cores' caches.

/// Pads a field group onto its own cache line. 128 bytes covers the
/// adjacent-line prefetcher on x86_64 and Apple Silicon's 128-byte granules.
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), repr(align(128)))]
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    repr(align(64))
)]
pub(crate) struct CachePadded<T>(pub(crate) T);
