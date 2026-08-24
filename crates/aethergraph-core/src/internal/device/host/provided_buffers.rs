//! K4.3 — `IORING_REGISTER_PBUF_RING` provided-buffer descriptors.
//!
//! [`crate::internal::uring::UringHandle`] uses these types to register a
//! userspace-backed descriptor ring and landing buffers, allowing the kernel
//! to pick a landing buffer per completion without userspace free-list
//! arbitration.
//!
//! The on-wire `io_uring_buf` entry is specified by the kernel ABI; packing
//! it here keeps the integration surface unit-testable on any host.

/// One entry in an `io_uring` provided-buffer ring (`struct io_uring_buf`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoUringBuf {
    /// Absolute address of the landing buffer.
    pub addr: u64,
    /// Usable length in bytes.
    pub len: u32,
    /// Buffer id returned in the CQE (`flags >> IORING_CQE_BUFFER_SHIFT`).
    pub bid: u16,
    /// Reserved; must be zero on publish.
    pub resv: u16,
}

impl IoUringBuf {
    /// Build a ring entry for `bid` covering `[addr, addr+len)`.
    #[must_use]
    pub const fn new(addr: u64, len: u32, bid: u16) -> Self {
        Self {
            addr,
            len,
            bid,
            resv: 0,
        }
    }
}

/// Registration parameters for `IORING_REGISTER_PBUF_RING`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProvidedBufferRingSpec {
    /// Buffer group id (`bgid`) referenced by SQEs with `IOSQE_BUFFER_SELECT`.
    pub bgid: u16,
    /// Number of entries in the ring (power of two on modern kernels).
    pub entries: u16,
    /// Ring flags (`IOU_PBUF_RING_MMAP` when the kernel maps the ring).
    pub flags: u16,
}

impl ProvidedBufferRingSpec {
    /// Construct a non-mmap userspace-backed ring of `entries` buffers.
    #[must_use]
    pub const fn userspace(bgid: u16, entries: u16) -> Self {
        Self {
            bgid,
            entries,
            flags: 0,
        }
    }

    /// True when `entries` is a non-zero power of two (kernel requirement).
    #[must_use]
    pub const fn entries_ok(self) -> bool {
        self.entries != 0 && self.entries.is_power_of_two()
    }
}

// Wired via `UringHandle::register_provided_buffer_ring` + `read_buffer_select`.
// Pair with DEFER_TASKRUN RingPool for completion work off the app thread.
// TODO(HARDWARE): end-to-end BUFFER_SELECT under load on a rooted Linux VM
// (≥5.19); assert CQE bids land in the published ring without userspace
// free-list pops.

#[cfg(test)]
mod tests {
    use super::{IoUringBuf, ProvidedBufferRingSpec};
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn io_uring_buf_matches_kernel_abi_layout() {
        assert_eq!(size_of::<IoUringBuf>(), 16);
        assert_eq!(align_of::<IoUringBuf>(), 8);
        assert_eq!(offset_of!(IoUringBuf, addr), 0);
        assert_eq!(offset_of!(IoUringBuf, len), 8);
        assert_eq!(offset_of!(IoUringBuf, bid), 12);
        assert_eq!(offset_of!(IoUringBuf, resv), 14);
    }

    #[test]
    fn provided_ring_requires_power_of_two_entries() {
        assert!(ProvidedBufferRingSpec::userspace(1, 64).entries_ok());
        assert!(!ProvidedBufferRingSpec::userspace(1, 0).entries_ok());
        assert!(!ProvidedBufferRingSpec::userspace(1, 3).entries_ok());
    }
}
