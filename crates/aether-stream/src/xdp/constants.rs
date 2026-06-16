//! AF_XDP kernel constants not yet exported by libc 0.2.
//!
//! These come from `linux/if_xdp.h` and `linux/socket.h`.
//! Pinned to kernel 5.10+ ABI (stable since 2020).

/// Address family for XDP sockets.
pub const AF_XDP: i32 = 44;

/// Socket option level for XDP.
pub const SOL_XDP: i32 = 283;

// --- setsockopt / getsockopt option names ---
// Verified against /usr/include/linux/if_xdp.h on rdma-core / Ubuntu 24.04.
// The previous values here had every option swapped — XDP_UMEM_REG was set
// to 1 (= XDP_MMAP_OFFSETS), so setsockopt(REG) actually requested mmap
// offsets and kernel returned ENOPROTOOPT, which masked the bug because
// nothing exercised the data path.

/// Get mmap offsets for all rings.
pub const XDP_MMAP_OFFSETS: i32 = 1;

/// Set RX ring size.
pub const XDP_RX_RING: i32 = 2;

/// Set TX ring size.
pub const XDP_TX_RING: i32 = 3;

/// Register UMEM area with the socket.
pub const XDP_UMEM_REG: i32 = 4;

/// Set UMEM fill ring size.
pub const XDP_UMEM_FILL_RING: i32 = 5;

/// Set UMEM completion ring size.
pub const XDP_UMEM_COMPLETION_RING: i32 = 6;

// --- mmap page offsets (pgoff parameter) ---

/// mmap pgoff for the RX ring.
pub const XDP_PGOFF_RX_RING: u64 = 0;

/// mmap pgoff for the TX ring.
pub const XDP_PGOFF_TX_RING: u64 = 0x80000000;

/// mmap pgoff for the FILL ring.
pub const XDP_PGOFF_FILL_RING: u64 = 0x100000000;

/// mmap pgoff for the COMPLETION ring.
pub const XDP_PGOFF_COMPLETION_RING: u64 = 0x180000000;

// --- bind flags ---

/// Share UMEM between multiple sockets.
pub const XDP_SHARED_UMEM: u16 = 1 << 0;

/// Copy mode (fallback when zero-copy not available).
pub const XDP_COPY: u16 = 1 << 1;

/// Zero-copy mode.
pub const XDP_ZEROCOPY: u16 = 1 << 2;

// --- Ring flags (in each ring's `flags` field) ---

/// Set by the kernel in a ring's `flags` field when the driver needs a
/// syscall to resume processing that ring (zero-copy / busy-poll drivers stop
/// draining FILL/TX once caught up). When observed, the userspace side must
/// issue `recvfrom`/`sendto` to kick the driver, or the ring livelocks.
pub const XDP_RING_NEED_WAKEUP: u32 = 1 << 0;

// --- Descriptor flags ---

/// Frame size for standard UMEM frames.
pub const XDP_FRAME_SIZE_2K: u32 = 2048;
pub const XDP_FRAME_SIZE_4K: u32 = 4096;
