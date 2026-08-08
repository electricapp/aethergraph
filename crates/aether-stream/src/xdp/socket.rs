//! AF_XDP socket lifecycle — create, configure, bind.

use super::constants::*;
use super::rings::{RxTxDesc, UmemDesc, XdpMmapOffsets, XdpRing};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

/// UMEM registration struct passed to `setsockopt(XDP_UMEM_REG)`.
#[repr(C)]
#[derive(Debug)]
pub struct XdpUmemReg {
    /// Base address of the UMEM area.
    pub addr: u64,
    /// Total size in bytes.
    pub len: u64,
    /// Size of each frame (must be 2048 or 4096).
    pub chunk_size: u32,
    /// Headroom before each frame's data (typically 0).
    pub headroom: u32,
    /// Flags (reserved, set to 0).
    pub flags: u32,
}

/// Address passed to `bind()` for AF_XDP.
#[repr(C)]
#[derive(Debug)]
pub struct SockaddrXdp {
    pub sxdp_family: u16,
    /// Bind flags (XDP_SHARED_UMEM, XDP_COPY, XDP_ZEROCOPY).
    pub sxdp_flags: u16,
    /// Interface index (from `if_nametoindex`).
    pub sxdp_ifindex: u32,
    /// NIC RX queue index.
    pub sxdp_queue_id: u32,
    /// Shared UMEM fd (when using XDP_SHARED_UMEM).
    pub sxdp_shared_umem_fd: u32,
}

/// An AF_XDP socket with its four kernel-mmap'd rings.
pub struct XdpSocket {
    fd: OwnedFd,
    pub fill: XdpRing<UmemDesc>,
    pub rx: XdpRing<RxTxDesc>,
    pub tx: XdpRing<RxTxDesc>,
    pub completion: XdpRing<UmemDesc>,
    /// Keep mmap'd regions alive
    _ring_mmaps: Vec<MmapRegion>,
}

/// Owned mmap region — unmapped on drop.
struct MmapRegion {
    ptr: *mut u8,
    len: usize,
}

impl Drop for MmapRegion {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr/len from our mmap call
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
        }
    }
}

// SAFETY: mmap regions are not aliased.
unsafe impl Send for MmapRegion {}
// SAFETY: mmap regions are not aliased.
unsafe impl Sync for MmapRegion {}

impl XdpSocket {
    /// Create and configure an AF_XDP socket.
    ///
    /// # Arguments
    /// * `umem_addr` - Base address of the UMEM region
    /// * `umem_size` - Total UMEM size in bytes
    /// * `frame_size` - Frame size (2048 or 4096)
    /// * `ring_size` - Size of each ring (must be power of two)
    /// * `ifindex` - Network interface index
    /// * `queue_id` - NIC RX queue to bind
    /// * `bind_flags` - XDP_COPY or XDP_ZEROCOPY
    ///
    /// # Safety
    /// `umem_addr` must point to a valid, mlock'd memory region of `umem_size` bytes.
    pub unsafe fn create(
        umem_addr: *mut u8,
        umem_size: usize,
        frame_size: u32,
        ring_size: u32,
        ifindex: u32,
        queue_id: u32,
        bind_flags: u16,
    ) -> std::io::Result<Self> {
        if !ring_size.is_power_of_two() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("ring_size {ring_size} must be a power of two"),
            ));
        }

        // 1. Create socket
        // SAFETY: standard socket(2) call; arguments are valid kernel constants.
        let fd = unsafe { libc::socket(AF_XDP, libc::SOCK_RAW, 0) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: socket() just succeeded and we have not consumed `fd`.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };

        // 2. Register UMEM
        let umem_reg = XdpUmemReg {
            addr: umem_addr as u64,
            len: umem_size as u64,
            chunk_size: frame_size,
            headroom: 0,
            flags: 0,
        };
        setsockopt_xdp(
            fd.as_raw_fd(),
            XDP_UMEM_REG,
            &umem_reg as *const _ as *const libc::c_void,
            std::mem::size_of::<XdpUmemReg>() as u32,
        )?;

        // 3. Set ring sizes
        let ring_size_val = ring_size;
        for opt in [
            XDP_UMEM_FILL_RING,
            XDP_UMEM_COMPLETION_RING,
            XDP_RX_RING,
            XDP_TX_RING,
        ] {
            setsockopt_xdp(
                fd.as_raw_fd(),
                opt,
                &ring_size_val as *const _ as *const libc::c_void,
                std::mem::size_of::<u32>() as u32,
            )?;
        }

        // 4. Get mmap offsets
        let offsets = getsockopt_mmap_offsets(fd.as_raw_fd())?;

        // 5. mmap each ring
        let mut ring_mmaps = Vec::with_capacity(4);

        // SAFETY: `fd` is the AF_XDP socket; `offsets` came from getsockopt above.
        let fill_mmap = unsafe {
            mmap_ring::<UmemDesc>(fd.as_raw_fd(), &offsets.fr, ring_size, XDP_PGOFF_FILL_RING)?
        };
        // SAFETY: fill_mmap.ptr is a valid mmap region for the fill ring.
        let fill = unsafe { XdpRing::<UmemDesc>::from_mmap(fill_mmap.ptr, &offsets.fr, ring_size) };
        ring_mmaps.push(fill_mmap);

        // SAFETY: `fd` is the AF_XDP socket; `offsets` came from getsockopt above.
        let comp_mmap = unsafe {
            mmap_ring::<UmemDesc>(
                fd.as_raw_fd(),
                &offsets.cr,
                ring_size,
                XDP_PGOFF_COMPLETION_RING,
            )?
        };
        // SAFETY: comp_mmap.ptr is a valid mmap region for the completion ring.
        let completion =
            unsafe { XdpRing::<UmemDesc>::from_mmap(comp_mmap.ptr, &offsets.cr, ring_size) };
        ring_mmaps.push(comp_mmap);

        // SAFETY: `fd` is the AF_XDP socket; `offsets` came from getsockopt above.
        let rx_mmap = unsafe {
            mmap_ring::<RxTxDesc>(fd.as_raw_fd(), &offsets.rx, ring_size, XDP_PGOFF_RX_RING)?
        };
        // SAFETY: rx_mmap.ptr is a valid mmap region for the rx ring.
        let rx = unsafe { XdpRing::<RxTxDesc>::from_mmap(rx_mmap.ptr, &offsets.rx, ring_size) };
        ring_mmaps.push(rx_mmap);

        // SAFETY: `fd` is the AF_XDP socket; `offsets` came from getsockopt above.
        let tx_mmap = unsafe {
            mmap_ring::<RxTxDesc>(fd.as_raw_fd(), &offsets.tx, ring_size, XDP_PGOFF_TX_RING)?
        };
        // SAFETY: tx_mmap.ptr is a valid mmap region for the tx ring.
        let tx = unsafe { XdpRing::<RxTxDesc>::from_mmap(tx_mmap.ptr, &offsets.tx, ring_size) };
        ring_mmaps.push(tx_mmap);

        // 6. Bind to NIC queue
        let addr = SockaddrXdp {
            sxdp_family: AF_XDP as u16,
            sxdp_flags: bind_flags,
            sxdp_ifindex: ifindex,
            sxdp_queue_id: queue_id,
            sxdp_shared_umem_fd: 0,
        };
        // SAFETY: `fd` is valid; `addr` is a kernel-defined sockaddr_xdp struct.
        let ret = unsafe {
            libc::bind(
                fd.as_raw_fd(),
                &addr as *const _ as *const libc::sockaddr,
                std::mem::size_of::<SockaddrXdp>() as u32,
            )
        };
        if ret < 0 {
            return Err(std::io::Error::last_os_error());
        }

        Ok(Self {
            fd,
            fill,
            rx,
            tx,
            completion,
            _ring_mmaps: ring_mmaps,
        })
    }

    /// Raw file descriptor (for `poll`/`epoll` if needed).
    pub fn fd(&self) -> i32 {
        self.fd.as_raw_fd()
    }
}

fn setsockopt_xdp(
    fd: i32,
    optname: i32,
    optval: *const libc::c_void,
    optlen: u32,
) -> std::io::Result<()> {
    // SAFETY: fd is valid, optval points to correct type
    let ret = unsafe { libc::setsockopt(fd, SOL_XDP, optname, optval, optlen) };
    if ret < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn getsockopt_mmap_offsets(fd: i32) -> std::io::Result<XdpMmapOffsets> {
    let mut offsets = XdpMmapOffsets::default();
    let mut len = std::mem::size_of::<XdpMmapOffsets>() as u32;
    // SAFETY: offsets is valid, len is correct
    let ret = unsafe {
        libc::getsockopt(
            fd,
            SOL_XDP,
            XDP_MMAP_OFFSETS,
            &mut offsets as *mut _ as *mut libc::c_void,
            &mut len,
        )
    };
    if ret < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(offsets)
    }
}

/// mmap a single ring region.
///
/// Ring size in bytes = offsets.desc + ring_size * sizeof(descriptor). The
/// kernel rejects (EINVAL) any request larger than the ring's page-aligned
/// allocation, so the size must be exact — no defensive padding.
unsafe fn mmap_ring<T>(
    fd: i32,
    offsets: &super::rings::XdpRingOffset,
    ring_size: u32,
    pgoff: u64,
) -> std::io::Result<MmapRegion> {
    let mmap_size = offsets.desc as usize + (ring_size as usize) * std::mem::size_of::<T>();

    // SAFETY: `fd` is the AF_XDP socket; `pgoff` selects a kernel-known ring
    // region whose size is at most `mmap_size`.
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            mmap_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_POPULATE,
            fd,
            pgoff as i64,
        )
    };

    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }

    Ok(MmapRegion {
        ptr: ptr as *mut u8,
        len: mmap_size,
    })
}
