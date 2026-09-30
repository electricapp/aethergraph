//! Cross-process shared feature cache over `memfd` + file seals.
//!
//! One process builds a feature cache in an anonymous, sealed
//! `memfd_create` region; N trainer processes map the *same physical
//! pages* read-only by receiving the memfd over a `SCM_RIGHTS` control
//! message. No copy, no serialization, no on-disk staging — the seqlock
//! slot format is already cross-process-correct (atomics on a shared
//! mapping), so a reader in another process sees writes the moment they
//! land.
//!
//! Seals make the sharing safe: `F_SEAL_SHRINK | F_SEAL_GROW` fix the
//! size for the memfd's whole lifetime, so no holder can `ftruncate` it
//! out from under a peer's mapping (which would fault them on access).
//! A region whose contents are final is additionally sealed against
//! writes ([`SharedRegion::seal_read_only`]); every peer receives an
//! `O_RDWR` descriptor, and only `F_SEAL_WRITE` stops one from mapping it
//! writable or `pwrite`-ing into every other process's view.

#![cfg(all(target_os = "linux", feature = "shm"))]

use anyhow::{Result, bail};
use std::os::unix::io::{AsRawFd, FromRawFd, OwnedFd, RawFd};

/// A shared memory region backed by a sealed `memfd`.
///
/// Created with [`SharedRegion::create`]; the owner writes through
/// [`SharedRegion::as_mut_slice`], then hands [`SharedRegion::raw_fd`] to
/// peers via [`send_fd`]. A peer maps it with [`SharedRegion::from_fd`]
/// after [`recv_fd`]. All mappings alias the same pages.
pub struct SharedRegion {
    fd: OwnedFd,
    base: *mut u8,
    len: usize,
    /// Whether this mapping may write (the creator) or is read-only
    /// (an attached peer).
    writable: bool,
}

// SAFETY: the region owns its fd and mapping; concurrent access across
// threads is the caller's responsibility exactly as for any shared buffer
// (the seqlock slot format coordinates it).
unsafe impl Send for SharedRegion {}
// SAFETY: see the Send impl above.
unsafe impl Sync for SharedRegion {}

impl SharedRegion {
    /// Create a sealed shared region of `len` bytes, mapped read-write.
    ///
    /// The memfd is created with sealing allowed, sized once, then sealed
    /// against shrink and grow so its size is fixed for every peer that
    /// later maps it.
    pub fn create(len: usize) -> Result<Self> {
        if len == 0 {
            bail!("SharedRegion length must be > 0");
        }
        let name = c"aether-shm";
        // SAFETY: `name` is a valid NUL-terminated C string; the flags are
        // constants.
        let raw = unsafe {
            libc::memfd_create(
                name.as_ptr(),
                (libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) as libc::c_uint,
            )
        };
        if raw < 0 {
            bail!("memfd_create failed: {}", std::io::Error::last_os_error());
        }
        // SAFETY: `raw` is a fresh fd we exclusively own.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };

        // Size it once.
        // SAFETY: `fd` is a valid memfd; `len` fits an off_t here.
        if unsafe { libc::ftruncate(raw, len as libc::off_t) } != 0 {
            bail!("ftruncate failed: {}", std::io::Error::last_os_error());
        }

        // Seal the size. Writes stay allowed (no F_SEAL_WRITE), so the
        // owner can still populate the cache; only resizing is forbidden.
        let seals = libc::F_SEAL_SHRINK | libc::F_SEAL_GROW;
        // SAFETY: `fd` allows sealing (MFD_ALLOW_SEALING) and has no
        // outstanding writable mappings yet.
        if unsafe { libc::fcntl(raw, libc::F_ADD_SEALS, seals) } != 0 {
            bail!("F_ADD_SEALS failed: {}", std::io::Error::last_os_error());
        }

        let base = mmap_fd(raw, len, Mapping::Writable)?;
        Ok(Self {
            fd,
            base,
            len,
            writable: true,
        })
    }

    /// Map an existing shared region from a received memfd, read-only.
    /// Takes ownership of `fd`.
    ///
    /// `len` is what the owner advertised alongside the fd; it is checked
    /// against the object rather than trusted. The descriptor arrives over
    /// a socket, so this is the edge where a shared region becomes a typed
    /// value, and everything the mapping depends on is established here:
    ///
    /// - the object really is `len` bytes, so the mapping cannot run past
    ///   its end;
    /// - `F_SEAL_SHRINK` and `F_SEAL_GROW` are already set, so the owner
    ///   cannot resize it afterwards and turn live reads into `SIGBUS`.
    ///
    /// Both are properties of the descriptor itself, which is why this
    /// needs no `unsafe` obligation from the caller.
    pub fn from_fd(fd: OwnedFd, len: usize) -> Result<Self> {
        Self::attach(fd, len, libc::F_SEAL_SHRINK | libc::F_SEAL_GROW)
    }

    /// Like [`from_fd`](Self::from_fd), but also require `F_SEAL_WRITE`:
    /// the contents are final, so no holder of the descriptor — this
    /// process included — can change the bytes the returned slice views.
    pub fn from_sealed_fd(fd: OwnedFd, len: usize) -> Result<Self> {
        Self::attach(
            fd,
            len,
            libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE,
        )
    }

    fn attach(fd: OwnedFd, len: usize, required: libc::c_int) -> Result<Self> {
        let raw = fd.as_raw_fd();

        // SAFETY: an all-zero `libc::stat` is a valid initial value; a
        // successful fstat overwrites every field it defines.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: `raw` is a live descriptor owned by `fd`, and `st` is a
        // valid out-pointer for the duration of the call.
        let rc = unsafe { libc::fstat(raw, &mut st) };
        if rc != 0 {
            bail!(
                "fstat on received memfd failed: {}",
                std::io::Error::last_os_error()
            );
        }
        let actual_len = usize::try_from(st.st_size).unwrap_or(0);
        if actual_len != len {
            bail!("received memfd is {actual_len} bytes, owner advertised {len}");
        }

        // SAFETY: `raw` is a live descriptor; F_GET_SEALS takes no argument
        // and returns the seal bits or -1.
        let seals = unsafe { libc::fcntl(raw, libc::F_GET_SEALS) };
        if seals < 0 {
            bail!(
                "received fd does not support seals, so it is not a memfd: {}",
                std::io::Error::last_os_error()
            );
        }
        let size_seals = libc::F_SEAL_SHRINK | libc::F_SEAL_GROW;
        if seals & size_seals != size_seals {
            bail!(
                "received memfd is not size-sealed (seals {seals:#x}); the owner \
                 could resize it under this mapping"
            );
        }
        if seals & required != required {
            bail!(
                "received memfd is not write-sealed (seals {seals:#x}); any holder \
                 of the descriptor could change the bytes under this mapping"
            );
        }

        let mapping = if seals & libc::F_SEAL_WRITE != 0 {
            Mapping::Frozen
        } else {
            Mapping::Live
        };
        let base = mmap_fd(raw, len, mapping)?;
        Ok(Self {
            fd,
            base,
            len,
            writable: false,
        })
    }

    /// Make the contents final: drop the creator's writable mapping, add
    /// `F_SEAL_WRITE`, and map the sealed contents read-only.
    ///
    /// The kernel refuses the seal while any shared mapping that could be
    /// made writable exists — a read-only `MAP_SHARED` view of an `O_RDWR`
    /// memfd included — so this runs with no mapping at all, before peers
    /// attach. That ordering is also what lets peers demand the seal
    /// through [`from_sealed_fd`](Self::from_sealed_fd). The region is
    /// consumed on failure.
    pub fn seal_read_only(mut self) -> Result<Self> {
        if !self.writable {
            bail!("only the creating process can seal a region against writes");
        }
        let raw = self.fd.as_raw_fd();
        // SAFETY: `base`/`len` is this region's own writable mapping, and
        // `self` is consumed, so no slice into it can still be live.
        unsafe { libc::munmap(self.base.cast::<libc::c_void>(), self.len) };
        // Unmapped: `Drop` skips a null base if an early return follows.
        self.base = std::ptr::null_mut();
        self.writable = false;
        // SAFETY: `raw` is this region's live memfd, created with
        // MFD_ALLOW_SEALING; nothing maps it any more.
        if unsafe { libc::fcntl(raw, libc::F_ADD_SEALS, libc::F_SEAL_WRITE) } != 0 {
            bail!(
                "F_ADD_SEALS(F_SEAL_WRITE) failed: {}",
                std::io::Error::last_os_error()
            );
        }
        self.base = mmap_fd(raw, self.len, Mapping::Frozen)?;
        Ok(self)
    }

    /// The memfd for passing to peers over [`send_fd`].
    pub fn raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Region length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the region is empty (never true for a live region).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read-only view of the shared bytes.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `base..base+len` is our mapping of the memfd, valid for
        // reads for the region's lifetime.
        unsafe { std::slice::from_raw_parts(self.base, self.len) }
    }

    /// Mutable view of the shared bytes. `None` for an attached read-only
    /// peer — only the creator may write.
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        if !self.writable {
            return None;
        }
        // SAFETY: the creator mapped PROT_WRITE; `&mut self` gives
        // exclusive access to this mapping (cross-process coordination is
        // the seqlock's job, not this borrow's).
        Some(unsafe { std::slice::from_raw_parts_mut(self.base, self.len) })
    }
}

impl Drop for SharedRegion {
    fn drop(&mut self) {
        if self.base.is_null() {
            return;
        }
        // SAFETY: `base`/`len` is our mapping; unmapping it doesn't affect
        // peers' independent mappings of the same memfd. The fd closes via
        // `OwnedFd`.
        unsafe {
            libc::munmap(self.base as *mut libc::c_void, self.len);
        }
    }
}

/// How a region maps its memfd.
#[derive(Clone, Copy)]
enum Mapping {
    /// The creator's writable view.
    Writable,
    /// A read-only view that sees the creator's later writes.
    Live,
    /// A read-only view of write-sealed contents. `MAP_PRIVATE` maps the
    /// same page-cache pages — nothing writes them, so none is ever copied —
    /// and, unlike `MAP_SHARED` of an `O_RDWR` descriptor, a write-sealed
    /// memfd accepts it on every kernel.
    Frozen,
}

fn mmap_fd(fd: RawFd, len: usize, mapping: Mapping) -> Result<*mut u8> {
    let (prot, flags) = match mapping {
        Mapping::Writable => (libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED),
        Mapping::Live => (libc::PROT_READ, libc::MAP_SHARED),
        Mapping::Frozen => (libc::PROT_READ, libc::MAP_PRIVATE),
    };
    // SAFETY: `fd` is a valid memfd of at least `len` bytes; every mode
    // maps its page cache, so all mappings see the same pages.
    let base = unsafe { libc::mmap(std::ptr::null_mut(), len, prot, flags, fd, 0) };
    if base == libc::MAP_FAILED {
        bail!("mmap failed: {}", std::io::Error::last_os_error());
    }
    Ok(base as *mut u8)
}

/// Send a file descriptor over a Unix socket via a `SCM_RIGHTS` control
/// message. The peer receives a fd referring to the same open file.
///
// The CMSG population is a single indivisible ritual — the header fields,
// the length, and the fd copy must all target the same control buffer in
// sequence — so it stays one unsafe block rather than fragmenting the
// pointer arithmetic across several.
#[allow(clippy::multiple_unsafe_ops_per_block)]
pub fn send_fd(sock: RawFd, fd: RawFd) -> Result<()> {
    // One data byte must accompany the control message — a zero-length
    // send carries no ancillary data on Linux.
    let mut iov_base = [0u8; 1];
    let mut iov = libc::iovec {
        iov_base: iov_base.as_mut_ptr() as *mut libc::c_void,
        iov_len: 1,
    };
    let mut cmsg_buf = CmsgBuf::new();
    // SAFETY: an all-zero msghdr is a valid empty message; fields are set below.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr();
    msg.msg_controllen = CMSG_SPACE_ONE_FD;

    // SAFETY: `msg.msg_control` points at `cmsg_buf`, aligned for and large
    // enough to hold one fd's CMSG header + payload.
    unsafe {
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<RawFd>() as u32) as usize;
        std::ptr::copy_nonoverlapping(
            &fd as *const RawFd as *const u8,
            libc::CMSG_DATA(cmsg),
            std::mem::size_of::<RawFd>(),
        );
    }

    // SAFETY: `msg` is fully initialized above.
    let n = unsafe { libc::sendmsg(sock, &msg, 0) };
    if n < 0 {
        bail!("sendmsg failed: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

/// Receive a file descriptor sent with [`send_fd`]. Returns an owning
/// handle to the received fd.
///
// Reading the control message back is the same indivisible CMSG ritual as
// `send_fd` — one unsafe block validates the header and extracts the fd.
#[allow(clippy::multiple_unsafe_ops_per_block)]
pub fn recv_fd(sock: RawFd) -> Result<OwnedFd> {
    let mut iov_base = [0u8; 1];
    let mut iov = libc::iovec {
        iov_base: iov_base.as_mut_ptr() as *mut libc::c_void,
        iov_len: 1,
    };
    let mut cmsg_buf = CmsgBuf::new();
    // SAFETY: an all-zero msghdr is a valid empty message; fields are set below.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr();
    msg.msg_controllen = CMSG_SPACE_ONE_FD;

    // MSG_CMSG_CLOEXEC: a received descriptor is never inherited by a
    // child spawned before this function takes ownership of it.
    // SAFETY: `msg` is initialized; the kernel fills the control buffer.
    let n = unsafe { libc::recvmsg(sock, &mut msg, libc::MSG_CMSG_CLOEXEC) };
    if n < 0 {
        bail!("recvmsg failed: {}", std::io::Error::last_os_error());
    }

    // Own every descriptor the kernel installed before judging the
    // message, so a rejected one closes them all instead of leaking them.
    let mut fds = take_scm_rights(&msg);

    // A peer that sent more descriptors than this buffer holds gets the
    // excess dropped by the kernel, which flags the message MSG_CTRUNC.
    if msg.msg_flags & libc::MSG_CTRUNC != 0 {
        bail!("control message truncated; peer sent more than one descriptor");
    }
    match fds.len() {
        1 => Ok(fds.remove(0)),
        0 => bail!("recvmsg returned no descriptor"),
        k => bail!("peer sent {k} descriptors, expected exactly one"),
    }
}

/// Take ownership of every descriptor in `msg`'s `SCM_RIGHTS` messages.
fn take_scm_rights(msg: &libc::msghdr) -> Vec<OwnedFd> {
    let mut fds = Vec::new();
    // SAFETY: `msg` was populated by recvmsg; the CMSG walk stays inside
    // its control buffer.
    let mut cmsg = unsafe { libc::CMSG_FIRSTHDR(msg) };
    while !cmsg.is_null() {
        // SAFETY: a non-null header from the walk lies wholly inside the
        // (cmsghdr-aligned) control buffer.
        let hdr = unsafe { &*cmsg };
        if hdr.cmsg_level == libc::SOL_SOCKET && hdr.cmsg_type == libc::SCM_RIGHTS {
            // SAFETY: `CMSG_LEN(0)` is pure arithmetic on the header size.
            let header = unsafe { libc::CMSG_LEN(0) } as usize;
            let count = (hdr.cmsg_len as usize).saturating_sub(header) / size_of::<RawFd>();
            // SAFETY: CMSG_DATA points at this message's payload.
            let data = unsafe { libc::CMSG_DATA(cmsg) }.cast::<RawFd>();
            for i in 0..count {
                // SAFETY: `i < count`, so the pointer stays inside the
                // payload the kernel reported.
                let slot = unsafe { data.add(i) };
                // SAFETY: `slot` is in bounds; the payload need not be
                // int-aligned, hence the unaligned read.
                let fd = unsafe { slot.read_unaligned() };
                if fd >= 0 {
                    // SAFETY: the kernel just installed `fd` in this
                    // process, and nothing else owns it.
                    fds.push(unsafe { OwnedFd::from_raw_fd(fd) });
                }
            }
        }
        // SAFETY: `cmsg` is a valid header within `msg`'s control buffer.
        cmsg = unsafe { libc::CMSG_NXTHDR(msg, cmsg) };
    }
    fds
}

/// Bytes needed to carry exactly one fd as ancillary data.
// SAFETY: CMSG_SPACE is pure arithmetic on its argument.
const CMSG_SPACE_ONE_FD: usize = unsafe { libc::CMSG_SPACE(size_of::<RawFd>() as u32) } as usize;

/// Control-message buffer for one fd. The `cmsghdr` member makes it
/// aligned for the header the CMSG macros read and write through.
#[repr(C)]
union CmsgBuf {
    _align: libc::cmsghdr,
    bytes: [u8; CMSG_SPACE_ONE_FD],
}

impl CmsgBuf {
    fn new() -> Self {
        Self {
            bytes: [0u8; CMSG_SPACE_ONE_FD],
        }
    }

    fn as_mut_ptr(&mut self) -> *mut libc::c_void {
        std::ptr::from_mut(self).cast()
    }
}

/// A cross-process fd for a Unix datagram/stream socket pair, split so the
/// two ends can hand off a memfd. Convenience for callers that don't
/// already have a socket channel.
pub fn socket_pair() -> Result<(OwnedFd, OwnedFd)> {
    let mut fds = [0 as RawFd; 2];
    // SAFETY: `fds` is a valid 2-element array; the domain/type/proto are
    // constants.
    let ret = unsafe {
        libc::socketpair(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_CLOEXEC,
            0,
            fds.as_mut_ptr(),
        )
    };
    if ret != 0 {
        bail!("socketpair failed: {}", std::io::Error::last_os_error());
    }
    // SAFETY: `fds[0]` is a fresh socket end we exclusively own.
    let a = unsafe { OwnedFd::from_raw_fd(fds[0]) };
    // SAFETY: `fds[1]` is the other fresh socket end we exclusively own.
    let b = unsafe { OwnedFd::from_raw_fd(fds[1]) };
    Ok((a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_sealed_against_resize() {
        let region = SharedRegion::create(8192).unwrap();
        // F_SEAL_SHRINK|GROW must reject any ftruncate to a new size.
        // SAFETY: valid memfd; the call is expected to fail with EPERM.
        // SAFETY: valid memfd; the call is expected to fail (sealed).
        let shrink = unsafe { libc::ftruncate(region.raw_fd(), 4096) };
        assert_ne!(shrink, 0, "shrink must be sealed off");
        // SAFETY: valid memfd; the call is expected to fail (sealed).
        let grow = unsafe { libc::ftruncate(region.raw_fd(), 16384) };
        assert_ne!(grow, 0, "grow must be sealed off");
        assert_eq!(region.len(), 8192);
    }

    #[test]
    fn two_mappings_of_one_memfd_alias() {
        let mut owner = SharedRegion::create(4096).unwrap();
        // Write a pattern through the owner mapping.
        for (i, b) in owner.as_mut_slice().unwrap().iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }

        // Duplicate the fd and map it again: same physical pages.
        // SAFETY: dup of a valid memfd returns a fresh fd or -1.
        let raw_dup = unsafe { libc::dup(owner.raw_fd()) };
        assert!(raw_dup >= 0, "dup failed");
        // SAFETY: `raw_dup` is a fresh fd we exclusively own.
        let dup = unsafe { OwnedFd::from_raw_fd(raw_dup) };
        let peer = SharedRegion::from_fd(dup, 4096).unwrap();

        assert_eq!(peer.as_slice(), owner.as_slice(), "peer sees owner writes");
        // A later owner write is visible to the peer with no copy.
        owner.as_mut_slice().unwrap()[0] = 0xAB;
        assert_eq!(peer.as_slice()[0], 0xAB, "live aliasing");
    }

    /// The advertised length is peer-supplied, so it is checked against
    /// the object rather than believed. A too-large value would otherwise
    /// map past the end of the memfd and fault on first touch.
    #[test]
    fn from_fd_rejects_a_length_the_object_does_not_have() {
        let owner = SharedRegion::create(4096).unwrap();
        // SAFETY: dup of a valid memfd returns a fresh fd or -1.
        let raw_dup = unsafe { libc::dup(owner.raw_fd()) };
        assert!(raw_dup >= 0, "dup failed");
        // SAFETY: `raw_dup` is a fresh fd we exclusively own.
        let dup = unsafe { OwnedFd::from_raw_fd(raw_dup) };

        let Err(err) = SharedRegion::from_fd(dup, 8192) else {
            panic!("mapping 8192 bytes of a 4096-byte object must be refused");
        };
        assert!(
            err.to_string().contains("advertised"),
            "unexpected error: {err}"
        );
    }

    /// Without the size seals the owner could shrink the object after the
    /// peer maps it, turning every subsequent read into SIGBUS. An unsealed
    /// descriptor must be refused at attach time.
    #[test]
    fn from_fd_rejects_an_unsealed_memfd() {
        // A plain memfd with no seals added — what `create` deliberately
        // does not produce.
        let name = c"aethergraph-unsealed";
        // SAFETY: `name` is a valid NUL-terminated C string.
        let raw = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_ALLOW_SEALING) };
        assert!(raw >= 0, "memfd_create failed");
        // SAFETY: `raw` is a fresh fd we exclusively own.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        // SAFETY: `raw` is a live memfd; sizing it is unrelated to sealing.
        assert_eq!(unsafe { libc::ftruncate(raw, 4096) }, 0, "ftruncate failed");

        let Err(err) = SharedRegion::from_fd(fd, 4096) else {
            panic!("an unsealed memfd must be refused");
        };
        assert!(
            err.to_string().contains("size-sealed"),
            "unexpected error: {err}"
        );
    }

    /// A sealed region's contents are final for every holder of the fd:
    /// no write through it, no writable mapping of it, and peers can insist
    /// on the seal.
    #[test]
    fn sealed_region_refuses_every_write_path() {
        let mut owner = SharedRegion::create(4096).unwrap();
        owner.as_mut_slice().unwrap().fill(0x42);
        let owner = owner.seal_read_only().unwrap();
        assert!(owner.as_slice().iter().all(|&b| b == 0x42));

        // A peer's descriptor is O_RDWR; the seal is what stops it.
        let byte = [0u8; 1];
        // SAFETY: valid memfd and a 1-byte buffer; expected to fail.
        let wrote = unsafe { libc::pwrite(owner.raw_fd(), byte.as_ptr().cast(), 1, 0) };
        assert!(wrote < 0, "pwrite into a write-sealed memfd must fail");
        // SAFETY: mapping request expected to fail; nothing is dereferenced.
        let map = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                4096,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                owner.raw_fd(),
                0,
            )
        };
        assert_eq!(map, libc::MAP_FAILED, "writable mapping must be refused");

        // SAFETY: dup of a valid memfd returns a fresh fd or -1.
        let raw_dup = unsafe { libc::dup(owner.raw_fd()) };
        assert!(raw_dup >= 0);
        // SAFETY: `raw_dup` is a fresh fd we exclusively own.
        let peer =
            SharedRegion::from_sealed_fd(unsafe { OwnedFd::from_raw_fd(raw_dup) }, 4096).unwrap();
        assert_eq!(peer.as_slice(), owner.as_slice());
        assert!(
            owner.seal_read_only().is_err(),
            "only a writable region seals"
        );
    }

    #[test]
    fn from_sealed_fd_rejects_a_writable_memfd() {
        let owner = SharedRegion::create(4096).unwrap();
        // SAFETY: dup of a valid memfd returns a fresh fd or -1.
        let raw_dup = unsafe { libc::dup(owner.raw_fd()) };
        assert!(raw_dup >= 0);
        // SAFETY: `raw_dup` is a fresh fd we exclusively own.
        let dup = unsafe { OwnedFd::from_raw_fd(raw_dup) };
        let Err(err) = SharedRegion::from_sealed_fd(dup, 4096) else {
            panic!("a memfd without F_SEAL_WRITE must be refused");
        };
        assert!(err.to_string().contains("write-sealed"), "{err}");
    }

    /// A peer that sends more than one descriptor is refused; the
    /// descriptors the kernel did install are owned, and so closed, before
    /// the message is judged.
    // Building the two-fd message is the same indivisible CMSG ritual as
    // `send_fd`, kept in one unsafe block.
    #[allow(clippy::multiple_unsafe_ops_per_block)]
    #[test]
    fn recv_fd_refuses_extra_descriptors_without_leaking_them() {
        let (a, b) = socket_pair().unwrap();
        let region = SharedRegion::create(4096).unwrap();
        let fds = [region.raw_fd(), region.raw_fd()];

        let mut iov_base = [0u8; 1];
        let mut iov = libc::iovec {
            iov_base: iov_base.as_mut_ptr().cast(),
            iov_len: 1,
        };
        // SAFETY: CMSG_SPACE is pure arithmetic.
        let space = unsafe { libc::CMSG_SPACE(size_of_val(&fds) as u32) } as usize;
        // SAFETY: CMSG_LEN is pure arithmetic.
        let len = unsafe { libc::CMSG_LEN(size_of_val(&fds) as u32) } as usize;
        let mut buf = vec![0u64; space.div_ceil(8)];
        // SAFETY: an all-zero msghdr is a valid empty message.
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = buf.as_mut_ptr().cast();
        msg.msg_controllen = space;
        // SAFETY: the control buffer is 8-aligned and `space` bytes long.
        unsafe {
            let cmsg = libc::CMSG_FIRSTHDR(&msg);
            (*cmsg).cmsg_level = libc::SOL_SOCKET;
            (*cmsg).cmsg_type = libc::SCM_RIGHTS;
            (*cmsg).cmsg_len = len;
            std::ptr::copy_nonoverlapping(
                fds.as_ptr().cast::<u8>(),
                libc::CMSG_DATA(cmsg),
                size_of_val(&fds),
            );
            assert!(libc::sendmsg(a.as_raw_fd(), &msg, 0) > 0);
        }

        // CMSG_SPACE of one int is padded to 8 bytes of payload, so the
        // kernel delivers both descriptors rather than truncating.
        let err = recv_fd(b.as_raw_fd()).unwrap_err().to_string();
        assert!(err.contains("2 descriptors"), "{err}");
    }

    #[test]
    fn scm_rights_fd_passing_shares_the_cache() {
        // Owner builds a cache, sends its memfd across a socket; the
        // receiver maps it and reads the same bytes.
        let (a, b) = socket_pair().unwrap();

        let mut owner = SharedRegion::create(8192).unwrap();
        let payload: Vec<u8> = (0..8192).map(|i| (i * 7 % 253) as u8).collect();
        owner.as_mut_slice().unwrap().copy_from_slice(&payload);

        send_fd(a.as_raw_fd(), owner.raw_fd()).unwrap();
        let received = recv_fd(b.as_raw_fd()).unwrap();
        let peer = SharedRegion::from_fd(received, 8192).unwrap();

        assert_eq!(peer.as_slice(), &payload[..], "received mapping matches");

        // Peer is read-only.
        // SAFETY: from_fd maps PROT_READ; we only assert the API contract.
        // (No write attempted — that would SIGSEGV, which is the point.)
        // Owner mutation still propagates to the peer.
        owner.as_mut_slice().unwrap()[100] = 0x5A;
        assert_eq!(peer.as_slice()[100], 0x5A);
    }
}
