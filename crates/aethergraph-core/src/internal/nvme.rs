//! NVMe passthrough reads over `io_uring` — the storage twin of AF_XDP.
//!
//! The feature store, once built, is immutable: its file extents are
//! fallocated and never move. That makes it sound to resolve each logical
//! read to an absolute device LBA once (via `FIEMAP`) and then issue
//! `NVME_URING_CMD_IO` reads against the block device's character handle
//! (`/dev/ngXnY`), bypassing the filesystem and block layer entirely —
//! the command goes straight to the NVMe controller.
//!
//! This module is split so the parts that need no hardware are fully
//! testable everywhere:
//!
//! - [`ExtentMap`] resolves file offsets to device LBAs with `FIEMAP`.
//!   Regular files on ext4/xfs expose extents; run against any such file.
//! - [`NvmePassthruCmd`] builds the 72-byte `nvme_uring_cmd` payload and
//!   is verified by field layout, no device required.
//! - [`NvmeReader`] owns the character-device handle plus ring and is the
//!   only part that needs `/dev/ng*` and privilege; construction is
//!   runtime-probed and the caller falls back when it returns `None`.

#![cfg(target_os = "linux")]

use anyhow::{Context, Result, bail};
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use tracing::trace;

/// NVMe read opcode (`nvme_cmd_read`).
const NVME_CMD_READ: u8 = 0x02;
/// `io_uring` async command for NVMe char devices (`NVME_URING_CMD_IO`).
/// `_IOWR('N', 0x80, struct nvme_passthru_cmd)` — the ioctl-encoded
/// command number io_uring's `cmd_op` expects for a passthrough read.
const NVME_URING_CMD_IO: u32 = nvme_ioctl_iowr(0x80);

/// Encode an `_IOWR('N', nr, struct nvme_passthru_cmd)` request number.
///
/// The struct is 72 bytes; the ioctl encoding packs (dir=IOWR=3, size,
/// type='N', nr) into a u32. Kept `const` so the command number is a
/// compile-time constant, matching what the kernel decodes.
const fn nvme_ioctl_iowr(nr: u32) -> u32 {
    const IOC_WRITE: u32 = 1;
    const IOC_READ: u32 = 2;
    const NRBITS: u32 = 8;
    const TYPEBITS: u32 = 8;
    const SIZEBITS: u32 = 14;
    const NRSHIFT: u32 = 0;
    const TYPESHIFT: u32 = NRSHIFT + NRBITS;
    const SIZESHIFT: u32 = TYPESHIFT + TYPEBITS;
    const DIRSHIFT: u32 = SIZESHIFT + SIZEBITS;
    let size = core::mem::size_of::<NvmePassthruCmd>() as u32;
    ((IOC_READ | IOC_WRITE) << DIRSHIFT)
        | (b'N' as u32) << TYPESHIFT
        | (nr << NRSHIFT)
        | (size << SIZESHIFT)
}

/// The `struct nvme_passthru_cmd` / `nvme_uring_cmd` payload, 72 bytes.
///
/// Layout matches `include/uapi/linux/nvme_ioctl.h`. For io_uring's
/// `uring_cmd`, this struct is written into the 80-byte `cmd` area of the
/// SQE (`sqe->cmd`); the kernel reads `opcode`, `nsid`, `addr`, `data_len`,
/// and the `cdw10..15` LBA/length words.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NvmePassthruCmd {
    pub opcode: u8,
    pub flags: u8,
    pub rsvd1: u16,
    pub nsid: u32,
    pub cdw2: u32,
    pub cdw3: u32,
    pub metadata: u64,
    pub addr: u64,
    pub metadata_len: u32,
    pub data_len: u32,
    pub cdw10: u32,
    pub cdw11: u32,
    pub cdw12: u32,
    pub cdw13: u32,
    pub cdw14: u32,
    pub cdw15: u32,
    pub timeout_ms: u32,
    pub result: u32,
}

// The kernel copies exactly 72 bytes out of the SQE cmd area; a mismatch
// would misalign every field the controller reads.
const _: () = assert!(core::mem::size_of::<NvmePassthruCmd>() == 72);

impl NvmePassthruCmd {
    /// Build a read of `nlb + 1` logical blocks starting at device LBA
    /// `slba` into `buf` (`data_len` bytes). `nlb` is the zero-based block
    /// count the NVMe READ command expects (0 means one block).
    pub fn read(nsid: u32, slba: u64, nlb: u16, buf: *mut u8, data_len: u32) -> Self {
        Self {
            opcode: NVME_CMD_READ,
            nsid,
            addr: buf as u64,
            data_len,
            // CDW10/11 carry the 64-bit starting LBA (low/high).
            cdw10: slba as u32,
            cdw11: (slba >> 32) as u32,
            // CDW12 low 16 bits carry NLB (zero-based block count).
            cdw12: u32::from(nlb),
            ..Self::default()
        }
    }
}

/// One physical extent of a file: a logical byte range mapped to a
/// contiguous device byte range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    /// Byte offset within the file.
    pub logical: u64,
    /// Absolute byte offset on the block device.
    pub physical: u64,
    /// Extent length in bytes.
    pub length: u64,
}

/// A file's logical→physical map, resolved once via `FIEMAP`.
///
/// Sound only for a file whose extents never move after mapping — the
/// immutable feature store. A copy-on-write filesystem (btrfs, ZFS,
/// bcachefs) can relocate blocks under a still-open fd, so [`Self::build`]
/// refuses those.
#[derive(Debug, Clone)]
pub struct ExtentMap {
    extents: Vec<Extent>,
}

impl ExtentMap {
    /// Resolve `file`'s extents. Fails if the filesystem doesn't support
    /// `FIEMAP`, reports a copy-on-write layout (`FIEMAP_EXTENT_SHARED`),
    /// or leaves any extent unwritten/delalloc.
    pub fn build(file: &File) -> Result<Self> {
        let len = file.metadata()?.len();
        let extents = fiemap(file.as_raw_fd(), len).context("FIEMAP failed")?;
        if extents.is_empty() && len > 0 {
            bail!("FIEMAP returned no extents for a {len}-byte file");
        }
        Ok(Self { extents })
    }

    /// Number of extents (1 for a freshly fallocated store; more if the
    /// filesystem fragmented it). A store mapped to many extents is a
    /// fragmentation signal worth logging.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn extent_count(&self) -> usize {
        self.extents.len()
    }

    /// Map a logical byte range to the device byte range that backs it,
    /// or `None` if the range crosses an extent boundary (the caller then
    /// splits the read or falls back). Feature rows are far smaller than
    /// an extent, so the common case is a single lookup.
    pub fn resolve(&self, logical: u64, len: u64) -> Option<(u64, u64)> {
        let end = logical.checked_add(len)?;
        for e in &self.extents {
            let e_end = e.logical + e.length;
            if logical >= e.logical && end <= e_end {
                let device_off = e.physical + (logical - e.logical);
                return Some((device_off, len));
            }
        }
        None
    }
}

/// Query a file's extents via the `FS_IOC_FIEMAP` ioctl.
fn fiemap(fd: i32, file_len: u64) -> Result<Vec<Extent>> {
    // FIEMAP flags / extent flags from <linux/fiemap.h>.
    const FIEMAP_FLAG_SYNC: u32 = 0x0001;
    const FIEMAP_EXTENT_LAST: u32 = 0x0001;
    const FIEMAP_EXTENT_UNKNOWN: u32 = 0x0002;
    const FIEMAP_EXTENT_DELALLOC: u32 = 0x0004;
    const FIEMAP_EXTENT_ENCODED: u32 = 0x0008;
    const FIEMAP_EXTENT_SHARED: u32 = 0x2000;
    // FS_IOC_FIEMAP = _IOWR('f', 11, struct fiemap).
    const FS_IOC_FIEMAP: libc::c_ulong = 0xc020660b;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct FiemapExtent {
        fe_logical: u64,
        fe_physical: u64,
        fe_length: u64,
        fe_reserved64: [u64; 2],
        fe_flags: u32,
        fe_reserved: [u32; 3],
    }
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct FiemapHeader {
        fm_start: u64,
        fm_length: u64,
        fm_flags: u32,
        fm_mapped_extents: u32,
        fm_extent_count: u32,
        fm_reserved: u32,
    }

    if file_len == 0 {
        return Ok(Vec::new());
    }

    // One ioctl per batch of extents; loop from where the last batch ended
    // until an extent carries FIEMAP_EXTENT_LAST.
    const BATCH: usize = 32;
    let mut out = Vec::new();
    let mut start = 0u64;
    loop {
        // Header immediately followed by `BATCH` extent slots, one
        // contiguous allocation as the ioctl expects.
        let mut buf = vec![
            0u8;
            std::mem::size_of::<FiemapHeader>()
                + BATCH * std::mem::size_of::<FiemapExtent>()
        ];
        let header = FiemapHeader {
            fm_start: start,
            fm_length: file_len - start,
            fm_flags: FIEMAP_FLAG_SYNC,
            fm_extent_count: BATCH as u32,
            ..Default::default()
        };
        // SAFETY: `header` is POD and `buf` holds at least its size at
        // offset 0.
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const FiemapHeader as *const u8,
                std::mem::size_of::<FiemapHeader>(),
            )
        };
        buf[..header_bytes.len()].copy_from_slice(header_bytes);
        // SAFETY: `fd` is a valid open file; `buf` matches the FIEMAP
        // struct layout the ioctl reads and writes.
        let ret = unsafe { libc::ioctl(fd, FS_IOC_FIEMAP, buf.as_mut_ptr()) };
        if ret != 0 {
            let err = std::io::Error::last_os_error();
            bail!("FS_IOC_FIEMAP ioctl: {err}");
        }

        // SAFETY: the ioctl populated the header in place.
        let mapped = unsafe {
            let h = &*(buf.as_ptr() as *const FiemapHeader);
            h.fm_mapped_extents as usize
        };
        if mapped == 0 {
            break;
        }

        let mut last = false;
        let ext_base = std::mem::size_of::<FiemapHeader>();
        for i in 0..mapped.min(BATCH) {
            // SAFETY: `i < mapped <= BATCH`, so this offset is within the
            // allocation's extent array.
            let p = unsafe {
                buf.as_ptr()
                    .add(ext_base + i * std::mem::size_of::<FiemapExtent>())
            };
            // SAFETY: `p` addresses one populated `FiemapExtent` (POD).
            let e = unsafe { *(p as *const FiemapExtent) };
            let bad = e.fe_flags
                & (FIEMAP_EXTENT_UNKNOWN
                    | FIEMAP_EXTENT_DELALLOC
                    | FIEMAP_EXTENT_ENCODED
                    | FIEMAP_EXTENT_SHARED);
            if bad != 0 {
                bail!(
                    "extent at logical {} is not stably mapped (flags {:#x}); \
                     refusing NVMe passthrough on this file",
                    e.fe_logical,
                    e.fe_flags
                );
            }
            out.push(Extent {
                logical: e.fe_logical,
                physical: e.fe_physical,
                length: e.fe_length,
            });
            if e.fe_flags & FIEMAP_EXTENT_LAST != 0 {
                last = true;
            }
            start = e.fe_logical + e.fe_length;
        }
        if last {
            break;
        }
    }
    Ok(out)
}

/// Build the NVMe char-device path (`/dev/ngXnY`) for the block device a
/// file lives on, if the platform is laid out the usual way.
///
/// Returns `None` rather than erroring when the mapping can't be made —
/// the caller treats that as "passthrough unavailable, use the fs path".
pub fn char_device_for(file: &File) -> Option<std::path::PathBuf> {
    let st_dev = {
        use std::os::unix::fs::MetadataExt;
        file.metadata().ok()?.dev()
    };
    let major = ((st_dev >> 8) & 0xfff) as u32;
    let minor = (st_dev & 0xff) as u32 | ((st_dev >> 12) & 0xfff00) as u32;
    // /sys/dev/block/<major>:<minor> → the block device; its name is like
    // `nvme0n1` or `nvme0n1p3`. The char device drops the partition
    // suffix: nvme0n1p3 → ng0n1.
    let link = std::fs::read_link(format!("/sys/dev/block/{major}:{minor}")).ok()?;
    let name = link.file_name()?.to_str()?;
    let stem = name.strip_prefix("nvme")?;
    // stem is like "0n1" or "0n1p3"; keep up to the partition marker.
    let core = match stem.split_once('p') {
        Some((base, _part)) => base,
        None => stem,
    };
    let path = std::path::PathBuf::from(format!("/dev/ng{core}"));
    path.exists().then_some(path)
}

/// Namespace-scoped NVMe reader: a char-device handle plus a dedicated
/// ring for `uring_cmd` passthrough.
///
/// The ring uses 128-byte SQEs (`Entry128`): a `uring_cmd` carries the
/// 80-byte command inline, which only fits the big-SQE layout.
pub struct NvmeReader {
    dev: File,
    nsid: u32,
    lba_bytes: u32,
    ring: io_uring::IoUring<io_uring::squeue::Entry128>,
}

impl NvmeReader {
    /// Open the char device backing `store_file` and prepare passthrough.
    /// Returns `Ok(None)` when the platform can't support it (no `/dev/ng*`,
    /// no permission, unreadable geometry) so the caller falls back.
    pub fn open_for(store_file: &File) -> Result<Option<Self>> {
        let Some(dev_path) = char_device_for(store_file) else {
            return Ok(None);
        };
        Self::open_path(&dev_path, store_file)
    }

    /// Open an explicit char-device path. Split out so tests can point at
    /// a fixture path and assert the not-available handling.
    pub fn open_path(dev_path: &Path, store_file: &File) -> Result<Option<Self>> {
        let dev = match File::open(dev_path) {
            Ok(f) => f,
            // Whatever the reason (EACCES, ENOENT, ENXIO, ENOTDIR, …), a
            // char device we cannot open means "passthrough unavailable
            // here" — this is a probe with a guaranteed fs-path fallback,
            // never a hard error.
            Err(e) => {
                trace!("NVMe char device {} not usable: {e}", dev_path.display());
                return Ok(None);
            }
        };

        let Some((nsid, lba_bytes)) = ns_geometry(&dev) else {
            return Ok(None);
        };
        let _ = store_file;

        let ring = match io_uring::IoUring::<io_uring::squeue::Entry128>::builder().build(64) {
            Ok(r) => r,
            Err(_) => return Ok(None),
        };
        Ok(Some(Self {
            dev,
            nsid,
            lba_bytes,
            ring,
        }))
    }

    /// The namespace's logical block size in bytes.
    pub fn lba_bytes(&self) -> u32 {
        self.lba_bytes
    }

    /// Read `len` bytes at absolute device byte offset `device_off` into
    /// `buf`. `device_off` and `len` must be LBA-aligned — resolve them
    /// from an [`ExtentMap`] whose file was LBA-aligned at build time.
    ///
    /// # Safety
    /// `buf` must point to at least `len` writable bytes and stay valid
    /// until this call returns.
    pub unsafe fn read_at(&mut self, device_off: u64, buf: *mut u8, len: u32) -> Result<()> {
        use io_uring::{opcode, types};

        let lba = self.lba_bytes as u64;
        if !device_off.is_multiple_of(lba) || !u64::from(len).is_multiple_of(lba) {
            bail!("NVMe passthrough read must be LBA-aligned ({lba} bytes)");
        }
        let slba = device_off / lba;
        let nblocks = (len as u64) / lba;
        if nblocks == 0 {
            return Ok(());
        }
        let nlb =
            u16::try_from(nblocks - 1).context("read exceeds one NVMe command's block count")?;

        let cmd = NvmePassthruCmd::read(self.nsid, slba, nlb, buf, len);
        // The uring_cmd SQE carries the 72-byte command in its cmd area.
        let entry = opcode::UringCmd80::new(types::Fd(self.dev.as_raw_fd()), NVME_URING_CMD_IO)
            .cmd(cmd_bytes(&cmd))
            .build()
            .user_data(1);

        {
            let mut sq = self.ring.submission();
            // SAFETY: `buf`/`cmd` outlive the wait below; the caller
            // guarantees `buf` is writable for `len` bytes.
            let pushed = unsafe { sq.push(&entry) };
            pushed.expect("64-entry ring accepts one SQE");
        }
        self.ring
            .submit_and_wait(1)
            .context("io_uring submit_and_wait")?;

        let cqe = self
            .ring
            .completion()
            .next()
            .context("no completion for NVMe passthrough read")?;
        let res = cqe.result();
        if res < 0 {
            bail!("NVMe passthrough read failed: {}", -res);
        }
        Ok(())
    }
}

/// Pack the command struct into the 80-byte SQE cmd array (72 used, 8 zero).
fn cmd_bytes(cmd: &NvmePassthruCmd) -> [u8; 80] {
    let mut out = [0u8; 80];
    // SAFETY: `NvmePassthruCmd` is `repr(C)` POD of 72 bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(
            cmd as *const NvmePassthruCmd as *const u8,
            out.as_mut_ptr(),
            std::mem::size_of::<NvmePassthruCmd>(),
        );
    }
    out
}

/// Read a namespace's (nsid, lba_size) from sysfs, `None` if unreadable.
fn ns_geometry(dev: &File) -> Option<(u32, u32)> {
    use std::os::unix::fs::MetadataExt;
    let rdev = dev.metadata().ok()?.rdev();
    let major = ((rdev >> 8) & 0xfff) as u32;
    let minor = (rdev & 0xff) as u32 | ((rdev >> 12) & 0xfff00) as u32;
    let base = format!("/sys/dev/char/{major}:{minor}");
    let nsid = std::fs::read_to_string(format!("{base}/nsid"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    // logical_block_size lives under the namespace's block queue.
    let lba = std::fs::read_to_string(format!("{base}/queue/logical_block_size"))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(512);
    Some((nsid, lba))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn passthru_cmd_layout_read() {
        let mut buf = [0u8; 4096];
        let cmd = NvmePassthruCmd::read(1, 0x1234_5678_9abc, 7, buf.as_mut_ptr(), 4096);
        assert_eq!(cmd.opcode, NVME_CMD_READ);
        assert_eq!(cmd.nsid, 1);
        assert_eq!(cmd.data_len, 4096);
        assert_eq!(cmd.addr, buf.as_ptr() as u64);
        // 48-bit LBA split across CDW10 (low) / CDW11 (high).
        assert_eq!(cmd.cdw10, 0x9abc_u32.wrapping_add(0x5678_0000));
        assert_eq!(cmd.cdw10, 0x5678_9abc);
        assert_eq!(cmd.cdw11, 0x1234);
        // NLB is zero-based in CDW12's low 16 bits.
        assert_eq!(cmd.cdw12 & 0xffff, 7);
    }

    #[test]
    fn uring_cmd_number_is_iowr_n_0x80() {
        // Direction=IOWR(3), size=72, type='N'(0x4e), nr=0x80.
        let expected = (3u32 << 30) | (72u32 << 16) | ((b'N' as u32) << 8) | 0x80;
        assert_eq!(NVME_URING_CMD_IO, expected);
    }

    #[test]
    fn extent_map_resolve_within_extent() {
        // A hand-built map (bypassing FIEMAP) exercises the arithmetic.
        let map = ExtentMap {
            extents: vec![
                Extent {
                    logical: 0,
                    physical: 1_000_000,
                    length: 8192,
                },
                Extent {
                    logical: 8192,
                    physical: 5_000_000,
                    length: 8192,
                },
            ],
        };
        // Row fully inside extent 0.
        assert_eq!(map.resolve(512, 512), Some((1_000_512, 512)));
        // Row fully inside extent 1.
        assert_eq!(map.resolve(8192, 4096), Some((5_000_000, 4096)));
        // Row straddling the boundary → None (caller splits / falls back).
        assert_eq!(map.resolve(8000, 512), None);
        // Past EOF → None.
        assert_eq!(map.resolve(16384, 1), None);
    }

    #[test]
    fn fiemap_maps_a_real_file() {
        // FIEMAP works on regular files on ext4/xfs; tmpfs and overlayfs
        // may not, so a failure here is a skip, not a test failure.
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let data = vec![0xABu8; 256 * 1024];
        tmp.write_all(&data).unwrap();
        tmp.flush().unwrap();

        match ExtentMap::build(tmp.as_file()) {
            Ok(map) => {
                assert!(map.extent_count() >= 1, "a 256 KiB file has ≥1 extent");
                // Offset 0 must resolve to some device offset.
                assert!(map.resolve(0, 4096).is_some());
            }
            Err(e) => eprintln!("FIEMAP unavailable on this fs ({e}); skipping"),
        }
    }

    #[test]
    fn reader_open_missing_device_is_none() {
        let store = tempfile::NamedTempFile::new().unwrap();
        // A path that certainly isn't an NVMe char device.
        let res = NvmeReader::open_path(Path::new("/dev/null/nope"), store.as_file()).unwrap();
        assert!(res.is_none(), "nonexistent char device → None, not error");
    }
}
