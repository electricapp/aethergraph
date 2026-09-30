//! NVMe passthrough reads over `io_uring` — the storage twin of AF_XDP.
//!
//! The feature store, once built, is immutable: its file extents are
//! fallocated and never move. That makes it sound to resolve each logical
//! read to an absolute namespace byte offset once (via `FIEMAP`) and then
//! issue `NVME_URING_CMD_IO` reads against the namespace's generic
//! character device (`/dev/ngXnY`), bypassing the filesystem and block
//! layer entirely — the command goes straight to the NVMe controller.
//!
//! This module is split so the parts that need no hardware are fully
//! testable everywhere:
//!
//! - [`NamespaceTarget`] resolves a file's block device to the namespace
//!   behind it — char device, nsid, LBA size, transfer limit, and the
//!   partition's offset — from sysfs, refusing anything it cannot pin down.
//! - [`ExtentMap`] resolves file offsets to namespace byte offsets with
//!   `FIEMAP`, shifted by the partition start and bounds-checked once.
//! - [`NvmePassthruCmd`] builds the 72-byte `nvme_uring_cmd` payload and
//!   is verified by field layout, no device required.
//! - [`NvmeReader`] owns the character-device handle plus ring and is the
//!   only part that needs `/dev/ng*` and privilege.

#![cfg(target_os = "linux")]

use anyhow::{Context, Result, bail, ensure};
use std::fs::File;
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use tracing::warn;

/// NVMe read opcode (`nvme_cmd_read`).
const NVME_CMD_READ: u8 = 0x02;
/// `io_uring` async command for NVMe char devices (`NVME_URING_CMD_IO`).
/// `_IOWR('N', 0x80, struct nvme_passthru_cmd)` — the ioctl-encoded
/// command number io_uring's `cmd_op` expects for a passthrough read.
const NVME_URING_CMD_IO: u32 = nvme_ioctl_iowr(0x80);
/// `NVME_IOCTL_ID` = `_IO('N', 0x40)`: returns the namespace id an NVMe
/// device handle addresses.
const NVME_IOCTL_ID: libc::c_ulong = ((b'N' as libc::c_ulong) << 8) | 0x40;
/// sysfs reports block-device geometry (`start`, `size`) in 512-byte
/// sectors whatever the namespace's LBA size.
const SYSFS_SECTOR: u64 = 512;

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

/// An NVMe completion status field, phase bit stripped — what the kernel
/// posts as a positive `uring_cmd` result when the controller fails a
/// command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvmeStatus(u16);

impl NvmeStatus {
    /// Status code type (generic, command-specific, media, path, vendor).
    pub fn sct(self) -> u8 {
        ((self.0 >> 8) & 0x7) as u8
    }

    /// Status code within its type.
    pub fn sc(self) -> u8 {
        (self.0 & 0xff) as u8
    }

    /// The controller's Do Not Retry bit.
    pub fn do_not_retry(self) -> bool {
        self.0 & 0x4000 != 0
    }

    fn describe(self) -> &'static str {
        match (self.sct(), self.sc()) {
            (0, 0x01) => "invalid command opcode",
            (0, 0x02) => "invalid field in command",
            (0, 0x04) => "data transfer error",
            (0, 0x06) => "internal error",
            (0, 0x0b) => "invalid namespace or format",
            (0, 0x80) => "LBA out of range",
            (0, 0x81) => "capacity exceeded",
            (0, 0x82) => "namespace not ready",
            (0, _) => "generic command status",
            (1, _) => "command specific status",
            (2, 0x81) => "unrecovered read error",
            (2, 0x82) => "end-to-end guard check error",
            (2, 0x83) => "end-to-end application tag check error",
            (2, 0x84) => "end-to-end reference tag check error",
            (2, 0x86) => "access denied",
            (2, 0x87) => "deallocated or unwritten logical block",
            (2, _) => "media or data integrity error",
            (3, _) => "path related status",
            (7, _) => "vendor specific status",
            _ => "reserved status code type",
        }
    }
}

impl std::fmt::Display for NvmeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NVMe status SCT {:#x} SC {:#04x} ({}{})",
            self.sct(),
            self.sc(),
            self.describe(),
            if self.do_not_retry() { ", DNR" } else { "" }
        )
    }
}

/// Why one passthrough command did not deliver its data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassthruFailure {
    /// Host-side failure: a negated errno from the driver.
    Errno(i32),
    /// The controller completed the command with a nonzero status.
    Status(NvmeStatus),
}

impl PassthruFailure {
    /// Classify a `uring_cmd` CQE result. Zero is the only success: the
    /// kernel reports a controller error as the positive status field and
    /// a host-side error as a negative errno.
    pub fn from_cqe_result(res: i32) -> Option<Self> {
        match res {
            0 => None,
            r if r < 0 => Some(Self::Errno(-r)),
            r => Some(Self::Status(NvmeStatus((r & 0x7fff) as u16))),
        }
    }
}

impl std::fmt::Display for PassthruFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Errno(e) => write!(f, "{}", std::io::Error::from_raw_os_error(*e)),
            Self::Status(s) => write!(f, "{s}"),
        }
    }
}

/// The namespace a file's blocks live on, resolved from sysfs once.
///
/// FIEMAP reports physical offsets relative to the filesystem's block
/// device — a partition, usually — while the generic char device addresses
/// the whole namespace from LBA 0. This carries the partition's start and
/// length so every extent is shifted and bounds-checked at map time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceTarget {
    char_dev: PathBuf,
    nsid: u32,
    lba_bytes: u32,
    /// Controller's maximum single-command transfer (MDTS, surfaced by
    /// the driver as `max_hw_sectors_kb`).
    max_transfer: u32,
    /// Byte offset of the filesystem's block device from namespace LBA 0.
    partition_start: u64,
    /// Byte length of the filesystem's block device.
    partition_len: u64,
}

impl NamespaceTarget {
    /// Resolve the namespace behind `file`'s filesystem. `None` when the
    /// device is not a plain NVMe namespace or partition of one (device
    /// mapper, md, loop, network filesystems) or any geometry attribute is
    /// missing — passthrough is then unavailable, never guessed at.
    pub fn for_file(file: &File) -> Option<Self> {
        let dev = file.metadata().ok()?.dev();
        let (major, minor) = (libc::major(dev), libc::minor(dev));
        let block_dir = std::fs::canonicalize(format!("/sys/dev/block/{major}:{minor}")).ok()?;
        Self::from_sysfs(&block_dir, Path::new("/dev"))
    }

    /// Resolve from a block device's sysfs directory. `dev_dir` holds the
    /// char device nodes. Split out so a fixture tree can stand in for
    /// sysfs in tests.
    fn from_sysfs(block_dir: &Path, dev_dir: &Path) -> Option<Self> {
        // A partition's sysfs directory nests inside its disk's and carries
        // a `partition` attribute; `start` and `size` are 512-byte sectors.
        let (disk_dir, partition_start, partition_len) = if block_dir.join("partition").exists() {
            let start = read_sysfs_u64(&block_dir.join("start"))?.checked_mul(SYSFS_SECTOR)?;
            let len = read_sysfs_u64(&block_dir.join("size"))?.checked_mul(SYSFS_SECTOR)?;
            (block_dir.parent()?, start, len)
        } else {
            let len = read_sysfs_u64(&block_dir.join("size"))?.checked_mul(SYSFS_SECTOR)?;
            (block_dir, 0, len)
        };
        let char_name = generic_char_name(disk_dir.file_name()?.to_str()?)?;

        let nsid = u32::try_from(read_sysfs_u64(&disk_dir.join("nsid"))?).ok()?;
        let lba_bytes =
            u32::try_from(read_sysfs_u64(&disk_dir.join("queue/logical_block_size"))?).ok()?;
        if nsid == 0 || lba_bytes < 512 || !lba_bytes.is_power_of_two() {
            return None;
        }
        let max_transfer = u32::try_from(
            read_sysfs_u64(&disk_dir.join("queue/max_hw_sectors_kb"))?.checked_mul(1024)?,
        )
        .ok()
        .filter(|&b| b >= lba_bytes)?;
        let namespace_len = read_sysfs_u64(&disk_dir.join("size"))?.checked_mul(SYSFS_SECTOR)?;

        // The partition must be LBA-aligned and lie inside the namespace,
        // or shifted extents could address blocks outside it.
        if !partition_start.is_multiple_of(u64::from(lba_bytes))
            || partition_start.checked_add(partition_len)? > namespace_len
            || partition_len == 0
        {
            return None;
        }

        Some(Self {
            char_dev: dev_dir.join(char_name),
            nsid,
            lba_bytes,
            max_transfer,
            partition_start,
            partition_len,
        })
    }

    /// The namespace's logical block size in bytes.
    pub fn lba_bytes(&self) -> u32 {
        self.lba_bytes
    }

    /// Largest single transfer the controller accepts, in bytes.
    pub fn max_transfer_bytes(&self) -> u32 {
        self.max_transfer
    }
}

/// `nvme<ctrl>n<ns>` → `ng<ctrl>n<ns>`. Hidden per-path disks
/// (`nvme0c1n1`) and anything else that is not a namespace disk map to
/// `None`.
fn generic_char_name(disk: &str) -> Option<String> {
    let (ctrl, ns) = disk.strip_prefix("nvme")?.split_once('n')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    (digits(ctrl) && digits(ns)).then(|| format!("ng{ctrl}n{ns}"))
}

fn read_sysfs_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// One physical extent of a file: a logical byte range mapped to a
/// contiguous range of namespace bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    /// Byte offset within the file.
    pub logical: u64,
    /// Byte offset from namespace LBA 0 (after [`ExtentMap::build`]) or
    /// from the filesystem's block device (as FIEMAP reports it).
    pub physical: u64,
    /// Extent length in bytes.
    pub length: u64,
}

/// A file's logical→namespace map, resolved once via `FIEMAP`.
///
/// Sound only for a file whose extents never move after mapping — the
/// immutable feature store. A copy-on-write filesystem (btrfs, ZFS,
/// bcachefs) can relocate blocks under a still-open fd, so [`Self::build`]
/// refuses those.
#[derive(Debug, Clone)]
pub struct ExtentMap {
    /// Sorted by `logical`, non-overlapping, `physical` in namespace bytes.
    extents: Vec<Extent>,
}

impl ExtentMap {
    /// Resolve `file`'s extents onto `target`'s namespace. Fails if the
    /// filesystem doesn't support `FIEMAP`, any extent is not plain data
    /// at a stable location (see [`check_extent_flags`]), or an extent
    /// falls outside the partition.
    pub fn build(file: &File, target: &NamespaceTarget) -> Result<Self> {
        let len = file.metadata()?.len();
        let extents = fiemap(file.as_raw_fd(), len).context("FIEMAP failed")?;
        if extents.is_empty() && len > 0 {
            bail!("FIEMAP returned no extents for a {len}-byte file");
        }
        Self::place(extents, target.partition_start, target.partition_len)
    }

    /// Shift partition-relative extents to namespace offsets, checking
    /// each lies inside the partition.
    fn place(mut extents: Vec<Extent>, partition_start: u64, partition_len: u64) -> Result<Self> {
        for e in &mut extents {
            let end = e
                .physical
                .checked_add(e.length)
                .context("extent end overflows")?;
            ensure!(
                end <= partition_len,
                "extent at logical {} ends at device byte {end}, past the \
                 {partition_len}-byte block device",
                e.logical
            );
            e.physical += partition_start;
        }
        extents.sort_unstable_by_key(|e| e.logical);
        for w in extents.windows(2) {
            ensure!(
                w[0].logical + w[0].length <= w[1].logical,
                "FIEMAP reported overlapping extents at logical {}",
                w[1].logical
            );
        }
        Ok(Self { extents })
    }

    /// Number of extents (1 for a freshly fallocated store; more if the
    /// filesystem fragmented it).
    #[cfg(test)]
    pub fn extent_count(&self) -> usize {
        self.extents.len()
    }

    /// Namespace byte offset backing `logical..logical + len`, or `None`
    /// if the range crosses an extent boundary or leaves the file (the
    /// caller then falls back).
    pub fn resolve(&self, logical: u64, len: u64) -> Option<u64> {
        let end = logical.checked_add(len)?;
        // First extent ending past `logical`; only it can contain the start.
        let i = self
            .extents
            .partition_point(|e| e.logical + e.length <= logical);
        let e = self.extents.get(i)?;
        (logical >= e.logical && end <= e.logical + e.length)
            .then(|| e.physical + (logical - e.logical))
    }
}

/// FIEMAP extent flags from `<linux/fiemap.h>`.
const FIEMAP_EXTENT_LAST: u32 = 0x0001;
const FIEMAP_EXTENT_MERGED: u32 = 0x1000;

/// Refuse an extent whose bytes are not plain data at `fe_physical`.
///
/// Only `LAST` and `MERGED` are harmless. Everything else means a raw
/// device read would not return the file's contents: an unwritten
/// (preallocated) extent reads as zeros through the filesystem but returns
/// stale disk blocks raw; delalloc/unknown extents have no stable
/// location; encoded, encrypted, inline, and tail-packed data does not sit
/// in plain blocks at that offset; not-aligned data does not start on a
/// block boundary; shared extents can be relocated by copy-on-write.
fn check_extent_flags(logical: u64, flags: u32) -> Result<()> {
    let bad = flags & !(FIEMAP_EXTENT_LAST | FIEMAP_EXTENT_MERGED);
    if bad != 0 {
        bail!(
            "extent at logical {logical} is not plain stable data (flags {flags:#x}); \
             refusing NVMe passthrough on this file"
        );
    }
    Ok(())
}

/// Query a file's extents via the `FS_IOC_FIEMAP` ioctl. Physical offsets
/// are relative to the filesystem's block device.
fn fiemap(fd: i32, file_len: u64) -> Result<Vec<Extent>> {
    const FIEMAP_FLAG_SYNC: u32 = 0x0001;
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

        // SAFETY: the ioctl populated the header in place; `buf` is only
        // byte-aligned, hence the unaligned read.
        let mapped = unsafe { (buf.as_ptr() as *const FiemapHeader).read_unaligned() }
            .fm_mapped_extents as usize;
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
            // SAFETY: `p` addresses one populated `FiemapExtent` (POD),
            // read unaligned from the byte buffer.
            let e = unsafe { (p as *const FiemapExtent).read_unaligned() };
            check_extent_flags(e.fe_logical, e.fe_flags)?;
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
        if last || start >= file_len {
            break;
        }
    }
    Ok(out)
}

/// Namespace-scoped NVMe reader: a char-device handle plus a dedicated
/// ring for `uring_cmd` passthrough.
///
/// The driver accepts `uring_cmd` only on a ring with both 128-byte SQEs
/// (the 80-byte command rides inline) and 32-byte CQEs (the completion
/// carries the command's result dword); either one missing fails every
/// command with `EOPNOTSUPP`.
pub struct NvmeReader {
    dev: File,
    nsid: u32,
    lba_bytes: u32,
    max_transfer: u32,
    ring: io_uring::IoUring<io_uring::squeue::Entry128, io_uring::cqueue::Entry32>,
}

/// Depth of the passthrough ring. Batches are submitted in chunks of this
/// many commands, so one chunk is one device round-trip.
const RING_ENTRIES: usize = 64;

impl NvmeReader {
    /// Open `target`'s char device and prepare passthrough. An error means
    /// passthrough is unavailable here (no device node, no permission, a
    /// device that is not the namespace sysfs described, or a kernel
    /// without big SQE/CQE rings).
    pub fn open(target: &NamespaceTarget) -> Result<Self> {
        let dev = File::open(&target.char_dev)
            .with_context(|| format!("open {}", target.char_dev.display()))?;
        ensure!(
            dev.metadata()?.file_type().is_char_device(),
            "{} is not a character device",
            target.char_dev.display()
        );
        // The char device is found by name; confirm it addresses the
        // namespace the geometry came from before trusting either.
        // SAFETY: `dev` is an open fd; NVME_IOCTL_ID takes no argument.
        let nsid = unsafe { libc::ioctl(dev.as_raw_fd(), NVME_IOCTL_ID) };
        ensure!(
            nsid >= 0,
            "NVME_IOCTL_ID on {}: {}",
            target.char_dev.display(),
            std::io::Error::last_os_error()
        );
        ensure!(
            nsid as u32 == target.nsid,
            "{} addresses nsid {nsid}, sysfs described nsid {}",
            target.char_dev.display(),
            target.nsid
        );

        let ring =
            io_uring::IoUring::<io_uring::squeue::Entry128, io_uring::cqueue::Entry32>::builder()
                .build(RING_ENTRIES as u32)
                .context("passthrough ring (SQE128 + CQE32)")?;
        Ok(Self {
            dev,
            nsid: target.nsid,
            lba_bytes: target.lba_bytes,
            max_transfer: target.max_transfer,
            ring,
        })
    }

    /// Read every `(device_off, buf, len)` request as one pipelined run of
    /// NVMe commands. Offsets and lengths must be LBA-aligned namespace
    /// byte offsets from an [`ExtentMap`].
    ///
    /// The whole batch is in flight at once, up to the ring's depth, so a
    /// gather costs one device round-trip per chunk rather than one per
    /// row. Every submitted command is reaped before returning, on success
    /// and on error alike: the drive DMAs into the caller's buffers, so
    /// returning while a command is outstanding would let it land in memory
    /// the caller has already reclaimed. Any nonzero completion — errno or
    /// controller status — fails the batch.
    ///
    /// # Safety
    /// Each `buf` must point to at least `len` writable bytes and stay
    /// valid until this call returns.
    pub unsafe fn read_batch(&mut self, reqs: &[(u64, *mut u8, u32)]) -> Result<()> {
        use io_uring::{opcode, types};

        if reqs.is_empty() {
            return Ok(());
        }

        // Validate up front so a bad request never strands earlier
        // commands in flight (see `command_for`).
        let commands = reqs
            .iter()
            .map(|&(off, buf, len)| {
                command_for(self.nsid, self.lba_bytes, self.max_transfer, off, buf, len)
            })
            .collect::<Result<Vec<_>>>()?;

        // A completion left over from an aborted batch indexes a request
        // slice that no longer exists; counting it here would let this
        // batch return before its own commands land.
        let stale = self.ring.completion().count();
        if stale > 0 {
            warn!("NVMe passthrough: discarded {stale} stale completions");
        }

        let fd = types::Fd(self.dev.as_raw_fd());
        let mut first_err: Option<anyhow::Error> = None;

        for (chunk, req_chunk) in commands.chunks(RING_ENTRIES).zip(reqs.chunks(RING_ENTRIES)) {
            let mut submitted = 0usize;
            for (i, cmd) in chunk.iter().enumerate() {
                let entry = opcode::UringCmd80::new(fd, NVME_URING_CMD_IO)
                    .cmd(cmd_bytes(cmd))
                    .build()
                    .user_data(i as u64);
                let mut sq = self.ring.submission();
                // SAFETY: the destination buffers are the caller's, valid
                // for the duration of this call, and the drain below runs
                // before it returns. The chunk is sized to the ring, so
                // the push cannot fail for lack of space.
                let pushed = unsafe { sq.push(&entry) };
                pushed.expect("chunk is sized to the ring's entry count");
                submitted += 1;
            }

            // Drain unconditionally: `submitted` commands are visible to
            // the kernel and each produces exactly one completion.
            let mut completed = 0usize;
            let mut wait_failures = 0u32;
            while completed < submitted {
                match self.ring.submit_and_wait(1) {
                    Ok(_) => wait_failures = 0,
                    Err(e) => {
                        // Transient errnos (EINTR/EAGAIN) clear on retry.
                        // A persistent failure means quiescence cannot be
                        // established, and returning would hand the drive
                        // memory the caller is free to reuse.
                        wait_failures += 1;
                        if wait_failures >= 1000 {
                            tracing::error!(
                                error = %e,
                                completed,
                                submitted,
                                "NVMe passthrough drain failed persistently; aborting to \
                                 keep the device from writing into reclaimed memory"
                            );
                            std::process::abort();
                        }
                        continue;
                    }
                }
                for cqe in self.ring.completion() {
                    completed += 1;
                    if let Some(failure) = PassthruFailure::from_cqe_result(cqe.result())
                        && first_err.is_none()
                    {
                        let idx = cqe.user_data() as usize;
                        let off = req_chunk.get(idx).map_or(0, |&(off, _, _)| off);
                        first_err = Some(anyhow::anyhow!(
                            "NVMe passthrough read at namespace byte {off} failed: {failure}"
                        ));
                    }
                }
            }
        }

        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Check one request against the namespace geometry and the controller's
/// transfer limit, returning the command it maps to.
///
/// Every request in a batch is checked before any of them is submitted: a
/// rejection partway through would leave earlier commands in flight
/// against buffers the caller is about to reclaim.
fn command_for(
    nsid: u32,
    lba_bytes: u32,
    max_transfer: u32,
    device_off: u64,
    buf: *mut u8,
    len: u32,
) -> Result<NvmePassthruCmd> {
    let lba = u64::from(lba_bytes);
    if len == 0 || !device_off.is_multiple_of(lba) || !u64::from(len).is_multiple_of(lba) {
        bail!("NVMe passthrough read must be a nonzero LBA-aligned span ({lba} bytes)");
    }
    if len > max_transfer {
        bail!(
            "NVMe passthrough read of {len} bytes exceeds the controller's \
             {max_transfer} byte maximum transfer"
        );
    }
    let nblocks = u64::from(len) / lba;
    let nlb = u16::try_from(nblocks - 1).context("read exceeds one NVMe command's block count")?;
    Ok(NvmePassthruCmd::read(nsid, device_off / lba, nlb, buf, len))
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
        assert_eq!(cmd.cdw10, 0x5678_9abc);
        assert_eq!(cmd.cdw11, 0x1234);
        // NLB is zero-based in CDW12's low 16 bits.
        assert_eq!(cmd.cdw12 & 0xffff, 7);
    }

    #[test]
    fn command_for_rejects_reads_past_the_controller_transfer_limit() {
        let mut buf = [0u8; 4096];
        let ptr = buf.as_mut_ptr();
        // MDTS is a hard controller limit: the drive rejects an oversized
        // command, so the batch must refuse it before anything is in
        // flight rather than discover it from a completion.
        let err = command_for(1, 512, 128 * 1024, 0, ptr, 256 * 1024).unwrap_err();
        assert!(
            err.to_string().contains("maximum transfer"),
            "unexpected error: {err}"
        );
        // Exactly at the limit is accepted.
        assert!(command_for(1, 512, 128 * 1024, 0, ptr, 128 * 1024).is_ok());
    }

    #[test]
    fn command_for_requires_nonzero_lba_aligned_spans() {
        let mut buf = [0u8; 4096];
        let ptr = buf.as_mut_ptr();
        for (off, len) in [(511u64, 512u32), (512, 511), (0, 1), (0, 0)] {
            let err = command_for(1, 512, 128 * 1024, off, ptr, len).unwrap_err();
            assert!(
                err.to_string().contains("LBA-aligned"),
                "offset {off} len {len} gave: {err}"
            );
        }
    }

    #[test]
    fn command_for_maps_offset_and_length_to_zero_based_blocks() {
        let mut buf = [0u8; 4096];
        let cmd = command_for(3, 512, 128 * 1024, 8192, buf.as_mut_ptr(), 2048).unwrap();
        assert_eq!(cmd.nsid, 3);
        // Byte offset 8192 over 512-byte blocks is LBA 16.
        assert_eq!(cmd.cdw10, 16);
        assert_eq!(cmd.cdw11, 0);
        // 2048 bytes is 4 blocks, and NLB is zero-based.
        assert_eq!(cmd.cdw12 & 0xffff, 3);
        assert_eq!(cmd.data_len, 2048);
    }

    #[test]
    fn uring_cmd_number_is_iowr_n_0x80() {
        // Direction=IOWR(3), size=72, type='N'(0x4e), nr=0x80.
        let expected = (3u32 << 30) | (72u32 << 16) | ((b'N' as u32) << 8) | 0x80;
        assert_eq!(NVME_URING_CMD_IO, expected);
        assert_eq!(NVME_IOCTL_ID, 0x4e40);
    }

    /// The kernel reports a controller failure as the positive status
    /// field, so only zero may count as a completed read.
    #[test]
    fn cqe_results_classify_errno_and_controller_status() {
        assert_eq!(PassthruFailure::from_cqe_result(0), None);
        assert_eq!(
            PassthruFailure::from_cqe_result(-libc::EINTR),
            Some(PassthruFailure::Errno(libc::EINTR))
        );

        let Some(PassthruFailure::Status(unrecovered)) = PassthruFailure::from_cqe_result(0x281)
        else {
            panic!("0x281 is a controller status");
        };
        assert_eq!((unrecovered.sct(), unrecovered.sc()), (2, 0x81));
        assert!(unrecovered.to_string().contains("unrecovered read error"));
        assert!(!unrecovered.do_not_retry());

        let Some(PassthruFailure::Status(lba_range)) = PassthruFailure::from_cqe_result(0x4080)
        else {
            panic!("0x4080 is a controller status");
        };
        assert_eq!((lba_range.sct(), lba_range.sc()), (0, 0x80));
        assert!(lba_range.do_not_retry());
        assert!(lba_range.to_string().contains("LBA out of range"));
    }

    #[test]
    fn extent_flags_admit_only_plain_stable_data() {
        assert!(check_extent_flags(0, 0).is_ok());
        assert!(check_extent_flags(0, FIEMAP_EXTENT_LAST | FIEMAP_EXTENT_MERGED).is_ok());
        // UNKNOWN, DELALLOC, ENCODED, DATA_ENCRYPTED, NOT_ALIGNED,
        // DATA_INLINE, DATA_TAIL, UNWRITTEN, SHARED.
        for flag in [0x2, 0x4, 0x8, 0x80, 0x100, 0x200, 0x400, 0x800, 0x2000] {
            assert!(
                check_extent_flags(0, flag | FIEMAP_EXTENT_LAST).is_err(),
                "flag {flag:#x} must be refused"
            );
        }
    }

    #[test]
    fn extent_map_resolve_within_extent() {
        let map = ExtentMap::place(
            vec![
                Extent {
                    logical: 8192,
                    physical: 5_000_000,
                    length: 8192,
                },
                Extent {
                    logical: 0,
                    physical: 1_000_000,
                    length: 8192,
                },
            ],
            0,
            u64::MAX,
        )
        .unwrap();
        // Row fully inside extent 0.
        assert_eq!(map.resolve(512, 512), Some(1_000_512));
        assert_eq!(map.resolve(0, 8192), Some(1_000_000));
        // Row fully inside extent 1.
        assert_eq!(map.resolve(8192, 4096), Some(5_000_000));
        assert_eq!(map.resolve(12288, 4096), Some(5_004_096));
        // Row straddling the boundary → None (caller falls back).
        assert_eq!(map.resolve(8000, 512), None);
        // Past EOF → None.
        assert_eq!(map.resolve(16384, 1), None);
        assert_eq!(map.resolve(u64::MAX, 2), None);
    }

    #[test]
    fn extents_shift_by_the_partition_start_and_stay_inside_it() {
        // FIEMAP offsets are partition-relative; a partition starting at
        // sector 2048 puts file block 0 one MiB into the namespace.
        let raw = vec![Extent {
            logical: 0,
            physical: 0,
            length: 65536,
        }];
        let map = ExtentMap::place(raw.clone(), 2048 * 512, 1 << 30).unwrap();
        assert_eq!(map.resolve(4096, 4096), Some(2048 * 512 + 4096));

        // An extent reaching past the partition's end is refused rather
        // than read from whatever follows it.
        let err = ExtentMap::place(raw, 2048 * 512, 32768).unwrap_err();
        assert!(err.to_string().contains("past the"), "{err}");
    }

    fn write(path: &Path, value: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, value).unwrap();
    }

    /// Lay out the sysfs attributes a namespace disk and one partition
    /// publish, under a temp dir.
    fn fixture_namespace(root: &Path, lba: u32) -> PathBuf {
        let disk = root.join("devices/nvme/nvme0/nvme0n1");
        write(&disk.join("nsid"), "1\n");
        write(&disk.join("size"), &format!("{}\n", 1u64 << 21)); // 1 GiB
        write(&disk.join("queue/logical_block_size"), &format!("{lba}\n"));
        write(&disk.join("queue/max_hw_sectors_kb"), "512\n");
        let part = disk.join("nvme0n1p3");
        write(&part.join("partition"), "3\n");
        write(&part.join("start"), "2048\n");
        write(&part.join("size"), "1048576\n"); // 512 MiB
        disk
    }

    #[test]
    fn partition_geometry_resolves_from_sysfs() {
        let root = tempfile::tempdir().unwrap();
        let disk = fixture_namespace(root.path(), 4096);
        let dev = root.path().join("dev");

        let t = NamespaceTarget::from_sysfs(&disk.join("nvme0n1p3"), &dev).unwrap();
        assert_eq!(t.char_dev, dev.join("ng0n1"));
        assert_eq!(t.nsid, 1);
        assert_eq!(t.lba_bytes(), 4096);
        assert_eq!(t.max_transfer_bytes(), 512 * 1024);
        assert_eq!(t.partition_start, 2048 * 512);
        assert_eq!(t.partition_len, 1048576 * 512);

        // The whole-namespace disk: no shift, the full namespace length.
        let whole = NamespaceTarget::from_sysfs(&disk, &dev).unwrap();
        assert_eq!(whole.partition_start, 0);
        assert_eq!(whole.partition_len, (1u64 << 21) * 512);
    }

    #[test]
    fn missing_or_inconsistent_geometry_refuses_passthrough() {
        let root = tempfile::tempdir().unwrap();
        let dev = root.path().join("dev");

        // No logical block size published: refuse rather than assume 512.
        let disk = fixture_namespace(root.path(), 4096);
        std::fs::remove_file(disk.join("queue/logical_block_size")).unwrap();
        assert!(NamespaceTarget::from_sysfs(&disk, &dev).is_none());

        // A partition not aligned to a 4096-byte LBA cannot be addressed.
        let root = tempfile::tempdir().unwrap();
        let disk = fixture_namespace(root.path(), 4096);
        write(&disk.join("nvme0n1p3/start"), "2049\n");
        assert!(NamespaceTarget::from_sysfs(&disk.join("nvme0n1p3"), &dev).is_none());

        // A partition reaching past the namespace is inconsistent.
        let root = tempfile::tempdir().unwrap();
        let disk = fixture_namespace(root.path(), 512);
        write(&disk.join("nvme0n1p3/size"), &format!("{}\n", 1u64 << 21));
        assert!(NamespaceTarget::from_sysfs(&disk.join("nvme0n1p3"), &dev).is_none());
    }

    #[test]
    fn only_namespace_disks_map_to_generic_char_devices() {
        assert_eq!(generic_char_name("nvme0n1").as_deref(), Some("ng0n1"));
        assert_eq!(generic_char_name("nvme12n3").as_deref(), Some("ng12n3"));
        // Hidden per-path disk, device mapper, and malformed names.
        for name in ["nvme0c1n1", "dm-0", "nvme0n", "nvmen1", "sda", "nvme0n1p3"] {
            assert_eq!(generic_char_name(name), None, "{name}");
        }
    }

    #[test]
    fn fiemap_maps_a_real_file() {
        // FIEMAP works on regular files on ext4/xfs; tmpfs and overlayfs
        // may not, so a failure here is a skip, not a test failure.
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let data = vec![0xABu8; 256 * 1024];
        tmp.write_all(&data).unwrap();
        tmp.as_file().sync_all().unwrap();

        match fiemap(tmp.as_file().as_raw_fd(), data.len() as u64) {
            Ok(extents) => {
                let map = ExtentMap::place(extents, 0, u64::MAX).unwrap();
                assert!(map.extent_count() >= 1, "a 256 KiB file has ≥1 extent");
                assert!(map.resolve(0, 4096).is_some());
            }
            Err(e) => eprintln!("FIEMAP unavailable on this fs ({e}); skipping"),
        }
    }

    #[test]
    fn reader_refuses_a_path_that_is_not_the_namespace_device() {
        let root = tempfile::tempdir().unwrap();
        let disk = fixture_namespace(root.path(), 512);
        let dev = root.path().join("dev");
        let target = NamespaceTarget::from_sysfs(&disk, &dev).unwrap();

        // Missing node.
        assert!(NvmeReader::open(&target).is_err());
        // A regular file where the char device should be.
        write(&dev.join("ng0n1"), "not a device");
        let err = NvmeReader::open(&target).err().unwrap();
        assert!(err.to_string().contains("not a character device"), "{err}");
    }
}
