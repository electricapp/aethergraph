//! Write-ahead log for [`DynamicGraph`] crash durability.
//!
//! Append-only log of every edge insertion. On startup, [`replay`] walks the
//! log and rebuilds the graph in memory. The log is the source of truth;
//! anything not in the log is lost on crash.
//!
//! # Durability contract
//!
//! - `WalWriter::append_edge` writes (buffered) but does not fsync.
//! - `WalWriter::sync` calls `fdatasync(2)` so all preceding edges survive a
//!   process / kernel / power crash.
//! - [`DynamicGraph`] integration: each clean `Writer::drop` calls `sync`
//!   exactly once. Crash mid-writer-guard loses every insert in that guard;
//!   crash after a clean drop loses zero inserts.
//! - A panicked writer guard never commits: its drop path calls
//!   [`WalWriter::discard_pending`], dropping the still-buffered records
//!   instead of flushing them. This is best-effort discard — records the
//!   buffer already spilled to the OS (a guard that appends more than the
//!   buffer holds) are out of reach and will replay as a valid prefix.
//!
//! # Record format
//!
//! Header (16 bytes, written once at file creation):
//!
//! ```text
//! [0..8 ] magic    = b"AGWAL\0\0\0"
//! [8..12] version  = u32 LE (currently 1)
//! [12..16] reserved = 0
//! ```
//!
//! Each record (12 bytes):
//!
//! ```text
//! [0..4 ] src      u32 LE
//! [4..8 ] dst      u32 LE
//! [8..12] crc32    u32 LE  — CRC32 of bytes [0..8] (IEEE polynomial)
//! ```
//!
//! The payload is exactly the edge: an epoch stamp would be identical for
//! every record in a writer guard and is ignored on replay, so carrying
//! it per record only inflated log I/O. A future record version can
//! reintroduce per-record metadata (epoch pinning, deletes) behind the
//! header's version field.
//!
//! Fixed-size records keep replay branchless. Torn writes are detected
//! two ways: a tail shorter than one record is caught by the fixed
//! stride, and a full-size but partially-flushed record is caught by its
//! CRC. Both surface as [`ReplayOutcome::truncate_to`] so recovery
//! truncates back to the last clean boundary before appending.
//!
//! # What is NOT in scope
//!
//! - **Checkpoints / log truncation**: the log grows unbounded today. A
//!   future checkpoint API will let callers snapshot the in-memory state
//!   and truncate everything below the checkpoint epoch.
//! - **Atomic batch semantics**: there is no "transaction begin / commit".
//!   A crash mid-batch can recover a prefix.
//! - **Concurrent writers**: a `WalWriter` holds an exclusive advisory
//!   lock on the file for its lifetime, so a second open of the same path
//!   fails with [`WalError::Locked`] instead of interleaving appends. The
//!   surrounding [`DynamicGraph`] additionally enforces single-writer
//!   in-process with the `Writer` guard.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub(crate) const MAGIC: [u8; 8] = *b"AGWAL\0\0\0";
pub(crate) const VERSION: u32 = 1;
pub(crate) const HEADER_LEN: u64 = 16;
pub(crate) const RECORD_LEN: usize = 12;

/// Writer/reader buffer size. Sized so a full commit interval's records
/// (65,536 edges from the ingest drivers) reach the file in one `write(2)`
/// and replay pulls the log in 1 MiB reads.
const BUF_CAPACITY: usize = 1 << 20;

/// Errors surfaced by WAL writes / replay. Corruption is not an error:
/// a torn or corrupt tail is reported through
/// [`ReplayOutcome::truncate_to`] so callers can truncate-and-recover.
#[derive(Debug)]
pub enum WalError {
    /// Underlying I/O failure (open, read, write, fsync).
    Io(io::Error),
    /// File exists but the magic header doesn't match — it isn't ours, or
    /// it's been overwritten by another process.
    BadMagic { found: [u8; 8] },
    /// The WAL file is exclusively locked by another `WalWriter` (in this
    /// process or another). Two writers appending to one log would
    /// silently overwrite each other, so only one may hold it open.
    Locked,
    /// Header says a version we don't know how to read.
    UnknownVersion(u32),
    /// A replayed record references a vertex at or beyond the
    /// `num_vertices` the graph was opened with — the WAL was written
    /// against a larger graph than the one being recovered into.
    RecordOutOfRange {
        src: u32,
        dst: u32,
        num_vertices: u64,
    },
    /// The arena filled up before every record was replayed. Reopen
    /// with a larger arena.
    ReplayArenaFull,
}

impl std::fmt::Display for WalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "WAL io error: {e}"),
            Self::BadMagic { found } => write!(f, "WAL bad magic: {found:?}"),
            Self::Locked => write!(f, "WAL file is locked by another writer"),
            Self::UnknownVersion(v) => write!(f, "WAL version {v} not supported"),
            Self::RecordOutOfRange {
                src,
                dst,
                num_vertices,
            } => write!(
                f,
                "WAL record ({src}, {dst}) exceeds num_vertices {num_vertices}; \
                 the log was written against a larger graph"
            ),
            Self::ReplayArenaFull => {
                write!(
                    f,
                    "arena filled during WAL replay; reopen with a larger arena"
                )
            }
        }
    }
}

impl std::error::Error for WalError {}

impl From<io::Error> for WalError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// One edge insertion event. The replay callback receives one of these per
/// successfully-deserialized record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeRecord {
    pub src: u32,
    pub dst: u32,
}

/// Append-only log writer. Single-threaded by construction — never
/// `Send + Sync` from multiple writers concurrently.
#[derive(Debug)]
pub struct WalWriter {
    file: File,
    /// Records staged in userspace, flushed to the file at `BUF_CAPACITY`
    /// or by `sync`. Owned outright rather than held in a `BufWriter` so
    /// the committed prefix can be dropped in place: the ring path writes
    /// straight out of this buffer, and `BufWriter` exposes no way to
    /// discard bytes it has already handed to the kernel.
    buf: Vec<u8>,
    /// Bytes appended since the last `sync()`. Used by the
    /// [`DynamicGraph`] integration to skip no-op fsyncs.
    pending: u64,
    /// Lazily-built ring for the linked write→fdatasync group commit.
    /// `None` before first use or after ring setup failed (the portable
    /// two-syscall path serves those cases).
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    ring: Option<UringCommit>,
    /// Set once ring construction has been attempted, so a kernel
    /// without io_uring is probed exactly once.
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    ring_probed: bool,
}

/// Wrapper existing solely because `io_uring::IoUring` has no `Debug`.
#[cfg(all(target_os = "linux", feature = "io-uring"))]
struct UringCommit {
    ring: io_uring::IoUring,
}

#[cfg(all(target_os = "linux", feature = "io-uring"))]
impl std::fmt::Debug for UringCommit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("UringCommit { .. }")
    }
}

impl WalWriter {
    /// Open an existing WAL or create one at `path`. The header is
    /// written and fsynced before this returns, so an interrupted
    /// open never leaves a half-initialized file.
    ///
    /// Takes an exclusive advisory lock on the file, held until the
    /// writer (and its `File`) is dropped. A second `create_or_open` on
    /// the same path fails with [`WalError::Locked`] while the lock is
    /// held.
    pub fn create_or_open(path: impl AsRef<Path>) -> Result<Self, WalError> {
        let path = path.as_ref();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;

        // Lock before reading or writing anything: a concurrent holder
        // may be mid-append, and our header write below would clobber it.
        file.try_lock().map_err(|e| match e {
            TryLockError::WouldBlock => WalError::Locked,
            TryLockError::Error(e) => WalError::Io(e),
        })?;

        let len = file.metadata()?.len();
        if len < HEADER_LEN {
            // Brand-new file (len == 0), or a header torn by a crash during
            // the very first creation (0 < len < HEADER_LEN). A partial
            // header carries no records — every append lands after a fully
            // written, fsynced header — so clearing it back to empty is
            // lossless. Truncate first so stray partial bytes don't sit
            // ahead of the header we write.
            if len != 0 {
                file.set_len(0)?;
                file.seek(SeekFrom::Start(0))?;
            }
            // Write the header and durably persist it before anyone appends
            // a record.
            let mut header = [0u8; HEADER_LEN as usize];
            header[..8].copy_from_slice(&MAGIC);
            header[8..12].copy_from_slice(&VERSION.to_le_bytes());
            file.write_all(&header)?;
            file.sync_data()?;
            // The new directory entry must also be durable: without an
            // fsync of the parent directory, a power loss can lose the
            // whole file even though every later sync() succeeded.
            #[cfg(unix)]
            {
                let parent = match path.parent() {
                    Some(p) if !p.as_os_str().is_empty() => p,
                    _ => Path::new("."),
                };
                File::open(parent)?.sync_all()?;
            }
        } else {
            file.seek(SeekFrom::Start(0))?;
            let mut header = [0u8; HEADER_LEN as usize];
            file.read_exact(&mut header)?;
            let mut magic = [0u8; 8];
            magic.copy_from_slice(&header[..8]);
            if magic != MAGIC {
                return Err(WalError::BadMagic { found: magic });
            }
            let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
            if version != VERSION {
                return Err(WalError::UnknownVersion(version));
            }
        }

        // Seek to end for append-only writes.
        file.seek(SeekFrom::End(0))?;
        Ok(Self {
            file,
            buf: Vec::with_capacity(BUF_CAPACITY),
            pending: 0,
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            ring: None,
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            ring_probed: false,
        })
    }

    /// Append one edge record. The bytes stage in the in-process buffer; a
    /// later [`sync`](Self::sync) writes and fsyncs them.
    pub fn append_edge(&mut self, rec: EdgeRecord) -> Result<(), WalError> {
        let mut buf = [0u8; RECORD_LEN];
        buf[0..4].copy_from_slice(&rec.src.to_le_bytes());
        buf[4..8].copy_from_slice(&rec.dst.to_le_bytes());
        let crc = crc32fast::hash(&buf[..8]);
        buf[8..12].copy_from_slice(&crc.to_le_bytes());

        if self.buf.len() + RECORD_LEN > BUF_CAPACITY {
            self.flush_buf()?;
        }
        self.buf.extend_from_slice(&buf);
        self.pending += RECORD_LEN as u64;
        Ok(())
    }

    /// Hand the staged bytes to the kernel and empty the buffer, without a
    /// durability barrier. `pending` is deliberately left alone: it tracks
    /// what is unsynced, and these bytes are in the page cache, not on the
    /// medium.
    fn flush_buf(&mut self) -> Result<(), WalError> {
        if !self.buf.is_empty() {
            self.file.write_all(&self.buf)?;
            self.buf.clear();
        }
        Ok(())
    }

    /// Write the staged bytes and fsync — every record durably written
    /// before this call survives a crash. No-op if nothing has been
    /// appended since the last sync.
    ///
    /// With the `io-uring` feature on Linux, the staged bytes and the
    /// fdatasync go to the kernel as one linked SQE chain — a single
    /// `io_uring_enter` replaces the write + fdatasync syscall pair, and
    /// the link makes the kernel enforce write-before-sync ordering.
    pub fn sync(&mut self) -> Result<(), WalError> {
        if self.pending == 0 {
            return Ok(());
        }
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        if self.sync_uring()? {
            return Ok(());
        }
        self.flush_buf()?;
        self.file.sync_data()?;
        self.pending = 0;
        Ok(())
    }

    /// Group commit over io_uring. Returns `Ok(true)` when the commit
    /// completed here, `Ok(false)` to route to the portable path (ring
    /// unavailable), `Err` on a real I/O failure.
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    fn sync_uring(&mut self) -> Result<bool, WalError> {
        use io_uring::{IoUring, opcode, types};
        use std::os::unix::io::AsRawFd;

        if !self.ring_probed {
            self.ring_probed = true;
            match IoUring::new(4) {
                // The write SQE below submits offset -1, which only means
                // "use and advance the file cursor" when the kernel reports
                // IORING_FEAT_RW_CUR_POS. Without it that offset is taken
                // literally and the record lands at 2^64-1, so the absence
                // of the feature has to disable the ring, not just the
                // absence of io_uring.
                Ok(ring) if ring.params().is_feature_rw_cur_pos() => {
                    self.ring = Some(UringCommit { ring })
                }
                Ok(_) => {
                    tracing::debug!(
                        "io_uring lacks IORING_FEAT_RW_CUR_POS; WAL uses write+fdatasync"
                    );
                }
                Err(e) => {
                    tracing::debug!(error = %e, "io_uring unavailable; WAL uses write+fdatasync");
                }
            }
        }
        if self.ring.is_none() {
            return Ok(false);
        }

        let len = self.buf.len();
        if len == 0 {
            // Capacity flushes already handed every record to the kernel;
            // only the durability barrier is outstanding, and a bare
            // fdatasync beats a two-SQE chain carrying a zero-length write.
            self.file.sync_data()?;
            self.pending = 0;
            return Ok(true);
        }
        // Raw pointer rather than a slice borrow: the SQE build below needs
        // `self.ring` mutably while the kernel reads these bytes.
        let buf_ptr = self.buf.as_ptr();
        let fd = types::Fd(self.file.as_raw_fd());
        let commit = self
            .ring
            .as_mut()
            .expect("ring presence checked immediately above");

        // Offset -1: append at the file's own cursor and advance it,
        // exactly like the write(2) the portable path would issue.
        let write_sqe = opcode::Write::new(fd, buf_ptr, len as u32)
            .offset(u64::MAX)
            .build()
            .flags(io_uring::squeue::Flags::IO_LINK)
            .user_data(1);
        let fsync_sqe = opcode::Fsync::new(fd)
            .flags(types::FsyncFlags::DATASYNC)
            .build()
            .user_data(2);

        {
            let mut sq = commit.ring.submission();
            // SAFETY: `buf_ptr` addresses `self.buf`, which nothing touches
            // until both CQEs are reaped below, and `submit_and_wait` does
            // not return until the kernel is done reading it.
            let pushed = unsafe { sq.push(&write_sqe) };
            pushed.expect("empty 4-entry ring accepts the write SQE");
            // SAFETY: the fsync SQE references only the fd, which stays
            // open for the life of `self.file`.
            let pushed = unsafe { sq.push(&fsync_sqe) };
            pushed.expect("4-entry ring accepts the linked fsync SQE");
        }
        if let Err(e) = commit.ring.submit_and_wait(2) {
            // Both SQEs may already be in flight, and their CQEs carry the
            // same user_data every commit does, so a later sync reaping
            // this ring could read this commit's results as its own.
            // Retire the ring; subsequent syncs take the portable path.
            self.ring = None;
            return Err(WalError::Io(e));
        }

        let mut written: Option<i32> = None;
        let mut synced: Option<i32> = None;
        for cqe in commit.ring.completion() {
            match cqe.user_data() {
                1 => written = Some(cqe.result()),
                2 => synced = Some(cqe.result()),
                _ => {}
            }
        }
        let (written, synced) = (
            written.expect("write CQE present after submit_and_wait(2)"),
            synced.expect("fsync CQE present after submit_and_wait(2)"),
        );

        if written < 0 {
            return Err(WalError::Io(io::Error::from_raw_os_error(-written)));
        }
        let written = written as usize;
        if written < len {
            // Rare (disk full mid-write): the linked fsync covered only
            // the short prefix. Finish the tail through the portable
            // path so its own fdatasync provides the guarantee.
            self.buf.drain(..written);
            self.flush_buf()?;
            self.file.sync_data()?;
            self.pending = 0;
            return Ok(true);
        }
        // A canceled link (-ECANCELED) cannot happen here: the write
        // completed fully, so the chain proceeded to the fsync.
        if synced < 0 {
            return Err(WalError::Io(io::Error::from_raw_os_error(-synced)));
        }

        self.buf.clear();
        self.pending = 0;
        Ok(true)
    }

    /// Discard every buffered record that has not yet been flushed to the
    /// OS, without writing it. Called from the panicked-writer drop path:
    /// a guard that never committed must not have its records persisted
    /// by a later flush. Bytes already flushed to the file are untouched
    /// — the discard reaches only the in-process buffer.
    pub fn discard_pending(&mut self) -> Result<(), WalError> {
        // The staging buffer holds exactly the records the kernel has not
        // seen; anything a capacity flush already wrote stays in the file.
        self.buf.clear();
        self.pending = 0;
        Ok(())
    }

    /// Bytes appended since the last successful `sync()`. Mainly for
    /// telemetry / tests.
    pub fn pending_bytes(&self) -> u64 {
        self.pending
    }
}

impl Drop for WalWriter {
    fn drop(&mut self) {
        // Best-effort flush on drop. If sync fails (disk full, etc.) the
        // error is logged via tracing — the panic path would re-poison an
        // already-panicking thread.
        if let Err(e) = self.sync() {
            tracing::error!(error = %e, "WAL sync on drop failed; recent edges may be lost");
        }
    }
}

/// Replay every record in the WAL at `path`, invoking `apply` for each;
/// an `Err` from `apply` aborts the walk and is returned as-is. Stops
/// cleanly at end-of-file or at the first torn record (kernel crashed
/// mid-write), whether the tear is a short tail (fewer than `RECORD_LEN`
/// trailing bytes) or a full-size record with a CRC mismatch. Returns the
/// number of records successfully applied and, if truncation is needed,
/// the byte offset to truncate to.
///
/// A missing file or a zero-byte file is treated as "no records yet" —
/// callers about to create a fresh WAL get an empty outcome rather than an
/// error. A file shorter than the header is a header torn during initial
/// creation; it holds no records and surfaces as `truncate_to: Some(0)` so
/// the caller can clear the partial header before reopening.
pub fn replay<F>(path: impl AsRef<Path>, mut apply: F) -> Result<ReplayOutcome, WalError>
where
    F: FnMut(EdgeRecord) -> Result<(), WalError>,
{
    let path = path.as_ref();
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Ok(ReplayOutcome {
                applied: 0,
                truncate_to: None,
            });
        }
        Err(e) => return Err(WalError::Io(e)),
    };
    let total_len = file.metadata()?.len();
    if total_len == 0 {
        return Ok(ReplayOutcome {
            applied: 0,
            truncate_to: None,
        });
    }
    // 1 MiB buffer: the default 8 KiB would issue a read(2) every ~680
    // records on a recovery path that must chew the whole log.
    let mut reader = BufReader::with_capacity(BUF_CAPACITY, file);

    // A header torn by a crash during initial creation (0 < len < HEADER_LEN,
    // since len == 0 returned above) holds no records. Report a truncate-to-0
    // so the caller clears the partial header before reopening; the writer's
    // create path then writes a fresh one.
    if total_len < HEADER_LEN {
        return Ok(ReplayOutcome {
            applied: 0,
            truncate_to: Some(0),
        });
    }

    let mut header = [0u8; HEADER_LEN as usize];
    reader.read_exact(&mut header)?;
    let mut magic = [0u8; 8];
    magic.copy_from_slice(&header[..8]);
    if magic != MAGIC {
        return Err(WalError::BadMagic { found: magic });
    }
    let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(WalError::UnknownVersion(version));
    }

    let mut applied: u64 = 0;
    let mut offset: u64 = HEADER_LEN;
    let mut buf = [0u8; RECORD_LEN];
    loop {
        match reader.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                // Clean end-of-file when the file ends exactly on a record
                // boundary. Anything shorter is a torn write: report the
                // boundary so callers truncate before appending — appends
                // on top of a partial record would land misaligned and be
                // discarded wholesale by the next replay's CRC check.
                if offset != total_len {
                    return Ok(ReplayOutcome {
                        applied,
                        truncate_to: Some(offset),
                    });
                }
                break;
            }
            Err(e) => return Err(WalError::Io(e)),
        }
        let crc_recorded = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let crc_computed = crc32fast::hash(&buf[..8]);
        if crc_recorded != crc_computed {
            // Partial or torn write detected. Stop replay; callers
            // truncate the file to `offset` to drop the corrupt tail.
            return Ok(ReplayOutcome {
                applied,
                truncate_to: Some(offset),
            });
        }
        let rec = EdgeRecord {
            src: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            dst: u32::from_le_bytes(buf[4..8].try_into().unwrap()),
        };
        apply(rec)?;
        applied += 1;
        offset += RECORD_LEN as u64;
    }

    Ok(ReplayOutcome {
        applied,
        truncate_to: None,
    })
}

/// Summary of a `replay()` call.
#[derive(Debug, Clone, Copy)]
pub struct ReplayOutcome {
    /// Number of records successfully delivered to the apply callback.
    pub applied: u64,
    /// If `Some`, the file ends in a corrupt/torn record; callers should
    /// truncate the file to this byte offset before opening a fresh
    /// `WalWriter` so future appends don't sit on top of garbage.
    pub truncate_to: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_wal() -> tempfile::NamedTempFile {
        tempfile::NamedTempFile::new().unwrap()
    }

    #[test]
    fn create_writes_header_and_fsyncs() {
        let tmp = tmp_wal();
        {
            let _w = WalWriter::create_or_open(tmp.path()).unwrap();
        }
        let bytes = std::fs::read(tmp.path()).unwrap();
        assert_eq!(&bytes[..8], &MAGIC);
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            VERSION
        );
    }

    #[test]
    fn append_replay_round_trip() {
        let tmp = tmp_wal();
        let mut w = WalWriter::create_or_open(tmp.path()).unwrap();
        for (src, dst) in [(0u32, 1u32), (2, 3), (4, 5)] {
            w.append_edge(EdgeRecord { src, dst }).unwrap();
        }
        w.sync().unwrap();
        drop(w);

        let mut got = Vec::new();
        let out = replay(tmp.path(), |r| {
            got.push(r);
            Ok(())
        })
        .unwrap();
        assert_eq!(out.applied, 3);
        assert!(out.truncate_to.is_none());
        assert_eq!(
            got,
            vec![
                EdgeRecord { src: 0, dst: 1 },
                EdgeRecord { src: 2, dst: 3 },
                EdgeRecord { src: 4, dst: 5 },
            ]
        );
    }

    #[test]
    fn reopen_preserves_existing_records() {
        let tmp = tmp_wal();
        {
            let mut w = WalWriter::create_or_open(tmp.path()).unwrap();
            w.append_edge(EdgeRecord { src: 7, dst: 8 }).unwrap();
            w.sync().unwrap();
        }
        {
            let mut w = WalWriter::create_or_open(tmp.path()).unwrap();
            w.append_edge(EdgeRecord { src: 9, dst: 10 }).unwrap();
            w.sync().unwrap();
        }
        let mut got = Vec::new();
        replay(tmp.path(), |r| {
            got.push(r);
            Ok(())
        })
        .unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].src, 7);
        assert_eq!(got[1].src, 9);
    }

    #[test]
    fn corrupt_tail_detected_via_crc() {
        let tmp = tmp_wal();
        {
            let mut w = WalWriter::create_or_open(tmp.path()).unwrap();
            w.append_edge(EdgeRecord { src: 0, dst: 1 }).unwrap();
            w.sync().unwrap();
        }
        // Simulate a torn write: append a "record" with the wrong CRC.
        // (16 bytes of payload then 4 bytes of zero CRC — definitely not
        // the real CRC32 of that payload.)
        {
            let mut f = OpenOptions::new().append(true).open(tmp.path()).unwrap();
            f.write_all(&[0xAA; RECORD_LEN]).unwrap();
            f.sync_data().unwrap();
        }
        let mut got = Vec::new();
        let out = replay(tmp.path(), |r| {
            got.push(r);
            Ok(())
        })
        .unwrap();
        assert_eq!(got.len(), 1, "first (good) record applied");
        assert_eq!(
            out.truncate_to,
            Some(HEADER_LEN + RECORD_LEN as u64),
            "truncate-to points at the start of the bad record"
        );
    }

    #[test]
    fn short_torn_tail_reports_truncation() {
        let tmp = tmp_wal();
        {
            let mut w = WalWriter::create_or_open(tmp.path()).unwrap();
            w.append_edge(EdgeRecord { src: 0, dst: 1 }).unwrap();
            w.sync().unwrap();
        }
        // Simulate a torn write that flushed only part of a record.
        {
            let mut f = OpenOptions::new().append(true).open(tmp.path()).unwrap();
            f.write_all(&[0xAA; 10]).unwrap();
            f.sync_data().unwrap();
        }
        let mut got = Vec::new();
        let out = replay(tmp.path(), |r| {
            got.push(r);
            Ok(())
        })
        .unwrap();
        assert_eq!(got.len(), 1, "good record applied");
        assert_eq!(
            out.truncate_to,
            Some(HEADER_LEN + RECORD_LEN as u64),
            "short tail must be reported for truncation, not ignored"
        );
    }

    #[test]
    fn bad_magic_rejected() {
        let tmp = tmp_wal();
        {
            let mut f = File::create(tmp.path()).unwrap();
            f.write_all(b"NOTAGWAL\0\0\0\0\0\0\0\0").unwrap();
        }
        match WalWriter::create_or_open(tmp.path()) {
            Err(WalError::BadMagic { .. }) => {}
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn torn_header_is_reinitialized_by_create_or_open() {
        // A crash during initial creation can leave a header shorter than
        // HEADER_LEN. It carries no records, so create_or_open clears it
        // and writes a fresh header rather than failing.
        let tmp = tmp_wal();
        {
            let mut f = File::create(tmp.path()).unwrap();
            f.write_all(&MAGIC[..4]).unwrap(); // half the magic, no version
        }
        {
            let mut w = WalWriter::create_or_open(tmp.path()).unwrap();
            w.append_edge(EdgeRecord { src: 0, dst: 1 }).unwrap();
            w.sync().unwrap();
        }
        let bytes = std::fs::read(tmp.path()).unwrap();
        assert_eq!(&bytes[..8], &MAGIC, "header rewritten");
        assert_eq!(bytes.len(), HEADER_LEN as usize + RECORD_LEN);

        let mut got = Vec::new();
        let out = replay(tmp.path(), |r| {
            got.push(r);
            Ok(())
        })
        .unwrap();
        assert_eq!(got.len(), 1);
        assert!(out.truncate_to.is_none());
    }

    #[test]
    fn torn_header_replay_reports_truncate_to_zero() {
        let tmp = tmp_wal();
        {
            let mut f = File::create(tmp.path()).unwrap();
            f.write_all(&MAGIC[..4]).unwrap();
        }
        let mut got = Vec::new();
        let out = replay(tmp.path(), |r| {
            got.push(r);
            Ok(())
        })
        .unwrap();
        assert!(got.is_empty(), "a torn header carries no records");
        assert_eq!(
            out.truncate_to,
            Some(0),
            "partial header must be cleared before reopening"
        );
    }

    #[test]
    fn second_open_fails_while_locked() {
        let tmp = tmp_wal();
        let w = WalWriter::create_or_open(tmp.path()).unwrap();
        match WalWriter::create_or_open(tmp.path()) {
            Err(WalError::Locked) => {}
            other => panic!("expected Locked, got {other:?}"),
        }
        // Dropping the writer releases the lock with its File.
        drop(w);
        WalWriter::create_or_open(tmp.path()).unwrap();
    }

    #[test]
    fn discard_pending_drops_unflushed_records() {
        let tmp = tmp_wal();
        let mut w = WalWriter::create_or_open(tmp.path()).unwrap();
        w.append_edge(EdgeRecord { src: 0, dst: 1 }).unwrap();
        w.sync().unwrap();
        // Buffered but never synced — must not survive the discard.
        w.append_edge(EdgeRecord { src: 2, dst: 3 }).unwrap();
        assert_eq!(w.pending_bytes(), RECORD_LEN as u64);
        w.discard_pending().unwrap();
        assert_eq!(w.pending_bytes(), 0);
        // Appends after a discard land on the last flushed boundary.
        w.append_edge(EdgeRecord { src: 4, dst: 5 }).unwrap();
        w.sync().unwrap();
        drop(w);

        let mut got = Vec::new();
        let out = replay(tmp.path(), |r| {
            got.push(r);
            Ok(())
        })
        .unwrap();
        assert_eq!(out.applied, 2);
        assert!(out.truncate_to.is_none());
        assert_eq!(got[0].src, 0);
        assert_eq!(got[1].src, 4);
    }

    #[test]
    fn empty_sync_is_noop() {
        let tmp = tmp_wal();
        let mut w = WalWriter::create_or_open(tmp.path()).unwrap();
        assert_eq!(w.pending_bytes(), 0);
        w.sync().unwrap();
        assert_eq!(w.pending_bytes(), 0);
    }

    /// Appending past `BUF_CAPACITY` spills to the file mid-guard. Every
    /// record must still replay exactly once and in order: the spill and
    /// the closing sync each write a disjoint slice of the buffer, so an
    /// off-by-one in either would duplicate or drop records at the seam.
    #[test]
    fn records_spanning_a_capacity_flush_replay_exactly_once() {
        let tmp = tmp_wal();
        let count = (BUF_CAPACITY / RECORD_LEN) as u32 + 1000;
        {
            let mut w = WalWriter::create_or_open(tmp.path()).unwrap();
            for i in 0..count {
                w.append_edge(EdgeRecord { src: i, dst: i + 1 }).unwrap();
            }
            // The spill is not a durability point: everything appended is
            // still counted as unsynced until sync() lands the barrier.
            assert_eq!(w.pending_bytes(), u64::from(count) * RECORD_LEN as u64);
            w.sync().unwrap();
            assert_eq!(w.pending_bytes(), 0);
        }

        let mut got = Vec::new();
        let out = replay(tmp.path(), |r| {
            got.push(r);
            Ok(())
        })
        .unwrap();
        assert!(out.truncate_to.is_none(), "no torn tail expected");
        assert_eq!(out.applied, u64::from(count));
        assert_eq!(got.len(), count as usize);
        for (i, rec) in got.iter().enumerate() {
            let i = i as u32;
            assert_eq!(*rec, EdgeRecord { src: i, dst: i + 1 }, "record {i}");
        }
    }

    /// A discard after the buffer has already spilled keeps the spilled
    /// prefix — it is in the file and out of reach — and drops only what
    /// is still staged. The durability contract documents this as a
    /// best-effort discard, so the replayed log must be a clean prefix.
    #[test]
    fn discard_after_a_capacity_flush_keeps_the_spilled_prefix() {
        let tmp = tmp_wal();
        let count = (BUF_CAPACITY / RECORD_LEN) as u32 + 1000;
        {
            let mut w = WalWriter::create_or_open(tmp.path()).unwrap();
            for i in 0..count {
                w.append_edge(EdgeRecord { src: i, dst: i + 1 }).unwrap();
            }
            w.discard_pending().unwrap();
            assert_eq!(w.pending_bytes(), 0);
        }

        let mut got = Vec::new();
        let out = replay(tmp.path(), |r| {
            got.push(r);
            Ok(())
        })
        .unwrap();
        assert!(
            out.truncate_to.is_none(),
            "a spilled prefix is record-aligned, not torn"
        );
        assert!(
            !got.is_empty() && got.len() < count as usize,
            "expected a strict prefix, got {} of {count}",
            got.len()
        );
        for (i, rec) in got.iter().enumerate() {
            let i = i as u32;
            assert_eq!(*rec, EdgeRecord { src: i, dst: i + 1 }, "record {i}");
        }
    }
}
