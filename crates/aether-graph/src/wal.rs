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
//! Each record (24 bytes):
//!
//! ```text
//! [0..8 ] epoch    u64 LE  — `EpochClock` value at the time of the insert
//! [8..12] src      u32 LE
//! [12..16] dst     u32 LE
//! [16..20] _pad    u32 LE  (0; reserved for record type when we add deletes)
//! [20..24] crc32   u32 LE  — CRC32 of bytes [0..20] (IEEE polynomial)
//! ```
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
//! - **Concurrent writers**: a single `WalWriter` is single-writer by
//!   construction — the surrounding [`DynamicGraph`] already enforces
//!   single-writer with the `Writer` guard.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub(crate) const MAGIC: [u8; 8] = *b"AGWAL\0\0\0";
pub(crate) const VERSION: u32 = 1;
pub(crate) const HEADER_LEN: u64 = 16;
pub(crate) const RECORD_LEN: usize = 24;

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
    pub epoch: u64,
    pub src: u32,
    pub dst: u32,
}

/// Append-only log writer. Single-threaded by construction — never
/// `Send + Sync` from multiple writers concurrently.
#[derive(Debug)]
pub struct WalWriter {
    inner: BufWriter<File>,
    /// Bytes appended since the last `sync()`. Used by the
    /// [`DynamicGraph`] integration to skip no-op fsyncs.
    pending: u64,
}

impl WalWriter {
    /// Open an existing WAL or create one at `path`. The header is
    /// written and fsynced before this returns, so an interrupted
    /// open never leaves a half-initialized file.
    pub fn create_or_open(path: impl AsRef<Path>) -> Result<Self, WalError> {
        let path = path.as_ref();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;

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
            inner: BufWriter::with_capacity(64 * 1024, file),
            pending: 0,
        })
    }

    /// Append one edge record. The bytes land in the BufWriter; a later
    /// [`sync`](Self::sync) flushes and fsyncs.
    pub fn append_edge(&mut self, rec: EdgeRecord) -> Result<(), WalError> {
        let mut buf = [0u8; RECORD_LEN];
        buf[0..8].copy_from_slice(&rec.epoch.to_le_bytes());
        buf[8..12].copy_from_slice(&rec.src.to_le_bytes());
        buf[12..16].copy_from_slice(&rec.dst.to_le_bytes());
        // buf[16..20] left zero (record-type tag for future versions).
        let crc = crc32fast::hash(&buf[..20]);
        buf[20..24].copy_from_slice(&crc.to_le_bytes());

        self.inner.write_all(&buf)?;
        self.pending += RECORD_LEN as u64;
        Ok(())
    }

    /// Flush the BufWriter to the file and fsync — every record durably
    /// written before this call survives a crash. No-op if nothing has
    /// been appended since the last sync.
    pub fn sync(&mut self) -> Result<(), WalError> {
        if self.pending == 0 {
            return Ok(());
        }
        self.inner.flush()?;
        self.inner.get_ref().sync_data()?;
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

/// Replay every record in the WAL at `path`, invoking `apply` for each.
/// Stops cleanly at end-of-file or at the first torn record (kernel
/// crashed mid-write), whether the tear is a short tail (fewer than
/// `RECORD_LEN` trailing bytes) or a full-size record with a CRC
/// mismatch. Returns the number of records successfully applied and, if
/// truncation is needed, the byte offset to truncate to.
///
/// A missing file or a zero-byte file is treated as "no records yet" —
/// callers about to create a fresh WAL get an empty outcome rather than an
/// error. A file shorter than the header is a header torn during initial
/// creation; it holds no records and surfaces as `truncate_to: Some(0)` so
/// the caller can clear the partial header before reopening.
pub fn replay<F>(path: impl AsRef<Path>, mut apply: F) -> Result<ReplayOutcome, WalError>
where
    F: FnMut(EdgeRecord),
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
    let mut reader = BufReader::new(file);

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
        let crc_recorded = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        let crc_computed = crc32fast::hash(&buf[..20]);
        if crc_recorded != crc_computed {
            // Partial or torn write detected. Stop replay; callers
            // truncate the file to `offset` to drop the corrupt tail.
            return Ok(ReplayOutcome {
                applied,
                truncate_to: Some(offset),
            });
        }
        let rec = EdgeRecord {
            epoch: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            src: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
            dst: u32::from_le_bytes(buf[12..16].try_into().unwrap()),
        };
        apply(rec);
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
        for (epoch, src, dst) in [(1, 0u32, 1u32), (1, 2, 3), (2, 4, 5)] {
            w.append_edge(EdgeRecord { epoch, src, dst }).unwrap();
        }
        w.sync().unwrap();
        drop(w);

        let mut got = Vec::new();
        let out = replay(tmp.path(), |r| got.push(r)).unwrap();
        assert_eq!(out.applied, 3);
        assert!(out.truncate_to.is_none());
        assert_eq!(
            got,
            vec![
                EdgeRecord {
                    epoch: 1,
                    src: 0,
                    dst: 1
                },
                EdgeRecord {
                    epoch: 1,
                    src: 2,
                    dst: 3
                },
                EdgeRecord {
                    epoch: 2,
                    src: 4,
                    dst: 5
                },
            ]
        );
    }

    #[test]
    fn reopen_preserves_existing_records() {
        let tmp = tmp_wal();
        {
            let mut w = WalWriter::create_or_open(tmp.path()).unwrap();
            w.append_edge(EdgeRecord {
                epoch: 1,
                src: 7,
                dst: 8,
            })
            .unwrap();
            w.sync().unwrap();
        }
        {
            let mut w = WalWriter::create_or_open(tmp.path()).unwrap();
            w.append_edge(EdgeRecord {
                epoch: 2,
                src: 9,
                dst: 10,
            })
            .unwrap();
            w.sync().unwrap();
        }
        let mut got = Vec::new();
        replay(tmp.path(), |r| got.push(r)).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].src, 7);
        assert_eq!(got[1].src, 9);
    }

    #[test]
    fn corrupt_tail_detected_via_crc() {
        let tmp = tmp_wal();
        {
            let mut w = WalWriter::create_or_open(tmp.path()).unwrap();
            w.append_edge(EdgeRecord {
                epoch: 1,
                src: 0,
                dst: 1,
            })
            .unwrap();
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
        let out = replay(tmp.path(), |r| got.push(r)).unwrap();
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
            w.append_edge(EdgeRecord {
                epoch: 1,
                src: 0,
                dst: 1,
            })
            .unwrap();
            w.sync().unwrap();
        }
        // Simulate a torn write that flushed only part of a record.
        {
            let mut f = OpenOptions::new().append(true).open(tmp.path()).unwrap();
            f.write_all(&[0xAA; 10]).unwrap();
            f.sync_data().unwrap();
        }
        let mut got = Vec::new();
        let out = replay(tmp.path(), |r| got.push(r)).unwrap();
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
            w.append_edge(EdgeRecord {
                epoch: 1,
                src: 0,
                dst: 1,
            })
            .unwrap();
            w.sync().unwrap();
        }
        let bytes = std::fs::read(tmp.path()).unwrap();
        assert_eq!(&bytes[..8], &MAGIC, "header rewritten");
        assert_eq!(bytes.len(), HEADER_LEN as usize + RECORD_LEN);

        let mut got = Vec::new();
        let out = replay(tmp.path(), |r| got.push(r)).unwrap();
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
        let out = replay(tmp.path(), |r| got.push(r)).unwrap();
        assert!(got.is_empty(), "a torn header carries no records");
        assert_eq!(
            out.truncate_to,
            Some(0),
            "partial header must be cleared before reopening"
        );
    }

    #[test]
    fn empty_sync_is_noop() {
        let tmp = tmp_wal();
        let mut w = WalWriter::create_or_open(tmp.path()).unwrap();
        assert_eq!(w.pending_bytes(), 0);
        w.sync().unwrap();
        assert_eq!(w.pending_bytes(), 0);
    }
}
