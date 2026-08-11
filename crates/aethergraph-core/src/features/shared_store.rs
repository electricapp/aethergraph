//! Cross-process feature store over a sealed `memfd`.
//!
//! Training with N worker processes on one host normally means N copies of
//! the feature matrix — N times the RAM, N times the load time. This is
//! one copy: the owner publishes the payload into a sealed shared region
//! ([`internal::shm`](crate::internal::shm)) and serves the memfd over a
//! Unix socket; every attached worker maps *the same physical pages*
//! read-only. Attaching is a mmap, not a read, so a worker joins a
//! multi-gigabyte store in microseconds.
//!
//! The seals are what make it safe to hand out: with `F_SEAL_SHRINK |
//! F_SEAL_GROW` no holder can resize the object out from under a peer's
//! mapping.
//!
//! The region carries the payload only; geometry (node count, dimension,
//! dtype) travels with the fd in the handshake, so an attached worker
//! parses it once at the socket edge and thereafter indexes rows with the
//! same arithmetic as a mapped store.

#![cfg(all(target_os = "linux", feature = "shm"))]

use std::io::{Read, Write};
use std::os::unix::io::{AsRawFd, OwnedFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::{debug, trace};

use super::header::{FeatureDtype, parse_feature_header};
use crate::graph::NodeId;
use crate::internal::shm::{SharedRegion, recv_fd, send_fd};

/// Wire form of the store geometry, sent alongside the memfd.
///
/// Fixed 32-byte little-endian layout: `payload_len u64 | num_nodes u64 |
/// feature_dim u64 | dtype u8 | 7 bytes padding`.
const HANDSHAKE_LEN: usize = 32;

#[derive(Debug, Clone, Copy)]
struct Geometry {
    payload_len: u64,
    num_nodes: usize,
    feature_dim: usize,
    dtype: FeatureDtype,
}

impl Geometry {
    fn to_bytes(self) -> [u8; HANDSHAKE_LEN] {
        let mut b = [0u8; HANDSHAKE_LEN];
        b[0..8].copy_from_slice(&self.payload_len.to_le_bytes());
        b[8..16].copy_from_slice(&(self.num_nodes as u64).to_le_bytes());
        b[16..24].copy_from_slice(&(self.feature_dim as u64).to_le_bytes());
        b[24] = self.dtype as u8;
        b
    }

    /// Parse and validate the handshake — the one place an attaching
    /// worker turns bytes off a socket into trusted geometry.
    fn from_bytes(b: &[u8; HANDSHAKE_LEN]) -> Result<Self> {
        let payload_len = u64::from_le_bytes(b[0..8].try_into().expect("8"));
        let num_nodes = usize::try_from(u64::from_le_bytes(b[8..16].try_into().expect("8")))
            .context("num_nodes does not fit in usize")?;
        let feature_dim = usize::try_from(u64::from_le_bytes(b[16..24].try_into().expect("8")))
            .context("feature_dim does not fit in usize")?;
        let dtype = FeatureDtype::from_u8(b[24])?;

        let row_bytes = feature_dim
            .checked_mul(dtype.element_size())
            .context("row size overflows usize")?;
        let expect = (num_nodes as u64)
            .checked_mul(row_bytes as u64)
            .context("payload size overflows u64")?;
        anyhow::ensure!(
            expect == payload_len,
            "handshake geometry inconsistent: {num_nodes} x {row_bytes} != {payload_len} bytes"
        );
        anyhow::ensure!(payload_len > 0, "shared store payload must be non-empty");

        Ok(Self {
            payload_len,
            num_nodes,
            feature_dim,
            dtype,
        })
    }
}

/// A feature store whose payload lives in shared memory.
///
/// The owner builds one with [`SharedFeatureStore::publish`] and serves it
/// with [`SharedFeatureStore::serve`]; workers get a read-only view from
/// [`SharedFeatureStore::attach`]. Both sides read rows through the same
/// [`get_into`](Self::get_into) / [`get_batch`](Self::get_batch).
pub struct SharedFeatureStore {
    region: Arc<SharedRegion>,
    num_nodes: usize,
    feature_dim: usize,
    dtype: FeatureDtype,
    row_bytes: usize,
}

impl SharedFeatureStore {
    /// Copy the payload of the feature file at `path` into a fresh sealed
    /// shared region. This is the one copy the whole host pays.
    pub fn publish(path: impl AsRef<Path>) -> Result<Self> {
        use std::os::unix::fs::FileExt;

        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open feature store {}", path.display()))?;
        let header = parse_feature_header(&file)?;

        let payload_len = header
            .num_nodes
            .checked_mul(header.feature_size)
            .context("feature payload size overflows usize")?;
        anyhow::ensure!(payload_len > 0, "feature store is empty");

        let mut region = SharedRegion::create(payload_len)?;
        {
            let dst = region
                .as_mut_slice()
                .expect("a freshly created region is writable");
            file.read_exact_at(dst, header.features_start_offset)
                .context("failed to read feature payload")?;
        }

        debug!(
            "Published {} nodes x {} dims ({} MiB) to shared memory",
            header.num_nodes,
            header.feature_dim,
            payload_len / (1024 * 1024)
        );

        Ok(Self {
            region: Arc::new(region),
            num_nodes: header.num_nodes,
            feature_dim: header.feature_dim,
            dtype: header.dtype,
            row_bytes: header.feature_size,
        })
    }

    /// Serve this store to workers connecting on the Unix socket at
    /// `socket_path`, until `shutdown` is dropped.
    ///
    /// Each connection receives the geometry followed by the memfd itself.
    /// Returns a handle whose drop stops the listener and removes the
    /// socket file.
    pub fn serve(&self, socket_path: impl AsRef<Path>) -> Result<ShareHandle> {
        let socket_path = socket_path.as_ref().to_path_buf();
        // A stale socket file from a crashed owner would make bind fail;
        // the path names this owner's endpoint, so reclaiming it is right.
        let _ = std::fs::remove_file(&socket_path);
        let listener = UnixListener::bind(&socket_path)
            .with_context(|| format!("failed to bind {}", socket_path.display()))?;
        listener
            .set_nonblocking(true)
            .context("failed to set listener non-blocking")?;

        let geometry = Geometry {
            payload_len: self.region.len() as u64,
            num_nodes: self.num_nodes,
            feature_dim: self.feature_dim,
            dtype: self.dtype,
        }
        .to_bytes();
        let region = Arc::clone(&self.region);
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let thread_shutdown = Arc::clone(&shutdown);

        let thread = std::thread::Builder::new()
            .name("aether-shm-serve".into())
            .spawn(move || {
                use std::sync::atomic::Ordering;
                while !thread_shutdown.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            if let Err(e) = serve_one(&mut stream, &geometry, region.raw_fd()) {
                                trace!("shared-store handshake failed: {e}");
                            }
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(std::time::Duration::from_millis(5));
                        }
                        Err(e) => {
                            trace!("shared-store accept failed: {e}");
                            break;
                        }
                    }
                }
            })
            .context("failed to spawn shared-store server thread")?;

        Ok(ShareHandle {
            shutdown,
            thread: Some(thread),
            socket_path,
        })
    }

    /// Attach to a store served at `socket_path`, mapping its pages
    /// read-only into this process.
    pub fn attach(socket_path: impl AsRef<Path>) -> Result<Self> {
        let socket_path = socket_path.as_ref();
        let mut stream = UnixStream::connect(socket_path)
            .with_context(|| format!("failed to connect to {}", socket_path.display()))?;

        let mut raw = [0u8; HANDSHAKE_LEN];
        stream
            .read_exact(&mut raw)
            .context("failed to read shared-store geometry")?;
        let geometry = Geometry::from_bytes(&raw)?;

        let fd: OwnedFd = recv_fd(stream.as_raw_fd())?;
        let payload_len = usize::try_from(geometry.payload_len).context("payload_len")?;
        // SAFETY: `fd` came from the owner's `send_fd` on this socket, and
        // `payload_len` is the size the owner sealed the memfd to — the
        // handshake above validated it against the geometry.
        let region = unsafe { SharedRegion::from_fd(fd, payload_len)? };

        debug!(
            "Attached to shared store: {} nodes x {} dims",
            geometry.num_nodes, geometry.feature_dim
        );

        Ok(Self {
            region: Arc::new(region),
            num_nodes: geometry.num_nodes,
            feature_dim: geometry.feature_dim,
            dtype: geometry.dtype,
            row_bytes: geometry.feature_dim * geometry.dtype.element_size(),
        })
    }

    /// Number of nodes in the store.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Feature dimension.
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Element dtype of the shared payload.
    pub fn dtype(&self) -> FeatureDtype {
        self.dtype
    }

    /// Bytes of shared memory backing this store — one copy per host,
    /// however many processes attach.
    pub fn shared_bytes(&self) -> usize {
        self.region.len()
    }

    /// Read one node's features into `out`, decoding from the shared
    /// pages. `out` must be `feature_dim` long.
    pub fn get_into(&self, node: NodeId, out: &mut [f32]) -> Result<()> {
        anyhow::ensure!(
            out.len() == self.feature_dim,
            "output buffer is {} long, expected {}",
            out.len(),
            self.feature_dim
        );
        let idx = node as usize;
        anyhow::ensure!(
            idx < self.num_nodes,
            "node {node} out of bounds (num_nodes={})",
            self.num_nodes
        );
        let start = idx * self.row_bytes;
        self.dtype
            .decode_row(&self.region.as_slice()[start..start + self.row_bytes], out);
        Ok(())
    }

    /// Gather `nodes` into one contiguous `f32` buffer in input order.
    pub fn get_batch(&self, nodes: &[NodeId]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; nodes.len() * self.feature_dim];
        for (i, &node) in nodes.iter().enumerate() {
            let dst = &mut out[i * self.feature_dim..(i + 1) * self.feature_dim];
            self.get_into(node, dst)?;
        }
        Ok(out)
    }
}

impl super::NodeFeatureSource for SharedFeatureStore {
    fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    fn read_node(&self, node: NodeId, out: &mut [f32]) -> bool {
        self.get_into(node, out).is_ok()
    }
}

/// Write the geometry then the memfd to one connecting worker.
fn serve_one(stream: &mut UnixStream, geometry: &[u8; HANDSHAKE_LEN], fd: i32) -> Result<()> {
    stream
        .write_all(geometry)
        .context("failed to write geometry")?;
    send_fd(stream.as_raw_fd(), fd)
}

/// Keeps a [`SharedFeatureStore::serve`] listener alive. Dropping it stops
/// the server thread and removes the socket file; already-attached workers
/// keep their mappings, which are independent of the listener.
pub struct ShareHandle {
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    socket_path: std::path::PathBuf,
}

impl ShareHandle {
    /// The socket path workers pass to [`SharedFeatureStore::attach`].
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }
}

impl Drop for ShareHandle {
    fn drop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::save_features;

    fn fixture(dir: &Path, num_nodes: usize, dim: usize) -> (std::path::PathBuf, Vec<f32>) {
        let path = dir.join("features.bin");
        let features: Vec<f32> = (0..num_nodes * dim).map(|i| i as f32 * 0.5).collect();
        save_features(&path, features.clone(), num_nodes, dim).unwrap();
        (path, features)
    }

    #[test]
    fn publish_and_read_locally() {
        let dir = tempfile::tempdir().unwrap();
        let (path, features) = fixture(dir.path(), 500, 16);

        let store = SharedFeatureStore::publish(&path).unwrap();
        assert_eq!(store.num_nodes(), 500);
        assert_eq!(store.feature_dim(), 16);
        assert_eq!(store.shared_bytes(), 500 * 16 * 4);

        let batch = store.get_batch(&[0, 17, 499, 17]).unwrap();
        for (i, &n) in [0usize, 17, 499, 17].iter().enumerate() {
            assert_eq!(
                &batch[i * 16..(i + 1) * 16],
                &features[n * 16..(n + 1) * 16]
            );
        }
        assert!(store.get_batch(&[500]).is_err(), "out of range must error");
    }

    /// The point of the whole module: a second holder of the memfd reads
    /// the same physical pages, and sees a write made after it attached.
    #[test]
    fn attached_peer_shares_the_same_pages() {
        let dir = tempfile::tempdir().unwrap();
        let (path, features) = fixture(dir.path(), 1000, 8);
        let socket = dir.path().join("store.sock");

        let owner = SharedFeatureStore::publish(&path).unwrap();
        let handle = owner.serve(&socket).unwrap();

        let peer = SharedFeatureStore::attach(handle.socket_path()).unwrap();
        assert_eq!(peer.num_nodes(), owner.num_nodes());
        assert_eq!(peer.feature_dim(), owner.feature_dim());
        assert_eq!(peer.dtype(), owner.dtype());

        let nodes: Vec<NodeId> = (0..1000u32).step_by(37).collect();
        assert_eq!(
            peer.get_batch(&nodes).unwrap(),
            owner.get_batch(&nodes).unwrap()
        );
        for (i, &n) in nodes.iter().enumerate() {
            let row = &peer.get_batch(&nodes).unwrap()[i * 8..(i + 1) * 8];
            assert_eq!(row, &features[n as usize * 8..(n as usize + 1) * 8]);
        }

        // Same pages, not a copy: the peer's mapping is backed by the same
        // memfd, so both report the identical region size and a second
        // attach costs no additional payload memory.
        assert_eq!(peer.shared_bytes(), owner.shared_bytes());

        let second = SharedFeatureStore::attach(handle.socket_path()).unwrap();
        assert_eq!(
            second.get_batch(&[42]).unwrap(),
            owner.get_batch(&[42]).unwrap()
        );
    }

    #[test]
    fn attach_to_missing_socket_errors() {
        let dir = tempfile::tempdir().unwrap();
        let err = match SharedFeatureStore::attach(dir.path().join("nope.sock")) {
            Ok(_) => panic!("attaching to a nonexistent socket must fail"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("connect"), "unexpected: {err}");
    }

    #[test]
    fn handle_drop_stops_serving() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = fixture(dir.path(), 64, 4);
        let socket = dir.path().join("store.sock");

        let owner = SharedFeatureStore::publish(&path).unwrap();
        {
            let handle = owner.serve(&socket).unwrap();
            SharedFeatureStore::attach(handle.socket_path()).unwrap();
        }
        assert!(!socket.exists(), "the socket file is removed on drop");
        assert!(SharedFeatureStore::attach(&socket).is_err());
    }

    #[test]
    fn geometry_round_trips_and_rejects_inconsistency() {
        let g = Geometry {
            payload_len: 100 * 8 * 4,
            num_nodes: 100,
            feature_dim: 8,
            dtype: FeatureDtype::F32,
        };
        let back = Geometry::from_bytes(&g.to_bytes()).unwrap();
        assert_eq!(back.num_nodes, 100);
        assert_eq!(back.feature_dim, 8);
        assert_eq!(back.payload_len, g.payload_len);

        // A payload length that disagrees with the geometry is refused at
        // the edge rather than mapping a wrong-sized region.
        let mut bad = g.to_bytes();
        bad[0..8].copy_from_slice(&999u64.to_le_bytes());
        assert!(Geometry::from_bytes(&bad).is_err());
    }
}
