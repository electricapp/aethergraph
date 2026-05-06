//! Async CSR graph with io_uring for parallel neighbor reads from NVMe.
//!
//! **The Problem**: Billion-scale graphs don't fit in RAM. When training GNNs, we need
//! to fetch neighbors for many nodes simultaneously, but synchronous disk I/O is too slow
//! and leaves the GPU starved.
//!
//! **The Solution**:
//! - Store CSR graph in file on NVMe
//! - Use io_uring (Linux) to issue many parallel async reads
//! - Batched I/O hides NVMe latency, keeping GPU fed
//!
//! **io_uring Performance Tiers**:
//! 1. SQPOLL: Kernel polls submission queue (reduced syscalls, check NEED_WAKEUP)
//! 2. Standard io_uring: Batched async I/O
//! 3. Tokio fallback: spawn_blocking for non-Linux platforms
//!
//! Note: We don't use O_DIRECT/IOPOLL for graph reads since adjacency lists are
//! variable-sized, making buffer alignment complex. SQPOLL still provides benefits.

use super::csr::{EdgeOffset, NodeId};
use anyhow::{Context, Result};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, trace};

#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(target_os = "linux")]
use tracing::warn;

#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;

/// Header size in the binary format (from mmap module)
const HEADER_SIZE: usize = 32;
/// Graph file magic (`AETH`).
const GRAPH_MAGIC: u32 = 0x4145_5448;
/// Supported graph file format version.
const GRAPH_VERSION: u32 = 1;
/// Defensive cap against malformed headers.
const MAX_NODES: u64 = 10_000_000_000;
/// Defensive cap against malformed headers.
const MAX_EDGES: u64 = 100_000_000_000;

/// Async CSR graph backed by NVMe storage with io_uring.
///
/// The graph structure lives in a file on NVMe. We keep the file descriptor
/// open and use io_uring with SQPOLL for parallel async reads.
///
/// **io_uring Design (Linux)**:
/// - SQPOLL: Kernel polls submission queue (only syscall if NEED_WAKEUP set)
/// - Registered FDs: Pre-register file descriptor for faster access
/// - Batch submission: Submit many reads at once for parallelism
pub struct AsyncCsrGraph {
    /// File descriptor for the graph file
    file: Arc<File>,

    /// Path to graph file (used for logging and debugging)
    #[allow(dead_code)]
    path: PathBuf,

    /// Number of nodes
    num_nodes: usize,

    /// Number of edges
    num_edges: usize,

    /// Byte offset where offsets array starts in file (reserved for future use)
    _offsets_start: u64,

    /// Byte offset where edges array starts in file
    edges_start: u64,

    /// Offsets array (we keep this in memory - it's small)
    /// offsets[i] = index in edges array where node i's neighbors start
    // Arc<Vec<_>> rather than Arc<[_]>: `Arc::from(Box<[T]>)` (the natural
    // conversion path) allocates a fresh ArcInner with the whole slice copied
    // inside, doubling peak memory for billion-node graphs (16 GB instead of
    // 8 GB on a 1 B × u64 offsets array). `Arc::new(vec)` just wraps the
    // existing heap allocation by reference, keeping the peak flat.
    offsets: Arc<Vec<EdgeOffset>>,

    /// Pool of io_uring handles for concurrent batch reads (Linux only)
    #[cfg(target_os = "linux")]
    uring_pool: Option<Vec<Arc<parking_lot::Mutex<crate::internal::uring::UringHandle>>>>,
    /// Round-robin selector for `uring_pool`
    #[cfg(target_os = "linux")]
    uring_next: AtomicUsize,
}

impl AsyncCsrGraph {
    /// Load graph from disk for async access.
    ///
    /// Reads the offsets array into memory (small), keeps file open for
    /// async neighbor reads via io_uring.
    pub async fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        debug!("Loading async graph from {}", path.display());

        // Open file (regular, not O_DIRECT - variable-sized reads make alignment complex)
        let path_owned = path.to_path_buf();
        let std_file = tokio::task::spawn_blocking(move || File::open(&path_owned))
            .await
            .context("spawn_blocking failed")??;

        // Read header using pread
        let mut header = [0u8; HEADER_SIZE];
        std_file
            .read_exact_at(&mut header, 0)
            .context("failed to read header")?;

        // Parse header
        let magic = u32::from_le_bytes(header[0..4].try_into()?);
        let version = u32::from_le_bytes(header[4..8].try_into()?);
        anyhow::ensure!(
            magic == GRAPH_MAGIC,
            "invalid graph magic: expected {:#x}, got {:#x}",
            GRAPH_MAGIC,
            magic
        );
        anyhow::ensure!(
            version == GRAPH_VERSION,
            "unsupported graph version: expected {}, got {}",
            GRAPH_VERSION,
            version
        );

        let num_nodes_u64 = u64::from_le_bytes(header[8..16].try_into()?);
        let num_edges_u64 = u64::from_le_bytes(header[16..24].try_into()?);
        let _has_weights = u32::from_le_bytes(header[24..28].try_into()?) != 0;
        anyhow::ensure!(
            num_nodes_u64 <= MAX_NODES,
            "num_nodes {} exceeds maximum {}",
            num_nodes_u64,
            MAX_NODES
        );
        anyhow::ensure!(
            num_edges_u64 <= MAX_EDGES,
            "num_edges {} exceeds maximum {}",
            num_edges_u64,
            MAX_EDGES
        );
        let num_nodes =
            usize::try_from(num_nodes_u64).context("num_nodes does not fit in usize")?;
        let num_edges =
            usize::try_from(num_edges_u64).context("num_edges does not fit in usize")?;

        debug!("Graph metadata: nodes={}, edges={}", num_nodes, num_edges);

        // Calculate offsets in file
        let offsets_size = num_nodes
            .checked_add(1)
            .and_then(|n| n.checked_mul(std::mem::size_of::<EdgeOffset>()))
            .ok_or_else(|| anyhow::anyhow!("offsets array size overflow"))?;
        let offsets_start = HEADER_SIZE as u64;
        let edges_start = offsets_start
            .checked_add(offsets_size as u64)
            .ok_or_else(|| anyhow::anyhow!("edges_start overflow"))?;
        let min_edges_bytes = (num_edges as u64)
            .checked_mul(std::mem::size_of::<NodeId>() as u64)
            .ok_or_else(|| anyhow::anyhow!("edge byte size overflow"))?;
        let min_file_size = edges_start
            .checked_add(min_edges_bytes)
            .ok_or_else(|| anyhow::anyhow!("minimum file size overflow"))?;
        let file_size = std_file
            .metadata()
            .context("failed to stat graph file")?
            .len();
        anyhow::ensure!(
            file_size >= min_file_size,
            "graph file truncated: expected at least {} bytes, got {}",
            min_file_size,
            file_size
        );

        // Read offsets array into memory. For billion-node graphs this is 8
        // GB, so the allocation pattern matters — go from file straight into
        // Vec<EdgeOffset> via a zero-copy &mut [u8] cast instead of a Vec<u8>
        // + .collect() hop that would temporarily double the footprint.
        //
        // Correctness: file is LE by spec; EdgeOffset is u64. On both targets
        // we support (x86_64 + aarch64) native byte order is LE so the bytes
        // land in the right shape. A big-endian host would need a byte-swap
        // pass here — asserted below so future ports don't silently miscorrupt.
        const { assert!(cfg!(target_endian = "little"), "graph file is little-endian") };
        let mut offsets: Vec<EdgeOffset> = vec![0; num_nodes + 1];
        {
            let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut offsets[..]);
            debug_assert_eq!(bytes.len(), offsets_size);
            std_file
                .read_exact_at(bytes, offsets_start)
                .context("failed to read offsets")?;
        }
        anyhow::ensure!(
            offsets.len() == num_nodes + 1,
            "invalid offsets length: expected {}, got {}",
            num_nodes + 1,
            offsets.len()
        );
        anyhow::ensure!(offsets[0] == 0, "invalid offsets: offsets[0] must be 0");
        for (i, window) in offsets.windows(2).enumerate() {
            anyhow::ensure!(
                window[0] <= window[1],
                "invalid offsets: offsets[{}]={} > offsets[{}]={}",
                i,
                window[0],
                i + 1,
                window[1]
            );
            anyhow::ensure!(
                window[1] <= num_edges as u64,
                "invalid offsets: offsets[{}]={} exceeds num_edges {}",
                i + 1,
                window[1],
                num_edges
            );
        }
        anyhow::ensure!(
            offsets[num_nodes] == num_edges as u64,
            "invalid offsets tail: offsets[last]={} != num_edges {}",
            offsets[num_nodes],
            num_edges
        );
        let offsets = Arc::new(offsets);

        debug!(
            "Loaded offsets array ({} MB in RAM)",
            offsets_size as f64 / 1_000_000.0
        );

        let file_arc = Arc::new(std_file);

        // Initialize io_uring (Linux only)
        #[cfg(target_os = "linux")]
        let uring_pool = Self::setup_uring(&file_arc)?;

        Ok(Self {
            file: file_arc,
            path: path.to_path_buf(),
            num_nodes,
            num_edges,
            _offsets_start: offsets_start,
            edges_start,
            offsets,
            #[cfg(target_os = "linux")]
            uring_pool,
            #[cfg(target_os = "linux")]
            uring_next: AtomicUsize::new(0),
        })
    }

    /// Setup io_uring with SQPOLL and registered file descriptors.
    ///
    /// Note: We use SQPOLL only (no IOPOLL) since we're not using O_DIRECT.
    /// Variable-sized adjacency reads make buffer alignment complex, so we
    /// use buffered I/O with SQPOLL for reduced syscalls.
    #[cfg(target_os = "linux")]
    fn setup_uring(
        file: &Arc<File>,
    ) -> Result<Option<Vec<Arc<parking_lot::Mutex<crate::internal::uring::UringHandle>>>>> {
        use crate::internal::uring::UringHandle;

        let pool_size = std::thread::available_parallelism()
            .map(|n| n.get().clamp(1, 4))
            .unwrap_or(1);
        let mut handles = Vec::with_capacity(pool_size);

        for idx in 0..pool_size {
            let mut handle = match UringHandle::new_sqpoll_only(crate::internal::uring::DEFAULT_RING_ENTRIES, 1000) {
                Ok(h) => h,
                Err(e) => {
                    if idx == 0 {
                        warn!("Failed to create io_uring: {}", e);
                        return Ok(None);
                    }
                    warn!("Failed to create extra io_uring handle {}: {}", idx, e);
                    break;
                }
            };

            if idx == 0 {
                if handle.is_sqpoll() {
                    debug!("io_uring: SQPOLL enabled (reduced syscalls)");
                } else {
                    debug!("io_uring: standard mode (batched I/O)");
                }
            }

            if let Err(e) = handle.register_fd(file) {
                warn!("Failed to register FD on handle {}: {}", idx, e);
            }
            handles.push(Arc::new(parking_lot::Mutex::new(handle)));
        }

        if handles.is_empty() {
            return Ok(None);
        }
        debug!("Initialized io_uring pool with {} handle(s)", handles.len());
        Ok(Some(handles))
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// True when `batch_neighbors` uses the io_uring fast path, false when it
    /// falls back to tokio spawn_blocking pread (non-Linux, or when ring
    /// setup failed e.g. under seccomp / missing CAP_SYS_ADMIN for SQPOLL).
    /// Exposed so perf tests can confirm which path they actually exercised.
    pub fn io_uring_active(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.uring_pool
                .as_ref()
                .map_or(false, |p| !p.is_empty())
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// Get degree of a node (number of neighbors)
    #[inline]
    pub fn degree(&self, node: NodeId) -> usize {
        let idx = node as usize;
        if idx >= self.num_nodes {
            return 0;
        }
        (self.offsets[idx + 1] - self.offsets[idx]) as usize
    }

    /// Async read neighbors for a single node from NVMe.
    ///
    /// This uses io_uring on Linux for zero-copy async I/O.
    /// On other platforms, falls back to tokio::fs.
    pub async fn neighbors(&self, node: NodeId) -> Result<Vec<NodeId>> {
        let idx = node as usize;
        if idx >= self.num_nodes {
            return Ok(Vec::new());
        }

        let start_edge = self.offsets[idx] as usize;
        let end_edge = self.offsets[idx + 1] as usize;
        let num_neighbors = end_edge - start_edge;

        if num_neighbors == 0 {
            return Ok(Vec::new());
        }

        // Calculate byte offset in file where this node's edges start
        let byte_offset = self.edges_start + (start_edge * std::mem::size_of::<NodeId>()) as u64;
        let byte_size = num_neighbors * std::mem::size_of::<NodeId>();

        trace!(
            "Reading {} neighbors for node {} at offset {}",
            num_neighbors,
            node,
            byte_offset
        );

        // Read from file asynchronously
        let neighbors = self.read_edges_at(byte_offset, byte_size).await?;

        Ok(neighbors)
    }

    /// **THE KEY FUNCTION**: Batch read neighbors for many nodes in parallel.
    ///
    /// This is where io_uring shines - we issue many read requests simultaneously,
    /// and io_uring completes them as fast as NVMe can serve them.
    ///
    /// This is 10-100x faster than reading sequentially because:
    /// - NVMe has high parallelism (many internal channels)
    /// - io_uring minimizes syscall overhead
    /// - Reads happen concurrently while we're processing other data
    pub async fn batch_neighbors(&self, nodes: &[NodeId]) -> Result<Vec<Vec<NodeId>>> {
        debug!(
            "Batch reading neighbors for {} nodes via io_uring",
            nodes.len()
        );

        #[cfg(target_os = "linux")]
        {
            self.batch_neighbors_io_uring(nodes).await
        }

        #[cfg(not(target_os = "linux"))]
        {
            self.batch_neighbors_tokio(nodes).await
        }
    }

    /// io_uring implementation (Linux only) with proper SQPOLL handling.
    ///
    /// Uses spawn_blocking since io_uring operations are blocking.
    /// Properly checks NEED_WAKEUP flag before syscalls with SQPOLL.
    #[cfg(target_os = "linux")]
    async fn batch_neighbors_io_uring(&self, nodes: &[NodeId]) -> Result<Vec<Vec<NodeId>>> {
        // If io_uring isn't available, fall back to tokio
        let uring = match &self.uring_pool {
            Some(pool) if !pool.is_empty() => {
                let idx = self.uring_next.fetch_add(1, Ordering::Relaxed) % pool.len();
                Arc::clone(&pool[idx])
            }
            None => return self.batch_neighbors_tokio(nodes).await,
            _ => return self.batch_neighbors_tokio(nodes).await,
        };

        // Clone what we need for the blocking task
        let file = Arc::clone(&self.file);
        let nodes = nodes.to_vec();
        let offsets = self.offsets.clone();
        let edges_start = self.edges_start;
        let num_nodes = self.num_nodes;

        // Run io_uring operations in spawn_blocking to not block tokio runtime
        let results = tokio::task::spawn_blocking(move || {
            let fd = file.as_raw_fd();

            // Pre-allocate buffers for all reads
            let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(nodes.len());
            let mut read_info: Vec<(usize, u64, usize)> = Vec::with_capacity(nodes.len());

            for (i, &node) in nodes.iter().enumerate() {
                let idx = node as usize;
                if idx >= num_nodes {
                    buffers.push(Vec::new());
                    continue;
                }

                let start_edge = offsets[idx] as usize;
                let end_edge = offsets[idx + 1] as usize;
                let num_neighbors = end_edge - start_edge;

                if num_neighbors == 0 {
                    buffers.push(Vec::new());
                    continue;
                }

                let byte_offset = edges_start + (start_edge * std::mem::size_of::<NodeId>()) as u64;
                let byte_size = num_neighbors * std::mem::size_of::<NodeId>();

                buffers.push(vec![0u8; byte_size]);
                read_info.push((i, byte_offset, byte_size));
            }

            if read_info.is_empty() {
                return Ok(vec![Vec::new(); nodes.len()]);
            }

            // Convert to (offset, ptr, len) tuples for batch_read.
            let reads: Vec<(u64, *mut u8, usize)> = read_info
                .iter()
                .map(|&(buf_idx, byte_offset, byte_size)| {
                    (byte_offset, buffers[buf_idx].as_mut_ptr(), byte_size)
                })
                .collect();

            // Hold the ring lock only for SQ/CQ interaction.
            {
                let mut handle = uring.lock();
                // SAFETY: each ptr in `reads` points into buffers[i] which lives
                // until this closure returns; batch_read completes synchronously.
                crate::internal::uring::batch_read(&mut handle, fd, &reads)?;
                trace!("Submitted {} reads via io_uring", read_info.len());
            }

            // Convert buffers to NodeId vectors
            let mut results = vec![Vec::new(); nodes.len()];
            for &(buf_idx, _, _) in &read_info {
                let buffer = &buffers[buf_idx];
                let neighbors: Vec<NodeId> = buffer
                    .chunks_exact(4)
                    .map(|chunk| NodeId::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                results[buf_idx] = neighbors;
            }

            trace!("Completed batch read of {} nodes via io_uring", nodes.len());

            Ok::<_, anyhow::Error>(results)
        })
        .await
        .context("spawn_blocking failed")??;

        Ok(results)
    }

    /// Tokio fallback implementation - still async but not io_uring
    async fn batch_neighbors_tokio(&self, nodes: &[NodeId]) -> Result<Vec<Vec<NodeId>>> {
        use tokio::task;

        trace!("Using tokio fallback for batch read (not io_uring)");

        let tasks: Vec<_> = nodes
            .iter()
            .map(|&node| {
                let file = Arc::clone(&self.file);
                let offsets = Arc::clone(&self.offsets);
                let edges_start = self.edges_start;
                let num_nodes = self.num_nodes;

                task::spawn_blocking(move || {
                    let idx = node as usize;
                    if idx >= num_nodes {
                        return Ok(Vec::new());
                    }

                    let start_edge = offsets[idx] as usize;
                    let end_edge = offsets[idx + 1] as usize;
                    let num_neighbors = end_edge - start_edge;

                    if num_neighbors == 0 {
                        return Ok(Vec::new());
                    }

                    let byte_offset =
                        edges_start + (start_edge * std::mem::size_of::<NodeId>()) as u64;
                    let byte_size = num_neighbors * std::mem::size_of::<NodeId>();

                    let mut buffer = vec![0u8; byte_size];
                    file.read_exact_at(&mut buffer, byte_offset)
                        .context("failed to read from file")?;

                    let neighbors: Vec<NodeId> = buffer
                        .chunks_exact(4)
                        .map(|chunk| {
                            NodeId::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                        })
                        .collect();

                    Ok::<_, anyhow::Error>(neighbors)
                })
            })
            .collect();

        let mut results = Vec::with_capacity(nodes.len());
        for task in tasks {
            results.push(task.await.context("blocking task failed")??);
        }

        Ok(results)
    }

    /// Helper: Read edges from file at given offset
    async fn read_edges_at(&self, offset: u64, size: usize) -> Result<Vec<NodeId>> {
        // Use blocking task since FileExt::read_exact_at is sync
        let file = Arc::clone(&self.file);
        let buffer = tokio::task::spawn_blocking(move || {
            let mut buf = vec![0u8; size];
            file.read_exact_at(&mut buf, offset)
                .context("failed to read from file")?;
            Ok::<_, anyhow::Error>(buf)
        })
        .await
        .context("blocking task panicked")?
        .context("read failed")?;

        // Convert bytes to NodeIds
        let neighbors: Vec<NodeId> = buffer
            .chunks_exact(4)
            .map(|chunk| NodeId::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(neighbors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{save_graph, Graph};
    use tempfile::NamedTempFile;

    async fn create_test_graph_file() -> NamedTempFile {
        let edges = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
        ];
        let graph = Graph::from_edges(5, &edges, None).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        save_graph(&graph, temp_file.path()).unwrap();

        temp_file
    }

    #[tokio::test]
    async fn test_async_neighbors() {
        let temp_file = create_test_graph_file().await;
        let graph = AsyncCsrGraph::load(temp_file.path()).await.unwrap();

        let neighbors = graph.neighbors(0).await.unwrap();
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
        assert!(neighbors.contains(&3));
    }

    #[tokio::test]
    async fn test_batch_neighbors() {
        let temp_file = create_test_graph_file().await;
        let graph = AsyncCsrGraph::load(temp_file.path()).await.unwrap();

        let batch = graph.batch_neighbors(&[0, 1, 2]).await.unwrap();

        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].len(), 3); // node 0 has 3 neighbors
        assert_eq!(batch[1].len(), 2); // node 1 has 2 neighbors
        assert_eq!(batch[2].len(), 2); // node 2 has 2 neighbors
    }

    #[tokio::test]
    async fn test_large_batch() {
        let temp_file = create_test_graph_file().await;
        let graph = AsyncCsrGraph::load(temp_file.path()).await.unwrap();

        // Simulate GNN sampling: fetch neighbors for 100 nodes in parallel
        let nodes: Vec<NodeId> = (0..100).map(|i| i % 5).collect();
        let batch = graph.batch_neighbors(&nodes).await.unwrap();

        assert_eq!(batch.len(), 100);
    }
}
