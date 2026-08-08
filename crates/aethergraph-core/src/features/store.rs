//! Memory-mapped feature store for billion-scale GNN training.
//!
//! Stores node features (embeddings) in a contiguous memory-mapped array for
//! zero-copy, TB-scale feature access.
//!
//! # File Format
//! ```text
//! [8 bytes] Magic: "AETHFEAT"
//! [8 bytes] num_nodes (u64 little-endian)
//! [8 bytes] feature_dim (u64 little-endian)
//! [8 bytes] data_offset (u64 little-endian, > 32; written files use 512)
//! [N bytes] Raw f32 feature data (num_nodes * feature_dim * 4)
//! ```
//!
//! # Design
//! - Single mmap'd file (not millions of tiny files)
//! - Zero-copy access via bytemuck slice casting
//! - Extensible header with explicit payload offset
//! - Batch-optimized: vectorized loads for sampled nodes
//!
//! # Performance
//! - 1B nodes × 768 dims × 4 bytes = 3TB → handled via mmap
//! - Random access: ~100ns (L3 cache) to ~10μs (NVMe)
//! - Batch of 1000 nodes: ~1-10ms depending on cache hits

use crate::FeatureDtype;
use crate::graph::NodeId;
use anyhow::{Context, Result};
use half::f16;
use memmap2::{Mmap, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::{debug, trace};

/// Magic bytes identifying feature files
const MAGIC: &[u8; 8] = b"AETHFEAT";

/// Header size in bytes (magic + num_nodes + feature_dim + data_offset)
const HEADER_SIZE: usize = 32;
/// Aligned start for feature payload in newly written files.
const ALIGNED_DATA_OFFSET: u64 = 512;

/// Feature data builder for incremental construction.
///
/// Use this to build features node-by-node, then save to disk.
#[derive(Debug, Clone)]
pub struct FeatureData {
    /// Number of nodes
    pub num_nodes: u64,

    /// Feature dimension per node
    pub feature_dim: u64,

    /// Flat feature array: [node0_f0, node0_f1, ..., node1_f0, ...]
    /// Row-major layout for cache-friendly access
    pub features: Vec<f32>,
}

impl FeatureData {
    /// Create new feature data with zeros.
    ///
    /// Panics if `num_nodes * feature_dim` overflows `usize` (this constructor
    /// has no error channel; the overflow would otherwise wrap silently).
    pub fn new(num_nodes: usize, feature_dim: usize) -> Self {
        let len = num_nodes
            .checked_mul(feature_dim)
            .expect("num_nodes * feature_dim overflows usize");
        Self {
            num_nodes: num_nodes as u64,
            feature_dim: feature_dim as u64,
            features: vec![0.0; len],
        }
    }

    /// Create from existing features
    pub fn from_features(features: Vec<f32>, num_nodes: usize, feature_dim: usize) -> Result<Self> {
        let expected = num_nodes
            .checked_mul(feature_dim)
            .ok_or_else(|| anyhow::anyhow!("num_nodes * feature_dim overflows usize"))?;
        anyhow::ensure!(
            features.len() == expected,
            "feature array size mismatch: expected {}, got {}",
            expected,
            features.len()
        );

        Ok(Self {
            num_nodes: num_nodes as u64,
            feature_dim: feature_dim as u64,
            features,
        })
    }

    /// Get features for a node
    #[inline(always)]
    pub fn get(&self, node: NodeId) -> Option<&[f32]> {
        let node_idx = node as usize;
        if node_idx >= self.num_nodes as usize {
            return None;
        }

        let feature_dim = self.feature_dim as usize;
        let start = node_idx * feature_dim;
        let end = start + feature_dim;

        Some(&self.features[start..end])
    }

    /// Get mutable features for a node
    #[inline(always)]
    pub fn get_mut(&mut self, node: NodeId) -> Option<&mut [f32]> {
        let node_idx = node as usize;
        if node_idx >= self.num_nodes as usize {
            return None;
        }

        let feature_dim = self.feature_dim as usize;
        let start = node_idx * feature_dim;
        let end = start + feature_dim;

        Some(&mut self.features[start..end])
    }
}

/// Telemetry for feature loading operations.
///
/// Tracks performance metrics for profiling and optimization.
#[derive(Debug, Default, Clone)]
pub struct FeatureLoadTelemetry {
    /// Number of single-node get() calls
    pub single_gets: u64,
    /// Number of batch get_batch() calls
    pub batch_gets: u64,
    /// Total nodes loaded (across all operations)
    pub total_nodes_loaded: u64,
    /// Total features loaded (nodes × feature_dim)
    pub total_features_loaded: u64,
    /// Total bytes loaded
    pub total_bytes_loaded: u64,
    /// Total time spent in get() operations (nanoseconds)
    pub get_time_ns: u64,
    /// Total time spent in get_batch() operations (nanoseconds)
    pub batch_time_ns: u64,
}

/// A `AtomicU64` padded to its own 64-byte cache line.
///
/// Concurrent loaders bump several of these counters at once; without padding
/// they would share cache lines and ping-pong them between cores (false
/// sharing). `Deref` keeps the `.fetch_add()` / `.load()` call sites unchanged.
#[repr(align(64))]
#[derive(Debug, Default)]
pub(crate) struct PaddedAtomicU64(pub(crate) AtomicU64);

impl std::ops::Deref for PaddedAtomicU64 {
    type Target = AtomicU64;
    #[inline(always)]
    fn deref(&self) -> &AtomicU64 {
        &self.0
    }
}

/// Atomic counters behind [`FeatureLoadTelemetry`].
///
/// The read paths bump these with relaxed atomics so enabling telemetry
/// does not put a lock on the per-node hot path; `snapshot()` assembles
/// the public summary struct. Each counter is cache-line padded to avoid
/// false sharing between concurrent loaders.
#[derive(Debug, Default)]
pub(crate) struct TelemetryCells {
    pub(crate) single_gets: PaddedAtomicU64,
    pub(crate) batch_gets: PaddedAtomicU64,
    pub(crate) total_nodes_loaded: PaddedAtomicU64,
    pub(crate) total_features_loaded: PaddedAtomicU64,
    pub(crate) total_bytes_loaded: PaddedAtomicU64,
    pub(crate) get_time_ns: PaddedAtomicU64,
    pub(crate) batch_time_ns: PaddedAtomicU64,
}

impl TelemetryCells {
    pub(crate) fn snapshot(&self) -> FeatureLoadTelemetry {
        FeatureLoadTelemetry {
            single_gets: self.single_gets.load(Ordering::Relaxed),
            batch_gets: self.batch_gets.load(Ordering::Relaxed),
            total_nodes_loaded: self.total_nodes_loaded.load(Ordering::Relaxed),
            total_features_loaded: self.total_features_loaded.load(Ordering::Relaxed),
            total_bytes_loaded: self.total_bytes_loaded.load(Ordering::Relaxed),
            get_time_ns: self.get_time_ns.load(Ordering::Relaxed),
            batch_time_ns: self.batch_time_ns.load(Ordering::Relaxed),
        }
    }
}

impl FeatureLoadTelemetry {
    /// Get total time in seconds
    #[inline]
    pub fn total_time_secs(&self) -> f64 {
        (self.get_time_ns + self.batch_time_ns) as f64 / 1e9
    }

    /// Get throughput in features/second
    #[inline]
    pub fn throughput_features_per_sec(&self) -> f64 {
        let total_time = self.total_time_secs();
        if total_time > 0.0 {
            self.total_features_loaded as f64 / total_time
        } else {
            0.0
        }
    }

    /// Get throughput in GB/second
    #[inline]
    pub fn throughput_gb_per_sec(&self) -> f64 {
        let total_time = self.total_time_secs();
        if total_time > 0.0 {
            (self.total_bytes_loaded as f64 / 1e9) / total_time
        } else {
            0.0
        }
    }

    /// Get average batch size (nodes per batch)
    #[inline]
    pub fn avg_batch_size(&self) -> f64 {
        if self.batch_gets > 0 {
            // saturating_sub: counters are bumped independently, so a snapshot
            // taken mid-update can momentarily show single_gets > total.
            self.total_nodes_loaded.saturating_sub(self.single_gets) as f64 / self.batch_gets as f64
        } else {
            0.0
        }
    }

    /// Print summary statistics
    pub fn print_summary(&self) {
        let total_ops = self.single_gets + self.batch_gets;
        if total_ops == 0 {
            debug!("No feature loads yet");
            return;
        }

        debug!("Feature Load Telemetry:");
        debug!(
            "  Operations:   {} total ({} single, {} batch)",
            total_ops, self.single_gets, self.batch_gets
        );
        debug!("  Nodes loaded: {}", self.total_nodes_loaded);
        debug!("  Features:     {}", self.total_features_loaded);
        debug!(
            "  Data:         {:.2} MB",
            self.total_bytes_loaded as f64 / 1e6
        );
        debug!("  Time:         {:.4}s", self.total_time_secs());
        debug!(
            "  Throughput:   {:.2} M features/sec",
            self.throughput_features_per_sec() / 1e6
        );
        debug!("  Throughput:   {:.2} GB/sec", self.throughput_gb_per_sec());
        if self.batch_gets > 0 {
            debug!("  Avg batch:    {:.1} nodes/batch", self.avg_batch_size());
        }
    }
}

/// Memory-mapped feature store with zero-copy access.
///
/// # File Format
/// Simple binary format with 32-byte header and explicit payload offset.
/// See module-level docs for format details.
///
/// # Optimization Notes
/// - Data stored in native little-endian format (no byte swapping on x86)
/// - Features accessed directly via mmap without deserialization
/// - bytemuck used for safe zero-copy slice casting
#[derive(Clone)]
pub struct FeatureStore {
    mmap: Arc<Mmap>,
    num_nodes: usize,
    feature_dim: usize,
    features_start_offset: usize,
    feature_data_len_bytes: usize,
    dtype: FeatureDtype,
    /// Optional telemetry collector
    telemetry: Option<Arc<TelemetryCells>>,
}

// No hand-written `unsafe impl Send/Sync`: every field (`Arc<Mmap>`,
// counters, plain data) is already Send + Sync, so the compiler derives
// both. Writing the impls by hand would silently suppress that checking
// if a non-thread-safe field were ever added.

impl FeatureStore {
    /// Load features from memory-mapped file.
    ///
    /// # Performance
    /// - O(1) load time (just mmaps + validates header)
    /// - Zero-copy access to features
    /// - OS handles paging from NVMe as needed
    #[tracing::instrument(skip(path), fields(path = %path.as_ref().display()))]
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        debug!("Loading feature store from {}", path.display());

        let file = File::open(path).context("failed to open feature file")?;
        // SAFETY: mmap requires the backing file to stay immutable for the
        // mapping's lifetime. The `file` handle is dropped once the mapping is
        // established (the kernel keeps the mapping valid regardless), so the
        // real contract is on the caller: the file at `path` must not be
        // truncated or modified while this store is alive. We never write to
        // the mapped region.
        let mmap = Arc::new(unsafe {
            MmapOptions::new()
                .map(&file)
                .context("failed to mmap feature file")?
        });

        // Validate header
        anyhow::ensure!(
            mmap.len() >= HEADER_SIZE,
            "feature file too small: {} bytes",
            mmap.len()
        );

        // Check magic
        anyhow::ensure!(
            &mmap[0..8] == MAGIC,
            "invalid feature file format (bad magic)"
        );

        // Read metadata
        let num_nodes_u64 = u64::from_le_bytes(mmap[8..16].try_into()?);
        let feature_dim_u64 = u64::from_le_bytes(mmap[16..24].try_into()?);
        let data_offset_u64 = u64::from_le_bytes(mmap[24..32].try_into()?);

        // Sanity check: prevent OOM from malicious headers
        const MAX_NODES: u64 = 10_000_000_000; // 10B nodes max
        const MAX_DIM: u64 = 100_000; // 100K feature dim max
        anyhow::ensure!(
            num_nodes_u64 <= MAX_NODES,
            "num_nodes {} exceeds maximum {}",
            num_nodes_u64,
            MAX_NODES
        );
        anyhow::ensure!(
            feature_dim_u64 <= MAX_DIM,
            "feature_dim {} exceeds maximum {}",
            feature_dim_u64,
            MAX_DIM
        );
        // The dtype tag lives at byte 32, so the payload must start past it.
        anyhow::ensure!(
            data_offset_u64 > HEADER_SIZE as u64,
            "invalid data_offset {} (must be > {})",
            data_offset_u64,
            HEADER_SIZE
        );
        let data_offset =
            usize::try_from(data_offset_u64).context("data_offset does not fit in usize")?;
        anyhow::ensure!(
            data_offset <= mmap.len(),
            "invalid data_offset {} for file size {}",
            data_offset,
            mmap.len()
        );
        anyhow::ensure!(
            data_offset % std::mem::align_of::<f32>() == 0,
            "invalid data_offset {} (must be {}-byte aligned)",
            data_offset,
            std::mem::align_of::<f32>()
        );

        // Read the dtype tag at byte 32 (first byte of the padding region).
        // In bounds: data_offset > HEADER_SIZE and data_offset <= mmap.len().
        let dtype = FeatureDtype::from_u8(mmap[HEADER_SIZE])?;

        // Validate data size entirely in u64 first, so a 32-bit usize can't
        // truncate the product before it's range-checked. Cast to usize only
        // after the mapped file is known large enough to hold the payload.
        let expected_data_size_u64 = num_nodes_u64
            .checked_mul(feature_dim_u64)
            .and_then(|n| n.checked_mul(dtype.element_size() as u64))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "feature data size overflow: {} nodes * {} dims * {} bytes",
                    num_nodes_u64,
                    feature_dim_u64,
                    dtype.element_size()
                )
            })?;
        let actual_data_size = (mmap.len() - data_offset) as u64;
        anyhow::ensure!(
            actual_data_size >= expected_data_size_u64,
            "feature file truncated: expected {} bytes of data, got {}",
            expected_data_size_u64,
            actual_data_size
        );
        let feature_data_end = (data_offset as u64)
            .checked_add(expected_data_size_u64)
            .ok_or_else(|| anyhow::anyhow!("feature_data_end overflow"))?;
        anyhow::ensure!(
            feature_data_end <= mmap.len() as u64,
            "feature payload end {} exceeds file size {}",
            feature_data_end,
            mmap.len()
        );

        let num_nodes =
            usize::try_from(num_nodes_u64).context("num_nodes does not fit in usize")?;
        let feature_dim =
            usize::try_from(feature_dim_u64).context("feature_dim does not fit in usize")?;
        let expected_data_size = usize::try_from(expected_data_size_u64)
            .context("feature data size does not fit in usize")?;

        debug!(
            "Feature store loaded: {} nodes, {} dims, dtype={:?}, offset={}, {:.2} GB",
            num_nodes,
            feature_dim,
            dtype,
            data_offset,
            mmap.len() as f64 / 1e9
        );

        Ok(Self {
            mmap,
            num_nodes,
            feature_dim,
            features_start_offset: data_offset,
            feature_data_len_bytes: expected_data_size,
            dtype,
            telemetry: None,
        })
    }

    /// Raw bytes of the feature payload.
    #[inline(always)]
    fn feature_data_raw(&self) -> &[u8] {
        let end = self.features_start_offset + self.feature_data_len_bytes;
        &self.mmap[self.features_start_offset..end]
    }

    /// Get the raw feature data as f32 slice (zero-copy, f32 dtype only).
    #[inline(always)]
    fn feature_data(&self) -> &[f32] {
        debug_assert!(
            self.dtype == FeatureDtype::F32,
            "feature_data() called on non-f32 store"
        );
        bytemuck::cast_slice(self.feature_data_raw())
    }

    /// Get number of nodes
    #[inline(always)]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get feature dimension
    #[inline(always)]
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Get the element data type.
    #[inline(always)]
    pub fn dtype(&self) -> FeatureDtype {
        self.dtype
    }

    /// Enable telemetry tracking.
    ///
    /// Call this before operations to track performance metrics.
    pub fn with_telemetry(mut self) -> Self {
        self.telemetry = Some(Arc::new(TelemetryCells::default()));
        self
    }

    /// Get telemetry statistics.
    pub fn telemetry(&self) -> Option<FeatureLoadTelemetry> {
        self.telemetry.as_ref().map(|t| t.snapshot())
    }

    /// Print telemetry summary.
    pub fn print_telemetry(&self) {
        if let Some(ref telemetry) = self.telemetry {
            telemetry.snapshot().print_summary();
        } else {
            debug!("Telemetry not enabled. Call with_telemetry() to enable.");
        }
    }

    /// Get features for a single node (zero-copy slice).
    ///
    /// # Performance
    /// - O(1) pointer arithmetic
    /// - Returns a slice into mmap (no allocation)
    /// - ~100ns if in cache, ~10μs if on NVMe
    #[inline(always)]
    pub fn get(&self, node: NodeId) -> Result<&[f32]> {
        anyhow::ensure!(
            self.dtype == FeatureDtype::F32,
            "get() is not supported for F16 features -- use get_batch() or get_into() instead"
        );

        // Clock reads cost ~20-30ns on a ~100ns documented hot path —
        // only pay for them when telemetry is enabled.
        let start = self.telemetry.is_some().then(Instant::now);

        let node_idx = node as usize;
        anyhow::ensure!(
            node_idx < self.num_nodes,
            "node {} out of bounds (num_nodes={})",
            node,
            self.num_nodes
        );

        let data = self.feature_data();
        let start_idx = node_idx.checked_mul(self.feature_dim).ok_or_else(|| {
            anyhow::anyhow!(
                "index overflow: node {} * feature_dim {}",
                node_idx,
                self.feature_dim
            )
        })?;
        let end_idx = start_idx.checked_add(self.feature_dim).ok_or_else(|| {
            anyhow::anyhow!(
                "index overflow: start {} + feature_dim {}",
                start_idx,
                self.feature_dim
            )
        })?;
        anyhow::ensure!(
            end_idx <= data.len(),
            "feature data bounds error: end_idx {} > data.len() {}",
            end_idx,
            data.len()
        );
        let features = &data[start_idx..end_idx];

        // Track telemetry
        if let (Some(stats), Some(start)) = (self.telemetry.as_deref(), start) {
            stats.single_gets.fetch_add(1, Ordering::Relaxed);
            stats.total_nodes_loaded.fetch_add(1, Ordering::Relaxed);
            stats
                .total_features_loaded
                .fetch_add(self.feature_dim as u64, Ordering::Relaxed);
            stats.total_bytes_loaded.fetch_add(
                self.feature_dim as u64 * std::mem::size_of::<f32>() as u64,
                Ordering::Relaxed,
            );
            stats
                .get_time_ns
                .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        Ok(features)
    }

    /// Decode a single node's features into a caller-provided buffer.
    ///
    /// Works for both F32 (zero-copy memcpy) and F16 (upcast). `out` must have
    /// length >= `feature_dim`.
    pub fn get_into(&self, node: NodeId, out: &mut [f32]) -> Result<()> {
        let node_idx = node as usize;
        anyhow::ensure!(
            node_idx < self.num_nodes,
            "node {} out of bounds (num_nodes={})",
            node,
            self.num_nodes
        );
        anyhow::ensure!(
            out.len() >= self.feature_dim,
            "output buffer too small: {} < {}",
            out.len(),
            self.feature_dim
        );

        let raw = self.feature_data_raw();
        let elem_size = self.dtype.element_size();
        let row_bytes = self.feature_dim * elem_size;
        let start = node_idx.checked_mul(row_bytes).ok_or_else(|| {
            anyhow::anyhow!(
                "index overflow: node {} * row_bytes {}",
                node_idx,
                row_bytes
            )
        })?;
        let end = start
            .checked_add(row_bytes)
            .ok_or_else(|| anyhow::anyhow!("index overflow: {} + {}", start, row_bytes))?;
        anyhow::ensure!(
            end <= raw.len(),
            "feature data bounds error: end {} > len {}",
            end,
            raw.len()
        );
        let row = &raw[start..end];

        match self.dtype {
            FeatureDtype::F32 => {
                let src: &[f32] = bytemuck::cast_slice(row);
                out[..self.feature_dim].copy_from_slice(src);
            }
            FeatureDtype::F16 => {
                for (i, chunk) in row.chunks_exact(2).enumerate() {
                    out[i] = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                }
            }
        }
        Ok(())
    }

    /// Get features for multiple nodes in batch (optimized).
    ///
    /// # Performance
    /// - Vectorized loads (better cache utilization)
    /// - ~1-10ms for 1000 nodes depending on cache hits
    #[inline]
    pub fn get_batch<T>(&self, nodes: &[T]) -> Result<Vec<f32>>
    where
        T: Copy + TryInto<NodeId>,
        <T as TryInto<NodeId>>::Error: std::fmt::Debug,
    {
        let start = self.telemetry.is_some().then(Instant::now);
        trace!("Batch loading {} node features", nodes.len());

        let mut result = Vec::with_capacity(nodes.len() * self.feature_dim);
        let raw = self.feature_data_raw();
        let elem_size = self.dtype.element_size();
        let row_bytes = self.feature_dim * elem_size;

        // Hot path: inline loop
        for &node in nodes {
            let node_id: NodeId = node
                .try_into()
                .map_err(|e| anyhow::anyhow!("invalid node id: {:?}", e))?;
            let node_idx = node_id as usize;

            anyhow::ensure!(node_idx < self.num_nodes, "node {} out of bounds", node_id);

            let byte_start = node_idx.checked_mul(row_bytes).ok_or_else(|| {
                anyhow::anyhow!(
                    "index overflow in batch: node {} * row_bytes {}",
                    node_idx,
                    row_bytes
                )
            })?;
            let byte_end = byte_start.checked_add(row_bytes).ok_or_else(|| {
                anyhow::anyhow!("index overflow in batch: {} + {}", byte_start, row_bytes)
            })?;
            anyhow::ensure!(
                byte_end <= raw.len(),
                "batch bounds error: end {} > len {}",
                byte_end,
                raw.len()
            );
            let row = &raw[byte_start..byte_end];

            match self.dtype {
                FeatureDtype::F32 => {
                    result.extend_from_slice(bytemuck::cast_slice(row));
                }
                FeatureDtype::F16 => {
                    for chunk in row.chunks_exact(2) {
                        result.push(f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
                    }
                }
            }
        }

        // Track telemetry (only if enabled)
        if let (Some(stats), Some(start)) = (self.telemetry.as_deref(), start) {
            stats.batch_gets.fetch_add(1, Ordering::Relaxed);
            stats
                .total_nodes_loaded
                .fetch_add(nodes.len() as u64, Ordering::Relaxed);
            stats
                .total_features_loaded
                .fetch_add((nodes.len() * self.feature_dim) as u64, Ordering::Relaxed);
            stats.total_bytes_loaded.fetch_add(
                result.len() as u64 * std::mem::size_of::<f32>() as u64,
                Ordering::Relaxed,
            );
            stats
                .batch_time_ns
                .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        Ok(result)
    }

    /// Gather features for multiple nodes into a caller-provided buffer.
    ///
    /// Writes node `i`'s feature row into `out[i * feature_dim .. (i+1) * feature_dim]`,
    /// so the result is packed contiguously in `nodes` order. `out` must have
    /// length >= `nodes.len() * feature_dim`. Works for both F32 (memcpy) and
    /// F16 (upcast). Avoids the per-call `Vec` that [`get_batch`] allocates.
    pub fn get_batch_into(&self, nodes: &[NodeId], out: &mut [f32]) -> Result<()> {
        let total = nodes
            .len()
            .checked_mul(self.feature_dim)
            .ok_or_else(|| anyhow::anyhow!("nodes.len() * feature_dim overflows usize"))?;
        anyhow::ensure!(
            out.len() >= total,
            "output buffer too small: {} < {}",
            out.len(),
            total
        );

        let start = self.telemetry.is_some().then(Instant::now);

        for (i, &node) in nodes.iter().enumerate() {
            // i * feature_dim is bounded by `total`, already checked above.
            let row_start = i * self.feature_dim;
            self.get_into(node, &mut out[row_start..row_start + self.feature_dim])?;
        }

        if let (Some(stats), Some(start)) = (self.telemetry.as_deref(), start) {
            stats.batch_gets.fetch_add(1, Ordering::Relaxed);
            stats
                .total_nodes_loaded
                .fetch_add(nodes.len() as u64, Ordering::Relaxed);
            stats
                .total_features_loaded
                .fetch_add(total as u64, Ordering::Relaxed);
            stats.total_bytes_loaded.fetch_add(
                total as u64 * std::mem::size_of::<f32>() as u64,
                Ordering::Relaxed,
            );
            stats
                .batch_time_ns
                .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Get all features as a slice (zero-copy, f32 only).
    ///
    /// Errors for F16 stores — the payload is half-precision and cannot
    /// be viewed as `&[f32]`; use `get_batch()` (which upcasts) instead.
    #[inline(always)]
    pub fn features(&self) -> Result<&[f32]> {
        anyhow::ensure!(
            self.dtype == FeatureDtype::F32,
            "features() is not supported for F16 stores -- the raw payload is not f32. \
             Use get_batch() to upcast."
        );
        let data = self.feature_data();
        let len = self
            .num_nodes
            .checked_mul(self.feature_dim)
            .ok_or_else(|| anyhow::anyhow!("num_nodes * feature_dim overflows"))?;
        anyhow::ensure!(
            len == data.len(),
            "feature data length mismatch: header says {} values, payload has {}",
            len,
            data.len()
        );
        Ok(data)
    }
}

impl super::NodeFeatureSource for FeatureStore {
    fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    fn read_node(&self, node: NodeId, out: &mut [f32]) -> bool {
        self.get_into(node, out).is_ok()
    }
}

/// Save features to file.
///
/// # Arguments
/// - `path`: Output file path
/// - `features`: Flat array in row-major order [node0_f0, ..., node1_f0, ...]
/// - `num_nodes`: Number of nodes
/// - `feature_dim`: Feature dimension
///
/// # Performance
/// - Single write, no fragmentation
/// - ~1-2 GB/s write throughput
pub fn save_features(
    path: impl AsRef<Path>,
    features: Vec<f32>,
    num_nodes: usize,
    feature_dim: usize,
) -> Result<()> {
    let path = path.as_ref();
    debug!(
        "Saving features to {}: {} nodes, {} dims",
        path.display(),
        num_nodes,
        feature_dim
    );

    let expected = num_nodes
        .checked_mul(feature_dim)
        .ok_or_else(|| anyhow::anyhow!("num_nodes * feature_dim overflows usize"))?;
    anyhow::ensure!(
        features.len() == expected,
        "feature array size mismatch: expected {}, got {}",
        expected,
        features.len()
    );

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .context("failed to create feature file")?;

    // Write header.
    // New files store an explicit, aligned payload offset to unlock O_DIRECT/IOPOLL paths.
    let data_offset = ALIGNED_DATA_OFFSET;
    anyhow::ensure!(
        data_offset >= HEADER_SIZE as u64,
        "invalid data_offset {}",
        data_offset
    );
    file.write_all(MAGIC)?;
    file.write_all(&(num_nodes as u64).to_le_bytes())?;
    file.write_all(&(feature_dim as u64).to_le_bytes())?;
    file.write_all(&data_offset.to_le_bytes())?;

    // Pad header to payload offset. Byte 0 of padding = dtype tag (0 = F32).
    let padding = data_offset as usize - HEADER_SIZE;
    if padding > 0 {
        let mut pad = vec![0u8; padding];
        pad[0] = FeatureDtype::F32 as u8;
        file.write_all(&pad)?;
    }

    // Write raw f32 data
    let feature_bytes: &[u8] = bytemuck::cast_slice(&features);
    file.write_all(feature_bytes)?;
    file.sync_all().context("failed to sync feature file")?;

    let file_size = data_offset as usize + feature_bytes.len();
    debug!("Features saved ({:.2} GB)", file_size as f64 / 1e9);
    Ok(())
}

/// Save features as f16 (half precision).
///
/// Accepts f32 input and converts to f16 on write. 2x I/O throughput improvement
/// with <0.5% accuracy loss for typical GNN embeddings.
pub fn save_features_f16(
    path: impl AsRef<Path>,
    features: &[f32],
    num_nodes: usize,
    feature_dim: usize,
) -> Result<()> {
    let path = path.as_ref();
    debug!(
        "Saving f16 features to {}: {} nodes, {} dims",
        path.display(),
        num_nodes,
        feature_dim
    );

    let expected = num_nodes
        .checked_mul(feature_dim)
        .ok_or_else(|| anyhow::anyhow!("num_nodes * feature_dim overflows usize"))?;
    anyhow::ensure!(
        features.len() == expected,
        "feature array size mismatch: expected {}, got {}",
        expected,
        features.len()
    );

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .context("failed to create feature file")?;

    let data_offset = ALIGNED_DATA_OFFSET;
    file.write_all(MAGIC)?;
    file.write_all(&(num_nodes as u64).to_le_bytes())?;
    file.write_all(&(feature_dim as u64).to_le_bytes())?;
    file.write_all(&data_offset.to_le_bytes())?;

    // Pad header to payload offset. Byte 0 of padding = dtype tag (1 = F16).
    let padding = data_offset as usize - HEADER_SIZE;
    let mut pad = vec![0u8; padding];
    pad[0] = FeatureDtype::F16 as u8;
    file.write_all(&pad)?;

    // Convert f32 -> f16 and write
    let mut buf = Vec::with_capacity(features.len() * 2);
    for &v in features {
        buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    file.write_all(&buf)?;
    file.sync_all().context("failed to sync feature file")?;

    let file_size = data_offset as usize + buf.len();
    debug!("F16 features saved ({:.2} GB)", file_size as f64 / 1e9);
    Ok(())
}

/// Create an empty feature file for incremental writing.
///
/// Returns mutable FeatureData that can be written node-by-node,
/// then saved with `save_feature_data()`.
pub fn create_features(num_nodes: usize, feature_dim: usize) -> FeatureData {
    FeatureData::new(num_nodes, feature_dim)
}

/// Save pre-constructed FeatureData to file.
pub fn save_feature_data(path: impl AsRef<Path>, data: &FeatureData) -> Result<()> {
    save_features(
        path,
        data.features.clone(),
        data.num_nodes as usize,
        data.feature_dim as usize,
    )
}

/// Save features from ndarray (convenience function).
///
/// # Example
/// ```no_run
/// use ndarray::Array2;
/// use aethergraph_core::save_features_ndarray;
///
/// let features = Array2::<f32>::zeros((1000, 768));
/// save_features_ndarray("features.bin", features.view())?;
/// # Ok::<_, anyhow::Error>(())
/// ```
pub fn save_features_ndarray<S>(
    path: impl AsRef<Path>,
    features: ndarray::ArrayBase<S, ndarray::Ix2>,
) -> Result<()>
where
    S: ndarray::Data<Elem = f32>,
{
    let (num_nodes, feature_dim) = features.dim();

    // Convert ndarray to contiguous row-major Vec<f32>
    let features_vec: Vec<f32> = if features.is_standard_layout() {
        features.as_slice().unwrap().to_vec()
    } else {
        features.iter().copied().collect()
    };

    save_features(path, features_vec, num_nodes, feature_dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_and_load_features() {
        let num_nodes = 1000;
        let feature_dim = 128;

        // Create synthetic features
        let mut features = Vec::with_capacity(num_nodes * feature_dim);
        for i in 0..num_nodes {
            for j in 0..feature_dim {
                features.push((i * feature_dim + j) as f32);
            }
        }

        // Save
        let temp_file = NamedTempFile::new().unwrap();
        save_features(temp_file.path(), features.clone(), num_nodes, feature_dim).unwrap();

        // Load
        let store = FeatureStore::load(temp_file.path()).unwrap();
        assert_eq!(store.num_nodes(), num_nodes);
        assert_eq!(store.feature_dim(), feature_dim);

        // Verify
        for i in 0..num_nodes {
            let node_features = store.get(i as NodeId).unwrap();
            assert_eq!(node_features.len(), feature_dim);
            for (j, &value) in node_features.iter().enumerate().take(feature_dim) {
                assert_eq!(value, (i * feature_dim + j) as f32);
            }
        }
    }

    #[test]
    fn test_batch_load() {
        let num_nodes = 1000;
        let feature_dim = 64;

        let mut features = vec![0.0; num_nodes * feature_dim];
        for i in 0..num_nodes {
            for j in 0..feature_dim {
                features[i * feature_dim + j] = (i + j) as f32;
            }
        }

        let temp_file = NamedTempFile::new().unwrap();
        save_features(temp_file.path(), features, num_nodes, feature_dim).unwrap();

        let store = FeatureStore::load(temp_file.path()).unwrap();

        // Batch load nodes [0, 10, 100]
        let nodes = vec![0, 10, 100];
        let batch = store.get_batch(&nodes).unwrap();

        assert_eq!(batch.len(), nodes.len() * feature_dim);

        // Verify node 0
        for (j, &value) in batch.iter().enumerate().take(feature_dim) {
            assert_eq!(value, j as f32);
        }

        // Verify node 10
        for j in 0..feature_dim {
            assert_eq!(batch[feature_dim + j], (10 + j) as f32);
        }
    }

    #[test]
    fn test_mutable_feature_data() {
        let mut data = create_features(10, 4);

        // Set features for node 5
        let slot = data.get_mut(5).unwrap();
        slot.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        // Verify
        let features = data.get(5).unwrap();
        assert_eq!(features, &[1.0, 2.0, 3.0, 4.0]);

        // Save and reload
        let temp_file = NamedTempFile::new().unwrap();
        save_feature_data(temp_file.path(), &data).unwrap();

        let store = FeatureStore::load(temp_file.path()).unwrap();
        let loaded = store.get(5).unwrap();
        assert_eq!(loaded, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_large_batch() {
        // Test with 10K nodes to verify performance
        let num_nodes = 10_000;
        let feature_dim = 256;

        let features = vec![1.0; num_nodes * feature_dim];
        let temp_file = NamedTempFile::new().unwrap();
        save_features(temp_file.path(), features, num_nodes, feature_dim).unwrap();

        let store = FeatureStore::load(temp_file.path()).unwrap();

        // Load batch of 1000 nodes
        let nodes: Vec<NodeId> = (0..1000).collect();
        let batch = store.get_batch(&nodes).unwrap();

        assert_eq!(batch.len(), 1000 * feature_dim);
        assert!(batch.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_zero_copy_access() {
        let num_nodes = 100;
        let feature_dim = 16;

        let features = vec![42.0; num_nodes * feature_dim];
        let temp_file = NamedTempFile::new().unwrap();
        save_features(temp_file.path(), features, num_nodes, feature_dim).unwrap();

        let store = FeatureStore::load(temp_file.path()).unwrap();

        // Get features (should be zero-copy slice)
        let node_features = store.get(50).unwrap();
        assert_eq!(node_features.len(), feature_dim);
        assert!(node_features.iter().all(|&x| x == 42.0));

        // All features slice (zero-copy)
        let all_features = store.features().unwrap();
        assert_eq!(all_features.len(), num_nodes * feature_dim);
    }

    #[test]
    fn test_file_format() {
        let num_nodes = 10;
        let feature_dim = 4;
        let features = vec![1.0f32; num_nodes * feature_dim];

        let temp_file = NamedTempFile::new().unwrap();
        save_features(temp_file.path(), features, num_nodes, feature_dim).unwrap();

        // Verify file format manually
        let bytes = std::fs::read(temp_file.path()).unwrap();

        // Check magic
        assert_eq!(&bytes[0..8], b"AETHFEAT");

        // Check num_nodes
        let stored_num_nodes = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(stored_num_nodes, num_nodes as u64);

        // Check feature_dim
        let stored_feature_dim = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        assert_eq!(stored_feature_dim, feature_dim as u64);

        // Check data offset (aligned for new files)
        let stored_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        assert_eq!(stored_offset, ALIGNED_DATA_OFFSET);

        // Check total size
        let expected_size = stored_offset as usize + num_nodes * feature_dim * 4;
        assert_eq!(bytes.len(), expected_size);
    }

    #[test]
    fn test_reject_zero_data_offset() {
        // The payload offset must clear the dtype tag at byte 32; a zero
        // offset is not a valid file.
        let temp_file = NamedTempFile::new().unwrap();
        let features = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(temp_file.path())
            .unwrap();

        file.write_all(MAGIC).unwrap();
        file.write_all(&(2u64).to_le_bytes()).unwrap();
        file.write_all(&(3u64).to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        let feature_bytes: &[u8] = bytemuck::cast_slice(&features);
        file.write_all(feature_bytes).unwrap();
        file.sync_all().unwrap();

        let err = match FeatureStore::load(temp_file.path()) {
            Err(e) => e,
            Ok(_) => panic!("zero data_offset must be rejected"),
        };
        assert!(
            err.to_string().contains("invalid data_offset"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_reject_unaligned_data_offset() {
        let mut bytes = vec![0u8; HEADER_SIZE + 32];
        bytes[0..8].copy_from_slice(MAGIC);
        bytes[8..16].copy_from_slice(&1u64.to_le_bytes()); // 1 node
        bytes[16..24].copy_from_slice(&1u64.to_le_bytes()); // 1 dim
        bytes[24..32].copy_from_slice(&33u64.to_le_bytes()); // unaligned offset

        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), &bytes).unwrap();

        let result = FeatureStore::load(temp_file.path());
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("aligned"));
    }

    #[test]
    fn test_truncated_file() {
        // Create a valid header but truncated data
        let mut bytes = vec![0u8; 64 + 10]; // payload region holds only 10 bytes
        bytes[0..8].copy_from_slice(b"AETHFEAT");
        bytes[8..16].copy_from_slice(&100u64.to_le_bytes()); // 100 nodes
        bytes[16..24].copy_from_slice(&64u64.to_le_bytes()); // 64 dims (needs 25600 bytes)
        bytes[24..32].copy_from_slice(&64u64.to_le_bytes()); // payload starts at 64
        // byte 32 stays 0 → dtype F32

        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), &bytes).unwrap();

        let result = FeatureStore::load(temp_file.path());
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("truncated"));
    }

    #[test]
    fn test_excessive_num_nodes() {
        let mut bytes = vec![0u8; HEADER_SIZE];
        bytes[0..8].copy_from_slice(b"AETHFEAT");
        bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes()); // Absurd num_nodes

        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), &bytes).unwrap();

        let result = FeatureStore::load(temp_file.path());
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("exceeds maximum"));
    }

    #[test]
    fn test_wrong_magic() {
        let mut bytes = vec![0u8; HEADER_SIZE + 100];
        bytes[0..8].copy_from_slice(b"BADMAGIC"); // Wrong magic
        bytes[8..16].copy_from_slice(&10u64.to_le_bytes());
        bytes[16..24].copy_from_slice(&4u64.to_le_bytes());

        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), &bytes).unwrap();

        let result = FeatureStore::load(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_feature_dim() {
        let mut bytes = vec![0u8; HEADER_SIZE];
        bytes[0..8].copy_from_slice(b"AETHFEAT");
        bytes[8..16].copy_from_slice(&10u64.to_le_bytes()); // 10 nodes
        bytes[16..24].copy_from_slice(&0u64.to_le_bytes()); // 0 dims (invalid)

        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), &bytes).unwrap();

        let result = FeatureStore::load(temp_file.path());
        // Zero feature_dim could be considered valid (empty features) or invalid
        // The important thing is it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_empty_file() {
        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), []).unwrap();

        let result = FeatureStore::load(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_overflow_in_data_size() {
        // num_nodes * feature_dim would overflow
        let mut bytes = vec![0u8; HEADER_SIZE];
        bytes[0..8].copy_from_slice(b"AETHFEAT");
        bytes[8..16].copy_from_slice(&(u64::MAX / 2).to_le_bytes()); // huge nodes
        bytes[16..24].copy_from_slice(&1000u64.to_le_bytes()); // dims that cause overflow

        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), &bytes).unwrap();

        let result = FeatureStore::load(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_save_and_load_f16() {
        let num_nodes = 100;
        let feature_dim = 64;

        let mut features = Vec::with_capacity(num_nodes * feature_dim);
        for i in 0..num_nodes {
            for j in 0..feature_dim {
                features.push(((i * feature_dim + j) as f32) * 0.01);
            }
        }

        let temp_file = NamedTempFile::new().unwrap();
        save_features_f16(temp_file.path(), &features, num_nodes, feature_dim).unwrap();

        let store = FeatureStore::load(temp_file.path()).unwrap();
        assert_eq!(store.num_nodes(), num_nodes);
        assert_eq!(store.feature_dim(), feature_dim);
        assert_eq!(store.dtype(), crate::FeatureDtype::F16);

        // get() should error for F16
        assert!(store.get(0).is_err());

        // get_into() should work
        let mut buf = vec![0.0f32; feature_dim];
        store.get_into(0, &mut buf).unwrap();
        for (j, &v) in buf.iter().enumerate() {
            let expected = (j as f32) * 0.01;
            assert!(
                (v - expected).abs() < 0.01,
                "node 0 dim {}: got {}, expected {}",
                j,
                v,
                expected
            );
        }
    }

    #[test]
    fn test_f16_batch_load() {
        let num_nodes = 200;
        let feature_dim = 32;

        let mut features = Vec::with_capacity(num_nodes * feature_dim);
        for i in 0..num_nodes {
            for j in 0..feature_dim {
                features.push((i + j) as f32 * 0.1);
            }
        }

        let temp_file = NamedTempFile::new().unwrap();
        save_features_f16(temp_file.path(), &features, num_nodes, feature_dim).unwrap();

        let store = FeatureStore::load(temp_file.path()).unwrap();
        let nodes = vec![0u32, 10, 100];
        let batch = store.get_batch(&nodes).unwrap();
        assert_eq!(batch.len(), 3 * feature_dim);

        // Verify node 10
        for j in 0..feature_dim {
            let expected = (10 + j) as f32 * 0.1;
            let got = batch[feature_dim + j];
            assert!(
                (got - expected).abs() < 0.1,
                "node 10 dim {}: got {}, expected {}",
                j,
                got,
                expected
            );
        }
    }

    #[test]
    fn test_f32_dtype_tag_loads() {
        // save_features writes dtype tag 0 (F32); load must honor it.
        let num_nodes = 50;
        let feature_dim = 16;
        let features = vec![42.0f32; num_nodes * feature_dim];

        let temp_file = NamedTempFile::new().unwrap();
        save_features(temp_file.path(), features.clone(), num_nodes, feature_dim).unwrap();

        let store = FeatureStore::load(temp_file.path()).unwrap();
        assert_eq!(store.dtype(), crate::FeatureDtype::F32);

        let node_features = store.get(0).unwrap();
        assert_eq!(node_features.len(), feature_dim);
        assert!(node_features.iter().all(|&x| x == 42.0));
    }
}
