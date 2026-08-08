//! Feature cache with GPU/CPU/NVMe tiering for billion-scale GNN training.
//!
//! Node features (embeddings) for large graphs can exceed GPU memory. This module
//! implements a three-tier cache:
//! - Hot tier: GPU memory (limited, fast access)
//! - Warm tier: CPU RAM (larger, medium latency)
//! - Cold tier: NVMe SSD (unlimited, high latency)
//!
//! Uses frequency-weighted eviction to keep hot nodes in faster tiers.
//! When warmup frequencies are provided, the hottest nodes are pinned in GPU.

use crate::graph::{Graph, NodeId};
use crate::loader::{NeighborSampler, SamplingConfig};
use ahash::AHashMap;
use anyhow::{Context, Result};
use parking_lot::RwLock;
use rustc_hash::FxHashSet;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use tracing::{debug, trace, warn};

/// Feature dimension
pub type FeatureDim = usize;

/// A node feature vector (e.g., 128-dim embedding)
pub type FeatureVector = Vec<f32>;

/// Configuration for the feature cache
#[derive(Debug, Clone)]
pub struct FeatureCacheConfig {
    /// Maximum number of features in GPU cache
    pub gpu_capacity: usize,

    /// Maximum number of features in CPU cache
    pub cpu_capacity: usize,

    /// Feature dimension
    pub feature_dim: FeatureDim,

    /// Path to NVMe storage for cold features (required, no default)
    pub nvme_path: Option<PathBuf>,

    /// Pre-computed node access frequencies from warmup pass.
    /// If provided, the cache uses frequency-weighted eviction and pins
    /// the hottest nodes in GPU tier.
    pub warmup_frequencies: Option<Arc<AHashMap<NodeId, u32>>>,

    /// Fraction of GPU capacity to pin for hot nodes (default 0.8).
    /// Only used when warmup_frequencies is provided.
    pub pin_ratio: f64,
}

impl Default for FeatureCacheConfig {
    fn default() -> Self {
        Self {
            gpu_capacity: 10_000,
            cpu_capacity: 1_000_000,
            feature_dim: 128,
            nvme_path: None,
            warmup_frequencies: None,
            pin_ratio: 0.8,
        }
    }
}

/// Rows per slab block. Growth appends a block; rows never move.
const SLAB_BLOCK_ROWS: usize = 4096;

/// Block-allocated row storage for one tier.
///
/// Rows live in fixed-size blocks addressed by a u32 row index, with a
/// free list recycling the rows of evicted entries — an owned
/// `Vec<f32>` per cache entry paid a heap allocation on every promote
/// and scattered rows across the allocator.
struct RowSlab {
    blocks: Vec<Vec<f32>>,
    dim: usize,
    rows: u32,
    free: Vec<u32>,
}

impl RowSlab {
    fn new(dim: usize) -> Self {
        Self {
            blocks: Vec::new(),
            dim,
            rows: 0,
            free: Vec::new(),
        }
    }

    fn alloc(&mut self) -> u32 {
        if let Some(r) = self.free.pop() {
            return r;
        }
        let r = self.rows;
        if (r as usize).is_multiple_of(SLAB_BLOCK_ROWS) {
            self.blocks.push(vec![0.0f32; SLAB_BLOCK_ROWS * self.dim]);
        }
        self.rows += 1;
        r
    }

    fn release(&mut self, r: u32) {
        self.free.push(r);
    }

    #[inline]
    fn row(&self, r: u32) -> &[f32] {
        let r = r as usize;
        let start = (r % SLAB_BLOCK_ROWS) * self.dim;
        &self.blocks[r / SLAB_BLOCK_ROWS][start..start + self.dim]
    }

    #[inline]
    fn row_mut(&mut self, r: u32) -> &mut [f32] {
        let r = r as usize;
        let start = (r % SLAB_BLOCK_ROWS) * self.dim;
        &mut self.blocks[r / SLAB_BLOCK_ROWS][start..start + self.dim]
    }
}

/// One cached row plus its runtime hit counter. The counter is atomic so
/// cache hits bump it under a read lock — a hit taking the tier's WRITE
/// lock serialized every concurrent loader on one lock.
struct CacheEntry {
    row: u32,
    hits: AtomicU32,
}

/// Outcome of a tier insert.
enum InsertOutcome {
    /// Stored without displacing anyone.
    Stored,
    /// Stored; the returned victim was evicted and its row copied into
    /// the caller's scratch buffer for demotion.
    StoredEvicting(NodeId),
    /// Not stored — every resident entry is pinned. The caller still owns
    /// the input slice and demotes it to the next tier directly.
    Refused,
}

/// Frequency-weighted cache for a single tier.
///
/// Replaces the old LRU cache. `get()` is O(1), takes `&self`, and is
/// called under a read lock. Eviction pops the lowest-frequency unpinned
/// node from a min-heap. The heap uses lazy deletion -- stale entries are
/// skipped on pop.
struct FreqCache {
    capacity: usize,
    cache: AHashMap<NodeId, CacheEntry>,
    /// Row storage for resident entries.
    slab: RowSlab,
    /// Static frequencies from warmup (set once, never mutated after init)
    warmup_freq: AHashMap<NodeId, u32>,
    /// Pinned nodes (never evicted)
    pinned: FxHashSet<NodeId>,
    /// Min-heap: (frequency, node_id). Lazy deletion on pop.
    eviction_heap: BinaryHeap<Reverse<(u32, NodeId)>>,
}

impl FreqCache {
    fn new(capacity: usize, dim: usize, warmup_freq: AHashMap<NodeId, u32>) -> Self {
        Self {
            capacity,
            cache: AHashMap::with_capacity(capacity),
            slab: RowSlab::new(dim),
            warmup_freq,
            pinned: FxHashSet::default(),
            eviction_heap: BinaryHeap::with_capacity(capacity),
        }
    }

    fn freq(&self, node: NodeId) -> u32 {
        let hits = self
            .cache
            .get(&node)
            .map(|e| e.hits.load(Ordering::Relaxed))
            .unwrap_or(0);
        self.warmup_freq.get(&node).copied().unwrap_or(0) + hits
    }

    /// Shared-access hit path: one map probe, one relaxed counter bump.
    /// The heap is NOT touched here — a hit's raised frequency only makes
    /// a node less likely to be evicted, and `evict_into` reconciles
    /// stale heap values lazily.
    fn get(&self, node: NodeId) -> Option<&[f32]> {
        let entry = self.cache.get(&node)?;
        entry.hits.fetch_add(1, Ordering::Relaxed);
        Some(self.slab.row(entry.row))
    }

    fn insert(
        &mut self,
        node: NodeId,
        features: &[f32],
        evict_out: &mut Vec<f32>,
    ) -> InsertOutcome {
        debug_assert_eq!(features.len(), self.slab.dim);
        if let Some(entry) = self.cache.get_mut(&node) {
            entry.hits.fetch_add(1, Ordering::Relaxed);
            let row = entry.row;
            self.slab.row_mut(row).copy_from_slice(features);
            return InsertOutcome::Stored;
        }

        // Evict if at capacity.
        let victim = if self.cache.len() >= self.capacity {
            match self.evict_into(evict_out) {
                Some(victim) => Some(victim),
                // Nothing is evictable (every entry is pinned). Inserting
                // anyway would push len() past capacity, so refuse the new
                // node; the caller demotes it to the next tier instead.
                None => return InsertOutcome::Refused,
            }
        } else {
            None
        };

        let row = self.slab.alloc();
        self.slab.row_mut(row).copy_from_slice(features);
        self.cache.insert(
            node,
            CacheEntry {
                row,
                hits: AtomicU32::new(0),
            },
        );
        let freq = self.freq(node);
        self.eviction_heap.push(Reverse((freq, node)));

        match victim {
            Some(v) => InsertOutcome::StoredEvicting(v),
            None => InsertOutcome::Stored,
        }
    }

    /// Pop the lowest-frequency unpinned node, copying its row into `out`
    /// (cleared first) and recycling the row. Lazy deletion: skip stale
    /// entries.
    ///
    /// Because hits raise a node's frequency without re-pushing the heap, a
    /// popped entry's recorded frequency can be stale (lower than the node's
    /// true current frequency). When that happens the entry is re-pushed at its
    /// true frequency and popping continues, so the node actually evicted is the
    /// current minimum rather than a stale one.
    ///
    /// Pinned entries popped along the way are simply dropped from the
    /// heap — they are not evictable, and re-pushing them would make this
    /// loop spin forever once every cached node is pinned (each pop would
    /// be matched by a push, and the caller holds the tier's write lock).
    /// `unpin` re-registers the node in the heap. Returns `None` when
    /// nothing is evictable.
    fn evict_into(&mut self, out: &mut Vec<f32>) -> Option<NodeId> {
        while let Some(Reverse((heap_freq, node))) = self.eviction_heap.pop() {
            // Skip if no longer in cache (already evicted)
            if !self.cache.contains_key(&node) {
                continue;
            }
            if self.pinned.contains(&node) {
                continue;
            }
            // Lazy staleness check: if the recorded frequency is below the
            // node's current frequency, this pop is stale. Re-push at the true
            // value and keep looking for the real minimum.
            let current_freq = self.freq(node);
            if heap_freq < current_freq {
                self.eviction_heap.push(Reverse((current_freq, node)));
                continue;
            }
            // Evict this node
            let entry = self.cache.remove(&node)?;
            out.clear();
            out.extend_from_slice(self.slab.row(entry.row));
            self.slab.release(entry.row);
            return Some(node);
        }
        None
    }

    fn pin(&mut self, node: NodeId) {
        self.pinned.insert(node);
    }

    #[allow(dead_code)]
    fn unpin(&mut self, node: NodeId) {
        if self.pinned.remove(&node) && self.cache.contains_key(&node) {
            // The node may have been dropped from the eviction heap while
            // pinned; make it evictable again.
            self.eviction_heap.push(Reverse((self.freq(node), node)));
        }
    }

    fn is_pinned(&self, node: NodeId) -> bool {
        self.pinned.contains(&node)
    }

    fn len(&self) -> usize {
        self.cache.len()
    }
}

/// Single-file NVMe spill tier.
///
/// One sparse `features.dat` holds every spilled node at byte offset
/// `node * record_bytes` — where a file per node paid an open, a close,
/// an inode, and a filesystem-block roundup on every miss and every
/// eviction. Presence is tracked in memory: the tier is a
/// process-lifetime spill, recreated empty at construction (a cache is
/// rebuildable by definition, so nothing is lost across restarts —
/// a cold start just misses).
///
/// Records are written at most with one value per node (features are
/// immutable per node in a training run), so a concurrent same-slot
/// read/write can only race identical bytes.
struct NvmeTier {
    file: std::fs::File,
    record_bytes: u64,
    dim: usize,
    present: RwLock<FxHashSet<NodeId>>,
}

impl NvmeTier {
    fn open(dir: &std::path::Path, dim: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create NVMe cache directory: {}", dir.display()))?;
        let path = dir.join("features.dat");
        // Truncate: presence lives in memory, so stale on-disk slots from
        // a previous process must not survive into this one.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .with_context(|| format!("failed to open NVMe slot file: {}", path.display()))?;
        Ok(Self {
            file,
            record_bytes: (dim * 4) as u64,
            dim,
            present: RwLock::new(FxHashSet::default()),
        })
    }

    /// Read one record, or `None` when the node was never spilled.
    fn load_blocking(&self, node: NodeId) -> Result<Option<FeatureVector>> {
        if !self.present.read().contains(&node) {
            return Ok(None);
        }
        let mut features = vec![0f32; self.dim];
        let offset = node as u64 * self.record_bytes;
        read_exact_at(&self.file, bytemuck::cast_slice_mut(&mut features), offset)
            .with_context(|| format!("failed to read NVMe slot for node {node}"))?;
        Ok(Some(features))
    }

    /// Write one record and mark it present. No fsync: this is a
    /// rebuildable cache, and an fsync per eviction serializes the write
    /// path on device flushes — a crash at worst loses cache entries.
    fn save_blocking(&self, node: NodeId, features: &[f32]) -> Result<()> {
        debug_assert_eq!(features.len(), self.dim);
        let offset = node as u64 * self.record_bytes;
        write_all_at(&self.file, bytemuck::cast_slice(features), offset)
            .with_context(|| format!("failed to write NVMe slot for node {node}"))?;
        self.present.write().insert(node);
        Ok(())
    }
}

#[cfg(unix)]
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(unix)]
fn write_all_at(file: &std::fs::File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &std::fs::File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = file.seek_read(buf, offset)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        offset += n as u64;
    }
    Ok(())
}

#[cfg(windows)]
fn write_all_at(file: &std::fs::File, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = file.seek_write(buf, offset)?;
        buf = &buf[n..];
        offset += n as u64;
    }
    Ok(())
}

/// Three-tier feature cache for GNN training
pub struct FeatureCache {
    config: FeatureCacheConfig,

    /// Cold tier: single-file NVMe slot store.
    nvme: Arc<NvmeTier>,

    /// GPU tier (hot)
    gpu_cache: Arc<RwLock<FreqCache>>,

    /// CPU tier (warm)
    cpu_cache: Arc<RwLock<FreqCache>>,

    /// Stats (atomic counters — the hot path must not take a lock to
    /// bump a hit counter)
    stats: Arc<CacheStatCells>,
}

/// Cache statistics for monitoring
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub gpu_hits: u64,
    pub cpu_hits: u64,
    pub nvme_hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Hits on pinned nodes (measures pin effectiveness)
    pub pinned_hits: u64,
}

/// Internal atomic counters behind [`CacheStats`].
#[derive(Debug, Default)]
struct CacheStatCells {
    gpu_hits: AtomicU64,
    cpu_hits: AtomicU64,
    nvme_hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    pinned_hits: AtomicU64,
}

impl CacheStatCells {
    fn snapshot(&self) -> CacheStats {
        CacheStats {
            gpu_hits: self.gpu_hits.load(Ordering::Relaxed),
            cpu_hits: self.cpu_hits.load(Ordering::Relaxed),
            nvme_hits: self.nvme_hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            pinned_hits: self.pinned_hits.load(Ordering::Relaxed),
        }
    }
}

impl FeatureCache {
    /// Create a new feature cache with the given configuration
    pub async fn new(config: FeatureCacheConfig) -> Result<Self> {
        let nvme_path = config
            .nvme_path
            .clone()
            .ok_or_else(|| anyhow::anyhow!("nvme_path must be configured"))?;

        let dim = config.feature_dim;
        let nvme = {
            let path = nvme_path.clone();
            tokio::task::spawn_blocking(move || NvmeTier::open(&path, dim))
                .await
                .context("task panicked")??
        };

        debug!("Initialized feature cache:");
        debug!("  GPU capacity: {} features", config.gpu_capacity);
        debug!("  CPU capacity: {} features", config.cpu_capacity);
        debug!("  Feature dim: {}", config.feature_dim);
        debug!("  NVMe path: {}", nvme_path.display());

        let warmup = config
            .warmup_frequencies
            .as_ref()
            .map_or_else(AHashMap::new, |m| (**m).clone());

        let mut gpu = FreqCache::new(config.gpu_capacity, dim, warmup.clone());
        let cpu = FreqCache::new(config.cpu_capacity, dim, warmup);

        // Pin hottest nodes in GPU tier when warmup frequencies are provided.
        // At least one slot stays unpinned: a fully-pinned tier has nothing
        // to evict, so every insert past capacity would overflow GPU memory.
        if let Some(ref freq_map) = config.warmup_frequencies {
            let requested = ((config.gpu_capacity as f64) * config.pin_ratio) as usize;
            let pin_count = requested.min(config.gpu_capacity.saturating_sub(1));
            if pin_count < requested {
                warn!(
                    "pin_ratio {} would pin the entire GPU tier ({} slots); \
                     clamping to {} so eviction stays possible",
                    config.pin_ratio, config.gpu_capacity, pin_count
                );
            }
            let mut nodes_by_freq: Vec<_> = freq_map.iter().collect();
            nodes_by_freq.sort_unstable_by(|a, b| b.1.cmp(a.1));
            for &(&node, _) in nodes_by_freq.iter().take(pin_count) {
                gpu.pin(node);
            }
            debug!(
                "  Pinned {} hot nodes in GPU tier (pin_ratio={:.2})",
                pin_count.min(nodes_by_freq.len()),
                config.pin_ratio
            );
        }

        Ok(Self {
            gpu_cache: Arc::new(RwLock::new(gpu)),
            cpu_cache: Arc::new(RwLock::new(cpu)),
            config,
            nvme: Arc::new(nvme),
            stats: Arc::new(CacheStatCells::default()),
        })
    }

    /// Get features for a node, loading from slower tiers if needed.
    pub async fn get(&self, node: NodeId) -> Result<FeatureVector> {
        let mut out = vec![0f32; self.config.feature_dim];
        self.get_into(node, &mut out).await?;
        Ok(out)
    }

    /// Copy features for a node into `out`, loading from slower tiers if
    /// needed. The zero-allocation variant of [`get`](Self::get) for
    /// callers gathering into a preallocated batch buffer.
    pub async fn get_into(&self, node: NodeId, out: &mut [f32]) -> Result<()> {
        debug_assert_eq!(out.len(), self.config.feature_dim);
        // Try GPU cache first (hot tier). Hits take only the READ lock —
        // the frequency bump is an atomic inside the entry — so concurrent
        // loaders hitting the hot tier no longer serialize on one writer
        // lock.
        {
            let gpu = self.gpu_cache.read();
            if let Some(row) = gpu.get(node) {
                out.copy_from_slice(row);
                let pinned = gpu.is_pinned(node);
                drop(gpu);
                self.stats.gpu_hits.fetch_add(1, Ordering::Relaxed);
                if pinned {
                    self.stats.pinned_hits.fetch_add(1, Ordering::Relaxed);
                }
                trace!("GPU cache hit for node {}", node);
                return Ok(());
            }
        }

        // Try CPU cache (warm tier). The row is copied straight into the
        // caller's buffer under the read lock; the promote then reads
        // from that copy, so a warm hit costs exactly one row copy plus
        // the promote's own slab write.
        let cpu_hit = {
            let cpu = self.cpu_cache.read();
            match cpu.get(node) {
                Some(row) => {
                    out.copy_from_slice(row);
                    true
                }
                None => false,
            }
        };
        if cpu_hit {
            self.stats.cpu_hits.fetch_add(1, Ordering::Relaxed);
            trace!("CPU cache hit for node {}", node);
            self.promote_to_gpu(node, out).await;
            return Ok(());
        }

        // Load from NVMe (cold tier).
        //
        // No single-flight guard: concurrent cold misses on the same node each
        // issue their own NVMe read and promote independently. The reads are
        // idempotent (a node's record is immutable once written) and the last
        // promote wins, so the only cost is redundant I/O on a simultaneous
        // miss — an acceptable trade vs. an in-flight future map.
        trace!("Loading node {} from NVMe", node);
        match self.load_from_nvme(node).await? {
            Some(features) => {
                out.copy_from_slice(&features);
                self.stats.nvme_hits.fetch_add(1, Ordering::Relaxed);
                self.promote_to_cpu(node, &features).await;
                Ok(())
            }
            None => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                Err(anyhow::anyhow!(
                    "features for node {node} are in no cache tier"
                ))
            }
        }
    }

    /// Get features for multiple nodes in batch (more efficient than
    /// per-node `get`).
    ///
    /// Each tier's lock is taken once for the whole batch instead of
    /// once per node, duplicate node IDs are resolved once, and only the
    /// distinct NVMe misses fan out as concurrent reads — cached hits
    /// never spawn a task.
    pub async fn get_batch(&self, nodes: &[NodeId]) -> Result<Vec<FeatureVector>> {
        trace!("Batch fetching {} node features", nodes.len());

        let mut resolved: AHashMap<NodeId, FeatureVector> = AHashMap::with_capacity(nodes.len());
        let mut missing: Vec<NodeId> = Vec::new();
        let mut missing_set: FxHashSet<NodeId> = FxHashSet::default();

        // GPU tier: one READ lock for the whole batch (hits don't mutate).
        {
            let gpu = self.gpu_cache.read();
            let mut gpu_hits = 0u64;
            let mut pinned_hits = 0u64;
            for &node in nodes {
                if resolved.contains_key(&node) {
                    continue;
                }
                if let Some(row) = gpu.get(node) {
                    gpu_hits += 1;
                    if gpu.is_pinned(node) {
                        pinned_hits += 1;
                    }
                    resolved.insert(node, row.to_vec());
                } else if missing_set.insert(node) {
                    missing.push(node);
                }
            }
            if gpu_hits > 0 {
                self.stats.gpu_hits.fetch_add(gpu_hits, Ordering::Relaxed);
            }
            if pinned_hits > 0 {
                self.stats
                    .pinned_hits
                    .fetch_add(pinned_hits, Ordering::Relaxed);
            }
        }

        // CPU tier for the remainder.
        let mut cpu_found: Vec<(NodeId, FeatureVector)> = Vec::new();
        if !missing.is_empty() {
            let cpu = self.cpu_cache.read();
            missing.retain(|&node| match cpu.get(node) {
                Some(row) => {
                    cpu_found.push((node, row.to_vec()));
                    false
                }
                None => true,
            });
        }
        if !cpu_found.is_empty() {
            self.stats
                .cpu_hits
                .fetch_add(cpu_found.len() as u64, Ordering::Relaxed);
            for (node, features) in cpu_found {
                self.promote_to_gpu(node, &features).await;
                resolved.insert(node, features);
            }
        }

        // NVMe for the distinct cold misses. A bounded set of chunked
        // blocking tasks rather than one spawn per node: NVME_FANOUT tasks
        // keep the device at a healthy queue depth, each chunk runs as ONE
        // blocking-pool dispatch issuing sequential positional reads, and
        // the spawn/JoinHandle overhead stays constant instead of scaling
        // with the miss count.
        if !missing.is_empty() {
            const NVME_FANOUT: usize = 64;
            let n_tasks = missing.len().min(NVME_FANOUT);
            let per_task = missing.len().div_ceil(n_tasks);
            let fetches: Vec<_> = missing
                .chunks(per_task)
                .map(|chunk| {
                    let tier = Arc::clone(&self.nvme);
                    let chunk = chunk.to_vec();
                    tokio::task::spawn_blocking(move || {
                        chunk
                            .into_iter()
                            .map(|node| {
                                let result = tier.load_blocking(node);
                                (node, result)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for fetch in fetches {
                for (node, result) in fetch.await.context("task panicked")? {
                    match result? {
                        Some(features) => {
                            self.stats.nvme_hits.fetch_add(1, Ordering::Relaxed);
                            self.promote_to_cpu(node, &features).await;
                            resolved.insert(node, features);
                        }
                        None => {
                            self.stats.misses.fetch_add(1, Ordering::Relaxed);
                            return Err(anyhow::anyhow!(
                                "features for node {node} are in no cache tier"
                            ));
                        }
                    }
                }
            }
        }

        // Assemble in input order (duplicates resolved from the map).
        Ok(nodes
            .iter()
            .map(|node| {
                resolved
                    .get(node)
                    .cloned()
                    .expect("every requested node was resolved or errored")
            })
            .collect())
    }

    /// Insert features for a node into the cache
    pub async fn insert(&self, node: NodeId, features: FeatureVector) -> Result<()> {
        // Persist first, then promote — the NVMe record is what makes the
        // node recoverable after it falls out of both memory tiers.
        self.save_to_nvme(node, &features).await?;
        self.promote_to_gpu(node, &features).await;
        Ok(())
    }

    /// Promote features to GPU cache (with eviction if needed)
    async fn promote_to_gpu(&self, node: NodeId, features: &[f32]) {
        let mut evicted = Vec::new();
        let outcome = {
            let mut gpu = self.gpu_cache.write();
            gpu.insert(node, features, &mut evicted)
        };

        match outcome {
            InsertOutcome::Stored => {}
            InsertOutcome::StoredEvicting(victim) => {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                trace!("Evicting node {} from GPU to CPU", victim);
                self.promote_to_cpu(victim, &evicted).await;
            }
            InsertOutcome::Refused => {
                // Every GPU slot is pinned: demote the newcomer directly.
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.promote_to_cpu(node, features).await;
            }
        }
    }

    /// Promote features to CPU cache (with eviction if needed)
    async fn promote_to_cpu(&self, node: NodeId, features: &[f32]) {
        let mut evicted = Vec::new();
        let outcome = {
            let mut cpu = self.cpu_cache.write();
            cpu.insert(node, features, &mut evicted)
        };

        match outcome {
            InsertOutcome::Stored => {}
            InsertOutcome::StoredEvicting(victim) => {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                trace!("Evicting node {} from CPU to NVMe", victim);
                if let Err(e) = self.save_to_nvme(victim, &evicted).await {
                    warn!("Failed to save evicted features to NVMe: {}", e);
                }
            }
            InsertOutcome::Refused => {
                if let Err(e) = self.save_to_nvme(node, features).await {
                    warn!("Failed to save demoted features to NVMe: {}", e);
                }
            }
        }
    }

    /// Load features from the NVMe slot file. `None` means the node was
    /// never spilled.
    async fn load_from_nvme(&self, node: NodeId) -> Result<Option<FeatureVector>> {
        let tier = Arc::clone(&self.nvme);
        tokio::task::spawn_blocking(move || tier.load_blocking(node))
            .await
            .context("task panicked")?
    }

    /// Save features to the NVMe slot file.
    async fn save_to_nvme(&self, node: NodeId, features: &[f32]) -> Result<()> {
        let tier = Arc::clone(&self.nvme);
        let owned = features.to_vec();
        tokio::task::spawn_blocking(move || tier.save_blocking(node, &owned))
            .await
            .context("task panicked")?
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.snapshot()
    }

    /// Print cache statistics
    pub fn print_stats(&self) {
        let stats = self.stats();
        let total_requests = stats.gpu_hits + stats.cpu_hits + stats.nvme_hits + stats.misses;

        if total_requests == 0 {
            debug!("No cache requests yet");
            return;
        }

        let gpu_hit_rate = (stats.gpu_hits as f64 / total_requests as f64) * 100.0;
        let cpu_hit_rate = (stats.cpu_hits as f64 / total_requests as f64) * 100.0;
        let nvme_hit_rate = (stats.nvme_hits as f64 / total_requests as f64) * 100.0;

        debug!("Feature Cache Statistics:");
        debug!("  Total requests: {}", total_requests);
        debug!("  GPU hits:  {} ({:.2}%)", stats.gpu_hits, gpu_hit_rate);
        debug!("  CPU hits:  {} ({:.2}%)", stats.cpu_hits, cpu_hit_rate);
        debug!("  NVMe hits: {} ({:.2}%)", stats.nvme_hits, nvme_hit_rate);
        debug!(
            "  Misses:    {} ({:.2}%)",
            stats.misses,
            (stats.misses as f64 / total_requests as f64) * 100.0
        );
        debug!("  Evictions: {}", stats.evictions);
        debug!("  Pinned hits: {}", stats.pinned_hits);

        let gpu_cache = self.gpu_cache.read();
        let cpu_cache = self.cpu_cache.read();
        debug!(
            "  GPU cache size: {} / {}",
            gpu_cache.len(),
            self.config.gpu_capacity
        );
        debug!(
            "  CPU cache size: {} / {}",
            cpu_cache.len(),
            self.config.cpu_capacity
        );
    }
}

/// Run one epoch of sampling to count node access frequencies.
/// Returns a map from `NodeId` to access count across all batches.
pub fn count_node_frequencies(
    graph: &Graph,
    config: &SamplingConfig,
    epoch_seeds: &[Vec<NodeId>],
) -> AHashMap<NodeId, u32> {
    let mut sampler = NeighborSampler::new(graph, config.clone());
    let mut freq: AHashMap<NodeId, u32> = AHashMap::new();

    for seeds in epoch_seeds {
        let subgraph = sampler.sample_neighbors(seeds);
        for &node in &subgraph.nodes {
            *freq.entry(node).or_insert(0) += 1;
        }
    }

    freq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_feature_cache_basic() {
        let dir = tempfile::tempdir().unwrap();
        let config = FeatureCacheConfig {
            gpu_capacity: 2,
            cpu_capacity: 5,
            feature_dim: 4,
            nvme_path: Some(dir.path().to_path_buf()),
            ..Default::default()
        };

        let cache = FeatureCache::new(config).await.unwrap();

        let features = vec![1.0, 2.0, 3.0, 4.0];
        cache.insert(0, features.clone()).await.unwrap();

        let retrieved = cache.get(0).await.unwrap();
        assert_eq!(retrieved, features);

        let stats = cache.stats();
        assert_eq!(stats.gpu_hits, 1);
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let config = FeatureCacheConfig {
            gpu_capacity: 2,
            cpu_capacity: 5,
            feature_dim: 4,
            nvme_path: Some(dir.path().to_path_buf()),
            ..Default::default()
        };

        let cache = FeatureCache::new(config).await.unwrap();

        for i in 0..3 {
            let features = vec![i as f32; 4];
            cache.insert(i, features).await.unwrap();
        }

        let retrieved = cache.get(0).await.unwrap();
        assert_eq!(retrieved, vec![0.0; 4]);

        let stats = cache.stats();
        assert!(stats.cpu_hits > 0 || stats.nvme_hits > 0);
    }

    #[tokio::test]
    async fn test_batch_get() {
        let dir = tempfile::tempdir().unwrap();
        let config = FeatureCacheConfig {
            gpu_capacity: 10,
            cpu_capacity: 20,
            feature_dim: 4,
            nvme_path: Some(dir.path().to_path_buf()),
            ..Default::default()
        };

        let cache = FeatureCache::new(config).await.unwrap();

        for i in 0..5 {
            let features = vec![i as f32; 4];
            cache.insert(i, features).await.unwrap();
        }

        let nodes = vec![0, 1, 2, 3, 4];
        let batch = cache.get_batch(&nodes).await.unwrap();

        assert_eq!(batch.len(), 5);
        for (i, features) in batch.iter().enumerate() {
            assert_eq!(features[0], i as f32);
        }
    }

    #[tokio::test]
    async fn test_pinned_nodes_not_evicted() {
        let dir = tempfile::tempdir().unwrap();

        let mut warmup = AHashMap::new();
        warmup.insert(0_u32, 100); // node 0 is very hot
        warmup.insert(1, 1); // node 1 is cold

        let config = FeatureCacheConfig {
            gpu_capacity: 2,
            cpu_capacity: 5,
            feature_dim: 4,
            nvme_path: Some(dir.path().to_path_buf()),
            warmup_frequencies: Some(Arc::new(warmup)),
            pin_ratio: 0.5, // pin 1 of 2 GPU slots
        };

        let cache = FeatureCache::new(config).await.unwrap();

        // Insert node 0 (pinned) and node 1
        cache.insert(0, vec![0.0; 4]).await.unwrap();
        cache.insert(1, vec![1.0; 4]).await.unwrap();

        // Insert node 2 -- should evict node 1 (cold, unpinned), not node 0 (pinned)
        cache.insert(2, vec![2.0; 4]).await.unwrap();

        // Node 0 should still be in GPU
        let retrieved = cache.get(0).await.unwrap();
        assert_eq!(retrieved, vec![0.0; 4]);

        let stats = cache.stats();
        assert_eq!(stats.gpu_hits, 1); // node 0 hit in GPU, not demoted
    }

    #[tokio::test]
    async fn fully_pinned_tier_does_not_hang() {
        // pin_ratio 1.0 with enough warmup nodes used to leave the GPU
        // tier 100% pinned; the next insert's eviction loop then spun
        // forever while holding the tier write lock. The pin count is now
        // clamped below capacity, and evict_one terminates regardless.
        let dir = tempfile::tempdir().unwrap();

        let mut warmup = AHashMap::new();
        for n in 0..4u32 {
            warmup.insert(n, 100 - n); // 4 hot nodes for 2 GPU slots
        }

        let config = FeatureCacheConfig {
            gpu_capacity: 2,
            cpu_capacity: 5,
            feature_dim: 4,
            nvme_path: Some(dir.path().to_path_buf()),
            warmup_frequencies: Some(Arc::new(warmup)),
            pin_ratio: 1.0,
        };

        let cache = FeatureCache::new(config).await.unwrap();
        // Fill past GPU capacity with pinned-candidate nodes, then keep
        // inserting. Must complete rather than livelock.
        for n in 0..8u32 {
            tokio::time::timeout(
                std::time::Duration::from_secs(10),
                cache.insert(n, vec![n as f32; 4]),
            )
            .await
            .expect("insert hung — eviction livelock")
            .unwrap();
        }
        for n in 0..8u32 {
            let f = cache.get(n).await.unwrap();
            assert_eq!(f, vec![n as f32; 4]);
        }
    }

    #[tokio::test]
    async fn batch_get_with_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let config = FeatureCacheConfig {
            gpu_capacity: 4,
            cpu_capacity: 8,
            feature_dim: 4,
            nvme_path: Some(dir.path().to_path_buf()),
            ..Default::default()
        };
        let cache = FeatureCache::new(config).await.unwrap();
        for i in 0..3u32 {
            cache.insert(i, vec![i as f32; 4]).await.unwrap();
        }
        // Duplicates and cold nodes mixed; output must match input order.
        let nodes = vec![2, 0, 2, 1, 0];
        let batch = cache.get_batch(&nodes).await.unwrap();
        assert_eq!(batch.len(), nodes.len());
        for (node, features) in nodes.iter().zip(&batch) {
            assert_eq!(features[0], *node as f32);
        }
    }

    #[tokio::test]
    async fn get_into_fills_caller_buffer() {
        let dir = tempfile::tempdir().unwrap();
        let config = FeatureCacheConfig {
            gpu_capacity: 2,
            cpu_capacity: 2,
            feature_dim: 4,
            nvme_path: Some(dir.path().to_path_buf()),
            ..Default::default()
        };
        let cache = FeatureCache::new(config).await.unwrap();
        cache.insert(7, vec![7.0; 4]).await.unwrap();

        let mut out = vec![0f32; 4];
        cache.get_into(7, &mut out).await.unwrap();
        assert_eq!(out, vec![7.0; 4]);
    }

    #[tokio::test]
    async fn spill_tier_is_one_slot_file() {
        let dir = tempfile::tempdir().unwrap();
        let config = FeatureCacheConfig {
            gpu_capacity: 1,
            cpu_capacity: 1,
            feature_dim: 4,
            nvme_path: Some(dir.path().to_path_buf()),
            ..Default::default()
        };
        let cache = FeatureCache::new(config).await.unwrap();

        // Tiny tiers force most nodes all the way down to NVMe.
        for i in 0..16u32 {
            cache.insert(i, vec![i as f32; 4]).await.unwrap();
        }
        for i in 0..16u32 {
            let f = cache.get(i).await.unwrap();
            assert_eq!(f, vec![i as f32; 4], "node {i} corrupted through spill");
        }

        // The whole tier is a single slot file — no per-node files.
        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(entries, vec![std::ffi::OsString::from("features.dat")]);
    }

    #[tokio::test]
    async fn absent_node_is_a_miss_error() {
        let dir = tempfile::tempdir().unwrap();
        let config = FeatureCacheConfig {
            gpu_capacity: 2,
            cpu_capacity: 2,
            feature_dim: 4,
            nvme_path: Some(dir.path().to_path_buf()),
            ..Default::default()
        };
        let cache = FeatureCache::new(config).await.unwrap();
        assert!(cache.get(999).await.is_err());
        assert_eq!(cache.stats().misses, 1);
    }

    #[tokio::test]
    async fn test_pinned_hits_tracked() {
        let dir = tempfile::tempdir().unwrap();

        let mut warmup = AHashMap::new();
        warmup.insert(0_u32, 100);

        let config = FeatureCacheConfig {
            gpu_capacity: 5,
            cpu_capacity: 5,
            feature_dim: 4,
            nvme_path: Some(dir.path().to_path_buf()),
            warmup_frequencies: Some(Arc::new(warmup)),
            pin_ratio: 1.0,
        };

        let cache = FeatureCache::new(config).await.unwrap();
        cache.insert(0, vec![0.0; 4]).await.unwrap();

        // Access pinned node
        let _ = cache.get(0).await.unwrap();
        let _ = cache.get(0).await.unwrap();

        let stats = cache.stats();
        assert_eq!(stats.pinned_hits, 2);
    }
}
