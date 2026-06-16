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
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, trace, warn};

/// Monotonic token for unique NVMe temp filenames during atomic save.
static NVME_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

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

/// Frequency-weighted cache for a single tier.
///
/// Replaces the old LRU cache. `get()` is O(1) instead of O(n).
/// Eviction pops the lowest-frequency unpinned node from a min-heap.
/// The heap uses lazy deletion -- stale entries are skipped on pop.
struct FreqCache {
    capacity: usize,
    cache: AHashMap<NodeId, FeatureVector>,
    /// Static frequencies from warmup (set once, never mutated after init)
    warmup_freq: AHashMap<NodeId, u32>,
    /// Runtime access counts
    access_counts: AHashMap<NodeId, u32>,
    /// Pinned nodes (never evicted)
    pinned: FxHashSet<NodeId>,
    /// Min-heap: (frequency, node_id). Lazy deletion on pop.
    eviction_heap: BinaryHeap<Reverse<(u32, NodeId)>>,
}

impl FreqCache {
    fn new(capacity: usize, warmup_freq: AHashMap<NodeId, u32>) -> Self {
        Self {
            capacity,
            cache: AHashMap::with_capacity(capacity),
            warmup_freq,
            access_counts: AHashMap::with_capacity(capacity),
            pinned: FxHashSet::default(),
            eviction_heap: BinaryHeap::with_capacity(capacity),
        }
    }

    fn freq(&self, node: NodeId) -> u32 {
        self.warmup_freq.get(&node).copied().unwrap_or(0)
            + self.access_counts.get(&node).copied().unwrap_or(0)
    }

    fn get(&mut self, node: NodeId) -> Option<&FeatureVector> {
        if self.cache.contains_key(&node) {
            // O(1) on a hit: only bump the access count. The heap is NOT
            // touched here — a hit's raised frequency only makes a node less
            // likely to be evicted, and `evict_one` reconciles stale heap
            // values lazily, so the heap never needs the per-hit push that
            // would otherwise grow it unboundedly.
            *self.access_counts.entry(node).or_insert(0) += 1;
            self.cache.get(&node)
        } else {
            None
        }
    }

    fn insert(&mut self, node: NodeId, features: FeatureVector) -> Option<(NodeId, FeatureVector)> {
        if self.cache.contains_key(&node) {
            *self.access_counts.entry(node).or_insert(0) += 1;
            self.cache.insert(node, features);
            return None;
        }

        // Evict if at capacity.
        let evicted = if self.cache.len() >= self.capacity {
            match self.evict_one() {
                Some(victim) => Some(victim),
                // Nothing is evictable (every entry is pinned). Inserting anyway
                // would push len() past capacity, so refuse the new node and hand
                // it back as the demotion candidate for the next tier instead.
                None => return Some((node, features)),
            }
        } else {
            None
        };

        // Insert new entry
        self.cache.insert(node, features);
        let freq = self.freq(node);
        self.eviction_heap.push(Reverse((freq, node)));

        evicted
    }

    /// Pop lowest-frequency unpinned node. Lazy deletion: skip stale entries.
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
    fn evict_one(&mut self) -> Option<(NodeId, FeatureVector)> {
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
            let features = self.cache.remove(&node)?;
            self.access_counts.remove(&node);
            return Some((node, features));
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

/// Three-tier feature cache for GNN training
pub struct FeatureCache {
    config: FeatureCacheConfig,

    /// Validated NVMe path (extracted from config)
    nvme_path: PathBuf,

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

        tokio::fs::create_dir_all(&nvme_path)
            .await
            .context("failed to create NVMe cache directory")?;

        debug!("Initialized feature cache:");
        debug!("  GPU capacity: {} features", config.gpu_capacity);
        debug!("  CPU capacity: {} features", config.cpu_capacity);
        debug!("  Feature dim: {}", config.feature_dim);
        debug!("  NVMe path: {}", nvme_path.display());

        let warmup = config
            .warmup_frequencies
            .as_ref()
            .map_or_else(AHashMap::new, |m| (**m).clone());

        let mut gpu = FreqCache::new(config.gpu_capacity, warmup.clone());
        let cpu = FreqCache::new(config.cpu_capacity, warmup);

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
            nvme_path,
            stats: Arc::new(CacheStatCells::default()),
        })
    }

    /// Get features for a node, loading from slower tiers if needed
    pub async fn get(&self, node: NodeId) -> Result<FeatureVector> {
        // Try GPU cache first (hot tier)
        {
            let mut gpu = self.gpu_cache.write();
            if let Some(features) = gpu.get(node) {
                let features = features.clone();
                let pinned = gpu.is_pinned(node);
                drop(gpu);
                self.stats.gpu_hits.fetch_add(1, Ordering::Relaxed);
                if pinned {
                    self.stats.pinned_hits.fetch_add(1, Ordering::Relaxed);
                }
                trace!("GPU cache hit for node {}", node);
                return Ok(features);
            }
        }

        // Try CPU cache (warm tier)
        let cpu_features = {
            let mut cpu = self.cpu_cache.write();
            cpu.get(node).cloned()
        };

        if let Some(features) = cpu_features {
            self.stats.cpu_hits.fetch_add(1, Ordering::Relaxed);
            trace!("CPU cache hit for node {}", node);
            self.promote_to_gpu(node, features.clone()).await;
            return Ok(features);
        }

        // Load from NVMe (cold tier).
        //
        // No single-flight guard: concurrent cold misses on the same node each
        // issue their own NVMe read and promote independently. The reads are
        // idempotent (the file is immutable once written) and the last promote
        // wins, so the only cost is redundant I/O on a simultaneous miss — an
        // acceptable trade vs. the complexity of an in-flight future map.
        trace!("Loading node {} from NVMe", node);
        match self.load_from_nvme(node).await {
            Ok(features) => {
                self.stats.nvme_hits.fetch_add(1, Ordering::Relaxed);
                self.promote_to_cpu(node, features.clone()).await;
                Ok(features)
            }
            Err(e) => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                Err(e).with_context(|| format!("failed to load features for node {}", node))
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

        // GPU tier: one lock for the whole batch.
        {
            let mut gpu = self.gpu_cache.write();
            let mut gpu_hits = 0u64;
            let mut pinned_hits = 0u64;
            for &node in nodes {
                if resolved.contains_key(&node) {
                    continue;
                }
                if let Some(features) = gpu.get(node) {
                    let features = features.clone();
                    gpu_hits += 1;
                    if gpu.is_pinned(node) {
                        pinned_hits += 1;
                    }
                    resolved.insert(node, features);
                } else if !missing.contains(&node) {
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
            let mut cpu = self.cpu_cache.write();
            missing.retain(|&node| match cpu.get(node) {
                Some(features) => {
                    cpu_found.push((node, features.clone()));
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
                self.promote_to_gpu(node, features.clone()).await;
                resolved.insert(node, features);
            }
        }

        // NVMe for the distinct cold misses, fetched concurrently.
        if !missing.is_empty() {
            let fetches: Vec<_> = missing
                .iter()
                .map(|&node| {
                    let cache = self.clone_arc();
                    tokio::spawn(async move {
                        let result = cache.load_from_nvme(node).await;
                        (node, result)
                    })
                })
                .collect();
            for fetch in fetches {
                let (node, result) = fetch.await.context("task panicked")?;
                match result {
                    Ok(features) => {
                        self.stats.nvme_hits.fetch_add(1, Ordering::Relaxed);
                        self.promote_to_cpu(node, features.clone()).await;
                        resolved.insert(node, features);
                    }
                    Err(e) => {
                        self.stats.misses.fetch_add(1, Ordering::Relaxed);
                        return Err(e)
                            .with_context(|| format!("failed to load features for node {}", node));
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
        self.promote_to_gpu(node, features.clone()).await;
        self.save_to_nvme(node, &features).await?;
        Ok(())
    }

    /// Promote features to GPU cache (with eviction if needed)
    async fn promote_to_gpu(&self, node: NodeId, features: FeatureVector) {
        let evicted = {
            let mut gpu = self.gpu_cache.write();
            gpu.insert(node, features)
        };

        if let Some((evicted_node, evicted_features)) = evicted {
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            trace!("Evicting node {} from GPU to CPU", evicted_node);
            self.promote_to_cpu(evicted_node, evicted_features).await;
        }
    }

    /// Promote features to CPU cache (with eviction if needed)
    async fn promote_to_cpu(&self, node: NodeId, features: FeatureVector) {
        let evicted = {
            let mut cpu = self.cpu_cache.write();
            cpu.insert(node, features)
        };

        if let Some((evicted_node, evicted_features)) = evicted {
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            trace!("Evicting node {} from CPU to NVMe", evicted_node);
            if let Err(e) = self.save_to_nvme(evicted_node, &evicted_features).await {
                warn!("Failed to save evicted features to NVMe: {}", e);
            }
        }
    }

    /// Load features from NVMe storage
    async fn load_from_nvme(&self, node: NodeId) -> Result<FeatureVector> {
        let path = self.nvme_feature_path(node);

        let mut file = File::open(&path)
            .await
            .with_context(|| format!("failed to open NVMe feature file: {}", path.display()))?;

        // Read straight into a properly-aligned f32 buffer. On little-endian
        // targets the on-disk bytes are already in native order, so this is a
        // bulk copy with no per-element decode.
        let mut features = vec![0f32; self.config.feature_dim];
        file.read_exact(bytemuck::cast_slice_mut(&mut features))
            .await
            .context("failed to read features from NVMe")?;

        Ok(features)
    }

    /// Save features to NVMe storage.
    ///
    /// Writes to a uniquely-named temp file then atomically renames it into
    /// place, so a torn write never leaves a full-length file of garbage that
    /// would read back as silent corruption — a reader sees either the old
    /// contents or the complete new ones.
    async fn save_to_nvme(&self, node: NodeId, features: &[f32]) -> Result<()> {
        let path = self.nvme_feature_path(node);
        // Unique temp name so concurrent writes (even of the same node) don't
        // clobber each other's in-progress file before the rename.
        let token = NVME_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_path = self
            .nvme_path
            .join(format!("node_{}.bin.tmp.{}", node, token));

        // Bulk little-endian copy: f32 is more aligned than u8, so casting the
        // slice to bytes is a plain memcpy with no per-element encode.
        let bytes: &[u8] = bytemuck::cast_slice(features);

        {
            let mut file = File::create(&tmp_path).await.with_context(|| {
                format!("failed to create NVMe temp file: {}", tmp_path.display())
            })?;
            file.write_all(bytes)
                .await
                .context("failed to write features to NVMe")?;
        }

        // Atomic publish. If it fails, drop the temp file rather than leak it.
        if let Err(e) = tokio::fs::rename(&tmp_path, &path).await {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(e).with_context(|| {
                format!("failed to rename NVMe feature file into place: {}", path.display())
            });
        }

        // No fsync: this is a rebuildable cache, and an fsync per inserted
        // or evicted node serializes the write path on device flushes. A
        // crash at worst loses a cache entry, which surfaces as a miss.
        Ok(())
    }

    /// Get NVMe path for a node's features
    fn nvme_feature_path(&self, node: NodeId) -> PathBuf {
        self.nvme_path.join(format!("node_{}.bin", node))
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.snapshot()
    }

    /// Helper to clone Arc references for async tasks
    fn clone_arc(&self) -> Self {
        Self {
            config: self.config.clone(),
            nvme_path: self.nvme_path.clone(),
            gpu_cache: Arc::clone(&self.gpu_cache),
            cpu_cache: Arc::clone(&self.cpu_cache),
            stats: Arc::clone(&self.stats),
        }
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
