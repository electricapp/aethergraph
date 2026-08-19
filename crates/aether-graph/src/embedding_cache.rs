//! Historical embedding cache for GNNAutoScale-style incremental training.
//!
//! Caches intermediate-layer embeddings per node. At each training step,
//! only nodes in the dirty set are recomputed -- the rest use cached values.
//! This turns a 230M-node problem into a ~100K-node problem per batch.

use rustc_hash::{FxHashMap, FxHashSet};

/// Row slot for one computed node.
#[derive(Debug, Clone, Copy)]
struct Slot {
    /// Row index into the row store.
    row: u32,
    /// Epoch when the node was last computed. 0 = allocated but never
    /// written through [`EmbeddingCache::update`].
    generation: u64,
}

/// Contiguous, lazily committed row storage.
///
/// On Unix the full `num_nodes × stride` matrix is *reserved* as one
/// anonymous mapping but committed page-by-page on first touch, so
/// resident memory tracks touched rows while row addressing is a single
/// `base + row * stride` — no per-block indirection in the gather loop,
/// no growth copies, and addresses are stable for the cache's lifetime.
/// Reserving is free: 118 GB of virtual space for 230M × 128 dims costs
/// nothing until rows are written. Non-Unix targets fall back to
/// block-allocated storage with the same interface.
#[derive(Debug)]
struct RowStore {
    #[cfg(unix)]
    base: std::ptr::NonNull<f32>,
    #[cfg(unix)]
    bytes: usize,
    #[cfg(not(unix))]
    blocks: Vec<Vec<f32>>,
    /// Row stride in f32s.
    stride: usize,
}

#[cfg(unix)]
// SAFETY: the mapping is owned by this store; `&self` access hands out
// disjoint row slices under Rust's usual borrow rules.
unsafe impl Send for RowStore {}
#[cfg(unix)]
// SAFETY: see Send.
unsafe impl Sync for RowStore {}

#[cfg(not(unix))]
/// Rows per storage block in the non-Unix fallback.
const BLOCK_ROWS: usize = 4096;

impl RowStore {
    #[cfg(unix)]
    fn new(num_rows: usize, stride: usize) -> Self {
        let bytes = num_rows
            .checked_mul(stride)
            .and_then(|n| n.checked_mul(4))
            .expect("row store size overflows usize")
            .max(4096);
        // SAFETY: anonymous private mapping with no file backing; length is
        // non-zero. MAP_NORESERVE (Linux) skips commit charging for the
        // reservation; macOS overcommits anonymous memory by default.
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | Self::map_noreserve(),
                -1,
                0,
            )
        };
        assert!(
            raw != libc::MAP_FAILED,
            "EmbeddingCache: failed to reserve {bytes} bytes of virtual space"
        );
        Self {
            base: std::ptr::NonNull::new(raw.cast::<f32>()).expect("mmap returned null"),
            bytes,
            stride,
        }
    }

    #[cfg(all(unix, target_os = "linux"))]
    fn map_noreserve() -> libc::c_int {
        libc::MAP_NORESERVE
    }

    #[cfg(all(unix, not(target_os = "linux")))]
    fn map_noreserve() -> libc::c_int {
        0
    }

    #[cfg(not(unix))]
    fn new(_num_rows: usize, stride: usize) -> Self {
        Self {
            blocks: Vec::new(),
            stride,
        }
    }

    /// Make row `r` addressable. On Unix the reservation already covers
    /// every row (the kernel commits the page on first write); the
    /// fallback appends a block when needed.
    #[inline]
    fn ensure_row(&mut self, r: u32) {
        #[cfg(unix)]
        {
            debug_assert!((r as usize + 1) * self.stride * 4 <= self.bytes);
        }
        #[cfg(not(unix))]
        {
            let r = r as usize;
            if r / BLOCK_ROWS == self.blocks.len() {
                self.blocks.push(vec![0.0f32; BLOCK_ROWS * self.stride]);
            }
        }
    }

    #[inline]
    fn row(&self, r: u32, dim: usize) -> &[f32] {
        #[cfg(unix)]
        {
            // SAFETY: `r` was handed out by the cache's allocator, so the
            // offset stays inside the mapping.
            let p = unsafe { self.base.as_ptr().add(r as usize * self.stride) };
            // SAFETY: anonymous pages read as zero before first write, so
            // the slice is always initialized memory.
            unsafe { std::slice::from_raw_parts(p, dim) }
        }
        #[cfg(not(unix))]
        {
            let r = r as usize;
            let start = (r % BLOCK_ROWS) * self.stride;
            &self.blocks[r / BLOCK_ROWS][start..start + dim]
        }
    }

    #[inline]
    fn row_mut(&mut self, r: u32, dim: usize) -> &mut [f32] {
        #[cfg(unix)]
        {
            // SAFETY: `r` was handed out by the cache's allocator, so the
            // offset stays inside the mapping.
            let p = unsafe { self.base.as_ptr().add(r as usize * self.stride) };
            // SAFETY: as `row`, and `&mut self` guarantees exclusivity.
            unsafe { std::slice::from_raw_parts_mut(p, dim) }
        }
        #[cfg(not(unix))]
        {
            let r = r as usize;
            let start = (r % BLOCK_ROWS) * self.stride;
            &mut self.blocks[r / BLOCK_ROWS][start..start + dim]
        }
    }
}

#[cfg(unix)]
impl Drop for RowStore {
    fn drop(&mut self) {
        // SAFETY: base/bytes came from the successful mmap in `new`.
        unsafe {
            libc::munmap(self.base.as_ptr().cast::<libc::c_void>(), self.bytes);
        }
    }
}

/// Per-layer embedding cache.
///
/// Stores one embedding vector per *computed* node for a single GNN
/// layer. Storage is sparse and lazy: the full matrix is reserved as
/// virtual space but committed per page on first write, so resident size
/// is `touched_nodes × stride × 4` bytes — `num_nodes` is only a
/// reservation bound, not an allocation. (An eagerly allocated dense
/// `num_nodes × dim` matrix would be ~118 GB per layer at 230M nodes ×
/// 128 dims whether or not a node is ever touched.)
///
/// The node → row map uses FxHash — this lookup runs once or twice per
/// node per layer per batch, and SipHash on a 4-byte key cost 3-5x more
/// than the whole probe needs. The row stride pads `dim` to a multiple
/// of 16 floats so rows of odd dims don't straddle cache lines.
#[derive(Debug)]
pub struct EmbeddingCache {
    /// Contiguous reserved row storage.
    store: RowStore,
    /// Rows allocated so far.
    rows: usize,
    /// node → row slot. Rows are never freed; a training epoch touches a
    /// bounded working set, so the map tracks distinct touched nodes.
    slots: FxHashMap<u32, Slot>,
    /// Embedding dimension.
    dim: usize,
    /// Number of addressable nodes (bounds the reservation).
    num_nodes: usize,
    /// Generation counter; nodes with `slot.generation < generation` are stale.
    generation: u64,
}

impl EmbeddingCache {
    /// Create a cache for up to `num_nodes` nodes with `dim`-dimensional
    /// embeddings. No per-node memory is committed until a node is
    /// written.
    ///
    /// # Panics
    /// Panics if `dim == 0`.
    pub fn new(num_nodes: usize, dim: usize) -> Self {
        assert!(dim > 0, "embedding dim must be > 0");
        let stride = dim.next_multiple_of(16);
        // Generation starts at 1 so slot generation 0 is automatically stale.
        Self {
            store: RowStore::new(num_nodes.max(1), stride),
            rows: 0,
            slots: FxHashMap::default(),
            dim,
            num_nodes,
            generation: 1,
        }
    }

    /// Allocate the next row. Rows never move; at most one row per node
    /// can be allocated, so the index stays within the reservation.
    #[inline]
    fn alloc_row(store: &mut RowStore, rows: &mut usize) -> u32 {
        assert!(
            *rows <= u32::MAX as usize,
            "EmbeddingCache row index {rows} overflows u32"
        );
        let row = *rows as u32;
        store.ensure_row(row);
        *rows += 1;
        row
    }

    /// Return the cached embedding for `node`, or `None` when the node
    /// was never computed (or was invalidated). One hash probe answers
    /// both "is it usable?" and "what is it?" — the hottest per-node loop
    /// of the training path asks exactly that question.
    #[inline]
    pub fn get_if_computed(&self, node: u32) -> Option<&[f32]> {
        let slot = self.slots.get(&node)?;
        if slot.generation == 0 {
            return None;
        }
        Some(self.store.row(slot.row, self.dim))
    }

    /// Write embedding and stamp current generation. One hash probe.
    ///
    /// # Panics
    /// Panics if `node >= num_nodes`.
    pub fn update(&mut self, node: u32, embedding: &[f32]) {
        debug_assert_eq!(embedding.len(), self.dim);
        assert!(
            (node as usize) < self.num_nodes,
            "EmbeddingCache: node {node} out of range (num_nodes {})",
            self.num_nodes
        );
        let generation = self.generation;
        let row = match self.slots.entry(node) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let s = e.get_mut();
                s.generation = generation;
                s.row
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                let row = Self::alloc_row(&mut self.store, &mut self.rows);
                e.insert(Slot { row, generation });
                row
            }
        };
        self.store.row_mut(row, self.dim).copy_from_slice(embedding);
    }

    /// True if node's cached embedding is from a previous generation.
    ///
    /// Out-of-range and never-computed nodes are reported as stale so
    /// callers iterating over an unfiltered candidate list don't panic —
    /// this matches `stale_nodes`, which skips out-of-range entries
    /// rather than indexing them.
    #[inline]
    pub fn is_stale(&self, node: u32) -> bool {
        match self.slots.get(&node) {
            Some(s) => s.generation < self.generation,
            None => true,
        }
    }

    /// Drop `node`'s cached embedding from service: its slot (if any) is
    /// reset to generation 0, so [`get_if_computed`](Self::get_if_computed)
    /// returns `None` until the next [`update`](Self::update). The row
    /// storage is retained and reused. No-op for nodes without a slot.
    pub fn invalidate(&mut self, node: u32) {
        if let Some(s) = self.slots.get_mut(&node) {
            s.generation = 0;
        }
    }

    /// Increment generation counter (call at epoch boundary).
    pub fn advance_epoch(&mut self) {
        self.generation += 1;
    }

    /// Nodes from `candidates` that are dirty OR stale (gen < current).
    ///
    /// The dirty list is loaded into a hash set once, so the scan is a
    /// genuine O(n + m) — per-candidate binary search over the dirty list
    /// was O(n log m) of random probes. The result preserves `candidates`
    /// order; out-of-range candidates are dropped silently.
    pub fn stale_nodes(&self, candidates: &[u32], dirty: &[u32]) -> Vec<u32> {
        let dirty_set: FxHashSet<u32> = dirty.iter().copied().collect();
        let mut out = Vec::with_capacity(candidates.len());
        for &n in candidates {
            if (n as usize) >= self.num_nodes {
                continue;
            }
            // Stale checks are O(1); consult the set only when needed.
            if self.is_stale(n) || dirty_set.contains(&n) {
                out.push(n);
            }
        }
        out
    }

    /// Embedding dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Current generation counter.
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Number of nodes with allocated rows (memory ≈ this × dim × 4 bytes).
    #[inline]
    pub fn computed_nodes(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_basic() {
        let mut cache = EmbeddingCache::new(10, 4);
        let emb = [1.0, 2.0, 3.0, 4.0];
        cache.update(0, &emb);
        assert_eq!(cache.get_if_computed(0), Some(&emb[..]));
        assert!(!cache.is_stale(0));
    }

    #[test]
    fn cache_stale_after_epoch() {
        let mut cache = EmbeddingCache::new(10, 4);
        cache.update(0, &[1.0, 2.0, 3.0, 4.0]);
        cache.advance_epoch();
        assert!(cache.is_stale(0));
        // Update again marks it fresh
        cache.update(0, &[5.0, 6.0, 7.0, 8.0]);
        assert!(!cache.is_stale(0));
    }

    #[test]
    fn stale_nodes_with_candidates() {
        let mut cache = EmbeddingCache::new(10, 4);
        cache.update(0, &[1.0; 4]);
        cache.update(1, &[2.0; 4]);
        cache.advance_epoch();
        // Check candidates [0, 1, 2] -- 0,1 are stale, 2 is dirty
        let candidates: Vec<u32> = vec![0, 1, 2];
        let dirty = vec![2];
        let mut stale = cache.stale_nodes(&candidates, &dirty);
        stale.sort_unstable();
        assert_eq!(stale, vec![0, 1, 2]);
    }

    #[test]
    fn stale_nodes_partial() {
        let mut cache = EmbeddingCache::new(5, 2);
        for i in 0..5u32 {
            cache.update(i, &[i as f32; 2]);
        }
        cache.advance_epoch();
        cache.update(0, &[10.0; 2]);
        cache.update(1, &[11.0; 2]);
        // Candidates: [0, 2, 3, 4]. 0 is fresh but dirty. 2,3,4 are stale.
        let candidates = vec![0, 2, 3, 4];
        let dirty = vec![0];
        let mut stale = cache.stale_nodes(&candidates, &dirty);
        stale.sort_unstable();
        assert_eq!(stale, vec![0, 2, 3, 4]);
    }

    #[test]
    fn allocation_is_lazy_and_proportional_to_touched_nodes() {
        // A billion-node cache costs nothing until nodes are written.
        let mut cache = EmbeddingCache::new(1_000_000_000, 128);
        assert_eq!(cache.computed_nodes(), 0);
        for n in 0..100u32 {
            cache.update(n * 1_000_000, &[1.0; 128]);
        }
        assert_eq!(cache.computed_nodes(), 100);
        assert!(cache.get_if_computed(5).is_none());
        assert!(cache.get_if_computed(0).is_some());
    }

    #[test]
    fn invalidate_hides_node_until_update() {
        let mut cache = EmbeddingCache::new(10, 4);
        cache.update(0, &[1.0; 4]);
        assert!(cache.get_if_computed(0).is_some());
        cache.invalidate(0);
        assert!(cache.get_if_computed(0).is_none());
        cache.update(0, &[2.0; 4]);
        assert_eq!(cache.get_if_computed(0), Some(&[2.0f32; 4][..]));
        // Invalidating a node without a slot is a no-op.
        cache.invalidate(9);
        assert!(cache.get_if_computed(9).is_none());
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn update_out_of_range_panics() {
        let mut cache = EmbeddingCache::new(10, 4);
        cache.update(10, &[0.0; 4]);
    }

    #[test]
    fn uncomputed_node_reads_none() {
        let cache = EmbeddingCache::new(10, 4);
        assert!(cache.get_if_computed(3).is_none());
    }
}
