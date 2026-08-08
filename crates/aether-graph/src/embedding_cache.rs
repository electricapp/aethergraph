//! Historical embedding cache for GNNAutoScale-style incremental training.
//!
//! Caches intermediate-layer embeddings per node. At each training step,
//! only nodes in the dirty set are recomputed -- the rest use cached values.
//! This turns a 230M-node problem into a ~100K-node problem per batch.

use std::collections::HashMap;

/// Row slot for one computed node.
#[derive(Debug, Clone, Copy)]
struct Slot {
    /// Row index into the flat storage (byte offset = row * dim * 4).
    row: u32,
    /// Epoch when the node was last computed. 0 = allocated but never
    /// written through [`EmbeddingCache::update`].
    generation: u64,
}

/// Per-layer embedding cache.
///
/// Stores one embedding vector per *computed* node for a single GNN
/// layer. Storage is sparse and lazy: memory is allocated per node on
/// first write, so resident size is `touched_nodes × dim × 4` bytes —
/// `num_nodes` is only a validity bound, not an allocation. (A dense
/// `num_nodes × dim` matrix would be ~118 GB per layer at 230M nodes ×
/// 128 dims, allocated up front whether or not a node is ever touched.)
#[derive(Debug)]
pub struct EmbeddingCache {
    /// Flat row storage for computed nodes: `computed × dim` f32 values.
    data: Vec<f32>,
    /// node → row slot. Rows are never freed; a training epoch touches a
    /// bounded working set, so the map tracks distinct touched nodes.
    slots: HashMap<u32, Slot>,
    /// Embedding dimension.
    dim: usize,
    /// Number of addressable nodes (validity bound only).
    num_nodes: usize,
    /// Generation counter; nodes with `slot.generation < generation` are stale.
    generation: u64,
}

impl EmbeddingCache {
    /// Create a cache for up to `num_nodes` nodes with `dim`-dimensional
    /// embeddings. No per-node memory is allocated until a node is
    /// written.
    ///
    /// # Panics
    /// Panics if `dim == 0`.
    pub fn new(num_nodes: usize, dim: usize) -> Self {
        assert!(dim > 0, "embedding dim must be > 0");
        // Generation starts at 1 so slot generation 0 is automatically stale.
        Self {
            data: Vec::new(),
            slots: HashMap::new(),
            dim,
            num_nodes,
            generation: 1,
        }
    }

    /// Row slice for an existing slot.
    #[inline]
    fn row(&self, slot: Slot) -> &[f32] {
        let start = slot.row as usize * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Allocate (or find) the row for `node`. The new row is zeroed and
    /// carries generation 0 until written via [`update`](Self::update).
    fn slot_or_insert(&mut self, node: u32) -> Slot {
        assert!(
            (node as usize) < self.num_nodes,
            "EmbeddingCache: node {node} out of range (num_nodes {})",
            self.num_nodes
        );
        if let Some(&s) = self.slots.get(&node) {
            return s;
        }
        let next_row = self.data.len() / self.dim;
        assert!(
            next_row <= u32::MAX as usize,
            "EmbeddingCache row index {next_row} overflows u32"
        );
        let row = next_row as u32;
        self.data.resize(self.data.len() + self.dim, 0.0);
        let s = Slot { row, generation: 0 };
        self.slots.insert(node, s);
        s
    }

    /// Return cached embedding slice for `node`.
    ///
    /// # Panics
    /// Panics if `node` has never been computed (check
    /// [`is_uninitialized`](Self::is_uninitialized) first).
    #[inline]
    pub fn get(&self, node: u32) -> &[f32] {
        let slot = self
            .slots
            .get(&node)
            .unwrap_or_else(|| panic!("EmbeddingCache::get({node}): node never computed"));
        self.row(*slot)
    }

    /// Return mutable embedding slice for `node`, allocating a zeroed row
    /// on first access (caller writes the new embedding). Does not stamp
    /// the generation — use [`update`](Self::update) for that.
    ///
    /// # Panics
    /// Panics if `node >= num_nodes`.
    #[inline]
    pub fn get_mut(&mut self, node: u32) -> &mut [f32] {
        let slot = self.slot_or_insert(node);
        let start = slot.row as usize * self.dim;
        &mut self.data[start..start + self.dim]
    }

    /// Write embedding and stamp current generation.
    ///
    /// # Panics
    /// Panics if `node >= num_nodes`.
    pub fn update(&mut self, node: u32, embedding: &[f32]) {
        debug_assert_eq!(embedding.len(), self.dim);
        let slot = self.slot_or_insert(node);
        let start = slot.row as usize * self.dim;
        self.data[start..start + self.dim].copy_from_slice(embedding);
        self.slots
            .get_mut(&node)
            .expect("slot just inserted")
            .generation = self.generation;
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

    /// True if node has never had an embedding computed, or its cached
    /// embedding was dropped via [`invalidate`](Self::invalidate).
    #[inline]
    pub fn is_uninitialized(&self, node: u32) -> bool {
        match self.slots.get(&node) {
            Some(s) => s.generation == 0,
            None => true,
        }
    }

    /// Drop `node`'s cached embedding from service: its slot (if any) is
    /// reset to generation 0, so [`is_uninitialized`](Self::is_uninitialized)
    /// reports true until the next [`update`](Self::update). The row
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
    /// `dirty_sorted` MUST be sorted ascending; this is required for the
    /// O(n+m) merge-style scan below. The result preserves `candidates`
    /// order; out-of-range candidates are dropped silently.
    pub fn stale_nodes(&self, candidates: &[u32], dirty_sorted: &[u32]) -> Vec<u32> {
        debug_assert!(
            dirty_sorted.windows(2).all(|w| w[0] <= w[1]),
            "dirty_sorted must be sorted ascending"
        );
        let mut out = Vec::with_capacity(candidates.len());
        for &n in candidates {
            if (n as usize) >= self.num_nodes {
                continue;
            }
            // Stale checks are O(1); avoid the binary search if possible.
            if self.is_stale(n) || dirty_sorted.binary_search(&n).is_ok() {
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
        assert_eq!(cache.get(0), &emb);
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
        stale.sort();
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
        stale.sort();
        assert_eq!(stale, vec![0, 2, 3, 4]);
    }

    #[test]
    fn get_mut_writes() {
        let mut cache = EmbeddingCache::new(5, 3);
        let slot = cache.get_mut(2);
        slot[0] = 7.0;
        slot[1] = 8.0;
        slot[2] = 9.0;
        assert_eq!(cache.get(2), &[7.0, 8.0, 9.0]);
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
        assert!(cache.is_uninitialized(5));
        assert!(!cache.is_uninitialized(0));
    }

    #[test]
    fn invalidate_marks_node_uninitialized_until_update() {
        let mut cache = EmbeddingCache::new(10, 4);
        cache.update(0, &[1.0; 4]);
        assert!(!cache.is_uninitialized(0));
        cache.invalidate(0);
        assert!(cache.is_uninitialized(0));
        cache.update(0, &[2.0; 4]);
        assert!(!cache.is_uninitialized(0));
        // Invalidating a node without a slot is a no-op.
        cache.invalidate(9);
        assert!(cache.is_uninitialized(9));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn update_out_of_range_panics() {
        let mut cache = EmbeddingCache::new(10, 4);
        cache.update(10, &[0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "never computed")]
    fn get_uncomputed_panics() {
        let cache = EmbeddingCache::new(10, 4);
        let _ = cache.get(3);
    }
}
