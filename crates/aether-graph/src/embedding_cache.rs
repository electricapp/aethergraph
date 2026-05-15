//! Historical embedding cache for GNNAutoScale-style incremental training.
//!
//! Caches intermediate-layer embeddings per node. At each training step,
//! only nodes in the dirty set are recomputed -- the rest use cached values.
//! This turns a 230M-node problem into a ~100K-node problem per batch.

/// Per-layer embedding cache.
///
/// Stores one embedding vector per node for a single GNN layer.
/// Embeddings are f32 row-major: `data[node * dim .. (node+1) * dim]`.
#[derive(Debug)]
pub struct EmbeddingCache {
    /// Flat storage: num_nodes * embedding_dim f32 values.
    data: Vec<f32>,
    /// Embedding dimension.
    dim: usize,
    /// Number of nodes.
    num_nodes: usize,
    /// Generation counter; nodes with `node_generation < generation` are stale.
    generation: u64,
    /// Per-node generation: `node_generation[i]` = epoch when node i was last computed.
    node_generation: Vec<u64>,
}

impl EmbeddingCache {
    /// Allocate a zeroed cache for `num_nodes` with `dim`-dimensional embeddings.
    ///
    /// # Panics
    /// Panics if `dim == 0` or if `num_nodes * dim` overflows `usize`.
    pub fn new(num_nodes: usize, dim: usize) -> Self {
        assert!(dim > 0, "embedding dim must be > 0");
        let total = num_nodes
            .checked_mul(dim)
            .expect("num_nodes * dim overflows usize");
        // Generation starts at 1 so node_generation == 0 is automatically stale.
        Self {
            data: vec![0.0; total],
            dim,
            num_nodes,
            generation: 1,
            node_generation: vec![0; num_nodes],
        }
    }

    /// Return cached embedding slice for `node`.
    #[inline]
    pub fn get(&self, node: u32) -> &[f32] {
        let i = node as usize;
        debug_assert!(i < self.num_nodes);
        let start = i * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Return mutable embedding slice for `node` (caller writes new embedding).
    #[inline]
    pub fn get_mut(&mut self, node: u32) -> &mut [f32] {
        let i = node as usize;
        debug_assert!(i < self.num_nodes);
        let start = i * self.dim;
        &mut self.data[start..start + self.dim]
    }

    /// Write embedding and stamp current generation.
    pub fn update(&mut self, node: u32, embedding: &[f32]) {
        debug_assert_eq!(embedding.len(), self.dim);
        let i = node as usize;
        debug_assert!(i < self.num_nodes);
        let start = i * self.dim;
        self.data[start..start + self.dim].copy_from_slice(embedding);
        self.node_generation[i] = self.generation;
    }

    /// True if node's cached embedding is from a previous generation.
    ///
    /// Out-of-range nodes are reported as stale so callers iterating over an
    /// unfiltered candidate list don't panic — this matches `stale_nodes`,
    /// which skips out-of-range entries rather than indexing them.
    #[inline]
    pub fn is_stale(&self, node: u32) -> bool {
        match self.node_generation.get(node as usize) {
            Some(&g) => g < self.generation,
            None => true,
        }
    }

    /// True if node has never had an embedding computed.
    ///
    /// Out-of-range nodes are reported as uninitialized for the same reason
    /// `is_stale` reports them as stale.
    #[inline]
    pub fn is_uninitialized(&self, node: u32) -> bool {
        match self.node_generation.get(node as usize) {
            Some(&g) => g == 0,
            None => true,
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
}
