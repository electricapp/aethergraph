//! Historical embedding sampler for incremental GNN training on live graphs.
//!
//! Combines DynamicGraph dirty-node tracking with EmbeddingCache to
//! minimize per-batch recomputation. Only nodes that received new edges
//! since the last epoch need fresh embeddings.

use crate::embedding_cache::EmbeddingCache;
use crate::graph::DynamicGraph;

/// A batch prepared for incremental training.
pub struct HistoricalBatch {
    /// Nodes that need fresh embedding computation (dirty or never computed).
    pub recompute_nodes: Vec<u32>,
    /// Nodes that can use cached embeddings.
    pub cached_nodes: Vec<u32>,
    /// Cached embeddings for `cached_nodes` (flattened: len = cached_nodes.len() * dim).
    pub cached_embeddings: Vec<f32>,
}

/// Incremental training state for one GNN layer.
///
/// Wraps an `EmbeddingCache` and provides the batch interface:
/// which nodes need recomputation vs cached lookup.
pub struct HistoricalSampler {
    /// Embedding cache for this layer.
    cache: EmbeddingCache,
    /// Dirty nodes collected at the last epoch boundary.
    dirty_set: Vec<u32>,
}

impl HistoricalSampler {
    /// Create a sampler for `num_nodes` with `embedding_dim`-dimensional embeddings.
    pub fn new(num_nodes: usize, embedding_dim: usize) -> Self {
        Self {
            cache: EmbeddingCache::new(num_nodes, embedding_dim),
            dirty_set: Vec::new(),
        }
    }

    /// Determine which nodes need recomputation vs cached lookup.
    ///
    /// `nodes` should include ALL nodes in the sampled subgraph (seeds +
    /// their k-hop neighbors), not just seeds. The caller is responsible
    /// for running the sampler first to get the full node set.
    ///
    /// A node needs recomputation if it is in the dirty set (received new
    /// edges since the last epoch) or has never been computed.
    /// Nodes computed in a prior epoch but NOT dirty use cached embeddings --
    /// this is the GNNAutoScale approximation: their neighbor structure is
    /// unchanged, so the cached embedding is a valid approximation.
    pub fn prepare_batch(&self, _graph: &DynamicGraph, nodes: &[u32]) -> HistoricalBatch {
        let mut recompute_nodes = Vec::new();
        let mut cached_nodes = Vec::new();
        let mut cached_embeddings = Vec::new();

        for &node in nodes {
            let is_dirty = self.dirty_set.binary_search(&node).is_ok();
            let never_computed = self.cache.is_uninitialized(node);

            if is_dirty || never_computed {
                recompute_nodes.push(node);
            } else {
                cached_nodes.push(node);
                cached_embeddings.extend_from_slice(self.cache.get(node));
            }
        }

        HistoricalBatch {
            recompute_nodes,
            cached_nodes,
            cached_embeddings,
        }
    }

    /// Write computed embeddings back to cache.
    ///
    /// `embeddings` is flattened: `batch.recompute_nodes.len() * dim` floats,
    /// in the same order as `batch.recompute_nodes`.
    pub fn commit_batch(&mut self, batch: &HistoricalBatch, embeddings: &[f32]) {
        let dim = self.cache.dim();
        debug_assert_eq!(embeddings.len(), batch.recompute_nodes.len() * dim);

        for (i, &node) in batch.recompute_nodes.iter().enumerate() {
            let start = i * dim;
            self.cache.update(node, &embeddings[start..start + dim]);
        }
    }

    /// Drain dirty set from graph and advance cache generation.
    ///
    /// Call once per epoch. After this, previously-dirty nodes will be
    /// stale in the cache (requiring recomputation) unless they were
    /// updated in this epoch.
    pub fn advance_epoch(&mut self, graph: &DynamicGraph) {
        self.dirty_set = graph.drain_dirty();
        self.dirty_set.sort_unstable();
        self.cache.advance_epoch();
    }

    /// Access the underlying embedding cache.
    pub fn cache(&self) -> &EmbeddingCache {
        &self.cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_prepare_batch() {
        let g = DynamicGraph::new(100, 1024);
        for i in 0..10 {
            g.insert_edge(i, i + 1).unwrap();
        }

        let sampler = HistoricalSampler::new(100, 4);
        let batch = sampler.prepare_batch(&g, &[0, 1, 2]);

        // All nodes should need recomputation (nothing cached yet)
        assert!(!batch.recompute_nodes.is_empty());
        assert!(batch.cached_nodes.is_empty());
    }

    #[test]
    fn historical_commit_and_reuse() {
        let g = DynamicGraph::new(100, 1024);
        g.insert_edge(0, 1).unwrap();
        g.insert_edge(1, 2).unwrap();

        let mut sampler = HistoricalSampler::new(100, 4);

        // Epoch 1: drain initial dirty set, compute batch
        sampler.advance_epoch(&g);
        let batch = sampler.prepare_batch(&g, &[0, 1]);
        assert_eq!(batch.cached_nodes.len(), 0, "first batch: nothing cached");
        let fake_embeddings: Vec<f32> = (0..batch.recompute_nodes.len() * 4)
            .map(|i| i as f32)
            .collect();
        sampler.commit_batch(&batch, &fake_embeddings);

        // Epoch 2: no new edges -- dirty set is empty
        sampler.advance_epoch(&g);

        // Second batch with same nodes: should reuse cached
        let batch2 = sampler.prepare_batch(&g, &[0, 1]);
        assert!(!batch2.cached_nodes.is_empty(), "should have cached nodes");
    }

    #[test]
    fn historical_dirty_invalidates() {
        let g = DynamicGraph::new(100, 1024);
        g.insert_edge(0, 1).unwrap();

        let mut sampler = HistoricalSampler::new(100, 4);
        let batch = sampler.prepare_batch(&g, &[0]);
        let fake_emb = vec![1.0; batch.recompute_nodes.len() * 4];
        sampler.commit_batch(&batch, &fake_emb);

        // New edge touching node 0
        g.insert_edge(0, 5).unwrap();
        sampler.advance_epoch(&g);

        // Node 0 should need recomputation (dirty), node 1 should be cached
        let batch2 = sampler.prepare_batch(&g, &[0, 1]);
        assert!(
            batch2.recompute_nodes.contains(&0),
            "dirty node should be recomputed"
        );
    }
}
