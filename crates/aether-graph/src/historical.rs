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
}

impl HistoricalSampler {
    /// Create a sampler for `num_nodes` with `embedding_dim`-dimensional embeddings.
    pub fn new(num_nodes: usize, embedding_dim: usize) -> Self {
        Self {
            cache: EmbeddingCache::new(num_nodes, embedding_dim),
        }
    }

    /// Determine which nodes need recomputation vs cached lookup.
    ///
    /// `nodes` should include ALL nodes in the sampled subgraph (seeds +
    /// their k-hop neighbors), not just seeds. The caller is responsible
    /// for running the sampler first to get the full node set.
    ///
    /// A node needs recomputation if its cache entry is invalid: never
    /// computed, or invalidated by `advance_epoch` because it received
    /// new edges. An invalidated node stays in need of recomputation
    /// across any number of epochs until a batch actually recomputes and
    /// commits it. Nodes with a valid entry use cached embeddings --
    /// this is the GNNAutoScale approximation: their neighbor structure is
    /// unchanged, so the cached embedding is a valid approximation.
    ///
    /// Invalidation state lives in the cache, populated by `advance_epoch`;
    /// the graph itself is not consulted, so callers should ensure
    /// `advance_epoch` has been invoked at least once for accurate results.
    pub fn prepare_batch(&self, nodes: &[u32]) -> HistoricalBatch {
        let dim = self.cache.dim();
        let mut recompute_nodes = Vec::with_capacity(nodes.len());
        let mut cached_nodes = Vec::with_capacity(nodes.len());
        let mut cached_embeddings = Vec::with_capacity(nodes.len() * dim);

        for &node in nodes {
            if self.cache.is_uninitialized(node) {
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
    /// Call once per epoch — and only for a **single-sampler** setup.
    /// `DynamicGraph::drain_dirty` destructively clears the graph's
    /// bitmap, so with one sampler per GNN layer (the normal
    /// GNNAutoScale arrangement) the first layer to call this would
    /// steal every dirty bit and leave the other layers serving stale
    /// embeddings. Multi-layer training must drain once and fan the
    /// list out via [`advance_epoch_with`](Self::advance_epoch_with):
    ///
    /// ```ignore
    /// let dirty = graph.drain_dirty();
    /// for layer in &mut samplers {
    ///     layer.advance_epoch_with(&dirty);
    /// }
    /// ```
    ///
    /// Each drained node's cache entry is invalidated, so it requires
    /// recomputation — in this epoch or any later one — before its
    /// cached embedding is served again.
    pub fn advance_epoch(&mut self, graph: &DynamicGraph) {
        let dirty = graph.drain_dirty();
        self.advance_epoch_with(&dirty);
    }

    /// Advance using an externally drained dirty list. Non-destructive:
    /// the same list can be fed to every layer's sampler, which is the
    /// correct pattern for multi-layer training (see
    /// [`advance_epoch`](Self::advance_epoch)).
    pub fn advance_epoch_with(&mut self, dirty: &[u32]) {
        // Invalidate rather than remember: a dirty node not recomputed
        // this epoch must still recompute in every later epoch until a
        // commit refreshes it.
        for &node in dirty {
            self.cache.invalidate(node);
        }
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

    fn fill_graph(g: &DynamicGraph, edges: &[(u32, u32)]) {
        let mut w = g.writer_or_panic();
        for &(s, d) in edges {
            w.insert_edge(s, d).unwrap();
        }
    }

    #[test]
    fn historical_prepare_batch() {
        let g = DynamicGraph::new(100, 1024);
        let edges: Vec<_> = (0u32..10).map(|i| (i, i + 1)).collect();
        fill_graph(&g, &edges);

        let sampler = HistoricalSampler::new(100, 4);
        let batch = sampler.prepare_batch(&[0, 1, 2]);

        assert!(!batch.recompute_nodes.is_empty());
        assert!(batch.cached_nodes.is_empty());
    }

    #[test]
    fn historical_commit_and_reuse() {
        let g = DynamicGraph::new(100, 1024);
        fill_graph(&g, &[(0, 1), (1, 2)]);

        let mut sampler = HistoricalSampler::new(100, 4);

        sampler.advance_epoch(&g);
        let batch = sampler.prepare_batch(&[0, 1]);
        assert_eq!(batch.cached_nodes.len(), 0, "first batch: nothing cached");
        let fake_embeddings: Vec<f32> = (0..batch.recompute_nodes.len() * 4)
            .map(|i| i as f32)
            .collect();
        sampler.commit_batch(&batch, &fake_embeddings);

        sampler.advance_epoch(&g);

        let batch2 = sampler.prepare_batch(&[0, 1]);
        assert!(!batch2.cached_nodes.is_empty(), "should have cached nodes");
    }

    #[test]
    fn multi_layer_samplers_share_one_drain() {
        let g = DynamicGraph::new(100, 4096);
        fill_graph(&g, &[(0, 1)]);

        let mut layers = [
            HistoricalSampler::new(100, 4),
            HistoricalSampler::new(100, 4),
        ];
        let dirty = g.drain_dirty();
        for layer in &mut layers {
            layer.advance_epoch_with(&dirty);
        }

        // Compute + commit embeddings for both layers, then dirty node 0
        // again. Every layer must see the invalidation — not just the
        // first one to advance.
        for layer in &mut layers {
            let batch = layer.prepare_batch(&[0, 1]);
            let emb = vec![1.0; batch.recompute_nodes.len() * 4];
            layer.commit_batch(&batch, &emb);
        }

        fill_graph(&g, &[(0, 5)]);
        let dirty = g.drain_dirty();
        for layer in &mut layers {
            layer.advance_epoch_with(&dirty);
        }

        for (i, layer) in layers.iter().enumerate() {
            let batch = layer.prepare_batch(&[0, 1]);
            assert!(
                batch.recompute_nodes.contains(&0),
                "layer {i} missed the node-0 invalidation"
            );
        }
    }

    #[test]
    fn dirty_node_missed_by_an_epoch_still_recomputes_later() {
        let g = DynamicGraph::new(100, 4096);
        fill_graph(&g, &[(0, 1)]);

        let mut sampler = HistoricalSampler::new(100, 4);
        sampler.advance_epoch(&g);
        let batch = sampler.prepare_batch(&[0]);
        let emb = vec![1.0; batch.recompute_nodes.len() * 4];
        sampler.commit_batch(&batch, &emb);

        // Node 0 receives a new edge, then sits out an entire epoch (no
        // batch samples it) before another epoch begins.
        fill_graph(&g, &[(0, 5)]);
        sampler.advance_epoch(&g);
        sampler.advance_epoch(&g);

        let batch2 = sampler.prepare_batch(&[0]);
        assert!(
            batch2.recompute_nodes.contains(&0),
            "unprocessed dirty node must stay invalidated across epochs"
        );
    }

    #[test]
    fn historical_dirty_invalidates() {
        let g = DynamicGraph::new(100, 1024);
        fill_graph(&g, &[(0, 1)]);

        let mut sampler = HistoricalSampler::new(100, 4);
        let batch = sampler.prepare_batch(&[0]);
        let fake_emb = vec![1.0; batch.recompute_nodes.len() * 4];
        sampler.commit_batch(&batch, &fake_emb);

        fill_graph(&g, &[(0, 5)]);
        sampler.advance_epoch(&g);

        let batch2 = sampler.prepare_batch(&[0, 1]);
        assert!(
            batch2.recompute_nodes.contains(&0),
            "dirty node should be recomputed"
        );
    }
}
