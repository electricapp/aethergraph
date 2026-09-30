//! Graph reordering for improved cache locality in GNN training.
//!
//! Implements Rabbit Order (Arai et al., IPDPS 2016): modularity-driven
//! incremental aggregation builds a community dendrogram, and a depth-first
//! walk of it emits a permutation that keeps every community — at every
//! level of the hierarchy — contiguous.
//!
//! 1. Degrees and the in-edge transpose are built in parallel; the graph is
//!    treated as undirected (an arc in either direction joins its
//!    endpoints), so directed inputs cluster on both edge directions.
//! 2. Vertices are visited in ascending degree order. Each one aggregates
//!    the edges of everything already merged into it, then merges into the
//!    neighbor community with the largest modularity gain — only while that
//!    gain is positive. Merging stops once no neighbor improves modularity,
//!    so communities do not collapse into connected components.
//! 3. A pre-order walk of the dendrogram (each community, then its merged
//!    children in merge order) emits the permutation.
//!
//! The merge phase is sequential and fully deterministic: the same graph
//! always yields the same permutation.

use crate::graph::csr::{
    EdgeOffset, Graph, GraphValidationMode, NodeId, alloc_hinted, as_atomic, par_fill_rows,
};
use anyhow::Result;
use rayon::prelude::*;
use std::sync::atomic::Ordering;

/// Marks "no vertex" in the dendrogram's child lists. Never a valid node id
/// (`MAX_NODES` keeps ids below `u32::MAX`).
const NONE: u32 = u32::MAX;

/// Rabbit Order dendrogram: each vertex either tops a community or was
/// merged into another.
struct Dendrogram {
    /// `parent[v] == v` for a top-level community; otherwise the community
    /// `v` merged into. Path-compressed as merges are resolved.
    parent: Vec<u32>,
    /// First child merged into each vertex, `NONE` if none.
    first_child: Vec<u32>,
    /// Next child of the same parent, `NONE` at the end of the list.
    next_sibling: Vec<u32>,
}

impl Dendrogram {
    /// Top-level community of `v`, compressing the path behind it.
    fn root(parent: &mut [u32], v: u32) -> u32 {
        let mut r = v;
        while parent[r as usize] != r {
            r = parent[r as usize];
        }
        let mut x = v;
        while parent[x as usize] != r {
            let next = parent[x as usize];
            parent[x as usize] = r;
            x = next;
        }
        r
    }

    /// Pre-order walk: each community, then its children in merge order.
    /// Every vertex belongs to exactly one tree, so the output is a
    /// permutation of `0..n`.
    fn permutation(&self) -> Vec<NodeId> {
        let n = self.parent.len();
        let mut perm = Vec::with_capacity(n);
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        for root in 0..n as u32 {
            if self.parent[root as usize] != root {
                continue;
            }
            perm.push(root);
            stack.push(self.first_child[root as usize]);
            while let Some(top) = stack.last_mut() {
                let c = *top;
                if c == NONE {
                    stack.pop();
                    continue;
                }
                *top = self.next_sibling[c as usize];
                perm.push(c);
                stack.push(self.first_child[c as usize]);
            }
        }
        debug_assert_eq!(perm.len(), n);
        perm
    }

    /// Dense community ids, numbered by each community's first node id.
    fn partitions(&mut self) -> Vec<u32> {
        let n = self.parent.len();
        let mut id_of_root = vec![NONE; n];
        let mut next = 0u32;
        let mut out = Vec::with_capacity(n);
        for v in 0..n as u32 {
            let r = Self::root(&mut self.parent, v) as usize;
            if id_of_root[r] == NONE {
                id_of_root[r] = next;
                next += 1;
            }
            out.push(id_of_root[r]);
        }
        out
    }
}

impl Graph {
    /// The in-edge transpose: `(offsets, sources)` with `sources[offsets[v]
    /// .. offsets[v + 1]]` the sources of arcs into `v`, in ascending
    /// source order.
    fn transpose(&self) -> (Vec<EdgeOffset>, Vec<NodeId>) {
        let n = self.num_nodes();
        let view = self.csr_view();
        let mut offsets: Vec<EdgeOffset> = alloc_hinted(n + 1);
        {
            let counts = as_atomic(&mut offsets[1..]);
            view.edges().par_iter().for_each(|&d| {
                counts[d as usize].fetch_add(1, Ordering::Relaxed);
            });
        }
        for i in 1..=n {
            offsets[i] += offsets[i - 1];
        }
        // Backward scatter: offsets[v + 1] starts as row v's end and each
        // arc claims the slot below it, leaving the row's start there.
        let mut sources: Vec<NodeId> = alloc_hinted(self.num_edges());
        for u in (0..n as NodeId).rev() {
            for &d in view.neighbors(u).iter().rev() {
                offsets[d as usize + 1] -= 1;
                sources[offsets[d as usize + 1] as usize] = u;
            }
        }
        offsets.copy_within(1.., 0);
        offsets[n] = self.num_edges() as EdgeOffset;
        (offsets, sources)
    }

    /// Rabbit Order community detection.
    ///
    /// Needs a homogeneous graph with every destination in range; proves
    /// the latter once (a no-op on an already `Full` graph).
    fn rabbit_dendrogram(&self) -> Result<Dendrogram> {
        anyhow::ensure!(
            self.is_homogeneous(),
            "Rabbit Order needs sources and destinations in one id space \
             (num_nodes {}, destination space {})",
            self.num_nodes(),
            self.num_dst_nodes()
        );
        self.validate_with_mode(GraphValidationMode::Full)?;

        let n = self.num_nodes();
        let mut dendro = Dendrogram {
            parent: (0..n as u32).collect(),
            first_child: vec![NONE; n],
            next_sibling: vec![NONE; n],
        };
        if n <= 1 || self.num_edges() == 0 {
            return Ok(dendro);
        }

        let view = self.csr_view();
        let (in_offsets, in_sources) = self.transpose();
        let in_neighbors = |v: u32| {
            &in_sources[in_offsets[v as usize] as usize..in_offsets[v as usize + 1] as usize]
        };

        // Undirected degree, self-loops excluded. Becomes the community
        // degree as vertices merge.
        let mut degree: Vec<u64> = (0..n as u32)
            .into_par_iter()
            .map(|v| {
                let out = view.neighbors(v).iter().filter(|&&d| d != v).count();
                let inn = in_neighbors(v).iter().filter(|&&s| s != v).count();
                (out + inn) as u64
            })
            .collect();
        // Twice the undirected edge count.
        let two_m: u64 = degree.par_iter().sum();
        if two_m == 0 {
            return Ok(dendro);
        }

        let mut order: Vec<u32> = (0..n as u32).collect();
        order.par_sort_unstable_by_key(|&v| (degree[v as usize], v));

        // Aggregated edge list of a vertex that merged into a community not
        // yet visited; the community absorbs it when its turn comes.
        let mut carried: Vec<Option<Vec<(u32, u64)>>> = vec![None; n];
        let mut visited = vec![false; n];
        let mut weight_to: Vec<u64> = vec![0; n];
        let mut touched: Vec<u32> = Vec::new();

        for &u in &order {
            touched.clear();
            let parent = &mut dendro.parent;
            let mut add = |c: u32, w: u64, touched: &mut Vec<u32>| {
                let c = Dendrogram::root(parent, c);
                if c == u {
                    return;
                }
                if weight_to[c as usize] == 0 {
                    touched.push(c);
                }
                weight_to[c as usize] += w;
            };
            for &v in view.neighbors(u) {
                add(v, 1, &mut touched);
            }
            for &v in in_neighbors(u) {
                add(v, 1, &mut touched);
            }
            let mut child = dendro.first_child[u as usize];
            while child != NONE {
                if let Some(list) = carried[child as usize].take() {
                    for (c, w) in list {
                        add(c, w, &mut touched);
                    }
                }
                child = dendro.next_sibling[child as usize];
            }

            // Modularity gain of joining community c, scaled by 2m²:
            // 2m·w(u,c) − d(u)·d(c). Exact in i128.
            let du = i128::from(degree[u as usize]);
            let mut best: Option<(i128, u32)> = None;
            for &c in &touched {
                let gain = i128::from(two_m) * i128::from(weight_to[c as usize])
                    - du * i128::from(degree[c as usize]);
                let better = match best {
                    None => gain > 0,
                    Some((g, b)) => gain > g || (gain == g && c < b),
                };
                if better {
                    best = Some((gain, c));
                }
            }

            if let Some((_, c)) = best {
                dendro.parent[u as usize] = c;
                degree[c as usize] += degree[u as usize];
                // Prepend: children are walked most-recent-first, which keeps
                // the child list O(1) per merge.
                dendro.next_sibling[u as usize] = dendro.first_child[c as usize];
                dendro.first_child[c as usize] = u;
                if !visited[c as usize] {
                    carried[u as usize] = Some(
                        touched
                            .iter()
                            .map(|&t| (t, weight_to[t as usize]))
                            .collect(),
                    );
                }
            }
            for &t in &touched {
                weight_to[t as usize] = 0;
            }
            visited[u as usize] = true;
        }
        Ok(dendro)
    }

    /// Compute the Rabbit Order permutation for improved cache locality.
    ///
    /// Returns `perm` where `perm[new_id] = old_id`. Errors on a graph whose
    /// destinations fail validation or that is not homogeneous.
    pub fn reorder_rabbit(&self) -> Result<Vec<NodeId>> {
        Ok(self.rabbit_dendrogram()?.permutation())
    }

    /// Compute Rabbit Order community partitions.
    ///
    /// Returns a vector of length `num_nodes` where `partitions[node] =
    /// partition_id`. Partition IDs are dense (0..num_partitions), numbered
    /// by each community's lowest node id. Nodes in the same top-level
    /// community share an ID.
    pub fn rabbit_partitions(&self) -> Result<Vec<u32>> {
        Ok(self.rabbit_dendrogram()?.partitions())
    }

    /// Compute Rabbit Order permutation AND partition assignments in one pass.
    ///
    /// Avoids running community detection twice when both the reordered
    /// graph and cluster-aligned batching are needed.
    pub fn reorder_rabbit_with_partitions(&self) -> Result<(Vec<NodeId>, Vec<u32>)> {
        let mut dendro = self.rabbit_dendrogram()?;
        let perm = dendro.permutation();
        Ok((perm, dendro.partitions()))
    }

    /// Apply a node permutation to produce a new reordered graph.
    ///
    /// `perm[new_id] = old_id`. Proves the source graph `Full` first (a
    /// no-op when it already is): the rebuilt rows are sized from the same
    /// neighbor ranges they are filled from, and every destination is
    /// relabeled through the inverse permutation.
    pub fn permute(&self, perm: &[NodeId]) -> Result<Self> {
        anyhow::ensure!(
            self.is_homogeneous(),
            "permute relabels sources and destinations together; this graph has \
             {} rows and a destination space of {}",
            self.num_nodes(),
            self.num_dst_nodes()
        );
        self.validate_with_mode(GraphValidationMode::Full)?;
        let n = self.num_nodes();
        anyhow::ensure!(perm.len() == n, "permutation length mismatch");

        // The duplicate/range check folds into the inverse-permutation
        // build: NONE marks unassigned slots (new_id < n <= u32::MAX, so it
        // can't collide).
        let mut inv_perm = vec![NONE; n];
        for (new_id, &old_id) in perm.iter().enumerate() {
            anyhow::ensure!((old_id as usize) < n, "out-of-range node {old_id}");
            anyhow::ensure!(inv_perm[old_id as usize] == NONE, "duplicate node {old_id}");
            inv_perm[old_id as usize] = new_id as u32;
        }

        // Row sizes come from `neighbors` — the same guarded range the fill
        // below reads — so a row's slot and its contents always agree.
        let mut new_offsets: Vec<EdgeOffset> = alloc_hinted(n + 1);
        new_offsets[1..]
            .par_iter_mut()
            .enumerate()
            .for_each(|(new_id, slot)| *slot = self.neighbors(perm[new_id]).len() as u64);
        for i in 1..=n {
            new_offsets[i] += new_offsets[i - 1];
        }
        let num_edges = self.num_edges();
        anyhow::ensure!(
            new_offsets[n] == num_edges as EdgeOffset,
            "permuted rows hold {} edges, graph has {num_edges}",
            new_offsets[n]
        );

        let mut new_edges: Vec<NodeId> = alloc_hinted(num_edges);
        par_fill_rows(&mut new_edges, &new_offsets, |new_id, row| {
            for (slot, &old_nbr) in row.iter_mut().zip(self.neighbors(perm[new_id])) {
                *slot = inv_perm[old_nbr as usize];
            }
        });
        let new_weights = self.weights().map(|_| {
            let mut w: Vec<f32> = alloc_hinted(num_edges);
            par_fill_rows(&mut w, &new_offsets, |new_id, row| {
                if let Some(src) = self.neighbor_weights(perm[new_id]) {
                    row.copy_from_slice(src);
                }
            });
            w
        });
        let new_timestamps = self.timestamps().map(|_| {
            let mut ts: Vec<f64> = alloc_hinted(num_edges);
            par_fill_rows(&mut ts, &new_offsets, |new_id, row| {
                if let Some(src) = self.neighbor_timestamps(perm[new_id]) {
                    row.copy_from_slice(src);
                }
            });
            ts
        });

        let mut graph = Self::from_trusted_parts(n, n, new_offsets, new_edges, new_weights);
        if let Some(ts) = new_timestamps {
            graph
                .set_timestamps(ts)
                .map_err(|e| anyhow::anyhow!("permuted timestamps: {e}"))?;
        }
        Ok(graph)
    }
}

/// Group seed nodes by Rabbit partition for cluster-aligned batching.
///
/// Seeds in the same community share neighbors, so grouping them into the
/// same batch makes feature loading nearly sequential (cache hit rates
/// approach 100% after Rabbit reordering).
///
/// Returns batches of seed node IDs, each batch containing at most
/// `batch_size` seeds. Seeds within each batch belong to the same or
/// nearby partitions. Errors on `batch_size == 0` or a seed outside
/// `partitions`.
///
/// If `shuffle_partitions` is true, partitions are visited in random order
/// (seeded by `seed`) to avoid bias from partition numbering. Within each
/// partition, seed order is preserved.
pub fn partition_aligned_batches(
    partitions: &[u32],
    seeds: &[NodeId],
    batch_size: usize,
    shuffle_partitions: bool,
    seed: u64,
) -> Result<Vec<Vec<NodeId>>> {
    anyhow::ensure!(batch_size > 0, "batch_size must be >= 1");

    // Group seeds by partition with one u64 sort — (partition << 32 |
    // original index) keys group by partition and stay stable within it.
    // The single key vec is the whole grouping state: a per-partition Vec
    // map would allocate one heap Vec per distinct community (Rabbit
    // produces many small ones), approaching one allocation per seed each
    // epoch.
    let mut keyed: Vec<u64> = Vec::with_capacity(seeds.len());
    for (i, &s) in seeds.iter().enumerate() {
        let Some(&p) = partitions.get(s as usize) else {
            anyhow::bail!(
                "seed {s} is out of range for partitions of length {}",
                partitions.len()
            );
        };
        keyed.push((u64::from(p) << 32) | i as u64);
    }
    keyed.sort_unstable();

    // Contiguous runs of one partition in the sorted keys.
    let mut runs: Vec<(u32, std::ops::Range<usize>)> = Vec::new();
    let mut run_start = 0usize;
    for i in 1..=keyed.len() {
        if i == keyed.len() || (keyed[i] >> 32) != (keyed[run_start] >> 32) {
            runs.push(((keyed[run_start] >> 32) as u32, run_start..i));
            run_start = i;
        }
    }

    // Optionally shuffle the partition visit order.
    if shuffle_partitions {
        // Fisher-Yates with a simple LCG seeded by `seed`.
        let mut rng = seed.wrapping_mul(0x517c_c1b7_2722_0a95).wrapping_add(1);
        for i in (1..runs.len()).rev() {
            rng = rng.wrapping_mul(0x517c_c1b7_2722_0a95).wrapping_add(1);
            let j = (rng >> 33) as usize % (i + 1);
            runs.swap(i, j);
        }
    }

    // Build batches: fill each batch from one partition at a time.
    // When a partition's seeds don't fill a batch, continue with the next
    // partition (cross-partition batches are rare and only at boundaries).
    let mut batches = Vec::new();
    let mut current_batch: Vec<NodeId> = Vec::with_capacity(batch_size);

    for (_, range) in runs {
        for &key in &keyed[range] {
            current_batch.push(seeds[key as u32 as usize]);
            if current_batch.len() == batch_size {
                batches.push(std::mem::replace(
                    &mut current_batch,
                    Vec::with_capacity(batch_size),
                ));
            }
        }
    }
    if !current_batch.is_empty() {
        batches.push(current_batch);
    }

    Ok(batches)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_cliques() -> Graph {
        let mut edges = Vec::new();
        for i in 0u32..4 {
            for j in 0u32..4 {
                if i != j {
                    edges.push((i, j));
                }
            }
        }
        for i in 4u32..8 {
            for j in 4u32..8 {
                if i != j {
                    edges.push((i, j));
                }
            }
        }
        edges.push((3, 4));
        edges.push((4, 3));
        Graph::from_edges(8, &edges, None).unwrap()
    }

    fn assert_bijection(perm: &[NodeId], n: usize) {
        assert_eq!(perm.len(), n);
        let mut seen = vec![false; n];
        for &id in perm {
            assert!(!seen[id as usize], "{id} appears twice");
            seen[id as usize] = true;
        }
    }

    #[test]
    fn test_permute_identity() {
        let graph = Graph::from_edges(3, &[(0, 1), (0, 2), (1, 2), (2, 0)], None).unwrap();
        let permuted = graph.permute(&[0, 1, 2]).unwrap();
        assert_eq!(permuted.num_nodes(), 3);
        assert_eq!(permuted.num_edges(), graph.num_edges());
        for n in 0..3u32 {
            let mut a: Vec<_> = graph.neighbors(n).to_vec();
            a.sort_unstable();
            let mut b: Vec<_> = permuted.neighbors(n).to_vec();
            b.sort_unstable();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_permute_reverse() {
        let graph = Graph::from_edges(3, &[(0, 1), (1, 2)], None).unwrap();
        let permuted = graph.permute(&[2, 1, 0]).unwrap();
        assert!(permuted.neighbors(2).contains(&1));
        assert!(permuted.neighbors(1).contains(&0));
    }

    #[test]
    fn test_permute_preserves_weights() {
        let graph =
            Graph::from_edges(3, &[(0, 1), (0, 2), (1, 2)], Some(&[1.0, 2.0, 3.0])).unwrap();
        let permuted = graph.permute(&[2, 1, 0]).unwrap();
        assert!(permuted.weights().is_some());
        assert_eq!(permuted.neighbor_weights(2).unwrap(), &[1.0, 2.0]);
        assert_eq!(permuted.neighbor_weights(1).unwrap(), &[3.0]);
    }

    #[test]
    fn test_permute_preserves_timestamps() {
        let mut graph = Graph::from_edges(3, &[(0, 1), (1, 2)], None).unwrap();
        graph.set_timestamps(vec![100.0, 200.0]).unwrap();
        let permuted = graph.permute(&[1, 0, 2]).unwrap();
        assert!((permuted.neighbor_timestamps(1).unwrap()[0] - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_permute_invalid() {
        let graph = Graph::from_edges(3, &[(0, 1)], None).unwrap();
        assert!(graph.permute(&[0, 1]).is_err());
        assert!(graph.permute(&[0, 1, 1]).is_err());
        assert!(graph.permute(&[0, 1, 5]).is_err());
    }

    /// A corrupt file admitted by header-only validation must be rejected by
    /// `permute`, never reach the rebuild: offsets [0, 4, 0, 12] give
    /// guarded degrees summing past the edge count.
    #[test]
    fn test_permute_rejects_corrupt_header_only_graph() {
        let path = tempfile::NamedTempFile::new().unwrap();
        let good = Graph::from_csr_arrays(3, vec![0, 4, 8, 12], vec![0; 12], None).unwrap();
        crate::internal::mmap::save_graph(&good, path.path()).unwrap();
        let mut bytes = std::fs::read(path.path()).unwrap();
        // offsets[2] lives at header (32) + 2 * 8.
        bytes[48..56].copy_from_slice(&0u64.to_le_bytes());
        // Drop the checksum so the body edit is what validation sees.
        bytes[28..32].copy_from_slice(&0u32.to_le_bytes());
        std::fs::write(path.path(), &bytes).unwrap();

        let corrupt =
            crate::internal::mmap::load_graph_mmap(path.path(), GraphValidationMode::HeaderOnly)
                .unwrap();
        assert_eq!(corrupt.validated(), GraphValidationMode::HeaderOnly);
        let err = corrupt.permute(&[0, 1, 2]).unwrap_err().to_string();
        assert!(err.contains("monotonic"), "got: {err}");
        assert!(corrupt.reorder_rabbit().is_err());
    }

    #[test]
    fn test_permute_rejects_bipartite() {
        let g = Graph::from_bipartite_src_dst(2, 5, &[0, 1], &[4, 3], None).unwrap();
        assert!(g.permute(&[1, 0]).is_err());
        assert!(g.reorder_rabbit().is_err());
    }

    #[test]
    fn test_permute_large_matches_reference() {
        let n = 20_000u32;
        let edges: Vec<(u32, u32)> = (0..200_000u64)
            .map(|i| {
                (
                    (i.wrapping_mul(2_654_435_761) % u64::from(n)) as u32,
                    (i.wrapping_mul(97) % u64::from(n)) as u32,
                )
            })
            .collect();
        let w: Vec<f32> = (0..edges.len()).map(|i| i as f32).collect();
        let g = Graph::from_edges(n as usize, &edges, Some(&w)).unwrap();
        let perm: Vec<u32> = (0..n).rev().collect();
        let p = g.permute(&perm).unwrap();
        for new_id in 0..n {
            let old = perm[new_id as usize];
            let expect: Vec<u32> = g.neighbors(old).iter().map(|&x| n - 1 - x).collect();
            assert_eq!(p.neighbors(new_id), &expect[..]);
            assert_eq!(p.neighbor_weights(new_id), g.neighbor_weights(old));
        }
    }

    #[test]
    fn test_rabbit_valid_permutation() {
        let graph = make_two_cliques();
        let perm = graph.reorder_rabbit().unwrap();
        assert_bijection(&perm, 8);
    }

    #[test]
    fn test_rabbit_clusters_communities() {
        let graph = make_two_cliques();
        let perm = graph.reorder_rabbit().unwrap();
        let mut inv = [0u32; 8];
        for (i, &old) in perm.iter().enumerate() {
            inv[old as usize] = i as u32;
        }

        let mut c1: Vec<u32> = (0..4).map(|i| inv[i]).collect();
        c1.sort_unstable();
        assert_eq!(c1[3] - c1[0], 3, "Clique 1 should be contiguous");

        let mut c2: Vec<u32> = (4..8).map(|i| inv[i]).collect();
        c2.sort_unstable();
        assert_eq!(c2[3] - c2[0], 3, "Clique 2 should be contiguous");
    }

    #[test]
    fn test_rabbit_is_deterministic() {
        let graph = make_two_cliques();
        assert_eq!(
            graph.reorder_rabbit().unwrap(),
            graph.reorder_rabbit().unwrap()
        );
    }

    #[test]
    fn test_rabbit_then_permute_preserves_structure() {
        let graph = make_two_cliques();
        let reordered = graph.permute(&graph.reorder_rabbit().unwrap()).unwrap();
        assert_eq!(reordered.num_nodes(), graph.num_nodes());
        assert_eq!(reordered.num_edges(), graph.num_edges());
    }

    #[test]
    fn test_empty_graph() {
        assert!(
            Graph::from_edges(0, &[], None)
                .unwrap()
                .reorder_rabbit()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_single_node() {
        assert_eq!(
            Graph::from_edges(1, &[], None)
                .unwrap()
                .reorder_rabbit()
                .unwrap(),
            vec![0]
        );
    }

    #[test]
    fn test_disconnected() {
        let perm = Graph::from_edges(5, &[(0, 1)], None)
            .unwrap()
            .reorder_rabbit()
            .unwrap();
        assert_bijection(&perm, 5);
    }

    // -- rabbit_partitions tests --

    #[test]
    fn test_partitions_length() {
        let graph = make_two_cliques();
        let parts = graph.rabbit_partitions().unwrap();
        assert_eq!(parts.len(), 8);
    }

    #[test]
    fn test_partitions_two_cliques_separated() {
        let graph = make_two_cliques();
        let parts = graph.rabbit_partitions().unwrap();
        let c1 = parts[0];
        for &n in &[1, 2, 3] {
            assert_eq!(parts[n], c1, "clique 1 nodes should share a partition");
        }
        let c2 = parts[4];
        for &n in &[5, 6, 7] {
            assert_eq!(parts[n], c2, "clique 2 nodes should share a partition");
        }
        // Joining across the single bridge lowers modularity.
        assert_ne!(c1, c2, "the cliques must stay separate communities");
    }

    /// Edges that only point from higher to lower ids (a citation graph's
    /// shape) still cluster: the aggregation sees both edge directions.
    #[test]
    fn test_partitions_follow_backward_edges() {
        let mut edges = Vec::new();
        for base in [0u32, 4] {
            for i in 0..4 {
                for j in 0..i {
                    edges.push((base + i, base + j));
                }
            }
        }
        edges.push((4, 3));
        let graph = Graph::from_edges(8, &edges, None).unwrap();
        let parts = graph.rabbit_partitions().unwrap();
        assert!(parts[..4].iter().all(|&p| p == parts[0]));
        assert!(parts[4..].iter().all(|&p| p == parts[4]));
        assert_ne!(parts[0], parts[4]);
    }

    /// A ring of cliques is one connected component. No clique is ever
    /// split, and the ring does not collapse into a single community —
    /// greedy aggregation may still pair an occasional clique with a
    /// neighbor whose bridge vertex has not been aggregated yet.
    #[test]
    fn test_partitions_do_not_collapse_to_components() {
        const CLIQUES: u32 = 8;
        const SIZE: u32 = 5;
        let mut edges = Vec::new();
        for c in 0..CLIQUES {
            let base = c * SIZE;
            for i in 0..SIZE {
                for j in 0..SIZE {
                    if i != j {
                        edges.push((base + i, base + j));
                    }
                }
            }
            // Bridge: this clique's last node to the next clique's first.
            let next = ((c + 1) % CLIQUES) * SIZE;
            edges.push((base + SIZE - 1, next));
            edges.push((next, base + SIZE - 1));
        }
        let graph = Graph::from_edges((CLIQUES * SIZE) as usize, &edges, None).unwrap();
        let parts = graph.rabbit_partitions().unwrap();
        for c in 0..CLIQUES {
            let base = (c * SIZE) as usize;
            let clique = &parts[base..base + SIZE as usize];
            assert!(
                clique.iter().all(|&p| p == clique[0]),
                "clique {c} split: {clique:?}"
            );
        }
        let communities = *parts.iter().max().unwrap() + 1;
        assert!(
            communities >= CLIQUES / 2,
            "ring collapsed into {communities} communities: {parts:?}"
        );
    }

    #[test]
    fn test_partitions_dense_ids() {
        let graph = make_two_cliques();
        let parts = graph.rabbit_partitions().unwrap();
        let max_id = *parts.iter().max().unwrap();
        // Partition IDs should be dense: max_id < num_unique_partitions
        let unique: std::collections::HashSet<u32> = parts.iter().copied().collect();
        assert_eq!(max_id + 1, unique.len() as u32);
    }

    #[test]
    fn test_partitions_empty_graph() {
        let graph = Graph::from_edges(0, &[], None).unwrap();
        assert!(graph.rabbit_partitions().unwrap().is_empty());
    }

    #[test]
    fn test_partitions_single_node() {
        let graph = Graph::from_edges(1, &[], None).unwrap();
        assert_eq!(graph.rabbit_partitions().unwrap(), vec![0]);
    }

    #[test]
    fn test_partitions_disconnected() {
        let graph = Graph::from_edges(5, &[(0, 1)], None).unwrap();
        let parts = graph.rabbit_partitions().unwrap();
        assert_eq!(parts.len(), 5);
        // Nodes 0 and 1 should share a partition (they're connected).
        assert_eq!(parts[0], parts[1]);
        // Isolated nodes 2, 3, 4 each get their own partition.
        let unique: std::collections::HashSet<u32> = parts.iter().copied().collect();
        assert_eq!(unique.len(), 4); // {0,1}, {2}, {3}, {4}
    }

    #[test]
    fn test_reorder_with_partitions_agrees_with_separate_calls() {
        let graph = make_two_cliques();
        let (perm, parts) = graph.reorder_rabbit_with_partitions().unwrap();
        assert_eq!(perm, graph.reorder_rabbit().unwrap());
        assert_eq!(parts, graph.rabbit_partitions().unwrap());
    }

    // -- partition_aligned_batches tests --

    #[test]
    fn test_batches_all_seeds_present() {
        let partitions = vec![0, 0, 1, 1, 2, 2];
        let seeds: Vec<NodeId> = vec![0, 1, 2, 3, 4, 5];
        let batches = partition_aligned_batches(&partitions, &seeds, 2, false, 0).unwrap();
        let mut all: Vec<NodeId> = batches.into_iter().flatten().collect();
        all.sort_unstable();
        assert_eq!(all, seeds);
    }

    #[test]
    fn test_batches_respects_batch_size() {
        let partitions = vec![0; 10];
        let seeds: Vec<NodeId> = (0..10).collect();
        let batches = partition_aligned_batches(&partitions, &seeds, 3, false, 0).unwrap();
        for (i, batch) in batches.iter().enumerate() {
            if i < batches.len() - 1 {
                assert_eq!(batch.len(), 3);
            } else {
                assert!(batch.len() <= 3);
            }
        }
        let total: usize = batches.iter().map(std::vec::Vec::len).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_batches_groups_by_partition() {
        // 3 partitions, batch_size=4 (big enough to hold all of partition 0)
        let partitions = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let seeds: Vec<NodeId> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let batches = partition_aligned_batches(&partitions, &seeds, 4, false, 0).unwrap();

        let batch_partitions: Vec<Vec<u32>> = batches
            .iter()
            .map(|b| b.iter().map(|&s| partitions[s as usize]).collect())
            .collect();

        // Partition 0's seeds should all appear before any partition 2 seed.
        let flat: Vec<u32> = batch_partitions.into_iter().flatten().collect();
        let first_p2 = flat.iter().position(|&p| p == 2).unwrap_or(flat.len());
        let last_p0 = flat.iter().rposition(|&p| p == 0).unwrap_or(0);
        assert!(
            last_p0 < first_p2,
            "partition 0 seeds should precede partition 2"
        );
    }

    #[test]
    fn test_batches_empty_seeds() {
        let partitions = vec![0, 1, 2];
        let batches = partition_aligned_batches(&partitions, &[], 128, false, 0).unwrap();
        assert!(batches.is_empty());
    }

    #[test]
    fn test_batches_single_seed() {
        let partitions = vec![0, 1, 2];
        let batches = partition_aligned_batches(&partitions, &[1], 128, false, 0).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], vec![1]);
    }

    #[test]
    fn test_batches_reject_bad_input() {
        let partitions = vec![0, 1, 2];
        assert!(partition_aligned_batches(&partitions, &[3], 2, false, 0).is_err());
        assert!(partition_aligned_batches(&partitions, &[0], 0, false, 0).is_err());
    }

    #[test]
    fn test_batches_shuffle_deterministic() {
        let partitions = vec![0, 0, 1, 1, 2, 2];
        let seeds: Vec<NodeId> = (0..6).collect();
        let b1 = partition_aligned_batches(&partitions, &seeds, 2, true, 42).unwrap();
        let b2 = partition_aligned_batches(&partitions, &seeds, 2, true, 42).unwrap();
        assert_eq!(b1, b2, "same seed should produce same shuffle");

        let b3 = partition_aligned_batches(&partitions, &seeds, 2, true, 99).unwrap();
        let mut all: Vec<NodeId> = b3.into_iter().flatten().collect();
        all.sort_unstable();
        assert_eq!(all, seeds);
    }

    // -- end-to-end: partitions + batching + sampling --

    #[test]
    fn test_end_to_end_partition_aligned_sampling() {
        let graph = make_two_cliques();
        let partitions = graph.rabbit_partitions().unwrap();

        // Seeds interleave the two cliques, so grouping is the batcher's work.
        let seeds: Vec<NodeId> = vec![0, 4, 1, 5, 2, 6, 3, 7];
        let batches = partition_aligned_batches(&partitions, &seeds, 4, false, 0).unwrap();

        assert_eq!(batches.len(), 2);
        let mut b0 = batches[0].clone();
        let mut b1 = batches[1].clone();
        b0.sort_unstable();
        b1.sort_unstable();
        let (lo, hi) = if b0[0] < b1[0] { (b0, b1) } else { (b1, b0) };
        assert_eq!(lo, vec![0, 1, 2, 3]);
        assert_eq!(hi, vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_partition_aligned_with_neighbor_loader() {
        let graph = std::sync::Arc::new(make_two_cliques());
        let partitions = graph.rabbit_partitions().unwrap();
        let seeds: Vec<NodeId> = (0..8).collect();
        let batches = partition_aligned_batches(&partitions, &seeds, 4, false, 0).unwrap();

        let config = crate::loader::SamplingConfig {
            fanout: vec![2],
            replace: false,
            seed: Some(42),
            ..Default::default()
        };
        let num_batches = batches.len();
        let loader = crate::loader::NeighborLoader::new(graph, config, 2, 1).unwrap();
        let batches = batches
            .into_iter()
            .map(|b| crate::loader::Seeds::new(b, loader.num_nodes()).unwrap())
            .collect();
        loader.submit_epoch(batches).unwrap();

        // Consume all results
        let mut total_seeds = 0;
        for _ in 0..num_batches {
            let sg = loader.next().unwrap().unwrap();
            total_seeds += sg.num_seeds();
        }
        assert_eq!(total_seeds, 8);
    }
}
