//! Graph reordering for improved cache locality in GNN training.
//!
//! Implements Rabbit Order (Arai et al., IPDPS 2016) with a parallel
//! merge phase using lock-free concurrent union-find.
//!
//! All phases are parallelized except the dendrogram construction:
//! 1. Each node picks its lowest-degree neighbor (parallel over V)
//! 2. Union-find merges (parallel over E) — logs merge events
//! 3. Build dendrogram from merge log + sequential traversal (O(V))

use crate::graph::csr::{EdgeOffset, Graph, NodeId};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

struct DendrogramNode {
    left: u32,
    right: u32,
}

/// Lock-free concurrent union-find with path halving and union by id.
///
/// The link direction is chosen purely from the immutable node ids — the
/// smaller id is always linked under the larger id. Because that choice is a
/// pure function of the unordered pair `{ra, rb}`, two threads unioning the
/// same two roots concurrently always pick the *same* direction, so they can
/// never install opposite links. That makes the parent forest provably
/// acyclic, which is what keeps `find`'s loop guaranteed to terminate.
/// Union-by-rank cannot offer the same guarantee here without a lock: rank
/// reads/increments are not atomic with the linking CAS, so concurrent unions
/// can pick opposing directions and form a cycle.
struct ConcurrentUnionFind {
    parent: Vec<AtomicU32>,
}

impl ConcurrentUnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).map(AtomicU32::new).collect(),
        }
    }

    #[inline]
    fn find(&self, mut x: u32) -> u32 {
        loop {
            let p = self.parent[x as usize].load(Ordering::Relaxed);
            if p == x {
                return x;
            }
            let gp = self.parent[p as usize].load(Ordering::Relaxed);
            let _ = self.parent[x as usize].compare_exchange_weak(
                p,
                gp,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            x = gp;
        }
    }

    /// Lock-free union. Returns `Some((winner, loser))` if a merge happened,
    /// `None` if already in the same set. `winner` is the surviving
    /// representative (the larger root id); `loser` is linked under it.
    #[inline]
    fn union(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        loop {
            let ra = self.find(a);
            let rb = self.find(b);
            if ra == rb {
                return None;
            }

            // Link smaller id under larger id. This direction is a pure
            // function of the pair, so concurrent unions can never create a
            // cycle (see the type-level comment).
            let (winner, loser) = if ra < rb { (rb, ra) } else { (ra, rb) };

            match self.parent[loser as usize].compare_exchange(
                loser,
                winner,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some((winner, loser)),
                Err(_) => continue,
            }
        }
    }
}

/// Merge log from Rabbit Order phases 1+2.
struct RabbitMergeLog {
    merges: Vec<(u32, u32)>,
}

impl Graph {
    /// Run Rabbit Order merge phases (parallel).
    ///
    /// Phase 1: each node picks lowest-degree neighbor.
    /// Phase 2: merge remaining cross-community edges.
    /// Returns the ordered merge log for dendrogram/partition construction.
    fn rabbit_merge(&self) -> Option<RabbitMergeLog> {
        let n = self.num_nodes();
        if n <= 1 || self.num_edges() == 0 {
            return None;
        }

        let degrees: Vec<u32> = (0..n)
            .into_par_iter()
            .map(|i| self.degree(i as NodeId) as u32)
            .collect();

        let m = self.num_edges() as u64;

        // Phase 1: Each node picks lowest-degree neighbor (parallel).
        let best_nbr: Vec<u32> = (0..n)
            .into_par_iter()
            .map(|u| {
                let neighbors = self.neighbors(u as NodeId);
                if neighbors.is_empty() {
                    return u32::MAX;
                }
                let max_deg_v = m / (degrees[u] as u64).max(1);
                let mut best_v = u32::MAX;
                let mut best_deg = u32::MAX;
                for &v in neighbors {
                    let dv = degrees[v as usize];
                    if (dv as u64) < max_deg_v && dv < best_deg {
                        best_deg = dv;
                        best_v = v;
                    }
                }
                best_v
            })
            .collect();

        // Phase 2: Parallel merges with lock-free union-find.
        let uf = ConcurrentUnionFind::new(n);

        // Pass 1: best-neighbor proposals (parallel)
        let merges_pass1: Vec<(u32, u32)> = (0..n)
            .into_par_iter()
            .filter_map(|u| {
                let v = best_nbr[u];
                if v == u32::MAX {
                    return None;
                }
                uf.union(u as u32, v)
            })
            .collect();

        // Pass 2: remaining cross-community edges (parallel)
        let degrees_ref = &degrees;
        let uf_ref = &uf;
        let merges_pass2: Vec<(u32, u32)> = (0..n)
            .into_par_iter()
            .flat_map_iter(|u| {
                let u = u as NodeId;
                let deg_u = degrees_ref[u as usize] as u64;
                let max_deg_v = m / deg_u.max(1);

                self.neighbors(u).iter().filter_map(move |&v| {
                    if u >= v {
                        return None;
                    }
                    let dv = degrees_ref[v as usize] as u64;
                    if dv >= max_deg_v {
                        return None;
                    }
                    // `union` already returns `None` for the same-set case, so a
                    // separate `find == find` pre-check would only add a
                    // redundant traversal on this hot loop.
                    uf_ref.union(u, v)
                })
            })
            .collect();

        // Append pass 2 into pass 1's allocation — a third full copy of the
        // merge log (8n bytes, 2x peak memory at billion-node scale) buys
        // nothing.
        let mut merges = merges_pass1;
        merges.extend_from_slice(&merges_pass2);
        drop(merges_pass2);

        Some(RabbitMergeLog { merges })
    }

    /// Compute the Rabbit Order permutation for improved cache locality.
    ///
    /// Parallel phases 1+2 (merge), sequential phase 3 (dendrogram traversal).
    /// Returns `perm` where `perm[new_id] = old_id`.
    ///
    /// Dendrogram indices (leaves `0..n`, internal nodes `n..n+merges`) are
    /// `u32`, so reordering is bounded to roughly 2.1B nodes even though the
    /// rest of the crate allows up to 4B (`NodeId` is `u32`).
    pub fn reorder_rabbit(&self) -> Vec<NodeId> {
        let n = self.num_nodes();
        let Some(merge_log) = self.rabbit_merge() else {
            return (0..n as NodeId).collect();
        };

        // Phase 3: Build dendrogram from merge log (sequential).
        //
        // Each merge (winner, loser) means: loser's subtree becomes a child
        // of winner's subtree. We process merges in order, building a binary tree.
        //
        // community_dendro[root] = current dendrogram index for that community.
        // Leaves: node i => index i. Internal: n + k.
        //
        // Internal-node indices are `n + k`; the largest is `n + merges - 1`,
        // and a forest over `n` leaves has at most `n - 1` internal nodes, so
        // the index space fits u32 whenever `n` does (up to ~2.1B leaves).
        debug_assert!(
            (n as u64 + merge_log.merges.len() as u64) <= u32::MAX as u64,
            "dendrogram index space exceeds u32"
        );
        let mut community_dendro: Vec<u32> = (0..n as u32).collect();
        let mut dendrogram: Vec<DendrogramNode> = Vec::with_capacity(merge_log.merges.len());

        // We need sequential find for dendrogram construction.
        // Convert the concurrent UF's parent array, but rebuild a fresh
        // sequential UF to replay merges in order.
        // Actually: just replay the merge log with a sequential UF.
        let mut seq_uf = SequentialUnionFind::new(n);

        for &(winner, loser) in &merge_log.merges {
            // The concurrent UF already merged these. Replay in sequential UF
            // to track roots for dendrogram construction.
            let rw = seq_uf.find(winner);
            let rl = seq_uf.find(loser);
            if rw == rl {
                continue; // duplicate merge (concurrent UF may log redundant pairs)
            }

            let internal_idx = n as u32 + dendrogram.len() as u32;
            dendrogram.push(DendrogramNode {
                left: community_dendro[rw as usize],
                right: community_dendro[rl as usize],
            });
            let new_root = seq_uf.union(rw, rl);
            community_dendro[new_root as usize] = internal_idx;
        }

        // In-order traversal of dendrogram forest.
        let mut perm = Vec::with_capacity(n);
        let mut stack = Vec::with_capacity(64);
        let mut visited = vec![false; n];

        let mut root_seen = vec![false; n];
        let mut roots = Vec::new();
        for i in 0..n as u32 {
            let root = seq_uf.find(i);
            if !root_seen[root as usize] {
                root_seen[root as usize] = true;
                roots.push(community_dendro[root as usize]);
            }
        }

        for root in roots {
            stack.clear();
            stack.push(root);
            while let Some(idx) = stack.pop() {
                if (idx as usize) < n {
                    if !visited[idx as usize] {
                        visited[idx as usize] = true;
                        perm.push(idx);
                    }
                } else {
                    let slot = (idx as usize) - n;
                    if slot < dendrogram.len() {
                        let node = &dendrogram[slot];
                        stack.push(node.right);
                        stack.push(node.left);
                    }
                }
            }
        }

        for i in 0..n as u32 {
            if !visited[i as usize] {
                perm.push(i);
            }
        }

        // The traversal flips `visited[idx]` before pushing, and the tail pushes
        // exactly the still-unvisited ids, so each id in `0..n` appears once.
        // A length check is enough to confirm the bijection given that gating.
        debug_assert_eq!(perm.len(), n);
        debug_assert!(perm.iter().all(|&id| (id as usize) < n));
        perm
    }

    /// Compute Rabbit Order community partitions.
    ///
    /// Returns a vector of length `num_nodes` where `partitions[node] = partition_id`.
    /// Partition IDs are dense (0..num_partitions). Nodes in the same community
    /// get the same partition ID.
    ///
    /// This is a free byproduct of the Rabbit merge phases -- no dendrogram needed.
    pub fn rabbit_partitions(&self) -> Vec<u32> {
        let n = self.num_nodes();
        let Some(merge_log) = self.rabbit_merge() else {
            // Each node is its own partition
            return (0..n as u32).collect();
        };

        // Replay merges in a sequential UF to recover community roots.
        let mut uf = SequentialUnionFind::new(n);
        for &(winner, loser) in &merge_log.merges {
            let rw = uf.find(winner);
            let rl = uf.find(loser);
            if rw != rl {
                uf.union(rw, rl);
            }
        }

        // Map roots to dense partition IDs.
        let mut root_to_id: FxHashMap<u32, u32> = FxHashMap::default();
        let mut partitions = Vec::with_capacity(n);
        for i in 0..n as u32 {
            let root = uf.find(i);
            let next_id = root_to_id.len() as u32;
            let id = *root_to_id.entry(root).or_insert(next_id);
            partitions.push(id);
        }
        partitions
    }

    /// Compute Rabbit Order permutation AND partition assignments in one pass.
    ///
    /// Avoids running the parallel merge phases twice when both the reordered
    /// graph and cluster-aligned batching are needed.
    ///
    /// Like [`Self::reorder_rabbit`], the `u32` dendrogram index space bounds
    /// reordering to roughly 2.1B nodes.
    pub fn reorder_rabbit_with_partitions(&self) -> (Vec<NodeId>, Vec<u32>) {
        let n = self.num_nodes();
        let Some(merge_log) = self.rabbit_merge() else {
            return ((0..n as NodeId).collect(), (0..n as u32).collect());
        };

        // Build the dendrogram permutation (same as reorder_rabbit). The
        // same sequential union-find drives the partition assignment below
        // — replaying the identical merge sequence into a second UF (two
        // more O(n) arrays and a full second replay of the sequential
        // phase) produced byte-identical roots.
        debug_assert!(
            (n as u64 + merge_log.merges.len() as u64) <= u32::MAX as u64,
            "dendrogram index space exceeds u32"
        );
        let mut community_dendro: Vec<u32> = (0..n as u32).collect();
        let mut dendrogram: Vec<DendrogramNode> = Vec::with_capacity(merge_log.merges.len());
        let mut seq_uf = SequentialUnionFind::new(n);
        for &(winner, loser) in &merge_log.merges {
            let rw = seq_uf.find(winner);
            let rl = seq_uf.find(loser);
            if rw == rl {
                continue;
            }
            let internal_idx = n as u32 + dendrogram.len() as u32;
            dendrogram.push(DendrogramNode {
                left: community_dendro[rw as usize],
                right: community_dendro[rl as usize],
            });
            let new_root = seq_uf.union(rw, rl);
            community_dendro[new_root as usize] = internal_idx;
        }

        // Partition assignments from the same union-find.
        let mut root_to_id: FxHashMap<u32, u32> = FxHashMap::default();
        let mut partitions = Vec::with_capacity(n);
        for i in 0..n as u32 {
            let root = seq_uf.find(i);
            let next_id = root_to_id.len() as u32;
            let id = *root_to_id.entry(root).or_insert(next_id);
            partitions.push(id);
        }

        let mut perm = Vec::with_capacity(n);
        let mut stack = Vec::with_capacity(64);
        let mut visited = vec![false; n];
        let mut root_seen = vec![false; n];
        let mut roots = Vec::new();
        for i in 0..n as u32 {
            let root = seq_uf.find(i);
            if !root_seen[root as usize] {
                root_seen[root as usize] = true;
                roots.push(community_dendro[root as usize]);
            }
        }
        for root in roots {
            stack.clear();
            stack.push(root);
            while let Some(idx) = stack.pop() {
                if (idx as usize) < n {
                    if !visited[idx as usize] {
                        visited[idx as usize] = true;
                        perm.push(idx);
                    }
                } else {
                    let slot = (idx as usize) - n;
                    if slot < dendrogram.len() {
                        let node = &dendrogram[slot];
                        stack.push(node.right);
                        stack.push(node.left);
                    }
                }
            }
        }
        for i in 0..n as u32 {
            if !visited[i as usize] {
                perm.push(i);
            }
        }
        debug_assert_eq!(perm.len(), n);
        debug_assert!(perm.iter().all(|&id| (id as usize) < n));

        (perm, partitions)
    }

    /// Apply a node permutation to produce a new reordered graph.
    pub fn permute(&self, perm: &[NodeId]) -> anyhow::Result<Graph> {
        let n = self.num_nodes();
        anyhow::ensure!(perm.len() == n, "permutation length mismatch");

        if n == 0 {
            return Graph::from_edges(0, &[], None);
        }

        // The duplicate/range check folds into the inverse-permutation
        // build: u32::MAX marks unassigned slots (new_id < n <= u32::MAX,
        // so it can't collide), replacing a separate bool array and a
        // second O(n) pass.
        let mut inv_perm = vec![u32::MAX; n];
        for (new_id, &old_id) in perm.iter().enumerate() {
            anyhow::ensure!((old_id as usize) < n, "out-of-range node {}", old_id);
            anyhow::ensure!(
                inv_perm[old_id as usize] == u32::MAX,
                "duplicate node {}",
                old_id
            );
            inv_perm[old_id as usize] = new_id as u32;
        }

        // The per-node degree lookup is a random access into the offsets
        // array — serial and latency-bound at billion-node scale. Gather
        // degrees in parallel, then run the cheap sequential prefix sum
        // over the dense result.
        let degrees: Vec<u64> = (0..n)
            .into_par_iter()
            .map(|new_id| self.degree(perm[new_id]) as u64)
            .collect();
        let mut new_offsets: Vec<EdgeOffset> = Vec::with_capacity(n + 1);
        new_offsets.push(0);
        let mut acc = 0u64;
        for &d in &degrees {
            acc += d;
            new_offsets.push(acc);
        }
        drop(degrees);

        let num_edges = self.num_edges();
        let has_weights = self.weights().is_some();
        let has_timestamps = self.has_timestamps();

        let mut new_edges = vec![0u32; num_edges];
        let mut new_weights = if has_weights {
            vec![0.0f32; num_edges]
        } else {
            Vec::new()
        };
        let mut new_timestamps = if has_timestamps {
            vec![0.0f64; num_edges]
        } else {
            Vec::new()
        };

        let ep = new_edges.as_mut_ptr() as usize;
        let wp = new_weights.as_mut_ptr() as usize;
        let tp = new_timestamps.as_mut_ptr() as usize;

        (0..n).into_par_iter().for_each(|new_id| {
            let old_id = perm[new_id];
            let dst = new_offsets[new_id] as usize;
            for (j, &old_nbr) in self.neighbors(old_id).iter().enumerate() {
                // SAFETY: dst+j < new_offsets[new_id+1] <= num_edges (loop bounded by degree), so the
                // pointer stays in-bounds of `new_edges`; each (new_id, j) pair is written by exactly
                // one rayon task, so the writes don't race.
                let p = unsafe { (ep as *mut NodeId).add(dst + j) };
                // SAFETY: p is in-bounds and uniquely owned by this task (see above).
                unsafe {
                    *p = inv_perm[old_nbr as usize];
                }
            }
            if has_weights && let Some(w) = self.neighbor_weights(old_id) {
                // SAFETY: `dst` is in-bounds of `new_weights` (parallel to
                // edges).
                let base = unsafe { (wp as *mut f32).add(dst) };
                // SAFETY: dst..dst+w.len() is uniquely owned by this task,
                // so a temporary exclusive slice is sound — and unlike an
                // element-at-a-time raw-pointer loop, `copy_from_slice`
                // lowers to one memcpy.
                let out = unsafe { std::slice::from_raw_parts_mut(base, w.len()) };
                out.copy_from_slice(w);
            }
            if has_timestamps && let Some(ts) = self.neighbor_timestamps(old_id) {
                // SAFETY: `dst` is in-bounds of `new_timestamps` (parallel
                // to edges).
                let base = unsafe { (tp as *mut f64).add(dst) };
                // SAFETY: same ownership argument as the weights slice.
                let out = unsafe { std::slice::from_raw_parts_mut(base, ts.len()) };
                out.copy_from_slice(ts);
            }
        });

        let weights_arc = if has_weights {
            Some(Arc::from(new_weights))
        } else {
            None
        };
        let mut graph = Graph::from_owned_parts(
            n,
            num_edges,
            new_offsets.into(),
            new_edges.into(),
            weights_arc,
        );
        if has_timestamps {
            // `new_timestamps` is built parallel to `new_edges` above, so the
            // length contract holds; surface a clear error if a future refactor
            // violates it rather than silently dropping the result.
            graph.set_timestamps(new_timestamps).map_err(|e| {
                anyhow::anyhow!("internal: reorder produced mismatched timestamps ({e})")
            })?;
        }
        Ok(graph)
    }
}

/// Simple sequential union-find for dendrogram replay.
struct SequentialUnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl SequentialUnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    #[inline(always)]
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            self.parent[x as usize] = self.parent[self.parent[x as usize] as usize];
            x = self.parent[x as usize];
        }
        x
    }

    #[inline(always)]
    fn union(&mut self, a: u32, b: u32) -> u32 {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        if self.rank[ra as usize] < self.rank[rb as usize] {
            self.parent[ra as usize] = rb;
            rb
        } else if self.rank[ra as usize] > self.rank[rb as usize] {
            self.parent[rb as usize] = ra;
            ra
        } else {
            self.parent[rb as usize] = ra;
            self.rank[ra as usize] += 1;
            ra
        }
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
/// nearby partitions.
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
) -> Vec<Vec<NodeId>> {
    assert!(batch_size > 0, "batch_size must be >= 1");

    // Group seeds by partition with one u64 sort — (partition << 32 |
    // original index) keys group by partition and stay stable within it.
    // The single key vec is the whole grouping state: a per-partition Vec
    // map would allocate one heap Vec per distinct community (Rabbit
    // produces many small ones), approaching one allocation per seed each
    // epoch. A seed id must index into `partitions` (length == num_nodes);
    // otherwise it would silently fall into partition 0 and be mis-grouped.
    let mut keyed: Vec<u64> = seeds
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            debug_assert!(
                (s as usize) < partitions.len(),
                "seed {s} is out of range for partitions of length {}",
                partitions.len()
            );
            let p = partitions.get(s as usize).copied().unwrap_or(0);
            ((p as u64) << 32) | i as u64
        })
        .collect();
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

    batches
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

    #[test]
    fn test_permute_identity() {
        let graph = Graph::from_edges(3, &[(0, 1), (0, 2), (1, 2), (2, 0)], None).unwrap();
        let permuted = graph.permute(&[0, 1, 2]).unwrap();
        assert_eq!(permuted.num_nodes(), 3);
        assert_eq!(permuted.num_edges(), graph.num_edges());
        for n in 0..3u32 {
            let mut a: Vec<_> = graph.neighbors(n).to_vec();
            a.sort();
            let mut b: Vec<_> = permuted.neighbors(n).to_vec();
            b.sort();
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
        assert_eq!(permuted.neighbor_weights(2).unwrap().len(), 2);
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

    #[test]
    fn test_rabbit_valid_permutation() {
        let graph = make_two_cliques();
        let perm = graph.reorder_rabbit();
        assert_eq!(perm.len(), 8);
        let mut seen = [false; 8];
        for &id in &perm {
            assert!(!seen[id as usize]);
            seen[id as usize] = true;
        }
    }

    #[test]
    fn test_rabbit_clusters_communities() {
        let graph = make_two_cliques();
        let perm = graph.reorder_rabbit();
        let mut inv = [0u32; 8];
        for (i, &old) in perm.iter().enumerate() {
            inv[old as usize] = i as u32;
        }

        let mut c1: Vec<u32> = (0..4).map(|i| inv[i]).collect();
        c1.sort();
        assert_eq!(c1[3] - c1[0], 3, "Clique 1 should be contiguous");

        let mut c2: Vec<u32> = (4..8).map(|i| inv[i]).collect();
        c2.sort();
        assert_eq!(c2[3] - c2[0], 3, "Clique 2 should be contiguous");
    }

    #[test]
    fn test_rabbit_then_permute_preserves_structure() {
        let graph = make_two_cliques();
        let reordered = graph.permute(&graph.reorder_rabbit()).unwrap();
        assert_eq!(reordered.num_nodes(), graph.num_nodes());
        assert_eq!(reordered.num_edges(), graph.num_edges());
    }

    #[test]
    fn test_empty_graph() {
        assert!(
            Graph::from_edges(0, &[], None)
                .unwrap()
                .reorder_rabbit()
                .is_empty()
        );
    }

    #[test]
    fn test_single_node() {
        assert_eq!(
            Graph::from_edges(1, &[], None).unwrap().reorder_rabbit(),
            vec![0]
        );
    }

    #[test]
    fn test_disconnected() {
        let perm = Graph::from_edges(5, &[(0, 1)], None)
            .unwrap()
            .reorder_rabbit();
        assert_eq!(perm.len(), 5);
        let mut seen = [false; 5];
        for &id in &perm {
            seen[id as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    // -- rabbit_partitions tests --

    #[test]
    fn test_partitions_length() {
        let graph = make_two_cliques();
        let parts = graph.rabbit_partitions();
        assert_eq!(parts.len(), 8);
    }

    #[test]
    fn test_partitions_two_cliques_separated() {
        let graph = make_two_cliques();
        let parts = graph.rabbit_partitions();
        // Nodes 0..4 are one clique, 4..8 are another.
        // They should get different partition IDs (bridge edge 3-4 is weak).
        let c1 = parts[0];
        for &n in &[1, 2, 3] {
            assert_eq!(parts[n], c1, "clique 1 nodes should share a partition");
        }
        let c2 = parts[4];
        for &n in &[5, 6, 7] {
            assert_eq!(parts[n], c2, "clique 2 nodes should share a partition");
        }
        // The two cliques may or may not merge (depends on Rabbit's threshold),
        // but we at least verify internal consistency.
    }

    #[test]
    fn test_partitions_dense_ids() {
        let graph = make_two_cliques();
        let parts = graph.rabbit_partitions();
        let max_id = *parts.iter().max().unwrap();
        // Partition IDs should be dense: max_id < num_unique_partitions
        let unique: std::collections::HashSet<u32> = parts.iter().copied().collect();
        assert_eq!(max_id + 1, unique.len() as u32);
    }

    #[test]
    fn test_partitions_empty_graph() {
        let graph = Graph::from_edges(0, &[], None).unwrap();
        assert!(graph.rabbit_partitions().is_empty());
    }

    #[test]
    fn test_partitions_single_node() {
        let graph = Graph::from_edges(1, &[], None).unwrap();
        assert_eq!(graph.rabbit_partitions(), vec![0]);
    }

    #[test]
    fn test_partitions_disconnected() {
        let graph = Graph::from_edges(5, &[(0, 1)], None).unwrap();
        let parts = graph.rabbit_partitions();
        assert_eq!(parts.len(), 5);
        // Nodes 0 and 1 should share a partition (they're connected).
        assert_eq!(parts[0], parts[1]);
        // Isolated nodes 2, 3, 4 each get their own partition.
        let unique: std::collections::HashSet<u32> = parts.iter().copied().collect();
        assert_eq!(unique.len(), 4); // {0,1}, {2}, {3}, {4}
    }

    // -- partition_aligned_batches tests --

    #[test]
    fn test_batches_all_seeds_present() {
        let partitions = vec![0, 0, 1, 1, 2, 2];
        let seeds: Vec<NodeId> = vec![0, 1, 2, 3, 4, 5];
        let batches = partition_aligned_batches(&partitions, &seeds, 2, false, 0);
        let mut all: Vec<NodeId> = batches.into_iter().flatten().collect();
        all.sort();
        assert_eq!(all, seeds);
    }

    #[test]
    fn test_batches_respects_batch_size() {
        let partitions = vec![0; 10];
        let seeds: Vec<NodeId> = (0..10).collect();
        let batches = partition_aligned_batches(&partitions, &seeds, 3, false, 0);
        for (i, batch) in batches.iter().enumerate() {
            if i < batches.len() - 1 {
                assert_eq!(batch.len(), 3);
            } else {
                assert!(batch.len() <= 3);
            }
        }
        let total: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_batches_groups_by_partition() {
        // 3 partitions, batch_size=4 (big enough to hold all of partition 0)
        let partitions = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let seeds: Vec<NodeId> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let batches = partition_aligned_batches(&partitions, &seeds, 4, false, 0);

        // First batch should be partition 0 (3 items) + 1 from partition 1
        // Verify: within each batch, most nodes share a partition.
        // More precisely: seeds from partition 0 are never split across batches
        // that have partition 2 seeds (no interleaving).
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
        let batches = partition_aligned_batches(&partitions, &[], 128, false, 0);
        assert!(batches.is_empty());
    }

    #[test]
    fn test_batches_single_seed() {
        let partitions = vec![0, 1, 2];
        let batches = partition_aligned_batches(&partitions, &[1], 128, false, 0);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], vec![1]);
    }

    #[test]
    fn test_batches_shuffle_deterministic() {
        let partitions = vec![0, 0, 1, 1, 2, 2];
        let seeds: Vec<NodeId> = (0..6).collect();
        let b1 = partition_aligned_batches(&partitions, &seeds, 2, true, 42);
        let b2 = partition_aligned_batches(&partitions, &seeds, 2, true, 42);
        assert_eq!(b1, b2, "same seed should produce same shuffle");

        let b3 = partition_aligned_batches(&partitions, &seeds, 2, true, 99);
        // Different seed may produce different order (not guaranteed but very likely
        // with 3 partitions). Just verify all seeds present.
        let mut all: Vec<NodeId> = b3.into_iter().flatten().collect();
        all.sort();
        assert_eq!(all, seeds);
    }

    // -- end-to-end: partitions + batching + sampling --

    #[test]
    fn test_end_to_end_partition_aligned_sampling() {
        let graph = make_two_cliques();
        let partitions = graph.rabbit_partitions();

        // Use all nodes as seeds
        let seeds: Vec<NodeId> = (0..8).collect();
        let batches = partition_aligned_batches(&partitions, &seeds, 4, false, 0);

        // Should produce 2 batches of 4
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 4);
        assert_eq!(batches[1].len(), 4);

        // Each batch should be dominated by one clique's nodes.
        // Count how many clique-0 nodes (0..4) in each batch.
        let c0_batch0 = batches[0].iter().filter(|&&n| n < 4).count();
        let c0_batch1 = batches[1].iter().filter(|&&n| n < 4).count();

        // One batch should have 3-4 clique-0 nodes, the other 0-1.
        let (hi, lo) = if c0_batch0 >= c0_batch1 {
            (c0_batch0, c0_batch1)
        } else {
            (c0_batch1, c0_batch0)
        };
        assert!(
            hi >= 3,
            "expected at least 3 same-clique nodes in aligned batch, got {hi}"
        );
        assert!(
            lo <= 1,
            "expected at most 1 cross-clique node in aligned batch, got {lo}"
        );
    }

    #[test]
    fn test_partition_aligned_with_neighbor_loader() {
        let graph = std::sync::Arc::new(make_two_cliques());
        let partitions = graph.rabbit_partitions();
        let seeds: Vec<NodeId> = (0..8).collect();
        let batches = partition_aligned_batches(&partitions, &seeds, 4, false, 0);

        let config = crate::loader::SamplingConfig {
            fanout: vec![2],
            replace: false,
            seed: Some(42),
            ..Default::default()
        };
        let num_batches = batches.len();
        let loader = crate::loader::NeighborLoader::new(graph, config, 2, 1).unwrap();
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
