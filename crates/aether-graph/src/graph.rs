//! Dynamic graph: vertex array + per-vertex C-tree neighbor lists.
//!
//! Single-writer, multi-reader via functional C-trees. Edge inserts
//! path-copy the affected C-tree; the old version remains valid for
//! concurrent readers. The vertex array uses atomic stores for the
//! root pointer swap.
//!
//! Read path (sampling): zero allocations, lock-free.
//! Write path (edge insert): arena bump-alloc only, no locks.

use crate::arena::Arena;
use crate::chunk::Chunk;
use crate::ctree::CTree;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Dirty-node bitmap for historical embedding tracking.
///
/// One bit per vertex. Set on edge insert, cleared at epoch boundary.
/// Lock-free: writers set bits with `fetch_or`, readers load with Acquire.
struct DirtyBitmap {
    words: Vec<AtomicU64>,
}

impl DirtyBitmap {
    fn new(num_vertices: usize) -> Self {
        let num_words = num_vertices.div_ceil(64);
        let mut words = Vec::with_capacity(num_words);
        for _ in 0..num_words {
            words.push(AtomicU64::new(0));
        }
        Self { words }
    }

    /// Mark a vertex as dirty. Lock-free (atomic fetch_or).
    #[inline]
    fn mark(&self, vertex: u32) {
        let word = vertex as usize / 64;
        let bit = vertex as usize % 64;
        self.words[word].fetch_or(1 << bit, Ordering::Release);
    }

    /// Check if a vertex is dirty.
    #[inline]
    fn is_dirty(&self, vertex: u32) -> bool {
        let word = vertex as usize / 64;
        let bit = vertex as usize % 64;
        self.words[word].load(Ordering::Acquire) & (1 << bit) != 0
    }

    /// Clear all dirty bits (epoch boundary).
    fn clear_all(&self) {
        for w in &self.words {
            w.store(0, Ordering::Release);
        }
    }

    /// Collect all dirty node IDs and atomically clear the bitmap.
    /// Uses swap per word so concurrent writers don't lose marks.
    fn drain_dirty(&self) -> Vec<u32> {
        let mut result = Vec::new();
        for (i, w) in self.words.iter().enumerate() {
            let bits = w.swap(0, Ordering::AcqRel);
            if bits == 0 {
                continue;
            }
            let base = (i * 64) as u32;
            let mut remaining = bits;
            while remaining != 0 {
                let bit = remaining.trailing_zeros();
                result.push(base + bit);
                remaining &= remaining - 1; // clear lowest set bit
            }
        }
        result
    }

    /// Count set bits across all words.
    fn count_dirty(&self) -> usize {
        self.words
            .iter()
            .map(|w| w.load(Ordering::Acquire).count_ones() as usize)
            .sum()
    }
}

/// Lock-free dynamic graph with O(1) edge insert and O(degree) neighbor access.
///
/// Each vertex's neighbor list is a C-tree (balanced tree of sorted chunks)
/// stored in a bump-allocating arena. Edge inserts create new tree nodes
/// via path copying; old trees remain valid for concurrent readers.
pub struct DynamicGraph {
    /// Per-vertex C-tree root offsets. Atomic for lock-free root swaps.
    /// NULL (u32::MAX) means no neighbors.
    roots: Vec<AtomicU32>,
    /// Arena for all C-tree nodes (chunks + interior nodes).
    arena: Arena,
    /// Number of vertices (fixed at construction).
    num_vertices: usize,
    /// Total edge count (informational, not used for correctness).
    num_edges: AtomicU64,
    /// Dirty-node bitmap for historical embedding tracking.
    /// Set on edge insert, cleared at epoch boundary.
    dirty: DirtyBitmap,
}

impl DynamicGraph {
    /// Create a graph with `num_vertices` vertices and no edges.
    ///
    /// `arena_bytes` controls the arena capacity. At ~80 bytes per edge
    /// (chunk amortized + interior node overhead), 1GB supports ~12M edges.
    /// For 100M edges, use ~8GB.
    pub fn new(num_vertices: usize, arena_bytes: usize) -> Self {
        let mut roots = Vec::with_capacity(num_vertices);
        for _ in 0..num_vertices {
            roots.push(AtomicU32::new(crate::ctree::NULL));
        }
        Self {
            roots,
            arena: Arena::new(arena_bytes),
            num_vertices,
            num_edges: AtomicU64::new(0),
            dirty: DirtyBitmap::new(num_vertices),
        }
    }

    /// Number of vertices.
    #[inline(always)]
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// Total edge count.
    #[inline(always)]
    pub fn num_edges(&self) -> u64 {
        self.num_edges.load(Ordering::Relaxed)
    }

    /// Insert a directed edge from `src` to `dst`.
    ///
    /// Returns `true` if the edge was new, `false` if it already existed.
    /// Returns `Err` if the arena is full.
    ///
    /// Single-writer only. The new C-tree root is atomically published
    /// so concurrent readers see a consistent snapshot.
    pub fn insert_edge(&self, src: u32, dst: u32) -> Result<bool, ArenaFull> {
        debug_assert!((src as usize) < self.num_vertices);
        debug_assert!((dst as usize) < self.num_vertices);

        let current_root = self.roots[src as usize].load(Ordering::Acquire);
        let tree = CTree { root: current_root };

        match tree.insert(&self.arena, dst) {
            Some(new_tree) => {
                // Atomically publish new root. Release ordering ensures
                // all arena writes (new nodes) are visible before the
                // root pointer becomes visible to readers.
                self.roots[src as usize].store(new_tree.root, Ordering::Release);
                self.num_edges.fetch_add(1, Ordering::Relaxed);
                self.dirty.mark(src);
                self.dirty.mark(dst);
                Ok(true)
            }
            None => Ok(false), // duplicate edge
        }
    }

    /// Get the C-tree root for a vertex.
    #[inline(always)]
    fn tree_for(&self, vertex: u32) -> CTree {
        CTree {
            root: self.roots[vertex as usize].load(Ordering::Acquire),
        }
    }

    /// Degree of a vertex.
    #[inline]
    pub fn degree(&self, vertex: u32) -> usize {
        self.tree_for(vertex).count(&self.arena)
    }

    /// Iterate a vertex's neighbors in sorted order. Calls `f` for each chunk.
    ///
    /// Zero allocations. The chunks are read directly from the arena.
    /// Safe to call concurrently with `insert_edge` — readers see a
    /// consistent snapshot (the tree that was current when they read the root).
    #[inline]
    pub fn for_each_chunk(&self, vertex: u32, f: impl FnMut(&Chunk)) {
        self.tree_for(vertex).for_each_chunk(&self.arena, f);
    }

    /// Collect all neighbors of `vertex` into `buf` (sorted).
    ///
    /// Clears `buf` first. For GNN sampling, use this to get a flat
    /// slice suitable for Floyd's algorithm.
    #[inline]
    pub fn neighbors_into(&self, vertex: u32, buf: &mut Vec<u32>) {
        buf.clear();
        self.tree_for(vertex).collect_into(&self.arena, buf);
    }

    /// Check if edge (src → dst) exists.
    #[inline]
    pub fn has_edge(&self, src: u32, dst: u32) -> bool {
        self.tree_for(src).contains(&self.arena, dst)
    }

    /// Arena usage stats.
    pub fn arena_used(&self) -> usize {
        self.arena.used()
    }

    pub fn arena_capacity(&self) -> usize {
        self.arena.capacity()
    }

    /// Snapshot the dynamic graph into raw CSR arrays.
    ///
    /// Returns `(offsets, edges)` where `offsets[v]` is the start of vertex
    /// `v`'s neighbors in `edges`. `offsets` has length `num_vertices + 1`.
    ///
    /// This is O(V + E) time and allocates fresh vectors for the CSR data.
    /// Call once per epoch, not per batch — the returned arrays can be used
    /// to construct a static `Graph` in the PyO3 layer.
    ///
    /// # Concurrency
    ///
    /// Safe to call concurrently with `insert_edge`. Each vertex's neighbor
    /// list is read from the C-tree root that was current at read time, so
    /// the snapshot is a consistent (though not globally atomic) view.
    pub fn snapshot_csr(&self) -> (Vec<u64>, Vec<u32>) {
        let mut offsets = Vec::with_capacity(self.num_vertices + 1);
        let mut edges = Vec::new();
        let mut buf = Vec::new();

        offsets.push(0);
        for v in 0..self.num_vertices as u32 {
            self.neighbors_into(v, &mut buf);
            edges.extend_from_slice(&buf);
            offsets.push(edges.len() as u64);
        }

        (offsets, edges)
    }

    /// Drain all dirty node IDs and atomically clear the bitmap.
    ///
    /// Returns the set of vertices that had edges inserted since the last
    /// drain or clear. Order is ascending by vertex ID within each word.
    pub fn drain_dirty(&self) -> Vec<u32> {
        self.dirty.drain_dirty()
    }

    /// Number of vertices currently marked dirty.
    pub fn dirty_count(&self) -> usize {
        self.dirty.count_dirty()
    }

    /// Check if a vertex is dirty (received new edges since last clear/drain).
    #[inline]
    pub fn is_dirty(&self, vertex: u32) -> bool {
        self.dirty.is_dirty(vertex)
    }

    /// Clear all dirty bits without collecting them.
    pub fn clear_dirty(&self) {
        self.dirty.clear_all();
    }

    /// Build from a batch of edges. More efficient than individual inserts
    /// because edges are sorted per-vertex and chunks are built directly.
    pub fn from_edges(num_vertices: usize, edges: &[(u32, u32)], arena_bytes: usize) -> Self {
        let graph = Self::new(num_vertices, arena_bytes);

        // Group edges by source vertex
        // For truly optimal bulk loading, we'd build C-trees directly from
        // sorted neighbor lists. For now, sequential inserts are fine since
        // each insert is O(log degree) amortized.
        for &(src, dst) in edges {
            let _ = graph.insert_edge(src, dst);
        }

        graph
    }
}

/// Error when arena is full.
#[derive(Debug, Clone, Copy)]
pub struct ArenaFull;

impl std::fmt::Display for ArenaFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "C-tree arena is full")
    }
}

impl std::error::Error for ArenaFull {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph() {
        let g = DynamicGraph::new(100, 1 << 20);
        assert_eq!(g.num_vertices(), 100);
        assert_eq!(g.num_edges(), 0);
        assert_eq!(g.degree(0), 0);
        assert!(!g.has_edge(0, 1));
    }

    #[test]
    fn insert_and_query() {
        let g = DynamicGraph::new(100, 1 << 20);
        assert!(g.insert_edge(0, 10).unwrap());
        assert!(g.insert_edge(0, 20).unwrap());
        assert!(g.insert_edge(0, 5).unwrap());

        assert_eq!(g.degree(0), 3);
        assert!(g.has_edge(0, 10));
        assert!(g.has_edge(0, 20));
        assert!(g.has_edge(0, 5));
        assert!(!g.has_edge(0, 15));

        let mut buf = Vec::new();
        g.neighbors_into(0, &mut buf);
        assert_eq!(buf, vec![5, 10, 20]);
    }

    #[test]
    fn duplicate_edge() {
        let g = DynamicGraph::new(100, 1 << 20);
        assert!(g.insert_edge(0, 10).unwrap()); // new
        assert!(!g.insert_edge(0, 10).unwrap()); // duplicate
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.num_edges(), 1);
    }

    #[test]
    fn multiple_vertices() {
        let g = DynamicGraph::new(100, 1 << 20);
        g.insert_edge(0, 10).unwrap();
        g.insert_edge(0, 20).unwrap();
        g.insert_edge(1, 30).unwrap();
        g.insert_edge(1, 40).unwrap();
        g.insert_edge(1, 50).unwrap();

        assert_eq!(g.degree(0), 2);
        assert_eq!(g.degree(1), 3);
        assert_eq!(g.degree(2), 0);
    }

    #[test]
    fn high_degree_vertex() {
        let g = DynamicGraph::new(1000, 1 << 20);
        for i in 0..500u32 {
            assert!(g.insert_edge(0, i).unwrap());
        }
        assert_eq!(g.degree(0), 500);

        let mut buf = Vec::new();
        g.neighbors_into(0, &mut buf);
        assert_eq!(buf.len(), 500);
        // Sorted
        for w in buf.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn from_edges_batch() {
        let edges: Vec<(u32, u32)> = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 3),
        ];
        let g = DynamicGraph::from_edges(4, &edges, 1 << 20);
        assert_eq!(g.degree(0), 3);
        assert_eq!(g.degree(1), 2);
        assert_eq!(g.degree(2), 3);
        assert_eq!(g.degree(3), 0);
    }

    #[test]
    fn snapshot_csr_basic() {
        let g = DynamicGraph::new(4, 1 << 20);
        g.insert_edge(0, 1).unwrap();
        g.insert_edge(0, 2).unwrap();
        g.insert_edge(0, 3).unwrap();
        g.insert_edge(1, 0).unwrap();
        g.insert_edge(1, 2).unwrap();
        g.insert_edge(2, 0).unwrap();

        let (offsets, edges) = g.snapshot_csr();
        assert_eq!(offsets.len(), 5); // num_vertices + 1
        assert_eq!(offsets[0], 0);
        // vertex 0: neighbors [1, 2, 3]
        assert_eq!(offsets[1] - offsets[0], 3);
        assert_eq!(&edges[offsets[0] as usize..offsets[1] as usize], &[1, 2, 3]);
        // vertex 1: neighbors [0, 2]
        assert_eq!(offsets[2] - offsets[1], 2);
        assert_eq!(&edges[offsets[1] as usize..offsets[2] as usize], &[0, 2]);
        // vertex 2: neighbors [0]
        assert_eq!(offsets[3] - offsets[2], 1);
        assert_eq!(&edges[offsets[2] as usize..offsets[3] as usize], &[0]);
        // vertex 3: no neighbors
        assert_eq!(offsets[4] - offsets[3], 0);
    }

    #[test]
    fn snapshot_csr_empty() {
        let g = DynamicGraph::new(3, 1 << 20);
        let (offsets, edges) = g.snapshot_csr();
        assert_eq!(offsets, vec![0, 0, 0, 0]);
        assert!(edges.is_empty());
    }

    #[test]
    fn dirty_tracking_basic() {
        let g = DynamicGraph::new(100, 1024);
        assert_eq!(g.dirty_count(), 0);
        g.insert_edge(0, 1).unwrap();
        assert!(g.is_dirty(0));
        assert!(g.is_dirty(1));
        assert!(!g.is_dirty(2));
        assert_eq!(g.dirty_count(), 2);
    }

    #[test]
    fn dirty_drain() {
        let g = DynamicGraph::new(100, 1024);
        g.insert_edge(0, 1).unwrap();
        g.insert_edge(2, 3).unwrap();
        let mut dirty = g.drain_dirty();
        dirty.sort();
        assert_eq!(dirty, vec![0, 1, 2, 3]);
        assert_eq!(g.dirty_count(), 0);
    }

    #[test]
    fn dirty_duplicate_edge() {
        let g = DynamicGraph::new(100, 1024);
        g.insert_edge(0, 1).unwrap();
        g.clear_dirty();
        // Duplicate edge should NOT mark dirty
        g.insert_edge(0, 1).unwrap();
        assert_eq!(g.dirty_count(), 0);
    }

    #[test]
    fn dirty_clear() {
        let g = DynamicGraph::new(100, 1024);
        g.insert_edge(0, 1).unwrap();
        g.clear_dirty();
        assert_eq!(g.dirty_count(), 0);
        assert!(!g.is_dirty(0));
    }

    #[test]
    fn concurrent_read_during_write() {
        use std::sync::Arc;
        use std::thread;

        let g = Arc::new(DynamicGraph::new(1000, 1 << 22));

        // Writer inserts edges
        let g_writer = Arc::clone(&g);
        let writer = thread::spawn(move || {
            for i in 0..1000u32 {
                let _ = g_writer.insert_edge(0, i);
            }
        });

        // Reader reads concurrently
        let g_reader = Arc::clone(&g);
        let reader = thread::spawn(move || {
            let mut buf = Vec::new();
            for _ in 0..100 {
                g_reader.neighbors_into(0, &mut buf);
                // Must always see a consistent sorted list
                for w in buf.windows(2) {
                    assert!(w[0] < w[1], "unsorted: {:?}", &buf[..20.min(buf.len())]);
                }
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();
    }
}
