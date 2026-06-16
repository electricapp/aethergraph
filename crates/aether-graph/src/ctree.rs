//! C-tree: scapegoat-balanced tree of sorted chunks for a single vertex's
//! neighbors.
//!
//! Functional (immutable) — inserts produce new nodes via path copying.
//! The old tree remains valid for concurrent readers.
//!
//! Tree shape:
//! - Leaf: a single Chunk (64 bytes, ≤15 neighbors)
//! - Interior: split_key + left/right child offsets + total count
//!
//! Balance: scapegoat scheme with α = 2/3. Every insert records its
//! root-to-leaf path; if the path exceeds the α-height bound
//! `log_{3/2}(count)`, the highest α-weight-unbalanced node on the path
//! is rebuilt into a perfectly weight-balanced subtree. Rebuilds are
//! ordinary path-copied allocations, so persistence is preserved: old
//! roots keep reading the old nodes. This bounds depth at
//! O(log count) regardless of insertion order (ascending neighbor IDs
//! are the common streaming case) and keeps per-insert path-copy cost
//! logarithmic.
//!
//! Nodes are arena-allocated (offsets, not pointers). A null offset (0xFFFFFFFF)
//! means "no node."

use crate::arena::Arena;
use crate::chunk::Chunk;

/// Sentinel for null/empty tree.
pub const NULL: u32 = u32::MAX;

/// Tag bit stored in the offset to distinguish leaves from interior nodes.
/// Bit 31 = 1 means interior node. Bit 31 = 0 means leaf (chunk).
/// [`Arena::MAX_CAPACITY`] caps arena offsets below this bit so a real
/// offset can never collide with the tag.
const INTERIOR_BIT: u32 = 1 << 31;

// The tag scheme is only sound if no arena offset can have bit 31 set.
const _: () = assert!(Arena::MAX_CAPACITY <= INTERIOR_BIT as usize);

/// Hard bound on tree depth (interior nodes on a root→leaf path).
///
/// The scapegoat invariant keeps depth ≤ log_{3/2}(count) + 1. The arena
/// holds at most [`Arena::MAX_CAPACITY`] = 2^31 bytes and each element
/// costs ≥ 4 bytes, so count < 2^29 and depth < 51. 64 leaves headroom.
pub(crate) const MAX_DEPTH: usize = 64;

/// `?`-friendly: turns `Option<u32>` from arena allocation into an
/// `InsertResult::ArenaFull` early return.
macro_rules! alloc_or_full {
    ($expr:expr) => {
        match $expr {
            Some(v) => v,
            None => return InsertResult::ArenaFull,
        }
    };
}

/// Result of inserting into a C-tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InsertResult {
    /// Insert succeeded; the new tree root.
    Inserted(CTree),
    /// Value was already present; the tree is unchanged.
    Duplicate,
    /// Arena is full; the tree is unchanged.
    ArenaFull,
}

/// Interior node of the C-tree. 16 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Interior {
    /// Offset to left child (leaf or interior).
    pub left: u32,
    /// Offset to right child (leaf or interior).
    pub right: u32,
    /// Total neighbor count in this subtree.
    pub count: u32,
    /// Split key: all values in left < split_key, all in right >= split_key.
    pub split_key: u32,
}

/// A C-tree root is just an offset into an arena.
/// NULL means empty (no neighbors).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CTree {
    pub root: u32,
}

impl CTree {
    /// Empty tree (no neighbors).
    #[inline]
    pub const fn empty() -> Self {
        Self { root: NULL }
    }

    /// Is this tree empty?
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.root == NULL
    }

    /// Total number of neighbors in this tree.
    #[inline]
    pub fn count(&self, arena: &Arena) -> usize {
        if self.root == NULL {
            return 0;
        }
        subtree_size(self.root, arena)
    }

    /// Iterate all neighbors in sorted order. Calls `f` for each chunk.
    /// Zero allocations.
    #[inline]
    pub fn for_each_chunk<F: FnMut(&Chunk)>(&self, arena: &Arena, mut f: F) {
        if self.root != NULL {
            visit_chunks(self.root, arena, &mut f);
        }
    }

    /// Collect all neighbors into a pre-allocated buffer.
    #[inline]
    pub fn collect_into(&self, arena: &Arena, buf: &mut Vec<u32>) {
        self.for_each_chunk(arena, |chunk| {
            buf.extend_from_slice(chunk.as_slice());
        });
    }

    /// Does this tree contain `val`?
    #[inline]
    pub fn contains(&self, arena: &Arena, val: u32) -> bool {
        let mut cur = self.root;
        if cur == NULL {
            return false;
        }
        while !is_leaf(cur) {
            // SAFETY: interior tag is set; strip_tag yields the Interior offset stored on insert.
            let node: &Interior = unsafe { arena.get(strip_tag(cur)) };
            cur = if val < node.split_key {
                node.left
            } else {
                node.right
            };
        }
        // SAFETY: leaf tag is clear, so cur is a Chunk allocated in this arena.
        let chunk: &Chunk = unsafe { arena.get(cur) };
        chunk.contains(val)
    }

    /// Build a balanced tree over a sorted, deduplicated slice.
    ///
    /// Used by graph compaction to rebuild adjacency lists with zero
    /// garbage. Returns `None` if the arena runs out of space.
    ///
    /// # Safety
    /// Single-writer invariant on the arena. See [`Arena::alloc`].
    pub(crate) unsafe fn from_sorted(arena: &Arena, vals: &[u32]) -> Option<CTree> {
        if vals.is_empty() {
            return Some(CTree::empty());
        }
        // SAFETY: caller upholds single-writer invariant.
        let root = unsafe { build_balanced(arena, vals)? };
        Some(CTree { root })
    }

    /// Insert `val` into the tree. Returns a new root (path-copied).
    ///
    /// Returns [`InsertResult::Duplicate`] if `val` already exists, or
    /// [`InsertResult::ArenaFull`] if a fresh node could not be allocated.
    /// All new nodes are allocated from `arena`. The old tree is untouched.
    ///
    /// Allocates a fresh scratch buffer per call for any rebalance.
    /// Callers on a hot insert path should use
    /// [`insert_with_scratch`](Self::insert_with_scratch) with a reused
    /// buffer instead.
    ///
    /// # Safety
    /// Single-writer invariant on the arena: only one thread may call `insert`
    /// concurrently for a given arena. See [`Arena::alloc`].
    pub unsafe fn insert(&self, arena: &Arena, val: u32) -> InsertResult {
        let mut scratch = Vec::new();
        // SAFETY: caller upholds the single-writer invariant.
        unsafe { self.insert_with_scratch(arena, val, &mut scratch) }
    }

    /// Insert `val`, reusing `scratch` for any scapegoat rebalance instead
    /// of allocating one. `scratch` is cleared on entry to a rebuild; its
    /// contents on return are unspecified. Semantics otherwise match
    /// [`insert`](Self::insert).
    ///
    /// # Safety
    /// Single-writer invariant on the arena: only one thread may insert
    /// concurrently for a given arena. See [`Arena::alloc`].
    pub(crate) unsafe fn insert_with_scratch(
        &self,
        arena: &Arena,
        val: u32,
        scratch: &mut Vec<u32>,
    ) -> InsertResult {
        if self.root == NULL {
            let chunk = Chunk::from_sorted(&[val]);
            // SAFETY: caller upholds single-writer invariant.
            return match unsafe { arena.alloc_write(chunk) } {
                Some(off) => InsertResult::Inserted(CTree { root: off }),
                None => InsertResult::ArenaFull,
            };
        }

        // Descend to the target leaf, recording the path of interior nodes.
        let mut path = [NULL; MAX_DEPTH];
        let mut went_left = [false; MAX_DEPTH];
        let mut depth = 0usize;
        let mut cur = self.root;
        while !is_leaf(cur) {
            assert!(depth < MAX_DEPTH, "C-tree exceeded MAX_DEPTH");
            // SAFETY: interior tag is set; strip_tag yields the Interior offset stored on insert.
            let node: &Interior = unsafe { arena.get(strip_tag(cur)) };
            path[depth] = cur;
            let left = val < node.split_key;
            went_left[depth] = left;
            depth += 1;
            cur = if left { node.left } else { node.right };
        }

        // Leaf insert (splitting a full chunk adds one level).
        // SAFETY: leaf tag is clear, so cur is a Chunk allocated in this arena.
        let chunk: &Chunk = unsafe { arena.get(cur) };
        let mut split_levels = 0usize;
        let mut new_child = if chunk.is_full() {
            let (left_c, right_c) = chunk.split();
            let split_key = right_c.min();
            let (new_left, new_right) = if val < split_key {
                match left_c.insert(val) {
                    Some(l) => (l, right_c),
                    None => return InsertResult::Duplicate,
                }
            } else {
                match right_c.insert(val) {
                    Some(r) => (left_c, r),
                    None => return InsertResult::Duplicate,
                }
            };
            // SAFETY: caller upholds single-writer invariant on the arena.
            let left_off = alloc_or_full!(unsafe { arena.alloc_write(new_left) });
            // SAFETY: same invariant.
            let right_off = alloc_or_full!(unsafe { arena.alloc_write(new_right) });
            let interior = Interior {
                left: left_off,
                right: right_off,
                count: (new_left.len() + new_right.len()) as u32,
                split_key,
            };
            // SAFETY: same invariant.
            let int_off = alloc_or_full!(unsafe { arena.alloc_write(interior) });
            split_levels = 1;
            tag_interior(int_off)
        } else {
            let new_chunk = match chunk.insert(val) {
                Some(c) => c,
                None => return InsertResult::Duplicate,
            };
            // SAFETY: caller upholds single-writer invariant.
            alloc_or_full!(unsafe { arena.alloc_write(new_chunk) })
        };

        // Path-copy ancestors bottom-up.
        let mut new_path = [NULL; MAX_DEPTH];
        for i in (0..depth).rev() {
            // SAFETY: path[i] was recorded as a tagged interior during descent.
            let old: &Interior = unsafe { arena.get(strip_tag(path[i])) };
            let (l, r) = if went_left[i] {
                (new_child, old.right)
            } else {
                (old.left, new_child)
            };
            let copied = Interior {
                left: l,
                right: r,
                count: old.count + 1,
                split_key: old.split_key,
            };
            // SAFETY: caller upholds single-writer invariant.
            new_child = tag_interior(alloc_or_full!(unsafe { arena.alloc_write(copied) }));
            new_path[i] = new_child;
        }
        let new_root = new_child;

        // Scapegoat rebalance: if this insert pushed the path past the
        // α-height bound, rebuild the highest α-weight-unbalanced node on
        // it. If the rebuild allocation fails we still return the (valid,
        // temporarily too-deep) new tree — the element is inserted either
        // way and a later insert retries the rebalance.
        let total = subtree_size(new_root, arena);
        if depth + split_levels > depth_limit(total) {
            // SAFETY: caller upholds single-writer invariant.
            match unsafe {
                rebalance(
                    arena,
                    new_root,
                    &new_path[..depth],
                    &went_left[..depth],
                    scratch,
                )
            } {
                Some(balanced_root) => {
                    return InsertResult::Inserted(CTree {
                        root: balanced_root,
                    });
                }
                None => return InsertResult::Inserted(CTree { root: new_root }),
            }
        }

        InsertResult::Inserted(CTree { root: new_root })
    }
}

// ---------------------------------------------------------------------------
// Tag helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn is_leaf(offset: u32) -> bool {
    offset & INTERIOR_BIT == 0
}

#[inline(always)]
fn tag_interior(offset: u32) -> u32 {
    offset | INTERIOR_BIT
}

#[inline(always)]
fn strip_tag(offset: u32) -> u32 {
    offset & !INTERIOR_BIT
}

// ---------------------------------------------------------------------------
// Balance helpers
// ---------------------------------------------------------------------------

/// Number of elements under `offset` (leaf or interior).
#[inline]
fn subtree_size(offset: u32, arena: &Arena) -> usize {
    if is_leaf(offset) {
        // SAFETY: leaf tag is clear, so offset is a Chunk allocated in this arena.
        let chunk: &Chunk = unsafe { arena.get(offset) };
        chunk.len()
    } else {
        // SAFETY: interior tag is set; strip_tag yields the Interior offset stored on insert.
        let node: &Interior = unsafe { arena.get(strip_tag(offset)) };
        node.count as usize
    }
}

/// α-height bound for a tree of `n` elements: floor(log_{3/2}(n)).
/// A path deeper than this proves some node on it is α-weight-unbalanced
/// (α = 2/3), i.e. a scapegoat exists.
#[inline]
fn depth_limit(n: usize) -> usize {
    // log_{3/2}(n) = log2(n) / log2(1.5) = log2(n) * (1 / log2(1.5)).
    // Multiplying by the precomputed reciprocal keeps this off the
    // floating-point divide on every insert's write path.
    const INV_LOG2_1_5: f64 = 1.709511291351454;
    ((n.max(2) as f64).log2() * INV_LOG2_1_5) as usize
}

/// Is `3 * child > 2 * parent` for either child — the α = 2/3 weight check.
fn alpha_unbalanced(node: &Interior, arena: &Arena) -> bool {
    let total = node.count as usize;
    let left = subtree_size(node.left, arena);
    let right = subtree_size(node.right, arena);
    3 * left > 2 * total || 3 * right > 2 * total
}

/// Find the highest α-weight-unbalanced interior on the freshly copied
/// path and rebuild it into a perfectly balanced subtree, then re-copy
/// the ancestors above it. Falls back to rebuilding the whole tree if no
/// path node trips the weight check. Returns the new root, or `None` if
/// the arena filled up mid-rebuild.
///
/// `scratch` is reused to gather the scapegoat subtree's elements; it is
/// cleared before use, so any prior contents are discarded.
///
/// # Safety
/// Single-writer invariant on the arena. See [`Arena::alloc`].
unsafe fn rebalance(
    arena: &Arena,
    new_root: u32,
    new_path: &[u32],
    went_left: &[bool],
    scratch: &mut Vec<u32>,
) -> Option<u32> {
    let mut scapegoat = 0usize; // rebuild the root unless a deeper node is unbalanced
    for (i, &off) in new_path.iter().enumerate() {
        // SAFETY: new_path holds tagged interiors created by this insert.
        let node: &Interior = unsafe { arena.get(strip_tag(off)) };
        if alpha_unbalanced(node, arena) {
            scapegoat = i;
            break;
        }
    }

    let target = if new_path.is_empty() {
        new_root
    } else {
        new_path[scapegoat]
    };

    // Collect the scapegoat subtree's elements (sorted by construction)
    // and rebuild it perfectly weight-balanced. The scratch buffer is
    // reused across inserts to keep the write path allocation-free.
    scratch.clear();
    scratch.reserve(subtree_size(target, arena));
    visit_chunks(target, arena, &mut |chunk: &Chunk| {
        scratch.extend_from_slice(chunk.as_slice());
    });
    // SAFETY: caller upholds single-writer invariant.
    let mut child = unsafe { build_balanced(arena, &scratch[..])? };

    // Re-copy the ancestors above the scapegoat (counts and split keys
    // are unchanged — only the child pointer moved).
    for i in (0..scapegoat).rev() {
        // SAFETY: new_path holds tagged interiors created by this insert.
        let old: &Interior = unsafe { arena.get(strip_tag(new_path[i])) };
        let (l, r) = if went_left[i] {
            (child, old.right)
        } else {
            (old.left, child)
        };
        let copied = Interior {
            left: l,
            right: r,
            count: old.count,
            split_key: old.split_key,
        };
        // SAFETY: caller upholds single-writer invariant.
        child = tag_interior(unsafe { arena.alloc_write(copied) }?);
    }
    Some(child)
}

/// Build a perfectly weight-balanced subtree over `vals` (sorted, unique,
/// non-empty). Splits at the element midpoint, so every interior node is
/// α-weight-balanced for any α ≥ 1/2 and depth is ceil(log2(len/15)) + 1.
///
/// # Safety
/// Single-writer invariant on the arena. See [`Arena::alloc`].
unsafe fn build_balanced(arena: &Arena, vals: &[u32]) -> Option<u32> {
    debug_assert!(!vals.is_empty());
    if vals.len() <= crate::chunk::CHUNK_CAP {
        // SAFETY: caller upholds single-writer invariant.
        return unsafe { arena.alloc_write(Chunk::from_sorted(vals)) };
    }
    let mid = vals.len() / 2;
    let (left_vals, right_vals) = vals.split_at(mid);
    // SAFETY: caller upholds single-writer invariant.
    let left = unsafe { build_balanced(arena, left_vals)? };
    // SAFETY: same invariant.
    let right = unsafe { build_balanced(arena, right_vals)? };
    let interior = Interior {
        left,
        right,
        count: vals.len() as u32,
        split_key: right_vals[0],
    };
    // SAFETY: same invariant.
    Some(tag_interior(unsafe { arena.alloc_write(interior) }?))
}

// ---------------------------------------------------------------------------
// Traversal (iterative — depth is bounded by the scapegoat invariant, but
// an explicit stack keeps the read path independent of thread stack size)
// ---------------------------------------------------------------------------

fn visit_chunks<F: FnMut(&Chunk)>(root: u32, arena: &Arena, f: &mut F) {
    let mut stack = [NULL; MAX_DEPTH];
    let mut sp = 0usize;
    let mut cur = root;
    loop {
        while !is_leaf(cur) {
            // SAFETY: interior tag is set; strip_tag yields the Interior offset stored on insert.
            let node: &Interior = unsafe { arena.get(strip_tag(cur)) };
            assert!(sp < MAX_DEPTH, "C-tree exceeded MAX_DEPTH");
            stack[sp] = node.right;
            sp += 1;
            cur = node.left;
        }
        // SAFETY: leaf tag is clear, so cur is a Chunk allocated in this arena.
        let chunk: &Chunk = unsafe { arena.get(cur) };
        f(chunk);
        if sp == 0 {
            break;
        }
        sp -= 1;
        cur = stack[sp];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::CHUNK_CAP;

    fn make_arena() -> Arena {
        Arena::new(1 << 20)
    }

    fn insert_assert(tree: CTree, arena: &Arena, val: u32) -> CTree {
        // SAFETY: tests are single-threaded.
        match unsafe { tree.insert(arena, val) } {
            InsertResult::Inserted(t) => t,
            other => panic!("expected Inserted, got {other:?}"),
        }
    }

    /// Depth of the deepest leaf (interior nodes on the path).
    fn max_depth(offset: u32, arena: &Arena) -> usize {
        if offset == NULL || is_leaf(offset) {
            return 0;
        }
        // SAFETY: tree invariant — tagged offsets point at valid Interiors.
        let node: &Interior = unsafe { arena.get(strip_tag(offset)) };
        1 + max_depth(node.left, arena).max(max_depth(node.right, arena))
    }

    #[test]
    fn empty_tree() {
        let arena = make_arena();
        let tree = CTree::empty();
        assert!(tree.is_empty());
        assert_eq!(tree.count(&arena), 0);
        assert!(!tree.contains(&arena, 42));
    }

    #[test]
    fn insert_one() {
        let arena = make_arena();
        let tree = insert_assert(CTree::empty(), &arena, 42);
        assert!(!tree.is_empty());
        assert_eq!(tree.count(&arena), 1);
        assert!(tree.contains(&arena, 42));
        assert!(!tree.contains(&arena, 99));
    }

    #[test]
    fn insert_sorted_order() {
        let arena = make_arena();
        let mut tree = CTree::empty();
        for i in 0..10u32 {
            tree = insert_assert(tree, &arena, i * 10);
        }
        assert_eq!(tree.count(&arena), 10);

        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        let expected: Vec<u32> = (0..10).map(|i| i * 10).collect();
        assert_eq!(buf, expected);
    }

    #[test]
    fn insert_reverse_order() {
        let arena = make_arena();
        let mut tree = CTree::empty();
        for i in (0..10u32).rev() {
            tree = insert_assert(tree, &arena, i * 10);
        }
        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        let expected: Vec<u32> = (0..10).map(|i| i * 10).collect();
        assert_eq!(buf, expected);
    }

    #[test]
    fn insert_random_order() {
        let arena = make_arena();
        let mut tree = CTree::empty();
        let vals = [50, 20, 80, 10, 30, 70, 90, 5, 15, 25, 35, 60, 75, 85, 95];
        for &v in &vals {
            tree = insert_assert(tree, &arena, v);
        }
        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        let mut expected = vals.to_vec();
        expected.sort();
        assert_eq!(buf, expected);
    }

    #[test]
    fn insert_duplicate_returns_duplicate() {
        let arena = make_arena();
        let tree = insert_assert(CTree::empty(), &arena, 42);
        // SAFETY: test is single-threaded.
        let result = unsafe { tree.insert(&arena, 42) };
        assert!(matches!(result, InsertResult::Duplicate));
    }

    #[test]
    fn arena_full_returns_arena_full() {
        // Tiny arena: only enough for a single chunk allocation, then split needs more.
        let arena = Arena::new(64);
        let mut tree = CTree::empty();
        // Fill until ArenaFull.
        let mut val = 0u32;
        loop {
            // SAFETY: test is single-threaded.
            match unsafe { tree.insert(&arena, val) } {
                InsertResult::Inserted(t) => tree = t,
                InsertResult::ArenaFull => return,
                InsertResult::Duplicate => unreachable!("monotonic vals"),
            }
            val += 1;
            assert!(val < 1_000_000, "should hit ArenaFull before 1M iterations");
        }
    }

    #[test]
    fn triggers_chunk_split() {
        let arena = make_arena();
        let mut tree = CTree::empty();
        for i in 0..=CHUNK_CAP as u32 {
            tree = insert_assert(tree, &arena, i);
        }
        assert_eq!(tree.count(&arena), CHUNK_CAP + 1);

        for i in 0..=CHUNK_CAP as u32 {
            assert!(tree.contains(&arena, i), "missing {}", i);
        }

        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        let expected: Vec<u32> = (0..=CHUNK_CAP as u32).collect();
        assert_eq!(buf, expected);
    }

    #[test]
    fn large_tree() {
        let arena = Arena::new(4 << 20);
        let mut tree = CTree::empty();
        let n = 1000u32;
        for i in 0..n {
            tree = insert_assert(tree, &arena, i * 3);
        }
        assert_eq!(tree.count(&arena), n as usize);

        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        assert_eq!(buf.len(), n as usize);
        for w in buf.windows(2) {
            assert!(w[0] < w[1]);
        }
        for i in 0..n {
            assert!(tree.contains(&arena, i * 3));
        }
    }

    #[test]
    fn functional_persistence() {
        let arena = make_arena();
        let tree1 = CTree::empty();
        let tree2 = insert_assert(tree1, &arena, 10);
        let tree3 = insert_assert(tree2, &arena, 20);

        assert_eq!(tree1.count(&arena), 0);
        assert_eq!(tree2.count(&arena), 1);
        assert!(tree2.contains(&arena, 10));
        assert!(!tree2.contains(&arena, 20));
        assert_eq!(tree3.count(&arena), 2);
        assert!(tree3.contains(&arena, 10));
        assert!(tree3.contains(&arena, 20));
    }

    #[test]
    fn for_each_chunk_visits_all() {
        let arena = make_arena();
        let mut tree = CTree::empty();
        for i in 0..50u32 {
            tree = insert_assert(tree, &arena, i);
        }

        let mut total = 0;
        tree.for_each_chunk(&arena, |chunk| {
            total += chunk.len();
            let s = chunk.as_slice();
            for w in s.windows(2) {
                assert!(w[0] < w[1]);
            }
        });
        assert_eq!(total, 50);
    }

    #[test]
    fn ascending_inserts_stay_balanced() {
        // The streaming-ingest worst case: monotonically increasing IDs.
        // Without rebalancing this builds an O(n/7.5)-deep right spine
        // (quadratic arena use + stack overflow); the scapegoat invariant
        // must keep depth logarithmic and arena use linear-ish.
        let n: u32 = 100_000;
        let arena = Arena::new(256 << 20);
        let mut tree = CTree::empty();
        for i in 0..n {
            tree = insert_assert(tree, &arena, i);
        }
        assert_eq!(tree.count(&arena), n as usize);

        let depth = max_depth(tree.root, &arena);
        let limit = depth_limit(n as usize);
        assert!(
            depth <= limit + 1,
            "depth {depth} exceeds scapegoat bound {limit}+1 for n={n}"
        );

        // Arena use must be far below the quadratic blowup (~10GB for
        // this workload without balancing).
        assert!(
            arena.used() < 128 << 20,
            "arena use {} suggests quadratic path-copying",
            arena.used()
        );

        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        assert_eq!(buf.len(), n as usize);
        for w in buf.windows(2) {
            assert!(w[0] < w[1]);
        }
        for i in (0..n).step_by(997) {
            assert!(tree.contains(&arena, i));
        }
    }

    #[test]
    fn descending_inserts_stay_balanced() {
        let n: u32 = 20_000;
        let arena = Arena::new(64 << 20);
        let mut tree = CTree::empty();
        for i in (0..n).rev() {
            tree = insert_assert(tree, &arena, i);
        }
        assert_eq!(tree.count(&arena), n as usize);
        let depth = max_depth(tree.root, &arena);
        let limit = depth_limit(n as usize);
        assert!(depth <= limit + 1, "depth {depth} > bound {limit}+1");
    }

    #[test]
    fn random_inserts_respect_depth_bound() {
        // xorshift so the test is deterministic without a rand dependency.
        let mut state = 0x9E37_79B9u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let arena = Arena::new(64 << 20);
        let mut tree = CTree::empty();
        let mut inserted = 0usize;
        for _ in 0..30_000 {
            // SAFETY: test is single-threaded.
            if let InsertResult::Inserted(t) = unsafe { tree.insert(&arena, next() % 1_000_000) } {
                tree = t;
                inserted += 1;
            }
        }
        assert_eq!(tree.count(&arena), inserted);
        let depth = max_depth(tree.root, &arena);
        let limit = depth_limit(inserted);
        assert!(depth <= limit + 1, "depth {depth} > bound {limit}+1");
    }

    #[test]
    fn from_sorted_builds_balanced_tree() {
        let arena = Arena::new(16 << 20);
        let vals: Vec<u32> = (0..50_000u32).map(|i| i * 2).collect();
        // SAFETY: test is single-threaded.
        let tree = unsafe { CTree::from_sorted(&arena, &vals) }.unwrap();
        assert_eq!(tree.count(&arena), vals.len());

        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        assert_eq!(buf, vals);

        let depth = max_depth(tree.root, &arena);
        // Perfectly balanced: depth ≈ log2(n / CHUNK_CAP) + 1.
        assert!(depth <= 14, "from_sorted produced depth {depth}");

        assert!(tree.contains(&arena, 0));
        assert!(tree.contains(&arena, 99_998));
        assert!(!tree.contains(&arena, 1));

        // Empty input → empty tree.
        // SAFETY: test is single-threaded.
        let empty = unsafe { CTree::from_sorted(&arena, &[]) }.unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn persistence_survives_rebalance() {
        // Snapshots taken before a rebuild must keep reading the old nodes.
        let arena = Arena::new(64 << 20);
        let mut tree = CTree::empty();
        let mut snapshots: Vec<(CTree, usize)> = Vec::new();
        for i in 0..10_000u32 {
            if i % 1000 == 0 {
                snapshots.push((tree, tree.count(&arena)));
            }
            tree = insert_assert(tree, &arena, i);
        }
        for (snap, expected_count) in snapshots {
            assert_eq!(snap.count(&arena), expected_count);
            let mut buf = Vec::new();
            snap.collect_into(&arena, &mut buf);
            assert_eq!(buf.len(), expected_count);
            for w in buf.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }
}
