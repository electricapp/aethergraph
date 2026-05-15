//! C-tree: balanced tree of sorted chunks for a single vertex's neighbors.
//!
//! Functional (immutable) — inserts produce new nodes via path copying.
//! The old tree remains valid for concurrent readers.
//!
//! Tree shape:
//! - Leaf: a single Chunk (64 bytes, ≤15 neighbors)
//! - Interior: split_key + left/right child offsets + total count
//!
//! Nodes are arena-allocated (offsets, not pointers). A null offset (0xFFFFFFFF)
//! means "no node."

use crate::arena::Arena;
use crate::chunk::Chunk;

/// Sentinel for null/empty tree.
pub const NULL: u32 = u32::MAX;

/// Tag bit stored in the offset to distinguish leaves from interior nodes.
/// Bit 31 = 1 means interior node. Bit 31 = 0 means leaf (chunk).
const INTERIOR_BIT: u32 = 1 << 31;

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
        if is_leaf(self.root) {
            // SAFETY: leaf tag bit is clear, so root is a Chunk offset previously returned by Arena::alloc_write.
            let chunk: &Chunk = unsafe { arena.get(self.root) };
            chunk.len()
        } else {
            // SAFETY: interior tag bit is set; strip_tag yields the Interior offset stored on insert.
            let interior: &Interior = unsafe { arena.get(strip_tag(self.root)) };
            interior.count as usize
        }
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
        if self.root == NULL {
            return false;
        }
        contains_rec(self.root, arena, val)
    }

    /// Insert `val` into the tree. Returns a new root (path-copied).
    ///
    /// Returns [`InsertResult::Duplicate`] if `val` already exists, or
    /// [`InsertResult::ArenaFull`] if a fresh node could not be allocated.
    /// All new nodes are allocated from `arena`. The old tree is untouched.
    ///
    /// # Safety
    /// Single-writer invariant on the arena: only one thread may call `insert`
    /// concurrently for a given arena. See [`Arena::alloc`].
    #[inline]
    pub unsafe fn insert(&self, arena: &Arena, val: u32) -> InsertResult {
        if self.root == NULL {
            let chunk = Chunk::from_sorted(&[val]);
            // SAFETY: caller upholds single-writer invariant.
            return match unsafe { arena.alloc_write(chunk) } {
                Some(off) => InsertResult::Inserted(CTree { root: off }),
                None => InsertResult::ArenaFull,
            };
        }
        // SAFETY: caller upholds single-writer invariant.
        match unsafe { insert_rec(self.root, arena, val) } {
            NodeInsert::New(new_root) => InsertResult::Inserted(CTree { root: new_root }),
            NodeInsert::Duplicate => InsertResult::Duplicate,
            NodeInsert::ArenaFull => InsertResult::ArenaFull,
        }
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
// Recursive operations (no heap allocation — stack depth bounded by tree height)
// ---------------------------------------------------------------------------

fn visit_chunks<F: FnMut(&Chunk)>(offset: u32, arena: &Arena, f: &mut F) {
    if is_leaf(offset) {
        // SAFETY: leaf tag is clear, so offset is a Chunk allocated in this arena.
        let chunk: &Chunk = unsafe { arena.get(offset) };
        f(chunk);
    } else {
        // SAFETY: interior tag is set; strip_tag yields the Interior offset stored on insert.
        let node: &Interior = unsafe { arena.get(strip_tag(offset)) };
        visit_chunks(node.left, arena, f);
        visit_chunks(node.right, arena, f);
    }
}

fn contains_rec(offset: u32, arena: &Arena, val: u32) -> bool {
    if is_leaf(offset) {
        // SAFETY: leaf tag is clear, so offset is a Chunk allocated in this arena.
        let chunk: &Chunk = unsafe { arena.get(offset) };
        chunk.contains(val)
    } else {
        // SAFETY: interior tag is set; strip_tag yields the Interior offset stored on insert.
        let node: &Interior = unsafe { arena.get(strip_tag(offset)) };
        if val < node.split_key {
            contains_rec(node.left, arena, val)
        } else {
            contains_rec(node.right, arena, val)
        }
    }
}

/// Internal recursive insert result. Distinguishes duplicate from arena-full.
enum NodeInsert {
    New(u32),
    Duplicate,
    ArenaFull,
}

/// `?`-friendly: turns `Option<u32>` from arena allocation into a `NodeInsert`.
macro_rules! alloc_or_full {
    ($expr:expr) => {
        match $expr {
            Some(v) => v,
            None => return NodeInsert::ArenaFull,
        }
    };
}

/// Insert into subtree rooted at `offset`.
///
/// # Safety
/// Single-writer invariant on the arena. See [`Arena::alloc`].
unsafe fn insert_rec(offset: u32, arena: &Arena, val: u32) -> NodeInsert {
    if is_leaf(offset) {
        // SAFETY: leaf tag is clear, so offset is a Chunk allocated in this arena.
        let chunk: &Chunk = unsafe { arena.get(offset) };

        if chunk.is_full() {
            let (left, right) = chunk.split();
            let split_key = right.min();

            let (new_left, new_right) = if val < split_key {
                match left.insert(val) {
                    Some(l) => (l, right),
                    None => return NodeInsert::Duplicate,
                }
            } else {
                match right.insert(val) {
                    Some(r) => (left, r),
                    None => return NodeInsert::Duplicate,
                }
            };

            // SAFETY: caller upholds single-writer invariant on the arena.
            let left_off = alloc_or_full!(unsafe { arena.alloc_write(new_left) });
            // SAFETY: same invariant.
            let right_off = alloc_or_full!(unsafe { arena.alloc_write(new_right) });
            let total = new_left.len() + new_right.len();
            let interior = Interior {
                left: left_off,
                right: right_off,
                count: total as u32,
                split_key,
            };
            // SAFETY: same invariant.
            let int_off = alloc_or_full!(unsafe { arena.alloc_write(interior) });
            NodeInsert::New(tag_interior(int_off))
        } else {
            let new_chunk = match chunk.insert(val) {
                Some(c) => c,
                None => return NodeInsert::Duplicate,
            };
            // SAFETY: caller upholds single-writer invariant.
            let off = unsafe { alloc_or_full!(arena.alloc_write(new_chunk)) };
            NodeInsert::New(off)
        }
    } else {
        // SAFETY: interior tag is set; strip_tag yields the Interior offset stored on insert.
        let node: &Interior = unsafe { arena.get(strip_tag(offset)) };

        let (new_left, new_right) = if val < node.split_key {
            // SAFETY: caller upholds single-writer invariant.
            let new_left = match unsafe { insert_rec(node.left, arena, val) } {
                NodeInsert::New(off) => off,
                other => return other,
            };
            (new_left, node.right)
        } else {
            // SAFETY: caller upholds single-writer invariant.
            let new_right = match unsafe { insert_rec(node.right, arena, val) } {
                NodeInsert::New(off) => off,
                other => return other,
            };
            (node.left, new_right)
        };

        let new_interior = Interior {
            left: new_left,
            right: new_right,
            count: node.count + 1,
            split_key: node.split_key,
        };
        // SAFETY: caller upholds single-writer invariant on the arena.
        let off = alloc_or_full!(unsafe { arena.alloc_write(new_interior) });
        NodeInsert::New(tag_interior(off))
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
}
