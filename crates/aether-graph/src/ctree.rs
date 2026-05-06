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
            let chunk: &Chunk = unsafe { arena.get(self.root) };
            chunk.len()
        } else {
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
    /// Returns the number of neighbors written.
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
    /// Returns None if `val` already exists (duplicate).
    ///
    /// All new nodes are allocated from `arena`. The old tree is untouched.
    #[inline]
    pub fn insert(&self, arena: &Arena, val: u32) -> Option<CTree> {
        if self.root == NULL {
            // First neighbor: create a single-element chunk.
            let chunk = Chunk::from_sorted(&[val]);
            let off = arena.alloc_write(chunk)?;
            return Some(CTree { root: off });
        }
        let new_root = insert_rec(self.root, arena, val)?;
        Some(CTree { root: new_root })
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
        let chunk: &Chunk = unsafe { arena.get(offset) };
        f(chunk);
    } else {
        let node: &Interior = unsafe { arena.get(strip_tag(offset)) };
        visit_chunks(node.left, arena, f);
        visit_chunks(node.right, arena, f);
    }
}

fn contains_rec(offset: u32, arena: &Arena, val: u32) -> bool {
    if is_leaf(offset) {
        let chunk: &Chunk = unsafe { arena.get(offset) };
        chunk.contains(val)
    } else {
        let node: &Interior = unsafe { arena.get(strip_tag(offset)) };
        if val < node.split_key {
            contains_rec(node.left, arena, val)
        } else {
            contains_rec(node.right, arena, val)
        }
    }
}

/// Insert into subtree rooted at `offset`. Returns new root offset.
/// Returns None if duplicate or arena full.
fn insert_rec(offset: u32, arena: &Arena, val: u32) -> Option<u32> {
    if is_leaf(offset) {
        let chunk: &Chunk = unsafe { arena.get(offset) };

        if chunk.is_full() {
            // Split the chunk, then insert into the appropriate half.
            let (left, right) = chunk.split();
            let split_key = right.min();

            let (new_left, new_right) = if val < split_key {
                (left.insert(val)?, right)
            } else {
                match right.insert(val) {
                    Some(r) => (left, r),
                    None => return None, // duplicate
                }
            };

            let left_off = arena.alloc_write(new_left)?;
            let right_off = arena.alloc_write(new_right)?;
            let total = new_left.len() + new_right.len();
            let interior = Interior {
                left: left_off,
                right: right_off,
                count: total as u32,
                split_key,
            };
            let int_off = arena.alloc_write(interior)?;
            Some(tag_interior(int_off))
        } else {
            // Chunk has space — insert and return new chunk.
            let new_chunk = chunk.insert(val)?;
            let off = arena.alloc_write(new_chunk)?;
            Some(off)
        }
    } else {
        let node: &Interior = unsafe { arena.get(strip_tag(offset)) };

        if val < node.split_key {
            let new_left = insert_rec(node.left, arena, val)?;
            let new_interior = Interior {
                left: new_left,
                right: node.right,
                count: node.count + 1,
                split_key: node.split_key,
            };
            let off = arena.alloc_write(new_interior)?;
            Some(tag_interior(off))
        } else {
            let new_right = insert_rec(node.right, arena, val)?;
            let new_interior = Interior {
                left: node.left,
                right: new_right,
                count: node.count + 1,
                split_key: node.split_key,
            };
            let off = arena.alloc_write(new_interior)?;
            Some(tag_interior(off))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::CHUNK_CAP;

    fn make_arena() -> Arena {
        Arena::new(1 << 20) // 1MB
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
        let tree = CTree::empty();
        let tree = tree.insert(&arena, 42).unwrap();
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
            tree = tree.insert(&arena, i * 10).unwrap();
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
            tree = tree.insert(&arena, i * 10).unwrap();
        }
        assert_eq!(tree.count(&arena), 10);

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
            tree = tree.insert(&arena, v).unwrap();
        }
        assert_eq!(tree.count(&arena), vals.len());

        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        let mut expected = vals.to_vec();
        expected.sort();
        assert_eq!(buf, expected);
    }

    #[test]
    fn insert_duplicate_returns_none() {
        let arena = make_arena();
        let tree = CTree::empty();
        let tree = tree.insert(&arena, 42).unwrap();
        assert!(tree.insert(&arena, 42).is_none());
    }

    #[test]
    fn triggers_chunk_split() {
        let arena = make_arena();
        let mut tree = CTree::empty();
        // Insert CHUNK_CAP + 1 elements to force a split
        for i in 0..=CHUNK_CAP as u32 {
            tree = tree.insert(&arena, i).unwrap();
        }
        assert_eq!(tree.count(&arena), CHUNK_CAP + 1);

        // Verify all elements present
        for i in 0..=CHUNK_CAP as u32 {
            assert!(tree.contains(&arena, i), "missing {}", i);
        }

        // Verify sorted order
        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        let expected: Vec<u32> = (0..=CHUNK_CAP as u32).collect();
        assert_eq!(buf, expected);
    }

    #[test]
    fn large_tree() {
        let arena = Arena::new(4 << 20); // 4MB — path copying creates many nodes
        let mut tree = CTree::empty();
        let n = 1000u32;
        for i in 0..n {
            tree = tree.insert(&arena, i * 3).unwrap();
        }
        assert_eq!(tree.count(&arena), n as usize);

        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        assert_eq!(buf.len(), n as usize);
        // Verify sorted
        for w in buf.windows(2) {
            assert!(w[0] < w[1]);
        }
        // Verify all present
        for i in 0..n {
            assert!(tree.contains(&arena, i * 3));
        }
    }

    #[test]
    fn functional_persistence() {
        let arena = make_arena();
        let tree1 = CTree::empty();
        let tree2 = tree1.insert(&arena, 10).unwrap();
        let tree3 = tree2.insert(&arena, 20).unwrap();

        // tree1 is still empty
        assert_eq!(tree1.count(&arena), 0);
        // tree2 has one element
        assert_eq!(tree2.count(&arena), 1);
        assert!(tree2.contains(&arena, 10));
        assert!(!tree2.contains(&arena, 20));
        // tree3 has two elements
        assert_eq!(tree3.count(&arena), 2);
        assert!(tree3.contains(&arena, 10));
        assert!(tree3.contains(&arena, 20));
    }

    #[test]
    fn for_each_chunk_visits_all() {
        let arena = make_arena();
        let mut tree = CTree::empty();
        for i in 0..50u32 {
            tree = tree.insert(&arena, i).unwrap();
        }

        let mut total = 0;
        tree.for_each_chunk(&arena, |chunk| {
            total += chunk.len();
            // Each chunk should be sorted
            let s = chunk.as_slice();
            for w in s.windows(2) {
                assert!(w[0] < w[1]);
            }
        });
        assert_eq!(total, 50);
    }
}
