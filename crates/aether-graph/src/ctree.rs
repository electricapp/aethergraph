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
//! means "no node." Mutation goes through the arena's [`ArenaWriter`]
//! handle, whose construction carries the single-writer proof — the write
//! operations here are safe functions over `&mut ArenaWriter`.
//!
//! Every tagged offset handled in this module refers to a live node of
//! the arena it is used with: offsets originate from this module's own
//! allocations and are only ever stored in nodes or published roots. The
//! read API leans on that crate invariant (see [`node_at`]).

use crate::arena::{Arena, ArenaWriter, RegionWriter, RetireLog};
use crate::chunk::Chunk;
use std::mem::MaybeUninit;

// The arena's slot classes are sized for exactly these node types.
const _: () = assert!(std::mem::size_of::<Chunk>() == crate::arena::CHUNK_SLOT);
const _: () = assert!(std::mem::align_of::<Chunk>() <= 64);
const _: () = assert!(std::mem::size_of::<Interior>() == crate::arena::INTERIOR_SLOT);
const _: () = assert!(std::mem::align_of::<Interior>() <= 16);

/// Hint the CPU to pull the line containing `ptr` toward L1. Prefetch is a
/// pure hint — any address is architecturally safe — so this wraps the
/// per-arch instruction without touching memory.
#[inline(always)]
fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetch has no memory effects; any address is safe.
    unsafe {
        core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0)
    };
    #[cfg(target_arch = "aarch64")]
    // SAFETY: prfm has no memory effects; any address is safe.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{p}]",
            p = in(reg) ptr.cast::<u8>(),
            options(nostack, preserves_flags),
        );
    };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = ptr;
}

/// Sentinel for null/empty tree.
pub const NULL: u32 = u32::MAX;

/// Tag bit stored in the offset to distinguish leaves from interior nodes.
/// Bit 31 = 1 means interior node. Bit 31 = 0 means leaf (chunk).
/// Offsets are *slot indices* into the arena's per-kind regions; the
/// arena's allocators never hand out an index with bit 31 set, so a real
/// index can never collide with the tag.
const INTERIOR_BIT: u32 = 1 << 31;

/// Hard bound on tree depth (interior nodes on a root→leaf path).
///
/// The scapegoat invariant keeps depth ≤ log_{3/2}(count) + 1. The arena
/// holds at most [`Arena::MAX_CAPACITY`] = 32 GiB and each element costs
/// ≥ 4 bytes, so count < 2^33 and depth < 58. 64 leaves headroom.
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

/// Decoded view of one tagged node offset.
///
/// The tag bit packs a two-variant union into a u32; this enum is its
/// typed form. [`node_at`] is the single point where the packed form is
/// parsed — traversals `match` on the result instead of re-deriving the
/// class and accessor at each site.
pub(crate) enum NodeRef<'a> {
    Leaf(&'a Chunk),
    Interior(&'a Interior),
}

/// Decode the node at a tagged offset.
///
/// Relies on the module invariant that every tagged offset in circulation
/// names a live node of `arena` (offsets are created only by this
/// module's allocations and stored only in nodes and published roots),
/// and on the caller-side reader contract of the safe read API: hold a
/// [`ReadGuard`](crate::ReadGuard) or the write handle across the
/// traversal.
#[inline(always)]
pub(crate) fn node_at(arena: &Arena, tagged: u32) -> NodeRef<'_> {
    debug_assert_ne!(tagged, NULL, "node_at on NULL offset");
    if is_leaf(tagged) {
        // SAFETY: module invariant above — the offset names a live,
        // initialized chunk slot.
        NodeRef::Leaf(unsafe { arena.chunk(tagged) })
    } else {
        // SAFETY: module invariant above, interior class per the tag.
        NodeRef::Interior(unsafe { arena.interior(strip_tag(tagged)) })
    }
}

/// [`node_at`] for offsets the caller knows carry the interior tag
/// (path entries recorded during a descent).
#[inline(always)]
fn interior_at(arena: &Arena, tagged: u32) -> &Interior {
    debug_assert!(!is_leaf(tagged), "interior_at on a leaf offset");
    // SAFETY: module invariant (see node_at); interior class per the tag.
    unsafe { arena.interior(strip_tag(tagged)) }
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
        buf.reserve(self.count(arena));
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
        let mut cur = self.root;
        loop {
            match node_at(arena, cur) {
                NodeRef::Interior(node) => {
                    cur = if val < node.split_key {
                        node.left
                    } else {
                        node.right
                    };
                }
                NodeRef::Leaf(chunk) => return chunk.contains(val),
            }
        }
    }

    /// Build a balanced tree over a sorted, deduplicated slice.
    ///
    /// Used by bulk inserts and graph compaction to rebuild adjacency
    /// lists with zero garbage. Returns `None` if the arena runs out of
    /// space.
    pub(crate) fn from_sorted(aw: &mut ArenaWriter<'_>, vals: &[u32]) -> Option<Self> {
        if vals.is_empty() {
            return Some(Self::empty());
        }
        let root = build_balanced(aw, vals)?;
        Some(Self { root })
    }

    /// Insert `val` into the tree. Returns a new root (path-copied).
    ///
    /// Returns [`InsertResult::Duplicate`] if `val` already exists, or
    /// [`InsertResult::ArenaFull`] if a fresh node could not be allocated.
    /// All new nodes are allocated through `aw`. The old tree is untouched.
    ///
    /// Allocates a fresh scratch buffer and a discarded retire log per
    /// call. Callers on a hot insert path should use
    /// [`insert_with_scratch`](Self::insert_with_scratch) with reused
    /// buffers instead — that is also what makes superseded nodes
    /// recyclable rather than garbage.
    pub fn insert(&self, aw: &mut ArenaWriter<'_>, val: u32) -> InsertResult {
        let mut scratch = Vec::new();
        let mut retire = RetireLog::new();
        self.insert_with_scratch(aw, val, &mut scratch, &mut retire)
    }

    /// Insert `val`, reusing `scratch` for any scapegoat rebalance instead
    /// of allocating one. `scratch` is cleared on entry to a rebuild; its
    /// contents on return are unspecified. Semantics otherwise match
    /// [`insert`](Self::insert).
    ///
    /// Every node this insert supersedes (the replaced leaf, the copied
    /// interior path, and — on a rebalance — the rebuilt subtree) is
    /// recorded in `retire`, and only on success: a failed insert leaves
    /// the old tree fully live, so nothing may be logged for reuse.
    pub(crate) fn insert_with_scratch(
        &self,
        aw: &mut ArenaWriter<'_>,
        val: u32,
        scratch: &mut Vec<u32>,
        retire: &mut RetireLog,
    ) -> InsertResult {
        let arena = aw.arena();
        if self.root == NULL {
            let chunk = Chunk::from_sorted(&[val]);
            return match aw.alloc_write_chunk(chunk) {
                Some(off) => InsertResult::Inserted(Self { root: off }),
                None => InsertResult::ArenaFull,
            };
        }

        // Descend to the target leaf, recording the path of interior nodes.
        // The scratch arrays are deliberately uninitialized: only the first
        // `depth` entries are ever written and read, and zeroing all 576
        // bytes on every insert costs more than the tree work of the
        // common single-chunk case.
        let mut path = [MaybeUninit::<u32>::uninit(); MAX_DEPTH];
        let mut went_left = [MaybeUninit::<bool>::uninit(); MAX_DEPTH];
        let mut depth = 0usize;
        let mut cur = self.root;
        let chunk: &Chunk = loop {
            match node_at(arena, cur) {
                NodeRef::Interior(node) => {
                    // A path this deep only exists when rebalances have
                    // been failing for lack of arena space (see the
                    // rebalance-failure note below), so report the root
                    // cause rather than panic.
                    if depth >= MAX_DEPTH {
                        return InsertResult::ArenaFull;
                    }
                    path[depth].write(cur);
                    let left = val < node.split_key;
                    went_left[depth].write(left);
                    depth += 1;
                    cur = if left { node.left } else { node.right };
                }
                NodeRef::Leaf(chunk) => break chunk,
            }
        };

        // The post-insert element count is known before any copying: the
        // current root's count plus one (duplicates return before this
        // matters). Deciding the rebalance up front means the path-copy
        // loop only records `new_path` when a rebuild will actually
        // consume it, and no re-read of the fresh root is needed.
        let total = subtree_size(self.root, arena) + 1;

        // Leaf insert (splitting a full chunk adds one level).
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
            let left_off = alloc_or_full!(aw.alloc_write_chunk(new_left));
            let right_off = alloc_or_full!(aw.alloc_write_chunk(new_right));
            let interior = Interior {
                left: left_off,
                right: right_off,
                count: (new_left.len() + new_right.len()) as u32,
                split_key,
            };
            let int_off = alloc_or_full!(aw.alloc_write_interior(interior));
            split_levels = 1;
            tag_interior(int_off)
        } else {
            let new_chunk = match chunk.insert(val) {
                Some(c) => c,
                None => return InsertResult::Duplicate,
            };
            alloc_or_full!(aw.alloc_write_chunk(new_chunk))
        };

        // Scapegoat check, decided before the copy loop (see `total`
        // above). The integer prefilter skips the floating-point log on
        // the overwhelming majority of inserts: depth_limit(n) is
        // ~1.71 x log2(n), so any path no deeper than floor(log2(n))
        // trivially passes the bound.
        let need_rebalance = {
            let d = depth + split_levels;
            d > (total.max(2).ilog2() as usize) && d > depth_limit(total)
        };

        // Path-copy ancestors bottom-up. `new_path` entries are recorded
        // only when the rebalance below will read them.
        let mut new_path = [MaybeUninit::<u32>::uninit(); MAX_DEPTH];
        for i in (0..depth).rev() {
            // SAFETY: `i < depth`, and every entry below `depth` was
            // written during the descent.
            let path_i = unsafe { path[i].assume_init() };
            // SAFETY: same bound for the direction array.
            let went_left_i = unsafe { went_left[i].assume_init() };
            let old = interior_at(arena, path_i);
            let (l, r) = if went_left_i {
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
            new_child = tag_interior(alloc_or_full!(aw.alloc_write_interior(copied)));
            if need_rebalance {
                new_path[i].write(new_child);
            }
        }
        let new_root = new_child;

        // Every fallible allocation has succeeded — the new root will be
        // returned (rebalance failure still returns it), so the old leaf
        // and the copied interior path are now superseded. Log them for
        // recycling; the writer stamps the log after publishing the root.
        retire.chunks.push(cur);
        for slot in path.iter().take(depth) {
            // SAFETY: entries below `depth` were written during the descent.
            let path_i = unsafe { slot.assume_init() };
            retire.interiors.push(strip_tag(path_i));
        }

        // Scapegoat rebalance: if this insert pushed the path past the
        // α-height bound, rebuild the highest α-weight-unbalanced node on
        // it. If the rebuild allocation fails we still return the (valid,
        // temporarily too-deep) new tree — the element is inserted either
        // way and a later insert retries the rebalance. Repeated failures
        // can only deepen the tree to MAX_DEPTH: the descent above returns
        // `ArenaFull` past that, so depth stays bounded.
        if need_rebalance {
            // SAFETY: every entry below `depth` in `new_path` was written in
            // the copy loop just above (under the same `need_rebalance`),
            // and MaybeUninit<u32> shares u32's layout.
            let new_path_init =
                unsafe { &*(&new_path[..depth] as *const [MaybeUninit<u32>] as *const [u32]) };
            // SAFETY: every entry below `depth` in `went_left` was written
            // during the descent, and MaybeUninit<bool> shares bool's layout.
            let went_left_init =
                unsafe { &*(&went_left[..depth] as *const [MaybeUninit<bool>] as *const [bool]) };
            match rebalance(aw, new_root, new_path_init, went_left_init, scratch, retire) {
                Some(balanced_root) => {
                    return InsertResult::Inserted(Self {
                        root: balanced_root,
                    });
                }
                None => return InsertResult::Inserted(Self { root: new_root }),
            }
        }

        InsertResult::Inserted(Self { root: new_root })
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
    match node_at(arena, offset) {
        NodeRef::Leaf(chunk) => chunk.len(),
        NodeRef::Interior(node) => node.count as usize,
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
/// On success the entire replaced subtree (which includes this insert's
/// fresh path copies below the scapegoat) and the superseded first copies
/// of the ancestors above it are logged in `retire`. On failure nothing
/// is logged: the un-rebalanced tree remains the live result, and the
/// partial rebuild's nodes leak until compact.
fn rebalance(
    aw: &mut ArenaWriter<'_>,
    new_root: u32,
    new_path: &[u32],
    went_left: &[bool],
    scratch: &mut Vec<u32>,
    retire: &mut RetireLog,
) -> Option<u32> {
    let arena = aw.arena();
    let mut scapegoat = 0usize; // rebuild the root unless a deeper node is unbalanced
    for (i, &off) in new_path.iter().enumerate() {
        if alpha_unbalanced(interior_at(arena, off), arena) {
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
    let mut child = build_balanced(aw, &scratch[..])?;

    // Re-copy the ancestors above the scapegoat (counts and split keys
    // are unchanged — only the child pointer moved).
    for i in (0..scapegoat).rev() {
        let old = interior_at(arena, new_path[i]);
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
        child = tag_interior(aw.alloc_write_interior(copied)?);
    }

    // Full success: the rebuilt subtree replaced `target` (whose nodes
    // include this insert's fresh copies below the scapegoat plus the old
    // off-path descendants), and the re-copy loop replaced the first
    // copies of the ancestors above it.
    // SAFETY: `target` and the first path copies are unreachable from the
    // root being returned; the caller publishes that root before stamping.
    unsafe { retire_subtree(arena, target, retire) };
    for &off in &new_path[..scapegoat] {
        retire.interiors.push(strip_tag(off));
    }
    Some(child)
}

/// Log every node of the subtree under `root` (leaves and interiors) as
/// superseded. Node fields are read before the node is logged, and logged
/// nodes are never revisited, so the walk never observes a recycled slot.
///
/// # Safety
/// The subtree must be (or be about to become) unreachable from every
/// tree the caller keeps live, and none of its slots may be logged twice
/// — once the log is stamped, the slots are rewritten under any surviving
/// reference.
pub(crate) unsafe fn retire_subtree(arena: &Arena, root: u32, retire: &mut RetireLog) {
    let mut stack = [MaybeUninit::<u32>::uninit(); MAX_DEPTH];
    let mut sp = 0usize;
    let mut cur = root;
    loop {
        while let NodeRef::Interior(node) = node_at(arena, cur) {
            debug_assert!(sp < MAX_DEPTH, "C-tree exceeded MAX_DEPTH");
            stack[sp].write(node.right);
            sp += 1;
            retire.interiors.push(strip_tag(cur));
            cur = node.left;
        }
        retire.chunks.push(cur);
        if sp == 0 {
            break;
        }
        sp -= 1;
        // SAFETY: the entry was written on the way down.
        cur = unsafe { stack[sp].assume_init() };
    }
}

// ---------------------------------------------------------------------------
// Balanced builds (shared by bulk insert, rebalance, and compaction)
// ---------------------------------------------------------------------------

/// Where a balanced build places its nodes: the fallible bump/free-list
/// allocator during normal writes, or a compaction thread's pre-reserved
/// (infallible) region. One recursion serves both, so the built shape —
/// which [`compact_slot_cost`] must predict exactly — exists once.
pub(crate) trait SlotSink {
    fn put_chunk(&mut self, c: Chunk) -> Option<u32>;
    fn put_interior(&mut self, i: Interior) -> Option<u32>;
}

impl SlotSink for ArenaWriter<'_> {
    #[inline(always)]
    fn put_chunk(&mut self, c: Chunk) -> Option<u32> {
        self.alloc_write_chunk(c)
    }
    #[inline(always)]
    fn put_interior(&mut self, i: Interior) -> Option<u32> {
        self.alloc_write_interior(i)
    }
}

impl SlotSink for RegionWriter<'_> {
    #[inline(always)]
    fn put_chunk(&mut self, c: Chunk) -> Option<u32> {
        Some(self.write_chunk(c))
    }
    #[inline(always)]
    fn put_interior(&mut self, i: Interior) -> Option<u32> {
        Some(self.write_interior(i))
    }
}

/// Build a perfectly weight-balanced subtree over `vals` (sorted, unique,
/// non-empty). Splits on a chunk-capacity boundary at the leaf-count
/// midpoint, so every leaf except the last is completely full — an
/// element-midpoint split would leave every leaf 8-15 of 15 full (~25%
/// wasted capacity, ~25% extra cache lines per scan, forever). Both sides
/// get within one leaf of half the leaves, so every interior node stays
/// α-weight-balanced for α = 2/3.
fn build_balanced<S: SlotSink>(sink: &mut S, vals: &[u32]) -> Option<u32> {
    debug_assert!(!vals.is_empty());
    if vals.len() <= crate::chunk::CHUNK_CAP {
        return sink.put_chunk(Chunk::from_sorted_unchecked(vals));
    }
    let leaves = vals.len().div_ceil(crate::chunk::CHUNK_CAP);
    let mid = (leaves / 2) * crate::chunk::CHUNK_CAP;
    let (left_vals, right_vals) = vals.split_at(mid);
    let left = build_balanced(sink, left_vals)?;
    let right = build_balanced(sink, right_vals)?;
    let interior = Interior {
        left,
        right,
        count: vals.len() as u32,
        split_key: right_vals[0],
    };
    Some(tag_interior(sink.put_interior(interior)?))
}

/// Exact slot cost of a perfectly balanced tree over `deg` elements:
/// `ceil(deg/CHUNK_CAP)` full-except-last leaves and one fewer interiors.
/// [`build_balanced`] produces exactly this shape, which is what lets
/// parallel compaction reserve disjoint regions up front.
pub(crate) fn compact_slot_cost(deg: usize) -> (usize, usize) {
    if deg == 0 {
        return (0, 0);
    }
    let leaves = deg.div_ceil(crate::chunk::CHUNK_CAP);
    (leaves, leaves - 1)
}

/// [`build_balanced`] into a compaction thread's private slot region. The
/// region was reserved from [`compact_slot_cost`]'s exact figures, so
/// allocation cannot fail (overrun is a debug panic).
pub(crate) fn build_balanced_region(region: &mut RegionWriter<'_>, vals: &[u32]) -> u32 {
    build_balanced(region, vals).expect("region reservation covers the exact build cost")
}

// ---------------------------------------------------------------------------
// Traversal (iterative — depth is bounded by the scapegoat invariant, but
// an explicit stack keeps the read path independent of thread stack size)
// ---------------------------------------------------------------------------

fn visit_chunks<F: FnMut(&Chunk)>(root: u32, arena: &Arena, f: &mut F) {
    // Fast path for the majority shape: a degree-≤15 vertex is one chunk,
    // and setting up the traversal stack would cost more than the visit.
    if let NodeRef::Leaf(chunk) = node_at(arena, root) {
        f(chunk);
        return;
    }

    // Uninitialized stack: entries are written before read (push before
    // pop), and zeroing 256 bytes per scan would dwarf the traversal work
    // for small trees. The depth bound is structural (see MAX_DEPTH), so
    // the release-mode check is a debug_assert.
    let mut stack = [MaybeUninit::<u32>::uninit(); MAX_DEPTH];
    let mut sp = 0usize;
    let mut cur = root;
    loop {
        let chunk = loop {
            match node_at(arena, cur) {
                NodeRef::Interior(node) => {
                    debug_assert!(sp < MAX_DEPTH, "C-tree exceeded MAX_DEPTH");
                    stack[sp].write(node.right);
                    sp += 1;
                    cur = node.left;
                }
                NodeRef::Leaf(chunk) => break chunk,
            }
        };
        // The pending right subtree is the next dependent miss; hint it
        // while the callback consumes the current chunk.
        if sp > 0 {
            // SAFETY: stack[sp - 1] was written on the way down.
            let next = unsafe { stack[sp - 1].assume_init() };
            let p = if is_leaf(next) {
                // SAFETY: `next` was read from a live interior node, so it
                // names a live chunk slot.
                unsafe { arena.chunk_ptr(next) }
            } else {
                // SAFETY: as above, for the interior region.
                unsafe { arena.interior_ptr(strip_tag(next)) }
            };
            prefetch_read(p);
        }
        f(chunk);
        if sp == 0 {
            break;
        }
        sp -= 1;
        // SAFETY: `sp` was decremented from a pushed position; the entry
        // was written on the way down.
        cur = unsafe { stack[sp].assume_init() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::CHUNK_CAP;

    fn make_arena() -> Arena {
        Arena::new(1 << 20)
    }

    /// The test-local write handle.
    fn writer(arena: &Arena) -> ArenaWriter<'_> {
        // SAFETY: each test constructs exactly one handle per arena and
        // runs single-threaded.
        unsafe { arena.writer() }
    }

    fn insert_assert(tree: CTree, aw: &mut ArenaWriter<'_>, val: u32) -> CTree {
        match tree.insert(aw, val) {
            InsertResult::Inserted(t) => t,
            other => panic!("expected Inserted, got {other:?}"),
        }
    }

    /// Depth of the deepest leaf (interior nodes on the path).
    fn max_depth(offset: u32, arena: &Arena) -> usize {
        if offset == NULL || is_leaf(offset) {
            return 0;
        }
        let node = interior_at(arena, offset);
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
        let mut aw = writer(&arena);
        let tree = insert_assert(CTree::empty(), &mut aw, 42);
        assert!(!tree.is_empty());
        assert_eq!(tree.count(&arena), 1);
        assert!(tree.contains(&arena, 42));
        assert!(!tree.contains(&arena, 99));
    }

    #[test]
    fn insert_sorted_order() {
        let arena = make_arena();
        let mut aw = writer(&arena);
        let mut tree = CTree::empty();
        for i in 0..10u32 {
            tree = insert_assert(tree, &mut aw, i * 10);
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
        let mut aw = writer(&arena);
        let mut tree = CTree::empty();
        for i in (0..10u32).rev() {
            tree = insert_assert(tree, &mut aw, i * 10);
        }
        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        let expected: Vec<u32> = (0..10).map(|i| i * 10).collect();
        assert_eq!(buf, expected);
    }

    #[test]
    fn insert_random_order() {
        let arena = make_arena();
        let mut aw = writer(&arena);
        let mut tree = CTree::empty();
        let vals = [50, 20, 80, 10, 30, 70, 90, 5, 15, 25, 35, 60, 75, 85, 95];
        for &v in &vals {
            tree = insert_assert(tree, &mut aw, v);
        }
        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        let mut expected = vals.to_vec();
        expected.sort_unstable();
        assert_eq!(buf, expected);
    }

    #[test]
    fn insert_duplicate_returns_duplicate() {
        let arena = make_arena();
        let mut aw = writer(&arena);
        let tree = insert_assert(CTree::empty(), &mut aw, 42);
        let result = tree.insert(&mut aw, 42);
        assert!(matches!(result, InsertResult::Duplicate));
    }

    #[test]
    fn arena_full_returns_arena_full() {
        // Tiny arena: only enough for a single chunk allocation, then split needs more.
        let arena = Arena::new(64);
        let mut aw = writer(&arena);
        let mut tree = CTree::empty();
        // Fill until ArenaFull.
        let mut val = 0u32;
        loop {
            match tree.insert(&mut aw, val) {
                InsertResult::Inserted(t) => tree = t,
                InsertResult::ArenaFull => return,
                InsertResult::Duplicate => unreachable!("monotonic vals"),
            }
            val += 1;
            assert!(val < 1_000_000, "should hit ArenaFull before 1M iterations");
        }
    }

    #[test]
    fn repeated_failed_rebuilds_return_arena_full_without_panicking() {
        // A nearly-full arena lets cheap path-copy inserts succeed while
        // every scapegoat rebuild (which needs O(subtree) fresh space)
        // fails, deepening the tree past its α-height bound. Inserts must
        // keep terminating and degrade to ArenaFull — never panic.
        let arena = Arena::new(8 << 10);
        let mut aw = writer(&arena);
        let mut tree = CTree::empty();
        let mut val = 0u32;
        loop {
            match tree.insert(&mut aw, val) {
                InsertResult::Inserted(t) => tree = t,
                InsertResult::ArenaFull => break,
                InsertResult::Duplicate => unreachable!("monotonic vals"),
            }
            val += 1;
            assert!(val < 1_000_000, "should hit ArenaFull before 1M inserts");
        }
        // Once full, every further insert keeps returning ArenaFull.
        for extra in val..val + 100 {
            let result = tree.insert(&mut aw, extra);
            assert_eq!(result, InsertResult::ArenaFull);
        }
        // The tree is still readable and intact.
        assert_eq!(tree.count(&arena), val as usize);
        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        assert_eq!(buf.len(), val as usize);
        for w in buf.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn triggers_chunk_split() {
        let arena = make_arena();
        let mut aw = writer(&arena);
        let mut tree = CTree::empty();
        for i in 0..=CHUNK_CAP as u32 {
            tree = insert_assert(tree, &mut aw, i);
        }
        assert_eq!(tree.count(&arena), CHUNK_CAP + 1);

        for i in 0..=CHUNK_CAP as u32 {
            assert!(tree.contains(&arena, i), "missing {i}");
        }

        let mut buf = Vec::new();
        tree.collect_into(&arena, &mut buf);
        let expected: Vec<u32> = (0..=CHUNK_CAP as u32).collect();
        assert_eq!(buf, expected);
    }

    #[test]
    fn large_tree() {
        let arena = Arena::new(4 << 20);
        let mut aw = writer(&arena);
        let mut tree = CTree::empty();
        let n = 1000u32;
        for i in 0..n {
            tree = insert_assert(tree, &mut aw, i * 3);
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
        let mut aw = writer(&arena);
        let tree1 = CTree::empty();
        let tree2 = insert_assert(tree1, &mut aw, 10);
        let tree3 = insert_assert(tree2, &mut aw, 20);

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
        let mut aw = writer(&arena);
        let mut tree = CTree::empty();
        for i in 0..50u32 {
            tree = insert_assert(tree, &mut aw, i);
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
        let mut aw = writer(&arena);
        let mut tree = CTree::empty();
        for i in 0..n {
            tree = insert_assert(tree, &mut aw, i);
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
        let mut aw = writer(&arena);
        let mut tree = CTree::empty();
        for i in (0..n).rev() {
            tree = insert_assert(tree, &mut aw, i);
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
        let mut aw = writer(&arena);
        let mut tree = CTree::empty();
        let mut inserted = 0usize;
        for _ in 0..30_000 {
            if let InsertResult::Inserted(t) = tree.insert(&mut aw, next() % 1_000_000) {
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
        let mut aw = writer(&arena);
        let vals: Vec<u32> = (0..50_000u32).map(|i| i * 2).collect();
        let tree = CTree::from_sorted(&mut aw, &vals).unwrap();
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
        let empty = CTree::from_sorted(&mut aw, &[]).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn persistence_survives_rebalance() {
        // Snapshots taken before a rebuild must keep reading the old nodes.
        let arena = Arena::new(64 << 20);
        let mut aw = writer(&arena);
        let mut tree = CTree::empty();
        let mut snapshots: Vec<(CTree, usize)> = Vec::new();
        for i in 0..10_000u32 {
            if i % 1000 == 0 {
                snapshots.push((tree, tree.count(&arena)));
            }
            tree = insert_assert(tree, &mut aw, i);
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
