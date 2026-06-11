//! Cache-line-sized sorted chunk of neighbor IDs.
//!
//! 64 bytes = exactly one cache line on x86_64 and aarch64.
//! Holds up to 15 sorted u32 neighbor IDs. ~90% of vertices in
//! a power-law graph fit in a single chunk.
//!
//! Chunks are always sorted. Binary search for contains/insert position.
//! No heap allocation — chunks live inside arena-allocated tree nodes.

/// Maximum neighbors per chunk. 15 × 4 = 60 bytes + 1-byte count + 3 pad = 64.
pub const CHUNK_CAP: usize = 15;

/// A sorted, cache-line-sized array of neighbor IDs.
///
/// Immutable after creation (functional style — insert produces a new chunk).
/// Laid out for single cache-line access.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Chunk {
    /// Number of valid entries in `data`.
    pub count: u8,
    _pad: [u8; 3],
    /// Sorted neighbor IDs. Only `data[0..count]` are valid.
    pub data: [u32; CHUNK_CAP],
}

impl Chunk {
    /// Empty chunk.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            count: 0,
            _pad: [0; 3],
            data: [0; CHUNK_CAP],
        }
    }

    /// Create a chunk from a sorted slice.
    ///
    /// # Panics
    /// Panics if `sorted.len() > CHUNK_CAP`. The slice is also expected to be
    /// sorted ascending (debug-asserted).
    #[inline]
    pub fn from_sorted(sorted: &[u32]) -> Self {
        assert!(
            sorted.len() <= CHUNK_CAP,
            "chunk: input len {} exceeds CHUNK_CAP {}",
            sorted.len(),
            CHUNK_CAP
        );
        debug_assert!(is_sorted(sorted), "chunk input must be sorted");
        let mut c = Self::empty();
        c.count = sorted.len() as u8;
        c.data[..sorted.len()].copy_from_slice(sorted);
        c
    }

    /// Valid neighbor slice.
    #[inline(always)]
    pub fn as_slice(&self) -> &[u32] {
        &self.data[..self.count as usize]
    }

    /// Number of neighbors.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count as usize
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.count as usize == CHUNK_CAP
    }

    /// Binary search for `val`. Returns Ok(index) if found, Err(insert_pos) if not.
    #[inline]
    pub fn search(&self, val: u32) -> Result<usize, usize> {
        self.as_slice().binary_search(&val)
    }

    /// Does this chunk contain `val`?
    #[inline]
    pub fn contains(&self, val: u32) -> bool {
        self.search(val).is_ok()
    }

    /// Insert `val` into a new chunk (sorted). Returns None if already present.
    /// Panics if chunk is full — caller must split first.
    #[inline]
    pub fn insert(&self, val: u32) -> Option<Chunk> {
        match self.search(val) {
            Ok(_) => None, // duplicate
            Err(pos) => {
                debug_assert!(!self.is_full(), "insert into full chunk");
                let mut new = *self;
                let n = new.count as usize;
                // Shift right to make room
                let dst = &mut new.data[..n + 1];
                dst.copy_within(pos..n, pos + 1);
                dst[pos] = val;
                new.count += 1;
                Some(new)
            }
        }
    }

    /// Split chunk into two halves. Returns (left, right).
    /// Left gets the first half, right gets the second half.
    #[inline]
    pub fn split(&self) -> (Chunk, Chunk) {
        let n = self.count as usize;
        let mid = n / 2;
        let left = Chunk::from_sorted(&self.data[..mid]);
        let right = Chunk::from_sorted(&self.data[mid..n]);
        (left, right)
    }

    /// Merge two sorted chunks into one. Panics if combined length > CHUNK_CAP.
    ///
    /// Requires the two inputs' key ranges to be disjoint — chunks are
    /// strictly sorted with unique keys, so a shared boundary key (e.g.
    /// `a=[1,5]`, `b=[5,9]`) would produce a duplicate `5` and violate the
    /// chunk invariant. The strict `<` comparison below would silently drop
    /// the `a` copy in that case; a debug assertion checks for it explicitly.
    #[inline]
    pub fn merge(a: &Chunk, b: &Chunk) -> Chunk {
        let an = a.len();
        let bn = b.len();
        debug_assert!(an + bn <= CHUNK_CAP);
        debug_assert!(
            an == 0 || bn == 0 || a.max() < b.min() || b.max() < a.min(),
            "Chunk::merge inputs must have disjoint key ranges",
        );
        let mut c = Chunk::empty();
        let mut ai = 0;
        let mut bi = 0;
        let mut ci = 0;
        while ai < an && bi < bn {
            if a.data[ai] < b.data[bi] {
                c.data[ci] = a.data[ai];
                ai += 1;
            } else {
                c.data[ci] = b.data[bi];
                bi += 1;
            }
            ci += 1;
        }
        while ai < an {
            c.data[ci] = a.data[ai];
            ai += 1;
            ci += 1;
        }
        while bi < bn {
            c.data[ci] = b.data[bi];
            bi += 1;
            ci += 1;
        }
        c.count = ci as u8;
        c
    }

    /// Smallest value in chunk (for split keys).
    #[inline(always)]
    pub fn min(&self) -> u32 {
        debug_assert!(!self.is_empty());
        self.data[0]
    }

    /// Largest value in chunk.
    #[inline(always)]
    pub fn max(&self) -> u32 {
        debug_assert!(!self.is_empty());
        self.data[self.count as usize - 1]
    }
}

#[inline]
fn is_sorted(s: &[u32]) -> bool {
    s.windows(2).all(|w| w[0] <= w[1])
}

// Verify cache-line alignment at compile time.
const _: () = assert!(std::mem::size_of::<Chunk>() == 64);
const _: () = assert!(std::mem::align_of::<Chunk>() == 64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chunk() {
        let c = Chunk::empty();
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert!(!c.is_full());
    }

    #[test]
    fn from_sorted() {
        let c = Chunk::from_sorted(&[10, 20, 30]);
        assert_eq!(c.len(), 3);
        assert_eq!(c.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn insert_maintains_sort() {
        let c = Chunk::from_sorted(&[10, 30, 50]);
        let c2 = c.insert(20).unwrap();
        assert_eq!(c2.as_slice(), &[10, 20, 30, 50]);
        let c3 = c2.insert(5).unwrap();
        assert_eq!(c3.as_slice(), &[5, 10, 20, 30, 50]);
        let c4 = c3.insert(100).unwrap();
        assert_eq!(c4.as_slice(), &[5, 10, 20, 30, 50, 100]);
    }

    #[test]
    fn insert_duplicate_returns_none() {
        let c = Chunk::from_sorted(&[10, 20, 30]);
        assert!(c.insert(20).is_none());
    }

    #[test]
    fn split_and_merge() {
        let c = Chunk::from_sorted(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let (left, right) = c.split();
        assert_eq!(left.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(right.as_slice(), &[5, 6, 7, 8]);
        let merged = Chunk::merge(&left, &right);
        assert_eq!(merged.as_slice(), c.as_slice());
    }

    #[test]
    fn fill_to_capacity() {
        let mut c = Chunk::empty();
        for i in 0..CHUNK_CAP as u32 {
            c = c.insert(i * 10).unwrap();
        }
        assert!(c.is_full());
        assert_eq!(c.len(), CHUNK_CAP);
    }

    #[test]
    fn search() {
        let c = Chunk::from_sorted(&[10, 20, 30, 40, 50]);
        assert_eq!(c.search(30), Ok(2));
        assert_eq!(c.search(25), Err(2));
        assert_eq!(c.search(5), Err(0));
        assert_eq!(c.search(55), Err(5));
    }

    #[test]
    fn size_and_alignment() {
        assert_eq!(std::mem::size_of::<Chunk>(), 64);
        assert_eq!(std::mem::align_of::<Chunk>(), 64);
    }
}
