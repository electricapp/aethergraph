//! Dirty-vertex tracking for historical embedding maintenance.

use std::sync::atomic::{AtomicU64, Ordering};

/// Dirty-node bitmap for historical embedding tracking.
///
/// One bit per vertex. Set on edge insert, cleared at epoch boundary.
/// Lock-free: writers set bits with `fetch_or`, readers load with Acquire.
pub(crate) struct DirtyBitmap {
    words: Vec<AtomicU64>,
    /// Number of vertices addressable. Caps the per-vertex bit index.
    num_vertices: usize,
}

impl DirtyBitmap {
    pub(crate) fn new(num_vertices: usize) -> Self {
        let num_words = num_vertices.div_ceil(64).max(1);
        let mut words = Vec::with_capacity(num_words);
        for _ in 0..num_words {
            words.push(AtomicU64::new(0));
        }
        Self {
            words,
            num_vertices,
        }
    }

    /// Mark a batch of vertices dirty from an ascending, deduplicated
    /// slice. Vertices sharing a 64-bit word coalesce into one `fetch_or`,
    /// and the sorted order walks the bitmap sequentially — marking one
    /// vertex at a time would pay a random-line atomic RMW per mark
    /// (a likely DRAM miss each on a multi-hundred-MB bitmap).
    /// Out-of-range IDs are skipped.
    pub(crate) fn mark_sorted(&self, sorted: &[u32]) {
        let mut i = 0;
        while i < sorted.len() {
            let v = sorted[i] as usize;
            if v >= self.num_vertices {
                // Sorted input: everything after is also out of range.
                return;
            }
            let word = v / 64;
            let mut mask = 0u64;
            while i < sorted.len() && (sorted[i] as usize) < self.num_vertices {
                let v = sorted[i] as usize;
                if v / 64 != word {
                    break;
                }
                mask |= 1u64 << (v % 64);
                i += 1;
            }
            self.words[word].fetch_or(mask, Ordering::Release);
        }
    }

    /// Check if a vertex is dirty. Returns `false` if `vertex` is out of bounds.
    #[inline]
    pub(crate) fn is_dirty(&self, vertex: u32) -> bool {
        let v = vertex as usize;
        if v >= self.num_vertices {
            return false;
        }
        let word = v / 64;
        let bit = v % 64;
        self.words[word].load(Ordering::Acquire) & (1u64 << bit) != 0
    }

    /// Clear all dirty bits (epoch boundary).
    ///
    /// Uses `swap(0, AcqRel)` per word so a `mark()` racing on the same word
    /// is not silently lost: the `fetch_or` either lands before our swap (its
    /// bit is included in the discarded value) or after (its bit survives in
    /// the cleared word). A plain `store(0)` would clobber the racing bit.
    pub(crate) fn clear_all(&self) {
        for w in &self.words {
            w.swap(0, Ordering::AcqRel);
        }
    }

    /// Collect all dirty node IDs and atomically clear the bitmap.
    /// Uses swap per word so concurrent writers don't lose marks.
    pub(crate) fn drain_dirty(&self) -> Vec<u32> {
        let mut result = Vec::new();
        for (i, w) in self.words.iter().enumerate() {
            let bits = w.swap(0, Ordering::AcqRel);
            if bits == 0 {
                continue;
            }
            let base = (i as u64) * 64;
            let mut remaining = bits;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as u64;
                let id = base + bit;
                // Mask off bits past num_vertices (last word can over-allocate).
                if (id as usize) < self.num_vertices {
                    result.push(id as u32);
                }
                remaining &= remaining - 1; // clear lowest set bit
            }
        }
        result
    }

    /// Count set bits across all words.
    pub(crate) fn count_dirty(&self) -> usize {
        self.words
            .iter()
            .map(|w| w.load(Ordering::Acquire).count_ones() as usize)
            .sum()
    }
}
