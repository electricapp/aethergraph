//! Block-allocated row storage backing the in-memory cache tiers.

/// Rows per slab block. Growth appends a block; rows never move.
const SLAB_BLOCK_ROWS: usize = 4096;

/// Block-allocated row storage for one tier.
///
/// Rows live in fixed-size blocks addressed by a u32 row index, with a
/// free list recycling the rows of evicted entries: a promote reuses a
/// released row instead of paying a heap allocation, and rows stay
/// clustered in large blocks instead of scattering across the allocator.
pub(super) struct RowSlab {
    blocks: Vec<Vec<f32>>,
    pub(super) dim: usize,
    rows: u32,
    free: Vec<u32>,
}

impl RowSlab {
    pub(super) fn new(dim: usize) -> Self {
        Self {
            blocks: Vec::new(),
            dim,
            rows: 0,
            free: Vec::new(),
        }
    }

    pub(super) fn alloc(&mut self) -> u32 {
        if let Some(r) = self.free.pop() {
            return r;
        }
        let r = self.rows;
        if (r as usize).is_multiple_of(SLAB_BLOCK_ROWS) {
            self.blocks.push(vec![0.0f32; SLAB_BLOCK_ROWS * self.dim]);
        }
        self.rows += 1;
        r
    }

    pub(super) fn release(&mut self, r: u32) {
        self.free.push(r);
    }

    #[inline]
    pub(super) fn row(&self, r: u32) -> &[f32] {
        let r = r as usize;
        let start = (r % SLAB_BLOCK_ROWS) * self.dim;
        &self.blocks[r / SLAB_BLOCK_ROWS][start..start + self.dim]
    }

    #[inline]
    pub(super) fn row_mut(&mut self, r: u32) -> &mut [f32] {
        let r = r as usize;
        let start = (r % SLAB_BLOCK_ROWS) * self.dim;
        &mut self.blocks[r / SLAB_BLOCK_ROWS][start..start + self.dim]
    }
}
