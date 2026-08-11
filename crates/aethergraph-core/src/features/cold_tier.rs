//! zstd-compressed cold feature tier.
//!
//! Hot features stay in the mmap/NVMe store read at full speed. Rarely
//! touched rows go in a cold tier: block-compressed with zstd against a
//! dictionary trained on the feature distribution at build time, so each
//! block shares statistics with its neighbors and small blocks still
//! compress well. A gather decompresses only the blocks it touches, into
//! the caller's arena; the hot tier is untouched.
//!
//! The tier is immutable once built — the same "parse at the edge, trust
//! downstream" contract as the rest of the store. Rows are fixed-size, so
//! a node ID maps to (block, offset) arithmetically with no per-row index.

use anyhow::{Context, Result, bail};

/// Rows per compression block. A block is the decompression unit: bigger
/// blocks compress better but force more waste when a gather wants one
/// row. 512 rows balances the two for typical feature dimensions.
pub const ROWS_PER_BLOCK: usize = 512;

/// A built cold tier: dictionary, per-block compressed payloads, and the
/// fixed geometry needed to locate any row.
#[derive(Debug, Clone)]
pub struct ColdTier {
    row_bytes: usize,
    num_rows: usize,
    dictionary: Vec<u8>,
    /// One compressed payload per block, block `b` covering rows
    /// `b*ROWS_PER_BLOCK .. min(num_rows, (b+1)*ROWS_PER_BLOCK)`.
    blocks: Vec<Vec<u8>>,
}

impl ColdTier {
    /// Compress `rows` (a flat `num_rows * row_bytes` buffer) into blocks,
    /// training a zstd dictionary on a sample of the blocks first.
    ///
    /// `level` is the zstd compression level (1–22; ~19 is a good archive
    /// setting for a cold tier written once and read many times).
    pub fn build(data: &[u8], row_bytes: usize, level: i32) -> Result<Self> {
        if row_bytes == 0 {
            bail!("row_bytes must be > 0");
        }
        if !data.len().is_multiple_of(row_bytes) {
            bail!(
                "data length {} is not a multiple of row_bytes {row_bytes}",
                data.len()
            );
        }
        let num_rows = data.len() / row_bytes;
        let block_size = ROWS_PER_BLOCK * row_bytes;

        let raw_blocks: Vec<&[u8]> = data.chunks(block_size).collect();

        // Train a dictionary on the raw blocks so every block decodes
        // against shared statistics. Dictionary training needs several
        // samples; with too few, skip it and compress without one.
        let dictionary = if raw_blocks.len() >= 8 {
            let max_dict = (data.len() / 100).clamp(4 * 1024, 112 * 1024);
            zstd::dict::from_samples(&raw_blocks, max_dict).unwrap_or_default()
        } else {
            Vec::new()
        };

        let mut blocks = Vec::with_capacity(raw_blocks.len());
        for raw in &raw_blocks {
            let compressed = if dictionary.is_empty() {
                zstd::bulk::compress(raw, level).context("zstd compress")?
            } else {
                let mut c = zstd::bulk::Compressor::with_dictionary(level, &dictionary)
                    .context("zstd compressor")?;
                c.compress(raw).context("zstd compress (dict)")?
            };
            blocks.push(compressed);
        }

        Ok(Self {
            row_bytes,
            num_rows,
            dictionary,
            blocks,
        })
    }

    /// Number of rows in the tier.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Bytes per row.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Total compressed size across all blocks plus the dictionary.
    pub fn compressed_bytes(&self) -> usize {
        self.dictionary.len() + self.blocks.iter().map(Vec::len).sum::<usize>()
    }

    /// Compression ratio vs the raw tier (raw / compressed).
    pub fn ratio(&self) -> f64 {
        let raw = (self.num_rows * self.row_bytes) as f64;
        raw / self.compressed_bytes().max(1) as f64
    }

    /// Gather the given rows into one contiguous buffer in `rows` order.
    ///
    /// Blocks are decompressed once each and reused across every requested
    /// row that falls in them, so a gather touching a handful of blocks
    /// pays for those blocks only — not the whole tier.
    pub fn gather_rows(&self, rows: &[u32]) -> Result<Vec<u8>> {
        let mut out = vec![0u8; rows.len() * self.row_bytes];
        // Decompress each needed block once, keyed by block index.
        let mut cache: std::collections::HashMap<usize, Vec<u8>> = std::collections::HashMap::new();

        for (i, &row) in rows.iter().enumerate() {
            let row = row as usize;
            if row >= self.num_rows {
                bail!("row {row} out of range (num_rows={})", self.num_rows);
            }
            let block = row / ROWS_PER_BLOCK;
            let within = row % ROWS_PER_BLOCK;

            let decompressed = match cache.get(&block) {
                Some(d) => d,
                None => {
                    let d = self.decompress_block(block)?;
                    cache.entry(block).or_insert(d)
                }
            };
            let src = &decompressed[within * self.row_bytes..(within + 1) * self.row_bytes];
            out[i * self.row_bytes..(i + 1) * self.row_bytes].copy_from_slice(src);
        }
        Ok(out)
    }

    /// Decompress one block back to its raw rows.
    fn decompress_block(&self, block: usize) -> Result<Vec<u8>> {
        let compressed = &self.blocks[block];
        // The final block may be short; every other block is full.
        let rows_here = if block == self.blocks.len() - 1 {
            self.num_rows - block * ROWS_PER_BLOCK
        } else {
            ROWS_PER_BLOCK
        };
        let capacity = rows_here * self.row_bytes;

        let raw = if self.dictionary.is_empty() {
            zstd::bulk::decompress(compressed, capacity).context("zstd decompress")?
        } else {
            let mut d = zstd::bulk::Decompressor::with_dictionary(&self.dictionary)
                .context("zstd decompressor")?;
            d.decompress(compressed, capacity)
                .context("zstd decompress (dict)")?
        };
        if raw.len() != capacity {
            bail!(
                "block {block} decompressed to {} bytes, expected {capacity}",
                raw.len()
            );
        }
        Ok(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(num_rows: usize, row_bytes: usize) -> Vec<u8> {
        // Structured, compressible rows: each row is a slowly varying
        // ramp, the sort of correlated feature data a dictionary exploits.
        let mut data = vec![0u8; num_rows * row_bytes];
        for r in 0..num_rows {
            for b in 0..row_bytes {
                data[r * row_bytes + b] = ((r / 4 + b) % 251) as u8;
            }
        }
        data
    }

    #[test]
    fn round_trips_every_row() {
        let (num_rows, row_bytes) = (2000usize, 128usize);
        let data = synthetic(num_rows, row_bytes);
        let tier = ColdTier::build(&data, row_bytes, 19).unwrap();
        assert_eq!(tier.num_rows(), num_rows);

        // Gather every row in order and compare to the source.
        let all: Vec<u32> = (0..num_rows as u32).collect();
        let got = tier.gather_rows(&all).unwrap();
        assert_eq!(got, data, "full gather must reproduce the source");
    }

    #[test]
    fn gather_subset_and_repeats() {
        let (num_rows, row_bytes) = (1500usize, 64usize);
        let data = synthetic(num_rows, row_bytes);
        let tier = ColdTier::build(&data, row_bytes, 12).unwrap();

        // A scattered subset with a repeat, crossing block boundaries.
        let rows = [0u32, 1, 511, 512, 513, 1499, 512, 0];
        let got = tier.gather_rows(&rows).unwrap();
        for (i, &row) in rows.iter().enumerate() {
            let expect = &data[row as usize * row_bytes..(row as usize + 1) * row_bytes];
            assert_eq!(
                &got[i * row_bytes..(i + 1) * row_bytes],
                expect,
                "row {row} at position {i}"
            );
        }
    }

    #[test]
    fn compresses_structured_data() {
        let data = synthetic(5000, 256);
        let tier = ColdTier::build(&data, 256, 19).unwrap();
        assert!(
            tier.ratio() > 3.0,
            "structured data should compress > 3x, got {:.2}x",
            tier.ratio()
        );
    }

    #[test]
    fn small_tier_without_dictionary() {
        // Fewer than 8 blocks: no dictionary trained, still round-trips.
        let data = synthetic(100, 32);
        let tier = ColdTier::build(&data, 32, 9).unwrap();
        let all: Vec<u32> = (0..100).collect();
        assert_eq!(tier.gather_rows(&all).unwrap(), data);
    }

    #[test]
    fn rejects_ragged_data() {
        let data = vec![0u8; 100];
        assert!(ColdTier::build(&data, 33, 3).is_err());
    }

    #[test]
    fn rejects_out_of_range_row() {
        let data = synthetic(50, 16);
        let tier = ColdTier::build(&data, 16, 3).unwrap();
        assert!(tier.gather_rows(&[49]).is_ok());
        assert!(tier.gather_rows(&[50]).is_err());
    }
}
