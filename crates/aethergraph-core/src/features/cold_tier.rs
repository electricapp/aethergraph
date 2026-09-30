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

use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use parking_lot::Mutex;
use zstd::dict::DecoderDictionary;

use super::header::{FeatureDtype, parse_feature_header};

/// Rows per compression block. A block is the decompression unit: bigger
/// blocks compress better but force more waste when a gather wants one
/// row. 512 rows balances the two for typical feature dimensions.
pub const ROWS_PER_BLOCK: usize = 512;

/// Most bytes fed to dictionary training. zstd wants on the order of 100x
/// the dictionary size in samples; past that, more only costs build time
/// and memory, and ZDICT rejects inputs beyond ~4 GiB outright.
const MAX_SAMPLE_BYTES: usize = 64 << 20;

/// Most samples drawn for dictionary training, spread evenly over the tier.
const MAX_SAMPLES: usize = 1024;

/// Decompressed blocks kept for reuse, so repeated single-row misses on a
/// hot block decompress it once.
const RECENT_BLOCKS: usize = 8;

/// A built cold tier: dictionary, per-block compressed payloads, and the
/// fixed geometry needed to locate any row.
pub struct ColdTier {
    row_bytes: usize,
    num_rows: usize,
    dictionary: Vec<u8>,
    /// The dictionary digested once for decoding; every gather borrows it
    /// instead of re-loading the raw bytes.
    prepared: Option<DecoderDictionary<'static>>,
    /// One compressed payload per block, block `b` covering rows
    /// `b*ROWS_PER_BLOCK .. min(num_rows, (b+1)*ROWS_PER_BLOCK)`.
    blocks: Vec<Vec<u8>>,
    /// Most recently decompressed blocks, newest last.
    recent: Mutex<VecDeque<(usize, Arc<Vec<u8>>)>>,
}

impl std::fmt::Debug for ColdTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColdTier")
            .field("row_bytes", &self.row_bytes)
            .field("num_rows", &self.num_rows)
            .field("dictionary_bytes", &self.dictionary.len())
            .field("blocks", &self.blocks.len())
            .finish_non_exhaustive()
    }
}

impl Clone for ColdTier {
    fn clone(&self) -> Self {
        Self {
            row_bytes: self.row_bytes,
            num_rows: self.num_rows,
            dictionary: self.dictionary.clone(),
            prepared: prepare(&self.dictionary),
            blocks: self.blocks.clone(),
            recent: Mutex::new(VecDeque::new()),
        }
    }
}

fn prepare(dictionary: &[u8]) -> Option<DecoderDictionary<'static>> {
    (!dictionary.is_empty()).then(|| DecoderDictionary::copy(dictionary))
}

impl ColdTier {
    /// Compress `rows` (a flat `num_rows * row_bytes` buffer) into blocks,
    /// training a zstd dictionary on a sample of the blocks first.
    ///
    /// `level` is the zstd compression level (1–22; ~19 is a good archive
    /// setting for a cold tier written once and read many times).
    pub fn build(data: &[u8], row_bytes: usize, level: i32) -> Result<Self> {
        let Some(row) = NonZeroUsize::new(row_bytes) else {
            bail!("row_bytes must be > 0");
        };
        if !data.len().is_multiple_of(row_bytes) {
            bail!(
                "data length {} is not a multiple of row_bytes {row_bytes}",
                data.len()
            );
        }
        let block_size = ROWS_PER_BLOCK * row_bytes;
        Self::build_from_blocks(data.len() / row_bytes, row, level, |b, buf| {
            let start = b * block_size;
            buf.clear();
            buf.extend_from_slice(&data[start..(start + block_size).min(data.len())]);
            Ok(())
        })
    }

    /// Build from a block source: `read_block(b, buf)` replaces `buf`'s
    /// contents with block `b`'s raw rows. Blocks are pulled one at a time
    /// — a bounded sample first, for dictionary training, then each block
    /// once to compress — so building never holds the raw tier in memory.
    fn build_from_blocks(
        num_rows: usize,
        row_bytes: NonZeroUsize,
        level: i32,
        mut read_block: impl FnMut(usize, &mut Vec<u8>) -> Result<()>,
    ) -> Result<Self> {
        let row_bytes = row_bytes.get();
        let block_size = ROWS_PER_BLOCK
            .checked_mul(row_bytes)
            .context("block size overflows usize")?;
        let total_bytes = num_rows
            .checked_mul(row_bytes)
            .context("tier size overflows usize")?;
        let num_blocks = num_rows.div_ceil(ROWS_PER_BLOCK);
        let mut buf = Vec::with_capacity(block_size);

        // Train a dictionary so every block decodes against shared
        // statistics. Training needs several samples; with too few blocks,
        // compress without one.
        let dictionary = if num_blocks >= 8 {
            let samples = num_blocks.min(MAX_SAMPLES);
            // Whole rows from the head of each sampled block, capped so the
            // sample set stays bounded however large the tier is.
            let per_sample = (MAX_SAMPLE_BYTES / samples / row_bytes).max(1) * row_bytes;
            let mut sample_data = Vec::with_capacity(per_sample.min(block_size) * samples);
            let mut sizes = Vec::with_capacity(samples);
            for i in 0..samples {
                read_block(i * num_blocks / samples, &mut buf)?;
                let take = buf.len().min(per_sample);
                sample_data.extend_from_slice(&buf[..take]);
                sizes.push(take);
            }
            let max_dict = (total_bytes / 100).clamp(4 * 1024, 112 * 1024);
            match zstd::dict::from_continuous(&sample_data, &sizes, max_dict) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(
                        "zstd dictionary training failed ({e}); compressing without one"
                    );
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        // One compressor for the whole tier: loading the dictionary digests
        // it into the encoder's tables, and repeating that per block would
        // dominate build time on a large store.
        let mut encoder = if dictionary.is_empty() {
            None
        } else {
            Some(
                zstd::bulk::Compressor::with_dictionary(level, &dictionary)
                    .context("zstd compressor")?,
            )
        };
        let mut blocks = Vec::with_capacity(num_blocks);
        for b in 0..num_blocks {
            read_block(b, &mut buf)?;
            let compressed = match encoder.as_mut() {
                Some(c) => c.compress(&buf).context("zstd compress (dict)")?,
                None => zstd::bulk::compress(&buf, level).context("zstd compress")?,
            };
            blocks.push(compressed);
        }

        Ok(Self {
            row_bytes,
            num_rows,
            prepared: prepare(&dictionary),
            dictionary,
            blocks,
            recent: Mutex::new(VecDeque::with_capacity(RECENT_BLOCKS)),
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
        if rows.is_empty() {
            return Ok(out);
        }
        for &row in rows {
            if row as usize >= self.num_rows {
                bail!("row {row} out of range (num_rows={})", self.num_rows);
            }
        }

        let mut needed: Vec<usize> = rows.iter().map(|&r| r as usize / ROWS_PER_BLOCK).collect();
        needed.sort_unstable();
        needed.dedup();

        // The dictionary was digested once at build; binding it to a
        // decoder is a pointer handoff, not a reload.
        let mut decoder: Option<zstd::bulk::Decompressor<'_>> = None;
        let mut cache: std::collections::HashMap<usize, Arc<Vec<u8>>> =
            std::collections::HashMap::with_capacity(needed.len());
        for &block in &needed {
            let raw = match self.recent_block(block) {
                Some(raw) => raw,
                None => {
                    if decoder.is_none() && self.prepared.is_some() {
                        decoder = self
                            .prepared
                            .as_ref()
                            .map(zstd::bulk::Decompressor::with_prepared_dictionary)
                            .transpose()
                            .context("zstd decompressor")?;
                    }
                    let raw = Arc::new(self.decompress_block(block, decoder.as_mut())?);
                    self.remember_block(block, Arc::clone(&raw));
                    raw
                }
            };
            cache.insert(block, raw);
        }

        for (i, &row) in rows.iter().enumerate() {
            let row = row as usize;
            let block = row / ROWS_PER_BLOCK;
            let within = row % ROWS_PER_BLOCK;
            let decompressed = &cache[&block];
            let src = &decompressed[within * self.row_bytes..(within + 1) * self.row_bytes];
            out[i * self.row_bytes..(i + 1) * self.row_bytes].copy_from_slice(src);
        }
        Ok(out)
    }

    /// A recently decompressed copy of `block`, refreshed as newest.
    fn recent_block(&self, block: usize) -> Option<Arc<Vec<u8>>> {
        let mut recent = self.recent.lock();
        let pos = recent.iter().position(|(b, _)| *b == block)?;
        let entry = recent.remove(pos)?;
        let raw = Arc::clone(&entry.1);
        recent.push_back(entry);
        Some(raw)
    }

    /// Keep `raw` as the newest recent block, dropping the oldest past
    /// [`RECENT_BLOCKS`].
    fn remember_block(&self, block: usize, raw: Arc<Vec<u8>>) {
        let mut recent = self.recent.lock();
        if recent.iter().any(|(b, _)| *b == block) {
            return;
        }
        if recent.len() == RECENT_BLOCKS {
            recent.pop_front();
        }
        recent.push_back((block, raw));
    }

    /// Decompress one block back to its raw rows. `decoder` carries the
    /// dictionary-bound decompressor when the tier was built with one.
    fn decompress_block(
        &self,
        block: usize,
        decoder: Option<&mut zstd::bulk::Decompressor<'_>>,
    ) -> Result<Vec<u8>> {
        let compressed = &self.blocks[block];
        // The final block may be short; every other block is full.
        let rows_here = if block == self.blocks.len() - 1 {
            self.num_rows - block * ROWS_PER_BLOCK
        } else {
            ROWS_PER_BLOCK
        };
        let capacity = rows_here * self.row_bytes;

        let raw = match decoder {
            Some(d) => d
                .decompress(compressed, capacity)
                .context("zstd decompress (dict)")?,
            None => zstd::bulk::decompress(compressed, capacity).context("zstd decompress")?,
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

/// A [`ColdTier`] built from a feature-store file, decoding gathered raw
/// rows through the store's dtype into `f32`.
///
/// This is the compressed in-memory backing tier of
/// `FeatureCache` (see `FeatureCacheConfig::cold_store_path`):
/// the whole feature matrix held resident at a fraction of its raw size,
/// so a node absent from every cache tier is a block decompression away
/// instead of an error.
#[derive(Debug)]
pub struct ColdStore {
    tier: ColdTier,
    dtype: FeatureDtype,
    feature_dim: usize,
}

impl ColdStore {
    /// Read the feature file at `path` and compress its payload into a
    /// resident cold store. `level` is the zstd compression level.
    ///
    /// The payload streams through one block-sized buffer, so building
    /// needs memory for the compressed tier, not for the raw store.
    pub fn build_from_store(path: &std::path::Path, level: i32) -> Result<Self> {
        use std::os::unix::fs::FileExt;

        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open feature store {}", path.display()))?;
        let header = parse_feature_header(&file)?;
        let Some(row_bytes) = NonZeroUsize::new(header.feature_size) else {
            bail!("feature store {} has zero-width rows", path.display());
        };
        let block_size = ROWS_PER_BLOCK * row_bytes.get();
        let tier = ColdTier::build_from_blocks(header.num_nodes, row_bytes, level, |b, buf| {
            // In bounds: the header proved the payload fits the file.
            let start = b * block_size;
            let len = block_size.min(header.payload_bytes - start);
            buf.resize(len, 0);
            file.read_exact_at(buf, header.features_start_offset + start as u64)
                .context("failed to read feature payload")
        })?;
        Ok(Self {
            tier,
            dtype: header.dtype,
            feature_dim: header.feature_dim,
        })
    }

    /// Rows (nodes) held by the store.
    pub fn num_rows(&self) -> usize {
        self.tier.num_rows()
    }

    /// Feature dimension of every row.
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Compression ratio vs the raw payload.
    pub fn ratio(&self) -> f64 {
        self.tier.ratio()
    }

    /// Gather `nodes` into one contiguous decoded `f32` buffer in input
    /// order — `nodes.len() * feature_dim` values.
    pub fn gather(&self, nodes: &[u32]) -> Result<Vec<f32>> {
        let raw = self.tier.gather_rows(nodes)?;
        let mut out = vec![0f32; nodes.len() * self.feature_dim];
        for (i, chunk) in raw.chunks_exact(self.tier.row_bytes()).enumerate() {
            self.dtype.decode_row(
                chunk,
                &mut out[i * self.feature_dim..(i + 1) * self.feature_dim],
            );
        }
        Ok(out)
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
    fn cold_store_matches_feature_store() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("feat.bin");
        let (num_nodes, dim) = (1200usize, 32usize);
        let features: Vec<f32> = (0..num_nodes * dim)
            .map(|i| (i % 251) as f32 * 0.25)
            .collect();
        crate::features::save_features(&path, features.clone(), num_nodes, dim).unwrap();

        let store = ColdStore::build_from_store(&path, 9).unwrap();
        assert_eq!(store.num_rows(), num_nodes);
        assert_eq!(store.feature_dim(), dim);

        // Scattered gather with a repeat, crossing block boundaries.
        let nodes = [0u32, 7, 511, 512, 1199, 7];
        let got = store.gather(&nodes).unwrap();
        for (i, &n) in nodes.iter().enumerate() {
            assert_eq!(
                &got[i * dim..(i + 1) * dim],
                &features[n as usize * dim..(n as usize + 1) * dim],
                "node {n} at position {i}"
            );
        }
    }

    /// Repeated single-row gathers on one block reuse its decompressed
    /// copy, and the reuse window stays bounded.
    #[test]
    fn single_row_gathers_reuse_recent_blocks() {
        let (num_rows, row_bytes) = (ROWS_PER_BLOCK * 20, 32usize);
        let data = synthetic(num_rows, row_bytes);
        let tier = ColdTier::build(&data, row_bytes, 3).unwrap();
        assert!(!tier.dictionary.is_empty(), "20 blocks train a dictionary");

        for row in [5u32, 6, 7] {
            let got = tier.gather_rows(&[row]).unwrap();
            assert_eq!(
                got,
                data[row as usize * row_bytes..(row as usize + 1) * row_bytes]
            );
        }
        assert_eq!(
            tier.recent.lock().len(),
            1,
            "three rows of block 0 decompress it once"
        );

        for block in 0..RECENT_BLOCKS * 2 {
            let row = (block * ROWS_PER_BLOCK) as u32;
            let got = tier.gather_rows(&[row]).unwrap();
            assert_eq!(
                got,
                data[row as usize * row_bytes..(row as usize + 1) * row_bytes]
            );
        }
        assert_eq!(tier.recent.lock().len(), RECENT_BLOCKS);
        // A clone decodes identically with its own, empty reuse window.
        let copy = tier.clone();
        assert_eq!(
            copy.gather_rows(&[1, 9000]).unwrap(),
            tier.gather_rows(&[1, 9000]).unwrap()
        );
    }

    /// Streaming the build from a file matches building from memory,
    /// including a short final block.
    #[test]
    fn streamed_store_build_matches_the_in_memory_build() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("feat.bin");
        let (num_nodes, dim) = (ROWS_PER_BLOCK * 9 + 37, 16usize);
        let features: Vec<f32> = (0..num_nodes * dim).map(|i| (i % 97) as f32).collect();
        crate::features::save_features(&path, &features, num_nodes, dim).unwrap();

        let store = ColdStore::build_from_store(&path, 5).unwrap();
        let direct = ColdTier::build(bytemuck::cast_slice(&features), dim * 4, 5).unwrap();
        assert_eq!(store.tier.blocks.len(), direct.blocks.len());
        let all: Vec<u32> = (0..num_nodes as u32).collect();
        assert_eq!(store.gather(&all).unwrap(), features);
    }

    #[test]
    fn rejects_out_of_range_row() {
        let data = synthetic(50, 16);
        let tier = ColdTier::build(&data, 16, 3).unwrap();
        assert!(tier.gather_rows(&[49]).is_ok());
        assert!(tier.gather_rows(&[50]).is_err());
    }
}
