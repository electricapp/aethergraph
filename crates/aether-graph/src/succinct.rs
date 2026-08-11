//! Succinct integer sequence codecs for neighbor lists.
//!
//! CSR neighbor lists are sorted, so they compress far below 32 bits per
//! edge without giving up random access. Two codecs live here, each right
//! for a different access pattern:
//!
//! - [`EliasFano`] stores a monotone sequence in ~`2 + ceil(log2(u/n))`
//!   bits per element with O(1) [`EliasFano::get`] — the encoding for a
//!   node's sorted neighbor IDs, where the reader indexes individual
//!   neighbors.
//! - [`StreamVByte`] delta-encodes a sequence into a control stream plus a
//!   byte stream, decoded sequentially at GB/s — the encoding for edge
//!   deltas in the WAL and ingest paths, where the reader wants the whole
//!   list back.
//!
//! Both are pure integer algorithms with no platform or hardware
//! dependency; they run and are tested everywhere.

/// Elias-Fano encoding of a monotone non-decreasing `u64` sequence.
///
/// Each value splits into `low_bits` low bits (stored verbatim, packed)
/// and the remaining high bits (stored as a unary gap sequence in a bit
/// vector). Space is close to the information-theoretic minimum for a
/// sorted set, and [`Self::get`] reconstructs any element in constant
/// time by a select over the high-bits vector.
#[derive(Debug, Clone)]
pub struct EliasFano {
    /// Number of stored elements.
    len: usize,
    /// Bits taken from the low end of each value.
    low_bits: u32,
    /// Packed low parts, `low_bits` each, little-endian bit order.
    low: Vec<u64>,
    /// High-bits unary sequence: for element i with high part h_i, a set
    /// bit sits at position `h_i + i`. Reading element i means finding the
    /// (i+1)-th set bit.
    high: Vec<u64>,
    /// Largest value stored (for `low_bits` reconstruction / bounds).
    universe: u64,
}

impl EliasFano {
    /// Encode `values`, which must be sorted non-decreasing.
    ///
    /// # Panics
    /// Panics if `values` is not sorted — the encoding is meaningless for
    /// an unsorted input and a silent wrong answer is worse than a loud
    /// failure at build time.
    pub fn encode(values: &[u64]) -> Self {
        for w in values.windows(2) {
            assert!(w[0] <= w[1], "EliasFano requires a sorted sequence");
        }
        let len = values.len();
        let universe = values.last().copied().unwrap_or(0);

        // Classic Elias-Fano low-bit count: floor(log2(u/n)), which
        // balances the low-array and high-array sizes.
        let low_bits = if len == 0 || universe < len as u64 {
            0
        } else {
            (universe / len as u64).ilog2()
        };

        let mut low = BitWriter::new();
        let high_len = len + (universe >> low_bits) as usize + 1;
        let mut high = vec![0u64; high_len.div_ceil(64)];

        let low_mask = if low_bits == 0 {
            0
        } else {
            (1u64 << low_bits) - 1
        };
        for (i, &v) in values.iter().enumerate() {
            if low_bits > 0 {
                low.push_bits(v & low_mask, low_bits);
            }
            let high_pos = (v >> low_bits) as usize + i;
            high[high_pos / 64] |= 1u64 << (high_pos % 64);
        }

        Self {
            len,
            low_bits,
            low: low.into_words(),
            high,
            universe,
        }
    }

    /// Number of encoded elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The largest value stored.
    pub fn universe(&self) -> u64 {
        self.universe
    }

    /// The `i`-th element, in O(1). Returns `None` for `i >= len`.
    pub fn get(&self, i: usize) -> Option<u64> {
        if i >= self.len {
            return None;
        }
        // Find the (i+1)-th set bit in `high`; its position minus i is the
        // high part. `select` scans word by word — bounded by the high
        // array, which is O(n/64) words for the whole sequence but here we
        // stop at the (i+1)-th bit.
        let set_pos = self.select(i);
        let high_part = (set_pos - i) as u64;
        let low_part = if self.low_bits == 0 {
            0
        } else {
            self.read_low(i)
        };
        Some((high_part << self.low_bits) | low_part)
    }

    /// Decode every element into a freshly allocated vector.
    pub fn to_vec(&self) -> Vec<u64> {
        (0..self.len)
            .map(|i| self.get(i).expect("i < len"))
            .collect()
    }

    /// Total heap bytes of the two backing arrays — for reporting the
    /// achieved bits-per-element.
    pub fn heap_bytes(&self) -> usize {
        self.low.len() * 8 + self.high.len() * 8
    }

    /// Position of the `(rank+1)`-th set bit in `high`.
    fn select(&self, rank: usize) -> usize {
        let mut remaining = rank as i64;
        for (word_idx, &word) in self.high.iter().enumerate() {
            let ones = word.count_ones() as i64;
            if remaining < ones {
                // The target bit is in this word: skip `remaining` set bits.
                let mut w = word;
                for _ in 0..remaining {
                    w &= w - 1; // clear lowest set bit
                }
                return word_idx * 64 + w.trailing_zeros() as usize;
            }
            remaining -= ones;
        }
        unreachable!("select rank {rank} exceeds stored ones")
    }

    /// Read the `low_bits`-wide low part of element `i`.
    fn read_low(&self, i: usize) -> u64 {
        let bit = i * self.low_bits as usize;
        let word = bit / 64;
        let off = bit % 64;
        let lb = self.low_bits as usize;
        let low_mask = if lb == 64 { u64::MAX } else { (1u64 << lb) - 1 };
        if off + lb <= 64 {
            (self.low[word] >> off) & low_mask
        } else {
            // Spans two words.
            let low = self.low[word] >> off;
            let high = self.low[word + 1] << (64 - off);
            (low | high) & low_mask
        }
    }
}

/// Little-endian bit packer used to build the Elias-Fano low array.
struct BitWriter {
    words: Vec<u64>,
    bit_len: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            words: Vec::new(),
            bit_len: 0,
        }
    }

    fn push_bits(&mut self, value: u64, bits: u32) {
        if bits == 0 {
            return;
        }
        let word = self.bit_len / 64;
        let off = self.bit_len % 64;
        if word >= self.words.len() {
            self.words.push(0);
        }
        self.words[word] |= value << off;
        if off + bits as usize > 64 {
            // Spilled into the next word.
            self.words.push(value >> (64 - off));
        }
        self.bit_len += bits as usize;
    }

    fn into_words(self) -> Vec<u64> {
        self.words
    }
}

/// StreamVByte codec: delta-encode a `u32` sequence into a control stream
/// (2 bits per value = byte length 1–4) and a data stream (that many
/// little-endian bytes). Sorted inputs delta down to mostly 1-byte
/// values, and the split-stream layout is what a SIMD decoder (SSSE3
/// `pshufb` / NEON `tbl`) shuffles at GB/s. The scalar decoder here is
/// the portable reference the SIMD path must match.
#[derive(Debug, Clone, Default)]
pub struct StreamVByte {
    /// Number of encoded values.
    len: usize,
    /// First value; deltas are relative to it (0 when empty).
    first: u32,
    /// Two control bits per delta, packed four to a byte.
    control: Vec<u8>,
    /// Variable-length little-endian delta bytes.
    data: Vec<u8>,
}

impl StreamVByte {
    /// Delta-encode `values`. The sequence may be arbitrary `u32`s, but
    /// gains most on sorted input where consecutive deltas are small.
    ///
    /// # Panics
    /// Panics if a delta would exceed `u32::MAX` (only possible for a
    /// non-monotone sequence with a huge backward step); callers encoding
    /// neighbor lists pass sorted input, where deltas are non-negative.
    pub fn encode_deltas(values: &[u32]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let first = values[0];
        let n_deltas = values.len() - 1;
        let mut control = vec![0u8; n_deltas.div_ceil(4)];
        let mut data = Vec::with_capacity(n_deltas);

        let mut prev = first;
        for (i, &v) in values[1..].iter().enumerate() {
            let delta = v
                .checked_sub(prev)
                .or_else(|| prev.checked_sub(v))
                .expect("delta fits in u32");
            let (nbytes, tag) = svb_len(delta);
            let le = delta.to_le_bytes();
            data.extend_from_slice(&le[..nbytes]);
            control[i / 4] |= tag << ((i % 4) * 2);
            prev = v;
        }

        Self {
            len: values.len(),
            first,
            control,
            data,
        }
    }

    /// Number of encoded values.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total heap bytes of the control and data streams.
    pub fn heap_bytes(&self) -> usize {
        self.control.len() + self.data.len()
    }

    /// Decode the full sequence back. The scalar reference decoder;
    /// walks the control stream two bits at a time, reads that many data
    /// bytes, and running-sums the deltas back to absolute values.
    pub fn decode(&self) -> Vec<u32> {
        if self.len == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(self.len);
        out.push(self.first);
        let mut acc = self.first;
        let mut pos = 0usize;
        for i in 0..self.len - 1 {
            let tag = (self.control[i / 4] >> ((i % 4) * 2)) & 0b11;
            let nbytes = tag as usize + 1;
            let mut buf = [0u8; 4];
            buf[..nbytes].copy_from_slice(&self.data[pos..pos + nbytes]);
            let delta = u32::from_le_bytes(buf);
            acc = acc.wrapping_add(delta);
            out.push(acc);
            pos += nbytes;
        }
        out
    }
}

/// Byte length (1–4) and 2-bit tag for a StreamVByte value.
#[inline]
fn svb_len(v: u32) -> (usize, u8) {
    if v < (1 << 8) {
        (1, 0)
    } else if v < (1 << 16) {
        (2, 1)
    } else if v < (1 << 24) {
        (3, 2)
    } else {
        (4, 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elias_fano_round_trips_sorted() {
        let cases: Vec<Vec<u64>> = vec![
            vec![],
            vec![0],
            vec![5],
            vec![0, 0, 0, 0],
            vec![1, 2, 3, 4, 5],
            vec![0, 10, 20, 30, 1000],
            (0..1000).map(|i| i * 7).collect(),
            vec![0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55],
        ];
        for values in cases {
            let ef = EliasFano::encode(&values);
            assert_eq!(ef.len(), values.len());
            assert_eq!(ef.to_vec(), values, "round-trip for {values:?}");
            for (i, &v) in values.iter().enumerate() {
                assert_eq!(ef.get(i), Some(v), "get({i}) for {values:?}");
            }
            assert_eq!(ef.get(values.len()), None);
        }
    }

    #[test]
    fn elias_fano_large_universe() {
        // Sparse, large values: high universe, few elements.
        let values: Vec<u64> = vec![0, 1_000, 1_000_000, 4_000_000_000];
        let ef = EliasFano::encode(&values);
        assert_eq!(ef.to_vec(), values);
        assert_eq!(ef.universe(), 4_000_000_000);
    }

    #[test]
    fn elias_fano_beats_flat_on_dense_sorted() {
        // 4096 sorted IDs in a 65536 universe: EF should use well under
        // the 8 bytes/elem a flat u64 array would.
        let values: Vec<u64> = (0..4096u64).map(|i| i * 16).collect();
        let ef = EliasFano::encode(&values);
        let bits_per = ef.heap_bytes() as f64 * 8.0 / values.len() as f64;
        assert!(bits_per < 16.0, "EF used {bits_per:.1} bits/elem");
    }

    #[test]
    #[should_panic(expected = "sorted")]
    fn elias_fano_rejects_unsorted() {
        EliasFano::encode(&[3, 1, 2]);
    }

    #[test]
    fn streamvbyte_round_trips() {
        let cases: Vec<Vec<u32>> = vec![
            vec![],
            vec![42],
            vec![0, 1, 2, 3, 4],
            vec![0, 255, 256, 65535, 65536, 16_777_215, 16_777_216],
            (0..1000).map(|i| i * 3).collect(),
            vec![100, 100, 100, 100],
            vec![0, u32::MAX],
        ];
        for values in cases {
            let svb = StreamVByte::encode_deltas(&values);
            assert_eq!(svb.len(), values.len());
            assert_eq!(svb.decode(), values, "round-trip for {values:?}");
        }
    }

    #[test]
    fn streamvbyte_shrinks_dense_sequence() {
        // Consecutive IDs: every delta is 1, so ~1 byte per value plus
        // control — well under 4 bytes/value flat.
        let values: Vec<u32> = (0..10_000).collect();
        let svb = StreamVByte::encode_deltas(&values);
        assert!(
            svb.heap_bytes() < values.len() * 2,
            "expected < 2 bytes/value, got {}",
            svb.heap_bytes() as f64 / values.len() as f64
        );
    }

    #[test]
    fn svb_len_boundaries() {
        assert_eq!(svb_len(0), (1, 0));
        assert_eq!(svb_len(255), (1, 0));
        assert_eq!(svb_len(256), (2, 1));
        assert_eq!(svb_len(65_535), (2, 1));
        assert_eq!(svb_len(65_536), (3, 2));
        assert_eq!(svb_len(16_777_215), (3, 2));
        assert_eq!(svb_len(16_777_216), (4, 3));
        assert_eq!(svb_len(u32::MAX), (4, 3));
    }
}
