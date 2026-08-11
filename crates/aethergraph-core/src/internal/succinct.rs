//! Succinct integer sequence codecs backing the compressed CSR format.
//!
//! The version-2 graph file (see `internal::compressed_graph`) stores the
//! two CSR arrays through these codecs:
//!
//! - [`EliasFano`] stores a monotone sequence in ~`2 + ceil(log2(u/n))`
//!   bits per element — the encoding for the offsets array, which is a
//!   prefix-sum and therefore always monotone.
//! - [`StreamVByte`] delta-encodes a sequence into a control stream plus a
//!   byte stream, decoded sequentially at GB/s — the encoding for the
//!   edges array, where sorted neighbor lists delta down to mostly
//!   single-byte values.
//!
//! Both are pure integer algorithms with no platform or hardware
//! dependency; they run and are tested everywhere. Each type also defines
//! its own little-endian wire format ([`EliasFano::write_into`] /
//! [`EliasFano::read_from`], and the [`StreamVByte`] equivalents) so the
//! file format has exactly one serializer per codec.

use anyhow::Context;

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
    ///
    /// Sequential: one pass over the high-bits words plus one low-bits
    /// read per element — O(n + words), unlike a loop over [`Self::get`]
    /// whose per-call select would make full decode quadratic.
    pub fn to_vec(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.len);
        let mut i = 0usize;
        'words: for (word_idx, &word) in self.high.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let set_pos = word_idx * 64 + w.trailing_zeros() as usize;
                w &= w - 1;
                let high_part = (set_pos - i) as u64;
                let low_part = if self.low_bits == 0 {
                    0
                } else {
                    self.read_low(i)
                };
                out.push((high_part << self.low_bits) | low_part);
                i += 1;
                if i == self.len {
                    break 'words;
                }
            }
        }
        debug_assert_eq!(out.len(), self.len, "high bit count matches len");
        out
    }

    /// Total heap bytes of the two backing arrays — for reporting the
    /// achieved bits-per-element.
    pub fn heap_bytes(&self) -> usize {
        self.low.len() * 8 + self.high.len() * 8
    }

    /// Serialize into `out` as little-endian:
    /// `len u64 | universe u64 | low_bits u32 | low word count u64 |
    /// low words | high word count u64 | high words`.
    pub fn write_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.len as u64).to_le_bytes());
        out.extend_from_slice(&self.universe.to_le_bytes());
        out.extend_from_slice(&self.low_bits.to_le_bytes());
        out.extend_from_slice(&(self.low.len() as u64).to_le_bytes());
        for w in &self.low {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&(self.high.len() as u64).to_le_bytes());
        for w in &self.high {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }

    /// Parse one serialized sequence from the front of `bytes`, returning
    /// it and the number of bytes consumed. Validates structural
    /// consistency (word counts sized for `len`/`universe`, set-bit count
    /// equal to `len`) so a decoded value can be trusted downstream.
    pub fn read_from(bytes: &[u8]) -> anyhow::Result<(Self, usize)> {
        let mut r = ByteReader::new(bytes);
        let len = usize::try_from(r.u64()?).context("EF len")?;
        let universe = r.u64()?;
        let low_bits = r.u32()?;
        anyhow::ensure!(low_bits < 64, "EF low_bits {low_bits} out of range");
        let low = r.u64_vec().context("EF low words")?;
        let high = r.u64_vec().context("EF high words")?;

        let expect_low_words = (len * low_bits as usize).div_ceil(64);
        anyhow::ensure!(
            low.len() >= expect_low_words,
            "EF low array truncated: {} words, need {expect_low_words}",
            low.len()
        );
        if len > 0 {
            let last_pos = (universe >> low_bits) as usize + len - 1;
            anyhow::ensure!(
                high.len() * 64 > last_pos,
                "EF high array truncated for universe {universe}"
            );
        }
        let ones: usize = high.iter().map(|w| w.count_ones() as usize).sum();
        anyhow::ensure!(
            ones == len,
            "EF high array has {ones} set bits for {len} elements"
        );

        Ok((
            Self {
                len,
                low_bits,
                low,
                high,
                universe,
            },
            r.consumed(),
        ))
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
    /// Delta-encode `values`. Deltas are wrapping (`v.wrapping_sub(prev)`),
    /// so any `u32` sequence round-trips exactly through [`Self::decode`]'s
    /// wrapping running sum; sorted input is where the encoding gains,
    /// because forward deltas are small. A backward step encodes as a
    /// large wrapping delta (4 bytes) — correct, merely uncompressed.
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
            let delta = v.wrapping_sub(prev);
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

impl StreamVByte {
    /// Serialize into `out` as little-endian:
    /// `len u64 | first u32 | control byte count u64 | control bytes |
    /// data byte count u64 | data bytes`.
    pub fn write_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.len as u64).to_le_bytes());
        out.extend_from_slice(&self.first.to_le_bytes());
        out.extend_from_slice(&(self.control.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.control);
        out.extend_from_slice(&(self.data.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.data);
    }

    /// Parse one serialized sequence from the front of `bytes`, returning
    /// it and the number of bytes consumed. Validates that the control
    /// stream covers `len - 1` deltas and that the data stream holds
    /// exactly the bytes the control tags call for, so [`Self::decode`]
    /// can run unchecked.
    pub fn read_from(bytes: &[u8]) -> anyhow::Result<(Self, usize)> {
        let mut r = ByteReader::new(bytes);
        let len = usize::try_from(r.u64()?).context("SVB len")?;
        let first = r.u32()?;
        let control = r.byte_vec().context("SVB control stream")?;
        let data = r.byte_vec().context("SVB data stream")?;

        let n_deltas = len.saturating_sub(1);
        anyhow::ensure!(
            control.len() == n_deltas.div_ceil(4),
            "SVB control stream is {} bytes for {n_deltas} deltas",
            control.len()
        );
        let need: usize = (0..n_deltas)
            .map(|i| (((control[i / 4] >> ((i % 4) * 2)) & 0b11) as usize) + 1)
            .sum();
        anyhow::ensure!(
            data.len() == need,
            "SVB data stream is {} bytes, control tags call for {need}",
            data.len()
        );

        Ok((
            Self {
                len,
                first,
                control,
                data,
            },
            r.consumed(),
        ))
    }
}

/// Bounds-checked little-endian cursor for the codec wire formats.
struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| anyhow::anyhow!("truncated: need {n} bytes at {}", self.pos))?;
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> anyhow::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4")))
    }

    fn u64(&mut self) -> anyhow::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8")))
    }

    fn byte_vec(&mut self) -> anyhow::Result<Vec<u8>> {
        let n = usize::try_from(self.u64()?).context("length prefix")?;
        Ok(self.take(n)?.to_vec())
    }

    fn u64_vec(&mut self) -> anyhow::Result<Vec<u64>> {
        let n = usize::try_from(self.u64()?).context("length prefix")?;
        let raw = self.take(n.checked_mul(8).context("word count overflow")?)?;
        Ok(raw
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().expect("8")))
            .collect())
    }

    fn consumed(&self) -> usize {
        self.pos
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
    fn streamvbyte_round_trips_non_monotone() {
        // Wrapping deltas make arbitrary sequences exact — backward steps
        // and large jumps included.
        let cases: Vec<Vec<u32>> = vec![
            vec![5, 3],
            vec![10, 0, u32::MAX, 1, 2, 1],
            vec![u32::MAX, 0, u32::MAX],
            (0..500)
                .map(|i| (i * 2_654_435_761u64 % 4_294_967_291) as u32)
                .collect(),
        ];
        for values in cases {
            let svb = StreamVByte::encode_deltas(&values);
            assert_eq!(svb.decode(), values, "round-trip for {values:?}");
        }
    }

    #[test]
    fn elias_fano_serialization_round_trips() {
        let cases: Vec<Vec<u64>> = vec![
            vec![],
            vec![0],
            vec![0, 0, 7, 7, 1_000_000],
            (0..2048).map(|i| i * 13).collect(),
        ];
        for values in cases {
            let ef = EliasFano::encode(&values);
            let mut buf = vec![0xAA; 3]; // leading noise the reader must skip past
            let start = buf.len();
            ef.write_into(&mut buf);
            buf.extend_from_slice(&[0xBB; 5]); // trailing bytes beyond one record
            let (back, consumed) = EliasFano::read_from(&buf[start..]).unwrap();
            assert_eq!(consumed, buf.len() - start - 5, "consumed for {values:?}");
            assert_eq!(back.to_vec(), values, "round-trip for {values:?}");
        }
    }

    #[test]
    fn streamvbyte_serialization_round_trips() {
        let values: Vec<u32> = (0..3000).map(|i| i * 3 + (i % 7)).collect();
        let svb = StreamVByte::encode_deltas(&values);
        let mut buf = Vec::new();
        svb.write_into(&mut buf);
        let (back, consumed) = StreamVByte::read_from(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(back.decode(), values);
    }

    #[test]
    fn serialization_rejects_corruption() {
        let ef = EliasFano::encode(&[1, 5, 9, 200]);
        let mut buf = Vec::new();
        ef.write_into(&mut buf);
        // Truncation.
        assert!(EliasFano::read_from(&buf[..buf.len() - 1]).is_err());
        // A cleared high word breaks the set-bit count invariant.
        let mut tampered = buf.clone();
        let last8 = tampered.len() - 8;
        tampered[last8..].fill(0);
        assert!(EliasFano::read_from(&tampered).is_err());

        let svb = StreamVByte::encode_deltas(&[1, 500, 9]);
        let mut buf = Vec::new();
        svb.write_into(&mut buf);
        assert!(StreamVByte::read_from(&buf[..buf.len() - 1]).is_err());
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
