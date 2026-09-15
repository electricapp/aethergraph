//! Block-scaled int8 codec for feature rows.
//!
//! [`internal::succinct`] compresses topology, the constant term in a batch's
//! byte budget. This compresses the feature payload, the per-batch term and
//! the larger one — a 128-seed `15x10` sample touches ~19k nodes, so at
//! `feature_dim = 128` the gather moves ~9.8 MB of features against ~1.5 MB
//! of adjacency. Only the second term moves the streaming asymptote.
//!
//! SoA and fixed-block so a device gather can read it with vector loads:
//! `codes` is one `i8` per element, row-major; `scales` one `f32` per
//! [`BlockScaledI8::BLOCK`] elements per row. 32 is a multiple of the 16
//! codes a lane pulls per `ld.global.cs.v4`, so a vector load never straddles
//! two scales.
//!
//! Symmetric: `scale = max|x| / 127`, `q = round(x / scale)` clamped to
//! `[-127, 127]`, decode `q as f32 * scale`. One IEEE multiply, so the device
//! decoder is bit-identical to [`BlockScaledI8::decode`], not merely close.
//!
//! The open question is accuracy, not throughput — whether GNN quality
//! tolerates int8 features is a sweep to run.

/// Rows of `f32` features stored as per-block-scaled `i8`.
///
/// One byte per element plus 4 bytes of scale per 32 elements: 1.125 bytes
/// per feature against `f32`'s 4, so 3.56x smaller for any `feature_dim`.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockScaledI8 {
    rows: usize,
    dim: usize,
    /// `rows * blocks_per_row(dim)` scales, row-major.
    scales: Vec<f32>,
    /// `rows * dim` codes, row-major.
    codes: Vec<i8>,
}

/// Why a slice of features could not be encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureCodecError {
    /// `dim` was zero.
    ZeroDim,
    /// `values.len()` was not a whole number of `dim`-element rows.
    Ragged,
    /// A value was NaN or infinite; a scale derived from it would poison
    /// the whole block.
    NotFinite,
}

impl std::fmt::Display for FeatureCodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDim => f.write_str("feature_dim must be non-zero"),
            Self::Ragged => f.write_str("values length is not a multiple of feature_dim"),
            Self::NotFinite => f.write_str("feature values must be finite"),
        }
    }
}

impl std::error::Error for FeatureCodecError {}

impl BlockScaledI8 {
    /// Elements sharing one scale.
    pub const BLOCK: usize = 32;

    /// Scales stored per row for `dim` features.
    #[must_use]
    pub const fn blocks_per_row(dim: usize) -> usize {
        dim.div_ceil(Self::BLOCK)
    }

    /// Quantize `values`, interpreted as `values.len() / dim` rows.
    pub fn encode_rows(values: &[f32], dim: usize) -> Result<Self, FeatureCodecError> {
        if dim == 0 {
            return Err(FeatureCodecError::ZeroDim);
        }
        if !values.len().is_multiple_of(dim) {
            return Err(FeatureCodecError::Ragged);
        }
        let rows = values.len() / dim;
        let per_row = Self::blocks_per_row(dim);
        let mut scales = Vec::with_capacity(rows * per_row);
        let mut codes = vec![0i8; values.len()];

        for (row, src) in values.chunks_exact(dim).enumerate() {
            for (b, block) in src.chunks(Self::BLOCK).enumerate() {
                let mut peak = 0.0f32;
                for &v in block {
                    if !v.is_finite() {
                        return Err(FeatureCodecError::NotFinite);
                    }
                    peak = peak.max(v.abs());
                }
                // An all-zero block gets a zero scale, which decodes back to
                // zeros without a division ever happening.
                let scale = if peak == 0.0 { 0.0 } else { peak / 127.0 };
                scales.push(scale);
                if scale == 0.0 {
                    continue;
                }
                let base = row * dim + b * Self::BLOCK;
                for (i, &v) in block.iter().enumerate() {
                    codes[base + i] = (v / scale).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
        Ok(Self {
            rows,
            dim,
            scales,
            codes,
        })
    }

    /// Reference decode. The device kernel reproduces this bit for bit.
    pub fn decode_into(&self, out: &mut [f32]) -> Result<(), FeatureCodecError> {
        if out.len() != self.codes.len() {
            return Err(FeatureCodecError::Ragged);
        }
        let per_row = Self::blocks_per_row(self.dim);
        for row in 0..self.rows {
            for e in 0..self.dim {
                let scale = self.scales[row * per_row + e / Self::BLOCK];
                out[row * self.dim + e] = f32::from(self.codes[row * self.dim + e]) * scale;
            }
        }
        Ok(())
    }

    /// Reference decode into a fresh buffer.
    #[must_use]
    pub fn decode(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.codes.len()];
        self.decode_into(&mut out).expect("sized from codes");
        out
    }

    /// Row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Features per row.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Per-block scales, row-major.
    #[must_use]
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Quantized codes, row-major.
    #[must_use]
    pub fn codes(&self) -> &[i8] {
        &self.codes
    }

    /// Encoded bytes per `f32` byte, scales included.
    #[must_use]
    pub fn bytes_per_f32_byte(&self) -> f64 {
        let encoded = self.codes.len() + self.scales.len() * 4;
        encoded as f64 / (self.codes.len() * 4) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockScaledI8, FeatureCodecError};

    fn ramp(rows: usize, dim: usize) -> Vec<f32> {
        (0..rows * dim).map(|i| (i as f32) * 0.25 - 3.0).collect()
    }

    #[test]
    fn decode_recovers_values_within_a_quantization_step() {
        let dim = 96;
        let src = ramp(4, dim);
        let enc = BlockScaledI8::encode_rows(&src, dim).unwrap();
        let out = enc.decode();
        for (block, (a, b)) in src
            .chunks(BlockScaledI8::BLOCK)
            .zip(out.chunks(BlockScaledI8::BLOCK))
            .enumerate()
        {
            let _ = block;
            let peak = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let step = peak / 127.0;
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() <= step * 0.5 + 1e-6, "{x} vs {y}");
            }
        }
    }

    #[test]
    fn scale_is_per_block_not_per_row() {
        // A row whose second block is tiny must not be crushed by the first
        // block's peak — that is the whole point of block scaling.
        let mut src = vec![0.0f32; 64];
        src[0] = 1000.0;
        src[32] = 0.5;
        src[33] = -0.25;
        let enc = BlockScaledI8::encode_rows(&src, 64).unwrap();
        assert_eq!(enc.scales().len(), 2);
        let out = enc.decode();
        assert!((out[32] - 0.5).abs() < 0.01, "{}", out[32]);
        assert!((out[33] + 0.25).abs() < 0.01, "{}", out[33]);
    }

    #[test]
    fn zero_block_round_trips_exactly() {
        let src = vec![0.0f32; 64];
        let enc = BlockScaledI8::encode_rows(&src, 32).unwrap();
        assert_eq!(enc.scales(), &[0.0, 0.0]);
        assert_eq!(enc.decode(), src);
    }

    #[test]
    fn partial_trailing_block_is_encoded() {
        let dim = 40; // one full block plus 8
        let src = ramp(2, dim);
        let enc = BlockScaledI8::encode_rows(&src, dim).unwrap();
        assert_eq!(enc.scales().len(), 2 * 2);
        assert_eq!(enc.codes().len(), 2 * dim);
        assert_eq!(enc.decode().len(), 2 * dim);
    }

    #[test]
    fn rejects_shapes_and_values_it_cannot_represent() {
        assert_eq!(
            BlockScaledI8::encode_rows(&[1.0], 0),
            Err(FeatureCodecError::ZeroDim)
        );
        assert_eq!(
            BlockScaledI8::encode_rows(&[1.0, 2.0, 3.0], 2),
            Err(FeatureCodecError::Ragged)
        );
        assert_eq!(
            BlockScaledI8::encode_rows(&[1.0, f32::NAN], 2),
            Err(FeatureCodecError::NotFinite)
        );
        assert_eq!(
            BlockScaledI8::encode_rows(&[f32::INFINITY, 1.0], 2),
            Err(FeatureCodecError::NotFinite)
        );
    }

    #[test]
    fn moves_an_eighth_fewer_than_a_quarter_of_the_bytes() {
        // 1 byte per element + 4 bytes per 32-element block = 1.125 B/feature
        // against f32's 4.
        let enc = BlockScaledI8::encode_rows(&ramp(8, 128), 128).unwrap();
        assert!((enc.bytes_per_f32_byte() - 1.125 / 4.0).abs() < 1e-9);
    }
}
