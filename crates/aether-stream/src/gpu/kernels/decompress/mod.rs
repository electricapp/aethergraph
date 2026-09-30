//! K5.5 StreamVByte + Elias-Fano GPU decode and CPU oracles.
//!
//! Device paths are warp-cooperative (SVB) / CTA-parallel (EF), still
//! bit-exact against the CPU codecs.
//!
//! The kernels trust their input's geometry, so the decoders take only
//! [`StreamVByteDevice`] and [`EliasFanoDeviceParts`], which can only be
//! built from the core codec types — themselves only produced by an encoder
//! or a validating `read_from`.

use aethergraph_core::{EliasFano, StreamVByte};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

pub(super) const KERNEL_SRC: &str = concat!(
    include_str!("../common.cuh"),
    "\n",
    include_str!("decompress.cu")
);

/// Byte count of the data stream a StreamVByte `control` stream describes
/// for `len` values, or why the control stream cannot describe them.
fn streamvbyte_data_len(control: &[u8], len: usize) -> Result<usize, &'static str> {
    if len == 0 {
        return if control.is_empty() {
            Ok(0)
        } else {
            Err("empty StreamVByte has control bytes")
        };
    }
    if control.len() != (len - 1).div_ceil(4) {
        return Err("control length does not cover all deltas");
    }
    Ok((0..len - 1)
        .map(|index| ((control[index / 4] >> ((index % 4) * 2)) & 3) as usize + 1)
        .sum())
}

/// Decode an aethergraph-core StreamVByte control/data pair without CUDA.
pub fn cpu_streamvbyte_delta_decode(
    first: u32,
    control: &[u8],
    data: &[u8],
    len: usize,
) -> Result<Vec<u32>, &'static str> {
    let data_len = streamvbyte_data_len(control, len)?;
    if len == 0 {
        return if data.is_empty() {
            Ok(Vec::new())
        } else {
            Err("empty StreamVByte has payload")
        };
    }
    if data.len() < data_len {
        return Err("truncated data");
    }
    if data.len() > data_len {
        return Err("trailing data");
    }
    let mut output = Vec::with_capacity(len);
    output.push(first);
    let mut acc = first;
    let mut pos = 0;
    for index in 0..len - 1 {
        let byte_count = ((control[index / 4] >> ((index % 4) * 2)) & 3) as usize + 1;
        let mut word = [0_u8; 4];
        word[..byte_count].copy_from_slice(&data[pos..pos + byte_count]);
        acc = acc.wrapping_add(u32::from_le_bytes(word));
        output.push(acc);
        pos += byte_count;
    }
    Ok(output)
}

/// One StreamVByte sequence resident in VRAM. Built only from a core
/// [`StreamVByte`], whose control stream covers exactly `len - 1` deltas and
/// whose data stream holds exactly the bytes those tags call for — the
/// bounds the kernel's reads rely on.
pub struct StreamVByteDevice {
    control: CudaSlice<u8>,
    data: CudaSlice<u8>,
    len: usize,
    first: u32,
}

impl StreamVByteDevice {
    /// Upload `svb` onto `stream`.
    pub fn upload(
        stream: &Arc<CudaStream>,
        svb: &StreamVByte,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // The kernel indexes values and data bytes with `int`.
        if i32::try_from(svb.len()).is_err() || i32::try_from(svb.data().len()).is_err() {
            return Err(format!(
                "{} values in {} data bytes exceed the kernel's i32 extents",
                svb.len(),
                svb.data().len()
            )
            .into());
        }
        let mut control = stream.alloc_zeros::<u8>(svb.control().len().max(1))?;
        if !svb.control().is_empty() {
            stream.memcpy_htod(svb.control(), &mut control)?;
        }
        let mut data = stream.alloc_zeros::<u8>(svb.data().len().max(1))?;
        if !svb.data().is_empty() {
            stream.memcpy_htod(svb.data(), &mut data)?;
        }
        Ok(Self {
            control,
            data,
            len: svb.len(),
            first: svb.first(),
        })
    }

    /// Number of encoded values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the sequence is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Compiled StreamVByte decoder and its output buffer.
pub struct StreamVByteDecoder {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    output: CudaSlice<u32>,
    max_len: usize,
}

impl StreamVByteDecoder {
    /// Compile the decoder and allocate its output capacity.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        max_len: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let module = ctx.load_module(super::compile_for_device(ctx, KERNEL_SRC)?)?;
        Ok(Self {
            stream: stream.clone(),
            func: module.load_function("streamvbyte_delta_decode")?,
            output: stream.alloc_zeros(max_len.max(1))?,
            max_len,
        })
    }

    /// Enqueue a warp-cooperative decode of `src` into the output buffer.
    pub fn decode(&mut self, src: &StreamVByteDevice) -> Result<(), Box<dyn std::error::Error>> {
        if src.len > self.max_len {
            return Err(format!("len {} exceeds {}", src.len, self.max_len).into());
        }
        if src.len == 0 {
            return Ok(());
        }
        let len_i32 = i32::try_from(src.len)?;
        // SAFETY: signature matches streamvbyte_delta_decode; `src` was
        // validated on upload, so the kernel's control and data reads stay
        // inside their buffers, and the output holds `len` values.
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(&src.control)
                .arg(&src.data)
                .arg(&mut self.output)
                .arg(&len_i32)
                .arg(&src.first)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Decoded device output.
    pub fn output(&self) -> &CudaSlice<u32> {
        &self.output
    }
}

/// Device buffers for one Elias-Fano sequence (matches core encoder layout).
/// Built only from a core [`EliasFano`], whose low array covers `len` fields
/// of `low_bits < 64` and whose high bitmap marks exactly `len` values — the
/// bounds the kernel's reads rely on.
pub struct EliasFanoDeviceParts {
    low: CudaSlice<u64>,
    high: CudaSlice<u64>,
    low_bits: u32,
    len: usize,
    /// Logical high-word count (may be less than `high.len()` padding).
    high_words: usize,
}

impl EliasFanoDeviceParts {
    /// Upload `ef` onto `stream`.
    pub fn upload(
        stream: &Arc<CudaStream>,
        ef: &EliasFano,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if i32::try_from(ef.len()).is_err() || i32::try_from(ef.high_words().len()).is_err() {
            return Err("Elias-Fano sequence exceeds the kernel's i32 extents".into());
        }

        let mut low = stream.alloc_zeros::<u64>(ef.low_words().len().max(1))?;
        let mut high = stream.alloc_zeros::<u64>(ef.high_words().len().max(1))?;
        if !ef.low_words().is_empty() {
            stream.memcpy_htod(ef.low_words(), &mut low)?;
        }
        if !ef.high_words().is_empty() {
            stream.memcpy_htod(ef.high_words(), &mut high)?;
        }
        Ok(Self {
            low,
            high,
            low_bits: ef.low_bits(),
            len: ef.len(),
            high_words: ef.high_words().len(),
        })
    }

    /// Number of encoded values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the sequence is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Compiled Elias-Fano full-sequence decoder (`to_vec` equivalent).
pub struct EliasFanoDecoder {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    output: CudaSlice<u64>,
    max_len: usize,
}

impl EliasFanoDecoder {
    /// Compile the decoder (shares PTX with StreamVByte) and allocate output.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        max_len: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let module = ctx.load_module(super::compile_for_device(ctx, KERNEL_SRC)?)?;
        Ok(Self {
            stream: stream.clone(),
            func: module.load_function("elias_fano_decode_all")?,
            output: stream.alloc_zeros(max_len.max(1))?,
            max_len,
        })
    }

    /// Decode every value into the internal output buffer.
    pub fn decode_all(
        &mut self,
        parts: &EliasFanoDeviceParts,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if parts.len > self.max_len {
            return Err(format!("len {} exceeds {}", parts.len, self.max_len).into());
        }
        if parts.len == 0 {
            return Ok(());
        }
        let len = i32::try_from(parts.len)?;
        let high_words = i32::try_from(parts.high_words)?;
        let low_bits = parts.low_bits;
        // SAFETY: matches elias_fano_decode_all; `parts` was validated on
        // upload, so every low-bit field and high word the kernel reads is
        // inside its buffer, and the output holds `len` values.
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(&parts.low)
                .arg(&parts.high)
                .arg(&mut self.output)
                .arg(&len)
                .arg(&high_words)
                .arg(&low_bits)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    // pops[256] + excl[256]
                    shared_mem_bytes: 256 * 2 * 4,
                })?;
        }
        Ok(())
    }

    /// Decoded device output.
    pub fn output(&self) -> &CudaSlice<u64> {
        &self.output
    }
}

#[cfg(test)]
mod tests {
    use super::{cpu_streamvbyte_delta_decode, streamvbyte_data_len};
    use aethergraph_core::{EliasFano, StreamVByte};

    #[test]
    fn cpu_decoder_matches_control_stream_layout() {
        assert_eq!(
            cpu_streamvbyte_delta_decode(10, &[0b0000_0100], &[1, 44, 1, 2], 4),
            Ok(vec![10, 11, 311, 313])
        );
    }

    #[test]
    fn cpu_decoder_rejects_mismatched_streams() {
        assert!(cpu_streamvbyte_delta_decode(10, &[0b0000_0100], &[1, 44, 1], 4).is_err());
        assert!(cpu_streamvbyte_delta_decode(10, &[0b0000_0100], &[1, 44, 1, 2, 9], 4).is_err());
        assert!(cpu_streamvbyte_delta_decode(10, &[], &[1], 4).is_err());
        assert!(cpu_streamvbyte_delta_decode(0, &[1], &[], 0).is_err());
    }

    /// The device upload's check is the CPU decoder's: every encoded
    /// sequence must pass it with exactly its own data length.
    #[test]
    fn encoded_streams_describe_their_data_exactly() {
        for values in [vec![], vec![7], vec![1, 300, 70_000, 70_001, 1 << 30]] {
            let svb = StreamVByte::encode_deltas(&values);
            assert_eq!(
                streamvbyte_data_len(svb.control(), svb.len()),
                Ok(svb.data().len())
            );
        }
    }

    #[test]
    fn elias_fano_accessors_round_trip_to_vec() {
        let values = [0u64, 1, 1, 4, 7, 10, 25];
        let ef = EliasFano::encode(&values);
        assert_eq!(ef.to_vec(), values);
        assert!(!ef.high_words().is_empty());
    }
}
