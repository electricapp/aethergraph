//! K5.5 StreamVByte + Elias-Fano GPU decode and CPU oracles.
//!
//! Device paths are warp-cooperative (SVB) / CTA-parallel (EF), still
//! bit-exact against the CPU codecs.

use aethergraph_core::EliasFano;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

const KERNEL_SRC: &str = concat!(
    include_str!("../common.cuh"),
    "\n",
    include_str!("decompress.cu")
);

/// Decode an aethergraph-core StreamVByte control/data pair without CUDA.
pub fn cpu_streamvbyte_delta_decode(
    first: u32,
    control: &[u8],
    data: &[u8],
    len: usize,
) -> Result<Vec<u32>, &'static str> {
    if len == 0 {
        return if control.is_empty() && data.is_empty() {
            Ok(Vec::new())
        } else {
            Err("empty StreamVByte has payload")
        };
    }
    if control.len() != (len - 1).div_ceil(4) {
        return Err("control length does not cover all deltas");
    }
    let mut output = Vec::with_capacity(len);
    output.push(first);
    let mut acc = first;
    let mut pos = 0;
    for index in 0..len - 1 {
        let byte_count = ((control[index / 4] >> ((index % 4) * 2)) & 3) as usize + 1;
        let bytes = data.get(pos..pos + byte_count).ok_or("truncated data")?;
        let mut word = [0_u8; 4];
        word[..byte_count].copy_from_slice(bytes);
        acc = acc.wrapping_add(u32::from_le_bytes(word));
        output.push(acc);
        pos += byte_count;
    }
    if pos != data.len() {
        return Err("trailing data");
    }
    Ok(output)
}

/// Compiled StreamVByte scalar decoder.
pub struct StreamVByteDecoder {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    output: CudaSlice<u32>,
    max_len: usize,
}

impl StreamVByteDecoder {
    /// Compile the baseline decoder and allocate its output capacity.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        max_len: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let module = ctx.load_module(cudarc::nvrtc::compile_ptx(KERNEL_SRC)?)?;
        Ok(Self {
            stream: stream.clone(),
            func: module.load_function("streamvbyte_delta_decode")?,
            output: stream.alloc_zeros(max_len)?,
            max_len,
        })
    }

    /// Enqueue warp-cooperative StreamVByte decode.
    pub fn decode(
        &mut self,
        control: &CudaSlice<u8>,
        data: &CudaSlice<u8>,
        len: usize,
        first: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if len > self.max_len {
            return Err(format!("len {len} exceeds {}", self.max_len).into());
        }
        let len_i32 = i32::try_from(len)?;
        // SAFETY: signature matches streamvbyte_delta_decode.
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(control)
                .arg(data)
                .arg(&mut self.output)
                .arg(&len_i32)
                .arg(&first)
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
pub struct EliasFanoDeviceParts {
    pub low: CudaSlice<u64>,
    pub high: CudaSlice<u64>,
    pub low_bits: u32,
    pub len: usize,
    /// Logical high-word count (may be less than `high.len()` padding).
    pub high_words: usize,
}

impl EliasFanoDeviceParts {
    /// Upload an [`EliasFano`] onto `stream`.
    pub fn upload(
        stream: &Arc<CudaStream>,
        ef: &EliasFano,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut low = unsafe { stream.alloc::<u64>(ef.low_words().len().max(1))? };
        let mut high = unsafe { stream.alloc::<u64>(ef.high_words().len().max(1))? };
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
        let module: CudaModule = ctx.load_module(cudarc::nvrtc::compile_ptx(KERNEL_SRC)?)?;
        Ok(Self {
            stream: stream.clone(),
            func: module.load_function("elias_fano_decode_all")?,
            output: stream.alloc_zeros(max_len)?,
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
        // SAFETY: matches elias_fano_decode_all; parts cover the encoded sequence.
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
    use super::cpu_streamvbyte_delta_decode;
    use aethergraph_core::EliasFano;

    #[test]
    fn cpu_decoder_matches_control_stream_layout() {
        assert_eq!(
            cpu_streamvbyte_delta_decode(10, &[0b0000_0100], &[1, 44, 1, 2], 4),
            Ok(vec![10, 11, 311, 313])
        );
    }

    #[test]
    fn elias_fano_accessors_round_trip_to_vec() {
        let values = [0u64, 1, 1, 4, 7, 10, 25];
        let ef = EliasFano::encode(&values);
        assert_eq!(ef.to_vec(), values);
        assert!(!ef.high_words().is_empty());
    }
}
