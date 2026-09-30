use aether_stream::gpu::kernels::harness::cuda_or_skip;
use aether_stream::gpu::kernels::{
    EliasFanoDecoder, EliasFanoDeviceParts, StreamVByteDecoder, StreamVByteDevice,
};
use aethergraph_core::{EliasFano, StreamVByte};

#[test]
fn streamvbyte_device_matches_cpu() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Crosses a 32-delta wave and mixes 1- to 4-byte deltas.
    let values: Vec<u32> = (0..100u32).map(|i| i * i * 997 + (i << 20)).collect();
    let svb = StreamVByte::encode_deltas(&values);
    let expect = svb.decode();

    let src = StreamVByteDevice::upload(&stream, &svb).unwrap();
    let mut dec = StreamVByteDecoder::new(&ctx, &stream, values.len()).unwrap();
    dec.decode(&src).unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0u32; values.len()];
    stream
        .memcpy_dtoh(&dec.output().slice(0..values.len()), &mut got)
        .unwrap();
    assert_eq!(got, expect);
}

#[test]
fn elias_fano_device_matches_to_vec() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let values = [0u64, 1, 1, 4, 7, 10, 25];
    let ef = EliasFano::encode(&values);
    let expect = ef.to_vec();
    let parts = EliasFanoDeviceParts::upload(&stream, &ef).unwrap();
    let mut dec = EliasFanoDecoder::new(&ctx, &stream, values.len()).unwrap();
    dec.decode_all(&parts).unwrap();
    stream.synchronize().unwrap();
    let mut got = vec![0u64; values.len()];
    stream
        .memcpy_dtoh(&dec.output().slice(0..values.len()), &mut got)
        .unwrap();
    assert_eq!(got, expect);
}
