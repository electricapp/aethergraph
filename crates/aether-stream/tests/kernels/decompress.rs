use aether_stream::gpu::kernels::harness::cuda_or_skip;
use aether_stream::gpu::kernels::{EliasFanoDecoder, EliasFanoDeviceParts, StreamVByteDecoder};
use aethergraph_core::{EliasFano, StreamVByte};

#[test]
fn streamvbyte_device_matches_cpu() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let values = [10u32, 11, 311, 313];
    let svb = StreamVByte::encode_deltas(&values);
    let expect = svb.decode();

    let mut d_ctrl = unsafe { stream.alloc::<u8>(svb.control().len().max(1)).unwrap() };
    let mut d_data = unsafe { stream.alloc::<u8>(svb.data().len().max(1)).unwrap() };
    if !svb.control().is_empty() {
        stream.memcpy_htod(svb.control(), &mut d_ctrl).unwrap();
    }
    if !svb.data().is_empty() {
        stream.memcpy_htod(svb.data(), &mut d_data).unwrap();
    }
    let mut dec = StreamVByteDecoder::new(&ctx, &stream, values.len()).unwrap();
    dec.decode(&d_ctrl, &d_data, values.len(), svb.first())
        .unwrap();
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
