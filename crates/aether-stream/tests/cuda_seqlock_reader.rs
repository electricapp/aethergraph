//! K5.3 direct device FeatureTable slot snapshot test.

#![cfg(all(target_os = "linux", feature = "gpudirect"))]

use aether_stream::gpu::seqlock_reader::SeqlockSnapshotReader;
use cudarc::driver::{CudaContext, CudaSlice, DevicePtrMut};

const FEATURE_DIM: usize = 3;
const FEAT: usize = aethergraph_core::FEATURE_SLOT_HEAD_BYTES;
const TAIL_OFFSET: usize = aethergraph_core::feature_slot_tail_offset(FEATURE_DIM);
const SLOT_SIZE: usize = aethergraph_core::feature_slot_stride(FEATURE_DIM);

fn slot(head: u64, features: [f32; FEATURE_DIM], tail: u64) -> [u8; SLOT_SIZE] {
    let mut bytes = [0; SLOT_SIZE];
    bytes[..8].copy_from_slice(&head.to_le_bytes());
    for (index, value) in features.into_iter().enumerate() {
        bytes[FEAT + index * 4..FEAT + 4 + index * 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes[TAIL_OFFSET..TAIL_OFFSET + 8].copy_from_slice(&tail.to_le_bytes());
    bytes
}

#[test]
fn snapshot_reader_marks_only_even_stable_slots_valid() {
    let ctx = CudaContext::new(0).expect("CUDA init");
    let stream = ctx.default_stream();
    let host = [slot(2, [1., 2., 3.], 2), slot(3, [4., 5., 6.], 3)];
    let mut staging: CudaSlice<u8> = stream.alloc_zeros(SLOT_SIZE * host.len()).expect("VRAM");
    stream
        .memcpy_htod(&host.concat(), &mut staging)
        .expect("H2D");
    let slots_ptr = {
        let (ptr, _guard) = staging.device_ptr_mut(&stream);
        ptr
    };

    let mut reader = SeqlockSnapshotReader::new(&ctx, &stream, host.len(), FEATURE_DIM)
        .expect("NVRTC reader compile");
    reader
        .snapshot(slots_ptr, SLOT_SIZE, host.len())
        .expect("launch");
    stream.synchronize().expect("snapshot completion");

    let mut valid = [0_i32; 2];
    stream
        .memcpy_dtoh(reader.valid_mask(), &mut valid)
        .expect("mask D2H");
    assert_eq!(valid, [1, 0]);
    let mut output = [0_f32; FEATURE_DIM * 2];
    stream
        .memcpy_dtoh(reader.output(), &mut output)
        .expect("output D2H");
    assert_eq!(&output[..FEATURE_DIM], &[1., 2., 3.]);
}
