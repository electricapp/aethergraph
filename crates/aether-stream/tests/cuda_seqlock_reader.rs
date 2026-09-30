//! K5.3 direct device FeatureTable slot snapshot test.

#![cfg(all(target_os = "linux", feature = "gpudirect"))]

use aether_stream::gpu::seqlock_reader::SeqlockSnapshotReader;
use cudarc::driver::{CudaContext, CudaSlice};

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

    let mut reader = SeqlockSnapshotReader::new(&ctx, &stream, host.len(), FEATURE_DIM, SLOT_SIZE)
        .expect("NVRTC reader compile");
    reader.snapshot(&staging, host.len()).expect("launch");
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

/// A writer republishing one slot under the reader must never get a row
/// accepted whose payload mixes two generations. Generation `g` writes head
/// `2g - 1`, a payload of all `g`, then tail and head `2g` — the table
/// writer's order — as separate copies on one stream.
#[test]
fn snapshot_reader_never_accepts_a_torn_row() {
    use cudarc::driver::DevicePtr;
    use cudarc::driver::result::memcpy_htod_async;
    use std::sync::Arc;

    const DIM: usize = 1024;
    const STRIDE: usize = aethergraph_core::feature_slot_stride(DIM);
    const TAIL: usize = aethergraph_core::feature_slot_tail_offset(DIM);
    const GENERATIONS: u64 = 20_000;

    let ctx = CudaContext::new(0).expect("CUDA init");
    let reader_stream = ctx.new_stream().expect("reader stream");
    let writer_stream = ctx.new_stream().expect("writer stream");
    // The race is the point; per-slice events would serialize the streams.
    // SAFETY: every slice below is used on `reader_stream` only, the
    // writer reaches the slot through a raw pointer, and the writer thread
    // is joined (its stream synchronized) before the slot drops.
    unsafe { ctx.disable_event_tracking() };

    let mut slot: CudaSlice<u8> = reader_stream.alloc_zeros(STRIDE).expect("VRAM");
    let mut first = vec![0u8; STRIDE];
    first[..8].copy_from_slice(&2u64.to_le_bytes());
    for lane in 0..DIM {
        first[FEAT + lane * 4..FEAT + 4 + lane * 4].copy_from_slice(&1f32.to_le_bytes());
    }
    first[TAIL..TAIL + 8].copy_from_slice(&2u64.to_le_bytes());
    reader_stream.memcpy_htod(&first, &mut slot).expect("H2D");
    reader_stream.synchronize().expect("publish generation 1");

    let base = {
        let (ptr, _record) = slot.device_ptr(&reader_stream);
        ptr
    };
    let writer = {
        let stream = Arc::clone(&writer_stream);
        std::thread::spawn(move || {
            stream.context().bind_to_thread().expect("bind");
            let cu = stream.cu_stream();
            for generation in 2..=GENERATIONS {
                let odd = [2 * generation - 1];
                let payload = vec![generation as f32; DIM];
                let even = [2 * generation];
                // SAFETY: this and the next three copies land inside the
                // live slot allocation; pageable sources are staged before
                // the call returns.
                unsafe { memcpy_htod_async(base, &odd, cu) }.expect("head odd");
                // SAFETY: as above.
                unsafe { memcpy_htod_async(base + FEAT as u64, &payload, cu) }.expect("payload");
                // SAFETY: as above.
                unsafe { memcpy_htod_async(base + TAIL as u64, &even, cu) }.expect("tail");
                // SAFETY: as above.
                unsafe { memcpy_htod_async(base, &even, cu) }.expect("head even");
            }
            stream.synchronize().expect("writer drain");
        })
    };

    let mut reader =
        SeqlockSnapshotReader::new(&ctx, &reader_stream, 1, DIM, STRIDE).expect("NVRTC");
    let mut accepted = 0u64;
    let mut output = vec![0f32; DIM];
    while !writer.is_finished() {
        reader.snapshot(&slot, 1).expect("launch");
        let mut valid = [0_i32];
        reader_stream
            .memcpy_dtoh(reader.valid_mask(), &mut valid)
            .expect("mask D2H");
        reader_stream
            .memcpy_dtoh(reader.output(), &mut output)
            .expect("output D2H");
        reader_stream.synchronize().expect("snapshot");
        if valid[0] == 1 {
            accepted += 1;
            assert!(
                output.iter().all(|&v| v == output[0]),
                "accepted a torn row: first {} vs {:?}",
                output[0],
                output.iter().find(|&&v| v != output[0])
            );
        }
    }
    writer.join().expect("writer");
    assert!(accepted > 0, "the reader never saw a stable row");
}
