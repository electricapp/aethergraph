use aether_stream::gpu::kernels::SeqlockValidator;
use aether_stream::gpu::kernels::harness::cuda_or_skip;
use cudarc::driver::DevicePtrMut;

const FEATURE_DIM: usize = 4;
const SLOT_SIZE: usize = 8 + FEATURE_DIM * 4 + 8;
const BATCH: usize = 4;

fn pack_slot(head: u64, features: [f32; FEATURE_DIM], tail: u64) -> [u8; SLOT_SIZE] {
    let mut out = [0u8; SLOT_SIZE];
    out[0..8].copy_from_slice(&head.to_le_bytes());
    for (i, f) in features.iter().enumerate() {
        out[8 + i * 4..8 + (i + 1) * 4].copy_from_slice(&f.to_le_bytes());
    }
    out[SLOT_SIZE - 8..SLOT_SIZE].copy_from_slice(&tail.to_le_bytes());
    out
}

#[test]
fn validate_graph_replay_skips_without_cuda() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let snap = pack_slot(2, [1.0, 2.0, 3.0, 4.0], 2);
    let mut host = Vec::with_capacity(SLOT_SIZE * BATCH * 2);
    for _ in 0..BATCH * 2 {
        host.extend_from_slice(&snap);
    }
    let mut staging = stream
        .alloc_zeros::<u8>(SLOT_SIZE * BATCH * 2)
        .expect("alloc");
    stream.memcpy_htod(&host, &mut staging).expect("H2D");
    let staging1 = {
        let (p, _) = staging.device_ptr_mut(&stream);
        p
    };
    let staging2 = staging1 + (SLOT_SIZE * BATCH) as u64;
    let mut v = SeqlockValidator::new(&ctx, &stream, BATCH, FEATURE_DIM).expect("nvrtc");
    assert_eq!(v.validate(staging1, staging2, SLOT_SIZE, BATCH).unwrap(), 0);
    assert!(v.has_captured_graph());
    assert_eq!(v.validate(staging1, staging2, SLOT_SIZE, BATCH).unwrap(), 0);
}
