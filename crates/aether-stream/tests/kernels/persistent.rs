use aether_stream::gpu::kernels::harness::cuda_or_skip;
use aether_stream::gpu::kernels::{PersistentWork, PersistentWorkKind, PersistentWorker};

#[test]
fn persistent_drain_counts_posted_work() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let mut worker = PersistentWorker::new(&ctx, &stream, 64).unwrap();
    worker.start().unwrap();
    let n = 16u64;
    for i in 0..n {
        assert!(
            worker
                .post(PersistentWork::new(PersistentWorkKind::Gather, i, 1))
                .unwrap()
        );
    }
    let completed = worker.stop_and_join().unwrap();
    assert_eq!(completed, n);
}
