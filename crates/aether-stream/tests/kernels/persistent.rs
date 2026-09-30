use aether_stream::gpu::kernels::harness::cuda_or_skip;
use aether_stream::gpu::kernels::{PersistentWork, PersistentWorkKind, PersistentWorker};
use std::time::{Duration, Instant};

#[test]
fn persistent_drain_counts_posted_work() {
    let Some((ctx, _stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let mut worker = PersistentWorker::new(&ctx, 64).unwrap();
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

/// Posting far past the ring's capacity only works if the kernel frees
/// slots as it claims them and the host sees it.
#[test]
fn persistent_drain_wraps_the_ring() {
    let Some((ctx, _stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let mut worker = PersistentWorker::new(&ctx, 8).unwrap();
    worker.start().unwrap();
    let n = 1000u64;
    let deadline = Instant::now() + Duration::from_secs(30);
    for i in 0..n {
        while !worker
            .post(PersistentWork::new(PersistentWorkKind::Complete, i, 1))
            .unwrap()
        {
            assert!(Instant::now() < deadline, "ring never drained");
            std::hint::spin_loop();
        }
    }
    assert_eq!(worker.stop_and_join().unwrap(), n);
}

/// Dropping a running worker must stop and join the kernel rather than
/// free the ring under it.
#[test]
fn dropping_a_running_worker_joins_the_kernel() {
    let Some((ctx, stream)) = cuda_or_skip() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    {
        let mut worker = PersistentWorker::new(&ctx, 16).unwrap();
        worker.start().unwrap();
        for i in 0..4u64 {
            assert!(
                worker
                    .post(PersistentWork::new(PersistentWorkKind::Validate, i, 1))
                    .unwrap()
            );
        }
    }
    ctx.synchronize().unwrap();
    stream.synchronize().unwrap();
}
