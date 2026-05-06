//! TEST_PLAN T1.2 — Seqlock FeatureTable stress test.
//!
//! 4 writer threads + 8 reader threads, each writer owning a disjoint range
//! of node IDs. Each write embeds a generation counter + a per-feature-index
//! checksum so a torn read (head/tail mismatch evading detection) shows up as
//! an invariant violation in the reader. Runs for `TARGET_ITERS` reader
//! iterations OR `MAX_RUNTIME` wall-clock, whichever comes first.
//!
//! This validates the head/tail seqlock protocol under contention and the
//! `Acquire` fence ordering on weak-memory architectures (aarch64). On x86,
//! TSO makes the fence a no-op — running there still exercises the protocol
//! state machine but provides weaker evidence; run on Graviton (aarch64) for
//! the strong memory-ordering check the comment in `feature_table.rs:118-123`
//! describes.

#![cfg(target_os = "linux")]

use aether_stream::feature_table::FeatureTable;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const NODE_COUNT: usize = 4096;
const FEATURE_DIM: usize = 16;
const NUM_WRITERS: usize = 4;
const NUM_READERS: usize = 8;
/// 5M reads is enough to catch a regression in the seqlock fence ordering —
/// the torn-read rate on a broken protocol is ~2 per 10M reads (empirically
/// observed on x86 before the Acquire-fence fix), so 5M yields ~1 per run
/// on a bad build. Paired with a 10s wall-clock cap, the test fits in a
/// default `cargo test` run without being the dominant cost.
const TARGET_ITERS: u64 = 5_000_000;
const MAX_RUNTIME: Duration = Duration::from_secs(10);

/// Encode (node_id, generation) into the feature vector so readers can
/// verify consistency without coordination. features[0] = generation;
/// features[i] = generation + (i as u32) * (node_id + 1) for i in 1..DIM.
#[inline]
fn encode(node: u32, gen: u32) -> [f32; FEATURE_DIM] {
    let mut out = [0f32; FEATURE_DIM];
    out[0] = gen as f32;
    for i in 1..FEATURE_DIM {
        let v = (gen as u64).wrapping_add(i as u64 * (node as u64 + 1));
        out[i] = (v & 0x00FF_FFFF) as f32; // keep within f32 exact-integer range
    }
    out
}

/// Verify a successfully-read slot is internally consistent for `node`.
/// Returns Some(generation) on success, None if the data is torn / mixed.
#[inline]
fn verify(node: u32, features: &[f32]) -> Option<u32> {
    let gen = features[0] as u64 as u32;
    for i in 1..FEATURE_DIM {
        let expected_v = (gen as u64).wrapping_add(i as u64 * (node as u64 + 1)) & 0x00FF_FFFF;
        let got_v = features[i] as u64;
        if got_v != expected_v {
            return None;
        }
    }
    Some(gen)
}

#[test]
fn stress_seqlock_concurrent_writers_readers() {
    // SAFETY/INVARIANT: each writer owns nodes [writer_id * SHARD .. (writer_id+1) * SHARD).
    // Disjoint ownership means no two writers ever touch the same node — exactly
    // the "writes to different nodes are concurrent-safe" guarantee the FeatureTable docs.
    const SHARD: usize = NODE_COUNT / NUM_WRITERS;
    assert_eq!(NODE_COUNT % NUM_WRITERS, 0);

    let table = Arc::new(FeatureTable::new(NODE_COUNT, FEATURE_DIM, vec![]).expect("alloc"));
    let stop = Arc::new(AtomicBool::new(false));
    let total_reads = Arc::new(AtomicU64::new(0));
    let total_writes = Arc::new(AtomicU64::new(0));
    let torn_observed = Arc::new(AtomicU64::new(0));

    let start = Instant::now();
    let mut workers = Vec::new();

    // ----- Writers --------------------------------------------------------
    for w in 0..NUM_WRITERS {
        let table = Arc::clone(&table);
        let stop = Arc::clone(&stop);
        let total_writes = Arc::clone(&total_writes);
        workers.push(thread::spawn(move || {
            let lo = w * SHARD;
            let hi = lo + SHARD;
            let mut gen: u32 = 1;
            let mut writes_local: u64 = 0;
            while !stop.load(Ordering::Relaxed) {
                for node in lo..hi {
                    let feats = encode(node as u32, gen);
                    table.write_node(node, &feats);
                    writes_local += 1;
                }
                gen = gen.wrapping_add(1);
                if writes_local % 100_000 == 0 {
                    total_writes.fetch_add(writes_local, Ordering::Relaxed);
                    writes_local = 0;
                }
            }
            total_writes.fetch_add(writes_local, Ordering::Relaxed);
        }));
    }

    // ----- Readers --------------------------------------------------------
    for r in 0..NUM_READERS {
        let table = Arc::clone(&table);
        let stop = Arc::clone(&stop);
        let total_reads = Arc::clone(&total_reads);
        let torn_observed = Arc::clone(&torn_observed);
        workers.push(thread::spawn(move || {
            // Per-thread xorshift PRNG — deterministic per reader for repro.
            let mut rng: u64 = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul((r as u64) + 1);
            let mut buf = vec![0f32; FEATURE_DIM];
            let mut reads_local: u64 = 0;
            loop {
                if reads_local % 8192 == 0 {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let cur = total_reads.fetch_add(reads_local, Ordering::Relaxed) + reads_local;
                    reads_local = 0;
                    if cur >= TARGET_ITERS {
                        stop.store(true, Ordering::Relaxed);
                        break;
                    }
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let node = (rng as usize) % NODE_COUNT;
                if table.read_node(node, &mut buf) {
                    if verify(node as u32, &buf).is_none() {
                        // CRITICAL: torn read passed read_node()'s head==tail check
                        // but the feature payload is internally inconsistent.
                        torn_observed.fetch_add(1, Ordering::Relaxed);
                    }
                }
                reads_local += 1;
            }
        }));
    }

    // Watchdog: enforce wall-clock cap.
    let watchdog_stop = Arc::clone(&stop);
    let watchdog = thread::spawn(move || {
        let deadline = Instant::now() + MAX_RUNTIME;
        while Instant::now() < deadline {
            if watchdog_stop.load(Ordering::Relaxed) {
                return;
            }
            thread::sleep(Duration::from_millis(50));
        }
        watchdog_stop.store(true, Ordering::Relaxed);
    });

    for w in workers {
        w.join().expect("worker panicked");
    }
    watchdog.join().ok();

    let elapsed = start.elapsed();
    let writes = total_writes.load(Ordering::Relaxed);
    let reads = total_reads.load(Ordering::Relaxed);
    let torn = torn_observed.load(Ordering::Relaxed);

    eprintln!(
        "stress_seqlock: {writes} writes, {reads} reads in {:?} ({:.0} reads/s, {:.0} writes/s); torn observed: {}",
        elapsed,
        reads as f64 / elapsed.as_secs_f64(),
        writes as f64 / elapsed.as_secs_f64(),
        torn,
    );

    // The hard invariant: zero torn reads ever pass verification.
    // A single failure here means the seqlock protocol is broken on this
    // architecture — most likely the Acquire fence in write_node / read_node
    // is missing or mis-ordered.
    assert_eq!(
        torn, 0,
        "{torn} torn reads passed read_node()'s head==tail gate — seqlock fence ordering is broken on {}",
        std::env::consts::ARCH
    );

    // Sanity: the workload actually ran. Don't pass the assertion above by
    // virtue of having done nothing.
    assert!(reads >= 1_000_000, "ran only {reads} reads — workload didn't get going");
    assert!(writes >= 100_000, "ran only {writes} writes — writer threads didn't make progress");
}
