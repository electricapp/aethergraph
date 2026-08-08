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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
fn encode(node: u32, gen_id: u32) -> [f32; FEATURE_DIM] {
    let mut out = [0f32; FEATURE_DIM];
    out[0] = gen_id as f32;
    for (i, slot) in out.iter_mut().enumerate().skip(1) {
        let v = (gen_id as u64).wrapping_add(i as u64 * (node as u64 + 1));
        *slot = (v & 0x00FF_FFFF) as f32; // keep within f32 exact-integer range
    }
    out
}

/// Verify a successfully-read slot is internally consistent for `node`.
/// Returns Some(generation) on success, None if the data is torn / mixed.
#[inline]
fn verify(node: u32, features: &[f32]) -> Option<u32> {
    let gen_id = features[0] as u64 as u32;
    for (i, &f) in features.iter().enumerate().take(FEATURE_DIM).skip(1) {
        let expected_v = (gen_id as u64).wrapping_add(i as u64 * (node as u64 + 1)) & 0x00FF_FFFF;
        let got_v = f as u64;
        if got_v != expected_v {
            return None;
        }
    }
    Some(gen_id)
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
            let mut gen_id: u32 = 1;
            let mut writes_local: u64 = 0;
            while !stop.load(Ordering::Relaxed) {
                for node in lo..hi {
                    let feats = encode(node as u32, gen_id);
                    table.write_node(node, &feats);
                    writes_local += 1;
                }
                gen_id = gen_id.wrapping_add(1);
                if writes_local.is_multiple_of(100_000) {
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
                if reads_local.is_multiple_of(8192) && stop.load(Ordering::Relaxed) {
                    break;
                }
                if reads_local.is_multiple_of(8192) {
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
                if table.read_node(node, &mut buf) && verify(node as u32, &buf).is_none() {
                    // CRITICAL: torn read passed read_node()'s head==tail check
                    // but the feature payload is internally inconsistent.
                    torn_observed.fetch_add(1, Ordering::Relaxed);
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
        torn,
        0,
        "{torn} torn reads passed read_node()'s head==tail gate — seqlock fence ordering is broken on {}",
        std::env::consts::ARCH
    );

    // Sanity: the workload actually ran. Don't pass the assertion above by
    // virtue of having done nothing.
    assert!(
        reads >= 1_000_000,
        "ran only {reads} reads — workload didn't get going"
    );
    assert!(
        writes >= 100_000,
        "ran only {writes} writes — writer threads didn't make progress"
    );
}

/// Multiple writers hammering the SAME node concurrently. `write_node`
/// serializes same-node writers on the head CAS, so every generation a
/// reader observes must be one complete write — all bytes from a single
/// `write_node` call, never a mix of two writers' payloads.
///
/// Encoding: `features[i] = base + i`, with a distinct `base` per
/// (writer, iteration). `features[i] - i` is constant across `i` within one
/// complete write and differs between any two writes, so a mixed row is
/// detected without coordination.
#[test]
fn stress_seqlock_same_node_concurrent_writers() {
    const SAME_NODE_WRITERS: usize = 4;
    const SAME_NODE_READERS: usize = 4;
    const DIM: usize = 16;
    const RUNTIME: Duration = Duration::from_secs(3);
    const NODE: usize = 0;

    let table = Arc::new(FeatureTable::new(4, DIM, vec![]).expect("alloc"));
    let stop = Arc::new(AtomicBool::new(false));
    let total_writes = Arc::new(AtomicU64::new(0));
    let total_reads = Arc::new(AtomicU64::new(0));
    let mixed_rows = Arc::new(AtomicU64::new(0));

    let mut workers = Vec::new();
    for w in 0..SAME_NODE_WRITERS {
        let table = Arc::clone(&table);
        let stop = Arc::clone(&stop);
        let total_writes = Arc::clone(&total_writes);
        workers.push(thread::spawn(move || {
            let mut iter: u64 = 0;
            let mut writes_local: u64 = 0;
            while !stop.load(Ordering::Relaxed) {
                // Distinct base per (writer, iteration), kept within f32's
                // exact-integer range so `base + i` round-trips exactly.
                let base = ((iter * SAME_NODE_WRITERS as u64 + w as u64) & 0x007F_FFFF) as f32;
                let mut feats = [0f32; DIM];
                for (i, slot) in feats.iter_mut().enumerate() {
                    *slot = base + i as f32;
                }
                table.write_node(NODE, &feats);
                iter += 1;
                writes_local += 1;
            }
            total_writes.fetch_add(writes_local, Ordering::Relaxed);
        }));
    }

    for _ in 0..SAME_NODE_READERS {
        let table = Arc::clone(&table);
        let stop = Arc::clone(&stop);
        let total_reads = Arc::clone(&total_reads);
        let mixed_rows = Arc::clone(&mixed_rows);
        workers.push(thread::spawn(move || {
            let mut buf = vec![0f32; DIM];
            let mut reads_local: u64 = 0;
            while !stop.load(Ordering::Relaxed) {
                if table.read_node(NODE, &mut buf) {
                    let base = buf[0];
                    if buf.iter().enumerate().any(|(i, &v)| v - i as f32 != base) {
                        mixed_rows.fetch_add(1, Ordering::Relaxed);
                    }
                }
                reads_local += 1;
            }
            total_reads.fetch_add(reads_local, Ordering::Relaxed);
        }));
    }

    thread::sleep(RUNTIME);
    stop.store(true, Ordering::Relaxed);
    for wkr in workers {
        wkr.join().expect("worker panicked");
    }

    let writes = total_writes.load(Ordering::Relaxed);
    let reads = total_reads.load(Ordering::Relaxed);
    let mixed = mixed_rows.load(Ordering::Relaxed);
    eprintln!("same-node stress: {writes} writes, {reads} reads, {mixed} mixed rows");

    assert_eq!(
        mixed,
        0,
        "{mixed} rows mixed bytes from different writes — same-node writer \
         serialization is broken on {}",
        std::env::consts::ARCH
    );
    // Sanity: real contention happened.
    assert!(
        writes >= 10_000,
        "ran only {writes} writes — writer threads didn't make progress"
    );
    assert!(
        reads >= 10_000,
        "ran only {reads} reads — reader threads didn't make progress"
    );
}
