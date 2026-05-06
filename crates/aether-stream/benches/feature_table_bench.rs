//! FeatureTable seqlock hot-path benchmarks.
//!
//! Measures write throughput, read throughput, and mixed read/write
//! contention for the head/tail seqlock protocol.
//!
//! Linux only (aether-stream is cfg(target_os = "linux")).
//!
//! ```
//! cargo bench -p aether-stream --bench feature_table_bench
//! BENCH_SCALE=large cargo bench -p aether-stream --bench feature_table_bench
//! ```

#![cfg(target_os = "linux")]

use aether_stream::feature_table::FeatureTable;
use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::hint::black_box;

fn is_large() -> bool {
    std::env::var("BENCH_SCALE").ok().as_deref() == Some("large")
}

fn node_count() -> usize {
    if is_large() { 1_000_000 } else { 100_000 }
}

fn bench_write_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_table_write");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));

    for dim in [128, 768] {
        let table = FeatureTable::new(node_count(), dim, vec![]).unwrap();
        let features: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &features,
            |b, features| {
                let mut node = 0u32;
                b.iter(|| {
                    table.write_node(node as usize % node_count(), features);
                    node = node.wrapping_add(1);
                    black_box(());
                });
            },
        );
    }
    group.finish();
}

fn bench_read_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_table_read");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));

    for dim in [128, 768] {
        let table = FeatureTable::new(node_count(), dim, vec![]).unwrap();
        let features: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        // Pre-write so reads succeed
        for i in 0..1000.min(node_count()) {
            table.write_node(i, &features);
        }
        let mut out = vec![0.0f32; dim];

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                let mut node = 0usize;
                b.iter(|| {
                    black_box(table.read_node(node % 1000, &mut out));
                    node += 1;
                });
            },
        );
    }
    group.finish();
}

fn bench_write_read_contention(c: &mut Criterion) {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let dim = 128;
    let table = Arc::new(FeatureTable::new(node_count(), dim, vec![]).unwrap());
    let features: Vec<f32> = (0..dim).map(|i| i as f32).collect();

    // Pre-write
    for i in 0..1000.min(node_count()) {
        table.write_node(i, &features);
    }

    let mut group = c.benchmark_group("feature_table_contention");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    // Benchmark reads while a background writer is hammering the table
    group.bench_function("read_under_write_pressure", |b| {
        let running = Arc::new(AtomicBool::new(true));
        let table_w = table.clone();
        let running_w = running.clone();
        let features_w = features.clone();

        let writer = thread::spawn(move || {
            let mut i = 0usize;
            while running_w.load(Ordering::Relaxed) {
                table_w.write_node(i % 1000, &features_w);
                i += 1;
            }
        });

        let mut out = vec![0.0f32; dim];
        let mut node = 0usize;
        b.iter(|| {
            black_box(table.read_node(node % 1000, &mut out));
            node += 1;
        });

        running.store(false, Ordering::Relaxed);
        writer.join().unwrap();
    });

    group.finish();
}

fn bench_batch_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_table_batch_write");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(3));

    let dim = 768;
    let table = FeatureTable::new(node_count(), dim, vec![]).unwrap();

    for batch in [64, 256, 1024] {
        let features: Vec<Vec<f32>> = (0..batch)
            .map(|i| (0..dim).map(|j| (i * dim + j) as f32 * 0.001).collect())
            .collect();

        group.throughput(Throughput::Elements(batch as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch),
            &features,
            |b, features| {
                b.iter(|| {
                    for (i, f) in features.iter().enumerate() {
                        table.write_node(i, f);
                    }
                    black_box(());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_write_throughput,
    bench_read_throughput,
    bench_write_read_contention,
    bench_batch_write,
);
criterion_main!(benches);
