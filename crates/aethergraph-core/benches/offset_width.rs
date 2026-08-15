//! Does narrowing the CSR offsets array from u64 to u32 buy sampling speed?
//!
//! `EdgeOffset` is u64, so the offsets array costs `8 * (num_nodes + 1)`
//! bytes. Any graph with fewer than 4 billion edges could store it in u32
//! and halve that. Halving the array is unambiguously a memory-capacity
//! win; whether it is also a *speed* win is the open question, because a
//! frontier node reads `offsets[n]` and `offsets[n+1]` — adjacent words
//! that almost always share one cache line at either width.
//!
//! This measures the two paths over identical edge data: the range lookup
//! alone, and the lookup plus the neighbour scan it gates. It deliberately
//! does not go through `Graph`, so no library change is needed to answer
//! the question.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;

const NUM_NODES: usize = 8_000_000;
const AVG_DEGREE: usize = 8;
const BATCH: usize = 4096;

struct Csr {
    offsets64: Vec<u64>,
    offsets32: Vec<u32>,
    edges: Vec<u32>,
}

fn build() -> Csr {
    let mut rng = StdRng::seed_from_u64(0x0FF5);
    let mut offsets64 = Vec::with_capacity(NUM_NODES + 1);
    let mut total = 0u64;
    offsets64.push(0);
    for _ in 0..NUM_NODES {
        total += rng.random_range(1..=(AVG_DEGREE * 2)) as u64;
        offsets64.push(total);
    }
    let edges: Vec<u32> = (0..total)
        .map(|_| rng.random_range(0..NUM_NODES as u32))
        .collect();
    let offsets32: Vec<u32> = offsets64.iter().map(|&o| o as u32).collect();
    Csr {
        offsets64,
        offsets32,
        edges,
    }
}

fn frontier() -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(0xB33F);
    (0..BATCH)
        .map(|_| rng.random_range(0..NUM_NODES as u32))
        .collect()
}

fn bench_offset_width(c: &mut Criterion) {
    let csr = build();
    let nodes = frontier();

    let mut group = c.benchmark_group("offset_width");

    // Range lookup only: the access the offsets array actually serves.
    group.bench_with_input(BenchmarkId::new("range_only", "u64"), &nodes, |b, nodes| {
        b.iter(|| {
            let mut acc = 0u64;
            for &n in nodes {
                let i = n as usize;
                acc += csr.offsets64[i + 1] - csr.offsets64[i];
            }
            black_box(acc)
        })
    });
    group.bench_with_input(BenchmarkId::new("range_only", "u32"), &nodes, |b, nodes| {
        b.iter(|| {
            let mut acc = 0u64;
            for &n in nodes {
                let i = n as usize;
                acc += (csr.offsets32[i + 1] - csr.offsets32[i]) as u64;
            }
            black_box(acc)
        })
    });

    // Lookup plus the neighbour scan it gates — the shape of real work,
    // where the edges array is identical at both widths.
    group.bench_with_input(BenchmarkId::new("with_scan", "u64"), &nodes, |b, nodes| {
        b.iter(|| {
            let mut acc = 0u64;
            for &n in nodes {
                let i = n as usize;
                let (s, e) = (csr.offsets64[i] as usize, csr.offsets64[i + 1] as usize);
                for &nb in &csr.edges[s..e] {
                    acc += nb as u64;
                }
            }
            black_box(acc)
        })
    });
    group.bench_with_input(BenchmarkId::new("with_scan", "u32"), &nodes, |b, nodes| {
        b.iter(|| {
            let mut acc = 0u64;
            for &n in nodes {
                let i = n as usize;
                let (s, e) = (csr.offsets32[i] as usize, csr.offsets32[i + 1] as usize);
                for &nb in &csr.edges[s..e] {
                    acc += nb as u64;
                }
            }
            black_box(acc)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_offset_width);
criterion_main!(benches);
