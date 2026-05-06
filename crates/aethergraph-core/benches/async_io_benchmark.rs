//! I/O benchmarks for CSR graph neighbor reads.
//!
//! Four read paths, same graph file, same node set:
//!   - mmap_lookup_1k       : mmap slice access (no I/O; memory baseline)
//!   - sync_pread_1k        : pread(2) in a loop, single thread
//!   - sync_pread_rayon_1k  : pread(2) fanned across rayon workers
//!   - async_iouring_1k     : AsyncCsrGraph::batch_neighbors (io_uring + SQPOLL)
//!
//! Environment knobs:
//!   AETHER_BENCH_GRAPH=/path/to/graph.bin   Use a pre-generated graph. If the
//!                                           file does not exist, a ~10GB graph
//!                                           is generated there (~2 min on NVMe).
//!   AETHER_BENCH_COLD=1                     posix_fadvise(DONTNEED) before each
//!                                           iteration to evict edge bytes from
//!                                           the page cache, measuring disk I/O
//!                                           rather than cached reads.

use aethergraph_core::{save_graph, AsyncCsrGraph, Graph, NodeId};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::fs::File;
use std::hint::black_box;
use std::os::unix::fs::FileExt;
#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::NamedTempFile;

const HEADER_SIZE: u64 = 32;
const OFFSET_BYTES: usize = 8; // EdgeOffset = u64
const NODE_ID_BYTES: usize = std::mem::size_of::<NodeId>();

/// Parsed graph file metadata for direct-pread benches.
struct GraphFileView {
    _keepalive: Option<NamedTempFile>,
    path: PathBuf,
    file: Arc<File>,
    num_nodes: usize,
    edges_start: u64,
    offsets: Arc<[u64]>,
}

impl GraphFileView {
    fn open(path: PathBuf, keepalive: Option<NamedTempFile>) -> Self {
        let file = File::open(&path).expect("open graph");
        let mut header = [0u8; HEADER_SIZE as usize];
        file.read_exact_at(&mut header, 0).expect("read header");
        let num_nodes = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;

        let offsets_size = (num_nodes + 1) * OFFSET_BYTES;
        let mut offsets_bytes = vec![0u8; offsets_size];
        file.read_exact_at(&mut offsets_bytes, HEADER_SIZE)
            .expect("read offsets");
        let offsets: Vec<u64> = offsets_bytes
            .chunks_exact(OFFSET_BYTES)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let edges_start = HEADER_SIZE + offsets_size as u64;

        Self {
            _keepalive: keepalive,
            path,
            file: Arc::new(file),
            num_nodes,
            edges_start,
            offsets: Arc::from(offsets.into_boxed_slice()),
        }
    }

    /// Byte offset + length for a node's neighbors in the file.
    fn read_extent(&self, node: NodeId) -> Option<(u64, usize)> {
        let idx = node as usize;
        if idx >= self.num_nodes {
            return None;
        }
        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;
        let len = end.saturating_sub(start);
        if len == 0 {
            return None;
        }
        let byte_off = self.edges_start + (start * NODE_ID_BYTES) as u64;
        Some((byte_off, len * NODE_ID_BYTES))
    }

    fn drop_page_cache(&self) {
        #[cfg(target_os = "linux")]
        unsafe {
            libc::posix_fadvise(self.file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
}

/// Tiny in-bench graph for the default (page-cache-friendly) case.
fn generate_small_graph(num_nodes: usize, avg_degree: usize) -> NamedTempFile {
    let mut rng = StdRng::seed_from_u64(42);
    let mut edges = Vec::new();
    for src in 0..num_nodes as NodeId {
        let degree = if rng.random::<f64>() < 0.1 {
            avg_degree * 10
        } else {
            rng.random_range(1..=avg_degree * 2)
        };
        for _ in 0..degree {
            let dst = rng.random_range(0..num_nodes as NodeId);
            if src != dst {
                edges.push((src, dst));
            }
        }
    }
    let graph = Graph::from_edges(num_nodes, &edges, None).unwrap();
    let temp_file = NamedTempFile::new().unwrap();
    save_graph(&graph, temp_file.path()).unwrap();
    temp_file
}

/// Generate a ~10GB graph: ~300M nodes × avg-degree-8 → ~2.4B edges × 4B = 9.6GB.
/// Writes directly to `path`. Takes ~2 min on NVMe.
fn generate_large_graph(path: &std::path::Path, num_nodes: usize, avg_degree: usize) {
    eprintln!(
        "AETHER_BENCH_GRAPH {}: generating {} nodes × ~{} degree...",
        path.display(),
        num_nodes,
        avg_degree
    );
    let mut rng = StdRng::seed_from_u64(42);
    let mut edges = Vec::with_capacity(num_nodes * avg_degree);
    for src in 0..num_nodes as NodeId {
        let degree = rng.random_range(1..=avg_degree * 2);
        for _ in 0..degree {
            let dst = rng.random_range(0..num_nodes as NodeId);
            if src != dst {
                edges.push((src, dst));
            }
        }
    }
    let graph = Graph::from_edges(num_nodes, &edges, None).unwrap();
    save_graph(&graph, path).unwrap();
    eprintln!("generated {} bytes", std::fs::metadata(path).unwrap().len());
}

/// Returns a GraphFileView, preferring $AETHER_BENCH_GRAPH if set.
/// `AETHER_BENCH_NODES` (default 300M) sets generation size — keep it well
/// under available RAM since generation buffers the full edge list.
fn open_bench_graph() -> GraphFileView {
    if let Ok(env_path) = std::env::var("AETHER_BENCH_GRAPH") {
        let path = PathBuf::from(&env_path);
        if !path.exists() {
            let nodes: usize = std::env::var("AETHER_BENCH_NODES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(300_000_000);
            generate_large_graph(&path, nodes, 8);
        }
        GraphFileView::open(path, None)
    } else {
        let tf = generate_small_graph(10_000, 20);
        let path = tf.path().to_path_buf();
        GraphFileView::open(path, Some(tf))
    }
}

fn sample_nodes(view: &GraphFileView, count: usize, seed: u64) -> Vec<NodeId> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| rng.random_range(0..view.num_nodes as NodeId))
        .collect()
}

fn cold_requested() -> bool {
    std::env::var("AETHER_BENCH_COLD").is_ok_and(|v| v == "1")
}

// ---------------------------------------------------------------------------
// mmap baseline: memory-only slice access, no I/O.
// ---------------------------------------------------------------------------

fn bench_mmap_lookup(c: &mut Criterion) {
    let view = open_bench_graph();
    let loaded_graph = aethergraph_core::load_graph(&view.path).unwrap();
    let nodes = sample_nodes(&view, 1000, 42);

    c.bench_function("mmap_lookup_1k_nodes", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(nodes.len());
            for &node in &nodes {
                results.push(loaded_graph.neighbors(node).to_vec());
            }
            black_box(results);
        });
    });
}

// ---------------------------------------------------------------------------
// Sync pread loop: single-threaded I/O baseline.
// ---------------------------------------------------------------------------

fn bench_sync_pread(c: &mut Criterion) {
    let view = open_bench_graph();
    let nodes = sample_nodes(&view, 1000, 42);
    let cold = cold_requested();

    c.bench_function("sync_pread_1k_nodes", |b| {
        b.iter(|| {
            if cold {
                view.drop_page_cache();
            }
            let mut results: Vec<Vec<NodeId>> = Vec::with_capacity(nodes.len());
            for &node in &nodes {
                let Some((off, len)) = view.read_extent(node) else {
                    results.push(Vec::new());
                    continue;
                };
                let mut buf = vec![0u8; len];
                view.file.read_exact_at(&mut buf, off).expect("pread");
                let neigh: Vec<NodeId> = buf
                    .chunks_exact(NODE_ID_BYTES)
                    .map(|c| NodeId::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                results.push(neigh);
            }
            black_box(results);
        });
    });
}

// ---------------------------------------------------------------------------
// Sync pread + rayon: thread-parallel I/O baseline.
// ---------------------------------------------------------------------------

fn bench_sync_pread_rayon(c: &mut Criterion) {
    let view = open_bench_graph();
    let nodes = sample_nodes(&view, 1000, 42);
    let file = Arc::clone(&view.file);
    let cold = cold_requested();

    c.bench_function("sync_pread_rayon_1k_nodes", |b| {
        b.iter(|| {
            if cold {
                view.drop_page_cache();
            }
            let results: Vec<Vec<NodeId>> = nodes
                .par_iter()
                .map(|&node| {
                    let Some((off, len)) = view.read_extent(node) else {
                        return Vec::new();
                    };
                    let mut buf = vec![0u8; len];
                    file.read_exact_at(&mut buf, off).expect("pread");
                    buf.chunks_exact(NODE_ID_BYTES)
                        .map(|c| NodeId::from_le_bytes(c.try_into().unwrap()))
                        .collect()
                })
                .collect();
            black_box(results);
        });
    });
}

// ---------------------------------------------------------------------------
// Async io_uring path.
// ---------------------------------------------------------------------------

fn bench_async_iouring(c: &mut Criterion) {
    let view = open_bench_graph();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let graph = rt.block_on(AsyncCsrGraph::load(&view.path)).unwrap();
    let nodes = sample_nodes(&view, 1000, 42);
    let cold = cold_requested();

    c.bench_function("async_iouring_1k_nodes", |b| {
        b.iter(|| {
            if cold {
                view.drop_page_cache();
            }
            rt.block_on(async {
                let results = graph.batch_neighbors(black_box(&nodes)).await.unwrap();
                black_box(results);
            })
        });
    });
}

// ---------------------------------------------------------------------------
// Batch-size sweep on the io_uring path.
// ---------------------------------------------------------------------------

fn bench_batch_sizes(c: &mut Criterion) {
    let view = open_bench_graph();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let graph = rt.block_on(AsyncCsrGraph::load(&view.path)).unwrap();
    let mut group = c.benchmark_group("batch_size_comparison");

    for &batch_size in &[10, 50, 100, 500] {
        let nodes = sample_nodes(&view, batch_size, 42);
        group.bench_with_input(
            BenchmarkId::new("async", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(graph.batch_neighbors(black_box(&nodes)).await.unwrap())
                    })
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// GNN-style 2-hop sampling workload.
// ---------------------------------------------------------------------------

fn bench_gnn_sampling_workload(c: &mut Criterion) {
    let view = open_bench_graph();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let graph = rt.block_on(AsyncCsrGraph::load(&view.path)).unwrap();

    let mut rng = StdRng::seed_from_u64(42);
    let batches: Vec<Vec<NodeId>> = (0..10)
        .map(|_| {
            (0..64)
                .map(|_| rng.random_range(0..view.num_nodes as NodeId))
                .collect()
        })
        .collect();

    c.bench_function("gnn_sampling_10_batches", |b| {
        b.iter(|| {
            rt.block_on(async {
                for batch in &batches {
                    let hop1 = graph.batch_neighbors(black_box(batch)).await.unwrap();
                    let hop1_flat: Vec<NodeId> = hop1
                        .iter()
                        .flat_map(|neighbors| neighbors.iter().take(10).copied())
                        .collect();
                    let _hop2 = graph.batch_neighbors(black_box(&hop1_flat)).await.unwrap();
                }
            })
        });
    });
}

criterion_group!(
    benches,
    bench_mmap_lookup,
    bench_sync_pread,
    bench_sync_pread_rayon,
    bench_async_iouring,
    bench_batch_sizes,
    bench_gnn_sampling_workload,
);
criterion_main!(benches);
