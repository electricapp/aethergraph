# AetherGraph Architecture

## Public API

```python
from aethergraph import Graph, DynamicGraph, HeteroGraph
from aethergraph.pytorch import NeighborLoader, HeteroNeighborLoader
```

## Crates

| Crate              | Purpose                                                           |
| ------------------ | ----------------------------------------------------------------- |
| `aethergraph-core` | Static CSR graph, homo + hetero sampling, feature store, io_uring |
| `aether-graph`     | Dynamic C-tree graph, lock-free concurrent R/W, arena bump-alloc  |
| `aether-stream`    | AF_XDP ingestion, seqlock features, ibverbs RDMA, GPUDirect       |
| `aether-mem`       | Lock-free slab allocator, HugePages, RDMA/CUDA memory hooks       |
| `aether-epoch`     | `EpochClock` — monotonic version counter shared by subsystems     |
| `aethergraph-py`   | PyO3 Python bindings for all of the above                         |
| `aethergraph-cli`  | CLI tools (convert, info, stats)                                  |

## aethergraph-core — Static Graph Engine

CSR-backed graph with SIMD-accelerated sampling. Memory-mapped from NVMe,
instant startup.

**Sampling design:**

- Generation-tagged direct array (< 1M nodes) or FxHashMap (> 1M nodes) for node
  dedup — O(1) lookup
- Pre-allocated buffers reused across calls via `clear()` — zero hot-path
  allocations
- Local indices assigned during sampling — no post-sort, no binary search for
  edge remapping
- Floyd's O(k) sampling with 256-byte stack bitmap for small degree nodes
- Double-buffered frontiers — `mem::swap` instead of allocating per hop

**Heterogeneous sampling:**

- Multiple CSR matrices (one per edge type) with per-type local node IDs.
- Per-edge-type fanout. Stack-copied edge type IDs in the hot loop to avoid
  borrow conflicts.
- Macro-inlined Floyd's to avoid `&mut self` conflicts with bitmap.

**Cache-locality reordering (Rabbit Order, Arai et al. IPDPS 2016):**

- Phase 1 (parallel over V): each node picks its lowest-degree neighbor as a
  merge candidate.
- Phase 2 (parallel over E): lock-free concurrent union-find with `AtomicU32`
  parent + rank, path-splitting `find`, CAS `union`. Logs `(winner, loser)`
  pairs in merge order.
- Phase 3 (sequential O(V)): replay the merge log into a dendrogram and emit the
  permutation by in-order traversal.
- Community partitions are a free byproduct of the merge log —
  `rabbit_partitions()` runs a sequential UF replay without rebuilding the
  dendrogram. `reorder_rabbit_with_partitions()` returns both in one pass.
- `partition_aligned_batches` uses the partition labels to construct seed
  batches whose neighborhoods overlap, amortizing destination-array reads across
  the batch.
- Measured in `benches/graph_benchmarks.rs::reorder_sampling_speedup` — sampling
  throughput delta on the same seeds before vs. after permutation.

## aether-graph — Dynamic Graph Engine

Lock-free C-tree for live-updating graphs. Single-writer, multi-reader via
functional persistence.

```
Chunk (64 bytes = 1 cache line):
┌────────────────────────────────────────────────┐
│ count: u8 │ pad: 3B │ neighbors: [u32; 15]     │
└────────────────────────────────────────────────┘

C-tree for degree 45 node:
        [Interior]
       /          \
  [Chunk: 15]   [Interior]
               /          \
          [Chunk: 15]   [Chunk: 15]
```

**Design:**

- 90% of nodes (degree < 15) fit in a single chunk — 1 cache line read, same as
  CSR
- Edge insert: path-copy affected nodes, arena bump-alloc (single `fetch_add`),
  atomic root swap
- Readers see a consistent snapshot via functional persistence — old tree valid
  for concurrent readers
- C-tree balance: scapegoat scheme (α = 2/3) — an insert that pushes its path
  past the α-height bound rebuilds the highest weight-unbalanced subtree, so
  depth stays O(log degree) even for monotonically increasing neighbor IDs
- Arena: pre-allocated contiguous region (max 2 GiB — u32 offsets with a tag
  bit), zero-alloc writes; superseded nodes are reclaimed in bulk by
  `DynamicGraph::compact`, which rebuilds live trees into a fresh arena
- 9M inserts/sec, 38M neighbor reads/sec (criterion benchmarks)

## aether-stream — Real-Time Feature Serving

Linux-only kernel-bypass engine. Compiles to empty crate on macOS/Windows.

**Seqlock slot layout (RDMA-safe):**

```
[head_version: u64] [features: f32 × dim] [tail_version: u64]
```

Writers: head odd → write features → tail even → head even. RDMA readers: single
bulk read, GPU kernel validates head == tail.

**ibverbs FFI:** 16 extern C functions, 12 `#[repr(C)]` structs. Non-opaque
`IbvMr` (rkey/lkey fields). `IbvQp` with correct field offsets through `qp_num`.
`IbvSendWrUnion` as a real union with atomic-variant padding.

**GPU pipeline:** `GpuGatherBuffer` (VRAM via cudarc, registered with NIC via
nvidia-peermem) → `SeqlockValidator` (CUDA kernel compiled at runtime via nvrtc
from standalone `.cu` file) → DLPack capsule → `torch.from_dlpack()`.

## Data Flow — Static Graph Training

```
Seed Nodes
    │
    ▼
NeighborLoader (prefetch thread)
    │
    ├── sample_neighbors() ─── CSR (mmap'd from NVMe)
    │                          Floyd's O(k), zero-alloc
    │
    ├── get_batch() ────────── FeatureStore (mmap or io_uring)
    │
    └── PyO3 boundary ──────── numpy via from_vec (zero-copy)
                                │
                                ▼
                          PyG Data object
                          (x, edge_index, n_id, ...)
```

## Data Flow — Dynamic Graph Training

```
Kafka / Event Stream
    │
    ▼
DynamicGraph.insert_edge()
    │ C-tree path copy
    │ Arena bump alloc
    │ Atomic root swap
    │
    ▼
Vertex Array [AtomicU32 roots]
    │
    │  (concurrent, lock-free)
    │
    ▼
Sampler reads C-tree
    │ Iterates sorted chunks
    │ Sees consistent snapshot
    │
    ▼
PyG Data / HeteroData
```

## Data Flow — Live Features (RDMA)

```
Raw Ethernet frames (UDP)
    │
    ▼
┌──────────────────────┐
│  AF_XDP Ingestion    │  ← busy-poll, per-NIC-queue thread, pinned to core
│  (XdpSocket + Umem)  │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│   FeatureTable       │  ← seqlock write (head odd → write → tail even → head even)
│  (HugePage RAM)      │
└──────────────────────┘
    │                         GPU Node
    ├── TCP Control Plane ──► fetch_advertisement() → base_addr, rkey, schema
    │                         connect_with_qp() → QP endpoint exchange
    │                                │
    └── RDMA READ ◄──────────────── RdmaFeatureClient::gather(node_ids)
                                     │
                                     ▼
                              GpuGatherBuffer (VRAM staging, nvidia-peermem)
                                     │
                                     ▼
                              SeqlockValidator (CUDA kernel)
                              if head == tail && even → compact to output
                              else → retry
                                     │
                                     ▼
                              DLPack → torch.from_dlpack()
                                     │
                                     ▼
                              batch.x on CUDA (< 5us total)
```

## Heterogeneous Sampling

```
HeteroGraph:
  node_types: [user(100K), post(500K), comment(1M), subreddit(10K)]
  edge_types:
    (user, votes, post)          → CSR [100K src × 500K dst]
    (user, writes, comment)      → CSR [100K src × 1M dst]
    (comment, reply_to, comment) → CSR [1M src × 1M dst]
    (post, belongs_to, subreddit)→ CSR [500K src × 10K dst]

HeteroNeighborSampler:
  For each hop:
    For each (node_type, local_id) in frontier:
      For each edge_type where src_type == node_type:
        sample fanout neighbors from CSR
        assign local indices during sampling (zero post-processing)
```

## Key Types

| Type                    | Crate         | Purpose                                          |
| ----------------------- | ------------- | ------------------------------------------------ |
| `Graph`                 | core          | Static CSR, mmap'd from NVMe                     |
| `HeteroGraph`           | core          | Multi-relational CSR (one per edge type)         |
| `DynamicGraph`          | aether-graph  | Lock-free C-tree, concurrent R/W                 |
| `NeighborSampler`       | core          | Floyd's O(k) sampling, zero-alloc                |
| `reorder_rabbit`        | core          | Rabbit Order permutation (parallel UF merge)     |
| `rabbit_partitions`     | core          | Community labels (free byproduct of merge log)   |
| `HeteroNeighborSampler` | core          | Typed multi-hop, pre-computed local indices      |
| `NeighborLoader`        | core          | Prefetch thread, io_uring features               |
| `FeatureTable`          | aether-stream | Seqlock feature table for RDMA                   |
| `RdmaFeatureClient`     | aether-stream | GPUDirect RDMA gather API                        |
| `Chunk`                 | aether-graph  | 64-byte sorted neighbor chunk (1 cache line)     |
| `Arena`                 | aether-graph  | Bump allocator for C-tree nodes                  |
| `CTree`                 | aether-graph  | Balanced tree of chunks (functional persistence) |
| `SharedMemoryRing`      | aether-mem    | Lock-free slab with HugePages                    |

## Platform Support

| Platform | Static Graph | Dynamic Graph | Feature I/O       | Streaming     |
| -------- | ------------ | ------------- | ----------------- | ------------- |
| Linux    | mmap         | C-tree        | io_uring + SQPOLL | AF_XDP + RDMA |
| macOS    | mmap         | C-tree        | mmap + madvise    | N/A           |
| Windows  | mmap         | C-tree        | mmap              | N/A           |

## Binary Formats

Every on-disk format starts with a 32-bit magic plus a 32-bit `version` integer.
The version is **monotonically increasing** — old wheels MUST refuse to load a
newer-version file rather than mis-parsing it. Adding backward-compatibility for
an older version requires a deliberate code change + a golden-file regression
test (see `crates/aethergraph-core/tests/golden/`).

### Graph (static CSR, `.bin`)

| Version | Magic                   | Status  | Notes                       |
| ------- | ----------------------- | ------- | --------------------------- |
| **v1**  | `0x41455448` (`"AETH"`) | current | only version emitted today. |

Layout (v1):

```
[Header: 32 bytes — see `aethergraph_core::internal::mmap::Header`]
  magic               u32  =  0x41455448 ("AETH")
  version             u32  =  1
  num_nodes           u64
  num_edges           u64
  has_weights         u32  (0 = absent, 1 = present)
  integrity_checksum  u32  (0 = absent; FNV-1a 32-bit of offsets+edges otherwise;
                            verified only under Full validation — large files
                            default to OffsetsOnly, which skips the body hash)

[Offsets: (num_nodes + 1) × 8 bytes, u64 LE]
[Edges:   num_edges × 4 bytes,        u32 LE]
[Weights: num_edges × 4 bytes,        f32 LE] (only if has_weights = 1)
```

Forward-compat policy:

- A wheel that knows version N MUST reject version N+1 with a clear error
  message ("file format version N+1 is newer than this wheel knows about (max
  N)"), not attempt a best-effort parse.
- Removing a field from a struct = bump major version.
- Adding a trailing optional section = same magic, bump version, MUST read
  header.version before deciding whether to consume the new bytes.

### Hetero graph (multi-relational CSR, `.bin`)

| Version | Magic        | Status  | Notes                             |
| ------- | ------------ | ------- | --------------------------------- |
| **v1**  | `"AETHHETG"` | current | One CSR section per edge type.    |

Layout (v1): 64-byte header (magic, `version` u32, type counts, reserved),
node/edge type tables, then per-edge-type CSR sections. u32 sections (edges,
weights) are zero-padded to 8-byte boundaries so every offsets array stays
u64-aligned. See `aethergraph_core::internal::mmap_hetero` for field-level
detail. The loader bounds-checks every section against the file length with
checked arithmetic — corrupt or truncated files fail at load, not at access.

### Features (`.bin`)

| Version | Magic        | Status  | Notes          |
| ------- | ------------ | ------- | -------------- |
| **v1**  | `"AETHFEAT"` | current | f32 row-major. |

Layout (v1):

```
[Header]
  magic        char[8] = "AETHFEAT"
  num_nodes    u64
  feature_dim  u64
  data_offset  u64
[Data: num_nodes × feature_dim × 4 bytes, f32 LE row-major]
```

### Golden-file regression tests

`crates/aethergraph-core/tests/golden_format.rs` round-trips the v1 graph and
feature formats from committed byte vectors. Any silent layout drift (field
reorder, padding change, magic typo) fails CI on the first affected commit.

To add support for v2 of any format:

1. Bump the `VERSION` constant in the producer.
2. Add the new struct alongside the old one (do NOT mutate the existing one).
3. Branch in the loader on `header.version`.
4. Commit a golden file at the new version and a regression test that loads BOTH
   v1 (existing golden) and v2 (new golden).

## PyG Compatibility

| Feature                 | Homogeneous  | Heterogeneous                     |
| ----------------------- | ------------ | --------------------------------- |
| `data`                  | Graph        | HeteroGraph                       |
| `num_neighbors`         | list[int]    | dict[(src, rel, dst) → list[int]] |
| `input_nodes`           | Tensor       | (node_type, Tensor)               |
| `batch_size`, `shuffle` | yes          | yes                               |
| `pin_memory`            | yes          | yes                               |
| `transform`             | yes          | yes                               |
| `weighted`              | yes          | —                                 |
| `temporal_strategy`     | uniform/last | —                                 |
| `disjoint`              | yes          | —                                 |
| `neighbor_sampler`      | callable     | —                                 |
| Returns                 | PyG Data     | PyG HeteroData                    |

## Determinism guarantees

| API                                          | Determinism                                                                                                                                                                                                                                      |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `NeighborSampler::sample_neighbors`          | Bit-deterministic given `seed`.                                                                                                                                                                                                                  |
| `NeighborSampler::sample_neighbors_temporal` | Bit-deterministic given `seed`.                                                                                                                                                                                                                  |
| `NeighborSampler::sample_neighbors_disjoint` | Bit-deterministic given `seed`.                                                                                                                                                                                                                  |
| `ParallelBatchSampler::sample_batches`       | **Statistically** reproducible given `seed`. Rayon work-stealing can permute per-batch order across runs. Set `SamplingConfig::deterministic = true` to fall back to serial execution and get bit-deterministic output at 2–8× lower throughput. |
| `Graph::reorder_rabbit`                      | **Statistically** reproducible. Concurrent union-find may converge to one of several valid dendrograms depending on thread interleaving. No deterministic mode (yet).                                                                            |
| `HeteroNeighborSampler::sample_neighbors`    | Bit-deterministic given `seed`.                                                                                                                                                                                                                  |
| RDMA gather paths                            | Statistical only (network reordering / NIC-level interleaving).                                                                                                                                                                                  |

Tests that need bit-equality MUST set `SamplingConfig::deterministic = true`
(Python: `SamplingConfig(..., deterministic=True)`). Tests that just need
statistical equivalence (mean fanout, expected hop coverage) can leave the
default.

## Epoch / version model

The `aether-epoch` crate ships a single primitive — `EpochClock`, an `AtomicU64`
wrapped in an ergonomic API. Subsystems opt in by accepting an `Arc<EpochClock>`
and bumping it on commit; readers snapshot `EpochClock::current()` to pin a
version before issuing a multi-source read.

| Subsystem                        | Role     | Contract                                                                                                                                                                      |
| -------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DynamicGraph`                   | Producer | Advances the clock once on each clean writer-guard drop. **Panic-poisoned drops do NOT advance** — readers must never observe a version that claims to include partial edits. |
| Future: versioned `FeatureStore` | Producer | Will advance on each batch commit, sharing the same clock with the graph.                                                                                                     |
| Reader code (sampler, RDMA path) | Consumer | Pins `current_epoch` before fan-out; passes it to subsystems for range-checked reads.                                                                                         |

Python: `DynamicGraph.current_epoch` is a `@property` returning `int`.

## Observability

Three independent layers, each used independently.

**1. Structured logging / spans (Rust + Python)**

- Rust uses the `tracing` crate. `#[tracing::instrument]` is attached to
  coarse-grained entry points: `NeighborLoader::new` /
  `NeighborLoader::shutdown`, `FeatureStore::load`, `load_graph`, and
  `Writer::drop` (trace-level event per commit, warn-level on panic-poisoned
  drop). Hot per-sample paths are NOT instrumented — they use atomic counters
  instead.
- Python uses OpenTelemetry through `aethergraph.tracing` (install with
  `pip install aethergraph[otel]`). `NeighborLoader.__iter__` and the Ray
  datasource emit spans named `epoch`, `ray_worker`, `hetero_epoch`.
- Both layers serialize to OTLP — point them at the same collector to get a
  unified trace across the Rust and Python sides of a sampling call.

**2. Atomic counters (zero-cost when unread)**

| Subsystem            | Collector              | Exposed via                                                                                  |
| -------------------- | ---------------------- | -------------------------------------------------------------------------------------------- |
| Neighborhood sampler | `SamplingTelemetry`    | `NeighborSampler::with_telemetry` (Rust); `SamplingTelemetry()` in `SamplingConfig` (Python) |
| Feature store        | `FeatureLoadTelemetry` | `FeatureStore::with_telemetry` (Rust); `FeatureStore.load(..., telemetry=True)` (Python)     |
| Prefetch loader      | `PrefetchStats`        | `NeighborLoader::stats` (always available)                                                   |

All three are `Arc`-shared, lock-free, and have negligible overhead. Call
`summary()` / `stats()` to drain at scrape time.

**3. Prometheus exposition**

`aethergraph_core::metrics::MetricsSnapshot` rolls the three collectors into one
snapshot and serializes to Prometheus 0.0.4 text format:

```rust
use aethergraph_core::metrics::MetricsSnapshot;
let snap = MetricsSnapshot::collect(
    Some(&sampling_telemetry),
    feature_store.telemetry(),
    Some(&loader.stats_arc()),
);
let exposition: String = snap.to_prometheus();   // serve at GET /metrics
```

Metric names follow `aethergraph_<subsystem>_<event>_total` for counters and
`aethergraph_<subsystem>_<event>` for gauges. There is no embedded HTTP server —
wire `exposition` into whichever framework you host (axum, warp, hyper, etc.).

Python consumers use `prometheus_client` directly against the existing telemetry
attributes (`loader.stats.hit_rate`, `store.telemetry().summary()`, etc.) — no
separate Python-side serializer.

## Durability — WAL

`aether-graph` ships an optional append-only write-ahead log behind the `wal`
Cargo feature. Default builds get no WAL code or runtime cost; callers who need
crash durability opt in:

```rust
use aether_graph::DynamicGraph;
let g = DynamicGraph::open_with_wal("/var/lib/aethergraph/edges.wal", 1_000_000, 8 << 20)?;
```

**Record format** (24 bytes per insert):

| Bytes  | Field | Notes                                                            |
| ------ | ----- | ---------------------------------------------------------------- |
| 0..8   | epoch | `EpochClock` value at insert time. Reserved for future MVCC use. |
| 8..12  | src   | u32 little-endian                                                |
| 12..16 | dst   | u32 little-endian                                                |
| 16..20 | \_pad | reserved (record-type tag for future deletes / annotations)      |
| 20..24 | crc32 | CRC32 (IEEE) of bytes [0..20]                                    |

Fixed-size records keep replay branchless; the per-record CRC catches partial
writes from a kernel that crashed mid-flush.

**Durability contract:**

- `Writer::insert_edge` writes one record to a `BufWriter`. No fsync.
- `Writer::drop` (clean path) calls `fdatasync(2)` — one syscall per
  writer-guard regardless of edge count. Returns control after the data is
  durable.
- WAL append failure inside `insert_edge`, or fsync failure on drop, **poisons
  the graph**. Subsequent `writer()` calls return `WriterError::Poisoned`.
  Recovery requires destroying the in-memory graph and re-opening the WAL.

**Recovery:**

`open_with_wal` reads every record in order, invoking `Writer::insert_edge`
internally to rebuild the in-memory state. The shared `EpochClock` advances once
per replayed record, so `current_epoch()` after recovery equals the number of
records replayed. A live run advances the clock once per writer-guard drop, so
the two agree only when each guard inserted exactly one edge — never compare
epoch values across a restart. If the file ends in a torn record (CRC mismatch /
short read), recovery truncates back to the last clean boundary so subsequent
appends sit on intact data. A record referencing a vertex ≥ the `num_vertices`
being recovered into, or an arena too small for the log's contents, aborts
recovery with an error rather than silently dropping records.

**What is NOT in scope today:**

- Checkpoints — the WAL grows unbounded; no mechanism truncates "everything
  before epoch N" yet. Production users with high ingest rates must rotate
  manually.
- Per-batch atomicity — a single `Writer` guard's inserts are not a transaction.
  A crash mid-guard recovers a prefix, not all-or-nothing.
- Async / `io_uring` — sync writes via `BufWriter` + `fdatasync`. Enough for the
  typical Kafka-driven ingest rate; the I/O path can be swapped later without
  changing the on-disk format.
