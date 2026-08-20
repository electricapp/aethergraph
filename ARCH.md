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

- Generation-tagged direct array for node dedup, chosen by memory budget: one
  `u64` slot per node up to 8M nodes (a 64 MiB table), FxHashMap above that. The
  table is `alloc_zeroed`, so slots a batch never touches are never faulted in.
- Per-call output buffers sized from the previous call plus an eighth of
  headroom, so a steady-state batch reallocates nothing. An exact fit would
  leave roughly half of all calls one element short, and falling short costs a
  full-array copy. Asserted at zero reallocations by
  `tests/sampler_allocations.rs`.
- Pre-allocated scratch reused across calls via `clear()` — zero hot-path
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

## Feature I/O paths

Four ways a feature row reaches a tensor, chosen at open time and each falling
back to the one below it.

**NVMe passthrough** (`nvme-passthru`). `IORING_OP_URING_CMD` against `/dev/ng*`
with rows resolved to LBAs through FIEMAP, so a read skips the block layer
entirely. Probed at store-open; a row that is not stably mapped drops the whole
batch to the io_uring gather.

**io_uring gather** (`io-uring`). O_DIRECT plus IOPOLL where the layout allows
it, falling back to SQPOLL-only on an unaligned payload. Each ring is owned by
one thread — work arrives as a closure over that thread's lane — so the ring
keeps a single submitter and can run `DEFER_TASKRUN` with `SINGLE_ISSUER`. Files
and landing buffers are registered, so reads land via `ReadFixed` and the kernel
skips the per-op page pin. The real O_DIRECT alignment comes from
`statx(STATX_DIOALIGN)` rather than a hardcoded 512, because a 4Kn device
rejects a merely 512-aligned layout with `EINVAL`.

**mmap** (portable). `MADV_RANDOM` so readahead does not fault 128 KiB per
useful row, plus `MADV_HUGEPAGE` on the payload, since TB-scale random gathers
are dTLB-bound at 4 KiB pages.

**userfaultfd pager** (`uffd`). Registers an anonymous region and services
faults from the file under a fixed residency budget, so the process holds a
bounded number of pages rather than whatever the kernel chose to keep. Eviction
is degree-weighted: pass node degrees and a hub's pages outlive a leaf's.

Above all four, `SharedFeatureStore` (`shm`) holds one copy of the matrix in a
sealed memfd and passes the descriptor over `SCM_RIGHTS`, so N worker processes
map the same physical pages read-only instead of each paying for their own.

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
- Readers see a consistent snapshot via functional persistence — the tree
  reached from the root a reader loaded stays intact for the lifetime of its
  `ReadGuard`, which is also what holds off recycling of the slots the writer
  has retired
- C-tree balance: scapegoat scheme (α = 2/3) — an insert that pushes its path
  past the α-height bound rebuilds the highest weight-unbalanced subtree, so
  depth stays O(log degree) even for monotonically increasing neighbor IDs
- Arena: pre-allocated contiguous region (max 32 GiB — u32 slot indices with a
  tag bit), zero-alloc writes; superseded nodes are logged as they are replaced
  and recycled once no reader or pinned snapshot can still observe them, so
  steady-state footprint tracks live edges rather than total inserts;
  `DynamicGraph::compact` repacks every live tree into a fresh arena and
  reclaims any slot the recycler missed
- Snapshots: each writer commit publishes an immutable `Snapshot` (CoW
  root-table pages, shared with the previous commit); `acquire()` clones it,
  pinning its epoch so reclamation holds off. Pinned reads are gate-free — zero
  per-vertex synchronization while ingest keeps committing
- 9M inserts/sec, 38M neighbor reads/sec (criterion benchmarks)

## aether-stream — Real-Time Feature Serving

Linux-only kernel-bypass engine. Compiles to empty crate on macOS/Windows.

**Seqlock slot layout (RDMA-safe):**

```
[head_version: u64] [features: f32 × dim] [tail_version: u64]
```

Writers claim the head by CAS (even → odd; same-node writers serialize by
spinning), copy features with volatile ops, then store tail and head to the next
even value. Local readers re-load head after the copy. RDMA readers issue two
sequential bulk reads into separate staging regions; a row is accepted only when
both snapshots carry the same even, nonzero head == tail AND identical payload
bytes — sound without any PCIe read-ordering assumption.

**ibverbs FFI:** 19 extern C functions, 16 `#[repr(C)]` structs. Non-opaque
`IbvMr` (rkey/lkey fields). `IbvQp` with correct field offsets through `qp_num`.
`IbvSendWrUnion` as a real union with atomic-variant padding.

**GPU pipeline:** `GpuGatherBuffer` (VRAM via cudarc, registered with the NIC
via nvidia-peermem, falling back to dma-buf export under IOMMU-mediated VMs) →
`cuFlushGPUDirectRDMAWrites` after each completion → `SeqlockValidator` (CUDA
kernel compiled at runtime via nvrtc from standalone `.cu` file) → per-batch
owned copy of the output (returned tensors survive subsequent gathers) → DLPack
capsule → `torch.from_dlpack()`.

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
│   FeatureTable       │  ← seqlock write (head CAS even→odd → write → tail even → head even)
│  (HugePage RAM)      │
└──────────────────────┘
    │                         GPU Node
    ├── TCP Control Plane ──► fetch_advertisement() → base_addr, rkey, schema
    │                         connect_with_qp() → QP endpoint exchange
    │                                │
    └── RDMA READ ×2 ◄───────────── RdmaFeatureClient::gather(node_ids)
                                     │  (two sequential snapshots per slot,
                                     │   cuFlushGPUDirectRDMAWrites after each)
                                     ▼
                              GpuGatherBuffer (VRAM staging ×2, nvidia-peermem
                              or dma-buf)
                                     │
                                     ▼
                              SeqlockValidator (CUDA kernel)
                              if both snapshots: head == tail, even, nonzero,
                              and payloads identical → compact to output
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

Linux-only subsystems, each behind a cargo feature and each runtime-probed so a
build carrying it still runs where the kernel or hardware does not:

| Subsystem          | Feature         | Falls back to                  |
| ------------------ | --------------- | ------------------------------ |
| NVMe passthrough   | `nvme-passthru` | O_DIRECT io_uring gather       |
| userfaultfd pager  | `uffd`          | `FeatureStore::load` (mmap)    |
| memfd shared store | `shm`           | per-process mmap               |
| `perf_event_open`  | `perf`          | a counter set with none active |
| NUMA placement     | `numa`          | default kernel placement       |
| GPUDirect Storage  | `gds`           | io_uring gather                |

Because these compile nowhere else, `scripts/check-linux.sh` cross-compiles them
with zig from a non-Linux machine. `cargo check` on macOS builds none of them.

## Binary Formats

Every on-disk format starts with a 32-bit magic plus a 32-bit `version` integer.
The version is **monotonically increasing** — old wheels MUST refuse to load a
newer-version file rather than mis-parsing it. Adding backward-compatibility for
an older version requires a deliberate code change + a golden-file regression
test (see `crates/aethergraph-core/tests/golden/`).

### Graph (static CSR, `.bin`)

| Version | Magic                   | Status  | Notes                                                 |
| ------- | ----------------------- | ------- | ----------------------------------------------------- |
| **v1**  | `0x41455448` (`"AETH"`) | current | Flat arrays, mmap-able in place.                      |
| **v2**  | `0x41455448` (`"AETH"`) | current | Succinct-coded. Same header, decodes to owned arrays. |

Both are emitted today and `load` dispatches on `header.version`. v2 replaces
the flat arrays with Elias-Fano offsets and StreamVByte-coded edge deltas,
typically 2-4x smaller at rest. It cannot be mapped in place — a v2 file always
decodes into owned arrays, so `storage="mmap"` raises on one. Written with
`graph.save(path, compressed=True)`.

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

| Version | Magic        | Status  | Notes                          |
| ------- | ------------ | ------- | ------------------------------ |
| **v1**  | `"AETHHETG"` | current | One CSR section per edge type. |

Layout (v1): 64-byte header (magic, `version` u32, type counts, reserved),
node/edge type tables, then per-edge-type CSR sections. u32 sections (edges,
weights) are zero-padded to 8-byte boundaries so every offsets array stays
u64-aligned. See `aethergraph_core::internal::mmap_hetero` for field-level
detail. The loader bounds-checks every section against the file length with
checked arithmetic — corrupt or truncated files fail at load, not at access.

### Features (`.bin`)

| Version | Magic        | Status  | Notes                        |
| ------- | ------------ | ------- | ---------------------------- |
| **v1**  | `"AETHFEAT"` | current | Row-major, f32 / f16 / bf16. |

Layout (v1):

```
[Header: 32 bytes]
  magic        char[8] = "AETHFEAT"
  num_nodes    u64
  feature_dim  u64
  data_offset  u64
[dtype tag: 1 byte at offset 32 — 0 = f32, 1 = f16, 2 = bf16]
[Data: num_nodes × feature_dim × element_size bytes, LE row-major]
```

`element_size` is 4 for f32 and 2 for the half-width types, so an f16 or bf16
file is half the size. bf16 keeps f32's exponent range and loses mantissa bits;
f16 does the reverse. Readers resolve the decoder from the tag, so nothing at
the call site changes.

Decoding dispatches on CPU features once per batch rather than per row: AVX-512
or F16C on x86-64 and NEON on aarch64 for f16, AVX2 for bf16, with a scalar
fallback. `FeatureDtype::row_decoder()` resolves that choice at the batch
boundary and the per-row path is a register-held branch.

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

| Subsystem                        | Role     | Contract                                                                                                                                                                                                            |
| -------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DynamicGraph`                   | Producer | Advances the clock once on each clean writer-guard drop. **Panic-poisoned drops do NOT advance** — readers must never observe a version that claims to include partial edits.                                       |
| Future: versioned `FeatureStore` | Producer | Will advance on each batch commit, sharing the same clock with the graph.                                                                                                                                           |
| Reader code (sampler)            | Consumer | Pins `current_epoch` before fan-out; passes it to subsystems for range-checked reads. The RDMA gather path serves latest-value reads only — a seqlock slot holds one version and cannot serve historical snapshots. |

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

- `Writer::insert_edge` appends one record to a `BufWriter` **before**
  publishing the edge; a failed append publishes nothing, poisons the graph, and
  returns `InsertError::WalAppend`. No fsync per record.
- `Writer::drop` (clean path) calls `fdatasync(2)` — one syscall per
  writer-guard regardless of edge count. Returns control after the data is
  durable.
- WAL append failure inside `insert_edge`, or fsync failure on drop, **poisons
  the graph**. Subsequent `writer()` calls return `WriterError::Poisoned`.
  Recovery requires destroying the in-memory graph and re-opening the WAL.
- A panic-poisoned guard discards its still-buffered records instead of flushing
  them; bytes the 64 KiB `BufWriter` already spilled to the OS may survive and
  replay as a valid prefix.
- The WAL file holds an exclusive advisory lock for its lifetime; a second
  `open_with_wal` on the same path fails with `WalError::Locked`.
- Streaming ingest (`aether_graph::ingest`) commits — drops and reacquires the
  writer guard, fsyncing and advancing the epoch clock — every 65,536 edges and
  at every batch boundary.

**Recovery:**

`open_with_wal` reads every record in order, invoking `Writer::insert_edge`
internally to rebuild the in-memory state. The shared `EpochClock` advances once
per replayed record, so `current_epoch()` after recovery equals the number of
records replayed. A live run advances the clock once per writer-guard drop, so
the two agree only when each guard inserted exactly one edge — never compare
epoch values across a restart. If the file ends in a torn record (CRC mismatch /
short read), recovery truncates back to the last clean boundary and fsyncs the
truncation before any new append, so a later crash cannot resurrect the
discarded bytes. A record referencing a vertex ≥ the `num_vertices` being
recovered into, or an arena too small for the log's contents, aborts recovery
with an error rather than silently dropping records.

**What is NOT in scope today:**

- Checkpoints — the WAL grows unbounded; no mechanism truncates "everything
  before epoch N" yet. Production users with high ingest rates must rotate
  manually.
- Per-batch atomicity — a single `Writer` guard's inserts are not a transaction.
  A crash mid-guard recovers a prefix, not all-or-nothing.
- Async / `io_uring` — sync writes via `BufWriter` + `fdatasync`. Enough for the
  typical Kafka-driven ingest rate; the I/O path can be swapped later without
  changing the on-disk format.
