# AetherGraph Architecture

## Public API

```python
from aethergraph import Graph, DynamicGraph, HeteroGraph
from aethergraph.pytorch import NeighborLoader, HeteroNeighborLoader
```

## Crates

| Crate              | Purpose                                                          |
|--------------------|------------------------------------------------------------------|
| `aethergraph-core` | Static CSR graph, homo + hetero sampling, feature store, io_uring|
| `aether-graph`     | Dynamic C-tree graph, lock-free concurrent R/W, arena bump-alloc |
| `aether-stream`    | AF_XDP ingestion, seqlock features, ibverbs RDMA, GPUDirect      |
| `aether-mem`       | Lock-free slab allocator, HugePages, RDMA/CUDA memory hooks      | 
| `aethergraph-py`   | PyO3 Python bindings for all of the above                        |
| `aethergraph-cli`  | CLI tools (convert, info, stats)                                 |

## aethergraph-core — Static Graph Engine

CSR-backed graph with SIMD-accelerated sampling. Memory-mapped from NVMe, instant startup.

**Sampling design:**
- Generation-tagged direct array (< 1M nodes) or FxHashMap (> 1M nodes) for node dedup — O(1) lookup
- Pre-allocated buffers reused across calls via `clear()` — zero hot-path allocations
- Local indices assigned during sampling — no post-sort, no binary search for edge remapping
- Floyd's O(k) sampling with 256-byte stack bitmap for small degree nodes
- Double-buffered frontiers — `mem::swap` instead of allocating per hop

**Heterogeneous sampling:** 
- Multiple CSR matrices (one per edge type) with per-type local node IDs. 
- Per-edge-type fanout. Stack-copied edge type IDs in the hot loop to avoid borrow conflicts. 
- Macro-inlined Floyd's to avoid `&mut self` conflicts with bitmap.

**Cache-locality reordering (Rabbit Order, Arai et al. IPDPS 2016):**
- Phase 1 (parallel over V): each node picks its lowest-degree neighbor as a merge candidate.
- Phase 2 (parallel over E): lock-free concurrent union-find with `AtomicU32` parent + rank, path-splitting `find`, CAS `union`. Logs `(winner, loser)` pairs in merge order.
- Phase 3 (sequential O(V)): replay the merge log into a dendrogram and emit the permutation by in-order traversal.
- Community partitions are a free byproduct of the merge log — `rabbit_partitions()` runs a sequential UF replay without rebuilding the dendrogram. `reorder_rabbit_with_partitions()` returns both in one pass.
- `partition_aligned_batches` uses the partition labels to construct seed batches whose neighborhoods overlap, amortizing destination-array reads across the batch.
- Measured in `benches/graph_benchmarks.rs::reorder_sampling_speedup` — sampling throughput delta on the same seeds before vs. after permutation.

## aether-graph — Dynamic Graph Engine

Lock-free C-tree for live-updating graphs. Single-writer, multi-reader via functional persistence.

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
- 90% of nodes (degree < 15) fit in a single chunk — 1 cache line read, same as CSR
- Edge insert: path-copy affected nodes, arena bump-alloc (single `fetch_add`), atomic root swap
- Readers see a consistent snapshot via functional persistence — old tree valid for concurrent readers
- Arena: pre-allocated contiguous region, zero-alloc writes, bulk free on epoch reclaim
- 9M inserts/sec, 38M neighbor reads/sec (criterion benchmarks)

## aether-stream — Real-Time Feature Serving

Linux-only kernel-bypass engine. Compiles to empty crate on macOS/Windows.

**Seqlock slot layout (RDMA-safe):**
```
[head_version: u64] [features: f32 × dim] [tail_version: u64]
```
Writers: head odd → write features → tail even → head even.
RDMA readers: single bulk read, GPU kernel validates head == tail.

**ibverbs FFI:** 16 extern C functions, 12 `#[repr(C)]` structs. Non-opaque `IbvMr` (rkey/lkey fields). `IbvQp` with correct field offsets through `qp_num`. `IbvSendWrUnion` as a real union with atomic-variant padding.

**GPU pipeline:** `GpuGatherBuffer` (VRAM via cudarc, registered with NIC via nvidia-peermem) → `SeqlockValidator` (CUDA kernel compiled at runtime via nvrtc from standalone `.cu` file) → DLPack capsule → `torch.from_dlpack()`.

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

| Type                    | Crate         | Purpose                                       |
|-------------------------|---------------|-----------------------------------------------|
| `Graph`                 | core          | Static CSR, mmap'd from NVMe                  |
| `HeteroGraph`           | core          | Multi-relational CSR (one per edge type)      |
| `DynamicGraph`          | aether-graph  | Lock-free C-tree, concurrent R/W              |
| `NeighborSampler`       | core          | Floyd's O(k) sampling, zero-alloc             |
| `reorder_rabbit`        | core          | Rabbit Order permutation (parallel UF merge)  |
| `rabbit_partitions`     | core          | Community labels (free byproduct of merge log)|
| `HeteroNeighborSampler` | core          | Typed multi-hop, pre-computed local indices   |
| `NeighborLoader`        | core          | Prefetch thread, io_uring features            |
| `FeatureTable`          | aether-stream | Seqlock feature table for RDMA                |
| `RdmaFeatureClient`     | aether-stream | GPUDirect RDMA gather API                     |
| `Chunk`                 | aether-graph  | 64-byte sorted neighbor chunk (1 cache line)  |
| `Arena`                 | aether-graph  | Bump allocator for C-tree nodes               |
| `CTree`                 | aether-graph  | Balanced tree of chunks (functional persistence)|
| `SharedMemoryRing`      | aether-mem    | Lock-free slab with HugePages                   |

## Platform Support

| Platform | Static Graph | Dynamic Graph | Feature I/O        | Streaming      |
|----------|-------------|---------------|--------------------|-----------------|
| Linux    | mmap        | C-tree        | io_uring + SQPOLL  | AF_XDP + RDMA   |
| macOS    | mmap        | C-tree        | mmap + madvise     | N/A             |
| Windows  | mmap        | C-tree        | mmap               | N/A             |

## Binary Formats

### Graph v1 (`.bin`, static CSR)

```
[Header: 32 bytes]
  magic: u32 = 0x41455448 ("AETH"), version: u32 = 1
  num_nodes: u64, num_edges: u64, has_weights: u32

[Offsets: (num_nodes + 1) × 8 bytes]
[Edges: num_edges × 4 bytes]
[Weights: num_edges × 4 bytes] (optional)
```

### Features (`.bin`)

```
[Header: "AETHFEAT" + num_nodes: u64 + feature_dim: u64 + data_offset: u64]
[Data: num_nodes × feature_dim × 4 bytes] (f32, row-major)
```

## PyG Compatibility

| Feature                | Homogeneous | Heterogeneous                     |
|------------------------|-------------|-----------------------------------|
| `data`                 | Graph       | HeteroGraph                       |
| `num_neighbors`        | list[int]   | dict[(src, rel, dst) → list[int]] |
| `input_nodes`          | Tensor      | (node_type, Tensor)               |
| `batch_size`, `shuffle`| yes         | yes                               |
| `pin_memory`           | yes         | yes                               |
| `transform`            | yes         | yes                               |
| `weighted`             | yes         | —                                 |
| `temporal_strategy`    | uniform/last| —                                 |
| `disjoint`             | yes         | —                                 |
| `neighbor_sampler`     | callable    | —                                 |
| Returns                | PyG Data    | PyG HeteroData                    |
