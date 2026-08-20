```
█████╗ ███████╗████████╗██╗  ██╗███████╗██████╗  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗
██╔══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║
███████║█████╗     ██║   ███████║█████╗  ██████╔╝██║  ███╗██████╔╝███████║██████╔╝███████║
██╔══██║██╔══╝     ██║   ██╔══██║██╔══╝  ██╔══██╗██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║
██║  ██║███████╗   ██║   ██║  ██║███████╗██║  ██║╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║
╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
```

**Rust core + Python bindings.** High-performance graph sampling and real-time
feature serving for GNN training on billion-scale evolving graphs.

> _"Because the CSR layout is simply 3 arrays, it scales on a single computer: a
> CSR matrix can be laid out on a disk instead of in-memory. You simply memory
> map the 3 arrays and use them on-disk from there._
>
> _With modern NVMe drives random seeks aren't slow anymore, much faster than
> distributed network calls like you do when scaling the linked list-based
> graph. **I haven't seen anyone actually implement this yet**, but it's in the
> roadmap for my implementation at least."_
>
> —
> [VodkaHaze, r/MachineLearning](https://www.reddit.com/r/MachineLearning/comments/kqazpd/d_why_im_lukewarm_on_graph_neural_networks/)
> (2021)

## Quick Start

```python
from aethergraph import Graph
from aethergraph.pytorch import NeighborLoader

# Static graph (CSR, mmap'd from NVMe)
graph = Graph.load("graph.bin")
graph.load_features("features.bin")

loader = NeighborLoader(graph, num_neighbors=[15, 10], batch_size=128)
for batch in loader:
    out = model(batch.x, batch.edge_index)
```

```python
from aethergraph import DynamicGraph

# Live graph (C-tree, single-writer ingest, lock-free readers)
graph = DynamicGraph(num_vertices=2_000_000_000, arena_mb=8192)
graph.insert_edge(user_id, post_id)  # from Kafka consumer
# Training reads concurrently — no locks, no stale snapshots
```

```python
from aethergraph import HeteroGraph
from aethergraph.pytorch import HeteroNeighborLoader

# Heterogeneous graph (Reddit: users, posts, comments, subreddits)
loader = HeteroNeighborLoader(
    hetero_graph,
    num_neighbors={
        ("user", "votes", "post"): [15, 10],
        ("user", "writes", "comment"): [10, 5],
        ("post", "belongs_to", "subreddit"): [3, 2],
    },
    input_nodes=("user", train_user_ids),
    batch_size=128,
)
for batch in loader:  # yields PyG HeteroData
    user_x = batch["user"].x
    edge_index = batch["user", "votes", "post"].edge_index
```

## Why AetherGraph?

Traditional GNN frameworks load entire graphs into GPU memory, hitting the VRAM
wall at millions-of-nodes scale. AetherGraph keeps topology on disk and streams
neighborhoods via memory-mapping + `io_uring`, enabling 100B+ edge graphs on
commodity hardware.

|                     | PyTorch Geometric    | AetherGraph                      |
| ------------------- | -------------------- | -------------------------------- |
| Max graph size      | ~40M nodes (VRAM)    | 2B+ nodes (NVMe)                 |
| Graph updates       | Full rebuild         | O(log degree) insert, no rebuild |
| Feature loading     | All in RAM           | Streamed on-demand               |
| Feature serving     | N/A                  | GPUDirect RDMA (<5us)            |
| Hetero graphs       | Python-level looping | Rust-native typed sampling       |
| Startup time        | Minutes (load all)   | Instant (mmap)                   |
| Sampling (1M nodes) | 341 us/batch         | 240 us/batch (1.4x faster)       |

**How it works:**

- **Static CSR**: Graph stored as 3 arrays (offsets, destinations, weights) —
  O(1) neighbor lookup, mmap'd from NVMe
- **Dynamic C-tree**: Balanced tree of cache-line-sized chunks — O(log degree)
  edge insert, lock-free reads via functional persistence and atomic root swap
- **Rabbit Order reordering**: Hierarchical community-detection vertex
  permutation (Arai et al., IPDPS 2016) for cache-friendly sampling on power-law
  graphs
- **io_uring (Linux)**: Batched async NVMe reads, ~10us per random access. Each
  ring is owned by one thread and set up with `DEFER_TASKRUN`, registered files
  and registered buffers, so a read is submitted and reaped without a syscall
- **NVMe passthrough**: `IORING_OP_URING_CMD` against `/dev/ng*` maps feature
  rows to LBAs and skips the block layer, falling back to the O_DIRECT gather
  when a row is not stably mapped
- **Compressed at rest**: Elias-Fano offsets plus StreamVByte edges typically
  cut a graph file 2-4x, and it loads through the same `Graph.load`
- **Half-precision features**: f16 and bf16 payloads halve the feature file and
  upcast on the way out through AVX-512 / F16C / NEON / AVX2 kernels
- **Out-of-core and shared features**: a userfaultfd pager bounds resident pages
  under a degree-weighted budget; a sealed memfd lets N worker processes map one
  copy of the feature matrix
- **NUMA placement**: graph body interleaved across nodes with the sampler pool
  pinned to them
- **AF_XDP + RDMA**: Kernel-bypass ingestion and GPUDirect feature serving
- **Rust core**: Zero-copy data pipeline from disk to PyTorch tensors

## Crates

| Crate              | Purpose                                                                |
| ------------------ | ---------------------------------------------------------------------- |
| `aethergraph-core` | Static CSR graph, homogeneous + heterogeneous sampling, feature store  |
| `aethergraph-cli`  | CLI tools (convert, info, stats)                                       |
| `aethergraph-py`   | PyO3 Python bindings                                                   |
| `aether-graph`     | Dynamic graph with C-tree neighbor lists, lock-free reader path        |
| `aether-mem`       | Lock-free slab allocator with HugePages and pluggable memory hooks     |
| `aether-stream`    | Kernel-bypass streaming: AF_XDP ingestion, seqlock feature table, RDMA |
| `aether-epoch`     | Monotonic version clock for consistent reads across subsystems         |

## Three Graph Modes

### 1. Static Graph (CSR) — `Graph`

For graphs that don't change during training. Memory-mapped from NVMe, instant
startup.

```python
graph = Graph.load("reddit_2B.bin")  # instant, no loading
loader = NeighborLoader(graph, num_neighbors=[15, 10], batch_size=128)
```

### 2. Dynamic Graph (C-tree) — `DynamicGraph`

For graphs that evolve during training. Single-writer ingest publishes new
C-tree roots atomically; readers run concurrently without taking locks. Edges
arrive from Kafka/Flink while training samples neighborhoods.

```python
graph = DynamicGraph(num_vertices=2_000_000_000, arena_mb=8192)

# Writer thread (Kafka consumer)
for edge in kafka_stream:
    graph.insert_edge(edge.src, edge.dst)

# Reader thread (training, concurrent)
neighbors = graph.neighbors(node_id)  # lock-free, consistent snapshot
```

### 3. Heterogeneous Graph — `HeteroGraph`

For multi-relational graphs (users, posts, comments, subreddits). Typed k-hop
sampling with per-edge-type fanout. Drop-in replacement for PyG's HeteroData
NeighborLoader.

```python
loader = HeteroNeighborLoader(
    hetero_graph,
    num_neighbors={("user", "votes", "post"): [15, 10], ...},
    input_nodes=("user", train_ids),
    batch_size=128,
)
```

## Live Features via RDMA

For real-time models (fraud detection, feed ranking), features change with every
user interaction. AetherGraph serves live features directly into GPU memory via
GPUDirect RDMA, bypassing CPU entirely.

```
NIC (AF_XDP)                  GPU Node
    │                             │
    ▼                             │
Ingestion Loop                    │
(busy-poll, per-queue thread)     │
    │                             │
    ▼                             │
FeatureTable                      │
(seqlock, HugePage RAM)           │
    │                             │
    ├── TCP Control Plane ──────► Discover base_addr + rkey
    │                             │
    └── RDMA READ ◄───────────── GPUDirect (NIC → PCIe → VRAM)
                                  │
                                  ▼
                            GPU Kernel
                        (validate + compact)
                                  │
                                  ▼
                          [batch, feature_dim]
                            output tensor
```

```python
# One config change — features arrive via RDMA instead of from disk
loader = NeighborLoader(
    graph, num_neighbors=[15, 10], batch_size=128,
    feature_source="rdma://10.0.0.1:9999",
)
for batch in loader:
    batch.x  # CUDA tensor, arrived via GPUDirect RDMA in <5us
```

## Cache-Locality Reordering (Rabbit Order)

Sampling a billion-node graph is memory-bound — the working set is the
destination array, and random neighbor lookups thrash the cache. Rabbit Order
(Arai et al., IPDPS 2016) computes a vertex permutation that places communities
contiguously, so neighbor reads stay in L2/L3 far longer.

```python
graph = Graph.load("graph.bin")
perm = graph.reorder_rabbit()        # returns the permutation
reordered = graph.permute(perm)      # build a new CSR in the new order
reordered.save("graph.reordered.bin")
```

**Implementation:**

- Phase 1 (parallel over V): each node picks its lowest-degree neighbor.
- Phase 2 (parallel over E): lock-free concurrent union-find merges remaining
  cross-community edges (`AtomicU32` parent + rank, path-splitting `find`, CAS
  `union`).
- Phase 3 (sequential, O(V)): replay the merge log into a dendrogram and emit
  the permutation.

The community partitions fall out of the merge log for free —
`graph.rabbit_partitions()` returns dense partition IDs without a second pass,
and `partition_aligned_batches` uses them to build seed batches that respect
locality.

See `crates/aethergraph-core/benches/graph_benchmarks.rs` (`rabbit_reorder`,
`reorder_sampling_speedup`) for the measurement methodology.

## Storage and Memory

Four independent levers for fitting a graph and its features on the hardware you
have. They compose: a compressed graph with bf16 features served out of a shared
region is the usual multi-worker setup.

**Compressed graph files.** Elias-Fano offsets and StreamVByte edges, typically
2-4x smaller at rest. `load` detects the format, so nothing at the call site
changes.

```python
graph.save("graph.bin", compressed=True)
graph = Graph.load("graph.bin")          # same call either way
```

**Half-precision features.** bf16 keeps f32's exponent range and loses mantissa
bits, which is the usual trade for trained embeddings; f16 does the reverse.
Both halve the file. Readers pick the decoder up from the header.

```python
from aethergraph import save_features
save_features("features.bin", x, dtype="bf16")   # "f32" | "f16" | "bf16"
```

**Out-of-core paging.** Bounds resident pages instead of letting the kernel
decide, and evicts by degree so hub nodes outlive leaves.

```python
store = FeatureStore.load_paged("features.bin", budget_pages=1 << 20,
                                degrees=graph.degrees())
faults, evictions = store.pager_stats()
```

**Shared feature memory.** One copy of the matrix for N workers: the owner seals
it into a memfd and serves the descriptor over a Unix socket, each worker maps
the same physical pages read-only.

```python
owner = SharedFeatureStore.publish("features.bin")   # in the parent
owner.serve("/tmp/ag.sock")

store = SharedFeatureStore.attach("/tmp/ag.sock")    # in each worker
```

The tiered `FeatureCache` (GPU / CPU / NVMe) can also take a `cold_store_path`,
which compresses the whole matrix into a resident zstd-backed tier underneath
the NVMe spill. That makes the cache a complete feature source: a node in no
other tier decompresses out of its block instead of raising.

## Installation

```bash
pip install aethergraph
```

For PyTorch Geometric integration:

```bash
pip install aethergraph[pytorch-geometric]
```

Wheels target the stable ABI (CPython 3.10+), with a separate free-threaded
`cp314t` wheel built without abi3 — the free-threaded ABI is not part of abi3.
The extension declares `gil_used = false`, so importing it into a no-GIL build
does not re-enable the GIL: every class is `Send` and guards its own state with
Rust locks rather than relying on the GIL.

### Build features

Platform-specific paths are cargo features. All of them degrade at runtime
rather than failing the build, so a wheel carrying them still runs where the
kernel or hardware does not.

| Feature         | Crate            | Effect                                          |
| --------------- | ---------------- | ----------------------------------------------- |
| `io-uring`      | aethergraph-core | Async NVMe reads (Linux)                        |
| `nvme-passthru` | aethergraph-core | Block-layer-bypassing gather over `/dev/ng*`    |
| `zstd-tier`     | aethergraph-core | Compressed resident cold feature tier           |
| `uffd`          | aethergraph-core | userfaultfd out-of-core pager                   |
| `shm`           | aethergraph-core | memfd cross-process shared feature store        |
| `perf`          | aethergraph-core | `perf_event_open` self-profiling counters       |
| `numa`          | aethergraph-core | Graph interleaving and sampler-pool pinning     |
| `gds`           | aethergraph-core | GPUDirect Storage (cuFile) feature reads        |
| `parquet`       | aethergraph-core | Parquet edge/feature import                     |
| `rdma`          | aether-stream    | RDMA feature transport (InfiniBand or RoCE)     |
| `gpudirect`     | aether-stream    | NIC-to-VRAM delivery, DLPack handoff to PyTorch |
| `xdp_bpf`       | aether-stream    | AF_XDP kernel-bypass ingestion                  |

Python builds enable `uffd`, `shm` and `perf` by default; `gpudirect` is opt-in.
Check what a wheel actually carries at runtime:

```python
from aethergraph import _core
_core.HAS_GPUDIRECT, _core.HAS_NUMA, _core.HAS_SHARED_STORE, _core.HAS_PERF_COUNTERS
```

## Ray Data (Distributed Training)

Replicated Topology: graph on each worker's NVMe, only gradients cross the
network.

```
Traditional:  Worker 1 ←── graph edges ──→ Worker 2   # Terabytes over network
AetherGraph:  Worker 1 ←── gradients ──→ Worker 2     # Megabytes over network
                  │                          │
                NVMe                       NVMe        # Graph is local
```

```python
import ray
from aethergraph.ray import create_sampling_dataset, collate_to_pyg

ray.init()
dataset = create_sampling_dataset(graph_path="graph.bin", num_neighbors=[25, 10], batch_size=1024)
for batch in dataset.iter_batches(batch_format="pyarrow"):
    data = collate_to_pyg(batch)
    out = model(data.x, data.edge_index)
```

## API Reference

### Graph (Static CSR)

```python
from aethergraph import Graph

graph = Graph.load("graph.bin")            # mmap, instant startup
graph = Graph.from_edges(num_nodes, src, dst)
graph.save("graph.bin")                    # add compressed=True for the succinct format
graph.load_features("features.bin")
graph.num_nodes  # int
graph.num_edges  # int
graph.degrees()                            # every node's degree
graph.degrees_of(nodes)                    # just these, in input order

# Cache-locality reordering (Rabbit Order)
perm = graph.reorder_rabbit()                       # permutation: new_id → old_id
reordered = graph.permute(perm)                     # new CSR in the permuted order
parts = graph.rabbit_partitions()                   # dense partition IDs per node
perm, parts = graph.reorder_rabbit_with_partitions()  # both in one pass
```

### DynamicGraph (C-tree)

```python
from aethergraph import DynamicGraph

graph = DynamicGraph(num_vertices=2_000_000_000, arena_mb=8192)
graph.insert_edge(src, dst)                # O(log degree), lock-free for readers
graph.insert_edges(src_array, dst_array)   # batch insert from numpy
graph.neighbors(node)                      # sorted numpy array
graph.degree(node)                         # O(1)
graph.has_edge(src, dst)                   # O(log degree)

snap = graph.acquire()                     # pin the latest committed snapshot
snap.neighbors(node)                       # immutable while inserts continue
snap.to_static()                           # atomic-cut CSR freeze
```

### NeighborLoader (Homogeneous)

```python
from aethergraph.pytorch import NeighborLoader

loader = NeighborLoader(
    graph,
    num_neighbors=[15, 10],
    batch_size=128,
    input_nodes=train_idx,
    shuffle=True,
)

for batch in loader:
    batch.x                  # [num_nodes, feat_dim] node features
    batch.edge_index         # [2, num_edges] COO edges
    batch.n_id               # [num_nodes] original node IDs
    batch.batch_size         # number of seed nodes
    batch.input_id           # seed indices in batch
    batch.num_sampled_nodes  # nodes per hop [num_hops]
    batch.num_sampled_edges  # edges per hop [num_hops]
```

### HeteroNeighborLoader (Heterogeneous)

```python
from aethergraph.pytorch import HeteroNeighborLoader

loader = HeteroNeighborLoader(
    hetero_graph,
    num_neighbors={
        ("user", "votes", "post"): [15, 10],
        ("user", "writes", "comment"): [10, 5],
    },
    input_nodes=("user", train_user_ids),
    batch_size=128,
)

for batch in loader:  # yields PyG HeteroData
    batch["user"].x
    batch["user", "votes", "post"].edge_index
```

### Features

```python
from aethergraph import FeatureStore, SharedFeatureStore, save_features

save_features("features.bin", x, dtype="bf16")   # "f32" | "f16" | "bf16"

store = FeatureStore.load("features.bin")        # mmap
store = FeatureStore.load_paged("features.bin", budget_pages, degrees)
store.get(node)                                  # one row
store.get_batch(nodes)                           # [len(nodes), feature_dim]
store.pager_stats()                              # (faults, evictions) when paged

owner = SharedFeatureStore.publish("features.bin")
owner.serve("/tmp/ag.sock")                      # until stop_serving() or drop
worker = SharedFeatureStore.attach("/tmp/ag.sock")
worker.shared_bytes                              # region size, mapped once per host
```

### PerfCounters

Hardware counters around a block of work. Thread-scoped, user space only. A host
that withholds PMU access grants fewer counters; those come back `None` rather
than raising, and `active` reports what was granted.

```python
from aethergraph import PerfCounters

with PerfCounters(["cycles", "LLC-misses", "dTLB-misses"]) as pc:
    for batch in loader:
        train_step(batch)
print(pc.readings())
```

### Differences from PyTorch Geometric

AetherGraph is a **high-performance replacement** for PyG's data loading
pipeline.

**Supported (homogeneous + heterogeneous):**

- `num_neighbors` (list or per-edge-type dict)
- `input_nodes` (Tensor, array, list, or `(node_type, Tensor)` for hetero)
- `batch_size`, `shuffle`, `replace`, `transform`
- `pin_memory`, `prefetch_factor`
- Weighted sampling (`weighted=True`, uses edge weights)
- Temporal sampling (`input_time`, `temporal_strategy="uniform"|"last"`)
- Disjoint subgraphs (`disjoint=True`, per-seed isolation with `batch` vector)
- Custom samplers (`neighbor_sampler=callable` to bypass Rust backend)
- Yields standard PyG `Data` / `HeteroData` objects

**Unique to AetherGraph:**

- Live graph updates via `DynamicGraph` (C-tree, single-writer + lock-free
  readers)
- `feature_source="rdma://..."` for GPUDirect RDMA live features
- mmap'd CSR for instant startup on billion-node graphs
- io_uring feature loading at NVMe line rate, with NVMe passthrough where the
  namespace allows it
- Rabbit Order vertex reordering with partition-aligned batching
- Compressed graph files and f16/bf16 feature payloads
- Out-of-core feature paging under an explicit residency budget
- One shared copy of the feature matrix across worker processes
- Free-threaded (no-GIL) CPython support
- 1.2-1.5x faster sampling than PyG's pyg-lib C++ kernel

## CLI

```bash
aethergraph convert -i edges.csv -o graph.bin -n 1000000
aethergraph info graph.bin
aethergraph stats graph.bin
```

## Architecture

```
                     ┌───────────────────────────────────────┐
                     │             Python API                │
                     │  Graph, DynamicGraph, HeteroGraph     │
                     │  NeighborLoader, HeteroNeighborLoader │
                     └──────────────────┬────────────────────┘
                                        │ PyO3
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
   ┌──────────▼──────────┐   ┌──────────▼──────────┐   ┌──────────▼──────────┐
   │   aethergraph-core  │   │    aether-graph     │   │   aether-stream     │
   │  Static CSR graph   │   │  Dynamic C-tree     │   │  AF_XDP ingestion   │
   │  Homo + hetero      │   │  Lock-free readers  │   │  Seqlock features   │
   │  sampling           │   │  Arena bump-alloc   │   │  RDMA + GPUDirect   │
   │  mmap, io_uring     │   │  Snapshot isolation │   │  DLPack → PyTorch   │
   └─────────────────────┘   └─────────────────────┘   └─────────────────────┘
              │
   ┌──────────▼──────────┐
   │     aether-mem      │
   │  HugePages, mlock   │
   │  RDMA/CUDA hooks    │
   └─────────────────────┘
```

See [ARCH.md](ARCH.md) for detailed component documentation.

## Benchmarks

Measured on Apple M-series, Python 3.12, PyG 2.6.1 with pyg-lib.

### Sampling (2-hop, fanout [15, 10], batch_size=128)

| Scale      | AetherGraph | PyG (pyg-lib C++) | Speedup |
| ---------- | ----------- | ----------------- | ------- |
| 10K nodes  | 170 us      | 210 us            | 1.23x   |
| 100K nodes | 185 us      | 218 us            | 1.18x   |
| 1M nodes   | 240 us      | 341 us            | 1.43x   |
| 10M nodes  | 302 us      | 358 us            | 1.17x   |

### Dynamic Graph (C-tree, 100K nodes / 1M edges)

| Operation     | Latency | Throughput       |
| ------------- | ------- | ---------------- |
| Edge insert   | 111 ns  | 9M inserts/sec   |
| Neighbor read | 26 ns   | 38M reads/sec    |
| Edge lookup   | 16 ns   | 63M lookups/sec  |
| Degree query  | 3.4 ns  | 294M queries/sec |

### Heterogeneous Sampling (4 edge types, 2-hop, batch=128)

| Operation                  | Latency |
| -------------------------- | ------- |
| Typed multi-hop sampling   | 130 us  |
| Edge index local remapping | 3 us    |
| Full Python roundtrip      | 130 us  |

## License

MIT OR Apache-2.0
