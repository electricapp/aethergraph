# AetherGraph Integration Guide

## Quick Start

```python
from aethergraph import Graph
from aethergraph.pytorch import NeighborLoader

# Load graph
graph = Graph.load("graph.bin")
graph.load_features("features.bin")  # optional

# Train (drop-in replacement for PyG's NeighborLoader)
loader = NeighborLoader(graph, num_neighbors=[15, 10], batch_size=128)
for batch in loader:
    out = model(batch.x, batch.edge_index)
    loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
```

## PyTorch Geometric Integration

### Architecture

```
User Training Loop
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    NeighborLoader                           │
│                    (IterableDataset)                        │
│                                                             │
│   __iter__():                                               │
│       1. Shuffle input_nodes                                │
│       2. Submit batches to Rust prefetch thread             │
│       3. Yield PyG Data objects                             │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                 Rust Prefetch Thread                        │
│                                                             │
│   - Parallel k-hop sampling (Rayon)                         │
│   - Feature loading (io_uring on Linux, mmap elsewhere)     │
│   - Bounded result queue (backpressure)                     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    PyG Data Output                          │
│                                                             │
│   Data(                                                     │
│       x=[num_nodes, feat_dim],      # Node features         │
│       edge_index=[2, num_edges],    # COO edges             │
│       n_id=[num_nodes],             # Original node IDs     │
│       batch_size=N,                 # Seed node count       │
│       input_id=[batch_size],        # Seed indices          │
│       num_sampled_nodes=[hops],     # Nodes per hop         │
│       num_sampled_edges=[hops],     # Edges per hop         │
│   )                                                         │
└─────────────────────────────────────────────────────────────┘
```

### API Compatibility

AetherGraph's `NeighborLoader` is a drop-in replacement for PyG's:

```python
# PyG
from torch_geometric.loader import NeighborLoader
loader = NeighborLoader(data, num_neighbors=[15, 10], batch_size=128)

# AetherGraph (same API)
from aethergraph.pytorch import NeighborLoader
loader = NeighborLoader(graph, num_neighbors=[15, 10], batch_size=128)
```

### Supported Parameters

| Parameter                  | Description                            |
|----------------------------|----------------------------------------|
| `data`                     | Graph object                           |
| `num_neighbors`            | Neighbors per hop, e.g. `[15, 10]`     |
| `input_nodes`              | Seed nodes (default: all)              |
| `batch_size`               | Seeds per batch                        |
| `shuffle`                  | Shuffle seeds each epoch               |
| `replace`                  | Sample with replacement                |

### Batch Attributes

| Attribute                  | Description                            |
|----------------------------|----------------------------------------|
| `x`                        | Node features `[num_nodes, feat_dim]`  |
| `edge_index`               | COO edges `[2, num_edges]`             |
| `n_id`                     | Original node IDs                      |
| `batch_size`               | Number of seed nodes                   |
| `input_id`                 | Seed indices in batch                  |
| `num_sampled_nodes`        | Nodes sampled per hop                  |
| `num_sampled_edges`        | Edges sampled per hop                  |

## Performance Optimizations

### Rust Sampling Engine

| Optimization               | Description                            |
|----------------------------|----------------------------------------|
| **Floyd's O(k) Algorithm** | Sample k items in O(k) instead of O(n) |
| **WyRand PRNG**            | ~2x faster than xoshiro                |
| **CSR Format**             | O(1) neighbor access                   |
| **Parallel Sampling**      | Rayon for multi-threaded sampling      |

### I/O Pipeline

| Optimization               | Description                            |
|----------------------------|----------------------------------------|
| **Memory-Mapped Graph**    | Zero-copy graph loading                |
| **io_uring (Linux)**       | Async feature loading with SQPOLL      |
| **Prefetch Thread**        | Overlaps sampling with training        |
| **Bounded Queue**          | Backpressure prevents memory blow-up   |

### Zero-Copy Data Path

```
Rust Vec<u32> → numpy int64 → torch.from_numpy() → PyG Data
     │              │                │
     └──────────────┴────────────────┘
              No copies (shared memory)
```

## CLI Usage

```bash
# Convert edge list to binary format
aethergraph convert -i edges.txt -o graph.bin -n 1000000

# Inspect graph
aethergraph info graph.bin
aethergraph stats graph.bin
```
