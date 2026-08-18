# AetherGraph Python

Python bindings for AetherGraph. See the [main README](../README.md) for full
documentation.

## Features

- **Zero-copy I/O**: Memory-mapped graphs, Arrow integration
- **io_uring async I/O**: Parallel NVMe reads, with NVMe passthrough where the
  namespace allows it (Linux)
- **PyTorch Geometric compatible**: Drop-in NeighborLoader replacement
- **Ray Data integration**: Distributed sampling with Replicated Topology
- **Compact at rest**: Compressed graph files, f16/bf16 feature payloads
- **Feature memory**: One shared copy across worker processes, or demand paging
  under a fixed residency budget (Linux)
- **Free-threaded CPython**: A `cp314t` wheel that does not re-enable the GIL

## Installation

```bash
pip install aethergraph

# Optional dependencies
pip install aethergraph[torch]              # PyTorch support
pip install aethergraph[ray]                # Ray Data support
pip install aethergraph[pytorch-geometric]  # Full PyG integration
```

## Quick Start

```python
from aethergraph import Graph
from aethergraph.pytorch import NeighborLoader

graph = Graph.load("graph.bin")
graph.load_features("features.bin")

loader = NeighborLoader(
    graph,
    num_neighbors=[25, 10],
    batch_size=1024,
)

for data in loader:
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out, data.y)
```

## Ray Distributed Training

AetherGraph uses **Replicated Topology Architecture**: the graph is replicated
on each worker's local NVMe, not partitioned across the network. This eliminates
the Sharding Tax.

```
Traditional:  Worker 1 ←── graph edges ──→ Worker 2   # Terabytes over network
AetherGraph:  Worker 1 ←── gradients ──→ Worker 2     # Megabytes over network
                  │                          │
                NVMe                       NVMe        # Graph is local
```

### Basic Usage

```python
import ray
from aethergraph.ray import create_sampling_dataset, collate_to_pyg

ray.init()

dataset = create_sampling_dataset(
    graph_path="graph.bin",
    num_neighbors=[25, 10],
    batch_size=1024,
)

for batch in dataset.iter_batches(batch_format="pyarrow"):
    data = collate_to_pyg(batch)
    out = model(data.x, data.edge_index)
```

### With Ray Train (Multi-GPU)

```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from aethergraph.ray import create_sampling_dataset, collate_to_pyg


def train_fn():
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = DDP(MyGNN())
    dataset = create_sampling_dataset("graph.bin", [25, 10])

    for batch in dataset.iter_batches(batch_format="pyarrow"):
        data = collate_to_pyg(batch)
        loss = model(data.x, data.edge_index)
        loss.backward()  # DDP syncs gradients here


trainer = TorchTrainer(
    train_fn,
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
)
trainer.fit()
```

## CLI

### `aethergraph convert`

Convert edge list files to binary format:

```bash
aethergraph convert -i edges.csv -o graph.bin -n 1000000
```

Options:

- `-i, --input`: Input edge list file (TSV/CSV, auto-detected)
- `-o, --output`: Output binary graph file
- `-n, --num-nodes`: Number of nodes in the graph
- `-d, --delimiter`: Delimiter (auto-detect if not set)
- `--skip-lines`: Skip first N lines (for headers)
- `-v, --verbose`: Verbose output (-v, -vv for more detail)
- `-q, --quiet`: Suppress non-error output

### `aethergraph info`

Display basic graph information:

```bash
aethergraph info graph.bin
```

Output:

```
Graph Information:
  Nodes: 1,000,000
  Edges: 5,000,000
  Avg degree: 5.00
  File size: 24.00 MB
```

### `aethergraph stats`

Detailed statistics including degree distribution:

```bash
aethergraph stats graph.bin
```

Output:

```
Graph Statistics:
  Nodes: 1,000,000
  Edges: 5,000,000
  Max degree: 1,234
  Avg degree: 5.00

File Information:
  Size: 24.00 MB
  Bytes per node: 24.00
  Bytes per edge: 4.80

Degree Distribution:
  50th percentile: 3
  90th percentile: 12
  99th percentile: 45
```

## Python API

```python
from aethergraph import Graph
import numpy as np

# Create from edges
src = np.array([0, 0, 1, 2], dtype=np.uint32)
dst = np.array([1, 2, 2, 3], dtype=np.uint32)
graph = Graph.from_edges(4, src, dst)
graph.save("graph.bin")

# Load existing graph
graph = Graph.load("graph.bin")

# Compressed at rest — load() detects the format
graph.save("graph.compressed.bin", compressed=True)
```

```python
from aethergraph import FeatureStore, SharedFeatureStore, save_features

# Half the file, decoded on the way out
save_features("features.bin", x, dtype="bf16")  # "f32" | "f16" | "bf16"

store = FeatureStore.load("features.bin")
store.get_batch(nodes)  # [len(nodes), feature_dim]

# One copy of the matrix for many workers (Linux)
owner = SharedFeatureStore.publish("features.bin")
owner.serve("/tmp/ag.sock")
worker = SharedFeatureStore.attach("/tmp/ag.sock")
```

Platform-gated classes are absent rather than broken where they do not apply.
Check before use:

```python
from aethergraph import _core

_core.HAS_GPUDIRECT, _core.HAS_NUMA, _core.HAS_SHARED_STORE, _core.HAS_PERF_COUNTERS
```

## License

Dual-licensed under MIT OR Apache-2.0.
