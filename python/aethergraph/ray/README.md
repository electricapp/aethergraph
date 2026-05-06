# AetherGraph Ray Integration

Distributed graph sampling using **Replicated Topology Architecture**. By treating the graph as a static asset on local NVMe (like model weights), we eliminate the Sharding Tax - reducing sampling latency from network RTT (milliseconds) to NVMe I/O (microseconds).

## Quick Start

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
    data = collate_to_pyg(batch)  # Zero-copy to PyG Data
    out = model(data.x, data.edge_index)
```

## Why This Wins

**Traditional distributed GNN:**
```
Worker 1 ←── graph edges ──→ Worker 2    # Terabytes over network
```

**AetherGraph approach:**
```
Worker 1 ←── gradients ──→ Worker 2      # Megabytes over network
    │                          │
  NVMe                       NVMe         # Graph is local
```

Linear scaling: 100 workers = 100x throughput. No diminishing returns from network congestion.

## API

### `create_sampling_dataset`

```python
create_sampling_dataset(
    graph_path: str,           # Path to graph.bin
    num_neighbors: list[int],  # Fanout per hop, e.g. [25, 10]
    input_nodes: list | None,  # Seed nodes (default: all)
    batch_size: int = 1024,
    replace: bool = False,
    features_path: str | None,
    parallelism: int | None,   # Default: num CPUs
) -> ray.data.Dataset
```

### `collate_to_pyg`

Zero-copy conversion from Ray Arrow batch to PyG Data:

```python
from aethergraph.ray import collate_to_pyg

for batch in dataset.iter_batches(batch_format="pyarrow"):
    data = collate_to_pyg(batch)
    # data.x, data.edge_index, data.n_id, data.e_id ready for GPU
```

### `AetherGraphDatasource`

Low-level datasource for custom pipelines:

```python
ds = ray.data.read_datasource(
    AetherGraphDatasource(
        graph_path="graph.bin",
        num_neighbors=[25, 10],
        batch_size=1024,
    )
)
```

## With Ray Train

```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from aethergraph.ray import create_sampling_dataset, collate_to_pyg

def train_fn():
    import torch
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

## Architecture

```
Ray Head
    │
    ├─► Worker 1: mmap(graph.bin) → NeighborLoader → Arrow → GPU
    │       └── local NVMe (zero network for sampling)
    │
    └─► Worker 2: mmap(graph.bin) → NeighborLoader → Arrow → GPU
            └── local NVMe (zero network for sampling)

    Only gradients cross the network (via NCCL/DDP)
```

## Requirements

```
pip install aethergraph[ray]
```
