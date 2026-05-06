# AetherGraph Examples

## Examples

| Script | Description |
|--------|-------------|
| `01_basic_sampling.py` | Load graph, sample neighborhoods, access data |
| `02_ray_distributed.py` | Distributed sampling with Ray Data |
| `03_pytorch_geometric_training.py` | Full PyG training with GraphSAGE |
| `simple_training.py` | Minimal training loop |

## Usage

```bash
pip install aethergraph[pytorch-geometric]
python 01_basic_sampling.py
```

For Ray:
```bash
pip install aethergraph[ray]
python 02_ray_distributed.py
```
