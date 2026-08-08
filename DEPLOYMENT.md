# Deployment tuning

Placement and build guidance for peak throughput on real training hosts.
Everything here is optional — the defaults are correct, just not always
placed optimally on multi-socket machines.

## NUMA placement

On a 2-socket trainer the sampler threads, the io_uring SQPOLL kernel
thread, and the feature/graph mappings should all live on the socket that
owns the NVMe drive and the NIC. Cross-socket traffic on the feature
gather path costs 1.5-2x latency per miss.

Find the right socket:

```bash
cat /sys/class/nvme/nvme0/device/numa_node
cat /sys/class/net/eth0/device/numa_node
```

Pin the training process (samplers inherit the mask):

```bash
numactl --cpunodebind=0 --membind=0 python train.py
```

- `--membind` places the CSR and feature `mmap` pages on the local node.
  If the feature file is larger than one node's RAM, prefer
  `--interleave=all` for the mapping-heavy process instead — remote-hit
  latency beats reclaim thrash.
- The io_uring SQPOLL poller is pinned via `AETHERGRAPH_SQPOLL_CPU=<cpu>`;
  pick a core on the NVMe's socket that the sampler pool does not use.
- `NeighborLoader(num_workers=N)` sizes the Rust sampler pool. Leave at
  least the SQPOLL core plus one core for the feature-loader thread free
  of samplers.
- Lock pages for RDMA (`ulimit -l unlimited`) before process start; GPU
  and NIC should share a PCIe root complex for GPUDirect paths (check
  `nvidia-smi topo -m`).

## Wheel variants

Baseline wheels are portable `x86-64` (SSE2). CI also builds an
`x86-64-v3` wheel (AVX2/FMA/BMI2, `wheel x86-64-v3` job artifact) — use it
on any host from roughly 2015 onward:

```bash
pip install aethergraph --no-index --find-links <artifact-dir>
```

Hot f16 decode paths dispatch on F16C at runtime in either wheel; the v3
wheel additionally vectorizes every other loop at compile time. Building
from source with `RUSTFLAGS="-C target-cpu=native" maturin develop
--release` is the ceiling.

## Huge pages

The mmap paths request `MADV_HUGEPAGE` themselves; make sure transparent
huge pages are at least `madvise`:

```bash
cat /sys/kernel/mm/transparent_hugepage/enabled   # want madvise or always
```
