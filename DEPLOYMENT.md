# Deployment tuning

Placement and build guidance for peak throughput on real training hosts.
Everything here is optional — the defaults are correct, just not always placed
optimally on multi-socket machines.

## NUMA placement

On a 2-socket trainer the sampler threads, the io_uring SQPOLL kernel thread,
and the feature/graph mappings should all live on the socket that owns the NVMe
drive and the NIC. Cross-socket traffic on the feature gather path costs 1.5-2x
latency per miss.

Find the right socket:

```bash
cat /sys/class/nvme/nvme0/device/numa_node
cat /sys/class/net/eth0/device/numa_node
```

Pin the training process (samplers inherit the mask):

```bash
numactl --cpunodebind=0 --membind=0 python train.py
```

- `--membind` places the CSR and feature `mmap` pages on the local node. If the
  feature file is larger than one node's RAM, prefer `--interleave=all` for the
  mapping-heavy process instead — remote-hit latency beats reclaim thrash.
- The io_uring SQPOLL poller is pinned via `AETHERGRAPH_SQPOLL_CPU=<cpu>`; pick
  a core on the NVMe's socket that the sampler pool does not use.
- `NeighborLoader(num_workers=N)` sizes the Rust sampler pool. Leave at least
  the SQPOLL core plus one core for the feature-loader thread free of samplers.
- Lock pages for RDMA (`ulimit -l unlimited`) before process start; GPU and NIC
  should share a PCIe root complex for GPUDirect paths (check
  `nvidia-smi topo -m`).

## Wheel variants

Baseline wheels are portable `x86-64` (SSE2), built by CI's `wheel x86-64` job
as the `aethergraph-wheel-x86-64` artifact. The `wheel x86-64-v3` job builds an
`x86-64-v3` wheel (AVX2/FMA/BMI2, `aethergraph-wheel-x86-64-v3` artifact) — use
it on any host from roughly 2015 onward:

```bash
pip install aethergraph --no-index --find-links <artifact-dir>
```

Half-precision decode dispatches at runtime in either wheel — AVX-512 then F16C
for f16, AVX2 for bf16 — so a v3 wheel is not required to get the vector path.
What v3 adds is compile-time vectorization of every other loop. Building from
source with `RUSTFLAGS="-C target-cpu=native" maturin develop --release` is the
ceiling.

A separate free-threaded `cp314t` wheel is built without abi3, since the
free-threaded ABI is not part of the stable ABI. The extension declares
`gil_used = false`, so importing it on a no-GIL build does not re-enable the
GIL: every class is `Send` and guards its state with Rust locks. On such a build
`num_workers` scales with threads rather than processes, which removes the
per-process copy of the feature matrix without needing the shared store below.

## Huge pages

Both the mmap paths and the in-memory CSR arrays request `MADV_HUGEPAGE`
themselves; make sure transparent huge pages are at least `madvise`:

```bash
cat /sys/kernel/mm/transparent_hugepage/enabled   # want madvise or always
```

At `never` the request is silently ignored and TB-scale random gathers stay
dTLB-bound.

## Feature memory across workers

Each worker process mapping its own copy of the feature matrix is usually the
largest avoidable line in a trainer's memory budget. Two ways out, both Linux:

**One shared copy.** The parent seals the matrix into a memfd and serves the
descriptor; each worker maps the same physical pages read-only. N workers then
cost one copy.

```python
owner = SharedFeatureStore.publish("features.bin")
owner.serve("/tmp/ag.sock")            # parent
store = SharedFeatureStore.attach("/tmp/ag.sock")   # each worker
```

**Bounded residency.** When the matrix exceeds RAM outright, page it in under an
explicit budget instead of letting the kernel decide what to keep.

```python
store = FeatureStore.load_paged("features.bin", budget_pages=1 << 20,
                                degrees=graph.degrees())
```

This needs userfaultfd available to an unprivileged process:

```bash
sysctl vm.unprivileged_userfaultfd    # want 1, else grant CAP_SYS_PTRACE
```

It raises rather than degrading silently, so a deployment that expects paging
finds out at open time; `FeatureStore.load` is the fallback.
