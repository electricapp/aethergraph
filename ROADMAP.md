# AetherGraph Test Plan

Outstanding work only. Delete rows from these tables as they land.

## Status

| ID  | What                    | Code written? | Blocker to run                        |
| --- | ----------------------- | ------------- | ------------------------------------- |
| R6  | T4.3 Ray Data multi-GPU | Yes           | ≥2 GPUs (Modal `gpu="T4:2"` suffices) |

All test code is written and gated. What's left is _running_ it on the right
hardware.

---

## R6 — T4.3 Ray Data multi-GPU

Skips unless `torch.cuda.device_count() >= 2`. Builds a sampling dataset at
parallelism = GPU count and asserts aggregate batches/sec is at least 1.5× the
single-worker baseline.

| File                                 |
| ------------------------------------ |
| `python/tests/test_ray_multi_gpu.py` |

---

## Performance structurals

Remaining larger redesigns; land each with its own tests.

| Area                   | Change                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- |
| FeatureCache NVMe tier | Route the slot file through the io_uring lane with `O_DIRECT` (it uses positional reads on the blocking pool today) |

---

## CI surrogates (catch drift without hardware)

| Job                      | What it does                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `markdown-format`        | `bunx prettier --check '**/*.md'`                                                                                              |
| `gpudirect-check`        | Compile-only `cargo check` of the rdma + gpudirect path on an `nvidia/cuda:12.5.0-devel-ubuntu22.04` container — no GPU needed |
| `rust` matrix (existing) | macOS + Ubuntu defaults; Ubuntu + `rdma` feature                                                                               |

---

## Environment gotchas

| Pitfall                                           | What to do                                                                                                                      |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| AL2023 kernel 6.18 does not ship `rdma_rxe`       | Use Ubuntu 22.04/24.04, or AL2023 kernel 6.1 + `kernel-modules-extra`                                                           |
| SoftRoCE on `lo` does not work                    | Bind `rxe` to a real ethernet device — loopback has no MAC for ARP/GID resolution                                               |
| GID 0 is link-local IPv6 on RoCE                  | Pick the IPv4-mapped GID (typically index 1; confirm via `show_gids`). `RdmaContext::open` requires an explicit `gid_index`     |
| `ulimit -l unlimited` required                    | Otherwise `ibv_reg_mr` / `mlock` fail                                                                                           |
| EFA needs SG self-reference on ingress AND egress | A generic egress `0.0.0.0/0` silently drops EFA                                                                                 |
| `cargo test -p ... -p ...` shares compile         | One crate's build failure cancels sibling test runs mid-build                                                                   |
| `xdp_bpf` feature needs `clang` + `libbpf-dev`    | `build.rs` invokes clang with `--target=bpf` to compile the redirect program                                                    |
| `ibv_reg_mr` on CUDA VAs EFAULTs in VMs           | nvidia-peermem needs bare metal; `reg_mr_cuda` falls back to dma-buf (`ibv_reg_dmabuf_mr`, needs rdma-core ≥ v34, driver ≥ 515) |
| auditwheel-bundled libibverbs sees 0 devices      | Bundled lib can't load the mlx5 provider plugin — build the extension with `maturin develop` so it links system libibverbs      |
| torch wheel CUDA flavor must match the driver     | e.g. driver 570 = CUDA 12.8 → install `+cu128` wheels from `download.pytorch.org/whl/cu128`                                     |

---

## Setup

```bash
# Tier 1 (SoftRoCE + veth + BPF)
sudo apt-get install -y rdma-core libibverbs-dev iproute2 clang libbpf-dev
sudo modprobe rdma_rxe
PRIMARY_NIC=$(ip -br link | awk '/UP/ && $1 !~ /^(lo|veth)/ {print $1; exit}')
sudo rdma link add rxe0 type rxe netdev "$PRIMARY_NIC"
sudo ip link add veth-tx type veth peer name veth-rx
sudo ip link set veth-tx up && sudo ip link set veth-rx up
ulimit -l unlimited

# Tier 2 adds:
sudo apt-get install -y nvidia-driver cuda-toolkit
nvidia-smi

# Tier 3 adds:
sudo modprobe nvidia-peermem
dmesg | grep peermem  # expect "nvidia-peermem registered"
```

---

## Competitive landscape

The real comparators are not vanilla PyG. When pitching or writing related work,
benchmark and position against these:

### Production systems

- **DGL‑GraphBolt** — closest competitor. Has an explicit `OnDiskDataset` with
  mmap'd CSC, `gpu_cached_feature` for partial caching, async prefetching, and a
  sampler designed around the disk path. This is "AetherGraph but in the DGL
  ecosystem and with two years of head start." If we can't point at concrete
  deltas (io_uring vs their `pread`+threadpool, Rabbit Order built into the
  format, dynamic ingest, Rust core), a reviewer will ask why we didn't just
  contribute upstream.
- **WholeGraph (cuGraph)** — NVIDIA's distributed shared‑memory store. Stripes
  features across host RAM on multi‑GPU boxes with NVLink/RDMA. Doesn't really
  do NVMe spill, but owns the GPUDirect / RDMA feature‑serving story we're
  claiming as a differentiator. If we say "GPUDirect RDMA, <5µs," the question
  back is "how is this not just WholeGraph?" Clean answer: single‑machine NVMe
  focus, dynamic ingest, billion‑edge tier where WholeGraph runs out of host
  RAM.
- **cuGraph‑PyG** — GPU sampler. Different niche (graph in GPU/host RAM, not on
  disk), but it's what people reach for when they say "PyG NeighborLoader is too
  slow." When we publish the 240 µs/batch number, the comparison they care about
  is "vs cuGraph‑PyG on the same hardware," not "vs PyG CPU sampler." 1.4× over
  CPU PyG is fine; cuGraph‑PyG can be 10×+ on small graphs.
- **Kùzu (PyG remote backend)** — disk‑based columnar graph DB with an official
  PyG `FeatureStore`/`GraphStore` integration. Already covers "static graph on
  NVMe, stream into PyG NeighborLoader." Differentiators are real but specific:
  Kùzu pays query‑engine overhead per fetch, no GPU‑direct path, no dynamic
  ingest tuned for streaming. Lead with those.

### Academic systems for the related‑work section

These will appear in our related‑work section whether we like it or not:

- **MariusGNN / Marius++** (Mohoney et al.) — disk‑based single‑machine
  billion‑edge GNN training. Same pitch.
- **Ginex** (Park et al., VLDB '22) — SSD‑resident GNN training with explicit
  feature cache eviction policy. Has the "graphs don't fit, NVMe is fast enough"
  thesis we're quoting.
- **GIDS** (Park et al.) — GPU‑initiated direct storage, exactly the
  io_uring/SPDK‑from‑GPU path. If we don't cite this and explain how AetherGraph
  differs, a PC reviewer will reject on novelty.
- **BaM** (NVIDIA) — same direction, GPU as storage initiator.
- **Legion / P3 / DistDGL** — distributed training comparators if we ever claim
  scale.
