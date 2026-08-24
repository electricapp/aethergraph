# AetherGraph Test Plan

Outstanding work only. Delete rows from these tables as they land.

## Status

| ID  | What                                  | Code written? | Blocker to run                            |
| --- | ------------------------------------- | ------------- | ----------------------------------------- |
| R6  | T4.3 Ray Data multi-GPU               | Yes           | ≥2 GPUs (Modal `gpu="T4:2"` suffices)     |
| H1  | GPU orchestration paths               | Yes           | Any CUDA GPU                              |
| H2  | NVMe passthrough gather               | Yes           | A drive exposing `/dev/ng*`               |
| H3  | NUMA placement chooses a node         | Yes           | A 2-socket host                           |
| H4  | USDT probe arguments                  | Yes           | Linux + `bpftrace`                        |
| H5  | io_uring setup wins                   | Bench only    | NVMe host under load                      |
| H6  | Native InfiniBand addressing          | Yes           | IB fabric + subnet manager                |
| H7  | AVX-512 / F16C f16 upcast             | Yes           | An x86-64 host with AVX-512               |
| H8  | Huge-page backing for in-memory CSR   | Yes           | Linux with `transparent_hugepage=madvise` |
| P1  | Whether node ordering speeds sampling | Bench only    | A quiet host with headroom to spare       |

All test code is written and gated. What's left is _running_ it on the right
hardware.

---

## Hardware verification debt

Each row below is compiled, linted, and (where the logic is portable) unit
tested — but the behaviour that motivates it has never executed. Kept explicit
because every one of these is a path where the code can look finished and do
nothing: a fallback that silently degrades, a placement call that is inert on
one socket, a probe that fires without its arguments.

| ID  | Never executed                                                                                           | Why CI cannot cover it                                                                                           | Rig                         |
| --- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------- |
| H1  | `gpu/gdrcopy.rs` BAR1 stores, `gpu/kernel.rs` seqlock validation, the CUDA half of `gpu/uvm.rs` prefetch | `gpudirect-check` runs `cargo check` in a CUDA container with no device — type-checked, never run                | Any CUDA GPU                |
| H2  | `NvmeReader::read_batch` submission and completion; MDTS rejection of an oversized command               | Runners have no NVMe character device, so `NvmeReader::open_for` returns `None` and the gather takes the fs path | Drive with `/dev/ng*`       |
| H3  | `interleave_region` spreading pages, `pin_current_thread` binding a worker to one socket's cores         | Runners are single-node: `nodes_online().len() < 2`, so both calls short-circuit before the syscall              | 2-socket host               |
| H4  | A tracer reading `arg1`…`arg4` off a probe                                                               | CI asserts the ELF note carries descriptors; nothing attaches to confirm a consumer resolves them                | Linux + `bpftrace`          |
| H5  | Whether `DEFER_TASKRUN` and the coalesced UVM prefetch are actually faster, not merely selected          | The tier assertion proves the setup was chosen; it says nothing about throughput                                 | NVMe host, GPU host         |
| H6  | A QP reaching a peer over LID routing, and `LinkLayer::InfiniBand` being taken at all                    | Every fabric available is Ethernet-link-layer — SoftRoCE, ConnectX-6 RoCE, EFA — so the IB branch never runs     | IB fabric + subnet manager  |
| H7  | `f16_le_to_f32_avx512` and `f16_le_to_f32_f16c`; the AVX2 bf16 kernel                                    | Development is on aarch64, where those blocks are `cfg`'d out entirely and NEON is the only path compiled        | x86-64 host with AVX-512    |
| H8  | Whether `MADV_HUGEPAGE` on the in-memory CSR arrays changes anything                                     | `advise_hugepage` compiles to a no-op off Linux, and is redundant under `transparent_hugepage=always`            | Linux, THP set to `madvise` |
| P1  | Whether Rabbit ordering makes sampling faster at all                                                     | Not a hardware gap — a noise-floor one; see below                                                                | Quiet host, ≥32 GB          |

**H3 is not covered by the Lambda A10** — that instance is single-socket, so it
exercises H1 and H2 but leaves NUMA placement inert exactly as CI does. A
separate 2-socket box is the only thing that shows `interleave_region` choosing
between nodes.

H6 is worth stating precisely, because "uses libibverbs" reads as "supports
InfiniBand" and does not mean it. The verbs API is common to IB, RoCE, iWARP,
and EFA, and `/sys/class/infiniband/` is Linux's name for every RDMA device
including pure Ethernet ones. What runs here is RoCE and EFA/SRD. The IB
addressing path is written and its decision logic is unit tested — same subnet
routes on the LID, crossing subnets adds a GRH — but no IB fabric has executed
it.

For H5, the numbers worth capturing are per-tier: run the feature gather with
the ring forced down each rung (`UringHandle::tier()` reports which one took
effect) and with prefetch coalescing disabled, so the claim is a measured delta
rather than a plausible mechanism.

P1 is a measurement gap, not a hardware one, and it is open in both directions —
no result yet shows reordering helping _or_ not helping.
`benches/sampling_locality.rs` is the instrument: two isomorphic R-MAT graphs,
one randomly permuted and one Rabbit-reordered, asserted to emit the same edge
count and visit the same node count before it reports, so a timing gap can only
be layout. Its one-hop arm reproduces to ~1%; its multi-hop arm does not, and
has returned differences of both signs across runs of the same binary. That arm
holds hundreds of thousands of output elements and dedup slots, which makes it
sensitive to memory pressure and to competing load. Read it only where repeated
runs agree on the sign and their intervals stay apart.

Two things to hold separate when it is run. The benchmark draws seeds uniformly,
which is the ordering-unfriendly case — at the first hop, consecutive frontier
entries are unrelated whatever the numbering, so only later hops have locality
to win back. And `partition_aligned_batches` would show a larger number, but it
samples a denser subgraph: part of that gain is doing less work, not touching
less memory, and it changes batch gradient statistics. Those are two claims, not
one.

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

_(none outstanding — FeatureCache NVMe spill now gathers through io_uring with
O_DIRECT when the padded slot stride allows.)_

---

## CI surrogates (catch drift without hardware)

| Job                      | What it does                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `markdown-format`        | `bunx prettier --check '**/*.md'`                                                                                              |
| `gpudirect-check`        | Compile-only `cargo check` of the rdma + gpudirect path on an `nvidia/cuda:12.5.0-devel-ubuntu22.04` container — no GPU needed |
| `rust` matrix (existing) | macOS + Ubuntu defaults; Ubuntu + `rdma` feature                                                                               |
| `numa placement`         | Exercises the mbind/affinity syscalls against node 0; the choice _between_ nodes stays untested (H3)                           |
| `perf counters`          | Reads `.note.stapsdt` back out of the build and requires at least one probe to declare arguments                               |

Two surrogates guard against a silent downgrade rather than a failure, which is
the failure mode these paths actually have:

- `UringHandle::tier()` names the setup rung the ring reached, and a test
  asserts the ladder climbs as high as the running kernel allows — so landing on
  a lesser rung is a test failure, not a benchmark that quietly fails to
  improve.
- The USDT check requires an argument descriptor, not just a probe name. A probe
  carrying no arguments still appears by name, which is how an empty descriptor
  survived being "checked".

---

## Environment gotchas

| Pitfall                                           | What to do                                                                                                                        |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| AL2023 kernel 6.18 does not ship `rdma_rxe`       | Use Ubuntu 22.04/24.04, or AL2023 kernel 6.1 + `kernel-modules-extra`                                                             |
| SoftRoCE on `lo` does not work                    | Bind `rxe` to a real ethernet device — loopback has no MAC for ARP/GID resolution                                                 |
| GID 0 is link-local IPv6 on RoCE                  | Pick the IPv4-mapped GID (typically index 1; confirm via `show_gids`). `RdmaContext::open` requires an explicit `gid_index`       |
| `ulimit -l unlimited` required                    | Otherwise pinned `ibv_reg_mr` / `mlock` fail. `reg_feature_mr` / `check_memlock_for` fail loud with this hint before the HCA call |
| EFA needs SG self-reference on ingress AND egress | A generic egress `0.0.0.0/0` silently drops EFA                                                                                   |
| `cargo test -p ... -p ...` shares compile         | One crate's build failure cancels sibling test runs mid-build                                                                     |
| `xdp_bpf` feature needs `clang` + `libbpf-dev`    | `build.rs` invokes clang with `--target=bpf` to compile the redirect program                                                      |
| `ibv_reg_mr` on CUDA VAs EFAULTs in VMs           | nvidia-peermem needs bare metal; `reg_mr_cuda` falls back to dma-buf and names both failures + driver/rdma-core requirements      |
| auditwheel-bundled libibverbs sees 0 devices      | Bundled lib can't load the mlx5 provider plugin — build the extension with `maturin develop` so it links system libibverbs        |
| torch wheel CUDA flavor must match the driver     | e.g. driver 570 = CUDA 12.8 → install `+cu128` wheels from `download.pytorch.org/whl/cu128`                                       |

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
