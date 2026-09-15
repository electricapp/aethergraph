# AetherGraph Test Plan

Outstanding work only. Delete rows from these tables as they land.

## Status

| ID  | What                           | Code written? | Blocker to run              |
| --- | ------------------------------ | ------------- | --------------------------- |
| H2  | NVMe passthrough gather (live) | Yes           | A drive exposing `/dev/ng*` |
| H3  | NUMA placement chooses a node  | Yes           | A 2-socket host             |
| H6  | Native InfiniBand addressing   | Yes           | IB fabric + subnet manager  |

Closed on 2026-09-02 (Modal T4:2 + Lambda `gpu_1x_a10`): **R6**, **H1** (seqlock
validate + snapshot reader), **H4**, **H5** (tier=`deferred`; async io_uring vs
pread on the A10 root volume), **H7**, **H8** (THP=`madvise`, CSR advise path
exercised), **P1** (Rabbit faster on both arms across 3 runs).

Still open inside H1: GDRCopy BAR1 stamping and UVM `prefetch_rows` need a live
RDMA feature server + `gdrdrv` (not exercised on this pass). CUDA graph capture
soft-falls back to eager launch on the A10; kernel correctness holds.

Lambda A10 is virtio root (`vda`), not NVMe — no `/dev/ng*`, so H2 live
`read_batch` remains open (MDTS / layout unit tests already pass). Single NUMA
node — H3 inert, same as CI.

---

## Hardware verification debt

Each row below is compiled, linted, and (where the logic is portable) unit
tested — but the behaviour that motivates it has never executed. Kept explicit
because every one of these is a path where the code can look finished and do
nothing: a fallback that silently degrades, a placement call that is inert on
one socket, a probe that fires without its arguments.

| ID  | Never executed                                                                                   | Why CI cannot cover it                                                                                           | Rig                        |
| --- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- | -------------------------- |
| H2  | `NvmeReader::read_batch` submission and completion                                               | Runners have no NVMe character device, so `NvmeReader::open_for` returns `None` and the gather takes the fs path | Drive with `/dev/ng*`      |
| H3  | `interleave_region` spreading pages, `pin_current_thread` binding a worker to one socket's cores | Runners are single-node: `nodes_online().len() < 2`, so both calls short-circuit before the syscall              | 2-socket host              |
| H6  | A QP reaching a peer over LID routing, and `LinkLayer::InfiniBand` being taken at all            | Every fabric available is Ethernet-link-layer — SoftRoCE, ConnectX-6 RoCE, EFA — so the IB branch never runs     | IB fabric + subnet manager |

**H3 is not covered by the Lambda A10** — that instance is single-socket, so it
leaves NUMA placement inert exactly as CI does. A separate 2-socket box is the
only thing that shows `interleave_region` choosing between nodes.

H6 is worth stating precisely, because "uses libibverbs" reads as "supports
InfiniBand" and does not mean it. The verbs API is common to IB, RoCE, iWARP,
and EFA, and `/sys/class/infiniband/` is Linux's name for every RDMA device
including pure Ethernet ones. What runs here is RoCE and EFA/SRD. The IB
addressing path is written and its decision logic is unit tested — same subnet
routes on the LID, crossing subnets adds a GRH — but no IB fabric has executed
it.

### P1 result (2026-09-02, Lambda A10, quiet)

`benches/sampling_locality.rs`, three consecutive runs. Rabbit faster on both
arms every time; one-hop intervals stay apart.

| Arm        | shuffled (median)      | rabbit (median)        |
| ---------- | ---------------------- | ---------------------- |
| 25 (1-hop) | ~68.6 / 68.7 / 69.2 µs | ~63.0 / 63.2 / 63.7 µs |
| 15×10      | ~944 / 928 / 921 µs    | ~896 / 891 / 876 µs    |

Two things to hold separate when reading it. The benchmark draws seeds
uniformly, which is the ordering-unfriendly case — at the first hop, consecutive
frontier entries are unrelated whatever the numbering, so only later hops have
locality to win back. And `partition_aligned_batches` would show a larger
number, but it samples a denser subgraph: part of that gain is doing less work,
not touching less memory, and it changes batch gradient statistics. Those are
two claims, not one.

### H5 result (2026-09-02, Lambda A10)

- Ladder: `io_uring setup tier reached: deferred` (SINGLE_ISSUER +
  DEFER_TASKRUN).
- `async_io_benchmark` on the root volume (virtio, not NVMe): async io_uring
  ~408 µs vs sync pread ~429 µs for 1k nodes — small win, not a cold-NVMe claim.
  Per-rung forced deltas still unmeasured.

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
| Lambda A10 root disk is virtio, not NVMe          | No `/dev/ng*` — H2 live passthrough cannot run; use a box with a real NVMe namespace char device                                  |
| Seqlock feature bytes start at offset 8           | `ld.global.*.v4` needs 16-byte alignment — validate kernel uses scalar `.cs` loads                                                |

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
