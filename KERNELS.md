# AetherGraph Device-Side Roadmap

The userspace roadmap takes every technology that can be reached from a normal
process on a normal box. This document covers the other half: the work that
requires writing device code, loading a kernel module, or taking a device away
from its host driver.

The organizing principle is **access model, not difficulty**. Every item below
is sorted first by what it needs from the machine it runs on, because that — not
the code — is what determines the schedule.

---

## The two tiers

| Tier  | Needs                                                            | Where it runs                                       |
| ----- | ---------------------------------------------------------------- | --------------------------------------------------- |
| **A** | A CUDA context and nothing else                                  | Any serverless GPU container (Modal, Runpod, Colab) |
| **B** | Root on the host, `insmod`, device unbind, PCIe topology control | Bare metal with root                                |

Tier A is container-native: iterate in seconds, tear down and respawn on a
wedged GPU, run the whole suite in CI against a rented device. Tier B is not
available on any serverless platform at any price — a container gets no
`insmod`, no NVMe namespace to unbind, no DEVX handle on a physical NIC, no
control over ACS. Tier B's schedule risk is provisioning and PCIe topology, not
engineering.

Split the work on that line and start Tier A immediately; Tier B's long pole is
securing one correctly-configured box.

---

## Tier A — device code only

### K5.0 Captured launch graphs and stream-ordered allocation

Capture the per-batch gather and compaction sequence once as a `cudaGraph_t` and
replay it with a single `cudaGraphLaunch`, rather than paying the host launch
cost of every node in that sequence on every batch. Pair it with
`cudaMallocAsync`/`cudaFreeAsync` so scratch allocation joins the stream
timeline instead of forcing a host synchronization to reclaim.

A captured graph reaches the launch-cost win without the forward-progress
reasoning a persistent kernel needs, which makes it both the cheapest item here
and — as K5.1 records — the reason the launch-cost case for K5.1 is already
spent. What a graph cannot do is branch on a device-computed value; that is the
boundary between the two.

### K5.1 Persistent megakernel loader with warp specialization

One long-lived kernel replaces per-batch launches. Warps are specialized into
roles — fetch, transform, compute — communicating through a ring in VRAM, so the
gather for batch _n+1_ overlaps the aggregation of batch _n_ without stream
ping-pong or launch latency between them. The hard part is forward-progress
reasoning: a persistent kernel that blocks on an empty ring must not starve the
producer warps on the same SM.

**The launch-cost argument for this does not hold here, and it is worth being
precise about why.** A megakernel earns its complexity when overhead swamps the
work it wraps — a per-op transformer decode stack issues hundreds of launches
against tens of microseconds of arithmetic, so deleting the launches is the
whole optimization. AetherGraph's device sequence is one kernel per batch
against a ~240 µs batch, and K5.0 already captured it. The overhead ratio is
~1.01, not ~27. Sizing K5.1 against launch cost would be optimizing a bubble
that is not there.

Two things _do_ justify it, and they are what it should be built against:

- **Data-dependent control flow.** A captured graph cannot branch on a value the
  device computed. Today the sequence is static, so K5.0 covers it. The moment
  K5.2 puts sampling on-device and K1.1/K2.1 make a cache miss a device-issued
  fetch, the per-batch sequence becomes _sample → gather → miss? → fetch →
  validate → sample → …_ with the branch decided in VRAM. That is the point
  where graphs stop being expressible and persistence is the only remaining
  answer.
- **Occupancy.** A batch of a few thousand rows cannot fill a modern part from
  one operation's tile count, however well that operation is written. The fix is
  to source parallelism from several batches in flight rather than from one
  batch's row count — which is what a resident kernel with a work ring gives and
  a launch-per-batch cadence cannot.

What exists today is a correctness scaffold, not that kernel:
`kernels/persistent/persistent.cu` runs one CTA (`blockIdx.x != 0` returns) with
three warps whose queue operations all happen on `lane == 0`. It is the right
shape for reasoning about forward progress and the wrong shape for bandwidth.
The version worth building partitions roles by cluster rank across the whole
device, uses arrival barriers rather than polled flags, and carries a heartbeat
counter in L2 that the host polls — because a persistent kernel that deadlocks
on a barrier-parity mismatch presents as "still working", and is the failure
mode that costs the most days.

`TODO:` build that on top of K5.2/K1.1/K2.1, not before them.

### K5.2 Warp-cooperative C-tree sampler

Neighbor sampling executed on-GPU against the C-tree layout. One warp per seed
node; `__ballot_sync`/`__shfl_sync` drive the reservoir and alias steps.
Counter-based RNG (Philox) keyed by `(seed, layer, node)` makes the output
bit-reproducible against the CPU sampler, which is what makes the whole thing
testable — see _Verification_ below.

Removes the sampler from the host critical path entirely: seeds in, node IDs and
offsets out, no round trip.

### K5.3 PTX seqlock reader

The feature table's head/tail seqlock read from device code with correct
acquire/release semantics (`ld.acquire.sys`). Lets a kernel snapshot a
live-updating feature table without host mediation, matching the guarantee the
host-side reader already provides.

`.acquire` is PTX ISA 6.0, so this unit is the one that sets `NVRTC_ARCH_FLOOR`.
A volatile load would compile anywhere and is not an acquire load, so the unit
`#error`s below sm_70 rather than quietly weakening the property the litmus
tests check.

This is the smallest item on the list and the highest-value one to do first: it
is the memory-model proof-of-competence that everything else in the compute
plane depends on.

### K5.4 Hopper TMA/WGMMA and Blackwell tcgen05

Bulk tile movement (TMA) and tensor-core MMA (WGMMA on Hopper, `tcgen05` plus
TMEM on Blackwell) for the aggregation stage that follows the gather.

Both are **dense** engines. They accelerate the matmul after neighbor features
are materialized into a tile; they cannot express a sparse gather. Scope them to
the aggregation kernel and nowhere else.

### K5.5 GPU-side decompression

StreamVByte and Elias-Fano decode in device code, so the compressed cold tier
travels to VRAM compressed and expands there — the PCIe link carries the
compressed bytes, not the expanded ones. Blackwell's hardware decompression
engine is the vendor path for LZ-family formats.

Pairs directly with the succinct codecs already used for the version-2
compressed graph file.

**Those codecs compress the wrong half of the batch.** Split a batch's bytes
into a term that is roughly constant and a term that scales with batch size.
Adjacency is the constant-ish one; the feature payload is the one that scales,
and it is the larger of the two — a 128-seed `15×10` sample touches ~19k nodes,
so at `feature_dim = 128` the gather moves ~9.8 MB of features against ~1.5 MB
of adjacency. Compressing the constant term buys latency at small batch;
compressing the per-batch term is what moves the streaming asymptote. Only the
second one changes what the loader converges to.

So K5.5 has two halves:

- **Topology** — `kernels/decompress`, StreamVByte + Elias-Fano, landed.
- **Payload** — `kernels/quant`, `aethergraph_core::BlockScaledI8`: symmetric
  int8 with an `f32` scale per 32 elements, 1.125 bytes per feature against
  `f32`'s 4. Decode is `q as f32 * scale`, one IEEE multiply, so the device
  output is bit-identical to the CPU codec rather than close to it; a lane's
  16-code `.v4` load sits inside one block and costs one scale fetch.

The open question on the payload half is accuracy, not throughput: whether GNN
quality survives int8 features is a sweep nobody has run. `TODO:` run it, per
layer and per feature-column distribution, before this is on by default.

### K5.6 Coalescing, streaming qualifiers, and in-flight depth

A feature row is read once per batch and never reused inside the kernel, so
routing it through L1 evicts the offsets and node ids that _are_ reused.
`ld.global.nc` (reachable as `__ldg`) sends the read down the read-only path;
`.cs` marks the line evict-first. A gather over a multi-gigabyte feature table
then stops competing with the working set that has to stay resident.

One qualifier per load site, applied only where the read is provably read-only
for the kernel's lifetime.

**A qualifier on an uncoalesced access is a rounding error on a factor of
eight**, and that is the order in which these have to be fixed:

- **One warp per row, not one thread per row.** Thread-per-row puts consecutive
  lanes a `slot_size` apart, so each lane lands in its own 32-byte sector and a
  warp fetches 32 sectors where four would do. `validate_and_compact` and
  `seqlock_snapshot_rows` are warp-per-row; so are the sampler and the feature
  dequantizer. Grids are sized in warps and grid-stride over the batch, so a
  small batch spreads across SMs instead of landing on one CTA.
- **Alignment is a format decision, not a kernel one.** The payload base sits at
  `FEATURE_SLOT_HEAD_BYTES` = 16, not 8, which is what makes `ld.global.cs.v4`
  legal at all; the 64-byte slot stride absorbs the pad, so it costs no DRAM.
  `aethergraph_core::feature_slot_stride` is the one definition,
  `SeqlockValidator::validate` rejects staging that violates it, and
  `common_cuh_matches_core_layout` fails if the device constant drifts.
- **Depth comes from the bandwidth-delay product, not from reuse.** There is no
  reuse to stage for — every feature row is touched once — so shared memory
  would buy nothing but in-flight bytes, and registers buy those more cheaply
  because nothing here is shared between lanes. `AETHER_GATHER_IN_FLIGHT` = 3
  rounds of 512 B per warp is ~2.4× the ~5 KB/SM the BDP asks for on an
  A10-class part. This is the knob to turn if a profile shows the DRAM
  controllers idling mid-batch.
- **Aggregate the atomics.** `retry_count` takes one `atomicAdd` per warp per
  grid-stride run, not one per torn row.

`TODO(HARDWARE):` confirm sector efficiency and achieved bandwidth with
`ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld` before and after,
rather than inferring the win from the access pattern.

---

## Tier B — privileged host

### K1.1 GPU-initiated NVMe (BaM / GIDS model)

GPU threads submit NVMe commands directly, with no CPU in the fetch loop. The
namespace is unbound from the kernel `nvme` driver, BAR0 doorbells are mapped
into the GPU's address space via
`cudaHostRegister(..., cudaHostRegisterIoMemory)`, SQ/CQ rings live in VRAM, and
doorbells are rung with `st.relaxed.mmio.sys`.

Displaces the io_uring + cuFile cold-tier path: a cache miss becomes a device
memory access rather than a host round trip. The reference implementation is
open source, so this is adaptation rather than register-level archaeology.

**Needs:** root, a sacrificial NVMe namespace, ACS disabled or an IOMMU domain
that permits peer-to-peer, GPU and NVMe under a compatible root complex.

### K1.2 NVMe FDP / streams directives

Place hot adjacency and cold features into separate reclaim units so device
garbage collection never mixes their lifetimes. Directly targets write
amplification on the append-heavy dynamic-graph path.

### K1.3 ZNS zone-append as a lockless WAL

Zone append returns the LBA the device assigned, which means concurrent writers
need no shared offset counter — the sequence allocator becomes the drive. The
WAL's offset arbitration disappears into the storage protocol.

### K2.1 IBGDA — GPU-constructed RDMA

GPU warps build mlx5 WQEs in VRAM and ring the NIC doorbell themselves, so a
remote feature fetch never touches the host. This is the hardest item here, but
the difficulty is WQE-layout fidelity, not access: NVSHMEM ships a working
implementation to adapt from.

Completes the picture the userspace RDMA work starts — one-sided READ with no
CPU on either end.

### K2.2 GPU-terminated Ethernet

A DEVX-created QP and CQ whose rings and doorbell records live in GPU memory,
plus flow steering rules that deliver matching packets straight into VRAM. DOCA
GPUNetIO is the vendor-supported equivalent.

The steering decision happens at QP creation, before any DMA is issued — that
placement is what makes GPU delivery possible at all.

### K2.3 BlueField-3 DPA / FlexIO

Edge parsing, dedup, and CSR delta staging executed on the NIC's 16 datapath
accelerator cores, so the host receives graph deltas rather than packets.

**Needs:** a BlueField-3 specifically. Note that BlueField's programmable
datapath is DPA/FlexIO, eBPF, and DOCA Flow — it is not a P4 target.

### K3.1 Open p2pdma module

An out-of-tree module that imports VRAM as a dma-buf, validates the path with
`pci_p2pdma_distance()`, and hands peer bus addresses to consumers for use in
NVMe PRP/SGL entries.

This is the enabling substrate for K1.1 and the natural first Tier B item: it is
where the topology constraints surface, and it fails fast and legibly on a box
that cannot support the rest.

### K3.2 CXL Type-3 pooled memory

Bring a Type-3 device online through `cxl_pci` → region → `dax`/`kmem` as a
CPU-less NUMA node, then bind the cold feature tier to it using the existing
`mbind` machinery.

No NVIDIA GPU is a CXL initiator. The GPU reaches pooled memory through the
host, which makes this a capacity-tier play — a memory tier between DRAM and
NVMe — not a GPU-direct one.

### K3.3 NVLink-C2C coherence

On Grace-Hopper and Grace-Blackwell parts, CPU and GPU share a coherent address
space and the copy-centric design stops being the right one. The staging pools
and pinned-memory machinery collapse into placement hints (`cuMemAdvise`,
`cuMemPrefetchAsync`) over a single allocation.

### K4.1 sched_ext BPF scheduler

A scheduler that knows the sampler's shard-to-core mapping and the io_uring
SQPOLL thread: the poller is never preempted, gather threads stay on the
NIC-local node, and the pinning the userspace layer requests becomes a policy
the kernel enforces rather than a hint it tolerates.

Runs on any rooted Linux VM — cheap, fast to iterate, verifier-checked.

### K4.2 DAMON schemes and MGLRU

Access-frequency-driven demotion of cold feature pages, replacing the
degree-weighted heuristic in the userfaultfd pager with measured access recency.
Also runs on any rooted VM.

### K4.3 Provided buffers and deferred completion work

`IORING_REGISTER_PBUF_RING` hands the kernel a ring of landing buffers and lets
it pick one per completion, so the buffer pool stops being a lock-free queue the
userspace layer has to arbitrate. `IORING_SETUP_DEFER_TASKRUN` (with
`SINGLE_ISSUER`) confines completion task work to the ring's own reap point
rather than letting it run at arbitrary task-work boundaries.

These are the remaining two items on the io_uring surface: registered files,
registered buffers with `ReadFixed`, SQPOLL, and IOPOLL are already in the uring
layer. Like the rest of K4 they need a rooted Linux VM and nothing else.

---

## Layout

| Layer                  | Path                                                                                                                                                                                                |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tier A CUDA kernels    | [`crates/aether-stream/src/gpu/kernels/`](crates/aether-stream/src/gpu/kernels/) (`validate`, `seqlock`, `sampler`, `decompress`, `quant`, `persistent`, `tma`, `ibgda`, `coherent` + `common.cuh`) |
| GPU infra (non-kernel) | [`crates/aether-stream/src/gpu/`](crates/aether-stream/src/gpu/) (`buffer`, `pool`, `uvm`, `vmm`, `ipc`, `gdrcopy`)                                                                                 |
| Tier B device paths    | [`crates/aethergraph-core/src/internal/device/`](crates/aethergraph-core/src/internal/device/) + [`modules/aether_p2pdma/`](modules/aether_p2pdma/) + [`modules/aether_dpa/`](modules/aether_dpa/)  |
| CUDA harness           | [`crates/aether-stream/tests/kernels/`](crates/aether-stream/tests/kernels/), [`benches/kernels/`](crates/aether-stream/benches/kernels/)                                                           |
| herd7 litmus (K5.3)    | [`crates/aether-stream/litmus/k5_3/`](crates/aether-stream/litmus/k5_3/)                                                                                                                            |
| Grind entrypoint       | [`scripts/kernels-verify.sh`](scripts/kernels-verify.sh)                                                                                                                                            |

CUDA units compile via NVRTC `include_str!` under the `gpudirect` feature — no
nvcc fatbin step in `build.rs`, so the toolkit is never a build dependency and
each unit is compiled for the compute capability of the device that will run it
(`compile_for_device`), not for NVRTC's default floor.

The cost is that `cargo` never reads a `.cu` file, so three gates stand in for
it, none of which needs a GPU:

| Gate                         | Needs      | Catches                             |
| ---------------------------- | ---------- | ----------------------------------- |
| `scripts/cu-syntax-check.sh` | clang only | C++ syntax and types                |
| `nvrtc_compiles_every_unit`  | `libnvrtc` | CUDA semantics, arch-gated builtins |
| `ptxas_assembles_every_unit` | `ptxas`    | the inline PTX bodies               |

The third is not redundant. NVRTC copies `asm volatile` bodies into its output
verbatim and never assembles them, and cudarc only ever asks it for PTX — so a
typo'd mnemonic clears both earlier gates and first appears as
`CUDA_ERROR_INVALID_PTX` when the driver JITs the module on the rig. `ptxas` is
that assembler, minus the rig. None of the three says anything about whether a
kernel computes the right answer.

---

## How to grind

On any box, before touching a `.cu` file:

```bash
scripts/cu-syntax-check.sh   # clang parses every unit; no CUDA, no GPU
```

On a CUDA box (Tier A):

```bash
scripts/kernels-verify.sh
# or manually:
cargo test  -p aether-stream --features gpudirect --test kernels -- --nocapture
cargo bench -p aether-stream --features gpudirect --bench kernels
```

Optional: `compute-sanitizer --tool racecheck …` and
`herd7 -model nvidia crates/aether-stream/litmus/k5_3/*.litmus` when those tools
are on `PATH`.

On a rooted Linux VM (K4): the same script runs PBUF / policy unit tests; load
`sched_ext_bpf` / live DAMON under `TODO(HARDWARE)`.

---

## Verification

The reason device work is slow is not that iteration is slow — on Tier A it
isn't — but that a wedged NVMe controller or a silently-dropped WQE emits no
signal. The strategy is therefore to **shrink the surface that needs hardware to
be observed**, until only a few hundred bytes of it remain.

| Technique                       | Applies to               | Effect                                                       |
| ------------------------------- | ------------------------ | ------------------------------------------------------------ |
| CPU reference model + diff test | K5.2, K5.5, K1.x codecs  | Bit-exact oracle; Philox keying makes the sampler comparable |
| Pure-logic command builders     | K1.1 NVMe SQE, K2.1 WQE  | Struct layout unit-tests on any machine against the spec     |
| clang nvptx parse               | every `.cu` unit         | Syntax + types with no CUDA toolkit; `cu-syntax-check.sh`    |
| NVRTC compile, no device        | every `.cu` unit         | CUDA semantics wherever `libnvrtc` exists, GPU or not        |
| `ptxas` assemble, no device     | every inline `asm` body  | Bad opcodes, which NVRTC passes through untouched            |
| Shared-geometry drift test      | slot layout, quant block | `common.cuh` constants checked against `aethergraph-core`    |
| `compute-sanitizer`             | all of Tier A            | racecheck / initcheck / synccheck via `kernels-verify.sh`    |
| herd7 litmus tests              | K5.3, ring protocols     | Sources in `litmus/k5_3/`; run when herd7 is installed       |
| syzkaller + KASAN/KCSAN         | K3.1                     | Module fuzzed before it touches a real namespace             |
| virtme-ng / QEMU harness        | K3.1, K4.x               | Module crash-iterate without reprovisioning                  |

Building K1.1 and K2.1 as an untestable doorbell ring around a fully unit-tested
command builder is the difference between a week and a month on each.

---

## Rig matrix

| Rig                                     | Covers                                      |
| --------------------------------------- | ------------------------------------------- |
| Serverless GPU container (B200/B300)    | All of Tier A, plus K5.4 architecture gates |
| Any rooted Linux VM                     | K4.1, K4.2, K4.3                            |
| virtme-ng / QEMU with virtual NVMe      | K3.1 development loop, K1.3 zone emulation  |
| Bare metal, root, ConnectX + spare NVMe | K3.1 validation, K1.1, K1.2, K2.1, K2.2     |
| BlueField-3                             | K2.3                                        |
| Grace-Hopper / Grace-Blackwell          | K3.3                                        |
| CXL Type-3 host                         | K3.2                                        |

---

## Sequencing

1. **K5.0** — done: graph replay + mapped pinned `retry_count` (device
   fallback).
2. **K5.3** — done: PTX acquire reader + litmus sources; herd7 run is
   `TODO(HARDWARE)`.
3. **K5.6, K5.2, K5.5** — warp-per-row gather with 16-byte-aligned payloads and
   BDP in-flight depth; warp prefetch sampler; warp StreamVByte + parallel EF
   for topology and block-scaled int8 for the payload. C-tree arena and the
   Blackwell decompression engine remain `TODO(HARDWARE)`; the int8 accuracy
   sweep is `TODO:`.
4. **K5.1** — deferred on purpose. The launch-cost case for it is spent (K5.0),
   and the cases that remain — device-computed control flow, multi-batch
   occupancy — only exist once K5.2 and K1.1/K2.1 land. Today's `persistent.cu`
   is a one-CTA forward-progress scaffold, not the kernel.
5. **K4.1, K4.2, K4.3** — `SchedExtLoader` + BPF struct_ops source, DAMON sysfs
   adapter, PBUF register + `read_buffer_select`; live attach / DAMON / load
   test remain `TODO(HARDWARE)`.
6. **K3.1** — `modules/aether_p2pdma/` + userspace ioctl client; virtme-ng
   crash-iterate `TODO(HARDWARE)`.
7. **K1.1** / **K1.2** / **K1.3** — `BamController`, FDP on SQE,
   `ZoneAppendWal`; BAR/doorbell / FDP drive / ZNS CQ `TODO(HARDWARE)`.
8. **K2.2**, then **K2.1** — `DevxGpuEthPlan` + `IbgdaQueue` + GPU WQE kernel;
   DEVX/IBGDA on ConnectX `TODO(HARDWARE)`.
9. **K5.4**, **K2.3**, **K3.3**, **K3.2** — GEMV + `FlexIoHost`/`aether_dpa` +
   CXL `mbind` apply + coherent hints; ISA/BF3/Grace/CXL box `TODO(HARDWARE)`.

The single highest-leverage action is securing one bare-metal box with a
ConnectX and a spare NVMe namespace, and verifying its ACS/IOMMU topology before
any Tier B doorbell code is written. That fact determines whether the
crown-jewel items are days-hard or months-hard.

---

## Implementation status (grind pass)

| Item      | Code                                                          | Verification                                               |
| --------- | ------------------------------------------------------------- | ---------------------------------------------------------- |
| K5.0      | `kernels/validate` — CUDA graph + mapped `retry_count`        | `tests/kernels` + graph unit tests                         |
| K5.1      | `kernels/persistent` — 1-CTA 3-warp scaffold, deferred        | `persistent_drain_counts_posted_work`                      |
| K5.2      | `kernels/sampler` — warp + `ld.cs` window + Philox R          | GPU↔CPU bit-diff in `tests/kernels`                        |
| K5.3      | `kernels/seqlock` — warp-per-row snapshot + `litmus/k5_3`     | Oracle tests; herd7 `TODO(HARDWARE)`                       |
| K5.4      | `kernels/tma` smem-B + `ld.cs.v4` GEMV                        | Smoke test; TMA/WGMMA ISA `TODO(HARDWARE)`                 |
| K5.5      | `kernels/decompress` (topology) + `kernels/quant` (payload)   | GPU↔CPU oracle tests; int8 accuracy sweep `TODO:`          |
| K5.6      | warp-per-row `.cs`/`.v4` gather, 16B payload base, BDP depth  | Validate + dequant tests; `ncu` sectors `TODO(HARDWARE)`   |
| K1.1–K1.3 | `device/nvme/` BaM + FDP-on-SQE + `ZoneAppendWal`             | Layout/unit tests; BAR/ZNS/FDP `TODO(HARDWARE)`            |
| K2.1–K2.3 | `device/rdma/` IBGDA + DEVX + FlexIO + `modules/aether_dpa`   | Unit + mock DEVX; ConnectX/BF3 `TODO(HARDWARE)`            |
| K3.1–K3.3 | `modules/aether_p2pdma/` + ioctl + CXL apply + coherent hints | Unit tests; module/CXL/GH `TODO(HARDWARE)`                 |
| K4.1      | `SchedExtLoader` + `bpf/src/sched_ext_aether.c`               | Missing-object test; load on VM `TODO(HARDWARE)`           |
| K4.2      | `DamonSysfs` adapter                                          | Temp-root unit test; live DAMON `TODO(HARDWARE)`           |
| K4.3      | `register_provided_buffer_ring` + `read_buffer_select`        | Register + BUFFER_SELECT smoke; load test `TODO(HARDWARE)` |

`TODO:` marks code still to write. `TODO(HARDWARE):` marks rig verification
only.

---

## Out of scope, with reasons

Items that look like they belong on this list but do not:

| Item                                   | Why not                                                                                                                                                                                                                                                                                    |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| XDP redirect into VRAM                 | XDP runs after the NIC has DMA'd the frame into host memory. A completed DMA cannot be retargeted. The mechanism that actually delivers to VRAM is flow steering at QP creation — K2.2.                                                                                                    |
| TMA / TMEM as a gather engine          | Both are dense tile movers. Sparse neighbor gather is not expressible in either; they belong to the aggregation stage only.                                                                                                                                                                |
| CXL load/store from a CUDA kernel      | No NVIDIA GPU is a CXL initiator, and `ld.global.nc` is a cache-coherence hint, not a CXL instruction. CXL memory reaches the GPU via host staging.                                                                                                                                        |
| P4 on BlueField-3                      | BlueField's programmable datapath is DPA/FlexIO, eBPF, and DOCA Flow. P4 targets are a different class of device.                                                                                                                                                                          |
| Speculative sampling with rollback     | Under counter-based RNG the sample for a given seed set is deterministic, so next-batch work is _pre-execution_, not speculation. Build the prefetch; the misprediction and rollback machinery has nothing to do.                                                                          |
| SPDK                                   | `IORING_OP_URING_CMD` covers the userspace path, and K1.1 covers the device path. SPDK sits between them with the drawbacks of both.                                                                                                                                                       |
| HMM `migrate_vma` fault-driven tiering | Fault-driven migration is the tool for an access set you cannot predict. Sampling emits the exact node id list before the gather runs, so explicit prefetch strictly dominates it: same transfers, no fault round trip.                                                                    |
| Intel DSA / `ENQCMD` gather offload    | Sapphire Rapids and newer only, which no rig in the matrix has. The premise also requires the feature gather to be CPU-bound — currently unmeasured, and the host sampling path profiles as memory-latency bound rather than issue-bound. Revisit when a rig and a measurement both exist. |
