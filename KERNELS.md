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

K5.1 subsumes this — a persistent kernel has no per-batch launch left to
amortize — but a captured graph reaches most of the same win without the
forward-progress reasoning. That makes it both the cheapest item here and the
baseline K5.1 has to beat to justify its complexity.

### K5.1 Persistent-kernel loader with warp specialization

One long-lived kernel replaces per-batch launches. Warps are specialized into
roles — fetch, transform, compute — communicating through a ring in VRAM, so the
gather for batch _n+1_ overlaps the aggregation of batch _n_ without stream
ping-pong or launch latency between them.

Displaces the per-batch H2D/launch cadence in the loader path. The hard part is
forward-progress reasoning: a persistent kernel that blocks on an empty ring
must not starve the producer warps on the same SM.

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
acquire/release semantics (`ld.acquire.sys`, `fence.acq_rel.sys`). Lets a kernel
snapshot a live-updating feature table without host mediation, matching the
guarantee the host-side reader already provides.

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

### K5.6 Non-coherent and streaming loads for the sparse gather

A feature row is read once per batch and never reused inside the kernel, so
routing it through L1 evicts the offsets and node ids that _are_ reused.
`ld.global.nc` (reachable as `__ldg`) sends the read down the read-only path;
`.cs` marks the line evict-first. A gather over a multi-gigabyte feature table
then stops competing with the working set that has to stay resident.

One qualifier per load site, applied only where the read is provably read-only
for the kernel's lifetime.

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

## Verification

The reason device work is slow is not that iteration is slow — on Tier A it
isn't — but that a wedged NVMe controller or a silently-dropped WQE emits no
signal. The strategy is therefore to **shrink the surface that needs hardware to
be observed**, until only a few hundred bytes of it remain.

| Technique                       | Applies to              | Effect                                                       |
| ------------------------------- | ----------------------- | ------------------------------------------------------------ |
| CPU reference model + diff test | K5.2, K5.5, K1.x codecs | Bit-exact oracle; Philox keying makes the sampler comparable |
| Pure-logic command builders     | K1.1 NVMe SQE, K2.1 WQE | Struct layout unit-tests on any machine against the spec     |
| `compute-sanitizer`             | all of Tier A           | racecheck / initcheck / synccheck in CI                      |
| herd7 litmus tests              | K5.3, ring protocols    | Memory-model claims proved, not asserted                     |
| syzkaller + KASAN/KCSAN         | K3.1                    | Module fuzzed before it touches a real namespace             |
| virtme-ng / QEMU harness        | K3.1, K4.x              | Module crash-iterate without reprovisioning                  |

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

1. **K5.0** first — the cheapest item on the list, and it fixes the launch
   cadence that K5.1 later has to justify replacing.
2. **K5.3** — smallest self-contained piece of device code, and it establishes
   the memory-model discipline the rest of the compute plane assumes.
3. **K5.2, K5.5, K5.6, K5.1** — Tier A in dependency order; each has a CPU
   oracle.
4. **K4.1, K4.2, K4.3** — cheap, rooted-VM, independent of everything else. Good
   parallel track.
5. **K3.1** — the first Tier B item and the topology canary. If a candidate box
   cannot run this, it cannot run K1.1 or K2.2 either.
6. **K1.1** on top of K3.1; **K1.2/K1.3** alongside it once the namespace is in
   hand.
7. **K2.2**, then **K2.1** — DEVX QP construction is the apprenticeship for
   GPU-side WQE construction.
8. **K5.4**, **K2.3**, **K3.3**, **K3.2** — hardware-gated, schedule them when
   the part is available.

The single highest-leverage action is securing one bare-metal box with a
ConnectX and a spare NVMe namespace, and verifying its ACS/IOMMU topology before
any Tier B code is written. That fact determines whether the crown-jewel items
are days-hard or months-hard.

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
