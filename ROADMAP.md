# AetherGraph Test Plan

Comprehensive validation plan covering every component from unit tests through production load testing. Organized by AWS tier to minimize cost.

Functionally this operates as scratch workspace / lab bench for personal and agent's notes about experiment results, bugs encountered along the way, and should not be considered readable for third parties. Refer to README.md, INTEGRATION.md, and other docs as needed to understand current project state.

## Component Status

Last Tier 1 run: 2026-04-14, t3.medium spot us-east-1c, AL2023 kernel 6.18, Rust 1.94.1.

| Component                      | Unit Tests | Integration | Hardware Validated | Production Load |
|--------------------------------|------------|-------------|--------------------|-----------------|
| CSR graph + mmap               | ✅ PASS    | ✅ PASS     | -                  | NO              |
| NeighborLoader (PyG drop-in)   | ✅ 26/26 + 89 related | ✅ PASS | -            | NO              |
| io_uring feature store         | ✅ PASS    | ✅ bench ran (EBS gp3)  | NO    | NO              |
| Ray Data integration           | PASS       | NO          | NO                 | NO              |
| AF_XDP ingestion               | COMPILES   | ✅ UMEM + XdpSocket (T1.7); T1.8 end-to-end still pending | NO | NO              |
| Seqlock FeatureTable           | ✅ 9/9     | ✅ `tests/stress_seqlock.rs` (4W+8R, 5M iters, 0 torn)    | NO | NO              |
| ibverbs FFI + QP lifecycle     | ✅ 1/1     | ✅ SoftRoCE on Ubuntu (`softroce_e2e.rs`) + SRD cross-node on EFA | ✅ c6gn.16xlarge | NO |
| GPUDirect RDMA gather          | COMPILES   | NO                 | NO (needs Tier 3)  | NO              |
| CUDA validation kernel         | COMPILES   | NO                 | NO (needs Tier 2)  | NO              |
| DLPack PyO3 bridge             | COMPILES   | NO                 | NO (needs Tier 2)  | NO              |
| `feature_source="rdma://..."`  | COMPILES   | NO                 | NO                 | NO              |

### Tier 1 run results (2026-04-14, i4i.large + Ubuntu 24.04)

- **T1.1 Compile + unit tests:** ✅ aether-mem 14/14, aether-stream --features rdma 9/9, aethergraph-core 131/131 + 1 doctest
- **T1.2 Seqlock stress:** ✅ `tests/stress_seqlock.rs::stress_seqlock_concurrent_writers_readers` — 4 writer × 8 reader threads, 5M-iter cap / 10s budget, 0 torn reads observed on x86_64 + aarch64
- **T1.3 SoftRoCE QP handshake:** ✅ `t1_3_context_open_and_qp_handshake` — two contexts × two QPs cross-connect, both reach RTS
- **T1.4 RDMA READ of FeatureTable:** ✅ `t1_4_rdma_read_feature_table_bytes_match` — 32 nodes × 16 f32 read via real RDMA, byte-for-byte verified, head==tail==2
- **T1.5 Control plane TCP + QP exchange:** ✅ `t1_5_control_plane_qp_exchange` — TCP advertisement + bidirectional QP endpoint exchange + connection
- **T1.6 CQ error recovery:** ✅ `t1_6_cq_error_recovery` — dereg MR mid-flight produces error WC, CQ drains cleanly, fresh QP+MR works after
- **T1.6a MR cross-thread:** ✅ `tests/mr_xthread_repro.rs` (4 thread-layout variants) + `tests/mr_xthread_sharded_repro.rs` (16 callers × 4 shards × 1k iters × 8 reads, MRs pre-registered on main)
- **T1.6b WR-chain correctness + pipelining:** ✅ `tests/rdma_wr_chain_stress.rs` — 1024-WR single batch with per-WR signaling; sustained 50×200 batches at signal_every_n=10; empty-post no-op; zero-length read succeeds without touching buffer
- **T1.6c Control plane robustness:** ✅ `tests/control_plane_robustness.rs` — server survives immediate-close, oversized length prefix, truncated body, malformed JSON; serves 10 sequential good clients after abuse
- **T1.6d Device enumeration + open-by-index:** ✅ `tests/device_enum.rs` — enumerates at least one device with contiguous indices and sane `numa_node`; `open_on_device` honors the index and rejects out-of-range with `NotFound`
- **T1.6e SRD (EFA) RDMA READ loopback:** ✅ `tests/srd_e2e.rs` — on c7g.16xlarge EFA (rdmap0s31): full state machine (RESET→INIT→RTR→RTS), endpoint shape (qpn + AHN + qkey + GID), byte-for-byte RDMA READ loopback, pipelined in-order 4-read drain. Backing FFI: `src/rdma/efa_ffi.rs` + C shim builder helpers in `csrc/ibv_shim.c`. ABI reference: `debug/efa_abi_probe.c`.

### EFA SRD bench vs naive baselines (2026-04-14, c7g.16xlarge loopback)

Single-inflight post+drain loop, 2 000 iters after 64 warm-up. TCP RPC is a
localhost `TcpStream::set_nodelay(true)` pair — client sends `[u32 len]`, server
replies with `len` bytes. memcpy is the same-host `copy_from_slice` floor.

| Payload |   memcpy   |  TCP RPC   | EFA SRD    | SRD / TCP |
|--------:|-----------:|-----------:|-----------:|:---------:|
|    64 B |    72 ns   |  14.38 µs  |   6.10 µs  |  **2.4×** |
|   512 B |    80 ns   |  14.84 µs  |   5.88 µs  |  **2.5×** |
|   4 KiB |   123 ns   |  15.65 µs  |   8.14 µs  |  **1.9×** |
|  16 KiB |   275 ns   |  16.83 µs  |   9.09 µs  |  **1.9×** |
|  64 KiB |   972 ns   |  24.07 µs  |  12.99 µs  |  **1.9×** |

Latency floors (min observed): SRD 4.25 µs @ 512 B vs TCP 10.68 µs — RDMA saves
~6–10 µs of kernel network-stack overhead per op, even on loopback where EFA is
not in its element. On inter-node traffic (where EFA is designed to run) the
gap widens further because TCP pays additional per-hop queueing / congestion.

`cargo test --features efa --release -p aether-stream --test srd_e2e -- --ignored --nocapture` reruns the three benches (`bench_memcpy_baseline`, `bench_tcp_rpc_baseline`, `bench_srd_rdma_read_latency`).
- **T1.7 AF_XDP primitives:** ✅ `tests/afxdp_veth.rs` — UMEM allocation at frame_size=4096, rejection of 2048 / arbitrary sizes, acquire/release roundtrip, `XdpSocket` creation bound to `veth-rx` queue 0
- **T1.8 AF_XDP → FeatureTable integration:** ❌ not written; needs UDP-send → `ingest_loop` → parse → seqlock-write → read assertion
- **T1.9 io_uring feature store:** ✅ regression closed. Fresh async_io_benchmark on c6gn.16xlarge (EBS gp3 root, small synthetic graph). Warm-cache: `async_iouring_1k_nodes` 870µs, `sync_pread_1k_nodes` 704µs, `sync_pread_rayon_1k_nodes` 552µs, `mmap_lookup_1k_nodes` 92µs — async pays a ~1.24× tag on fully-cached data (spawn_blocking + buffer-allocation overhead). Cold-cache (AETHER_BENCH_COLD=1, posix_fadvise DONTNEED per iter): async 103.7ms, rayon 104.5ms, sync_pread 147ms — io_uring **1.4× faster than sync** and matches per-thread rayon parallelism on EBS gp3. `DEFAULT_RING_ENTRIES = 4096` (`crates/aethergraph-core/src/internal/uring.rs`) is the load-bearing bit — smaller values thrash the SQ overflow slow-path on batches > depth.
- **T1.10 NeighborLoader E2E (Python):** ✅ 115 passed, 1 skipped (pin_memory needs CUDA)

**Total: 159 Rust tests + 115 Python tests green, 0 failed.**

### FFI bugs found and fixed during T1.3–T1.6 implementation

The first real run of the RDMA data path (vs in-memory struct-layout tests) surfaced ten production-critical bugs in the hand-rolled ibverbs FFI:

| # | File | Bug | Effect |
|---|---|---|---|
| 1 | `aether-stream/src/rdma/ffi.rs` | `IBV_WR_RDMA_READ = 1` (was the `WRITE_WITH_IMM` opcode); should be 4 | "READ" actually performs RDMA WRITE — silent data corruption |
| 2 | `aether-stream/src/rdma/ffi.rs` | `IBV_SEND_SIGNALED = 4` (was `SOLICITED`); should be 2 | No completions ever delivered to CQ |
| 3 | `aether-stream/src/rdma/ffi.rs` | `IBV_ACCESS_REMOTE_READ` ↔ `IBV_ACCESS_REMOTE_WRITE` swapped | MRs grant wrong remote permissions to peers |
| 4 | `aether-stream/src/rdma/ffi.rs` | `IbvSendWr` 72 bytes; C struct is 128 | Driver reads past Rust allocation (UB); chained-WR traversal corruption |
| 5 | `aether-stream/src/rdma/ffi.rs` | `IbvPortAttr` 52 bytes; C struct is 56 | Kernel writes 4 bytes past allocation on every `ibv_query_port` |
| 6 | `aether-stream/src/rdma/ffi.rs` | `IbvQpAttr` 140 bytes; C struct is 144 | Kernel reads 4 bytes past allocation on every `ibv_modify_qp` |
| 7 | `aether-stream/src/rdma/ffi.rs` + `aether-mem/src/hooks/rdma.rs` | `extern "C"` blocks missing `#[link(name="ibverbs")]` | Any binary actually calling FFI fails to link (latent — unit tests didn't trip it) |
| 8 | `aether-stream/src/rdma/ffi.rs` | `ibv_post_send`/`ibv_poll_cq` declared as ABI symbols (they're `static inline` in `<infiniband/verbs.h>`) | Link error on data-path code; required new `csrc/ibv_shim.c` + `build.rs` |
| 9 | `aether-stream/src/rdma/context.rs` | `RdmaContext::open` hardcoded GID index 0 (link-local IPv6 on RoCE) | RoCE packets undeliverable — destination unreachable via Ethernet |
| 10 | `aether-stream/src/rdma/qp.rs` | `to_rtr` hardcoded `sgid_index = 0` | Mismatched SGID in advertised packets even after caller picks correct GID |

After all ten fixes, T1.3–T1.6 pass and the full 155 unit-test suite still passes (no regressions).

### Known environment gotchas

- **AL2023 kernel 6.18 does not ship `rdma_rxe`.** Use Ubuntu 22.04/24.04 (default kernel has it) or downgrade to AL2023 kernel 6.1 + `kernel-modules-extra`.
- **SoftRoCE on `lo` does not work.** Bind rxe to a real ethernet device (e.g. `ens5`); loopback has no MAC for ARP/GID resolution → "Failed to modify QP to RTR".
- **GID 0 is link-local IPv6 on RoCE; not routable over Ethernet.** Pick the IPv4-mapped GID (typically index 1 — confirm via `show_gids`). `RdmaContext::open` requires explicit `gid_index`; no auto-probing in the data path.
- **`cargo test -p ... -p ...` shares compile** — if one crate fails to compile, sibling tests get canceled mid-build.
- **t3.medium is CPU-starved for maturin builds** — first `uv sync` (release-mode PyO3 wheel) takes 5–10 min. Cache persists across runs.

### ibverbs perf-gap closures (2026-04-14)

Tractable wins applied after the FFI bug round:

| Change | Before | After |
|---|---|---|
| `RdmaContext::reg_mr` returns `*mut IbvMr` | manual `ibv_dereg_mr` everywhere; easy leak | RAII `RegisteredMr` with Drop (`crates/aether-stream/src/rdma/context.rs`) |
| `RdmaQp::create(ctx)` hardcoded `max_send_wr=4096` | one-size-fits-all | `create(ctx, &cap)` + `DEFAULT_QP_CAP` const for ergonomics; callers tune per HCA |
| `post_reads` only ever signals last WR | drains-once pattern only | `post_reads_with_signaling(reads, every_n)` for sustained pipelining; old `post_reads` preserved as the common case |
| Test poll loops used `thread::yield_now()` + `[IbvWc; 4]` | ~1µs yield overhead/iter | `std::hint::spin_loop()` (PAUSE/YIELD) + `[IbvWc; 32]`, matches production `client.rs` |
| `RdmaContext::open` GID auto-scanned (added then reverted per "no dynamic" rule) | runtime probing | explicit `gid_index: u8` param; caller picks routable GID |

SoftRoCE bench delta on i4i.large (4096 nodes, 768-dim, slot=4096B):

| Batch | Before | After | Per-read |
|---|---|---|---|
| 1 | 7.9 µs | 5.3 µs | 5.26 µs |
| 8 | 38.6 µs | 30.5 µs | 3.82 µs |
| 64 | 268.6 µs | 235.1 µs | 3.67 µs |
| 256 | 1050.3 µs | 916.9 µs | 3.58 µs |
| 1024 | 4520.2 µs | 4092.5 µs | 4.00 µs |

3.58 µs/read at batch=256 is within 4% of `ib_read_lat`'s 3.45 µs baseline — at the SoftRoCE software-emulation ceiling.

### Still-open architectural items (deferred — design notes below)

#### Item 8 — Multi-threaded post/poll

`RdmaQp::post_reads` + `RdmaFeatureClient::post_and_wait` are sequential. At
>1 GB/s sustained you'll bottleneck on single-thread CQ poll. Real fix:

- **Sharded QPs:** N QPs per peer (one per worker thread), each with its own
  send CQ. Each thread polls only its own CQ → no contention.
- **Where to put it:** new `crates/aether-stream/src/rdma/sharded_client.rs`
  paralleling `client.rs`. Holds `Vec<RdmaQp>` and `Vec<*mut IbvCq>`.
  Constructor takes `num_shards` and `Vec<core_affinity::CoreId>`.
- **Server side:** `serve_control_plane_with_qp` loops accepts already; for
  N shards client opens N TCP connections, each gets a server QP. Server
  needs to track which QPs map to which shard for fairness.
- **Effort:** ~1 day, isolated module. Existing single-QP path stays for
  the "one batch at a time" common case.

#### Item 7 — EFA / SRD RDMA READ (DONE this session, loopback)

Single-node SRD RDMA READ is working end-to-end on c7g.16xlarge EFA
(`rdmap0s31`, libefa 1.4.61, kernel 6.17). Files:

- `src/rdma/efa_ffi.rs` — FFI bindings for `efadv_query_device`,
  `efadv_query_ah`, `efadv_create_qp_ex`, the extended CQ/QP handles, and
  the shim-routed builder API. Struct sizes (`IbvQpInitAttrEx = 136`,
  `IbvCqInitAttrEx = 56`, `EfadvQpInitAttr = 16`, `EfadvDeviceAttr = 32`,
  `EfadvAhAttr = 16`) guarded by compile-time `const _` assertions.
- `src/rdma/srd.rs` — `SrdContext` (device open + PD + extended CQ),
  `SrdQp` (create via `efadv_create_qp_ex`, full RESET→INIT→RTR→RTS state
  machine with qkey), `SrdAddressHandle` (carries AHN from `efadv_query_ah`),
  one-shot RDMA READ via the builder API.
- `csrc/ibv_shim.c` — wraps `ibv_create_cq_ex`, `ibv_qp_to_qp_ex`,
  `ibv_cq_ex_to_cq`, and the
  `wr_start`/`wr_rdma_read`/`wr_set_ud_addr`/`wr_set_sge`/`wr_complete`
  builder sequence into a single FFI entry point (`aether_ibv_post_rdma_read_srd`),
  plus `aether_ibv_poll_cq_ex_one` that drains one extended CQE into a
  snapshot struct. Built behind `AETHER_EFA_SHIM` in `build.rs` and linked
  against `libefa`.
- `tests/srd_e2e.rs` — three integration tests: byte-for-byte loopback
  RDMA READ, endpoint-shape assertion, 4-WR pipelined in-order completion.
- `debug/efa_abi_probe.c` — committed C probe that prints every struct
  size / field offset / constant value the Rust FFI depends on. Run after
  kernel / rdma-core / libefa bumps and reconcile drift with the `const _`
  compile-time assertions.

**Cross-node SRD (DONE this session)**:
`serve_srd_control_plane` + `SrdFeatureClient::connect` now drive the
bidirectional handshake: client ships its `SrdEndpoint` first, server
creates a local `ibv_ah` for the client (populates EFA's hardware peer
table), server then sends the `SrdAdvertisement`. Without this,
inbound RDMA READ completes with `vendor_err=14`
(`REMOTE_ERROR_UNKNOWN_PEER`).

Verified on two c6gn.16xlarge in the same subnet (us-east-2c), same SG
with self-reference on BOTH ingress AND egress (generic egress
`0.0.0.0/0` silently drops EFA traffic).

### Cross-node SRD RDMA READ bench (c6gn.16xlarge ↔ c6gn.16xlarge)

Same-subnet, `examples/srd_client.rs`, 4 KiB slots, random-gather
workload (no locality). `--shards N` drives the `SrdShardedFeatureClient`
pool (N independent contexts/QPs/AHs with N TCP handshakes, each shard
runs its post+drain on its own OS thread via `std::thread::scope`).

| shards | per-shard batch | total batch | MB/s | µs/read |
|-------:|----------------:|------------:|-----:|--------:|
|     1  |           1024  |       1024  | 4165 |  0.98   |
|     2  |           1024  |       2048  | 5499 |  0.74   |
|     4  |           1024  |       4096  | 6566 |  0.62   |
|     4  |           4096  |      16384  | 6921 |  0.59   |
|     8  |           1024  |       8192  | 6802 |  0.60   |
|     8  |           2048  |      16384  | 6880 |  0.60   |

Plateau ≈ **6.9 GB/s = 55 Gbps** regardless of shard count past 4 or
per-shard batch past 1024. That's **~55% of the advertised 100 Gbps
line rate** — the remaining headroom is only recoverable by (a)
coalescing many feature slots into fewer larger RDMA READ WRs so the
per-WR doorbell cost amortizes over more bytes, or (b) running on
newer hardware with faster MMIO (c7gn.16xlarge / Graviton3).

Comparison on the same two c6gn.16xlarge nodes:

|      Path        |  Pattern   | Throughput | 4 KiB latency |
|------------------|------------|-----------:|--------------:|
| TCP 8-stream bulk (iperf3) | streaming |  4.85 GB/s |        —     |
| UDP 4 KiB dgram  | one-shot   |  0.62 GB/s |        —     |
| TCP RPC (localhost 4 KiB) | req/resp  |  0.26 GB/s |  15–24 µs   |
| EFA SRD 1-shard  | req/resp   |  4.17 GB/s |   0.97 µs   |
| EFA SRD 8-shard  | req/resp   |  **6.88 GB/s** | **0.60 µs** |

vs naive TCP RPC: **~25× lower per-read latency, ~26× higher
throughput**. vs iperf3 bulk TCP (8 parallel streams): **1.4× higher
throughput** with zero server CPU involvement per read (RDMA READ is
one-sided — the server's cores never see the data-plane traffic).

**QP-cap sweep** (ablation in `tests/srd_e2e.rs::srd_qp_cap_ablation`):
every shape up to `max_send_wr = 4096 / max_recv_wr = 1 / cq = 4096`
passes on c6gn.16xlarge. `DEFAULT_SRD_QP_CAP` is currently
`max_send_wr=1024, max_recv_wr=1`.

#### Item 10 — Tier 3 (real-hardware RDMA + GPU)

Blocked on item 7 (SRD path) AND item 6 (GPU quota). Two viable paths:

- **A) AWS via EFA + GPU quota.** Once AWS approves the pending GPU vCPU
  quota requests (filed via `aws service-quotas request-service-quota-increase`
  for codes L-DB2E81BA, L-3819A6DF, L-417A185B, L-7212CCBC at 192 vCPUs in
  us-east-1 / us-east-2 / us-west-1) AND item 7's SRD path is implemented:
  - `c7gn.xlarge` (~$0.20/hr spot) — hardware RDMA validation on EFA, no GPU
  - `p4d.24xlarge` (~$10/hr spot) — GPUDirect RDMA + 8× A100 (Tier 3 proper)

- **B) On-prem ConnectX.** Existing RC code path works on real Mellanox /
  NVIDIA NICs. No EFA work, no quota wait — just access to a 2-node
  ConnectX cluster (or a single-node loopback test on real hardware).

T3.1–T3.5 unrunnable until A or B lands. Auto-approval status of the GPU
quota requests can be polled with `aws service-quotas
list-requested-service-quota-change-history --service-code ec2`.

#### Item 8 — Sharded multi-thread post/poll (DONE this session)

`crates/aether-stream/src/rdma/sharded.rs` lands a `ShardedQpPool` with N
(QP, CQ, worker thread) shards and per-shard CQs so polling threads don't
contend. New supporting primitives: `RegisteredCq` RAII, `RdmaQp::create_with_cqs`,
`RdmaQp::poll_cq_on`. Verified on i4i SoftRoCE: 8 caller × 4 shards × 200
iters × 8 reads = 12,800 successful gathers, byte-for-byte verified.

**MR cross-thread**: `RegisteredMr` is `Send` — allocate + `reg_mr` on any
thread, ship to any other thread, post from any thread. Regression guards
are `tests/mr_xthread_repro.rs` (four alloc/post thread-layout variants)
and `tests/mr_xthread_sharded_repro.rs` (16 caller threads × 4 shards × 1k
iters × 8 reads with MRs pre-registered on the main thread). Both verified
on i4i SoftRoCE.

#### Item 9 — NUMA-aware MR + QP placement (DONE this session)

Added `RdmaContext::open_on_device(cq_size, device_index, gid_index)` and
`enumerate_devices() -> Vec<RdmaDeviceInfo { index, name, numa_node }>` so
callers can pick the NIC co-located with their pinned threads + memory.
No probing in the hot path — caller is responsible for picking correctly.
Cross-NUMA RDMA costs ~2-3× latency on real hardware.

#### Connection model — RC at scale

RC needs O(N²) QPs for N peers — fine for small clusters; for >100 peers
consider DC (Dynamically Connected) or SRD with application-level
reliability. Folded into items 7/8 above.

### Bugs found by stress tests this session — fixed at the protocol level

- **Seqlock head/tail ordering bug (`feature_table.rs:143`).** Stress test
  with 4 writers + 8 readers + 10M iters caught **2 torn reads** on x86_64.
  Cause: writer's `Release` stores can sit in the CPU's store buffer; the
  reader can observe both the old `tail` AND the old `head` even though
  the writer's `head→odd` RMW already happened. Single `head == tail`
  check passes with mixed-generation feature bytes.
  **Fix:** added a second `head.load(Acquire)` after the feature copy
  (standard Linux-kernel seqlock pattern). Since the writer's first op is
  a LOCK-prefixed RMW that's globally visible immediately, h2 cannot
  equal the old even h1 if a writer ran. RDMA-compatible `h == t` check
  is preserved as a cheap early-out. Stress test now reports 0 torn
  reads in 10M iters at ~1.4M reads/s + ~14M writes/s.

- **Umem 2048-frame stride mismatch (`umem.rs`).** AF_XDP frame-size 2048
  silently produced 4096-byte stride because `SharedMemoryRing` rounds
  `slot_size` to the page boundary. Kernel was told `chunk_size=2048`,
  our `frame_ptr(N)` returned `base + N*4096` — half the frames pointed
  to the wrong UMEM region.
  **Fix:** `Umem::new` now rejects `frame_size != 4096` with a clear
  message. (A future change can add a sub-page allocator path if the
  kernel-min 2048 is actually needed; today nothing in the codebase does.)

### io_uring consolidation

`crates/aethergraph-core/src/internal/uring.rs::batch_read` was a speculative reference impl with no callers; meanwhile `prefetch.rs`, `async_store.rs`, and `async_graph.rs` each inlined ~50 LOC of duplicated SQ submit + CQ poll loops with subtly different short-read handling. Refactored to a single canonical `batch_read(handle, fd, &[(offset, ptr, len)])` so SQPOLL/short-read fixes apply in one place.

---

## Tier 1 — `t3.medium` ($0.04/hr, ~4 hours, ~$0.16)

No GPU. Amazon Linux 2023. SoftRoCE + veth pairs.

### Setup

```bash
sudo dnf install -y rdma-core libibverbs-devel iproute clang

# SoftRoCE (emulated RDMA). Bind to the primary NIC, NOT lo — loopback has no
# MAC so GID/ARP resolution fails with "Failed to modify QP to RTR" during the
# RC state transition. Use whatever `ip -br link` shows as your UP ethernet.
sudo modprobe rdma_rxe
PRIMARY_NIC=$(ip -br link | awk '/UP/ && $1 !~ /^(lo|veth)/ {print $1; exit}')
sudo rdma link add rxe0 type rxe netdev "$PRIMARY_NIC"
ibv_devices       # verify rxe0 appears
ibv_devinfo       # port should be PORT_ACTIVE, MTU 4096

# Smoke test (loopback via real NIC):
#   (ibv_rc_pingpong -d rxe0 -g 1 -s 4096 -n 1000 &); sleep 2;
#   ibv_rc_pingpong -d rxe0 -g 1 -s 4096 -n 1000 <private-ip>
# Expected: ~200 Mbit/s, ~300 µs/iter on a small instance.

# Also: AL2023 kernel 6.18 ships WITHOUT rdma_rxe module.
# Use Ubuntu 22.04/24.04 (has it built in) or pinned AL2023 kernel 6.1 + kernel-modules-extra.

# veth pair for AF_XDP
sudo ip link add veth-tx type veth peer name veth-rx
sudo ip link set veth-tx up
sudo ip link set veth-rx up

# Locked memory for ibv_reg_mr + mlock
ulimit -l unlimited
```

### T1.1 — Compile + existing tests

```bash
cargo test -p aether-mem
cargo test -p aether-stream --features rdma
cargo test -p aethergraph-core
```

**Expected:** 14 aether-mem + 9 aether-stream + all core tests pass.

### T1.2 — FeatureTable stress test (concurrent writers + readers)

Write a test binary (`tests/stress_seqlock.rs`):

- Spawn 4 writer threads, each writing to its own set of nodes
- Spawn 8 reader threads, each reading random nodes in a loop
- Run for 10M iterations
- Assert: no reader ever sees `read_node()` return true with data that doesn't match what any writer wrote (use checksums)
- Assert: `fence(Acquire)` between feature copy and tail load prevents torn reads on aarch64

**Validates:** Seqlock correctness under contention, memory ordering on ARM (Graviton).

### T1.3 — SoftRoCE: RdmaContext + QP handshake

Write a test binary (`tests/softroce_handshake.rs`):

1. `RdmaContext::open(256)` — must find `rxe0` device
2. Create two QPs (server + client) on the same context
3. Exchange endpoints via in-memory (no TCP needed for test)
4. `qp.connect()` both sides (RESET → INIT → RTR → RTS)
5. Assert: both QPs reach RTS without error

**Validates:** `IbvQp` struct layout (qp_num at correct offset), `IBV_QP_DEST_QPN` constant (1 << 20), QP state transition masks.

### T1.4 — SoftRoCE: RDMA READ of FeatureTable

Write a test binary (`tests/softroce_feature_read.rs`):

1. Allocate `FeatureTable` (1024 nodes, dim=128)
2. Write known features to nodes 0-99
3. `ibv_reg_mr` the FeatureTable memory (host memory, not GPU)
4. Create server QP + client QP, connect them
5. Post RDMA READs for nodes 0-99 (client reads from server's MR)
6. Poll CQ for completion
7. Validate: read features match written features byte-for-byte

**Validates:** ibv_post_send, IbvSendWr union layout, chained WR traversal, IbvSge addresses, CQ polling, rkey/lkey from non-opaque IbvMr.

### T1.5 — SoftRoCE: Control plane TCP handshake

Write a test binary (`tests/softroce_control_plane.rs`):

1. Spawn `serve_control_plane_with_qp` in a thread
2. Call `connect_with_qp` from main thread
3. Assert: returns valid `RdmaAdvertisement` with correct schema
4. Assert: both QPs connected (post a dummy RDMA READ, verify completion)

**Validates:** TCP length-prefixed protocol, ServerHello/ClientHello serde, QP endpoint exchange, `MAX_MSG_SIZE` limit, TCP timeouts.

### T1.6 — SoftRoCE: CQ error recovery

1. Connect two QPs
2. Deregister the server's MR (makes remote memory invalid)
3. Client posts RDMA READs (will fail with remote access error)
4. Verify `post_and_wait` returns error AND the CQ is properly drained
5. Re-register MR, post new RDMA READs
6. Verify the new batch succeeds (no stale CQEs from previous error)

**Validates:** CQ drain loop waits for signaled WR, no desync on subsequent gathers.

### T1.7 — AF_XDP ingestion with veth pair

Write a test binary (`tests/afxdp_veth.rs`):

1. Create `Umem` (256 frames, 2048 bytes each)
2. Create `XdpSocket` bound to `veth-rx`, queue 0, `XDP_COPY` mode
3. Spawn `ingest_loop` on a dedicated thread
4. From main thread: send 100 UDP packets into `veth-tx` via raw socket
5. Receive `InboundFrame` from the crossbeam channel
6. Assert: received 100 frames, payload matches sent data

**Validates:** XdpSocket creation, UMEM registration, FILL/RX/COMPLETION ring operations, busy-poll ingestion loop, frame lifecycle.

### T1.8 — AF_XDP → FeatureTable integration

1. Setup veth pair + XdpSocket + FeatureTable
2. Send UDP packets containing `(node_id: u32, features: [f32; 128])`
3. Ingestion loop parses packets, writes to FeatureTable
4. Reader thread reads from FeatureTable, verifies features match

**Validates:** Full ingestion pipeline: NIC → AF_XDP → parse → seqlock write → read.

### T1.9 — io_uring feature store

```bash
# Create a test feature file (100K nodes, dim=768)
cargo run --example create_test_features -- --nodes 100000 --dim 768 --output /tmp/test_features.bin

# Run io_uring benchmark
cargo bench -p aethergraph-core -- feature
```

Also write a test that:
1. Creates `SyncFeatureStore` with the test file
2. Calls `get_batch` for random batches of 1024 nodes
3. Verifies features match expected values
4. Measures throughput (should be >1M nodes/sec on NVMe)

**Validates:** AETHFEAT file format, io_uring SQPOLL, O_DIRECT alignment, batch read correctness.

### T1.10 — NeighborLoader E2E (no GPU)

```bash
cd python
uv run pytest tests/ -k "neighbor_loader" -v
```

Also write a stress test:
1. Load a synthetic graph (1M nodes, 50M edges)
2. Run NeighborLoader for 100 epochs
3. Verify no crashes, no memory leaks, correct PyG Data shapes
4. Measure prefetch hit rate (should be >95%)

**Validates:** Python → Rust → sampling → features → PyG Data pipeline.

---

## Tier 2 — `g4dn.xlarge` spot (~$0.16/hr, ~4 hours, ~$0.64)

T4 GPU (16GB VRAM). SoftRoCE for RDMA (host memory, not GPUDirect).

### Setup

```bash
# Same as Tier 1, plus:
sudo dnf install -y nvidia-driver cuda-toolkit

# Verify GPU
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

### T2.1 — CUDA validation kernel with injected torn reads

Write a test (`tests/cuda_seqlock_kernel.rs`):

1. `CudaContext::new(0)`, get stream
2. Allocate VRAM staging buffer (3 slots, dim=128)
3. Build slot data on CPU:
   - Slot 0: head=2, features=[1.0; 128], tail=2 (VALID)
   - Slot 1: head=3, features=[2.0; 128], tail=2 (TORN — head odd)
   - Slot 2: head=4, features=[3.0; 128], tail=2 (TORN — head != tail)
4. `memcpy_htod` slot data to VRAM
5. Launch `SeqlockValidator::validate()`
6. Assert: `retry_count == 2`
7. Assert: `retry_indices() == [1, 2]`
8. Copy output tensor back to CPU
9. Assert: slot 0 features are `[1.0; 128]` (compacted correctly)

**Validates:** CUDA kernel correctness, tail offset computation matches Rust, atomicAdd retry counter, feature compaction.

### T2.2 — DLPack capsule → PyTorch tensor

Write a Python test (`tests/test_dlpack.py`):

1. From Rust (via PyO3): allocate VRAM, fill with known f32 values
2. Create DLPack capsule via `create_dlpack_capsule()`
3. `x = torch.from_dlpack(capsule)`
4. Assert: `x.device == torch.device('cuda:0')`
5. Assert: `x.shape == (num_nodes, feature_dim)`
6. Assert: `x.dtype == torch.float32`
7. Assert: `torch.allclose(x, expected_tensor)`

**Validates:** DLPack struct layout, PyCapsule creation, CUDA device/dtype metadata, zero-copy transfer.

### T2.3 — SoftRoCE RDMA READ → host buffer → GPU memcpy → validate

End-to-end test without nvidia-peermem:

1. Server: FeatureTable (1024 nodes, dim=128), write known features
2. Client: SoftRoCE RDMA READ into **host** pinned memory (cudaMallocHost)
3. `memcpy_htod` from host buffer to VRAM staging buffer
4. Launch CUDA validation kernel
5. Assert: all reads valid, features match

**Validates:** Full pipeline except nvidia-peermem. Proves the logic works even if RDMA target is host memory instead of VRAM.

### T2.4 — `feature_source="rdma://..."` Python E2E (SoftRoCE, host memory fallback)

This requires a small shim: on Tier 2 without nvidia-peermem, `GpuGatherBuffer::new` will fail at `ibv_reg_mr` on GPU memory. Write a host-memory fallback test:

1. Server: FeatureTable + `serve_control_plane_with_qp` on localhost
2. Client Python script:
```python
loader = NeighborLoader(graph, num_neighbors=[5, 3], batch_size=32,
                        feature_source="rdma://127.0.0.1:9999")
for batch in loader:
    assert batch.x is not None
    assert batch.x.device.type == 'cuda'
    assert batch.x.shape[1] == feature_dim
```

If nvidia-peermem isn't available, test with a modified `GpuGatherBuffer` that uses host pinned memory + `memcpy_htod`. This validates the full Python → Rust → RDMA → CUDA → DLPack → PyTorch path.

**Validates:** The entire E2E wiring from `feature_source` param to `batch.x` on GPU.

### T2.5 — NeighborLoader + PyG training loop (GPU)

```python
import torch
from aethergraph import Graph
from aethergraph.pytorch import NeighborLoader

graph = Graph.load("test_graph.bin")
graph.load_features("test_features.bin")

loader = NeighborLoader(graph, num_neighbors=[15, 10], batch_size=128,
                        pin_memory=True)

# Simple 2-layer GCN
model = GCN(in_channels=768, hidden=256, out_channels=10).cuda()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(5):
    for batch in loader:
        batch = batch.to('cuda')
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print("Training complete — no crashes, shapes correct")
```

**Validates:** Real GNN training works end-to-end with AetherGraph as the data loader.

---

## Tier 3 — `p3dn.24xlarge` spot (~$9/hr, ~2 hours, ~$18)

8x V100 + EFA (Elastic Fabric Adapter). True GPUDirect RDMA.

### Setup

```bash
sudo modprobe nvidia-peermem
sudo modprobe ib_uverbs

# Verify EFA
ibv_devices  # should show efa0
fi_info -p efa  # EFA fabric info

# Verify nvidia-peermem
dmesg | grep peermem  # "nvidia-peermem registered"

ulimit -l unlimited
```

### T3.1 — ibv_reg_mr on CUDA device pointer

```rust
let ctx = RdmaContext::open(256)?;
let cuda_ctx = CudaContext::new(0)?;
let stream = cuda_ctx.default_stream();
let buf: CudaSlice<u8> = stream.alloc_zeros(1 << 20)?; // 1MB VRAM
let (ptr, _guard) = buf.device_ptr(&stream);
let mr = ctx.reg_mr(ptr as *mut u8, 1 << 20, IBV_ACCESS_LOCAL_WRITE)?;
assert!(!mr.is_null());
// If this passes, nvidia-peermem is working
```

**Validates:** nvidia-peermem kernel module, ibv_reg_mr on GPU pointers, lkey valid for RDMA WRs.

### T3.2 — GPUDirect RDMA READ loopback

Single instance, server + client on same box:

1. Server: FeatureTable (10K nodes, dim=768), write random features
2. Server: `ibv_reg_mr` FeatureTable memory, serve control plane
3. Client: `RdmaFeatureClient::connect("127.0.0.1:9999", gpu_id=0, batch=1024)`
4. Client: `gather(&node_ids)` for 1024 random nodes
5. Copy output tensor to CPU, compare against server's FeatureTable
6. Assert: byte-for-byte match

**Validates:** True GPUDirect RDMA: NIC reads from host memory, writes to VRAM via PCIe, CUDA kernel validates, features correct.

### T3.3 — GPUDirect torn read detection

1. Server writes to FeatureTable in a tight loop (continuous updates)
2. Client does repeated `gather()` calls concurrently
3. Run for 10M iterations
4. Track: how many times the CUDA kernel detects torn reads (retry_count > 0)
5. Track: how many retries succeed on second attempt
6. Assert: ZERO cases where torn data passes validation (no corrupted features reach output)

**Validates:** Head/tail seqlock protocol over real RDMA under contention. The critical correctness test.

### T3.4 — Latency profiling

```rust
let start = Instant::now();
for _ in 0..10_000 {
    client.gather(&node_ids_1024)?;
}
let elapsed = start.elapsed();
println!("p50: {}μs", elapsed / 10_000);
```

**Expected:**
- RDMA READ (1024 nodes × 3088 bytes/slot): <10μs
- CUDA validation + compaction: <5μs
- Total gather latency: <20μs

If >50μs: check PCIe ACS (BIOS setting), NUMA topology (GPU and NIC on same socket), MTU (should be 4096).

### T3.5 — Full E2E: Python training with live features

Two terminals on same instance:

Terminal 1 (server):
```bash
# Run FeatureTable server with synthetic live updates
cargo run --release --features gpudirect --example feature_server -- \
    --nodes 100000 --dim 768 --update-rate 10000 --bind 0.0.0.0:9999
```

Terminal 2 (client):
```python
loader = NeighborLoader(graph, num_neighbors=[15, 10], batch_size=128,
                        feature_source="rdma://127.0.0.1:9999")

for epoch in range(10):
    for batch in loader:
        assert batch.x.device.type == 'cuda'
        assert batch.x.shape == (batch.num_nodes, 768)
        # Features are LIVE — different each epoch because server is updating
        out = model(batch.x, batch.edge_index)
        loss.backward()
```

**Validates:** The entire production path. Live-updating features, GPUDirect RDMA, CUDA validation, DLPack, PyTorch training loop. This is the demo.

---

## Tier 4 — Production benchmarks (optional, same instance)

### T4.1 — Billion-node graph loading

```bash
# Generate 1B node graph (CSR, ~32GB on NVMe)
cargo run --release -p aethergraph-cli -- generate \
    --nodes 1000000000 --avg-degree 50 --output /nvme/graph_1B.bin

# Load and sample
python3 -c "
from aethergraph import Graph
g = Graph.load('/nvme/graph_1B.bin')
print(f'Loaded: {g.num_nodes:,} nodes, {g.num_edges:,} edges')
# Startup should be instant (mmap, no loading)
"
```

### T4.2 — NeighborLoader throughput at scale

```python
loader = NeighborLoader(graph_1B, num_neighbors=[15, 10], batch_size=1024)
batches = 0
start = time.time()
for batch in loader:
    batches += 1
    if batches >= 1000:
        break
elapsed = time.time() - start
print(f"{1000/elapsed:.0f} batches/sec, {1000*1024/elapsed:.0f} seeds/sec")
```

**Target:** >500 batches/sec (>500K seeds/sec) on single instance.

### T4.3 — Ray Data multi-GPU

```python
import ray
from aethergraph.ray import create_sampling_dataset

ray.init()
dataset = create_sampling_dataset(
    graph_path="/nvme/graph_1B.bin",
    num_neighbors=[15, 10],
    batch_size=1024,
)
# Distribute across 8 GPUs on p3dn
for batch in dataset.iter_batches():
    # Each GPU worker gets batches in parallel
    pass
```

**Validates:** Ray Data integration at scale, multi-GPU throughput.

---

## Test execution order

```
Tier 1 ($0.16)                    Tier 2 ($0.64)                  Tier 3 ($18)
─────────────                     ──────────────                  ────────────
T1.1 Compile + unit tests         T2.1 CUDA kernel torn reads     T3.1 ibv_reg_mr on GPU ptr
T1.2 Seqlock stress               T2.2 DLPack → PyTorch          T3.2 GPUDirect RDMA READ
T1.3 SoftRoCE QP handshake        T2.3 SoftRoCE → host → GPU     T3.3 Torn read detection
T1.4 SoftRoCE feature read        T2.4 feature_source E2E        T3.4 Latency profiling
T1.5 Control plane TCP            T2.5 PyG training loop          T3.5 Full E2E live features
T1.6 CQ error recovery
T1.7 AF_XDP veth ingestion
T1.8 AF_XDP → FeatureTable
T1.9 io_uring feature store
T1.10 NeighborLoader E2E
```

**Total budget: ~$19. Stop at any tier if blockers found.**

Each tier's tests are ordered by dependency — if T1.3 (QP handshake) fails, skip T1.4-T1.6 and fix the FFI first. If T2.1 (CUDA kernel) fails, skip T2.3-T2.4. Never burn Tier 3 hours until Tier 1+2 are green.

---

## Remaining work

Everything above this line has been exercised to the "Integration" column. This section is the explicit punch list of what is NOT yet done, split by whether the blocker is code or external (GPU quota / hardware access).

### Not GPU-blocked — can be driven today

These three items are CPU/NVMe-only and can run on whatever Linux box is handy.

#### R1 — T1.8 AF_XDP → FeatureTable integration

**Gap:** `tests/afxdp_veth.rs` has five `t1_7_*` tests for UMEM allocation + `XdpSocket` creation, but no `t1_8_*` covering the ingest → parse → seqlock-write → reader loop.

**Steps:**
1. Write `tests/afxdp_t1_8_integration.rs` (or extend `afxdp_veth.rs`) gated on `skip_if_no_veth()` + `CAP_NET_ADMIN`.
2. Bring up `veth-tx` / `veth-rx` (already in `scripts/bootstrap_node.sh`).
3. Construct `Umem::new(256 frames, 4096 frame_size)` + `XdpSocket::bind("veth-rx", queue=0, XDP_COPY)`.
4. Allocate `FeatureTable::new(num_nodes=1024, feature_dim=128)` — the ingest target.
5. Spawn `ingest_loop` (from `crates/aether-stream/src/ingest.rs`) on its own thread with a crossbeam `Receiver<InboundFrame>`; route frames into a small parser that reads `(node_id: u32, features: [f32; 128])` and calls `FeatureTable::write_node`.
6. From the main thread, open a raw UDP socket on `veth-tx` and send 100 packets whose payload is `(node_id, known_features)` — use a deterministic payload so the reader can verify.
7. Join-with-timeout: wait up to 5s for 100 frames to be ingested.
8. Reader thread: for each sent `node_id`, call `FeatureTable::read_node` and assert bytes match the sent features.
9. Assert: zero dropped frames (COMPLETION ring drained == 100), zero torn reads, ingest loop exits cleanly on shutdown signal.

**Validates:** The full packet path — XdpSocket RX ring → FILL/COMPLETION frame lifecycle → parse → seqlock write → reader round-trip. Closes the last Tier 1 integration gap.

**Effort:** ~2-3 hrs. Needs a Linux box with veth support; no special hardware.

#### R2 — T4.1 billion-node graph loading ✅

`crates/aethergraph-core/tests/billion_node_load.rs`, gated on
`AETHER_BIG_GRAPH=<path>`. The test streams a synthetic uniform-degree CSR
directly to disk (no in-memory edge list), then loads via both default
(`OffsetsOnly`) and `HeaderOnly` paths.

**Run results (g4dn.xlarge, 1× instance-store NVMe):**
- Graph: 1B nodes × avg_degree 15 = **15B edges, 68.00 GB on disk**
- Generation: **417s (163 MB/s sustained)** — write-bound on instance-store
- `load_graph` (default → `OffsetsOnly`): **23.611s cold** — dominated by
  the monotonicity walk over 8 GB of offsets at ~350 MB/s disk reads
- `load_graph_mmap(..., HeaderOnly)`: **0.000s** — pure mmap + header parse
- Cold walk over 1000 sampled nodes (`degree()`): 3 ms post-validation

**Note vs original plan:** The ≥32 GB / avg_degree 50 estimate was wrong —
offsets are u64 (8B) and edges u32 (4B), so 1B nodes × deg 50 is 208 GB,
not 32 GB. Dropped avg_degree to 15 to fit g4dn.xlarge's 125 GB local NVMe.
The "near-zero mmap load" claim only holds for `HeaderOnly`; the default
validation walks offsets and is I/O-bound on cold caches.

#### R3 — T4.2 NeighborLoader throughput at scale

`crates/aethergraph-core/tests/billion_node_throughput.rs`, gated on
`AETHER_BIG_GRAPH=<path>`. Loads `AsyncCsrGraph`, warms up 20 batches of
1024 random nodes, measures 200 more. Reports batches/sec, seeds/sec, and
per-batch p50/p90/p99 latency.

Note: we did **not** go through `async_io_benchmark` / criterion. Criterion
registers every `bench_*` function's setup code (each allocates an 8 GB
`GraphFileView` offsets `Vec<u64>`) before honoring name filters, so even
with a single-bench filter it OOMs on a 15 GB box. The standalone test is
also much easier to interpret for a single scale-run.

**Run results (g4dn.xlarge, 1× instance-store NVMe, 1B × avg_degree 15):**
- `AsyncCsrGraph::load`: **30.6s** — 8 GB offsets + io_uring pool setup
- Batches/sec at bs=1024: **53.3**
- Seeds/sec: **54,619**
- Per-batch latency: mean **18.7 ms**, p50 23.7 ms, p90 24.0 ms, p99 25.8 ms

**Target was >500 batches/sec (>500K seeds/sec).** Below target by ~10× on
this box, for a structural reason: each batch triggers ~1024 random 4 KB
reads across the 60 GB edges file. g4dn.xlarge's instance-store NVMe caps at
~100K random-read IOPS; we saturate ~half of that. An i4i.4xlarge (2× NVMe
at ~500K IOPS/device) should clear the target — no code change needed, just
the right instance class. Filed as a hardware follow-up rather than a bug.

**Library fix shipped alongside the test:** `AsyncCsrGraph::load` used to
allocate 16 GB peak when reading offsets (a `Vec<u8>` buffer + `.collect()`
into `Vec<EdgeOffset>`), which OOMs on a 1B-node graph on anything under a
24 GB-RAM box. Now reads straight into `Vec<EdgeOffset>` via
`bytemuck::cast_slice_mut` — 8 GB peak, halved.

---

### GPU-blocked — on AWS quota approval or ConnectX access

These cannot be started until hardware is available. GPU vCPU quota requests have been filed (see "Item 10" above); ConnectX access is an alternative if the quota stays stuck.

#### R4 — Tier 2 (g4dn.xlarge, T4 GPU, SoftRoCE)

Five tests, documented in full above at T2.1–T2.5. Condensed:

1. **T2.1 CUDA seqlock kernel** — craft 3 slots (valid, head-odd torn, head≠tail torn), memcpy to VRAM, assert `retry_count == 2` and correct compaction.
2. **T2.2 DLPack → PyTorch** — PyO3 capsule round-trip; assert device/dtype/shape/values match.
3. **T2.3 SoftRoCE → host pinned → GPU memcpy** — end-to-end feature read into VRAM without `nvidia-peermem`.
4. **T2.4 `feature_source="rdma://..."` Python E2E** — NeighborLoader with a host-memory fallback shim.
5. **T2.5 Full PyG GCN training loop** — 5 epochs, real GNN model, no crashes.

**Unblockers:** approved AWS GPU vCPU quota (currently pending); g4dn.xlarge spot.

#### R5 — Tier 3 (p4d.24xlarge or p3dn.24xlarge, GPUDirect RDMA)

Five tests, T3.1–T3.5 above. Condensed:

1. **T3.1** `ibv_reg_mr` on a CUDA device pointer — proves `nvidia-peermem` is loaded.
2. **T3.2** GPUDirect loopback — server/client on same box, byte-for-byte match.
3. **T3.3** Torn-read detection under contention — 10M iters, assert ZERO corrupted features reach output.
4. **T3.4** Latency profiling — target <20 µs total gather, <10 µs RDMA + <5 µs CUDA.
5. **T3.5** Python training with live-updating features — the full demo.

**Unblockers:** GPU + EFA on the same instance (p4d/p5) AND quota; OR access to a ConnectX cluster on-prem (skips the EFA detour — existing RC code path already passes on SoftRoCE).

#### R6 — T4.3 Ray Data multi-GPU

**Depends on R4 or R5** (needs multiple GPUs in one box).

**Steps:**
1. On p3dn.24xlarge (8× V100) or p4d.24xlarge (8× A100): `ray.init()` with all GPUs visible.
2. `dataset = create_sampling_dataset(graph_path, num_neighbors=[15,10], batch_size=1024)`.
3. Iterate batches across all 8 GPU workers in parallel.
4. Record aggregate batches/sec and confirm scaling (should be ~6-8× single-GPU given PCIe contention).

**Validates:** Ray Data integration under real multi-GPU load.

---

### One-line summary

- **Not GPU-blocked, roughly a day of work:** R1, R2, R3.
- **GPU-blocked, burn a few hundred dollars of spot to clear:** R4, R5, R6.



## Roadmap — Research-Informed Optimizations

### GPUDirect Storage (NVIDIA GDS)

The RDMA path handles remote feature servers. For single-machine training,
NVIDIA GPUDirect Storage loads features from NVMe directly to GPU, bypassing
the CPU bounce buffer. Throughput goes from ~3 GB/s (CPU-bounce) to ~6 GB/s
(direct DMA). Complementary to the io_uring code — same NVMe, but the DMA
target is VRAM, not host memory.

Implementation: new feature-store variant using `cuFile` API from NVIDIA's
GDS SDK. Requires Linux + CUDA 11.4+ + NVMe with GDS driver support.

- **Impact:** 2x feature load on single-machine GPU training
- **Why:** Existing io_uring plumbing already handles async I/O; only the destination changes

### Historical Embeddings (GNNAutoScale, ICML 2021)

For multi-layer GNNs, the 2-hop fan-out means each batch touches 100K+ nodes.
Most of those nodes' representations did not change since the previous epoch.
GNNAutoScale caches intermediate layer embeddings and only recomputes the
subgraph that actually changed. Perfect fit for the `DynamicGraph` use case:
the C-tree tracks which nodes received new edges; only those nodes need
fresh embeddings, everyone else uses cached values.

This turns a 230M-node problem into a 100K-node problem per batch for
incremental training on live graphs.

- **Impact:** 5-10x for dynamic-graph training
- **Why:** The `DynamicGraph` C-tree already provides the dirty-node set

### Priority Summary

| Idea                         | Effort | Impact                               |
|------------------------------|--------|--------------------------------------|
| Feature quantization (fp16)  | 2 days | 2x throughput                        |
| Pre-sampling cache warmup    | 3 days | 1.5x for hot nodes                   |
| Cluster-aligned batching     | 1 day  | 1.2x on top of reorder               |
| GDS integration              | 3 days | 2x feature load (single-machine GPU) |
| Triple-buffered prefetch     | 4 days | Eliminates pipeline stalls           |
| Historical embeddings        | 1 week | 5-10x for dynamic graphs             |

Quick wins: feature quantization + cluster-aligned batching.
Moonshot: historical embeddings for dynamic-graph training.


PYTHON TESTING:

# Testing & Benchmarking

## Correctness Tests

```bash
uv run pytest tests/ -v
```

### What to verify
- Sampled subgraphs match PyG's NeighborLoader given same seeds
- Edge indices valid, node mappings correct
- Edge cases: isolated nodes, degree-0/1, high-degree hubs

## Benchmark Datasets

| Dataset | Nodes | Edges | Features | Source |
|---------|-------|-------|----------|--------|
| ogbn-products | 2.4M | 61M | 100 | OGB |
| Reddit | 233K | 114M | 602 | PyG |
| ogbn-papers100M | 111M | 1.6B | 128 | OGB |
| MAG240M | 244M | 1.7B | 768 | OGB |

Download:
```python
from ogb.nodeproppred import NodePropPredDataset
dataset = NodePropPredDataset(name="ogbn-papers100M")
```

## Key Metrics

| Metric | How to measure |
|--------|----------------|
| Sampling throughput | batches/sec, nodes/sec |
| Memory usage | `htop` RSS, should stay flat |
| NVMe utilization | `iostat -x 1` |
| GPU utilization | `nvidia-smi dmon` |
| Epoch time | wall clock |

## Benchmarks to Run

### 1. Sampling throughput (no training)
```python
import time
from aethergraph import Graph
from aethergraph.pytorch import NeighborLoader

graph = Graph.load("graph.bin")
loader = NeighborLoader(graph, num_neighbors=[15, 10], batch_size=1024)

start = time.perf_counter()
batches = sum(1 for _ in loader)
elapsed = time.perf_counter() - start
print(f"{batches / elapsed:.1f} batches/sec")
```

### 2. vs PyG in-memory baseline
```python
# PyG baseline (requires graph in RAM)
from torch_geometric.loader import NeighborLoader as PyGLoader
pyg_loader = PyGLoader(pyg_data, num_neighbors=[15, 10], batch_size=1024)
```

### 3. Rust microbenchmarks (Criterion)
Use targeted benches for fast iteration:
```bash
cargo bench -p aethergraph-core --bench graph_benchmarks -- --list
cargo bench -p aethergraph-core --bench graph_benchmarks -- \
  neighbor_sampling/single_hop/25 --sample-size 10 --quick --noplot --discard-baseline
cargo bench -p aethergraph-core --bench async_io_benchmark -- \
  sync_batch_1k_nodes --sample-size 10 --quick --noplot --discard-baseline
```

### 4. DDP scaling efficiency
```bash
# Single GPU baseline
python examples/03_pytorch_geometric_training.py

# Multi-GPU (expect near-linear speedup)
torchrun --nproc_per_node=2 examples/05_multi_gpu_ddp_training.py
torchrun --nproc_per_node=4 examples/05_multi_gpu_ddp_training.py
torchrun --nproc_per_node=8 examples/05_multi_gpu_ddp_training.py
```

## Claims to Validate

| Claim | Test |
|-------|------|
| Zero RAM overhead | Memory flat as graph grows |
| Faster than DistDGL | Compare epoch time on papers100M |
| Linear DDP scaling | 8 GPU ≈ 8x throughput |
| No cold start | First batch time vs PyG full load |

## Hardware Requirements

| Test | Minimum |
|------|---------|
| Correctness | Any machine |
| Basic benchmarks | 1 GPU + NVMe |
| DDP scaling | 8x GPU node |
| papers100M | 500GB+ NVMe |
| MAG240M | 2TB+ NVMe |

