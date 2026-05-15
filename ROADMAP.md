# AetherGraph Test Plan

Outstanding test work only. Delete rows from these tables as they land.

## Status

| ID  | What                       | Code written? | Blocker to run                     |
| --- | -------------------------- | ------------- | ---------------------------------- |
| R1  | T1.8 AF_XDP → FeatureTable | Yes           | Linux + veth + clang + caps        |
| R4  | Tier 2 tests (T2.2–T2.5)   | Yes           | g4dn.xlarge spot (CUDA, AWS quota) |
| R5  | Tier 3 tests (T3.1–T3.5)   | Yes           | GPU + RDMA same box, or ConnectX   |
| R6  | T4.3 Ray Data multi-GPU    | Yes           | Multi-GPU box (after R4 or R5)     |

All test code is written and gated. What's left is _running_ it on the right
hardware.

---

## R1 — T1.8 AF_XDP → FeatureTable

Closes the last Tier-1 integration gap. Loads the bundled XDP redirect program
(`crates/aether-stream/bpf/src/xsk_redirect.c`) via `aya`, attaches it to
`veth-rx`, inserts the AF_XDP socket fd into `xsks_map[0]`, then injects 100 raw
Ethernet frames from `veth-tx` and asserts they all land in a `FeatureTable`.

| File                                                   | Notes                                                                              |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `crates/aether-stream/bpf/src/xsk_redirect.c`          | Minimal XDP redirect program (~25 LOC)                                             |
| `crates/aether-stream/build.rs`                        | Compiles the BPF object with `clang --target=bpf` when `--features xdp_bpf` is set |
| `crates/aether-stream/tests/afxdp_xdp_redirect_e2e.rs` | End-to-end test, gated on `linux + xdp_bpf`                                        |

### Run on any Linux box (no GPU)

```bash
sudo apt-get install -y clang libbpf-dev
sudo ip link add veth-tx type veth peer name veth-rx
sudo ip link set veth-tx up && sudo ip link set veth-rx up
ulimit -l unlimited
sudo -E cargo test --features xdp_bpf -p aether-stream \
    --test afxdp_xdp_redirect_e2e -- --nocapture
```

`CAP_NET_ADMIN` (XDP attach) and `CAP_NET_RAW` (`AF_PACKET` inject) are both
required — easiest path is `sudo`.

---

## R4 — Tier 2 tests (g4dn.xlarge, T4 GPU + SoftRoCE)

Spot ~$0.16/hr. Unblocker: AWS GPU vCPU quota approval.

```bash
cargo test --release -p aether-stream --features "rdma gpudirect" -- --nocapture
cd python && uv run pytest tests/test_dlpack.py \
    tests/test_rdma_feature_source.py tests/test_gcn_training_loop.py -v
```

| ID   | File                                                        |
| ---- | ----------------------------------------------------------- |
| T2.2 | `python/tests/test_dlpack.py`                               |
| T2.3 | `crates/aether-stream/tests/softroce_host_pinned_to_gpu.rs` |
| T2.4 | `python/tests/test_rdma_feature_source.py`                  |
| T2.5 | `python/tests/test_gcn_training_loop.py`                    |

T2.4 spawns `cargo run -p aether-stream --example rdma_feature_server` as a
subprocess and reads its `READY port=<…>` line to synchronize.

---

## R5 — Tier 3 tests (GPUDirect RDMA)

Same feature flags as R4. Run with:

```bash
cargo test --release -p aether-stream --features "rdma gpudirect" -- --nocapture
cargo test --release -p aether-stream --features "rdma gpudirect" \
    --test gpudirect_latency -- --ignored --nocapture
cd python && uv run pytest tests/test_rdma_live_training.py -v
```

| ID   | File                                                       | Notes       |
| ---- | ---------------------------------------------------------- | ----------- |
| T3.1 | `crates/aether-stream/tests/ibv_reg_mr_on_cuda.rs`         |             |
| T3.2 | `crates/aether-stream/tests/gpudirect_loopback.rs`         |             |
| T3.3 | `crates/aether-stream/tests/gpudirect_torn_read_stress.rs` |             |
| T3.4 | `crates/aether-stream/tests/gpudirect_latency.rs`          | `#[ignore]` |
| T3.5 | `python/tests/test_rdma_live_training.py`                  |             |

T3.5 reuses the `rdma_feature_server` example with `--live-rate` set so the
server overwrites features in the background while training runs.

---

## R6 — T4.3 Ray Data multi-GPU

Skips unless `torch.cuda.device_count() >= 2`. Builds a sampling dataset at
parallelism = GPU count and asserts aggregate batches/sec is at least 1.5× the
single-worker baseline.

| File                                 |
| ------------------------------------ |
| `python/tests/test_ray_multi_gpu.py` |

---

## CI surrogates (catch drift without hardware)

| Job                      | What it does                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `markdown-format`        | `bunx prettier --check '**/*.md'`                                                                                              |
| `gpudirect-check`        | Compile-only `cargo check` of the rdma + gpudirect path on an `nvidia/cuda:12.5.0-devel-ubuntu22.04` container — no GPU needed |
| `rust` matrix (existing) | macOS + Ubuntu defaults; Ubuntu + `rdma` feature                                                                               |

---

## Environment gotchas

| Pitfall                                           | What to do                                                                                                                  |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| AL2023 kernel 6.18 does not ship `rdma_rxe`       | Use Ubuntu 22.04/24.04, or AL2023 kernel 6.1 + `kernel-modules-extra`                                                       |
| SoftRoCE on `lo` does not work                    | Bind `rxe` to a real ethernet device — loopback has no MAC for ARP/GID resolution                                           |
| GID 0 is link-local IPv6 on RoCE                  | Pick the IPv4-mapped GID (typically index 1; confirm via `show_gids`). `RdmaContext::open` requires an explicit `gid_index` |
| `ulimit -l unlimited` required                    | Otherwise `ibv_reg_mr` / `mlock` fail                                                                                       |
| EFA needs SG self-reference on ingress AND egress | A generic egress `0.0.0.0/0` silently drops EFA                                                                             |
| `cargo test -p ... -p ...` shares compile         | One crate's build failure cancels sibling test runs mid-build                                                               |
| `xdp_bpf` feature needs `clang` + `libbpf-dev`    | `build.rs` invokes clang with `--target=bpf` to compile the redirect program                                                |

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
