#!/usr/bin/env bash
# Run the cloud CI compile/lint gates locally, before pushing.
#
# Mirrors .github/workflows/ci.yml. Stage labels match CI job names, so a
# failure here names the job that would fail there. Host-native gates run
# as-is; the Linux-gated ones (io_uring, NVMe passthrough, userfaultfd,
# memfd, perf, NUMA, XDP) cross-compile through zig — no container, no
# emulation.
#
# Gates needing headers zig cannot supply (linux/bpf.h, infiniband/verbs.h,
# CUDA) are delegated to the lima VM, so one run covers every CI job.
#
# It compiles; it does not execute. To RUN the Linux tests use
# scripts/linux-test.sh.
#
# Usage:
#   scripts/check-linux.sh              # everything below
#   scripts/check-linux.sh --host       # host gates only (fmt/clippy/doc)
#   scripts/check-linux.sh --cross      # Linux cross gates only
#   scripts/check-linux.sh --arm        # cross against aarch64-linux
#   scripts/check-linux.sh --quick      # skip --all-targets (faster)
#   scripts/check-linux.sh --vm         # header-dependent gates only (lima)
#   scripts/check-linux.sh --no-vm      # skip the lima gates
#
# Requires zig (brew install zig) and:
#   rustup target add x86_64-unknown-linux-gnu   # (aarch64-... for --arm)

set -euo pipefail

TARGET=x86_64-unknown-linux-gnu
ZIG_TARGET=x86_64-linux-gnu
RUN_HOST=1
RUN_CROSS=1
RUN_VM=1
VM=aethergraph
ALL_TARGETS=(--all-targets)

for arg in "$@"; do
  case "$arg" in
    --arm)    TARGET=aarch64-unknown-linux-gnu; ZIG_TARGET=aarch64-linux-gnu ;;
    --host)   RUN_CROSS=0; RUN_VM=0 ;;
    --cross)  RUN_HOST=0; RUN_VM=0 ;;
    --vm)     RUN_HOST=0; RUN_CROSS=0 ;;
    --no-vm)  RUN_VM=0 ;;
    --quick)  ALL_TARGETS=() ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done
export ZIG_TARGET

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ci.yml sets RUSTFLAGS workflow-wide, so every plain `cargo check` there is
# a warnings gate too; the rustdoc steps add RUSTDOCFLAGS on top.
export RUSTFLAGS="${RUSTFLAGS:--D warnings}"
export RUSTDOCFLAGS="${RUSTDOCFLAGS:--D warnings}"

FAILED=()
stage() { # stage <ci-job-label> <cmd...>
  local label="$1"; shift
  echo "==> [$label] $*"
  if ! "$@"; then
    FAILED+=("$label: $*")
  fi
}

vm_stage() { # vm_stage <ci-job-label> <cargo-cmd...>
  local label="$1"; shift
  echo "==> [$label] (vm) $*"
  if ! limactl shell --workdir "$REPO_ROOT" "$VM" bash -c \
      'export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR="$HOME/aethergraph-target" \
         RUSTFLAGS="-D warnings"; exec "$@"' \
      -- "$@"; then
    FAILED+=("$label: $*")
  fi
}

# Feature matrix from ci.yml's rust job.
MATRIX_FEATURES="rdma"
# Keep in step with the feature table in README.md.
CORE_FEATURES="io-uring nvme-passthru uffd shm perf numa zstd-tier"

if [ "$RUN_HOST" = 1 ]; then
  echo "### host gates (rust macos/ubuntu-stable-default rows)"
  stage "rust:rustfmt" cargo fmt --all -- --check
  stage "rust:clippy (no features)" \
    cargo clippy --workspace --all-targets --no-deps -- -D warnings
  stage "rust:clippy (with features)" \
    cargo clippy --workspace --all-targets --no-deps --features "$MATRIX_FEATURES" -- -D warnings
  stage "rust:rustdoc (no features)" cargo doc --workspace --no-deps
  stage "rust:rustdoc (with features)" \
    cargo doc --workspace --no-deps --features "$MATRIX_FEATURES"
  stage "rust:bench compile check" cargo check --workspace --benches

  if command -v prettier >/dev/null 2>&1; then
    stage "prettier markdown" prettier --check '**/*.md'
  elif command -v bunx >/dev/null 2>&1; then
    stage "prettier markdown" bunx prettier --check '**/*.md'
  else
    echo "!! skipping prettier markdown (install prettier or bun)"
  fi
fi

if [ "$RUN_CROSS" = 1 ]; then
  if ! command -v zig >/dev/null 2>&1; then
    echo "error: zig not found. Install it with: brew install zig" >&2
    exit 1
  fi
  if ! rustup target list --installed | grep -qx "$TARGET"; then
    echo "error: Rust target $TARGET not installed. Add it with:" >&2
    echo "  rustup target add $TARGET" >&2
    exit 1
  fi

  ZIGCC="$REPO_ROOT/scripts/zigcc"
  chmod +x "$ZIGCC" 2>/dev/null || true
  export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$ZIGCC"
  export CC_x86_64_unknown_linux_gnu="$ZIGCC"
  export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$ZIGCC"
  export CC_aarch64_unknown_linux_gnu="$ZIGCC"

  echo
  echo "### cross gates ($TARGET)"

  # CI builds these one at a time, and a helper used only under some other
  # feature is dead code in that build — which `-D warnings` rejects.
  for f in $CORE_FEATURES; do
    stage "rust:clippy [core/$f]" \
      cargo clippy -p aethergraph-core --target "$TARGET" \
        --features "$f" ${ALL_TARGETS[@]+"${ALL_TARGETS[@]}"} -- -D warnings
  done

  stage "rust:clippy [core/none]" \
    cargo clippy -p aethergraph-core --target "$TARGET" \
      ${ALL_TARGETS[@]+"${ALL_TARGETS[@]}"} -- -D warnings
  stage "rust:clippy [core/all]" \
    cargo clippy -p aethergraph-core --target "$TARGET" \
      --features "$CORE_FEATURES" ${ALL_TARGETS[@]+"${ALL_TARGETS[@]}"} -- -D warnings
  stage "rust:clippy [workspace]" \
    cargo clippy --workspace --target "$TARGET" \
      ${ALL_TARGETS[@]+"${ALL_TARGETS[@]}"} -- -D warnings

  # Dedicated per-feature jobs. These clippy aethergraph-py too — a py-side
  # break under a Linux feature is invisible to the core-only checks above.
  stage "numa placement" \
    cargo clippy -p aethergraph-core --target "$TARGET" \
      --features numa --all-targets -- -D warnings
  stage "userfaultfd pager" \
    cargo clippy -p aethergraph-core -p aethergraph-py --target "$TARGET" \
      --all-targets --features uffd -- -D warnings
  stage "perf counters" \
    cargo clippy -p aethergraph-core -p aethergraph-py --target "$TARGET" \
      --all-targets --features perf -- -D warnings
  stage "shm shared store" \
    cargo clippy -p aethergraph-core -p aethergraph-py --target "$TARGET" \
      --all-targets --features shm -- -D warnings
  stage "nvme passthrough" \
    cargo clippy -p aethergraph-core --target "$TARGET" \
      --all-targets --features nvme-passthru -- -D warnings
  stage "zstd cold tier" \
    cargo clippy --workspace --target "$TARGET" \
      --all-targets --features aethergraph-core/zstd-tier -- -D warnings

  stage "wal recovery [wal]" \
    cargo clippy -p aether-graph --target "$TARGET" \
      --features wal ${ALL_TARGETS[@]+"${ALL_TARGETS[@]}"} -- -D warnings
  stage "wal recovery [wal io-uring]" \
    cargo clippy -p aether-graph --target "$TARGET" \
      --features "wal,io-uring" ${ALL_TARGETS[@]+"${ALL_TARGETS[@]}"} -- -D warnings

  # Docs only exist for cfg'd-in code, so host doc runs never see these.
  stage "rust:rustdoc [cross/none]" cargo doc --workspace --no-deps --target "$TARGET"
  stage "rust:rustdoc [cross/core]" \
    cargo doc --workspace --no-deps --target "$TARGET" --features "$CORE_FEATURES"
  stage "wal recovery:rustdoc" \
    cargo doc -p aether-graph --no-deps --target "$TARGET" --features "wal,io-uring"
fi

# Gates needing headers zig does not ship (linux/bpf.h, infiniband/verbs.h,
# CUDA) run in the lima VM instead. Same headers CI installs; arm64 rather
# than x86_64, which still catches API and type breaks.
if [ "$RUN_VM" = 1 ]; then
  echo
  if ! command -v limactl >/dev/null 2>&1; then
    echo "!! skipping VM gates (brew install lima), CI-only: xdp_bpf, mlx5dv, rdma, gpudirect"
  else
    "$REPO_ROOT/scripts/linux-test.sh" --ensure-vm
    echo "### vm gates (header-dependent CI jobs)"
    vm_stage "rust:xdp_bpf compile check" \
      cargo check -p aether-stream --tests --features xdp_bpf
    vm_stage "rust:mlx5dv compile check" \
      cargo check -p aether-stream --tests --features mlx5dv
    vm_stage "cargo check (rdma + gpudirect)" \
      cargo check -p aether-stream --tests --features "rdma gpudirect"
    vm_stage "cargo check (rdma + gpudirect) [gdrcopy]" \
      cargo check -p aether-stream --features "rdma gpudirect gdrcopy"
    vm_stage "cargo check (rdma + gpudirect) [py]" \
      cargo check -p aethergraph-py --features gpudirect
    vm_stage "rust:clippy [rdma]" \
      cargo clippy -p aether-stream --all-targets --features rdma -- -D warnings
  fi
fi

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All local CI gates passed."
else
  echo "FAILED (${#FAILED[@]}):"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi
