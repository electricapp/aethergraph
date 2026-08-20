#!/usr/bin/env bash
# Compile the Linux-only code on a non-Linux machine, before cloud CI does.
#
# The io_uring gather, NVMe passthrough, userfaultfd pager, memfd shared store,
# perf counters and NUMA placement are all `cfg(target_os = "linux")`. A plain
# `cargo check` on macOS compiles none of them, so a break there looks clean
# locally and only surfaces in CI. zig supplies the cross toolchain, so this
# needs no container and no emulation.
#
# Run it yourself before pushing; it is deliberately not a pre-commit or
# pre-push hook.
#
# It compiles; it does not execute. The binaries it builds are x86-64 Linux and
# will not run here, so a Linux-only test that compiles but fails still reaches
# CI. This closes the "does it build" gap, not the "does it work" one.
#
# Usage:
#   scripts/check-linux.sh            # type-check the Linux surface
#   scripts/check-linux.sh --clippy   # the same, as clippy with -D warnings
#   scripts/check-linux.sh --tests    # include test and bench targets
#   scripts/check-linux.sh --arm      # aarch64-linux instead of x86_64
#
# Requires zig (brew install zig) and:
#   rustup target add x86_64-unknown-linux-gnu   # (aarch64-... for --arm)
#
# To actually RUN the Linux tests, use scripts/linux-test.sh (lima VM).

set -euo pipefail

TARGET=x86_64-unknown-linux-gnu
ZIG_TARGET=x86_64-linux-gnu
for arg in "$@"; do
  if [ "$arg" = --arm ]; then
    TARGET=aarch64-unknown-linux-gnu
    ZIG_TARGET=aarch64-linux-gnu
  fi
done
export ZIG_TARGET
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

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

CMD=check
TARGETS=()
TRAILING=()
for arg in "$@"; do
  case "$arg" in
    --clippy) CMD=clippy; TRAILING=(-- -D warnings) ;;
    --tests)  TARGETS=(--all-targets) ;;
    --arm)    ;; # handled above
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Keep in step with the feature table in README.md.
CORE_FEATURES="io-uring nvme-passthru uffd shm perf numa zstd-tier"

# CI builds these one at a time, and a helper used only under some other
# feature is dead code in that build — which `-D warnings` rejects. Checking
# only the union would miss it, so check each on its own first.
for f in $CORE_FEATURES; do
  echo "==> $CMD aethergraph-core [$f]"
  cargo "$CMD" -p aethergraph-core --target "$TARGET" \
    --features "$f" ${TARGETS[@]+"${TARGETS[@]}"} ${TRAILING[@]+"${TRAILING[@]}"}
done

echo "==> $CMD aethergraph-core [no features]"
cargo "$CMD" -p aethergraph-core --target "$TARGET" \
  ${TARGETS[@]+"${TARGETS[@]}"} ${TRAILING[@]+"${TRAILING[@]}"}

echo "==> $CMD aethergraph-core [$CORE_FEATURES]"
cargo "$CMD" -p aethergraph-core --target "$TARGET" \
  --features "$CORE_FEATURES" ${TARGETS[@]+"${TARGETS[@]}"} ${TRAILING[@]+"${TRAILING[@]}"}

echo "==> $CMD aether-graph [wal, io-uring]"
cargo "$CMD" -p aether-graph --target "$TARGET" \
  --features "wal,io-uring" ${TARGETS[@]+"${TARGETS[@]}"} ${TRAILING[@]+"${TRAILING[@]}"}

echo "==> $CMD workspace [default features]"
cargo "$CMD" --workspace --target "$TARGET" ${TARGETS[@]+"${TARGETS[@]}"} ${TRAILING[@]+"${TRAILING[@]}"}

# The rustdoc gate CI runs: docs only exist for cfg'd-in code, so macOS
# doc runs never see the Linux-gated modules. rdma needs verbs headers
# and stays CI-only.
export RUSTDOCFLAGS="-D warnings"
echo "==> doc workspace [no features]"
cargo doc --workspace --no-deps --target "$TARGET"
echo "==> doc workspace [$CORE_FEATURES]"
cargo doc --workspace --no-deps --target "$TARGET" --features "$CORE_FEATURES"
echo "==> doc aether-graph [wal, io-uring]"
cargo doc -p aether-graph --no-deps --target "$TARGET" --features "wal,io-uring"

echo
echo "Linux cross-check passed."
