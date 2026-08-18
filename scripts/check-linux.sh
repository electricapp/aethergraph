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
# Usage:
#   scripts/check-linux.sh            # type-check the Linux surface
#   scripts/check-linux.sh --clippy   # the same, as clippy with -D warnings
#   scripts/check-linux.sh --tests    # include test and bench targets
#
# Requires zig (brew install zig) and:
#   rustup target add x86_64-unknown-linux-gnu

set -euo pipefail

TARGET=x86_64-unknown-linux-gnu
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

CMD=check
TARGETS=()
TRAILING=()
for arg in "$@"; do
  case "$arg" in
    --clippy) CMD=clippy; TRAILING=(-- -D warnings) ;;
    --tests)  TARGETS=(--all-targets) ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Keep in step with the feature table in README.md.
CORE_FEATURES="io-uring nvme-passthru uffd shm perf numa zstd-tier"

echo "==> $CMD aethergraph-core [$CORE_FEATURES]"
cargo "$CMD" -p aethergraph-core --target "$TARGET" \
  --features "$CORE_FEATURES" ${TARGETS[@]+"${TARGETS[@]}"} ${TRAILING[@]+"${TRAILING[@]}"}

echo "==> $CMD aether-graph [wal, io-uring]"
cargo "$CMD" -p aether-graph --target "$TARGET" \
  --features "wal,io-uring" ${TARGETS[@]+"${TARGETS[@]}"} ${TRAILING[@]+"${TRAILING[@]}"}

echo "==> $CMD workspace [default features]"
cargo "$CMD" --workspace --target "$TARGET" ${TARGETS[@]+"${TARGETS[@]}"} ${TRAILING[@]+"${TRAILING[@]}"}

echo
echo "Linux cross-check passed."
