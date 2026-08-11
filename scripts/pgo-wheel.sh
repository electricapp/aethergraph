#!/usr/bin/env bash
# Build a profile-guided wheel: instrument, run the workload, rebuild with
# the merged profile. When llvm-bolt is on PATH, additionally re-lay-out
# the shared object from a BOLT-instrumented second workload run and
# repack the wheel (linux only — BOLT needs ELF + --emit-relocs).
#
# The default workload is the pytest suite; point AETHERGRAPH_PGO_WORKLOAD
# at a heavier representative command for release builds.
#
# Requires: uv, rustup component llvm-tools. Optional: llvm-bolt, merge-fdata.
# Usage: scripts/pgo-wheel.sh [out_dir]      (out_dir relative to python/,
#                                             default dist-pgo)
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
OUT_DIR="${1:-dist-pgo}"
cd "$ROOT/python"

SYSROOT="$(rustc --print sysroot)"
PROFDATA="$(find "$SYSROOT" -name llvm-profdata -type f | head -1)"
if [ -z "$PROFDATA" ]; then
  echo "llvm-profdata not found; run: rustup component add llvm-tools" >&2
  exit 1
fi

PROF_DIR="$(mktemp -d)"
trap 'rm -rf "$PROF_DIR"' EXIT

workload() {
  if [ -n "${AETHERGRAPH_PGO_WORKLOAD:-}" ]; then
    $AETHERGRAPH_PGO_WORKLOAD
  else
    uv run --no-sync pytest tests/ -q --tb=short
  fi
}

echo "== stage 1: instrumented build + profile run"
uv sync --group dev --no-install-project
RUSTFLAGS="-Cprofile-generate=$PROF_DIR" uv run --no-sync maturin develop --release
LLVM_PROFILE_FILE="$PROF_DIR/aethergraph-%p-%m.profraw" workload
"$PROFDATA" merge -o "$PROF_DIR/merged.profdata" "$PROF_DIR"/*.profraw

BOLT="$(command -v llvm-bolt || true)"

echo "== stage 2: PGO-optimized wheel"
PGO_RUSTFLAGS="-Cprofile-use=$PROF_DIR/merged.profdata"
if [ -n "$BOLT" ]; then
  # BOLT needs relocations and symbols in the final object; the wheel is
  # stripped again after the re-layout.
  RUSTFLAGS="$PGO_RUSTFLAGS -Clink-arg=-Wl,--emit-relocs" \
    CARGO_PROFILE_RELEASE_STRIP=none \
    uv run --no-sync maturin build --release --out "$OUT_DIR"
else
  RUSTFLAGS="$PGO_RUSTFLAGS" uv run --no-sync maturin build --release --out "$OUT_DIR"
  echo "llvm-bolt not on PATH; emitted PGO-only wheel in python/$OUT_DIR"
  exit 0
fi

echo "== stage 3: BOLT re-layout"
WHL="$(ls "$OUT_DIR"/*.whl | head -1)"
WORK="$PROF_DIR/wheel"
uv run --no-sync --with wheel python -m wheel unpack --dest "$WORK" "$WHL"
UNPACKED="$(ls -d "$WORK"/*/ | head -1)"
SO="$(find "$UNPACKED" -name '_core*.so' | head -1)"

"$BOLT" "$SO" -instrument -o "$SO.inst" \
  -instrumentation-file="$PROF_DIR/bolt.fdata" -instrumentation-file-append-pid

# The dev venv extension is swapped for the instrumented object so the
# workload exercises it, then restored.
VENV_SO="$(find .venv -name '_core*.so' | head -1)"
cp "$VENV_SO" "$PROF_DIR/venv_so.orig"
cp "$SO.inst" "$VENV_SO"
workload
cp "$PROF_DIR/venv_so.orig" "$VENV_SO"

if command -v merge-fdata >/dev/null; then
  merge-fdata "$PROF_DIR"/bolt.fdata* > "$PROF_DIR/bolt.merged"
else
  cat "$PROF_DIR"/bolt.fdata* > "$PROF_DIR/bolt.merged"
fi

"$BOLT" "$SO" -data="$PROF_DIR/bolt.merged" -o "$SO.bolt" \
  -reorder-blocks=ext-tsp -reorder-functions=cdsort \
  -split-functions -split-all-cold -icf=1 -dyno-stats
strip "$SO.bolt" || true
mv "$SO.bolt" "$SO"
rm -f "$SO.inst"

uv run --no-sync --with wheel python -m wheel pack --dest-dir "$OUT_DIR" "$UNPACKED"
echo "BOLT+PGO wheel in python/$OUT_DIR"
