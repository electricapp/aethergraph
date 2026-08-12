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
if [ -n "$BOLT" ]; then
  # -instrument links the binary against BOLT's runtime archive, which BOLT
  # resolves as <prefix>/lib of the path it was invoked through — symlinks
  # are not followed, so reaching it through a foreign prefix loses the
  # archive. A miss only surfaces in stage 3, after stage 2 has already
  # built an unstripped wheel, so settle the question before that branch.
  BOLT_PREFIX="$(dirname "$(dirname "$BOLT")")"
  BOLT_LIB=""
  for cand in "$BOLT_PREFIX/lib/libbolt_rt_instr.a" \
              "$BOLT_PREFIX/lib64/libbolt_rt_instr.a"; do
    if [ -f "$cand" ]; then
      BOLT_LIB="$cand"
      break
    fi
  done
  if [ -z "$BOLT_LIB" ]; then
    echo "llvm-bolt found but no libbolt_rt_instr.a under $BOLT_PREFIX"
    BOLT=""
  fi
fi

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

# The extension the workload imports is swapped for the instrumented
# object, then restored. `maturin develop` on this mixed project installs
# a .pth pointing back at the source tree, so the live module is not under
# .venv at all — ask the interpreter where it is rather than guess.
DEV_SO="$(uv run --no-sync python -c 'import aethergraph._core as c; print(c.__file__)')"
if [ ! -f "$DEV_SO" ]; then
  echo "could not locate the development _core extension" >&2
  exit 1
fi
cp "$DEV_SO" "$PROF_DIR/dev_so.orig"
# Restore on any exit path; leaving the instrumented object installed
# would otherwise break the working tree if the workload fails.
trap 'cp "$PROF_DIR/dev_so.orig" "$DEV_SO" 2>/dev/null || true; rm -rf "$PROF_DIR"' EXIT
cp "$SO.inst" "$DEV_SO"
workload
cp "$PROF_DIR/dev_so.orig" "$DEV_SO"

# An instrumented run that produced no profile leaves nothing to re-lay-out
# against; keep the PGO wheel rather than failing the build over it.
if [ "$(find "$PROF_DIR" -maxdepth 1 -name 'bolt.fdata*' | wc -l)" -eq 0 ]; then
  echo "BOLT instrumentation produced no profile; keeping PGO-only wheel"
  exit 0
fi

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
