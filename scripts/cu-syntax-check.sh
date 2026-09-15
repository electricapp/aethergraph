#!/usr/bin/env bash
# Parse every Tier A CUDA unit with clang++ -fsyntax-only.
#
# NVRTC compiles the kernels from source strings at runtime, so a typo in a
# .cu file is invisible to cargo. This runs the same concatenation the Rust
# modules build (common.cuh + the unit) through clang, with
# scripts/cu_host_shim.hpp standing in for the CUDA builtins.
#
# Targets nvptx64-nvidia-cuda, which is what makes clang accept the PTX
# register constraints (`l`, `r`, `f`); a host target rejects every one.
#
# Syntax and types only — the inline PTX bodies are opaque strings here and
# NVRTC passes them through unassembled too, so a bad opcode needs ptxas
# (`ptxas_assembles_every_unit`) and a wrong answer needs a GPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNELS="$REPO_ROOT/crates/aether-stream/src/gpu/kernels"
SHIM="$REPO_ROOT/scripts/cu_host_shim.hpp"

if ! command -v clang++ >/dev/null 2>&1; then
  echo "skipping: clang++ not on PATH"
  exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

FAILED=()
for unit in "$KERNELS"/*/*.cu; do
  name="$(basename "$unit")"
  out="$WORK/$name.cpp"
  {
    echo "#include \"$SHIM\""
    cat "$KERNELS/common.cuh"
    echo
    cat "$unit"
  } >"$out"
  if clang++ -std=c++17 -fsyntax-only -nostdinc++ -ffreestanding \
      --target=nvptx64-nvidia-cuda \
      -Wno-unused-function -Wno-pragma-once-outside-header \
      "$out" 2>"$WORK/$name.err"; then
    echo "ok   $name"
  else
    echo "FAIL $name"
    sed 's/^/     /' "$WORK/$name.err"
    FAILED+=("$name")
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo
  echo "CUDA syntax check failed: ${FAILED[*]}"
  exit 1
fi
echo "All CUDA units parse."
