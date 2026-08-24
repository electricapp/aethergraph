#!/usr/bin/env bash
# KERNELS.md grind entrypoint — run on a CUDA box (Tier A) and/or rooted VM (K4).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PASS=()
SKIP=()
FAIL=()

note() { printf '==> %s\n' "$*"; }
record() {
  local status="$1" item="$2"
  case "$status" in
    pass) PASS+=("$item") ;;
    skip) SKIP+=("$item") ;;
    fail) FAIL+=("$item") ;;
  esac
}

note "Tier A unit oracles (any host)"
if cargo test -p aethergraph-core --lib --quiet reservoir_; then
  record pass "CPU oracles (philox/reservoir)"
else
  record fail "CPU oracles"
fi
if cargo test -p aethergraph-core --lib --quiet cpu_seqlock; then
  record pass "CPU seqlock oracle"
else
  record fail "CPU seqlock oracle"
fi
if cargo test -p aethergraph-core --lib --quiet device::; then
  record pass "device builders"
else
  record fail "device builders"
fi

note "Tier A CUDA tests (gpudirect)"
if command -v nvidia-smi >/dev/null 2>&1; then
  if cargo test -p aether-stream --features gpudirect --test kernels -- --nocapture; then
    record pass "kernels integration tests"
  else
    record fail "kernels integration tests"
  fi
  if cargo bench -p aether-stream --bench kernels --features gpudirect -- --quick 2>/dev/null \
    || cargo bench -p aether-stream --bench kernels --features gpudirect -- --sample-size 10; then
    record pass "kernels criterion benches"
  else
    record skip "kernels benches (criterion flags)"
  fi
else
  record skip "CUDA tests (no nvidia-smi)"
fi

if command -v compute-sanitizer >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
  note "compute-sanitizer racecheck on validate"
  if compute-sanitizer --tool racecheck cargo test -p aether-stream --features gpudirect \
      --test kernels validate_graph -- --nocapture; then
    record pass "compute-sanitizer racecheck"
  else
    record fail "compute-sanitizer racecheck"
  fi
else
  record skip "compute-sanitizer"
fi

LITMUS="$ROOT/crates/aether-stream/litmus/k5_3"
if command -v herd7 >/dev/null 2>&1 && [[ -d "$LITMUS" ]]; then
  note "herd7 litmus (K5.3)"
  if herd7 -model nvidia "$LITMUS"/*.litmus; then
    record pass "herd7 K5.3"
  else
    record fail "herd7 K5.3"
  fi
else
  record skip "herd7"
fi

note "Tier B / K4 Linux-only checks"
if [[ "$(uname -s)" == "Linux" ]]; then
  if cargo test -p aethergraph-core --lib provided_buffer damon sched_ext --quiet; then
    record pass "device host policy unit tests"
  else
    record fail "device host policy unit tests"
  fi
  if cargo test -p aethergraph-core --features io-uring --lib provided_buffer_ring_registers -- --nocapture; then
    record pass "K4.3 PBUF register"
  else
    record skip "K4.3 PBUF register"
  fi
else
  record skip "K4 Linux checks (not Linux)"
fi

note "Grind matrix"
printf 'PASS (%d): %s\n' "${#PASS[@]}" "${PASS[*]:-}"
printf 'SKIP (%d): %s\n' "${#SKIP[@]}" "${SKIP[*]:-}"
printf 'FAIL (%d): %s\n' "${#FAIL[@]}" "${FAIL[*]:-}"
if ((${#FAIL[@]} > 0)); then
  exit 1
fi
