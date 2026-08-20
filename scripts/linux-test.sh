#!/usr/bin/env bash
# RUN the Linux-only tests locally, in a lima VM (native arm64 Linux via
# Virtualization.framework — no docker, no emulation). Complements
# scripts/check-linux.sh, which only proves the Linux surface compiles.
#
# The repo is mounted read-only at its host path; builds go to a VM-local
# CARGO_TARGET_DIR. First run provisions the VM (Ubuntu LTS, rustup, the
# apt deps CI installs) and takes a few minutes; later runs go straight
# to the tests.
#
# A real kernel means io_uring, uffd, memfd and friends actually execute.
# RDMA/XDP still need the SoftRoCE tier (see ROADMAP.md) — those tests
# self-skip. x86_64-specific paths still need CI: this VM is arm64.
#
# Usage:
#   scripts/linux-test.sh                                  # workspace default features
#   scripts/linux-test.sh -p aethergraph-core --features io-uring
#   scripts/linux-test.sh -p aether-graph --features "wal io-uring"
#
# Requires lima (brew install lima).

set -euo pipefail

VM=aethergraph
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v limactl >/dev/null 2>&1; then
  echo "error: limactl not found. Install it with: brew install lima" >&2
  exit 1
fi

if ! limactl list --format '{{.Name}}' 2>/dev/null | grep -qx "$VM"; then
  echo "==> creating lima VM '$VM' (one-time)"
  limactl create --name="$VM" --tty=false template://ubuntu-lts
fi

if [ "$(limactl list --format '{{.Status}}' "$VM")" != "Running" ]; then
  limactl start "$VM"
fi

# One-time provisioning, keyed on rustup's presence.
if ! limactl shell "$VM" bash -c 'command -v cargo >/dev/null 2>&1 || test -x "$HOME/.cargo/bin/cargo"'; then
  echo "==> provisioning toolchain + apt deps (one-time)"
  limactl shell "$VM" bash -c 'sudo apt-get update && sudo apt-get install -y --no-install-recommends build-essential clang pkg-config libbpf-dev libibverbs-dev libnuma-dev curl ca-certificates'
  limactl shell "$VM" bash -c 'curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal'
fi

ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
  ARGS=(--workspace)
fi

echo "==> cargo test ${ARGS[*]} (inside $VM)"
limactl shell --workdir "$REPO_ROOT" "$VM" bash -c \
  'export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR="$HOME/aethergraph-target"; exec cargo test "$@"' \
  -- "${ARGS[@]}"
