#!/usr/bin/env bash
# Run the AF_XDP → FeatureTable end-to-end test from `aether-stream`.
#
# Brings up a `veth-tx` / `veth-rx` pair if it's not already there, raises
# the locked-memory limit, and runs the test with `sudo -E` so it gets
# CAP_NET_ADMIN (XDP attach) and CAP_NET_RAW (AF_PACKET inject).
#
# Requires Linux + clang + libbpf headers. On macOS this script exits with
# a clear message — the test is `cfg`-gated to Linux anyway.

set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "this test is Linux-only (XDP / AF_XDP / aya)" >&2
    exit 0
fi

# Bring up veth peers if not present.
if ! ip link show veth-rx >/dev/null 2>&1; then
    echo ">> creating veth-tx / veth-rx"
    sudo ip link add veth-tx type veth peer name veth-rx
    sudo ip link set veth-tx up
    sudo ip link set veth-rx up
fi

ulimit -l unlimited

# Pass through any extra args (e.g. `--nocapture`).
exec sudo -E "$(command -v cargo)" test \
    --features xdp_bpf \
    -p aether-stream \
    --test afxdp_xdp_redirect_e2e \
    -- --nocapture "$@"
