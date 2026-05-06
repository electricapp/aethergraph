#!/usr/bin/env bash
# Bootstrap a fresh AWS Ubuntu 24.04 instance for aether-graph testing.
#
# Installs (in order):
#   - build toolchain (gcc, pkg-config, cmake, clang dev)
#   - ibverbs / rdma-core dev headers (SoftRoCE + EFA both need libibverbs)
#   - optional AWS EFA driver (`aws-efa-installer`) if --efa is passed
#   - optional SoftRoCE (rdma_rxe) if --rxe is passed
#   - optional NVIDIA driver + CUDA toolkit if --cuda is passed (g4dn/p3/p4)
#   - optional mount of instance-store NVMe at /nvme if --nvme is passed
#   - Rust 1.94.1 via rustup (minimal profile)
#
# Idempotent: re-running skips apt packages already installed and detects
# an existing rustup toolchain. Safe to rerun after a base-AMI refresh.
#
# Usage:
#   scripts/bootstrap_node.sh                # build deps + ibverbs + Rust
#   scripts/bootstrap_node.sh --efa          # also install AWS EFA driver
#   scripts/bootstrap_node.sh --rxe          # also load SoftRoCE on ens5
#   scripts/bootstrap_node.sh --cuda         # also install nvidia-driver + CUDA
#   scripts/bootstrap_node.sh --nvme         # mount instance-store NVMe at /nvme
#   scripts/bootstrap_node.sh --rxe --cuda --nvme  # g4dn.xlarge for R1-R4
#
# Typical remote invocation from the workstation:
#   rsync -a scripts/ ubuntu@<ip>:/home/ubuntu/aether-scripts/
#   ssh ubuntu@<ip> bash /home/ubuntu/aether-scripts/bootstrap_node.sh --efa

set -euo pipefail

WANT_EFA=0
WANT_RXE=0
WANT_CUDA=0
WANT_NVME=0
RUST_TOOLCHAIN=${RUST_TOOLCHAIN:-1.94.1}
RXE_LINK=${RXE_LINK:-ens5}
NVME_DEV=${NVME_DEV:-/dev/nvme1n1}
NVME_MNT=${NVME_MNT:-/nvme}

while [ $# -gt 0 ]; do
    case "$1" in
        --efa) WANT_EFA=1 ;;
        --rxe) WANT_RXE=1 ;;
        --cuda) WANT_CUDA=1 ;;
        --nvme) WANT_NVME=1 ;;
        --toolchain) shift; RUST_TOOLCHAIN=$1 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done

echo "[bootstrap] apt update + build deps"
sudo apt-get update -q >/dev/null
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    build-essential pkg-config libclang-dev cmake libudev-dev libssl-dev \
    autoconf automake libtool curl wget git \
    rdma-core libibverbs-dev ibverbs-utils infiniband-diags \
    >/dev/null

if [ "$WANT_EFA" -eq 1 ]; then
    if [ ! -e /usr/lib/aarch64-linux-gnu/libefa.so ] && [ ! -e /usr/lib/x86_64-linux-gnu/libefa.so ]; then
        echo "[bootstrap] installing AWS EFA driver"
        tmpdir=$(mktemp -d)
        (
            cd "$tmpdir"
            wget -q https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
            tar xf aws-efa-installer-latest.tar.gz
            cd aws-efa-installer
            # Pingpong test fails on single-node boxes (no peer); -y accepts defaults.
            sudo ./efa_installer.sh -y >"$tmpdir/efa.log" 2>&1 || true
        )
        echo "[bootstrap] EFA driver install log: $tmpdir/efa.log"
    else
        echo "[bootstrap] libefa already present, skipping EFA driver install"
    fi
fi

if [ "$WANT_NVME" -eq 1 ]; then
    if [ ! -b "$NVME_DEV" ]; then
        echo "[bootstrap] --nvme requested but $NVME_DEV not present; skipping"
    else
        # Format only if blank. Instance-store NVMe drops data on stop/start but
        # survives reboot, so the xfs is typically still there after a --cuda
        # reboot cycle — just remount.
        if ! sudo blkid "$NVME_DEV" >/dev/null 2>&1; then
            echo "[bootstrap] formatting $NVME_DEV as xfs"
            sudo apt-get install -y -q xfsprogs >/dev/null
            sudo mkfs.xfs -f -q "$NVME_DEV"
        fi
        sudo mkdir -p "$NVME_MNT"
        if ! mountpoint -q "$NVME_MNT"; then
            sudo mount "$NVME_DEV" "$NVME_MNT"
        fi
        sudo chown "$(id -u):$(id -g)" "$NVME_MNT"
        echo "[bootstrap] $NVME_DEV mounted at $NVME_MNT ($(df -h "$NVME_MNT" | awk 'NR==2 {print $4}') free)"
    fi
fi

if [ "$WANT_RXE" -eq 1 ]; then
    # Ubuntu 24.04 AWS kernels (6.17+) ship rdma_rxe in linux-modules-extra,
    # not the base linux-modules package. Install it up front so modprobe +
    # `rdma link add` below don't die under `set -e`.
    if ! modinfo rdma_rxe >/dev/null 2>&1; then
        echo "[bootstrap] installing linux-modules-extra for rdma_rxe"
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
            "linux-modules-extra-$(uname -r)" >/dev/null
    fi
    echo "[bootstrap] loading SoftRoCE on $RXE_LINK"
    sudo modprobe rdma_rxe
    if ! rdma link show | grep -q rxe0; then
        sudo rdma link add rxe0 type rxe netdev "$RXE_LINK"
    fi
fi

if [ "$WANT_CUDA" -eq 1 ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[bootstrap] installing nvidia driver + CUDA toolkit"
        # Ubuntu's `ubuntu-drivers` picks the right driver for the detected GPU.
        # On g4dn/p3/p4 this is 580-server; pulls kernel modules + CUDA runtime.
        # nvcc comes from nvidia-cuda-toolkit; nvidia-smi from nvidia-utils-*.
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
            ubuntu-drivers-common nvidia-cuda-toolkit >/dev/null
        sudo ubuntu-drivers install --gpgpu >/dev/null
        # `ubuntu-drivers --gpgpu` omits nvidia-smi (it's in the non-headless
        # utils pkg). Install the server variant matching the driver series.
        driver_series=$(dpkg -l 2>/dev/null \
            | awk '/^ii[[:space:]]+nvidia-headless-no-dkms-[0-9]+-server-open/{print $2}' \
            | head -1 \
            | sed 's/.*-\([0-9]*\)-server-open/\1/')
        if [ -n "$driver_series" ]; then
            sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
                "nvidia-utils-${driver_series}-server" >/dev/null
        fi
        echo "[bootstrap] CUDA installed — reboot required for driver to load"
    fi
fi

if ! command -v rustc >/dev/null 2>&1 || ! rustc --version | grep -q "$RUST_TOOLCHAIN"; then
    echo "[bootstrap] installing Rust $RUST_TOOLCHAIN"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain "$RUST_TOOLCHAIN" --profile minimal >/dev/null
fi
source "$HOME/.cargo/env"

echo "[bootstrap] done:"
echo "  rust = $(rustc --version)"
echo "  ibv_devices:"
ibv_devices 2>&1 | sed 's/^/    /'
if [ "$WANT_CUDA" -eq 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "  nvidia-smi:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | sed 's/^/    /'
fi
