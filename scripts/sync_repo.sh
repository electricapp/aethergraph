#!/usr/bin/env bash
# Rsync the aether-graph source tree to a remote host, skipping .git + build
# artifacts. Run from the repo root.
#
# Usage:
#   scripts/sync_repo.sh <user@host>
#   scripts/sync_repo.sh ubuntu@18.206.176.58
#
# Env:
#   KEY=~/.ssh/foo.pem  SSH key (default: ~/.ssh/phonon-coldstart-test.pem)
#   DEST=/home/ubuntu/aethergraph  remote path (default per user below)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <user@host> [rsync-args...]" >&2
    exit 2
fi

HOST=$1
shift

KEY=${KEY:-$HOME/.ssh/phonon-coldstart-test.pem}
DEST=${DEST:-/home/${HOST%%@*}/aethergraph}

rsync -az \
    --exclude='.git' \
    --exclude='target' \
    --exclude='node_modules' \
    --exclude='*.pem' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    "$@" \
    ./ "$HOST:$DEST/"
echo "synced repo to $HOST:$DEST"
