#!/usr/bin/env bash
# Audit the workflow files with zizmor at its widest setting.
#
# auditor persona, lowest severity and confidence: every finding zizmor can
# produce. .github/zizmor.yml carries the reason for each standing exception.
#
# The binary is fetched by release URL and checksummed rather than installed
# by version, so a re-tagged release or a compromised index cannot change what
# runs here. Same shape as the actionlint step in ci.yml.

set -euo pipefail

VER="1.30.1"

case "$(uname -s)/$(uname -m)" in
  Linux/x86_64)   ASSET="zizmor-x86_64-unknown-linux-gnu.tar.gz"
                  SHA="e65324f4430c2717591937edcec90ccbefaf14c174f8ec9415e03ca875b46e1a" ;;
  Linux/aarch64)  ASSET="zizmor-aarch64-unknown-linux-gnu.tar.gz"
                  SHA="7ff1dce33bdd18fd2a4affe63bdd47efcccca97b2cec1c1863ec26e9e2647540" ;;
  Darwin/arm64)   ASSET="zizmor-aarch64-apple-darwin.tar.gz"
                  SHA="e28d22b087f9ebb8d99da6e740d348c930f559961c7c3f12badda54f882195a2" ;;
  Darwin/x86_64)  ASSET="zizmor-x86_64-apple-darwin.tar.gz"
                  SHA="10e6b18b11ea07e515a16f0f0518c7b07527bc9977c1fd5698181ce7f3554202" ;;
  *) echo "no pinned zizmor build for $(uname -s)/$(uname -m)" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

curl -fsSL "https://github.com/zizmorcore/zizmor/releases/download/v${VER}/${ASSET}" \
  -o "$WORK/zizmor.tgz"
if command -v sha256sum >/dev/null 2>&1; then
  echo "${SHA}  $WORK/zizmor.tgz" | sha256sum -c -
else
  echo "${SHA}  $WORK/zizmor.tgz" | shasum -a 256 -c -
fi
tar -xzf "$WORK/zizmor.tgz" -C "$WORK"

exec "$(find "$WORK" -name zizmor -type f -perm -u+x | head -1)" \
  --config "$REPO_ROOT/.github/zizmor.yml" \
  --persona=auditor \
  --min-severity=informational \
  --min-confidence=low \
  --no-online-audits \
  "$REPO_ROOT/.github/workflows/"
