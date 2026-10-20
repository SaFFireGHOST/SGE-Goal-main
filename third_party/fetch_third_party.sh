#!/usr/bin/env bash
# Fetch the upstream repositories SGE-Goal depends on, pinned to the exact commits used in
# the paper, into the locations the code expects, and apply our local patches.
#
#   bash third_party/fetch_third_party.sh            # everything
#   bash third_party/fetch_third_party.sh xmem sam   # a subset: xmem sam susie clover
#
# Model weights are NOT fetched here — see third_party/README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES="$ROOT/third_party/patches"

XMEM_URL=https://github.com/hkchengrex/XMem.git;               XMEM_REV=f3b841d
SAM_URL=https://github.com/facebookresearch/segment-anything.git; SAM_REV=dca509f
SUSIE_URL=https://github.com/kvablack/susie.git;               SUSIE_REV=8177f63
CLOVER_URL=https://github.com/OpenDriveLab/CLOVER.git;         CLOVER_REV=4b2a75e

# fetch <url> <rev> <dest> [patch]
fetch() {
  local url=$1 rev=$2 dest=$3 patch=${4:-}
  if [[ -e "$dest" ]]; then
    echo "skip: $dest already exists"
    return
  fi
  echo "clone: $url@$rev -> ${dest#$ROOT/}"
  git clone --quiet "$url" "$dest"
  git -C "$dest" checkout --quiet "$rev"
  if [[ -n "$patch" ]]; then
    echo "patch: $(basename "$patch")"
    git -C "$dest" apply "$patch"
  fi
}

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(xmem sam susie clover)

for t in "${targets[@]}"; do
  case "$t" in
    xmem)
      fetch "$XMEM_URL" "$XMEM_REV" "$ROOT/data_pipeline/XMem" "$PATCHES/xmem.patch"
      fetch "$XMEM_URL" "$XMEM_REV" "$ROOT/evaluator/XMem" "$PATCHES/xmem.patch"
      ;;
    sam)
      fetch "$SAM_URL" "$SAM_REV" "$ROOT/data_pipeline/segment-anything" "$PATCHES/segment-anything_data_pipeline.patch"
      fetch "$SAM_URL" "$SAM_REV" "$ROOT/evaluator/segment-anything"
      ;;
    susie)
      fetch "$SUSIE_URL" "$SUSIE_REV" "$ROOT/baselines/susie/susie" "$PATCHES/susie.patch"
      ;;
    clover)
      fetch "$CLOVER_URL" "$CLOVER_REV" "$ROOT/baselines/clover/CLOVER"
      fetch "$CLOVER_URL" "$CLOVER_REV" "$ROOT/downstream/calvin/gr_mg/CLOVER-repo"
      ;;
    *)
      echo "unknown target: $t (expected: xmem sam susie clover)" >&2
      exit 1
      ;;
  esac
done
