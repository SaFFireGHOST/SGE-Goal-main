#!/usr/bin/env bash
#
# Launch parallel test-set inference for BOTH ablations across the 6 inference
# accounts. Each account processes a disjoint shard of the 1087 samples with up
# to 10 GPU workers (Modal per-account GPU limit). Outputs are written locally
# into each sample dir as edited_image.jpg. Idempotent: re-running only fills gaps.
#
# Partition (1087 samples -> 3 disjoint shards):
#   A = 1..362, B = 363..725, C = 726..1087
#
#   i2 (original + initial mask):  acct1->A   acct2->B   acct3->C
#   i1 (original only):            acct4->A   acct5->B   acct6->C
#
# Auth: env-var only (MODAL_TOKEN_ID/SECRET). The global ~/.modal.toml is untouched.
#
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
I2="$ROOT/flux2_klein_calvin_bft_i2"
I1="$ROOT/flux2_klein_calvin_bft_i1"

# folder | token_id | token_secret | shard_start | shard_end | tag
# Credentials come from MODAL_TOKEN_ID / MODAL_TOKEN_SECRET (empty = default Modal profile).
JOBS=(
  "$I2|${MODAL_TOKEN_ID:-}|${MODAL_TOKEN_SECRET:-}|1|362|i2_acct1"
  "$I2|${MODAL_TOKEN_ID:-}|${MODAL_TOKEN_SECRET:-}|363|725|i2_acct2"
  "$I2|${MODAL_TOKEN_ID:-}|${MODAL_TOKEN_SECRET:-}|726|1087|i2_acct3"
  "$I1|${MODAL_TOKEN_ID:-}|${MODAL_TOKEN_SECRET:-}|1|362|i1_acct4"
  "$I1|${MODAL_TOKEN_ID:-}|${MODAL_TOKEN_SECRET:-}|363|725|i1_acct5"
  "$I1|${MODAL_TOKEN_ID:-}|${MODAL_TOKEN_SECRET:-}|726|1087|i1_acct6"
)

# Optional filter: pass "i2" or "i1" to launch only that group; default launches all.
FILTER="${1:-all}"

for job in "${JOBS[@]}"; do
  IFS='|' read -r folder tid tsec s e tag <<< "$job"
  case "$FILTER" in
    i2) [[ "$tag" == i2_* ]] || continue ;;
    i1) [[ "$tag" == i1_* ]] || continue ;;
    all) ;;
    *) ;;
  esac
  log="$folder/infer_${tag}.log"
  pidf="$folder/infer_${tag}.pid"
  echo "launching $tag : $folder shard [$s,$e] -> $log"
  (
    cd "$folder" || exit 1
    [[ -n "$tid" ]] && [[ -n "$tid" ]] && [[ -n "$tid" ]] && export MODAL_TOKEN_ID="$tid" MODAL_TOKEN_SECRET="$tsec"
    unset MODAL_PROFILE 2>/dev/null || true
    nohup python -m modal run modal_infer.py --shard-start "$s" --shard-end "$e" \
      > "$log" 2>&1 < /dev/null &
    echo $! > "$pidf"
  )
  sleep 1
done
echo "All requested inference jobs launched. Tail logs with: tail -f <folder>/infer_*.log"
