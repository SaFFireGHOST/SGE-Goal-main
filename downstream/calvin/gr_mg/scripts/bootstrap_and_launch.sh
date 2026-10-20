#!/bin/bash
# bootstrap_and_launch.sh — bootstrap one account and launch its eval runs.
# Usage: ./scripts/bootstrap_and_launch.sh <account> <method1> [method2] [--gt]
# The --gt flag makes the last method use --use-hf-parquet-goal and 10 shards.
set -euo pipefail

ACCOUNT="$1"; shift
METHODS=()
GT=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gt) GT=true; shift;;
    *) METHODS+=("$1"); shift;;
  esac
done

if [[ ${#METHODS[@]} -eq 0 ]]; then
  echo "Usage: $0 <account> <method1> [method2] [--gt]"; exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOGDIR="$ROOT/.runlogs/orchestrator"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/${ACCOUNT}__bootstrap.log"
TOPLOG="$LOGDIR/${ACCOUNT}__bootstrap_top.log"
export MODAL_PROFILE="$ACCOUNT"

log() { echo "[boot] $(date '+%H:%M:%S') $ACCOUNT: $*" | tee -a "$TOPLOG"; }

run_modal() {
  local tag="$1"; shift
  log "running: modal $*"
  echo "=== $(date -Iseconds) $ modal $* ===" >> "$LOG"
  if ! modal "$@" >> "$LOG" 2>&1; then
    log "FAILED: $tag (rc=$?)"
    return 1
  fi
  log "OK: $tag"
}

# --- Step 1: CALVIN data ---
if modal volume ls rm-policy-calvin-data / 2>/dev/null | grep -q task_D_D; then
  log "skip: calvin-data already has task_D_D"
else
  run_modal "fetch_calvin_d" run modal_app.py::fetch_calvin_d || exit 1
fi

# --- Step 2: Checkpoints ---
if modal volume ls rm-policy-ckpts / 2>/dev/null | grep -q gr_mg; then
  log "skip: ckpts already present"
else
  run_modal "fetch_grmg_ckpts" run modal_app.py::fetch_grmg_ckpts --policy --no-goal-gen --no-mae || exit 1
fi

# --- Step 3: Mapping ---
MAPPING="$ROOT/.runlogs/hf_calvin_mapping.json"
if modal volume ls rm-policy-runs / 2>/dev/null | grep -q hf_calvin_mapping; then
  log "skip: mapping already uploaded"
else
  if [[ ! -f "$MAPPING" ]]; then
    log "FATAL: mapping JSON missing at $MAPPING"; exit 1
  fi
  run_modal "upload_mapping" run modal_app.py::upload_mapping --local-path "$MAPPING" || exit 1
fi

# --- Step 4: HF parquet ---
if modal volume ls rm-policy-hf-cache / 2>/dev/null | grep -q parquet; then
  log "skip: HF parquet already cached"
elif $GT; then
  run_modal "download_hf_parquet" run modal_app.py::download_hf_parquet || exit 1
else
  # Non-GT accounts still need the parquet for eval_method_shard
  run_modal "download_hf_parquet" run modal_app.py::download_hf_parquet || exit 1
fi

# --- Step 5: Upload chain_skeletons.json ---
if modal volume ls rm-policy-runs / 2>/dev/null | grep -q chain_skeletons; then
  log "skip: chain_skeletons already uploaded"
else
  run_modal "upload_chain_skeletons" run modal_app.py::upload_chain_skeletons || log "WARN: chain_skeletons upload failed (non-fatal)"
fi

# --- Step 6: Per-method eval data ---
EVALS_ROOT="$ROOT/../CALVIN-EVALS"
for m in "${METHODS[@]}"; do
  if [[ "$m" == "ground_truth" ]]; then
    log "skip eval-data upload for ground_truth"
    continue
  fi
  if modal volume ls rm-policy-eval-data / 2>/dev/null | grep -q "$m"; then
    log "skip: eval-data/$m already uploaded"
    continue
  fi
  MDIR="$EVALS_ROOT/$m"
  if [[ ! -d "$MDIR" ]]; then
    log "FATAL: CALVIN-EVALS/$m missing at $MDIR"; exit 1
  fi
  run_modal "upload_eval_method($m)" run modal_app.py::upload_eval_method --local-path "$MDIR" --method-name "$m" || exit 1
done

log "bootstrap complete."

# --- Launch eval runs ---
for m in "${METHODS[@]}"; do
  LAUNCHED="$LOGDIR/${ACCOUNT}__run_${m}.LAUNCHED"
  if [[ -f "$LAUNCHED" ]]; then
    log "skip launch: $m already launched"
    continue
  fi
  RUNLOG="$LOGDIR/${ACCOUNT}__run_${m}.log"
  SHARDS=5
  EXTRA_ARGS=""
  if [[ "$m" == "ground_truth" ]]; then
    SHARDS=10
    EXTRA_ARGS="--use-hf-parquet-goal"
  fi
  echo "=== $(date -Iseconds) launching $m ===" >> "$RUNLOG"
  nohup env MODAL_PROFILE="$ACCOUNT" modal run --detach \
    modal_app.py::run_method --method-name "$m" --num-shards $SHARDS --ep-len 360 $EXTRA_ARGS \
    >> "$RUNLOG" 2>&1 &
  echo "$!" > "$LAUNCHED"
  log "launched $m PID=$! shards=$SHARDS"
  sleep 2
done

log "all methods launched."
