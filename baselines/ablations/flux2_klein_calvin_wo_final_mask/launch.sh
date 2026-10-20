#!/usr/bin/env bash
#
# Launch the i2 ablation (flux_kontext_calvin_bft_i2) on Modal account 1.
#
# What this does:
#   1. Sets this account's Modal credentials as ENV VARS only (MODAL_TOKEN_ID /
#      MODAL_TOKEN_SECRET). It NEVER runs `modal token set` and NEVER reads/writes
#      the shared global profile (~/.modal.toml). MODAL_PROFILE is unset so nothing
#      can override the token env vars.
#   2. Starts training DETACHED (`modal run --detach`) under nohup, so it keeps
#      running after you log off this server, and Modal auto-restarts + resumes it
#      from the last checkpoint if preempted.
#   3. Starts the real-time sync poller (sync_from_modal.py) under nohup, so new
#      checkpoints + logs land in ./models and ./logs here in near real time.
#
# Safe to re-run: it refuses to start a second training process if one is alive.
# Stop everything with ./stop.sh
#
set -u
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

# ---- Modal account 1 credentials (env-var auth only; global profile untouched) ----
# Modal credentials: export MODAL_TOKEN_ID / MODAL_TOKEN_SECRET beforehand, or rely on
# your default `modal token` profile.
unset MODAL_PROFILE 2>/dev/null || true

TRAIN_PID_FILE="$SCRIPT_DIR/train.pid"
SYNC_PID_FILE="$SCRIPT_DIR/sync.pid"
TRAIN_LOG="$SCRIPT_DIR/flux_kontext_calvin_bft_i2_modalrun.log"
SYNC_LOG="$SCRIPT_DIR/flux_kontext_calvin_bft_i2_sync.log"

is_alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

if is_alive "$TRAIN_PID_FILE"; then
  echo "Training already running (PID $(cat "$TRAIN_PID_FILE")). Use ./stop.sh first." >&2
  exit 1
fi

echo "=== flux_kontext_calvin_bft_i2 : launching on Modal account 1 ==="

# 1) Detached training (auto-resume + auto-retry on preemption are built into the app)
nohup python -m modal run --detach modal_flux_2_klein_train.py --action train \
  > "$TRAIN_LOG" 2>&1 < /dev/null &
echo $! > "$TRAIN_PID_FILE"
echo "  training  PID $(cat "$TRAIN_PID_FILE")  -> $TRAIN_LOG"

# 2) Real-time checkpoint/log sync poller
if is_alive "$SYNC_PID_FILE"; then
  echo "  sync poller already running (PID $(cat "$SYNC_PID_FILE"))."
else
  nohup python sync_from_modal.py > "$SYNC_LOG" 2>&1 < /dev/null &
  echo $! > "$SYNC_PID_FILE"
  echo "  sync      PID $(cat "$SYNC_PID_FILE")  -> $SYNC_LOG"
fi

echo
echo "You can safely log off now. Follow progress with:"
echo "  tail -f \"$TRAIN_LOG\"        # remote training console (client side)"
echo "  tail -f \"$SYNC_LOG\"         # poller activity"
echo "  tail -f \"$SCRIPT_DIR/flux_kontext_calvin_bft_i2.log\"   # synced training log"
echo "  ls models/flux-2-klein-lora-calvin-i2/                  # synced checkpoints"
