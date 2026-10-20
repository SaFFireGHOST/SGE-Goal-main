#!/usr/bin/env bash
# ==========================================================================
# launch_training.sh — One-command launcher for the full training pipeline
#
# What it does:
#   1. Sets the Modal authentication token
#   2. Launches training on Modal in detached mode (-d)
#      - Automatically uploads local checkpoints to the Modal volume
#      - Auto-resumes from the latest checkpoint on the volume
#   3. Starts the sync daemon (downloads checkpoints/viz from volume to local)
#   4. Starts the watchdog daemon (monitors & restarts training if Modal
#      stops the app)
#
# All output is appended to flux2_klein_bridge_fer_mask_prediction.log
#
# Usage:
#   bash launch_training.sh
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/flux2_klein_bridge_fer_mask_prediction.log"
SYNC_PID_FILE="$SCRIPT_DIR/.sync_pid"
WATCHDOG_PID_FILE="$SCRIPT_DIR/.watchdog_pid"

SYNC_INTERVAL=180      # 3 minutes
WATCHDOG_INTERVAL=300  # 5 minutes

# --------------------------------------------------------------------------
# Activate the lgie conda environment
# --------------------------------------------------------------------------
eval "$(conda shell.bash hook)"
conda activate lgie
PYTHON="$(which python)"
MODAL="$(which modal)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using conda env: lgie" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Python: $PYTHON" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Modal:  $MODAL" | tee -a "$LOG_FILE"

# --------------------------------------------------------------------------
# Helper: kill old background processes if still running
# --------------------------------------------------------------------------
kill_old_process() {
    local pid_file="$1"
    local label="$2"
    if [[ -f "$pid_file" ]]; then
        local old_pid
        old_pid=$(cat "$pid_file")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stopping old $label (PID $old_pid)..."
            kill "$old_pid" 2>/dev/null || true
            sleep 2
            kill -9 "$old_pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi
}

# --------------------------------------------------------------------------
echo ""
echo "============================================================" | tee -a "$LOG_FILE"
echo " TRAINING LAUNCH — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# --------------------------------------------------------------------------
# Step 0: Kill any existing sync/watchdog daemons from prior runs
# --------------------------------------------------------------------------
kill_old_process "$SYNC_PID_FILE" "sync daemon"
kill_old_process "$WATCHDOG_PID_FILE" "watchdog daemon"

# --------------------------------------------------------------------------
# Step 1: Set Modal token via env vars (does NOT touch the global profile)
# --------------------------------------------------------------------------
# Modal credentials: export MODAL_TOKEN_ID / MODAL_TOKEN_SECRET beforehand, or rely on
# your default `modal token` profile.
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Modal token set via env vars (global profile untouched)" | tee -a "$LOG_FILE"

# --------------------------------------------------------------------------
# Step 2: Launch training on Modal in detached mode
#
# The local_entrypoint main() will:
#   - Upload any local checkpoints to the Modal volume
#   - Spawn the training function and return immediately
# The training function on Modal will:
#   - Auto-detect the latest checkpoint on the volume
#   - Resume training from there (or start fresh if no checkpoints)
#   - Commit the volume after each checkpoint / validation / test
# --------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching training on Modal (detached)..." | tee -a "$LOG_FILE"
cd "$SCRIPT_DIR"
"$MODAL" run -d modal_flux_2_klein_erm_train.py --action train 2>&1 | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Modal training app launched." | tee -a "$LOG_FILE"

# --------------------------------------------------------------------------
# Step 3: Start the background sync daemon
# --------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting sync daemon (interval: ${SYNC_INTERVAL}s)..." | tee -a "$LOG_FILE"
nohup "$PYTHON" "$SCRIPT_DIR/sync_volume.py" --interval "$SYNC_INTERVAL" \
    >> "$LOG_FILE" 2>&1 &
SYNC_PID=$!
echo "$SYNC_PID" > "$SYNC_PID_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sync daemon started (PID: $SYNC_PID)" | tee -a "$LOG_FILE"

# --------------------------------------------------------------------------
# Step 4: Start the watchdog daemon
# --------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting watchdog daemon (interval: ${WATCHDOG_INTERVAL}s)..." | tee -a "$LOG_FILE"
nohup "$PYTHON" "$SCRIPT_DIR/watchdog.py" --interval "$WATCHDOG_INTERVAL" \
    >> "$LOG_FILE" 2>&1 &
WATCHDOG_PID=$!
echo "$WATCHDOG_PID" > "$WATCHDOG_PID_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watchdog daemon started (PID: $WATCHDOG_PID)" | tee -a "$LOG_FILE"

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------
echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " ALL PROCESSES LAUNCHED SUCCESSFULLY" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "  Training:  running on Modal (detached)" | tee -a "$LOG_FILE"
echo "  Sync PID:  $SYNC_PID  (log: $LOG_FILE)" | tee -a "$LOG_FILE"
echo "  Watchdog:  $WATCHDOG_PID  (log: $LOG_FILE)" | tee -a "$LOG_FILE"
echo "  Log file:  $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "To monitor:" | tee -a "$LOG_FILE"
echo "  tail -f $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "To stop everything:" | tee -a "$LOG_FILE"
echo "  bash $SCRIPT_DIR/stop_training.sh" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
