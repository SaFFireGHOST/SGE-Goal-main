#!/bin/bash
# =============================================================================
# launch_training.sh — Master launcher for Flux Kontext training on Modal
#
# Orchestrates:
#   1. Modal token setup
#   2. Upload local checkpoints + spawn training (modal run -d --launch-only)
#   3. Background volume sync daemon (sync_modal_volume.py)
#   4. Background watchdog daemon (watchdog_training.py)
#
# All output is appended to the log file. Use  tail -f <log_file>  to monitor.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/flux_kontext_dev_bridge_image_editing.log"
PID_FILE="$SCRIPT_DIR/.training_pids"

# Conda environment name
CONDA_ENV="lgie"

# ── helpers ──────────────────────────────────────────────────────────────────
ts() { date "+%Y-%m-%d %H:%M:%S"; }

log() {
    echo "[LAUNCHER $(ts)] $*" | tee -a "$LOG_FILE"
}

cleanup_old_processes() {
    log "Cleaning up any previously running sync/watchdog processes..."
    # Kill by PID file first
    if [[ -f "$PID_FILE" ]]; then
        while IFS= read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                log "  Killing old process PID=$pid"
                kill "$pid" 2>/dev/null || true
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    # Also kill by process name to catch orphans
    pkill -f "sync_modal_volume.py" 2>/dev/null || true
    pkill -f "watchdog_training.py" 2>/dev/null || true
    # Kill any leftover modal app logs processes
    pkill -f "modal app logs $APP_NAME" 2>/dev/null || true
    sleep 2
}

APP_NAME="flux-kontext-dev-bridge-image-editing"

# ── main ─────────────────────────────────────────────────────────────────────
log "============================================================"
log "Flux Kontext Training Launch"
log "============================================================"

# Step 0: Activate conda environment
log "Activating conda environment: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
log "  Python: $(which python)"
log "  Modal:  $(which modal)"

# Step 1: Set Modal token
log "Setting Modal token..."
if [[ -n "${MODAL_TOKEN_ID:-}" && -n "${MODAL_TOKEN_SECRET:-}" ]]; then
    modal token set \
        --token-id "$MODAL_TOKEN_ID" \
        --token-secret "$MODAL_TOKEN_SECRET" 2>&1 | tee -a "$LOG_FILE"
else
    log "  MODAL_TOKEN_ID/SECRET not set; using the existing Modal profile"
fi

# Step 2: Clean up old processes
cleanup_old_processes

# Step 3: Kill any stale Modal apps with the same name
log "Stopping any stale Modal apps matching '$APP_NAME'..."
APP_LIST_OUTPUT=$(modal app list 2>/dev/null || true)
echo "$APP_LIST_OUTPUT" | grep -oP 'ap-[A-Za-z0-9]+' | while read -r app_id; do
    # Check if this app_id line also contains our APP_NAME
    if echo "$APP_LIST_OUTPUT" | grep "$app_id" | grep -q "$APP_NAME"; then
        log "  Stopping stale app: $app_id"
        modal app stop "$app_id" 2>&1 | tee -a "$LOG_FILE" || true
    fi
done || true
sleep 5

# Step 4: Launch training (upload local checkpoints + spawn remote training + exit)
log "Launching training on Modal (detached + launch-only)..."
cd "$SCRIPT_DIR"
modal run -d modal_flux_kontext_train.py \
    --action train \
    --single-gpu \
    --launch-only 2>&1 | tee -a "$LOG_FILE"

log "Training spawned on Modal."

# Step 5: Start volume sync daemon in background
log "Starting volume sync daemon (interval=120s)..."
nohup python "$SCRIPT_DIR/sync_modal_volume.py" --interval 120 \
    >> "$LOG_FILE" 2>&1 &
SYNC_PID=$!
log "  Sync daemon PID: $SYNC_PID"

# Step 6: Start watchdog daemon in background
# The watchdog also manages log streaming via `modal app logs`
log "Starting watchdog daemon (check-interval=120s, grace-period=600s)..."
nohup python "$SCRIPT_DIR/watchdog_training.py" \
    --check-interval 120 \
    --grace-period 600 \
    >> "$LOG_FILE" 2>&1 &
WATCHDOG_PID=$!
log "  Watchdog PID: $WATCHDOG_PID"

# Save PIDs for cleanup
echo "$SYNC_PID" > "$PID_FILE"
echo "$WATCHDOG_PID" >> "$PID_FILE"

log "============================================================"
log "All processes launched successfully."
log "  Sync PID:     $SYNC_PID"
log "  Watchdog PID: $WATCHDOG_PID"
log "  Log file:     $LOG_FILE"
log ""
log "Monitor with:  tail -f $LOG_FILE"
log "Stop all with: kill $SYNC_PID $WATCHDOG_PID"
log "============================================================"
