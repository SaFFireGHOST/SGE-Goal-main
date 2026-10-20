#!/usr/bin/env bash
# =============================================================================
# Watchdog Script for flux2-klein-bridge-bft-i2 Modal Training
# =============================================================================
# Monitors the Modal training process and automatically relaunches it if it
# crashes or gets stopped by Modal. Training always resumes from the latest
# checkpoint on the volume (auto-resume is built into the training code).
#
# Usage:
#   nohup bash watchdog.sh > watchdog.log 2>&1 &
#
# To stop:
#   kill $(cat watchdog.pid)
# =============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODAL_BIN="${MODAL_BIN:-modal}"
TRAIN_SCRIPT="modal_flux_2_klein_train.py"
LOG_FILE="${PROJECT_DIR}/flux2_klein_bft_i2_ablation_no_final_mask_training.log"
WATCHDOG_PID_FILE="${PROJECT_DIR}/watchdog.pid"
TRAIN_PID_FILE="${PROJECT_DIR}/train.pid"

POLL_INTERVAL=120     # seconds between health checks
STARTUP_GRACE=300     # seconds to wait after launch before checking health
MAX_RESTARTS=50       # max consecutive restarts before giving up
RESTART_COOLDOWN=30   # seconds to wait before restarting after a crash

# ── Write our own PID so the watchdog can be stopped ─────────────────────────
echo $$ > "${WATCHDOG_PID_FILE}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WATCHDOG] $*"
}

cleanup() {
    log "Watchdog shutting down."
    rm -f "${WATCHDOG_PID_FILE}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Helpers ──────────────────────────────────────────────────────────────────

is_training_running() {
    # Check if the training process (modal run) is alive
    if [[ -f "${TRAIN_PID_FILE}" ]]; then
        local pid
        pid=$(cat "${TRAIN_PID_FILE}")
        if kill -0 "$pid" 2>/dev/null; then
            return 0  # running
        fi
    fi
    return 1  # not running
}

is_training_complete() {
    # Check if the log file indicates training finished successfully
    if [[ -f "${LOG_FILE}" ]]; then
        if grep -q "Training stream completed" "${LOG_FILE}" 2>/dev/null; then
            return 0  # completed
        fi
        if grep -q '>>> Training Result:' "${LOG_FILE}" 2>/dev/null && \
           grep -q 'Status: success' "${LOG_FILE}" 2>/dev/null; then
            return 0  # completed
        fi
    fi
    return 1  # not completed
}

launch_training() {
    log "Launching Modal training (will auto-resume from latest checkpoint)..."
    cd "${PROJECT_DIR}"

    # Truncate the log so we get a clean view for this run
    # (previous logs are already in the terminal scrollback / old log)
    : > "${LOG_FILE}"

    nohup "${MODAL_BIN}" run -d "${TRAIN_SCRIPT}" --action train --single-gpu \
        >> "${LOG_FILE}" 2>&1 &
    local pid=$!
    echo "$pid" > "${TRAIN_PID_FILE}"
    log "Training launched with PID ${pid}."
}

# ── Main Loop ────────────────────────────────────────────────────────────────

restart_count=0

log "Watchdog started (PID $$). Monitoring project: ${PROJECT_DIR}"
log "Poll interval: ${POLL_INTERVAL}s | Max restarts: ${MAX_RESTARTS}"

# Check if training is already complete before doing anything
if is_training_complete; then
    log "Training already completed successfully. Nothing to do."
    cleanup
fi

# Check if there's already a running training process
if is_training_running; then
    log "Training process already running (PID $(cat "${TRAIN_PID_FILE}"))."
    log "Will monitor existing process."
else
    # No running process — check if it crashed or was never started
    log "No running training process found. Launching..."
    launch_training
    log "Waiting ${STARTUP_GRACE}s for startup grace period..."
    sleep "${STARTUP_GRACE}"
fi

while true; do
    # Check if training completed successfully
    if is_training_complete; then
        log "Training completed successfully! Watchdog exiting."
        cleanup
    fi

    # Check if the process is still alive
    if is_training_running; then
        # Process is alive — all good
        restart_count=0  # reset consecutive restart counter
        sleep "${POLL_INTERVAL}"
        continue
    fi

    # Process is dead and training is not complete — need to restart
    restart_count=$((restart_count + 1))
    log "Training process is NOT running (restart #${restart_count}/${MAX_RESTARTS})."

    if [[ ${restart_count} -ge ${MAX_RESTARTS} ]]; then
        log "ERROR: Max restarts (${MAX_RESTARTS}) reached. Giving up."
        log "Check the log file for details: ${LOG_FILE}"
        cleanup
    fi

    # Check what caused the stop
    if [[ -f "${LOG_FILE}" ]]; then
        local_tail=$(tail -5 "${LOG_FILE}" 2>/dev/null || true)
        log "Last 5 lines of log:"
        echo "${local_tail}" | while IFS= read -r line; do
            log "  > ${line}"
        done
    fi

    log "Waiting ${RESTART_COOLDOWN}s cooldown before restarting..."
    sleep "${RESTART_COOLDOWN}"

    log "Restarting training (attempt ${restart_count})..."
    launch_training

    log "Waiting ${STARTUP_GRACE}s for startup grace period..."
    sleep "${STARTUP_GRACE}"
done
