#!/usr/bin/env bash
# ============================================================================
# watchdog.sh — Monitor the single bft-i1 Modal training app and restart
#               it ONLY if it has stopped. Never creates duplicates.
# ============================================================================
#
# Usage:
#   nohup bash watchdog.sh > watchdog.log 2>&1 &
#
# How it works:
#   - Every CHECK_INTERVAL seconds, queries `modal app list --json`
#   - Counts how many apps with description "flux2-klein-bridge-bft-i1" are
#     in a non-stopped state (ephemeral, detached, running, deployed, etc.)
#   - If count == 0  →  relaunch training (auto-resumes from latest checkpoint)
#   - If count >= 1  →  do nothing (app is alive)
#   - Also keeps volume_sync.sh alive for local checkpoint downloads
#
# ============================================================================

set -uo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODAL_BIN="${SCRIPT_DIR}/.venv/bin/modal"
TRAIN_SCRIPT="${SCRIPT_DIR}/modal_flux_2_klein_train.py"
TRAIN_LOG="${SCRIPT_DIR}/flux2_klein_bft_i1_train.log"
VOLUME_SYNC_SCRIPT="${SCRIPT_DIR}/volume_sync.sh"
VOLUME_SYNC_LOG="${SCRIPT_DIR}/volume_sync.log"

APP_DESCRIPTION="flux2-klein-bridge-bft-i1"   # exact match

CHECK_INTERVAL=120        # seconds between health checks
RELAUNCH_COOLDOWN=180     # min seconds between relaunches
MAX_CONSECUTIVE_FAILS=3   # consecutive failures before backoff
BACKOFF_INTERVAL=600      # seconds to wait after MAX_CONSECUTIVE_FAILS

# ── State ──────────────────────────────────────────────────────────────────
last_relaunch_ts=0
consecutive_fails=0
total_relaunches=0

# ── Helpers ────────────────────────────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

ensure_volume_sync() {
    if ! pgrep -f "bash.*volume_sync.sh" > /dev/null 2>&1; then
        log "volume_sync.sh not running. Restarting..."
        nohup bash "${VOLUME_SYNC_SCRIPT}" >> "${VOLUME_SYNC_LOG}" 2>&1 &
        log "volume_sync.sh restarted (PID $!)."
    fi
}

count_running_apps() {
    # Returns the number of non-stopped apps matching our APP_DESCRIPTION.
    # Uses `modal app list --json` and filters by exact description match.
    local json_output
    json_output=$("${MODAL_BIN}" app list --json 2>/dev/null) || {
        log "WARNING: 'modal app list --json' failed."
        echo "-1"
        return
    }

    # Count apps where Description == APP_DESCRIPTION and State != "stopped"
    local count
    count=$(echo "${json_output}" | python3 -c "
import sys, json
apps = json.load(sys.stdin)
count = sum(1 for a in apps
            if a.get('Description') == '${APP_DESCRIPTION}'
            and 'stopped' not in a.get('State', 'stopped').lower())
print(count)
" 2>/dev/null) || {
        log "WARNING: JSON parsing failed."
        echo "-1"
        return
    }

    echo "${count}"
}

launch_training() {
    local now
    now=$(date +%s)

    # Cooldown check
    local elapsed=$(( now - last_relaunch_ts ))
    if (( elapsed < RELAUNCH_COOLDOWN )); then
        log "Cooldown active (${elapsed}s < ${RELAUNCH_COOLDOWN}s). Skipping."
        return 1
    fi

    # Backoff check
    if (( consecutive_fails >= MAX_CONSECUTIVE_FAILS )); then
        log "BACKOFF: ${consecutive_fails} consecutive failures. Waiting ${BACKOFF_INTERVAL}s."
        sleep "${BACKOFF_INTERVAL}"
        consecutive_fails=0
    fi

    log "=== RELAUNCHING training (relaunch #$((total_relaunches + 1))) ==="
    log "Training will auto-resume from latest checkpoint on volume."

    # Launch in detached mode (auto-resumes via --resume_from_checkpoint logic)
    nohup "${MODAL_BIN}" run --detach "${TRAIN_SCRIPT}" --action train \
        >> "${TRAIN_LOG}" 2>&1 &
    local launch_pid=$!
    log "modal run --detach launched (local PID ${launch_pid})."

    last_relaunch_ts=$(date +%s)
    total_relaunches=$((total_relaunches + 1))

    # Wait for the local modal process to finish dispatching
    sleep 45

    # Verify the app actually appeared
    local running
    running=$(count_running_apps)
    if [[ "${running}" == "0" ]]; then
        log "ERROR: App did not appear after launch."
        consecutive_fails=$((consecutive_fails + 1))
        return 1
    fi

    log "App confirmed running after relaunch."
    consecutive_fails=0
    return 0
}

# ── Main Loop ──────────────────────────────────────────────────────────────
log "=========================================="
log "Watchdog started"
log "Monitoring: ${APP_DESCRIPTION}"
log "Interval:   ${CHECK_INTERVAL}s"
log "Cooldown:   ${RELAUNCH_COOLDOWN}s"
log "PID:        $$"
log "=========================================="

while true; do
    ensure_volume_sync

    running=$(count_running_apps)

    if [[ "${running}" == "-1" ]]; then
        log "Could not determine app state. Will retry next cycle."
    elif (( running >= 1 )); then
        log "OK — ${running} running instance(s) of '${APP_DESCRIPTION}'. (Total relaunches: ${total_relaunches})"
    else
        log "ALERT — 0 running instances of '${APP_DESCRIPTION}'. Triggering relaunch..."
        launch_training
    fi

    sleep "${CHECK_INTERVAL}"
done
