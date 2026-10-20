#!/usr/bin/env bash
# ==========================================================================
# stop_training.sh — Stops the sync daemon and watchdog
#
# NOTE: This does NOT stop the Modal training app itself (it runs on
# Modal's infrastructure independently). To stop the Modal app, use:
#   modal app stop <app-id>
#
# Usage:
#   bash stop_training.sh
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/flux2_klein_bridge_fer_mask_prediction.log"
SYNC_PID_FILE="$SCRIPT_DIR/.sync_pid"
WATCHDOG_PID_FILE="$SCRIPT_DIR/.watchdog_pid"

echo ""
echo "============================================================"
echo " STOPPING LOCAL DAEMONS — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

stop_process() {
    local pid_file="$1"
    local label="$2"
    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $label (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force-killing $label..."
                kill -9 "$pid" 2>/dev/null || true
            fi
            echo "  $label stopped."
        else
            echo "$label (PID $pid) is not running."
        fi
        rm -f "$pid_file"
    else
        echo "No PID file for $label found."
    fi
}

stop_process "$SYNC_PID_FILE" "Sync daemon"
stop_process "$WATCHDOG_PID_FILE" "Watchdog daemon"

echo ""
echo "Local daemons stopped."
echo "NOTE: The Modal training app may still be running on Modal."
echo "      To stop it: modal app list  then  modal app stop <app-id>"
echo "============================================================"
