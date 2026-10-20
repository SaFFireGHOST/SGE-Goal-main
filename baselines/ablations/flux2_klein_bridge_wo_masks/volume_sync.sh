#!/usr/bin/env bash
# ============================================================================
# volume_sync.sh - Periodically download models/checkpoints from Modal volume
# ============================================================================
# Usage:
#   nohup bash volume_sync.sh > volume_sync.log 2>&1 &
#
# This script polls the Modal volume every INTERVAL seconds and downloads
# any new checkpoints, validation visualizations, and test artifacts.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODAL_BIN="${MODAL_BIN:-modal}"
VOLUME_NAME="flux2-klein-bridge-bft-i1-volume"
REMOTE_DIR="models"
LOCAL_DIR="${SCRIPT_DIR}"
INTERVAL=120  # Poll every 2 minutes

echo "=========================================="
echo "Modal Volume Sync Started"
echo "Volume:    ${VOLUME_NAME}"
echo "Remote:    ${REMOTE_DIR}"
echo "Local:     ${LOCAL_DIR}"
echo "Interval:  ${INTERVAL}s"
echo "PID:       $$"
echo "=========================================="

download_count=0

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo ""
    echo "[${timestamp}] Checking for updates on Modal volume..."

    if "${MODAL_BIN}" volume get --force "${VOLUME_NAME}" "${REMOTE_DIR}" "${LOCAL_DIR}" 2>&1; then
        download_count=$((download_count + 1))
        echo "[${timestamp}] Download #${download_count} complete."
    else
        echo "[${timestamp}] Warning: Download failed or no changes."
    fi

    echo "[${timestamp}] Sleeping ${INTERVAL}s..."
    sleep "${INTERVAL}"
done
