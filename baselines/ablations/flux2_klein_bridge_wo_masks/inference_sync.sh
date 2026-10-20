#!/usr/bin/env bash
# ============================================================================
# inference_sync.sh — Real-time sync of inference results from Modal volume
# ============================================================================
# Downloads edited_image.jpg files from the Modal inference volume into the
# local test data directory as they are produced.
#
# Usage:
#   nohup bash inference_sync.sh > inference_sync.log 2>&1 &
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODAL_BIN="${SCRIPT_DIR}/.venv/bin/modal"
VOLUME_NAME="flux2-klein-bft-i1-ie-inference-volume"
LOCAL_TEST_DATA_DIR="${SCRIPT_DIR}/bridge_data_v2_tel_test_data"
INTERVAL=90  # Poll every 90 seconds

echo "=========================================="
echo "Inference Sync Started"
echo "Volume:    ${VOLUME_NAME}"
echo "Local dir: ${LOCAL_TEST_DATA_DIR}"
echo "Interval:  ${INTERVAL}s"
echo "PID:       $$"
echo "=========================================="

sync_count=0

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo ""
    echo "[${timestamp}] Syncing inference results from Modal volume..."

    # Download to a temp directory, then copy only edited_image.jpg files
    TMPDIR=$(mktemp -d)
    if "${MODAL_BIN}" volume get --force "${VOLUME_NAME}" "test_data" "${TMPDIR}" 2>&1; then
        sync_count=$((sync_count + 1))

        # Determine where files landed
        SRC_DIR="${TMPDIR}/test_data"
        if [[ ! -d "${SRC_DIR}" ]]; then
            SRC_DIR="${TMPDIR}"
        fi

        # Copy only edited_image.jpg files into the local test data folder
        copied=0
        for case_dir in "${SRC_DIR}"/*/; do
            case_name=$(basename "${case_dir}")
            src_file="${case_dir}edited_image.jpg"
            dst_file="${LOCAL_TEST_DATA_DIR}/${case_name}/edited_image.jpg"
            if [[ -f "${src_file}" && -d "${LOCAL_TEST_DATA_DIR}/${case_name}" ]]; then
                cp -f "${src_file}" "${dst_file}"
                copied=$((copied + 1))
            fi
        done

        # Count total edited images locally
        total_local=$(find "${LOCAL_TEST_DATA_DIR}" -name "edited_image.jpg" 2>/dev/null | wc -l)
        echo "[${timestamp}] Sync #${sync_count}: copied ${copied} edited images. Total local: ${total_local}/2000"
    else
        echo "[${timestamp}] Warning: Download failed or volume not yet available."
    fi

    rm -rf "${TMPDIR}"
    echo "[${timestamp}] Sleeping ${INTERVAL}s..."
    sleep "${INTERVAL}"
done
