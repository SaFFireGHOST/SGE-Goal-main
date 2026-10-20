#!/usr/bin/env bash
# ===========================================================================
# Launch the bft_i1 ablation finetuning on Modal.
#   Ablation: NEITHER the initial NOR the final edit-region mask is fed to the
#             model (only original image + edit instruction are fed). SSC loss
#             removed.
#   Modal account: #2
#
# This uses PER-PROCESS env-var auth (MODAL_TOKEN_ID / MODAL_TOKEN_SECRET) so
# the shared machine's GLOBAL modal profile (~/.modal.toml) is NOT touched or
# relied upon.
#
# Runs DETACHED (modal run --detach) + under nohup so that:
#   * the remote training keeps running on Modal even if this local driver dies;
#   * the local driver (which polls the volume and downloads checkpoints/viz in
#     real time) survives you logging off this server;
#   * the full training log is streamed in real time into the local .log file.
#
# Re-running this script after a preemption/stop is safe: the Modal training
# function auto-detects the latest checkpoint on the volume and resumes from it.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Modal credentials: export MODAL_TOKEN_ID / MODAL_TOKEN_SECRET beforehand, or rely on
# your default `modal token` profile.

LOG="flux_kontext_calvin_bft_i1.log"

echo "Launching bft_i1 ablation (account #2), detached. Log -> $LOG"
# setsid + nohup + stdin from /dev/null => fully detached from the launching
# shell/session, survives logoff. --detach keeps the remote alive regardless.
setsid nohup modal run --detach modal_flux_kontext_train.py --action train >> "$LOG" 2>&1 < /dev/null &
echo "Started. Local driver PID: $!"
echo "Follow with:  tail -f $(pwd)/$LOG"
