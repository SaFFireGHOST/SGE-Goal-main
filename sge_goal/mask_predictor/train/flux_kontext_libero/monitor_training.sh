#!/bin/bash
###############################################################################
# monitor_training.sh — Watchdog for flux_kontext_lora_finetuning.py
#
# • Polls every CHECK_INTERVAL seconds to see if the training process is alive.
# • If the process has died without producing the TRAINING_COMPLETE marker,
#   it automatically restarts from the latest checkpoint.
# • Once training is fully done (marker file exists), the monitor exits.
#
# Usage:
#   nohup bash monitor_training.sh &> monitor_training.log &
#   # PID is written to monitor_training.pid for easy cleanup
###############################################################################

set -euo pipefail

# ─── Ensure HuggingFace token is available ───────────────────────────────────
if [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]] && [[ -z "${HF_TOKEN:-}" ]]; then
    # Read token from huggingface-cli cache
    HF_TOKEN_FILE="${HOME}/.cache/huggingface/token"
    if [[ -f "${HF_TOKEN_FILE}" ]]; then
        export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
    fi
fi

# ─── Configuration ───────────────────────────────────────────────────────────
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="lgie"
OUTPUT_DIR="${WORKDIR}/kontext-i2i-libero"
LOG_FILE="${WORKDIR}/flux_kontext_libero_finetuning.log"
MARKER_FILE="${OUTPUT_DIR}/TRAINING_COMPLETE"
CHECK_INTERVAL=120          # seconds between checks
MIN_RESTART_INTERVAL=300    # minimum seconds between restarts (cool-down)
PROCESS_GREP_PATTERN="flux_kontext_lora_finetuning.py"

# The base training command (--resume_from_checkpoint added dynamically)
BASE_TRAIN_CMD="accelerate launch flux_kontext_lora_finetuning.py \
  --pretrained_model_name_or_path=black-forest-labs/FLUX.1-Kontext-dev \
  --dataset_name=RiddhiCh/LGRM_Libero_10_90_Combined \
  --output_dir=kontext-i2i-libero \
  --resolution=128 \
  --train_batch_size=43 \
  --val_batch_size=8 \
  --test_batch_size=8 \
  --dataloader_num_workers=0 \
  --num_train_epochs=75 \
  --checkpointing_epochs=1 \
  --checkpoints_total_limit=75 \
  --validation_epoch_list=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75 \
  --val_metrics_samples=25 \
  --val_viz_num=10 \
  --test_viz_num=10 \
  --cache_text_embeddings_to_disk \
  --cache_latents_to_disk \
  --disk_cache_dir=disk_cache \
  --gradient_checkpointing \
  --mixed_precision=bf16 \
  --learning_rate=1e-4 \
  --rank=16 \
  --lora_alpha=16 \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --seed=42 \
  --report_to=tensorboard \
  --allow_tf32"

# ─── Helpers ─────────────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] MONITOR: $*"
}

is_training_running() {
    # Returns 0 (true) if the training process is alive
    pgrep -f "${PROCESS_GREP_PATTERN}" > /dev/null 2>&1
}

is_training_complete() {
    # Returns 0 (true) if the completion marker exists
    [[ -f "${MARKER_FILE}" ]]
}

get_latest_checkpoint() {
    # Print the name of the latest checkpoint dir, or empty string
    if [[ -d "${OUTPUT_DIR}" ]]; then
        ls -1d "${OUTPUT_DIR}"/checkpoint-epoch-* 2>/dev/null \
            | sort -t'-' -k3 -n \
            | tail -1 \
            | xargs -r basename
    fi
}

launch_training() {
    local ckpt
    ckpt=$(get_latest_checkpoint)

    # Build command: add --resume_from_checkpoint only if checkpoints exist
    local TRAIN_CMD="${BASE_TRAIN_CMD}"
    if [[ -n "${ckpt}" ]]; then
        TRAIN_CMD="${TRAIN_CMD} --resume_from_checkpoint=latest"
        log "Resuming from latest checkpoint: ${ckpt}"
    else
        log "No checkpoint found — starting fresh training"
    fi

    cd "${WORKDIR}"

    # Activate conda env and launch. Append to log file (not overwrite).
    # Run in background so the monitor can keep polling.
    (
        eval "$(conda shell.bash hook 2>/dev/null)"
        conda activate "${CONDA_ENV}"
        ${TRAIN_CMD} 2>&1 | tee -a "${LOG_FILE}"
    ) &

    TRAINING_PID=$!
    log "Training process launched (PID=${TRAINING_PID})"

    # Give the process time to initialize before we start checking again
    sleep 60
}

# ─── Main Loop ───────────────────────────────────────────────────────────────

# Write our own PID for external management
echo $$ > "${WORKDIR}/monitor_training.pid"
log "Monitor started (PID=$$). PID written to monitor_training.pid"

LAST_RESTART_TIME=0

while true; do
    # 1. Check if training is already complete
    if is_training_complete; then
        log "TRAINING_COMPLETE marker found — full run finished. Monitor exiting."
        rm -f "${WORKDIR}/monitor_training.pid"
        exit 0
    fi

    # 2. Check if training process is still alive
    if is_training_running; then
        # Everything is fine — sleep and check again
        sleep "${CHECK_INTERVAL}"
        continue
    fi

    # 3. Process is dead and training is not complete — need to restart
    log "Training process NOT running and TRAINING_COMPLETE not found."

    # Cool-down: avoid rapid restart loops (e.g., if there's a config error)
    NOW=$(date +%s)
    ELAPSED=$(( NOW - LAST_RESTART_TIME ))
    if (( ELAPSED < MIN_RESTART_INTERVAL )); then
        WAIT_TIME=$(( MIN_RESTART_INTERVAL - ELAPSED ))
        log "Cool-down: last restart was ${ELAPSED}s ago. Waiting ${WAIT_TIME}s before retry."
        sleep "${WAIT_TIME}"
    fi

    # Double-check completion again after cool-down
    if is_training_complete; then
        log "TRAINING_COMPLETE marker found after cool-down — Monitor exiting."
        rm -f "${WORKDIR}/monitor_training.pid"
        exit 0
    fi

    # Restart training
    LAST_RESTART_TIME=$(date +%s)
    launch_training

    # Continue the monitoring loop
    sleep "${CHECK_INTERVAL}"
done
