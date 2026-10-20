#!/bin/bash
# Fine-tune Octo-Base on the CALVIN RLDS dataset (the paper's Octo-ft run).
#
# usage: bash finetune.sh [mode] [task] [steps] [extra --config.* overrides]
#   mode  : full (paper) | head_only | head_mlp_only
#   task  : multimodal (paper; goal image OR instruction per sample, p=0.5)
#           | image_conditioned | language_conditioned
#   steps : 30000 (paper)
#
# env:
#   OCTO_DIR         Octo checkout from setup_octo.sh      (default ./octo)
#   CALVIN_RLDS_DIR  RLDS dataset from calvin_rlds/build_rlds.py (default ./rlds)
#   SAVE_DIR         checkpoint directory                 (default ./checkpoints)
#   CUDA_VISIBLE_DEVICES  GPUs; the paper used one 48 GB GPU with batch 32
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE=${1:-full}; TASK=${2:-multimodal}; STEPS=${3:-30000}
OCTO_DIR=${OCTO_DIR:-$HERE/octo}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
# Give the goal image its own augmentation seed (requires the patched Octo).
export OCTO_DESYNC_TASK_AUG=${OCTO_DESYNC_TASK_AUG:-1}
export PYTHONPATH="$HERE:$OCTO_DIR:${PYTHONPATH:-}"
cd "$HERE"
exec python -u "$OCTO_DIR/scripts/finetune.py" \
  --config calvin_rlds/calvin_finetune_config.py:"$MODE","$TASK" \
  --config.pretrained_path=hf://rail-berkeley/octo-base \
  --config.dataset_kwargs.data_dir="${CALVIN_RLDS_DIR:-$HERE/rlds}" \
  --config.save_dir="${SAVE_DIR:-$HERE/checkpoints}" \
  --config.num_steps="$STEPS" \
  "${@:4}"
