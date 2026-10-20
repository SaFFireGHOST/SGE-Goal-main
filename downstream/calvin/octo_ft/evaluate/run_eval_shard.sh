#!/bin/bash
# Evaluate the fine-tuned Octo policy on one shard of the 1,087 CALVIN test rows,
# with the same protocol as GR-MG and GHIL-Glue (seed 0, 360-step horizon).
#
# usage: bash run_eval_shard.sh <method> <row_start> <row_end> <ckpt_dir> <out_dir>
#   method   : name of a goal-image folder under $GOAL_DIR (e.g. flux-kontext-dev-ft),
#              or "ground_truth" to use the dataset's own goal images (oracle)
#   ckpt_dir : Octo-ft checkpoint directory (the final step is loaded)
# The paper used shards [0,217) [217,434) [434,652) [652,869) [869,1087).
#
# env:
#   CALVIN_DATASET   CALVIN task_D_D folder containing validation/      (required)
#   CALVIN_REPO      clone of mees/calvin (for the task oracle config)   (required)
#   MAPPING_JSON     hf_calvin_mapping.json from gr_mg/evaluate/build_hf_calvin_mapping.py
#   TEST_PARQUET     CALVIN HF test parquet (only for method=ground_truth)
#   GOAL_DIR         folder holding one sub-folder of goal images per method
#   OCTO_TASK_MODE   goal_only (paper) | both | text_only
#   OCTO_DIR         Octo checkout (default ../octo)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
METHOD=$1; RS=$2; RE=$3; CKPT=$4; OUT=$5
: "${CALVIN_DATASET:?}" "${CALVIN_REPO:?}" "${MAPPING_JSON:?}"
export OCTO_TASK_MODE=${OCTO_TASK_MODE:-goal_only}
export OCTO_ACTION_SPACE=calvin          # Octo-ft emits CALVIN rel_actions directly
export PYOPENGL_PLATFORM=egl MESA_GL_VERSION_OVERRIDE=3.3
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.25
export PYTHONPATH="$HERE:${OCTO_DIR:-$HERE/../octo}:${PYTHONPATH:-}"
cd "$HERE"
[ -e calvin ] || ln -s "$CALVIN_REPO" calvin   # eval_method.py reads ./calvin/calvin_models/conf
ARGS=(--dataset_path "$CALVIN_DATASET" --ckpt_dir "$CKPT" --config_path unused
      --mapping_json "$MAPPING_JSON" --output_dir "$OUT/$METHOD"
      --row_start "$RS" --row_end "$RE" --ep_len 360 --method_name "$METHOD")
if [ "$METHOD" = ground_truth ]; then
  ARGS+=(--hf_parquet_goal "${TEST_PARQUET:?}")
else
  ARGS+=(--method_dir "${GOAL_DIR:?}/$METHOD")
fi
exec python -u eval_method.py "${ARGS[@]}"
