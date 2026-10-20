#!/usr/bin/env bash
# Build the per-generator folder layout that orchestrate_goals.py expects, as symlinks into this
# repository and into the checkpoints downloaded from Hugging Face.
#
#   huggingface-cli download RiddhiCh/SGE-Goal --local-dir sge-goal-checkpoints
#   bash make_inferencers_layout.sh /abs/path/sge-goal-checkpoints [OUT_DIR]
#   export SGE_INFERENCERS_DIR=OUT_DIR        # default: downstream/simpler_bridge/inferencers
#
# Checkpoints for the ablation baselines are linked inside their code directories
# (baselines/ablations/*/models/), which git ignores.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /abs/path/sge-goal-checkpoints [OUT_DIR]" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CKPT="$(cd "$1" && pwd)"
OUT="${2:-$HERE/inferencers}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

link() {  # link TARGET LINK_PATH
  [[ -e "$1" ]] || { echo "missing: $1" >&2; exit 1; }
  mkdir -p "$(dirname "$2")"
  ln -sfn "$1" "$2"
}

# SGE-Goal, FLUX.2 klein (BridgeDataV2 adapters)
K="$OUT/flux2-klein-ft"
link "$REPO/sge_goal/grounding/flux2_klein"                  "$K/ier-mask-prediction-code"
link "$REPO/sge_goal/mask_predictor/inference/flux2_klein"   "$K/fer-mask-prediction-code"
link "$REPO/sge_goal/synthesis/inference/flux2_klein"        "$K/image-editing-code"
link "$CKPT/flux2-klein/bridge/step2_final_mask"  "$K/bridge-fer-mask-prediction-checkpoint/klein-fer-checkpoint-epoch-17"
link "$CKPT/flux2-klein/bridge/step3_goal_image"  "$K/bridge-image-editing-checkpoint/klein-bridge-image-editing-checkpoint-epoch-9"

# SGE-Goal, FLUX.1 Kontext (BridgeDataV2 adapters)
F="$OUT/flux-kontext-dev-ft/bridge_and_calvin"
link "$REPO/sge_goal/grounding/flux_kontext_bridge_calvin"            "$F/ier-mask-prediction-code"
link "$REPO/sge_goal/mask_predictor/inference/flux_kontext_bridge_calvin" "$F/fer-mask-prediction-code"
link "$REPO/sge_goal/synthesis/inference/flux_kontext_bridge_calvin"  "$F/image-editing-code"
link "$CKPT/flux1-kontext-dev/bridge/step2_final_mask"  "$F/bridge-fer-mask-prediction-checkpoint/kontext-bridge-fer-checkpoint-epoch-38"
link "$CKPT/flux1-kontext-dev/bridge/step3_goal_image"  "$F/bridge-image-editing-checkpoint/kontext-ie-checkpoint-epoch-18"

# Baselines
link "$REPO/baselines/flux_kontext_base"   "$OUT/flux-kontext-dev"
link "$REPO/baselines/flux2_klein_base"    "$OUT/flux2-klein"
link "$REPO/baselines/qwen_image_edit"     "$OUT/qwen-image-edit"
link "$REPO/baselines/susie"               "$OUT/SuSIE"
link "$REPO/baselines/clover"              "$OUT/Clover"
link "$REPO/baselines/gemini_3_pro_image"  "$OUT/gemini-3-pro-image"
link "$REPO/baselines/gpt_image_1_5"       "$OUT/gpt-image-1.5"

# Mask ablations (BridgeDataV2)
A="$REPO/baselines/ablations"
link "$A/flux_kontext_bridge_wo_masks"       "$OUT/flux-kontext-dev-bft-i1"
link "$A/flux_kontext_bridge_wo_final_mask"  "$OUT/flux-kontext-dev-bft-i2"
link "$A/flux2_klein_bridge_wo_masks"        "$OUT/flux2-klein-bft-i1"
link "$A/flux2_klein_bridge_wo_final_mask"   "$OUT/flux2-klein-bft-i2"
link "$CKPT/ablations/flux1-kontext-dev/bridge/wo_masks"       "$A/flux_kontext_bridge_wo_masks/models/kontext-i2i/checkpoint-epoch-13"
link "$CKPT/ablations/flux1-kontext-dev/bridge/wo_final_mask"  "$A/flux_kontext_bridge_wo_final_mask/models/kontext-i2i/checkpoint-epoch-9"
link "$CKPT/ablations/flux2-klein/bridge/wo_masks"             "$A/flux2_klein_bridge_wo_masks/models/flux-2-klein-lora-bridge-v2-ablation-no-masks/checkpoint-epoch-10"
link "$CKPT/ablations/flux2-klein/bridge/wo_final_mask"        "$A/flux2_klein_bridge_wo_final_mask/models/flux-2-klein-lora-bridge-v2/checkpoint-epoch-9"

echo "Layout ready in $OUT"
echo "export SGE_INFERENCERS_DIR=$OUT"
