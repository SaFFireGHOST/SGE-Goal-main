#!/bin/bash
# Overfit test for counterfactual Flow-DPO (train_step2_cfdpo.py). cfdpo_step2 left train-pair DPO accuracy at chance,
# so: can the Step 2 LoRA fit the instruction preference on 16 train pairs at all? DPO terms only (no anchor rows), no
# M_i damage, higher LR, clipping loosened (cfdpo_step2 clipped norms of 4-47 to 1.0). Train metrics are scored on the
# same 16 pairs (orientation a, 3 fixed noise draws); mask IoU on 32 val rows as a drift check.
# Pass: train text_acc / dpo_acc well above 0.5 (~0.9) within ~30 epochs. Fail: the text signal does not get through.
# Overrides via env: NAME, INIT, LR, BETA, CLIP, STEPS, NPAIRS, VAL_EVERY, EXTRA, PY, CUDA_VISIBLE_DEVICES. Supervised variant (no DPO, no reference):
#   NAME=cfsft_overfit CLIP=1 EXTRA=--sft-cf
# Launch detached: setsid nohup bash run/final_mask/run_cfdpo_overfit.sh > run/logs/fm_cfdpo_overfit.log 2>&1 < /dev/null &
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-/home/sgeextn/.conda/envs/sgegen/bin/python}; [ -x "$PY" ] || PY=$(command -v python)
R=$REPO/run
NAME=${NAME:-cfdpo_overfit}
LR=${LR:-2e-4}
BETA=${BETA:-300}
CLIP=${CLIP:-50}
STEPS=${STEPS:-120}
NPAIRS=${NPAIRS:-16}
EXTRA=${EXTRA:-}   # e.g. EXTRA=--sft-cf for the supervised counterfactual overfit
OUT=$R/final_mask/training
INIT=${INIT:-$OUT/midrop_step2/last}   # or run/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin/step2_final_mask
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $REPO
mkdir -p $R/logs
while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -lt 30500 ]; do sleep 60; done
echo "[$(date '+%F %T')] start $NAME $EXTRA lr=$LR beta=$BETA clip=$CLIP steps=$STEPS pairs=$NPAIRS"
$PY -u sge_goal/final_mask/train_step2_cfdpo.py --name $NAME --init-lora $INIT --train-gt-mi \
  --n-train-pairs $NPAIRS --no-anchor --skip-held --pair-mi-damage 0 --cf-beta $BETA --lambda-cf 1 --lr $LR \
  --max-grad-norm $CLIP --warmup-steps 10 --pairs-batch 2 --micro-batch 2 --grad-accum 4 --max-steps $STEPS \
  --val-every ${VAL_EVERY:-20} --val-iou-samples 32 --val-loss-samples 16 --pair-noise 3 --log-every 5 --num-workers 4 $EXTRA \
  > $R/logs/fm_cfdpo_train_$NAME.log 2>&1
rc=$?
grep '"type": "val"' $OUT/$NAME/metrics.jsonl | cut -c1-400
[ $rc -eq 0 ] && echo OVERFIT_DONE || { echo "OVERFIT_FAILED exit=$rc"; tail -20 $R/logs/fm_cfdpo_train_$NAME.log; }
