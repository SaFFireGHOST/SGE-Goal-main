#!/bin/bash
# Flow-DPO on Step 2 with IoU-ranked pairs (the A-sparse pilot's trainer and settings, clean labels instead of
# rollout successes). Waits for run_sample_step2.sh, builds the pairs (build_iou_pairs.py), then 1 epoch from the
# released Step 2 LoRA: reference = frozen released LoRA, anchor = original Step 2 losses on HF tuples with
# ground-truth M_i. Validation every 53 updates (~4x per epoch): final-mask IoU on 256 HF validation tuples with
# learned M_i (checkpoint selection, best/), held-out pair loss / accuracy; a checkpoint is kept at each.
# The GPU is shared: train_flux_lora_lowmem.py keeps the startup peak at ~29 GB (the trainer alone needs ~39 GB),
# and a crash is retried with --resume once enough memory is free.
# Overrides via env: NAME, LR, EPOCHS, VAL_EVERY (run 2: NAME=iou_dpo_step2_lr5e5 LR=5e-5 EPOCHS=2 VAL_EVERY=74).
# Launch detached: setsid nohup bash run/final_mask/run_dpo_step2.sh > run/logs/fm_dpo_step2.log 2>&1 < /dev/null &
set -u
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
R=$REPO/run
PY=/home/sgeextn/.conda/envs/sgegen/bin/python
NAME=${NAME:-iou_dpo_step2}
LR=${LR:-1e-5}
EPOCHS=${EPOCHS:-1}
VAL_EVERY=${VAL_EVERY:-53}
OUT=$R/final_mask/training
NEED_MIB=29500

until grep -qE "SAMPLE_STEP2_(DONE|FAILED)" $R/logs/fm_sample_step2.log; do sleep 60; done
grep -q SAMPLE_STEP2_DONE $R/logs/fm_sample_step2.log || { echo "DPO_STEP2_FAILED: sampling failed"; exit 1; }

echo "===== [$(date '+%F %T')] pairs ====="
[ -f $R/final_mask/iou_pairs.json ] || $PY $R/final_mask/build_iou_pairs.py || { echo "DPO_STEP2_FAILED: pairs"; exit 1; }

RESUME=""
for try in $(seq 1 10); do
  while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -lt $NEED_MIB ]; do sleep 60; done
  echo "===== [$(date '+%F %T')] DPO try $try $RESUME ====="
  (cd $REPO && TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore::FutureWarning \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u sge_goal/final_mask/train_flux_lora_lowmem.py --step 2 --mode dpo --name $NAME --out-dir $OUT \
    --pairs $R/final_mask/iou_pairs.json --train-gt-mi --lr $LR --beta 1000 --warmup-steps 5 \
    --pairs-batch 2 --micro-batch 2 --grad-accum 4 --epochs $EPOCHS --val-every $VAL_EVERY \
    --val-iou-samples 256 --val-loss-samples 64 --log-every 5 $RESUME)
  grep -q '"type": "done"' $OUT/$NAME/metrics.jsonl 2>/dev/null && break
  RESUME="--resume"
  sleep 120
done
grep -q '"type": "done"' $OUT/$NAME/metrics.jsonl || { echo "DPO_STEP2_FAILED: training"; exit 1; }
grep '"type": "val"' $OUT/$NAME/metrics.jsonl
echo DPO_STEP2_DONE
