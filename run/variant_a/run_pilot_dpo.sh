#!/bin/bash
# A-sparse pilot update (plan.md §6.2): one round of success Flow-DPO from the released LoRAs, Step 2 then
# Step 3, on run/variant_a/pilot/pairs.json, anchor = original losses on HF tuples with ground-truth M_i.
# Smoke test (lr 5e-5, beta 2000) overshot (grad norm ~100, held-out pair loss 0.69 -> 2.3 in 2 steps),
# hence lr 1e-5, beta 1000. A checkpoint is kept at every validation (once per epoch over the pairs);
# selection is by rollouts on held-out contexts.
# Launch detached: setsid nohup bash run/variant_a/run_pilot_dpo.sh > run/logs/va_pilot_dpo.log 2>&1 < /dev/null &
set -u
R=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main/run
COMMON="--mode dpo --train-gt-mi --lr 1e-5 --beta 1000 --warmup-steps 5 --pairs-batch 2 --micro-batch 2 --grad-accum 4 --log-every 1 --val-loss-samples 64"
for spec in "2 6 9 --val-iou-samples 64" "3 5 12 --val-iou-samples 0"; do
  set -- $spec
  step=$1; epochs=$2; every=$3; shift 3
  echo "===== [$(date '+%F %T')] Step $step DPO ====="
  bash $R/variant_a/train_flux_lora.sh --step $step --name pilot_dpo_step$step $COMMON --epochs $epochs --val-every $every "$@" \
    > $R/logs/pilot_dpo_step$step.log 2>&1
  echo "step $step exit=$?"
  grep '"type": "val"' $R/variant_a/training/pilot_dpo_step$step/metrics.jsonl
done
echo PILOT_DPO_DONE
