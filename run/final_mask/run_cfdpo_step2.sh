#!/bin/bash
# Counterfactual Flow-DPO on Step 2 (sge_goal/final_mask/train_step2_cfdpo.py, plan_final_mask.md §5.1) on the
# same-block, same-place pairs of run/final_mask/cf_pairs.json (build_cf_pairs.py). Starts from midrop_step2/last,
# which is also the reference and the control. Stages:
#   0  pair metrics of the released LoRA (baseline; midrop's are the training run's step-0 validation)
#   1  training: lambda_cf 1, beta 300, lr 5e-5, 8 pair items + 8 anchor rows per update, EPOCHS over both
#      orientations of the train pairs; validation every VAL_EVERY updates (pair metrics, mask IoU on 256 val rows)
#   2  language test with real M_i (same 96 rows as language_test.json)
#   3  IoU on the 710 held-out validation rows vs released
# Checks: held-out text_acc / cf_acc rise (and on train pairs); language test close_text_vs_base below ~0.7 (released
# 0.975, midrop 0.971); held-out IoU not below 0.340. Every stage skips finished work, so rerunning resumes.
# Overrides via env: NAME, LR, BETA, EPOCHS, VAL_EVERY, PAIR_NOISE (noise draws per pair in validation).
# Launch detached: setsid nohup bash run/final_mask/run_cfdpo_step2.sh > run/logs/fm_cfdpo.log 2>&1 < /dev/null &
set -u
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
R=$REPO/run
PY=/home/sgeextn/.conda/envs/sgegen/bin/python
NAME=${NAME:-cfdpo_step2}
LR=${LR:-5e-5}
BETA=${BETA:-300}
EPOCHS=${EPOCHS:-1.5}
VAL_EVERY=${VAL_EVERY:-50}
PAIR_NOISE=${PAIR_NOISE:-3}
OUT=$R/final_mask/training
LORA=$OUT/$NAME/last
W=$R/final_mask/cfdpo_val/$NAME
NEED_MIB=30500
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $REPO
mkdir -p $W $R/logs

stage() { echo; echo "===== [$(date '+%F %T')] $1 ====="; }
wait_mem() { while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -lt $1 ]; do sleep 60; done; }
run() {
  local name=$1 rc=1; shift
  # The GPU is shared: another job can take the memory after the check, so a crash is retried.
  for try in $(seq 1 10); do
    wait_mem $NEED_MIB
    "$@" > $R/logs/$name.log 2>&1
    rc=$?
    echo "[$(date '+%F %T')] $name try $try exit=$rc"
    [ $rc -eq 0 ] && return 0
    sleep 300
  done
  echo "CFDPO_FAILED at $name"; tail -20 $R/logs/$name.log; exit $rc
}
TRAIN_ARGS="--train-gt-mi --cf-beta $BETA --lambda-cf 1 --lr $LR --warmup-steps 10 --pairs-batch 2 --micro-batch 2
  --grad-accum 4 --val-iou-samples 256 --val-loss-samples 64 --pair-noise $PAIR_NOISE --n-train-eval 115 --log-every 5 --num-workers 4"

stage "0/3 released baseline (pair metrics)"
grep -q eval_only $OUT/cfdpo_eval_released/metrics.jsonl 2>/dev/null || run fm_cfdpo_released \
  $PY -u sge_goal/final_mask/train_step2_cfdpo.py --name cfdpo_eval_released --eval-only $TRAIN_ARGS
tail -1 $OUT/cfdpo_eval_released/metrics.jsonl

stage "1/3 training"
RESUME=""
[ -f $OUT/$NAME/last_state.pt ] && RESUME="--resume"
for try in $(seq 1 10); do
  grep -q '"type": "done"' $OUT/$NAME/metrics.jsonl 2>/dev/null && break
  wait_mem $NEED_MIB
  echo "[$(date '+%F %T')] try $try $RESUME"
  $PY -u sge_goal/final_mask/train_step2_cfdpo.py --name $NAME --init-lora $OUT/midrop_step2/last $TRAIN_ARGS \
    --epochs $EPOCHS --val-every $VAL_EVERY $RESUME > $R/logs/fm_cfdpo_train_$NAME.log 2>&1
  RESUME="--resume"
  sleep 120
done
grep -q '"type": "done"' $OUT/$NAME/metrics.jsonl || { echo "CFDPO_FAILED at training"; tail -20 $R/logs/fm_cfdpo_train_$NAME.log; exit 1; }
grep '"type": "val"' $OUT/$NAME/metrics.jsonl | cut -c1-400

stage "2/3 language test, real M_i"
[ -f $W/language_real.json ] || run fm_cfdpo_lang_$NAME $PY $R/final_mask/diag_language.py --per-pair 16 \
  --checkpoint-dir $LORA --work $R/CALVIN-EVALS/fm_lang_$NAME --out $W/language_real.json

stage "3/3 held-out IoU vs released"
[ -f $W/heldout_iou.json ] || run fm_cfdpo_heldout_$NAME $PY $R/final_mask/score_step2_ckpts.py --out $W/heldout_iou.json \
  --ref released=$R/final_mask/ioudpo_val/heldout_iou.json --ckpt cfdpo=$LORA
echo CFDPO_DONE
