#!/bin/bash
# Step 2 with M_i dropout (sge_goal/final_mask/train_step2_midrop.py): 1 epoch from the released LoRA, settings of
# run/variant_a/training/diag_gtmi_step2 (ground-truth M_i, lr 5e-5, batch 32), 25% black + 25% damaged M_i. Then:
#   language test with real M_i and with black M_i (same 96 rows as language_test.json), IoU on the 710 held-out rows.
# Pass: close_text_vs_base falls from 0.975 to below ~0.7 with real M_i, held-out IoU not below released. Rollouts only
# after that. Every stage skips finished work, so rerunning resumes.
# Launch detached: setsid nohup bash run/final_mask/run_midrop_step2.sh > run/logs/fm_midrop.log 2>&1 < /dev/null &
set -u
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
R=$REPO/run
PY=/home/sgeextn/.conda/envs/sgegen/bin/python
NAME=${NAME:-midrop_step2}
OUT=$R/final_mask/training
LORA=$OUT/$NAME/last
W=$R/final_mask/midrop_val
NEED_MIB=29500
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore::FutureWarning
cd $REPO
mkdir -p $W $R/logs

stage() { echo; echo "===== [$(date '+%F %T')] $1 ====="; }
wait_mem() { while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -lt $1 ]; do sleep 60; done; }
run() {
  local name=$1 rc=1; shift
  # The GPU is shared: another job can take the memory after the check, so a crash is retried.
  for try in $(seq 1 10); do
    wait_mem 26000
    "$@" > $R/logs/$name.log 2>&1
    rc=$?
    echo "[$(date '+%F %T')] $name try $try exit=$rc"
    [ $rc -eq 0 ] && return 0
    sleep 300
  done
  echo "MIDROP_FAILED at $name"; tail -20 $R/logs/$name.log; exit $rc
}

stage "1/4 training"
RESUME=""
[ -f $OUT/$NAME/last_state.pt ] && RESUME="--resume"
for try in $(seq 1 10); do
  grep -q '"type": "done"' $OUT/$NAME/metrics.jsonl 2>/dev/null && break
  wait_mem $NEED_MIB
  echo "[$(date '+%F %T')] try $try $RESUME"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u sge_goal/final_mask/train_step2_midrop.py --step 2 \
    --name $NAME --out-dir $OUT --train-gt-mi --lr 5e-5 --warmup-steps 100 --micro-batch 4 --grad-accum 8 --epochs 1 \
    --val-every 172 --val-iou-samples 256 --val-loss-samples 64 --log-every 10 --num-workers 4 $RESUME \
    > $R/logs/fm_midrop_train.log 2>&1
  RESUME="--resume"
  sleep 120
done
grep -q '"type": "done"' $OUT/$NAME/metrics.jsonl || { echo "MIDROP_FAILED at training"; tail -20 $R/logs/fm_midrop_train.log; exit 1; }
grep '"type": "val"' $OUT/$NAME/metrics.jsonl | cut -c1-140

stage "2/4 language test, real M_i"
[ -f $W/language_real.json ] || run fm_midrop_lang_real $PY $R/final_mask/diag_language.py --per-pair 16 \
  --checkpoint-dir $LORA --work $R/CALVIN-EVALS/fm_lang_midrop --out $W/language_real.json

stage "3/4 language test, black M_i"
[ -f $W/language_black.json ] || run fm_midrop_lang_black $PY $R/final_mask/diag_language.py --per-pair 16 --black-mi \
  --checkpoint-dir $LORA --work $R/CALVIN-EVALS/fm_lang_midrop --out $W/language_black.json

stage "4/4 held-out IoU vs released"
[ -f $W/heldout_iou.json ] || run fm_midrop_heldout $PY $R/final_mask/score_step2_ckpts.py --out $W/heldout_iou.json \
  --ref released=$R/final_mask/ioudpo_val/heldout_iou.json --ckpt midrop=$LORA
echo MIDROP_DONE
