#!/bin/bash
# Step 2 samples for IoU-ranked Flow-DPO pairs: 2000 HF train tuples x 4 samples, learned M_i, scored by IoU
# against the ground-truth final mask (sge_goal/final_mask/sample_step2.py) -> run/final_mask/step2_samples/.
# The GPU is shared, so an OOM is retried (resumable) once enough memory is free.
# Launch detached: setsid nohup bash run/final_mask/run_sample_step2.sh > run/logs/fm_sample_step2.log 2>&1 < /dev/null &
set -u
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
PY=/home/sgeextn/.conda/envs/sgegen/bin/python
OUT=$REPO/run/final_mask/step2_samples
NEED_MIB=26000

cd $REPO
for try in $(seq 1 20); do
  while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -lt $NEED_MIB ]; do sleep 60; done
  echo "===== [$(date '+%F %T')] try $try ====="
  $PY -u sge_goal/final_mask/sample_step2.py --out-dir $OUT --num-rows 2000 --k 4 --batch-rows 4 2>&1 \
    | grep -v "truncated because CLIP" | grep -v "^\['"
  [ "$(wc -l < $OUT/samples.jsonl)" -ge 2000 ] && { echo SAMPLE_STEP2_DONE; exit 0; }
  sleep 120
done
echo SAMPLE_STEP2_FAILED
exit 1
