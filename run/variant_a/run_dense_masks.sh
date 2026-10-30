#!/bin/bash
# A-dense conditions for all Variant A training demos: learned M_i, then Step 2 M_f (seed 1000, as the
# pilot's m0). Waits until the pilot (run_pilot.sh) has finished generating and >= 22 GB of GPU memory is
# free, so it can share the GPU with the pilot's rollouts. Resumable (--skip-existing).
# Launch detached: setsid nohup bash run/variant_a/run_dense_masks.sh > run/logs/va_dense_masks.log 2>&1 < /dev/null &
set -u
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
R=$REPO/run
PY=/home/sgeextn/.conda/envs/sgegen/bin/python
V=$R/CALVIN-EVALS/va_dense
SUB=calvin_abcd_test_data
PILOT=$R/CALVIN-EVALS/va_pilot/m0a0/$SUB
CK=$R/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin
FER_DIR=$REPO/sge_goal/mask_predictor/inference/flux_kontext_bridge_calvin

stage() { echo; echo "===== [$(date '+%F %T')] $1 ====="; }
run() {
  local name=$1; shift
  "$@" > $R/logs/$name.log 2>&1
  local rc=$?
  echo "$name exit=$rc"
  if [ $rc -ne 0 ]; then echo "DENSE_MASKS_FAILED at $name"; tail -20 $R/logs/$name.log; exit $rc; fi
}
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }

stage "1/4 folders"
run va_dense_folders $PY $R/variant_a/make_dense_folders.py
tail -1 $R/logs/va_dense_folders.log

stage "2/4 wait for the pilot's generation to finish and GPU memory"
until grep -qE "4/4 rollouts|PILOT_DONE|PILOT_FAILED" $R/logs/va_pilot.log; do sleep 60; done
sleep 300  # let the rollout workers load first
until [ "$(free_mb)" -ge 22000 ]; do echo "[$(date '+%T')] free $(free_mb) MiB, waiting"; sleep 300; done
echo "free $(free_mb) MiB"

stage "3/4 learned M_i (pilot demos reuse m0a0)"
mkdir -p $V/cond/$SUB
for d in $PILOT/*/; do
  c=$(basename $d); mkdir -p $V/cond/$SUB/$c
  for f in original_image.png edit_instruction.txt initial_edit_region_mask.jpg final_edit_region_mask.jpg; do
    cp -n $d$f $V/cond/$SUB/$c/
  done
done
run va_dense_ier bash $R/predict_ier.sh --checkpoint $R/ier_training/evf_union/best.pt \
  --data-dir $V/src/$SUB --out-dir $V/cond/$SUB --skip-existing
echo "M_i: $(ls $V/cond/$SUB/*/initial_edit_region_mask.jpg | wc -l)"

stage "4/4 Step 2 M_f"
run va_dense_fer bash -c "cd $FER_DIR && $PY local_fer_mask_inference.py --data-dir $V/cond/$SUB --checkpoint-dir $CK/step2_final_mask --seed 1000 --skip-existing --low-vram"
echo "M_f: $(ls $V/cond/$SUB/*/final_edit_region_mask.jpg | wc -l)"
echo DENSE_MASKS_DONE
