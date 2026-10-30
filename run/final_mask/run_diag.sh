#!/bin/bash
# Final-mask (Step 2) diagnostics, both local:
#   A. oracle: learned M_i + ground-truth final mask -> Step 3 (seed 0, as sge_goal_union_ier) -> GR-MG eval on
#      all 1087 test rows (6 shards, so Step 2 sampling fits on the GPU next to it) -> CL1-5 table.
#      Only M_f differs from sge_goal_union_ier (62.74 CL1): the causal effect of a perfect final mask.
#   B. Step 2 sampling on HF tuples (make_folders.py): 8 seeds x 256 val tuples with learned M_i, plus
#      ground-truth M_i on 256 val and 256 train tuples -> analyze_fer_diag.py.
# Every stage skips finished work, so rerunning resumes.
# Launch detached: setsid nohup bash run/final_mask/run_diag.sh > run/logs/final_mask_diag.log 2>&1 < /dev/null &
set -u
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
R=$REPO/run
PY=/home/sgeextn/.conda/envs/sgegen/bin/python
EVAL_PY=/home/sgeextn/.conda/envs/gate0/bin/python
AGG=$REPO/downstream/calvin/gr_mg/evaluate/aggregate_chain_success.py
CK=$R/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin
FER_DIR=$REPO/sge_goal/mask_predictor/inference/flux_kontext_bridge_calvin
GOAL_DIR=$REPO/sge_goal/synthesis/inference/flux_kontext_bridge_calvin
SUB=calvin_abcd_test_data
METHOD=sge_goal_union_gtmf
ORACLE=$R/CALVIN-EVALS/$METHOD/$SUB
EVAL_OUT=$R/eval_$METHOD
FD=$R/CALVIN-EVALS/fer_diag

stage() { echo; echo "===== [$(date '+%F %T')] $1 ====="; }
run() {
  local name=$1; shift
  "$@" > $R/logs/$name.log 2>&1
  local rc=$?
  echo "$name exit=$rc"
  if [ $rc -ne 0 ]; then echo "DIAG_FAILED at $name"; tail -20 $R/logs/$name.log; exit $rc; fi
}
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
mkdir -p $R/logs

stage "1/5 folders"
[ -f $R/final_mask/fer_diag_index.json ] || run fm_folders $PY $R/final_mask/make_folders.py
tail -2 $R/logs/fm_folders.log

stage "2/5 oracle Step 3 (learned M_i + ground-truth M_f)"
run fm_goal_oracle bash -c "cd $GOAL_DIR && $PY local_flux_kontext_inference.py --data-dir $ORACLE --checkpoint-dir $CK/step3_goal_image --skip-existing --low-vram"
echo "oracle goals: $(ls $ORACLE/*/edited_image.jpg | wc -l)"

stage "3/5 oracle rollouts (background, 6 shards)"
bash $R/run_eval_full.sh 6 $EVAL_OUT 0 1087 $METHOD > $R/logs/fm_eval_oracle.log 2>&1 &
EVAL_PID=$!
sleep 300  # let the rollout workers load before measuring free memory
until [ "$(free_mb)" -ge 24500 ]; do echo "[$(date '+%T')] free $(free_mb) MiB, waiting"; sleep 120; done
echo "free $(free_mb) MiB"

stage "4/5 Step 2 sampling"
for s in 0 1 2 3 4 5 6 7; do
  run fm_fer_val_pred_s$s bash -c "cd $FER_DIR && $PY local_fer_mask_inference.py --data-dir $FD/val_pred_s$s/$SUB --checkpoint-dir $CK/step2_final_mask --seed $((1000*s)) --skip-existing --low-vram"
done
for set in val_gt_s0 train_gt_s0; do
  run fm_fer_$set bash -c "cd $FER_DIR && $PY local_fer_mask_inference.py --data-dir $FD/$set/$SUB --checkpoint-dir $CK/step2_final_mask --seed 0 --skip-existing --low-vram"
done
run fm_fer_analysis $PY $R/final_mask/analyze_fer_diag.py
cat $R/logs/fm_fer_analysis.log
echo FER_DIAG_DONE

stage "5/5 wait for oracle rollouts, aggregate"
wait $EVAL_PID
n=$(cat $EVAL_OUT/per_sample_*.jsonl 2>/dev/null | wc -l)
echo "$METHOD rows=$n"
if [ "$n" -ne 1087 ]; then echo "DIAG_FAILED: oracle eval has $n/1087 rows"; tail -5 $R/logs/fm_eval_oracle.log; exit 1; fi
ln -sfn ../eval_$METHOD $R/eval_results/$METHOD
$EVAL_PY -I $AGG --chain_skeletons $R/eval_inputs/chain_skeletons.json --results_dir $R/eval_results \
  --methods $METHOD --out_csv $EVAL_OUT/table.csv --out_json $EVAL_OUT/table.json | tail -6
for d in eval_ground_truth eval_$METHOD eval_sge_goal_union_ier eval_sge_goal_gt_ier eval_sge_goal; do
  tail -n 1 $R/$d/table.csv
done
echo DIAG_DONE
