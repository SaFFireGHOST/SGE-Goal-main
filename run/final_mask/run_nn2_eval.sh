#!/bin/bash
# Tuned retrieval final masks (tune_retrieval.py: same-task pool, alpha 0.3, k 25, vote 0.4; test IoU 0.494 vs
# 0.422 for nn_baseline's nn5_vote and 0.341 for Step 2) + learned M_i -> Step 3 (seed 0, as sge_goal_union_ier)
# -> GR-MG eval on all 1087 test rows -> CL1-5 table. Only M_f differs from sge_goal_union_ier (62.74),
# sge_goal_union_nnmf (66.33) and sge_goal_union_gtmf (72.86).
# The GPU is shared: each GPU stage waits for enough free memory. Resumable.
# Launch detached: setsid nohup bash run/final_mask/run_nn2_eval.sh > run/logs/final_mask_nn2_eval.log 2>&1 < /dev/null &
set -u
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
R=$REPO/run
PY=/home/sgeextn/.conda/envs/sgegen/bin/python
EVAL_PY=/home/sgeextn/.conda/envs/gate0/bin/python
AGG=$REPO/downstream/calvin/gr_mg/evaluate/aggregate_chain_success.py
CK=$R/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin
GOAL_DIR=$REPO/sge_goal/synthesis/inference/flux_kontext_bridge_calvin
METHOD=sge_goal_union_nnmf2
DIR=$R/CALVIN-EVALS/$METHOD/calvin_abcd_test_data
EVAL_OUT=$R/eval_$METHOD

stage() { echo; echo "===== [$(date '+%F %T')] $1 ====="; }
wait_mem() { while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -lt $1 ]; do sleep 60; done; }
run() {
  local name=$1; shift
  "$@" > $R/logs/$name.log 2>&1
  local rc=$?
  echo "$name exit=$rc"
  if [ $rc -ne 0 ]; then echo "NN2_EVAL_FAILED at $name"; tail -20 $R/logs/$name.log; exit $rc; fi
}
mkdir -p $R/logs

stage "1/3 tuned retrieval final masks"
[ "$(ls $DIR/*/final_edit_region_mask.png 2>/dev/null | wc -l)" -eq 1087 ] || run fm_nn2_folder $PY $R/final_mask/tune_retrieval.py --write-test-folder
tail -4 $R/logs/fm_nn2_folder.log

stage "2/3 Step 3"
for try in 1 2 3 4 5; do
  wait_mem 25000
  [ "$(ls $DIR/*/edited_image.jpg 2>/dev/null | wc -l)" -eq 1087 ] && break
  (cd $GOAL_DIR && $PY local_flux_kontext_inference.py --data-dir $DIR --checkpoint-dir $CK/step3_goal_image \
    --skip-existing --low-vram) > $R/logs/fm_goal_nn2.log 2>&1 || sleep 120
done
echo "goals: $(ls $DIR/*/edited_image.jpg | wc -l)"
[ "$(ls $DIR/*/edited_image.jpg | wc -l)" -eq 1087 ] || { echo "NN2_EVAL_FAILED at Step 3"; tail -20 $R/logs/fm_goal_nn2.log; exit 1; }

stage "3/3 rollouts (8 shards) + aggregate"
wait_mem 21000
bash $R/run_eval_full.sh 8 $EVAL_OUT 0 1087 $METHOD
n=$(cat $EVAL_OUT/per_sample_*.jsonl 2>/dev/null | wc -l)
echo "$METHOD rows=$n"
if [ "$n" -ne 1087 ]; then echo "NN2_EVAL_FAILED: $n/1087 rows"; exit 1; fi
ln -sfn ../eval_$METHOD $R/eval_results/$METHOD
$EVAL_PY -I $AGG --chain_skeletons $R/eval_inputs/chain_skeletons.json --results_dir $R/eval_results \
  --methods $METHOD --out_csv $EVAL_OUT/table.csv --out_json $EVAL_OUT/table.json | tail -6
for d in eval_ground_truth eval_sge_goal_union_gtmf eval_$METHOD eval_sge_goal_union_nnmf eval_sge_goal_union_ier; do
  tail -n 1 $R/$d/table.csv
done
echo NN2_EVAL_DONE
