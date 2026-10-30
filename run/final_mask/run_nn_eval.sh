#!/bin/bash
# Does a better final mask from another source raise success? Retrieval final masks (nn_baseline.py nn5_vote,
# test IoU 0.42 vs Step 2's 0.34) + learned M_i -> Step 3 (seed 0, as sge_goal_union_ier) -> GR-MG eval on all
# 1087 test rows -> CL1-5 table. Only M_f differs from sge_goal_union_ier (62.74) and sge_goal_union_gtmf (72.86).
# Resumable. Launch detached: setsid nohup bash run/final_mask/run_nn_eval.sh > run/logs/final_mask_nn_eval.log 2>&1 < /dev/null &
set -u
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
R=$REPO/run
PY=/home/sgeextn/.conda/envs/sgegen/bin/python
EVAL_PY=/home/sgeextn/.conda/envs/gate0/bin/python
AGG=$REPO/downstream/calvin/gr_mg/evaluate/aggregate_chain_success.py
CK=$R/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin
GOAL_DIR=$REPO/sge_goal/synthesis/inference/flux_kontext_bridge_calvin
METHOD=sge_goal_union_nnmf
DIR=$R/CALVIN-EVALS/$METHOD/calvin_abcd_test_data
EVAL_OUT=$R/eval_$METHOD

stage() { echo; echo "===== [$(date '+%F %T')] $1 ====="; }
run() {
  local name=$1; shift
  "$@" > $R/logs/$name.log 2>&1
  local rc=$?
  echo "$name exit=$rc"
  if [ $rc -ne 0 ]; then echo "NN_EVAL_FAILED at $name"; tail -20 $R/logs/$name.log; exit $rc; fi
}
mkdir -p $R/logs

stage "1/3 retrieval final masks"
[ "$(ls $DIR/*/final_edit_region_mask.png 2>/dev/null | wc -l)" -eq 1087 ] || run fm_nn_folder $PY $R/final_mask/nn_baseline.py --write-test-folder
cat $R/logs/fm_nn_folder.log

stage "2/3 Step 3"
run fm_goal_nn bash -c "cd $GOAL_DIR && $PY local_flux_kontext_inference.py --data-dir $DIR --checkpoint-dir $CK/step3_goal_image --skip-existing --low-vram"
echo "goals: $(ls $DIR/*/edited_image.jpg | wc -l)"

stage "3/3 rollouts (8 shards) + aggregate"
bash $R/run_eval_full.sh 8 $EVAL_OUT 0 1087 $METHOD
n=$(cat $EVAL_OUT/per_sample_*.jsonl 2>/dev/null | wc -l)
echo "$METHOD rows=$n"
if [ "$n" -ne 1087 ]; then echo "NN_EVAL_FAILED: $n/1087 rows"; exit 1; fi
ln -sfn ../eval_$METHOD $R/eval_results/$METHOD
$EVAL_PY -I $AGG --chain_skeletons $R/eval_inputs/chain_skeletons.json --results_dir $R/eval_results \
  --methods $METHOD --out_csv $EVAL_OUT/table.csv --out_json $EVAL_OUT/table.json | tail -6
for d in eval_ground_truth eval_sge_goal_union_gtmf eval_$METHOD eval_sge_goal_union_ier; do tail -n 1 $R/$d/table.csv; done
echo NN_EVAL_DONE
