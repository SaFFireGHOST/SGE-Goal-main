#!/bin/bash
# CALVIN eval (GR-MG, all 1087 rows) of the learned-IER goal sets, run detached:
#   sge_goal_union_ier  (learned union M_i + predicted M_f)
#   sge_goal_gt_ier     (ground-truth M_i + predicted M_f)
# then aggregates CL1-5 and prints them next to the baseline and ground-truth-goal results.
# Resumable: finished shards are skipped. Progress: run/logs/ier_evals.log
# Launch: setsid nohup bash run/run_ier_evals.sh > run/logs/ier_evals.log 2>&1 < /dev/null &
set -u
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
R=$REPO/run
PY=/home/sgeextn/.conda/envs/gate0/bin/python
AGG=$REPO/downstream/calvin/gr_mg/evaluate/aggregate_chain_success.py
mkdir -p $R/eval_results

for METHOD in sge_goal_union_ier sge_goal_gt_ier; do
  OUT=$R/eval_$METHOD
  echo; echo "===== [$(date '+%F %T')] eval $METHOD ====="
  bash $R/run_eval_full.sh 8 $OUT 0 1087 $METHOD
  n=$(cat $OUT/per_sample_*.jsonl 2>/dev/null | wc -l)
  echo "$METHOD rows=$n"
  if [ "$n" -ne 1087 ]; then echo "EVALS_FAILED: $METHOD has $n/1087 rows"; exit 1; fi
  ln -sfn ../eval_$METHOD $R/eval_results/$METHOD
  $PY -I $AGG --chain_skeletons $R/eval_inputs/chain_skeletons.json --results_dir $R/eval_results \
    --methods $METHOD --out_csv $OUT/table.csv --out_json $OUT/table.json | tail -6
done

echo; echo "===== [$(date '+%F %T')] summary (GR-MG SR%, CL1..CL5) ====="
for d in eval_ground_truth eval_sge_goal_gt_ier eval_sge_goal_union_ier eval_sge_goal; do
  tail -n 1 $R/$d/table.csv
done
echo EVALS_DONE
