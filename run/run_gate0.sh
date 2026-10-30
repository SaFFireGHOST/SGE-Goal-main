#!/bin/bash
# Gate 0 (smaller design): 300 test contexts x 4 goal samples x 2 start states = 2,400 rollouts.
#   1. folders: v0 = existing learned-IER goals (seed 0); v1-v3 = Steps 2-3 re-sampled with seeds 1000/2000/3000
#   2. generate v1-v3 (Step 2 then Step 3, low-VRAM, resumable)
#   3. rollouts: 8 shards, every variant x start state per row (eval_gate0.py)
#   4. analysis -> run/eval_gate0/gate0_result.json
# Launch detached: setsid nohup bash run/run_gate0.sh > run/logs/gate0.log 2>&1 < /dev/null &
set -u
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
R=$REPO/run
G=$R/gate0
GEN_PY=/home/sgeextn/.conda/envs/sgegen/bin/python
EVAL_PY=/home/sgeextn/.conda/envs/gate0/bin/python
CK=$R/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin
FER_DIR=$REPO/sge_goal/mask_predictor/inference/flux_kontext_bridge_calvin
GOAL_DIR=$REPO/sge_goal/synthesis/inference/flux_kontext_bridge_calvin
OUT=$R/eval_gate0
N_SHARDS=8

stage() { echo; echo "===== [$(date '+%F %T')] $1 ====="; }
run() {
  local name=$1; shift
  "$@" > $R/logs/$name.log 2>&1
  local rc=$?
  echo "$name exit=$rc"
  if [ $rc -ne 0 ]; then echo "GATE0_FAILED at $name"; tail -20 $R/logs/$name.log; exit $rc; fi
}

stage "1/4 select contexts and build goal-sample folders"
[ -f $G/rows.json ] || run gate0_folders $GEN_PY $G/make_gate0_folders.py

stage "2/4 re-sample Steps 2-3 for v1-v3"
for k in 1 2 3; do
  D=$R/CALVIN-EVALS/gate0/v$k/calvin_abcd_test_data
  run gate0_fer_v$k bash -c "cd $FER_DIR && $GEN_PY local_fer_mask_inference.py --data-dir $D --checkpoint-dir $CK/step2_final_mask --seed $((1000*k)) --skip-existing --low-vram"
  run gate0_goal_v$k bash -c "cd $GOAL_DIR && $GEN_PY local_flux_kontext_inference.py --data-dir $D --checkpoint-dir $CK/step3_goal_image --seed $((1000*k)) --skip-existing --low-vram"
  echo "v$k: $(ls $D/*/edited_image.jpg | wc -l) goal images"
done

stage "3/4 rollouts (${N_SHARDS} shards)"
mkdir -p $OUT/logs
$GEN_PY -c "
import json
rows = json.load(open('$G/rows.json'))
n = $N_SHARDS
for i in range(n):
    open(f'$G/rows_shard{i}.txt', 'w').write(','.join(map(str, rows[len(rows)*i//n:len(rows)*(i+1)//n])))
"
source $R/eval_env.sh
export PYTHONPATH=$PYTHONPATH:$G
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd $R/gr_mg_ws
pids=()
for ((i=0;i<N_SHARDS;i++)); do
  if [ -f $OUT/shard_done_s$i.json ]; then echo "shard $i already done"; continue; fi
  $EVAL_PY $G/eval_gate0.py --dataset_path $R/eval_inputs/task_D_D --ckpt_dir $R/eval_inputs/ckpts/gr_mg \
    --epoch 47 --config_path policy/config/train.json --mapping_json $R/eval_inputs/hf_calvin_mapping.json \
    --method_dir $R/CALVIN-EVALS/gate0 --variants v0,v1,v2,v3 --rows_file $G/rows_shard$i.txt \
    --tag s$i --num_perturb 2 --perturb_std 0.02 --output_dir $OUT > $OUT/logs/shard_s$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
n=$(cat $OUT/per_sample_*.jsonl | wc -l)
echo "rollouts=$n (expected 2400) shard_fail=$fail"
if [ $fail -ne 0 ]; then echo "GATE0_FAILED at rollouts"; exit 1; fi

stage "4/4 analysis"
cd $REPO
$GEN_PY -I $G/analyze_gate0.py
echo GATE0_DONE
