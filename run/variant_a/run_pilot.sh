#!/bin/bash
# A-sparse pilot data (plan.md §6.2): 300 D-training contexts, K_m = 2 final masks x K_a = 2 goal images
# per mask x K_s = 2 start states = 8 rollouts per context (+ 1 ground-truth-goal rollout as reference).
#   1. contexts + folders (make_pilot_contexts.py)          CALVIN-EVALS/va_pilot/{src,gt}
#   2. learned M_i (union model)                              CALVIN-EVALS/va_pilot/base
#   3. Step 2 per mask k (seed 1000(k+1)) in m<k>a0, copied to m<k>a1; Step 3 per folder (own seed)
#   4. rollouts, 8 shards (eval_contexts.py)                  run/variant_a/pilot/rollouts
# Every stage skips finished work, so rerunning resumes.
# Launch detached: setsid nohup bash run/variant_a/run_pilot.sh > run/logs/va_pilot.log 2>&1 < /dev/null &
set -u
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
R=$REPO/run
P=$R/variant_a/pilot
V=$R/CALVIN-EVALS/va_pilot
GEN_PY=/home/sgeextn/.conda/envs/sgegen/bin/python
EVAL_PY=/home/sgeextn/.conda/envs/gate0/bin/python
CK=$R/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin
FER_DIR=$REPO/sge_goal/mask_predictor/inference/flux_kontext_bridge_calvin
GOAL_DIR=$REPO/sge_goal/synthesis/inference/flux_kontext_bridge_calvin
OUT=$P/rollouts
N_SHARDS=8
SUB=calvin_abcd_test_data

stage() { echo; echo "===== [$(date '+%F %T')] $1 ====="; }
run() {
  local name=$1; shift
  "$@" > $R/logs/$name.log 2>&1
  local rc=$?
  echo "$name exit=$rc"
  if [ $rc -ne 0 ]; then echo "PILOT_FAILED at $name"; tail -20 $R/logs/$name.log; exit $rc; fi
}
mkdir -p $R/logs

stage "1/4 contexts"
[ -f $P/contexts.json ] || run va_pilot_contexts $GEN_PY $R/variant_a/make_pilot_contexts.py

stage "2/4 learned M_i"
run va_pilot_ier bash $R/predict_ier.sh --checkpoint $R/ier_training/evf_union/best.pt \
  --data-dir $V/src/$SUB --out-dir $V/base/$SUB --skip-existing
echo "M_i: $(ls $V/base/$SUB/*/initial_edit_region_mask.jpg | wc -l)"

stage "3/4 Step 2 (2 masks) and Step 3 (2 goal images per mask)"
for k in 0 1; do
  A0=$V/m${k}a0/$SUB
  A1=$V/m${k}a1/$SUB
  mkdir -p $A0 && cp -rn $V/base/$SUB/. $A0/
  run va_pilot_fer_m$k bash -c "cd $FER_DIR && $GEN_PY local_fer_mask_inference.py --data-dir $A0 --checkpoint-dir $CK/step2_final_mask --seed $((1000*(k+1))) --skip-existing --low-vram"
  mkdir -p $A1
  for d in $A0/*/; do
    c=$(basename $d); mkdir -p $A1/$c
    for f in $d*; do case $(basename $f) in edited_image.*) ;; *) cp -n $f $A1/$c/ ;; esac; done
  done
  for j in 0 1; do
    run va_pilot_goal_m${k}a$j bash -c "cd $GOAL_DIR && $GEN_PY local_flux_kontext_inference.py --data-dir $V/m${k}a$j/$SUB --checkpoint-dir $CK/step3_goal_image --seed $((5000 + 1000*(2*k+j))) --skip-existing --low-vram"
    echo "m${k}a$j: $(ls $V/m${k}a$j/$SUB/*/edited_image.jpg | wc -l) goal images"
  done
done

stage "4/4 rollouts (${N_SHARDS} shards)"
mkdir -p $OUT/logs
$GEN_PY -c "
import json
eps = json.load(open('$P/contexts.json'))
n = $N_SHARDS
for i in range(n):
    open(f'$P/eps_shard{i}.txt', 'w').write(','.join(map(str, eps[len(eps)*i//n:len(eps)*(i+1)//n])))
"
source $R/eval_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd $R/gr_mg_ws
pids=()
for ((i=0;i<N_SHARDS;i++)); do
  if [ -f $OUT/shard_done_s$i.json ]; then echo "shard $i already done"; continue; fi
  $EVAL_PY $R/variant_a/eval_contexts.py --dataset_path $R/eval_inputs/task_D_D --ckpt_dir $R/eval_inputs/ckpts/gr_mg \
    --epoch 47 --config_path policy/config/train.json --method_dir $V --variants m0a0,m0a1,m1a0,m1a1,gt:1 \
    --episodes_file $P/eps_shard$i.txt --tag s$i --num_perturb 2 --perturb_std 0.02 --output_dir $OUT \
    > $OUT/logs/shard_s$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
n=$(cat $OUT/per_sample_*.jsonl | wc -l)
echo "rollouts=$n (expected 2700) shard_fail=$fail"
if [ $fail -ne 0 ]; then echo "PILOT_FAILED at rollouts"; exit 1; fi
echo PILOT_DONE
