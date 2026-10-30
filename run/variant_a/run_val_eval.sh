#!/bin/bash
# Variant A held-out comparison on the 300 `val` contexts (make_val_contexts.py). Only Step 3 differs:
#   ref     released step3_goal_image
#   asparse pilot Flow-DPO Step 3, last epoch  (training/pilot_dpo_step3/ckpt_00060)
#   adense  A-dense, 2 epochs, best val_beh    (training/adense_2ep/ckpt_02290)
# Shared across variants: learned M_i, Step 2 M_f (released model, seed 1000), Step 3 seed 5000, the
# 2 start states per context (eval_contexts.py), so the comparison is paired. gt:1 = end-frame goal ceiling.
# Every stage skips finished work, so rerunning resumes.
# Launch detached: setsid nohup bash run/variant_a/run_val_eval.sh > run/logs/va_val_eval.log 2>&1 < /dev/null &
set -u
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
R=$REPO/run
P=$R/variant_a/val
V=$R/CALVIN-EVALS/va_val
GEN_PY=/home/sgeextn/.conda/envs/sgegen/bin/python
EVAL_PY=/home/sgeextn/.conda/envs/gate0/bin/python
CK=$R/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin
T=$R/variant_a/training
FER_DIR=$REPO/sge_goal/mask_predictor/inference/flux_kontext_bridge_calvin
GOAL_DIR=$REPO/sge_goal/synthesis/inference/flux_kontext_bridge_calvin
OUT=$P/rollouts
N_SHARDS=8
SUB=calvin_abcd_test_data
declare -A LORA=([ref]=$CK/step3_goal_image [asparse]=$T/pilot_dpo_step3/ckpt_00060 [adense]=$T/adense_2ep/ckpt_02290)
VARIANTS="ref asparse adense"

stage() { echo; echo "===== [$(date '+%F %T')] $1 ====="; }
run() {
  local name=$1; shift
  "$@" > $R/logs/$name.log 2>&1
  local rc=$?
  echo "$name exit=$rc"
  if [ $rc -ne 0 ]; then echo "VAL_EVAL_FAILED at $name"; tail -20 $R/logs/$name.log; exit $rc; fi
}
mkdir -p $R/logs

stage "1/4 contexts"
[ -f $P/contexts.json ] || run va_val_contexts $GEN_PY $R/variant_a/make_val_contexts.py

stage "2/4 learned M_i + Step 2 M_f (shared)"
run va_val_ier bash $R/predict_ier.sh --checkpoint $R/ier_training/evf_union/best.pt \
  --data-dir $V/src/$SUB --out-dir $V/base/$SUB --skip-existing
run va_val_fer bash -c "cd $FER_DIR && $GEN_PY local_fer_mask_inference.py --data-dir $V/base/$SUB --checkpoint-dir $CK/step2_final_mask --seed 1000 --skip-existing --low-vram"
echo "M_i: $(ls $V/base/$SUB/*/initial_edit_region_mask.jpg | wc -l)  M_f: $(ls $V/base/$SUB/*/final_edit_region_mask.jpg | wc -l)"

stage "3/4 Step 3 per variant"
for v in $VARIANTS; do
  mkdir -p $V/$v/$SUB && cp -rn $V/base/$SUB/. $V/$v/$SUB/
  run va_val_goal_$v bash -c "cd $GOAL_DIR && $GEN_PY local_flux_kontext_inference.py --data-dir $V/$v/$SUB --checkpoint-dir ${LORA[$v]} --seed 5000 --skip-existing --low-vram"
  echo "$v: $(ls $V/$v/$SUB/*/edited_image.jpg | wc -l) goal images"
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
    --epoch 47 --config_path policy/config/train.json --method_dir $V --variants ref,asparse,adense,gt:1 \
    --episodes_file $P/eps_shard$i.txt --tag s$i --num_perturb 2 --perturb_std 0.02 --output_dir $OUT \
    > $OUT/logs/shard_s$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
n=$(cat $OUT/per_sample_*.jsonl | wc -l)
echo "rollouts=$n (expected 2100) shard_fail=$fail"
if [ $fail -ne 0 ]; then echo "VAL_EVAL_FAILED at rollouts"; exit 1; fi
echo VAL_EVAL_DONE
