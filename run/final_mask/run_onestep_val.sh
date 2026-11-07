#!/bin/bash
# One-step Step 2 masks (predict_onestep.py, threshold 64) on the 300 va_val contexts, compared with Ref (64.2).
# Shared with Ref (run/variant_a/run_val_eval.sh): learned M_i, released Step 3, Step 3 seed 5000, the 2 start states
# per context. Only M_f differs. Every stage skips finished work, so rerunning resumes.
# Launch detached: setsid nohup bash run/final_mask/run_onestep_val.sh > run/logs/fm_onestep_val.log 2>&1 < /dev/null &
set -u
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
R=$REPO/run
V=$R/CALVIN-EVALS/va_val
SUB=calvin_abcd_test_data
DIR=$V/onestep/$SUB
P=$R/variant_a/val
OUT=$R/final_mask/onestep_val/rollouts
GEN_PY=/home/sgeextn/.conda/envs/sgegen/bin/python
EVAL_PY=/home/sgeextn/.conda/envs/gate0/bin/python
CK=$R/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin
GOAL_DIR=$REPO/sge_goal/synthesis/inference/flux_kontext_bridge_calvin
N_SHARDS=8
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

stage() { echo; echo "===== [$(date '+%F %T')] $1 ====="; }
wait_mem() { while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -lt $1 ]; do sleep 60; done; }
run() {
  local name=$1; shift
  "$@" > $R/logs/$name.log 2>&1
  local rc=$?
  echo "$name exit=$rc"
  if [ $rc -ne 0 ]; then echo "ONESTEP_VAL_FAILED at $name"; tail -20 $R/logs/$name.log; exit $rc; fi
}
mkdir -p $R/logs $OUT/logs $R/final_mask/onestep_val

stage "1/5 one-step IoU on all HF validation tuples"
JSON=$R/final_mask/onestep_val/validation_iou.json
wait_mem 26000
[ -f $JSON ] || run fm_onestep_iou $GEN_PY $R/final_mask/predict_onestep.py --validation $JSON
cat $JSON

stage "2/5 one-step masks for va_val"
for case in $(ls $V/base/$SUB); do
  mkdir -p $DIR/$case
  for f in original_image.png edit_instruction.txt initial_edit_region_mask.jpg; do
    cp -n $V/base/$SUB/$case/$f $DIR/$case/
  done
done
wait_mem 26000
run fm_onestep_masks $GEN_PY $R/final_mask/predict_onestep.py --data-dir $DIR
echo "M_f: $(ls $DIR/*/final_edit_region_mask.png | wc -l)"

stage "3/5 Step 3 (released, seed 5000)"
for try in 1 2 3 4 5; do
  wait_mem 25000
  [ "$(ls $DIR/*/edited_image.jpg 2>/dev/null | wc -l)" -eq 300 ] && break
  # load_lora_weights(dir) without a weight_name refuses to run with HF_HUB_OFFLINE set.
  (cd $GOAL_DIR && env -u HF_HUB_OFFLINE $GEN_PY local_flux_kontext_inference.py --data-dir $DIR --checkpoint-dir $CK/step3_goal_image \
    --seed 5000 --skip-existing --low-vram) > $R/logs/fm_onestep_goal.log 2>&1 || sleep 120
done
echo "goals: $(ls $DIR/*/edited_image.jpg | wc -l)"
[ "$(ls $DIR/*/edited_image.jpg | wc -l)" -eq 300 ] || { echo "ONESTEP_VAL_FAILED at Step 3"; tail -20 $R/logs/fm_onestep_goal.log; exit 1; }

stage "4/5 rollouts (${N_SHARDS} shards)"
wait_mem 21000
source $R/eval_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd $R/gr_mg_ws
pids=()
for ((i=0;i<N_SHARDS;i++)); do
  if [ -f $OUT/shard_done_s$i.json ]; then echo "shard $i already done"; continue; fi
  $EVAL_PY $R/variant_a/eval_contexts.py --dataset_path $R/eval_inputs/task_D_D --ckpt_dir $R/eval_inputs/ckpts/gr_mg \
    --epoch 47 --config_path policy/config/train.json --method_dir $V --variants onestep \
    --episodes_file $P/eps_shard$i.txt --tag s$i --num_perturb 2 --perturb_std 0.02 --output_dir $OUT \
    > $OUT/logs/shard_s$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
n=$(cat $OUT/per_sample_*.jsonl | wc -l)
echo "rollouts=$n (expected 600) shard_fail=$fail"
if [ $fail -ne 0 ]; then echo "ONESTEP_VAL_FAILED at rollouts"; exit 1; fi

stage "5/5 paired comparison with Ref"
cd $REPO
$GEN_PY $R/final_mask/analyze_onestep_val.py
echo ONESTEP_VAL_DONE
