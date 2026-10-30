#!/bin/bash
# Local ground-truth CALVIN eval: same as run_eval_full.sh but goals come from the HF test parquet's
# ``edited_image`` column (run/calvin/hf_dataset) instead of generated images.
# Usage: bash run/run_eval_gt.sh [N_SHARDS] [OUT_DIR]
R=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main/run
IN=$R/eval_inputs
PY=/home/sgeextn/.conda/envs/gate0/bin/python
N=${1:-8}
OUT=${2:-$R/eval_ground_truth}
PARQUET=$(ls $R/calvin/hf_dataset/data/test-*.parquet | head -1)
source $R/eval_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p $OUT/logs
cd $R/gr_mg_ws
pids=()
for ((i=0;i<N;i++)); do
  s=$((1087*i/N)); e=$((1087*(i+1)/N))
  if [ -f $OUT/shard_done_${s}_${e}.json ]; then echo "shard $s-$e already done"; continue; fi
  $PY evaluate/eval_method.py --dataset_path $IN/task_D_D --ckpt_dir $IN/ckpts/gr_mg --epoch 47 \
    --config_path policy/config/train.json --mapping_json $IN/hf_calvin_mapping.json \
    --output_dir $OUT --row_start $s --row_end $e --ep_len 360 --method_name ground_truth \
    --conditioning both --hf_parquet_goal $PARQUET > $OUT/logs/shard_${s}_${e}.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
echo "EVAL_EXIT=$fail"
