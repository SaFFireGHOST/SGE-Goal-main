#!/bin/bash
# Local SGE-Goal CALVIN eval: 1087 rows split into N contiguous shards (same split as modal_app.py: 1087*i//N).
# Everything it reads lives under run/ (eval_inputs/, CALVIN-EVALS/) plus ~/calvin (simulator).
# Usage: bash run/run_eval_full.sh [N_SHARDS] [OUT_DIR] [ROW_START ROW_END] [METHOD]
#   METHOD = goal-image folder under run/CALVIN-EVALS/ (default sge_goal)
R=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main/run
IN=$R/eval_inputs
PY=/home/sgeextn/.conda/envs/gate0/bin/python
N=${1:-8}
OUT=${2:-$R/eval_sge_goal}
RS=${3:-0}; RE=${4:-1087}
METHOD=${5:-sge_goal}
source $R/eval_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p $OUT/logs
cd $R/gr_mg_ws
pids=()
for ((i=0;i<N;i++)); do
  s=$((RS+(RE-RS)*i/N)); e=$((RS+(RE-RS)*(i+1)/N))
  if [ -f $OUT/shard_done_${s}_${e}.json ]; then echo "shard $s-$e already done"; continue; fi
  $PY evaluate/eval_method.py --dataset_path $IN/task_D_D --ckpt_dir $IN/ckpts/gr_mg --epoch 47 \
    --config_path policy/config/train.json --mapping_json $IN/hf_calvin_mapping.json \
    --output_dir $OUT --row_start $s --row_end $e --ep_len 360 --method_name $METHOD \
    --conditioning both --method_dir $R/CALVIN-EVALS/$METHOD > $OUT/logs/shard_${s}_${e}.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
echo "EVAL_EXIT=$fail"
