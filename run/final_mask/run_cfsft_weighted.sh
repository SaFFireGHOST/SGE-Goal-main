#!/bin/bash
# Supervised counterfactual overfit with the flow loss weighted on the cells where M_c and M_o differ (x 1+DW), 16 train
# pairs, 300 updates; then 10-step sampled masks on the same pairs + 32 held-out pairs (sample_cf_pairs.py).
# Pass: sampled cf_hit on the 16 pairs well above midrop's 0.27 (~0.9 = memorised); text_shift well above 0.04.
# Launch detached: setsid nohup bash run/final_mask/run_cfsft_weighted.sh > run/logs/fm_cfsft_weighted.log 2>&1 < /dev/null &
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-/home/sgeextn/.conda/envs/sgegen/bin/python}; [ -x "$PY" ] || PY=$(command -v python)
cd $REPO
NAME=${NAME:-cfsft_w20}
DW=${DW:-20}
NAME=$NAME CLIP=1 STEPS=${STEPS:-300} VAL_EVERY=50 EXTRA="--sft-cf --diff-weight $DW" bash run/final_mask/run_cfdpo_overfit.sh
[ -f run/final_mask/training/$NAME/last/pytorch_lora_weights.safetensors ] || { echo "WEIGHTED_FAILED training"; exit 1; }
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false $PY -u \
  run/final_mask/sample_cf_pairs.py --ckpt $NAME=run/final_mask/training/$NAME/last \
  --out run/final_mask/cf_sample/$NAME.json > run/logs/fm_cf_sample_$NAME.log 2>&1 || { echo "WEIGHTED_FAILED sampling"; exit 1; }
grep "^$NAME {" run/logs/fm_cf_sample_$NAME.log
echo WEIGHTED_DONE
