#!/bin/bash
# Predict initial masks with a trained learned-IER checkpoint into a new CALVIN-EVALS folder.
# Usage: bash run/predict_ier.sh --checkpoint ... --data-dir ... --out-dir ... [--gt-parquet ...]
R=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main/run
export PYTHONPATH=$R/pydeps:$R/third_party/EVF-SAM${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore::FutureWarning
cd $R/..
exec /home/sgeextn/.conda/envs/sgegen/bin/python sge_goal/grounding/learned_ier/predict_ier_mask.py "$@"
