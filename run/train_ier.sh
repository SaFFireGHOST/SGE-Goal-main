#!/bin/bash
# Train the learned IER (initial mask) predictor on CALVIN with EVF-SAM.
# EVF-SAM code lives in run/third_party/EVF-SAM; its extra deps (torchscale, fairscale) in run/pydeps.
# Usage: bash run/train_ier.sh --name evf_union [--epochs 5 --batch-size 8 ...]
R=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main/run
export PYTHONPATH=$R/pydeps:$R/third_party/EVF-SAM${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore::FutureWarning
cd $R/..
exec /home/sgeextn/.conda/envs/sgegen/bin/python sge_goal/grounding/learned_ier/train_ier_mask.py "$@"
