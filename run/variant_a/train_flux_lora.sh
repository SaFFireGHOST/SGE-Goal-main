#!/bin/bash
# Variant A LoRA trainer for SGE-Goal Steps 2-3 (sge_goal/variant_a/flux_lora_trainer.py).
# Usage: bash run/variant_a/train_flux_lora.sh --step 2 --name asup_step2 [--epochs 1 ...]
REPO=/home/sgeextn/SGE-Goal-Cleaned/SGE-Goal-main
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore::FutureWarning
cd $REPO
exec /home/sgeextn/.conda/envs/sgegen/bin/python -u sge_goal/variant_a/flux_lora_trainer.py "$@"
