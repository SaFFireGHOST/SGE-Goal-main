# `sge_goal/` — the SGE-Goal framework

Three-step, spatially-grounded goal-image synthesis: initial mask `M_i` → final mask `M_f` →
goal image `Î_e`. Each leaf directory is a **self-contained unit**: its scripts import their
siblings and the Modal launchers upload the unit's own directory (`add_local_dir(".")`), so run
every script from inside its own directory.

```
sge_goal/
├── grounding/                 # Step 1 — training-free M_i (Florence-2 + Grounding DINO + BLIP-2 + SAM)
│   ├── flux_kontext_bridge_calvin/
│   ├── flux_kontext_libero/
│   └── flux2_klein/
├── mask_predictor/            # Step 2 — M_f, LoRA + channel-consistency loss (λ = 25)
│   ├── train/{flux_kontext,flux2_klein}_{bridge,calvin,libero}/
│   └── inference/{flux_kontext_bridge_calvin,flux_kontext_libero,flux2_klein}/
├── synthesis/                 # Step 3 — Î_e, LoRA + static-scene-consistency loss (λ = 7)
│   ├── train/{flux_kontext,flux2_klein}_{bridge,calvin,libero}/
│   └── inference/{flux_kontext_bridge_calvin,flux_kontext_libero,flux2_klein}/
├── pipelines/                 # end-to-end drivers that chain Steps 1 → 2 → 3
└── requirements.txt
```

## Training

Both fine-tuned stages are trained on [Modal](https://modal.com) with `modal_*_train.py`, which
holds the run configuration (hyperparameters, dataset repo, resolution) in its `TRAINING_CONFIG` dict and
calls the stage's trainer (`flux_kontext_lora_finetuning.py`, `flux_2_klein_lora_finetuning.py`,
`flux_2_klein_erm_finetuning.py`). The FLUX.1 Kontext LIBERO units
(`*/train/flux_kontext_libero/`) were run locally instead; see `monitor_training.sh` /
`benchmark_batch_sizes.py` there for the command lines.

Training data are the processed datasets on Hugging Face (see the top-level README).

## Inference

```bash
cd sge_goal/pipelines/flux2_klein
python run_pipeline.py \
  --data-dir /abs/path/to/test_data \
  --fer-checkpoint-dir /abs/path/to/step2_lora_checkpoint \
  --ie-checkpoint-dir  /abs/path/to/step3_lora_checkpoint
```

`--data-dir` holds one numbered sub-directory per test case containing `original_image.*` and
`edit_instruction.txt`; each step writes `initial_edit_region_mask.jpg`,
`final_edit_region_mask.jpg` and `edited_image.jpg` next to them. The FLUX.1 Kontext drivers
(`pipelines/flux_kontext_*/run_pipeline.py`) take `--image-editing-checkpoint-dir` instead of
`--ie-checkpoint-dir`.

The `*_exp1` drivers are the BridgeDataV2 variants of the pipeline. `run_pipeline_local_exp1.py`
runs all three steps locally and records inference-time benchmarks, and
`run_pipeline_bridge_dummy_latency.py` reports per-step latency on a small dummy set; raw timing
outputs are in `scripts/runtime/results/`.

LoRA checkpoints are not stored in git; they will be released on Hugging Face.
