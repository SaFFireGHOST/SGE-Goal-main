# Quickstart: generate a goal image

This runs the full SGE-Goal pipeline (Step 1 initial mask → Step 2 final mask → Step 3 goal image)
on your own images with the released FLUX.2 klein adapters. Complete the
[installation](installation.md) first (Modal login and the `huggingface-secret`).

## 1. Download the adapters

```bash
huggingface-cli download RiddhiCh/SGE-Goal --local-dir sge-goal-checkpoints
```

## 2. Prepare the input folder

One numbered sub-directory per example, each holding the current observation and one **atomic**
instruction:

```
my_goals/
├── 1/
│   ├── original_image.jpg
│   └── edit_instruction.txt        # e.g. "put the spoon on the towel"
└── 2/
    ├── original_image.jpg
    └── edit_instruction.txt
```

Use the adapters trained on the dataset closest to your scene (BridgeDataV2 for real-world
tabletop scenes, CALVIN or LIBERO for those simulators).

## 3. Run the pipeline

The paths must be absolute.

```bash
cd sge_goal/pipelines/flux2_klein
python run_pipeline.py \
  --data-dir           "$PWD/../../../my_goals" \
  --fer-checkpoint-dir "$PWD/../../../sge-goal-checkpoints/flux2-klein/bridge/step2_final_mask" \
  --ie-checkpoint-dir  "$PWD/../../../sge-goal-checkpoints/flux2-klein/bridge/step3_goal_image"
```

Each step runs as a Modal job and writes its output next to the inputs:

```
my_goals/1/
├── original_image.jpg
├── edit_instruction.txt
├── initial_edit_region_mask.jpg     # Step 1
├── final_edit_region_mask.jpg       # Step 2
└── edited_image.jpg                 # Step 3 — the goal image
```

A step is skipped when every example already has its output, so a re-run resumes where it stopped.

## FLUX.1 Kontext variant

```bash
cd sge_goal/pipelines/flux_kontext_bridge_calvin        # flux_kontext_libero for LIBERO
python run_pipeline.py \
  --data-dir /abs/path/my_goals \
  --fer-checkpoint-dir /abs/path/sge-goal-checkpoints/flux1-kontext-dev/bridge/step2_final_mask \
  --image-editing-checkpoint-dir /abs/path/sge-goal-checkpoints/flux1-kontext-dev/bridge/step3_goal_image
```

To run all three steps on a local GPU instead of Modal, use
`sge_goal/pipelines/flux_kontext_bridge_calvin/run_pipeline_local_exp1.py` (its defaults point at
`dummy_test_data/` and the `sge-goal-checkpoints/` download above).

## Multi-step instructions

SGE-Goal works on atomic instructions. For a composite task, split it into atomic steps and run
them in order, using each step's `edited_image.jpg` as the next step's `original_image.jpg`.
