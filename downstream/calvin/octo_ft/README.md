# `octo_ft/` — Octo fine-tuned on CALVIN (Octo-ft)

The third CALVIN policy in the paper's supplement (Sec. A). GR-MG and GHIL-Glue are released checkpoints;
to check that the ranking of goal-image generators does not depend on them, we fine-tune
[Octo-Base](https://github.com/octo-models/octo) on real CALVIN frames and evaluate it, frozen and
conditioned on the **goal image only**, on the same 1,087 test rows and chains.

| Result (CALVIN, goal image only) | |
|---|---|
| Ground-truth goal (oracle), chain length 1 | 36.2% |
| SGE-Goal (FLUX.2 klein / FLUX.1 Kontext), chain length 1 | 35.6% / 34.7% |
| Best other generator, chain length 1 | 21.3% (Clover) |
| Spearman ρ of the generator ranking vs GR-MG / GHIL-Glue | 0.95 / 0.88 |

The policy never sees a generator's output during training, so it cannot favour any of them.

## What is in this folder

| Path | Purpose |
|---|---|
| `data/calvin_train_fetch.py` | Downloads only the needed CALVIN `task_D_D/training` frames from the official ZIP via HTTP Range requests (resumable). |
| `data/select_frames.py` | Picks 1,214 language-annotated windows (~64k frames), balanced over all 34 tasks. |
| `calvin_rlds/` | TFDS builder that turns the windows into an RLDS dataset (one episode per window), Octo's standardization transform, and the fine-tuning config. |
| `octo_patch/` + `setup_octo.sh` | Octo at commit `241fb35` with two small patches (below). |
| `finetune.sh` | Fine-tuning launcher; defaults reproduce the paper's run. |
| `evaluate/` | `eval_method.py` + `wrapper/octo_model_wrapper.py` (rollouts on the 1,087 test rows) and `run_eval_shard.sh`. |
| `requirements.txt` | Package versions of the environment we used. |

Chain building and chain-length aggregation are shared with GR-MG: use
`downstream/calvin/gr_mg/evaluate/{build_hf_calvin_mapping,build_chains,aggregate_chain_success}.py`.

## Checkpoint

The fine-tuned policy from the paper (step 30,000) is on Hugging Face at
[`RiddhiCh/SGE-Goal/octo-ft/calvin`](https://huggingface.co/RiddhiCh/SGE-Goal/tree/main/octo-ft/calvin):

```bash
huggingface-cli download RiddhiCh/SGE-Goal --include "octo-ft/calvin/*" --local-dir sge-goal-checkpoints
bash evaluate/run_eval_shard.sh <method> 0 217 sge-goal-checkpoints/octo-ft/calvin results
```

## Recipe used in the paper

| | |
|---|---|
| Initialization | Octo-Base (`hf://rail-berkeley/octo-base`), all parameters fine-tuned |
| Data | 1,214 windows (1,154 train / 60 val episodes), ~64k frames, CALVIN `task_D_D` training split (disjoint from the validation episodes the test rows come from) |
| Inputs | static camera 256², gripper camera 128²; 2-step observation window; 4-step action chunk |
| Actions | CALVIN's native `rel_actions`; gripper left unnormalised |
| Task conditioning | per sample, goal image **or** instruction (p = 0.5); goals relabeled uniformly from the future of the window |
| Goal augmentation | independent seed from the observation (`OCTO_DESYNC_TASK_AUG=1`) |
| Optimisation | AdamW (wd 0.01, clip 1.0), cosine LR peak 3e-4 with 1k warmup, batch 32, 30k steps (~14 epochs), ~4 h on one 48 GB GPU |
| Checkpoint | final step (30k), chosen by closed-loop success, not validation loss |

Two choices matter for using a policy to rank goal images:

1. **Goal-image or instruction, never both, during training.** With both available the policy learns to
   solve tasks from language. Evaluated with goal + instruction, Octo-ft reaches 50.1% with the
   ground-truth goal but the nine generators differ by only 4.0 points; goal-image-only they differ by 20.0.
   So evaluate with `OCTO_TASK_MODE=goal_only`.
2. **Separate augmentation for the goal image.** Octo's default augments observation and goal with the
   same random seed, so a policy can match them through shared crop and colour artifacts — a cue that
   does not exist when the goal comes from a separate generator.

Do not pick an early checkpoint to "avoid overfitting": under the cosine schedule the step-15k
checkpoint solved 0 of the first 40 test rows (ground-truth goal and instruction), the final one 30.

## Octo patches (`octo_patch/octo-241fb35-calvin.patch`)

- `octo/data/dataset.py`: when `OCTO_DESYNC_TASK_AUG=1`, the task (goal) image gets its own
  augmentation seed. Default behaviour is unchanged.
- `octo/utils/visualization_lib.py`: `tostring_rgb()` → `buffer_rgba()` for matplotlib ≥ 3.10.

## Reproducing

```bash
cd downstream/calvin/octo_ft
pip install -r requirements.txt          # Python 3.10, CUDA 12; see the jaxlib line
bash setup_octo.sh                       # clones Octo into ./octo and applies the patch
export CALVIN_ROOT=$PWD/calvin_data       # where frames and selection files go

# 1. training data (~19 GB, instead of downloading the whole task_D_D archive)
python data/calvin_train_fetch.py list
python data/calvin_train_fetch.py fetch 0          # metadata + language annotations only
python data/select_frames.py                      # writes calvin_train_selection.json
python data/calvin_train_fetch.py fetch_selected

# 2. RLDS dataset
CALVIN_RLDS_DIR=$PWD/rlds python calvin_rlds/build_rlds.py

# 3. fine-tune (paper settings are the defaults)
CUDA_VISIBLE_DEVICES=0 CALVIN_RLDS_DIR=$PWD/rlds SAVE_DIR=$PWD/checkpoints bash finetune.sh

# 4. evaluate one method on one shard (repeat for the five shards and every method)
export CALVIN_DATASET=/path/to/calvin/dataset/task_D_D   # needs validation/
export CALVIN_REPO=/path/to/calvin                         # clone of mees/calvin
export MAPPING_JSON=/path/to/hf_calvin_mapping.json        # gr_mg/evaluate/build_hf_calvin_mapping.py
export GOAL_DIR=/path/to/goal_images                       # <GOAL_DIR>/<method>/<one sub-folder>/<row+1>/edited_image.*
bash evaluate/run_eval_shard.sh flux-kontext-dev-ft 0 217 $PWD/checkpoints/<run> $PWD/results

# 5. chain-length success rates (same skeletons and aggregator as GR-MG)
python ../gr_mg/evaluate/aggregate_chain_success.py --chain_skeletons chain_skeletons.json \
  --results_dir results --methods flux-kontext-dev-ft ... --out_csv octo_ft.csv --out_json octo_ft.json
```

The paper used seed 0, a 360-step horizon and the shards `[0,217) [217,434) [434,652) [652,869) [869,1087)`.
For the ground-truth oracle pass `ground_truth` as the method and set `TEST_PARQUET` to the CALVIN test parquet.

`jaxlib 0.4.20` does not support NVIDIA Blackwell GPUs (sm_120); use an Ada/Ampere-class GPU.
Import JAX before PyTorch in the evaluation process (`eval_method.py` already does): loading PyTorch's
CUDA runtime first makes JAX fall back to the CPU.
