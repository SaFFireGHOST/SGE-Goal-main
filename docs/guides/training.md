# Training

SGE-Goal fine-tunes two LoRA adapters on a frozen base editor:

- **Step 2 — final mask** `M_f` from `(I_o, T, M_i)`: rectified-flow loss + channel-consistency
  loss (λ = 25).
- **Step 3 — goal image** `Î_e` from `(I_o, T, M_i, M_f)`: rectified-flow loss +
  static-scene-consistency loss (λ = 7).

Both auxiliary losses apply only for noise levels σ < 0.35. Step 1 is training-free.

## Where the training code is

| Stage | Base editor | Dataset | Directory |
|---|---|---|---|
| Step 2 | FLUX.1 Kontext [dev] | BridgeDataV2 / CALVIN / LIBERO | `sge_goal/mask_predictor/train/flux_kontext_{bridge,calvin,libero}/` |
| Step 2 | FLUX.2 klein | BridgeDataV2 / CALVIN / LIBERO | `sge_goal/mask_predictor/train/flux2_klein_{bridge,calvin,libero}/` |
| Step 3 | FLUX.1 Kontext [dev] | BridgeDataV2 / CALVIN / LIBERO | `sge_goal/synthesis/train/flux_kontext_{bridge,calvin,libero}/` |
| Step 3 | FLUX.2 klein | BridgeDataV2 / CALVIN / LIBERO | `sge_goal/synthesis/train/flux2_klein_{bridge,calvin,libero}/` |

Each directory is self-contained: a Modal launcher (`modal_*_train.py`) and the trainer it runs
(`flux_kontext_lora_finetuning.py`, `flux_2_klein_lora_finetuning.py` or
`flux_2_klein_erm_finetuning.py`). Run every command from inside the directory.

## Run on Modal

```bash
cd sge_goal/synthesis/train/flux2_klein_bridge
modal run modal_flux_2_klein_train.py                       # train (the default action)
modal run modal_flux_2_klein_train.py --action download     # copy checkpoints back from the volume
```

Checkpoints are written to a Modal volume once per epoch. The FLUX.2 klein launchers resume
automatically from the latest checkpoint on the volume when restarted (for example after
preemption); the FLUX.1 Kontext launchers do this with `--action resume`. Every launcher also has
`--action inference` for a quick test; the full list of actions is in the docstring of its `main()`.

The run configuration lives in the launcher's `TRAINING_CONFIG` dictionary: dataset, resolution,
batch size, epochs, learning rate, LoRA rank (and, for FLUX.2 klein, the loss weights). The FLUX.1 Kontext LIBERO Step 2 and
Step 3 runs were launched locally instead (see `monitor_training.sh` and
`benchmark_batch_sizes.py` in the `*_libero` directories, with `pip install -r sge_goal/requirements.txt`).

## Configuration

| | |
|---|---|
| LoRA rank / α / dropout | 16 / 16 / 0 |
| Optimizer | 8-bit AdamW |
| Learning rate | 1 × 10⁻⁴, constant, 200 warm-up steps |
| Batch size / gradient accumulation | 160 / 1 |
| Epochs | up to 75, one checkpoint per epoch |
| Precision | bf16 |
| Max. text tokens | 512 |
| Resolution | 256 (BridgeDataV2, CALVIN) / 128 (LIBERO) |
| Seed | 0 |
| Guidance scale, Step 2 | 4.0 (FLUX.2 klein) / 1.0 (FLUX.1 Kontext) |
| Guidance scale, Step 3 | 1.0 |

The released adapters are the checkpoints from these epochs:

| Stage | Base editor | BridgeDataV2 | CALVIN-ABCD | LIBERO-10/90 |
|---|---|---|---|---|
| Step 2 | FLUX.1 Kontext [dev] | 38 | 22 | 56 |
| Step 2 | FLUX.2 klein | 17 | 22 | 59 |
| Step 3 | FLUX.1 Kontext [dev] | 18 | 18 | 64 |
| Step 3 | FLUX.2 klein | 9 | 15 | 71 |

## Mask ablations

`baselines/ablations/` holds the Step 3 variants trained without the final mask
(`*_wo_final_mask`) or without either mask (`*_wo_masks`) on BridgeDataV2 and CALVIN, for both base
editors. Each directory has the same layout as above plus a `README.md` describing what changed.
