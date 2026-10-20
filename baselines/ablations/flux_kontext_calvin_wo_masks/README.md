# Ablation: `flux_kontext_calvin_bft_i1`

Finetuning of `black-forest-labs/FLUX.1-Kontext-dev` (LoRA) on the Calvin dataset
**without either edit-region mask** as a model input.

## What changed vs. the baseline
Baseline model input (concatenated along the sequence dim of the transformer
`hidden_states`):

```
[ noisy target (id 0) | original image (id 1) | initial mask (id 2) | final mask (id 3) ]
```

This ablation (`bft_i1`) feeds:

```
[ noisy target (id 0) | original image (id 1) ]
```

i.e. **both the initial and the final edit-region masks are removed from the set
of model inputs.** Everything else (original image, edit instruction) is fed
exactly as in the baseline.

Because the static-scene-consistency (SSC) auxiliary loss requires both masks to
define the "preserve" region, it is **removed**. The training (and validation)
loss is therefore the **rectified-flow loss only**.

The edit-region masks are *still loaded from the dataset* — but only so the
evaluation metrics (masked / unmasked PSNR, SSIM, LPIPS) and the validation
visualizations stay identical and comparable to the baseline. They never reach
the model.

All changes are marked with `# [ABLATION bft_i1]` comments in:
- `flux_kontext_lora_finetuning.py` (training loop + loss)
- `flux_kontext_eval_utils.py` (validation-loss + inference)

`pipeline_flux_kontext_multi_imgs.py` is unchanged (it is already None-safe for
absent masks; no masks are passed at inference).

## Run it
```bash
./run_bft_i1.sh
```
- Uses **Modal account #2** via per-process env-var auth — the shared machine's
  global modal profile is never touched.
- Runs **detached** under `nohup`: you can log off this server; the remote
  training continues, and the local driver keeps polling + downloading.
- Trains **21 epochs** (indices 0..20, ≥ 20 as required).

## Auto-resume after preemption
If Modal preempts/restarts the run with the same input, the remote training
function auto-detects the latest `checkpoint-epoch-N` on its volume and resumes
from the next epoch (state, optimizer, scheduler, RNG all restored via
`accelerator.load_state` + `training_state.json`). Just re-run `./run_bft_i1.sh`
if the local driver itself ever stops.

## Real-time data landing locally
The local driver polls the Modal volume and downloads as soon as each checkpoint
/ validation completes:
- `models/kontext-i2i/checkpoint-epoch-N/` — LoRA weights + optimizer/scheduler/RNG + `training_state.json`
- `models/kontext-i2i/validation_visualization/` — per-epoch viz
- `models/kontext-i2i/test_visualization/` — final test viz
- `flux_kontext_calvin_bft_i1.log` — full training log (loss curves, signals), streamed live

Follow progress: `tail -f flux_kontext_calvin_bft_i1.log`
