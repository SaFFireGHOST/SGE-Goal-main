# Ablation i2 — `flux_kontext_calvin_bft_i2`

Mask-input ablation of the Flux 2 Klein LoRA finetune on the Calvin dataset.

## What this ablation changes

Baseline (full model) feeds the transformer **3 conditioning images**:
`[original_image, initial_edit_region_mask, final_edit_region_mask]`.

**i2 drops the final mask.** The model receives **2 conditioning images**:
`[original_image, initial_edit_region_mask]`. Everything else about the data
pipeline is unchanged.

Because the **Static Scene Consistency (SSC) loss** is only defined when *both*
masks are available (it uses `max(initial, final)` to mark the preserved
background), it is **removed** in this ablation. Training uses the **rectified
flow-matching loss only**.

| Variant | Conditioning inputs | Training loss |
|---|---|---|
| Baseline (full) | original + initial mask + final mask | flow + SSC |
| **i2 (this)** | original + initial mask | **flow only** |
| i1 | original only | flow only |

Exact code changes vs. the baseline are confined to
`flux_2_klein_lora_finetuning.py` (search for `ABLATION i2`):
`compute_batch_losses` (conditioning concat + SSC removal) and the validation/
inference `cond_images` lists.

Trained for **20 epochs**, **one checkpoint saved per epoch**.

## Modal account / resource names

- Modal account: **#1**
- App: `flux2-klein-calvin-bft-i2`
- Volume: `flux2-klein-calvin-bft-i2-volume`
- HF secret used: `huggingface-secret` (this is how account 1 names it)
- Remote output dir: `/data/models/flux-2-klein-lora-calvin-i2`
- Remote logs dir: `/data/logs/flux-2-klein-calvin-i2`

## How to run

```bash
./launch.sh      # starts detached training + the real-time sync poller
```

You can then log off the server. To watch progress:

```bash
tail -f flux_kontext_calvin_bft_i2_modalrun.log   # client view of remote console
tail -f flux_kontext_calvin_bft_i2.log            # synced training console log
tail -f flux_kontext_calvin_bft_i2_sync.log       # poller activity
ls models/flux-2-klein-lora-calvin-i2/            # checkpoints as they arrive
```

To stop the **local** processes (training keeps running on Modal because it is
detached):

```bash
./stop.sh
```

To stop the **remote** training run entirely:

```bash
  python -m modal app stop flux2-klein-calvin-bft-i2
```

## Local data layout (mirrors the baseline codebase)

The poller writes into:

```
models/flux-2-klein-lora-calvin-i2/
  checkpoint-epoch-1/  ... checkpoint-epoch-20/   (each: pytorch_lora_weights.safetensors,
                                                   optimizer.bin, random_states_0.pkl,
                                                   training_state.json)
  validation_visualization/
  test_visualization/
logs/flux-2-klein-calvin-i2/
  training_console.log
  <tensorboard event files>
flux_kontext_calvin_bft_i2.log                    (convenience copy of training_console.log)
```

## Robustness (preemption + logout)

- **Auto-resume**: on start (and on every Modal-initiated restart) the training
  container auto-detects the latest `checkpoint-epoch-N` on the volume and resumes
  from it.
- **Auto-restart on preemption**: the training function is decorated with
  `retries=`, so if Modal preempts or the run crashes it is restarted with the
  same input and resumes from the last checkpoint — no manual action needed.
- **Detached**: launched with `modal run --detach`, so the run survives this
  client/SSH session ending.
- **Real-time sync without the global profile**: `sync_from_modal.py` polls the
  volume every 30 s and pulls new checkpoints + logs here. Both training and the
  poller authenticate purely via `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` env vars
  set by `launch.sh`; the shared global Modal profile (`~/.modal.toml`) is never
  read or modified.

## Files

| File | Purpose |
|---|---|
| `flux_2_klein_lora_finetuning.py` | Training/eval script (i2-modified) |
| `modal_flux_2_klein_train.py` | Modal app: image, volume, training function, entrypoint |
| `sync_from_modal.py` | Standalone real-time volume→local poller |
| `launch.sh` | Start detached training + poller (account 1, env-var auth) |
| `stop.sh` | Stop the local processes |
