#!/usr/bin/env python
"""Parallel validation visualization across all checkpoint epochs.

Runs one L40S GPU per checkpoint on Modal, producing a separate
validation_visualization_epoch_<N> folder for each checkpoint so results
can be compared visually.

LoRA loading mirrors the training code exactly:
  1. Load Flux2Transformer2DModel from pretrained
  2. transformer.add_adapter(LoraConfig(...))   — same config as training
  3. set_peft_model_state_dict(transformer, safetensors.load(lora_path))
  4. Construct Flux2KleinPipeline(scheduler, vae, text_encoder, tokenizer, transformer)
  5. Run inference

Usage:
    export MODAL_TOKEN_ID=<token-id> MODAL_TOKEN_SECRET=<token-secret>   # or use `modal token new`
    modal run modal_parallel_val_viz.py
"""

import modal
import os
import re

# ── Local paths ──────────────────────────────────────────────────────
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODELS_DIR = os.path.join(LOCAL_DIR, "models", "flux-2-klein-erm")

# ── Modal setup ──────────────────────────────────────────────────────
app = modal.App("flux2-klein-erm-parallel-val-viz")

VOLUME_NAME = "flux2-klein-erm-val-viz-volume"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

huggingface_secret = modal.Secret.from_name("huggingface-secret")

# ── Config (matching training exactly) ───────────────────────────────
MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"
DATASET_ID = "RiddhiCh/LGRM_BridgeDataV2_Teleoperated"
GUIDANCE_SCALE = 4.0
NUM_INFERENCE_STEPS = 50
EVAL_VIZ_NUM = 10
SEED = 0

# LoRA config — must match training exactly
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0", "proj_mlp"]

# ── ERM prompt (must match training) ────────────────────────────────
DEFAULT_ERM_INSTRUCTION = """TASK: Edit Region Mask Prediction for Robotic Manipulation

You are provided with the following inputs:
1. ORIGINAL IMAGE: A scene containing objects, including a robotic arm and dynamic objects that can be manipulated.
2. INITIAL EDIT REGION MASK: A binary mask (white=1, black=0) showing the current locations of dynamic objects in the original image. White pixels indicate regions occupied by objects that will be manipulated/moved.
3. EDIT INSTRUCTION (above): A text description of the manipulation/edit to perform on the dynamic objects.

YOUR TASK: Generate a FINAL EDIT REGION MASK - a binary mask (white=1, black=0) predicting where the dynamic objects should appear AFTER the edit instruction is executed.
- White pixels (1): Regions where dynamic objects will be located after manipulation
- Black pixels (0): Background regions that remain unchanged

IMPORTANT:
- Focus ONLY on predicting object locations, not colors or textures
- The mask should be consistent across all RGB channels
- Use the initial mask to understand which objects are dynamic
- Use the edit instruction to determine the target locations
- Output should be a clean binary mask showing final object positions"""


def build_erm_prompt(edit_instruction: str) -> str:
    return f"{edit_instruction}\n\n{DEFAULT_ERM_INSTRUCTION.strip()}"


# ── Image definition ─────────────────────────────────────────────────
CUDA_TAG = "12.8.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .uv_pip_install("torch==2.8.0", "torchvision", "torchaudio")
    .uv_pip_install(
        "accelerate",
        "transformers",
        "peft",
        "datasets",
        "pillow",
        "numpy",
        "scipy",
        "huggingface-hub",
        "hf_transfer",
        "safetensors",
        "tqdm",
        "packaging",
        "wheel",
        "sentencepiece",
    )
    .run_commands("apt-get update && apt-get install -y git")
    .run_commands("pip install --upgrade pip setuptools wheel")
    .run_commands("pip install git+https://github.com/huggingface/diffusers.git")
    .run_commands("pip install flash-attn --no-build-isolation")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/data/.cache"})
)


# ── Step 1: Pre-cache model + dataset on the volume ─────────────────
@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[huggingface_secret],
    timeout=7200,
    memory=65536,
)
def prepare_model_and_dataset():
    """Download base model and validation dataset into the volume cache."""
    import os

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    from huggingface_hub import snapshot_download

    print(f"Downloading base model: {MODEL_ID}")
    snapshot_download(MODEL_ID)

    from datasets import load_dataset

    print(f"Downloading validation dataset: {DATASET_ID}")
    load_dataset(DATASET_ID, split="validation")

    volume.commit()
    print("Model and dataset cached on volume.")


# ── Step 2: Run viz for one epoch ────────────────────────────────────
@app.function(
    image=image,
    gpu="L40S",
    volumes={"/data": volume},
    secrets=[huggingface_secret],
    timeout=3600,
    memory=65536,
    max_containers=16,
)
def run_viz_for_epoch(epoch: int) -> dict:
    """Generate validation visualization for a single checkpoint epoch.

    Mirrors the training code's evaluate_and_save_metrics() exactly:
    load components individually → add LoRA adapter → load LoRA state dict
    via set_peft_model_state_dict → construct pipeline → run inference.
    """
    import copy
    import os
    import importlib
    import traceback

    import torch
    import diffusers
    from PIL import Image
    from datasets import load_dataset
    from peft import LoraConfig, set_peft_model_state_dict
    from safetensors.torch import load_file as st_load

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    try:
        # ── Resolve diffusers classes (mirrors training imports) ──
        def _get(name):
            obj = getattr(diffusers, name, None)
            if obj is None:
                for mod in ("diffusers.pipelines", "diffusers.models",
                            "diffusers.schedulers"):
                    try:
                        obj = getattr(importlib.import_module(mod), name)
                        break
                    except (ImportError, AttributeError):
                        continue
            return obj

        Flux2KleinPipeline = _get("Flux2KleinPipeline")
        Flux2Transformer2DModel = _get("Flux2Transformer2DModel")
        AutoencoderKLFlux2 = _get("AutoencoderKLFlux2")
        FlowMatchEulerDiscreteScheduler = _get("FlowMatchEulerDiscreteScheduler")

        from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

        # ── Check LoRA exists ──
        lora_dir = f"/data/lora/checkpoint-epoch-{epoch}"
        lora_file = os.path.join(lora_dir, "transformer_lora.safetensors")
        if not os.path.exists(lora_file):
            return {
                "epoch": epoch,
                "status": "error",
                "message": f"LoRA not found: {lora_file}",
            }

        # ── Load components individually (mirrors training code) ──
        print(f"[Epoch {epoch}] Loading model components...")
        weight_dtype = torch.bfloat16

        tokenizer = Qwen2TokenizerFast.from_pretrained(
            MODEL_ID, subfolder="tokenizer"
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            MODEL_ID, subfolder="text_encoder", torch_dtype=weight_dtype
        )
        vae = AutoencoderKLFlux2.from_pretrained(
            MODEL_ID, subfolder="vae", torch_dtype=weight_dtype
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            MODEL_ID, subfolder="scheduler"
        )
        transformer = Flux2Transformer2DModel.from_pretrained(
            MODEL_ID, subfolder="transformer", torch_dtype=weight_dtype
        )

        # ── Add LoRA adapter (same config as training) ──
        transformer.requires_grad_(False)
        transformer_lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            init_lora_weights="gaussian",
            target_modules=LORA_TARGET_MODULES,
        )
        transformer.add_adapter(transformer_lora_config)

        # ── Load LoRA weights (same as training's load_model_hook) ──
        print(f"[Epoch {epoch}] Loading LoRA weights from {lora_file}...")
        lora_state = st_load(lora_file, device="cpu")
        incompatible = set_peft_model_state_dict(
            transformer, lora_state, adapter_name="default"
        )
        if incompatible is not None:
            unexpected = getattr(incompatible, "unexpected_keys", None)
            if unexpected:
                print(f"[Epoch {epoch}] WARNING: Unexpected LoRA keys: {unexpected}")

        # ── Construct pipeline (mirrors training's validation code) ──
        inference_scheduler = copy.deepcopy(scheduler)
        pipeline = Flux2KleinPipeline(
            scheduler=inference_scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        pipeline = pipeline.to("cuda")
        pipeline.set_progress_bar_config(disable=True)

        # ── Load dataset ──
        print(f"[Epoch {epoch}] Loading validation dataset...")
        dataset = load_dataset(DATASET_ID, split="validation")

        # ── Generate visualizations ──
        viz_dir = f"/data/viz/validation_visualization_epoch_{epoch}"
        os.makedirs(viz_dir, exist_ok=True)

        n = min(EVAL_VIZ_NUM, len(dataset))
        for i in range(n):
            example = dataset[i]
            original = example["original_image"].convert("RGB")
            init_mask = example["initial_edit_region_mask"].convert("RGB")
            final_mask = example["final_edit_region_mask"].convert("RGB")
            prompt = build_erm_prompt(example["edit_instruction"])

            orig_w, orig_h = original.size
            gen_h = (orig_h // 16) * 16
            gen_w = (orig_w // 16) * 16

            original_r = original.resize((gen_w, gen_h), resample=Image.BILINEAR)
            init_mask_r = init_mask.resize((gen_w, gen_h), resample=Image.NEAREST)
            final_mask_r = final_mask.resize(
                (gen_w, gen_h), resample=Image.NEAREST
            )

            blank = Image.new("RGB", (gen_w, gen_h), color=(0, 0, 0))
            cond_images = [original_r, init_mask_r, blank]

            gen = torch.Generator(device="cuda").manual_seed(SEED + i)

            with torch.autocast(device_type="cuda", dtype=weight_dtype):
                pred = pipeline(
                    prompt=prompt,
                    image=cond_images,
                    height=gen_h,
                    width=gen_w,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    generator=gen,
                    guidance_scale=GUIDANCE_SCALE,
                ).images[0]

            pred.save(os.path.join(viz_dir, f"{i}_pred.png"))
            final_mask_r.save(os.path.join(viz_dir, f"{i}_gt.png"))
            original_r.save(os.path.join(viz_dir, f"{i}_orig.png"))
            init_mask_r.save(os.path.join(viz_dir, f"{i}_init_mask.png"))
            with open(os.path.join(viz_dir, f"{i}_prompt.txt"), "w") as pf:
                pf.write(prompt)

            print(f"[Epoch {epoch}] Sample {i + 1}/{n} done.")

        volume.commit()
        print(f"[Epoch {epoch}] All {n} visualizations saved.")
        return {"epoch": epoch, "status": "success", "samples": n}

    except Exception as e:
        traceback.print_exc()
        return {"epoch": epoch, "status": "error", "message": str(e)}


# ── Local entrypoint ─────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    import shutil
    import subprocess
    from pathlib import Path

    modal_bin = shutil.which("modal") or "modal"

    # ── 1. Discover local checkpoints ──
    ckpt_dirs = sorted(
        [
            d
            for d in os.listdir(LOCAL_MODELS_DIR)
            if d.startswith("checkpoint-epoch-")
            and os.path.isdir(os.path.join(LOCAL_MODELS_DIR, d))
        ],
        key=lambda d: int(re.search(r"(\d+)$", d).group(1)),
    )
    epochs = [int(re.search(r"(\d+)$", d).group(1)) for d in ckpt_dirs]
    first, last = min(epochs), max(epochs)
    print(f"Found {len(epochs)} local checkpoints (epochs {first}-{last})")

    # ── 2. Upload LoRA weights to volume ──
    print("\nUploading LoRA weights to volume...")
    uploaded = 0
    with volume.batch_upload(force=True) as batch:
        for d in ckpt_dirs:
            local_lora = os.path.join(
                LOCAL_MODELS_DIR, d, "transformer_lora.safetensors"
            )
            if os.path.exists(local_lora):
                remote = f"lora/{d}/transformer_lora.safetensors"
                batch.put_file(Path(local_lora), remote)
                size_mb = os.path.getsize(local_lora) / (1024 * 1024)
                print(f"  {d} ({size_mb:.1f} MB)")
                uploaded += 1
            else:
                print(f"  WARNING: {d}/transformer_lora.safetensors not found")
    print(f"Upload complete: {uploaded}/{len(ckpt_dirs)} checkpoints.")

    # ── 3. Pre-cache model & dataset on volume ──
    print("\nPre-caching base model and dataset on volume (first-time only)...")
    prepare_model_and_dataset.remote()
    print("Preparation complete.")

    # ── 4. Parallel visualization ──
    print(f"\nLaunching {len(epochs)} parallel viz jobs on L40S GPUs...")
    results = list(run_viz_for_epoch.map(epochs))

    succeeded = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    print(f"\n{'='*60}")
    print(f"Results: {len(succeeded)} succeeded, {len(failed)} failed")
    print(f"{'='*60}")
    for r in sorted(succeeded, key=lambda x: x["epoch"]):
        print(f"  Epoch {r['epoch']:>3}: OK ({r['samples']} samples)")
    for r in sorted(failed, key=lambda x: x["epoch"]):
        print(f"  Epoch {r['epoch']:>3}: FAILED - {r.get('message', '?')}")

    # ── 5. Download results from volume ──
    if not succeeded:
        print("\nNo successful results to download.")
        return

    print(f"\nDownloading {len(succeeded)} visualization folders...")
    downloaded = 0
    for epoch in sorted(r["epoch"] for r in succeeded):
        remote_dir = f"viz/validation_visualization_epoch_{epoch}"
        result = subprocess.run(
            [modal_bin, "volume", "get", VOLUME_NAME, remote_dir, LOCAL_MODELS_DIR],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            downloaded += 1
            print(f"  Epoch {epoch}: downloaded")
        else:
            print(f"  Epoch {epoch}: DOWNLOAD FAILED - {result.stderr.strip()}")

    print(f"\nDone! {downloaded} viz folders saved to {LOCAL_MODELS_DIR}/")
