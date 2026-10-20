"""
Modal script to run validation visualization for ALL checkpoints in parallel.

Each checkpoint gets its own L40S GPU instance. Results are saved to the Modal volume
and then downloaded locally.

Usage:
    modal run modal_parallel_viz.py
"""

import modal
import os

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App("flux-kontext-parallel-viz")

VOLUME_NAME = "flux-kontext-dev-bridge-image-editing-volume"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# CUDA Configuration (match training script)
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision",
        "torchaudio",
    )
    .uv_pip_install(
        "accelerate",
        "diffusers",
        "transformers",
        "peft",
        "datasets",
        "pillow",
        "numpy",
        "scipy",
        "huggingface-hub",
        "hf_transfer",
        "bitsandbytes",
        "scikit-image",
        "lpips",
        "tqdm",
        "packaging",
        "wheel",
        "sentencepiece",
    )
    .run_commands("pip install flash-attn --no-build-isolation")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/data/datasets/.cache",
        }
    )
    .add_local_dir(
        ".",
        remote_path="/root/flux_finetuning",
        ignore=[
            "*.git*",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".env",
            ".venv",
            "*.log",
            ".DS_Store",
            "*.egg-info",
            ".pytest_cache",
            "models",
            "models/**",
            "*.safetensors",
            "*.bin",
            "*.pkl",
        ],
    )
)

huggingface_secret = modal.Secret.from_name("huggingface-secret")

# Dataset and model config (match training)
DATASET_NAME = "RiddhiCh/LGRM_BridgeDataV2_Teleoperated"
BASE_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
CACHE_DIR = "/data/datasets/.cache"
VOLUME_CHECKPOINT_DIR = "/data/models/kontext-i2i"
VIZ_OUTPUT_BASE = "/data/models/kontext-i2i"  # viz folders go here
NUM_VIZ_SAMPLES = 10
NUM_INFERENCE_STEPS = 28  # Full quality


@app.function(
    image=image,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    gpu="L40S",
    timeout=3600,
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0, initial_delay=10.0),
)
def run_viz_for_epoch(epoch: int) -> dict:
    """
    Run validation visualization for a single checkpoint epoch.
    
    Loads base model + LoRA weights, runs inference on the first NUM_VIZ_SAMPLES
    validation samples, saves results to volume.
    
    Returns dict with epoch number and status.
    """
    import sys
    sys.path.insert(0, "/root/flux_finetuning")

    import torch
    import numpy as np
    from PIL import Image
    from datasets import load_dataset
    from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline
    from flux_kontext_eval_utils import build_enhanced_prompt

    print(f"[Epoch {epoch}] Starting visualization...")
    
    # Check checkpoint exists on volume
    ckpt_dir = os.path.join(VOLUME_CHECKPOINT_DIR, f"checkpoint-epoch-{epoch}")
    lora_path = os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors")
    if not os.path.exists(lora_path):
        print(f"[Epoch {epoch}] ERROR: LoRA weights not found at {lora_path}")
        return {"epoch": epoch, "status": "error", "message": "LoRA weights not found"}

    # Output directory for this epoch's visualization
    viz_dir = os.path.join(VIZ_OUTPUT_BASE, f"validation_visualization_epoch_{epoch}")
    if os.path.exists(viz_dir):
        import shutil
        shutil.rmtree(viz_dir)
    os.makedirs(viz_dir, exist_ok=True)

    # Load dataset
    print(f"[Epoch {epoch}] Loading validation dataset...")
    dataset = load_dataset(DATASET_NAME, cache_dir=CACHE_DIR)
    val_dataset = dataset["validation"]
    total_samples = min(NUM_VIZ_SAMPLES, len(val_dataset))
    print(f"[Epoch {epoch}] Will process {total_samples} validation samples")

    # Load pipeline
    print(f"[Epoch {epoch}] Loading base model...")
    weight_dtype = torch.bfloat16
    pipeline = FluxKontextPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=weight_dtype,
    )

    # Load LoRA weights for this checkpoint
    print(f"[Epoch {epoch}] Loading LoRA weights from {ckpt_dir}...")
    pipeline.load_lora_weights(ckpt_dir)

    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    # Seed generator for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(0)

    def _maybe_load_image(x):
        if x is None:
            return None
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        if isinstance(x, np.ndarray):
            return Image.fromarray(x.astype("uint8")).convert("RGB")
        if hasattr(x, "convert"):
            return x.convert("RGB")
        return x

    # Run inference on each sample
    print(f"[Epoch {epoch}] Running inference ({NUM_INFERENCE_STEPS} steps per sample)...")
    for i in range(total_samples):
        sample = val_dataset[i]

        edit_instruction = sample.get("edit_instruction", "")
        prompt = build_enhanced_prompt(edit_instruction) if edit_instruction else ""

        orig_image = _maybe_load_image(sample.get("original_image"))
        initial_edit_region_mask = _maybe_load_image(sample.get("initial_edit_region_mask"))
        final_edit_region_mask = _maybe_load_image(sample.get("final_edit_region_mask"))
        gt_image = _maybe_load_image(sample.get("edited_image"))

        # Pipeline call args
        call_args = {
            "original_image": orig_image,
            "initial_edit_region_mask": initial_edit_region_mask,
            "final_edit_region_mask": final_edit_region_mask,
            "num_inference_steps": NUM_INFERENCE_STEPS,
        }

        # Match ground truth dimensions
        if gt_image is not None:
            gt_w, gt_h = gt_image.size
            call_args["height"] = gt_h
            call_args["width"] = gt_w
            call_args["_auto_resize"] = False
            call_args["max_area"] = gt_h * gt_w

        # Encode prompt
        with torch.no_grad():
            try:
                prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                    prompt, prompt_2=None
                )
                prompt_embeds = prompt_embeds.to("cuda")
                pooled_prompt_embeds = pooled_prompt_embeds.to("cuda")
            except Exception:
                prompt_embeds = None
                pooled_prompt_embeds = None

        # Run inference
        with torch.autocast("cuda"):
            try:
                if prompt_embeds is not None:
                    out = pipeline(
                        **call_args,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        generator=generator,
                    )
                else:
                    out = pipeline(**call_args, prompt=prompt, generator=generator)
            except Exception as e:
                print(f"[Epoch {epoch}] Pipeline error on sample {i}: {e}")
                continue

        # Extract predicted image
        pred_img = None
        if hasattr(out, "images"):
            pred_img = out.images[0]
        elif isinstance(out, (list, tuple)):
            candidate = out[0]
            pred_img = candidate[0] if isinstance(candidate, (list, tuple)) else candidate
        else:
            pred_img = out

        if not isinstance(pred_img, Image.Image):
            pred_img = _maybe_load_image(pred_img)

        # Resize prediction to ground truth size if different
        if pred_img is not None and gt_image is not None:
            try:
                gt_w, gt_h = gt_image.size
                pred_w, pred_h = pred_img.size
                if (pred_w, pred_h) != (gt_w, gt_h):
                    pred_img = pred_img.resize((gt_w, gt_h), resample=Image.BILINEAR)
            except Exception:
                pass

        # Save everything
        sample_dir = os.path.join(viz_dir, str(i))
        os.makedirs(sample_dir, exist_ok=True)

        if orig_image is not None:
            orig_image.save(os.path.join(sample_dir, "original_image.png"))
        if final_edit_region_mask is not None:
            final_edit_region_mask.save(os.path.join(sample_dir, "final_edit_region_mask.png"))
        if initial_edit_region_mask is not None:
            initial_edit_region_mask.save(os.path.join(sample_dir, "initial_edit_region_mask.png"))
        if pred_img is not None:
            pred_img.save(os.path.join(sample_dir, "predicted_image.png"))
        if gt_image is not None:
            gt_image.save(os.path.join(sample_dir, "ground_truth_image.png"))
        with open(os.path.join(sample_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(str(prompt))
        with open(os.path.join(sample_dir, "edit_instruction.txt"), "w", encoding="utf-8") as f:
            f.write(str(edit_instruction) if edit_instruction else "")

        print(f"[Epoch {epoch}] Sample {i}/{total_samples} done")

    # Commit volume to persist results
    volume.commit()
    print(f"[Epoch {epoch}] All {total_samples} samples done. Volume committed.")
    return {"epoch": epoch, "status": "success", "samples": total_samples}


@app.local_entrypoint()
def main():
    """Launch visualization for all checkpoints in parallel."""
    # All 31 checkpoints (epoch 0-30)
    epochs = list(range(31))
    print(f"Launching parallel visualization for {len(epochs)} checkpoints on L40S GPUs...")
    print(f"Epochs: {epochs}")

    # Use starmap to run all in parallel
    results = list(run_viz_for_epoch.map(epochs))

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    successes = 0
    failures = 0
    for r in sorted(results, key=lambda x: x["epoch"]):
        status = r["status"]
        ep = r["epoch"]
        if status == "success":
            print(f"  Epoch {ep:>2}: OK ({r['samples']} samples)")
            successes += 1
        else:
            print(f"  Epoch {ep:>2}: FAILED - {r.get('message', 'unknown')}")
            failures += 1

    print(f"\nTotal: {successes} succeeded, {failures} failed out of {len(epochs)}")
    print("\nVisualization folders saved on volume at:")
    print(f"  /data/models/kontext-i2i/validation_visualization_epoch_{{0..30}}/")
    print("\nTo download, run the download script.")
