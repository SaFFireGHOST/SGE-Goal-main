"""
Modal script for running Flux 2 Klein LoRA image editing inference on test data.

This script:
1. Uploads the entire test data folder and checkpoint to a Modal volume
2. Runs image editing inference on all test cases on a GPU (as a generator)
3. Yields progress signals after every batch so the local entrypoint can
   download edited_image.jpg files in real-time
4. Saves edited_image.jpg in each test case subdirectory on the volume

Each test case subdirectory must contain:
    - original_image.jpg
    - edit_instruction.txt
    - initial_edit_region_mask.jpg  (binary mask)

Usage:
    modal run modal_flux2_klein_ie_inference.py \\
        --data-dir bridge_data_v2_tel_test_data \\
        --checkpoint-dir models/flux-2-klein-lora-bridge-v2/checkpoint-epoch-9
"""

import modal
import os

# ============================================================================
# GPU Configuration
# ============================================================================
GPU_TYPE = "B200"
GPU_CONFIG = f"{GPU_TYPE}:1"

# ============================================================================
# Volume Configuration
# ============================================================================
VOLUME_NAME = "flux2-klein-bft-i2-inference-volume"
VOLUME_MOUNT = "/data"
REMOTE_TEST_DATA_DIR = "/data/test_data"
REMOTE_CHECKPOINT_DIR = "/data/checkpoint"

# ============================================================================
# Model Configuration
# ============================================================================
BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 1.0
SEED = int(os.environ.get("SGE_GEN_SEED", "0"))

# How many test cases to process before yielding a progress signal
SYNC_BATCH_SIZE = 10

# ============================================================================
# CUDA / Image Configuration (mirrors training script)
# ============================================================================
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App("flux2-klein-bft-i2-inference")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    # Install PyTorch
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision",
        "torchaudio",
    )
    # Install main dependencies
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
        "bitsandbytes",
        "tqdm",
        "packaging",
        "wheel",
        "sentencepiece",
    )
    # Install diffusers from git for Flux 2 Klein support
    .run_commands("apt-get update && apt-get install -y git")
    .run_commands("pip install --upgrade pip setuptools wheel")
    .run_commands("pip install git+https://github.com/huggingface/diffusers.git")
    # Install flash-attn (needs torch present)
    .run_commands("pip install flash-attn --no-build-isolation")
    # Set environment variables
    .env(
        {
            "SGE_GEN_SEED": os.environ.get("SGE_GEN_SEED", "0"),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/data/.cache",
        }
    )
)

# HuggingFace secret for gated model access
huggingface_secret = modal.Secret.from_name("huggingface-secret")


# ============================================================================
# Enhanced Prompt Builder (extracted from flux_2_klein_lora_finetuning.py)
# ============================================================================
def build_enhanced_prompt(edit_instruction: str) -> str:
    """
    Build an enhanced prompt for the Flux Kontext model.
    Matches the prompt template used during training.
    (Ablation: no final edit region mask guidance)
    """
    enhanced_prompt = f"""{edit_instruction}

Task: Robotic Manipulation Image Editing

You are performing an image editing task in the domain of robotic manipulation. Your goal is to generate the final edited scene after a robotic arm performs the manipulation action described above.

Inputs Provided:
1. Original Image: The initial scene before any manipulation occurs. This is your reference for the scene layout, lighting, and static elements.
2. Initial Edit Region Mask: A binary mask highlighting the dynamic object(s) in the original image that will be manipulated by the robotic arm. White pixels (value 1) indicate dynamic objects; black pixels (value 0) indicate static background.

Expected Output:
Generate the final edited scene image where:
- The dynamic object(s) identified in the Initial Edit Region Mask have been moved/manipulated according to the edit instruction
- All static background elements from the original image should be preserved exactly
- The edited image should look photorealistic and physically plausible
- Maintain consistent lighting, shadows, and visual style with the original image"""

    return enhanced_prompt


# ============================================================================
# Remote Inference Function (generator — yields signals for real-time sync)
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={VOLUME_MOUNT: volume},
    timeout=86400,  # 24 hours
    memory=65536,   # 64 GB RAM
    is_generator=True,
)
def run_image_editing_inference():
    """
    Run image editing inference on all test cases in the uploaded test data folder.
    Yields progress signals after every SYNC_BATCH_SIZE cases so the local
    entrypoint can download results in real-time.

    Expects the volume to already contain:
        /data/test_data/<case_id>/original_image.jpg
        /data/test_data/<case_id>/edit_instruction.txt
        /data/test_data/<case_id>/initial_edit_region_mask.jpg
    And:
        /data/checkpoint/pytorch_lora_weights.safetensors

    Produces:
        /data/test_data/<case_id>/edited_image.jpg
    """
    import importlib
    import torch
    from PIL import Image
    from pathlib import Path

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    test_data_dir = Path(REMOTE_TEST_DATA_DIR)
    checkpoint_dir = Path(REMOTE_CHECKPOINT_DIR)

    # ---- Validate volume contents ----
    if not test_data_dir.exists():
        raise FileNotFoundError(
            f"Test data directory not found at {test_data_dir}. "
            "Ensure the local entrypoint uploaded it correctly."
        )
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found at {checkpoint_dir}. "
            "Ensure the local entrypoint uploaded it correctly."
        )

    lora_weights_path = checkpoint_dir / "pytorch_lora_weights.safetensors"
    if not lora_weights_path.exists():
        raise FileNotFoundError(
            f"LoRA weights not found at {lora_weights_path}. "
            "Ensure the checkpoint directory contains pytorch_lora_weights.safetensors."
        )

    # ---- Load pipeline ----
    print("=" * 70)
    print("Loading Flux 2 Klein pipeline...")
    print(f"  Base model: {BASE_MODEL_ID}")
    print(f"  LoRA checkpoint: {checkpoint_dir}")
    print("=" * 70)

    diffusers_mod = importlib.import_module("diffusers")
    Flux2KleinPipeline = getattr(diffusers_mod, "Flux2KleinPipeline")

    pipeline = Flux2KleinPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    print(f"Loading LoRA weights from {checkpoint_dir} ...")
    pipeline.load_lora_weights(str(checkpoint_dir))
    pipeline.set_progress_bar_config(disable=False)
    print("Pipeline loaded successfully.\n")

    # ---- Discover test cases ----
    test_case_dirs = sorted(
        [d for d in test_data_dir.iterdir() if d.is_dir()],
        key=lambda x: int(x.name) if x.name.isdigit() else x.name,
    )
    total_cases = len(test_case_dirs)
    print(f"Found {total_cases} test case(s) in {test_data_dir}\n")

    processed = 0
    skipped = 0
    batch_cases = []  # case names processed since last yield

    for case_dir in test_case_dirs:
        case_name = case_dir.name

        # Required files (ablation: no final_edit_region_mask)
        original_image_path = case_dir / "original_image.jpg"
        edit_instruction_path = case_dir / "edit_instruction.txt"
        init_mask_path = case_dir / "initial_edit_region_mask.jpg"
        output_path = case_dir / "edited_image.jpg"

        # Check all required files exist
        missing = []
        if not original_image_path.exists():
            missing.append("original_image.jpg")
        if not edit_instruction_path.exists():
            missing.append("edit_instruction.txt")
        if not init_mask_path.exists():
            missing.append("initial_edit_region_mask.jpg")

        if missing:
            print(f"[SKIP] Test case {case_name}: missing {', '.join(missing)}")
            skipped += 1
            continue

        # Skip if already processed (resume support)
        if output_path.exists():
            print(f"[SKIP] Test case {case_name}: edited_image.jpg already exists (resume)")
            skipped += 1
            continue

        print(f"[{processed + 1}/{total_cases}] Processing test case {case_name}...")

        # Load inputs (ablation: no final_mask)
        original = Image.open(original_image_path).convert("RGB")
        init_mask = Image.open(init_mask_path).convert("RGB")
        edit_instruction = edit_instruction_path.read_text().strip()

        print(f"    Edit instruction: {edit_instruction[:100]}{'...' if len(edit_instruction) > 100 else ''}")
        print(f"    Image size: {original.size}")

        # Build enhanced prompt (matches training-time prompt template)
        enhanced_prompt = build_enhanced_prompt(edit_instruction)

        # Prepare multimodal conditioning: [original, init_mask]
        cond_images = [original, init_mask]

        generator = torch.Generator(device="cuda").manual_seed(SEED)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = pipeline(
                prompt=enhanced_prompt,
                image=cond_images,
                height=original.height,
                width=original.width,
                num_inference_steps=NUM_INFERENCE_STEPS,
                generator=generator,
                guidance_scale=GUIDANCE_SCALE,
            )

        # Save edited image
        edited_image = result.images[0]
        edited_image.save(str(output_path), quality=95)
        print(f"    Saved: {output_path}")
        processed += 1
        batch_cases.append(case_name)

        # Yield a progress signal every SYNC_BATCH_SIZE cases
        if len(batch_cases) >= SYNC_BATCH_SIZE:
            print(f"    [Sync] Committing volume after {processed} cases …")
            volume.commit()
            yield {
                "signal": "batch_complete",
                "cases": list(batch_cases),
                "processed": processed,
                "skipped": skipped,
                "total": total_cases,
            }
            batch_cases = []

    # ---- Final commit & yield for any remaining cases ----
    if batch_cases:
        print(f"    [Sync] Final commit ({processed} cases total) …")
        volume.commit()
        yield {
            "signal": "batch_complete",
            "cases": list(batch_cases),
            "processed": processed,
            "skipped": skipped,
            "total": total_cases,
        }

    print(f"\n{'=' * 70}")
    print(f"Inference complete. Processed: {processed}, Skipped: {skipped}")
    print(f"{'=' * 70}")

    yield {
        "signal": "inference_complete",
        "processed": processed,
        "skipped": skipped,
        "total": total_cases,
    }


# ============================================================================
# Local Entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    data_dir: str = "bridge_data_v2_tel_test_data",
    checkpoint_dir: str = "models/flux-2-klein-lora-bridge-v2/checkpoint-epoch-9",
):
    """
    Run image editing inference on a local test data folder with real-time
    download of results.

    Steps:
        1. Upload test data folder to Modal volume
        2. Upload checkpoint to Modal volume
        3. Run inference on GPU (generator — yields after every batch)
        4. After each batch, download new edited_image.jpg files locally

    Args:
        data_dir:       Path to test data directory (each subdirectory is a test case)
        checkpoint_dir: Path to checkpoint directory containing pytorch_lora_weights.safetensors
    """
    import subprocess
    import sys
    import time

    data_dir_abs = os.path.abspath(data_dir)
    checkpoint_dir_abs = os.path.abspath(checkpoint_dir)

    # Validate local paths
    if not os.path.isdir(data_dir_abs):
        raise FileNotFoundError(f"Test data directory not found: {data_dir_abs}")
    if not os.path.isdir(checkpoint_dir_abs):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir_abs}")

    lora_file = os.path.join(checkpoint_dir_abs, "pytorch_lora_weights.safetensors")
    if not os.path.isfile(lora_file):
        raise FileNotFoundError(
            f"pytorch_lora_weights.safetensors not found in {checkpoint_dir_abs}"
        )

    # Count test cases
    test_cases = [
        d for d in os.listdir(data_dir_abs)
        if os.path.isdir(os.path.join(data_dir_abs, d))
    ]
    print("=" * 70)
    print("Flux 2 Klein Image Editing Inference (Real-Time Sync)")
    print("=" * 70)
    print(f"  Test data dir:   {data_dir_abs}")
    print(f"  Checkpoint dir:  {checkpoint_dir_abs}")
    print(f"  Test cases:      {len(test_cases)}")
    print(f"  Volume:          {VOLUME_NAME}")
    print(f"  GPU:             {GPU_CONFIG}")
    print(f"  Sync batch size: {SYNC_BATCH_SIZE}")
    print("=" * 70)

    modal_cmd_base = [sys.executable, "-m", "modal"]

    # ==================================================================
    # Step 1: Upload test data to Modal volume
    # ==================================================================
    print(f"\n[Step 1/3] Uploading test data to Modal volume ({VOLUME_NAME})...")
    t0 = time.time()
    result = subprocess.run(
        modal_cmd_base + ["volume", "put", "--force", VOLUME_NAME, data_dir_abs, "test_data"],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  ERROR uploading test data (exit {result.returncode}, {elapsed:.1f}s)")
        if result.stderr:
            print(f"  stderr: {result.stderr.strip()}")
        raise RuntimeError("Failed to upload test data to Modal volume.")
    print(f"  Upload complete ({elapsed:.1f}s)")

    # ==================================================================
    # Step 2: Upload checkpoint to Modal volume
    # ==================================================================
    print(f"\n[Step 2/3] Uploading checkpoint to Modal volume ({VOLUME_NAME})...")
    t0 = time.time()
    result = subprocess.run(
        modal_cmd_base + ["volume", "put", "--force", VOLUME_NAME, checkpoint_dir_abs, "checkpoint"],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  ERROR uploading checkpoint (exit {result.returncode}, {elapsed:.1f}s)")
        if result.stderr:
            print(f"  stderr: {result.stderr.strip()}")
        raise RuntimeError("Failed to upload checkpoint to Modal volume.")
    print(f"  Upload complete ({elapsed:.1f}s)")

    # ==================================================================
    # Step 3: Run inference with real-time sync
    # ==================================================================
    print(f"\n[Step 3/3] Running inference on Modal ({GPU_CONFIG}) with real-time sync...")
    t0 = time.time()
    download_count = 0
    total_downloaded_cases = 0

    for event in run_image_editing_inference.remote_gen():
        signal = event.get("signal", "unknown")

        if signal == "batch_complete":
            cases = event.get("cases", [])
            proc = event.get("processed", "?")
            total = event.get("total", "?")
            download_count += 1

            print(f"\n>>> [SYNC #{download_count}] Batch complete: {len(cases)} cases "
                  f"(total processed: {proc}/{total})")

            # Download each edited_image.jpg from the volume
            sync_ok = 0
            for case_name in cases:
                remote_path = f"test_data/{case_name}/edited_image.jpg"
                local_dest = os.path.join(data_dir_abs, case_name)
                dl_result = subprocess.run(
                    modal_cmd_base + [
                        "volume", "get",
                        VOLUME_NAME,
                        remote_path,
                        local_dest,
                    ],
                    capture_output=True,
                    text=True,
                )
                if dl_result.returncode == 0:
                    sync_ok += 1
                else:
                    print(f"    WARNING: Failed to download {remote_path}: {dl_result.stderr.strip()}")

            total_downloaded_cases += sync_ok
            print(f"    Downloaded {sync_ok}/{len(cases)} edited images "
                  f"(cumulative: {total_downloaded_cases})")

        elif signal == "inference_complete":
            proc = event.get("processed", "?")
            skipped = event.get("skipped", "?")
            total = event.get("total", "?")
            elapsed = time.time() - t0
            print(f"\n{'=' * 70}")
            print(f"Inference complete ({elapsed:.1f}s)")
            print(f"  Processed: {proc}")
            print(f"  Skipped:   {skipped}")
            print(f"  Total:     {total}")
            print(f"  Downloaded locally: {total_downloaded_cases}")
            print(f"{'=' * 70}")

        else:
            print(f"\n>>> [SYNC] Unknown signal: {signal} — {event}")

    elapsed_total = time.time() - t0
    print(f"\n>>> Done. Total time: {elapsed_total:.1f}s")
    print(f">>> Total sync downloads: {download_count}")
    print(f">>> Edited images saved to: {data_dir_abs}")

