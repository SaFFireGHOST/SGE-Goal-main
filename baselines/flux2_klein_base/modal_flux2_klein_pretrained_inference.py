"""
Modal script for running inference with the original pre-trained Flux 2 Klein
pipeline on test data (no finetuning / LoRA — just the base model).

Each test case subdirectory must contain:
    - original_image.jpg   (the image to edit)
    - edit_instruction.txt (the natural-language edit instruction)

The script produces:
    - edited_image.jpg     (the pipeline prediction) in each test case dir

Usage:
    modal run modal_flux2_klein_pretrained_inference.py \
        --data-dir dummy_test_data
"""

import modal
import os

# ============================================================================
# GPU Configuration
# ============================================================================
GPU_TYPE = "L40S"
GPU_CONFIG = f"{GPU_TYPE}:1"

# ============================================================================
# Volume Configuration
# ============================================================================
VOLUME_NAME = "flux2-klein-pretrained-calvin-inference-volume"
VOLUME_MOUNT = "/data"
REMOTE_TEST_DATA_DIR = "/data/test_data"

# ============================================================================
# Model Configuration
# ============================================================================
BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 1.0
# Goal-gen seed: overridable per run via SGE_GEN_SEED.
# Default 0 reproduces the original single-seed table. Read at use-site from the
# container env, which is baked from the launch env via the image .env() below.
SEED = int(os.environ.get("SGE_GEN_SEED", "0"))

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
app = modal.App("flux2-klein-pretrained-calvin-inference")

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
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/data/.cache",
            # Bake the launch-time goal-gen seed into the container env (final
            # layer -> cheap rebuild per seed). Read back as SGE_GEN_SEED above.
            "SGE_GEN_SEED": os.environ.get("SGE_GEN_SEED", "0"),
        }
    )
)

# HuggingFace secret for gated model access
huggingface_secret = modal.Secret.from_name("huggingface-secret")


# ============================================================================
# Enhanced Prompt Builder (for pre-trained pipeline — no mask references)
# ============================================================================
def build_enhanced_prompt(edit_instruction: str) -> str:
    """
    Build an enhanced prompt for the pre-trained Flux 2 Klein pipeline.

    The raw edit instruction is placed at the very beginning of the prompt
    (the first characters) to minimise the chance of it being truncated by
    token-length limits.  The remaining text provides auxiliary context that
    steers the model towards faithful image editing behaviour.
    """
    enhanced_prompt = (
        f"{edit_instruction}\n"
        "\n"
        "You are given the original image above. Edit it according to the "
        "instruction provided at the top of this prompt. Apply only the "
        "changes that are explicitly described — preserve every other element "
        "of the scene (background, lighting, camera angle, object identities, "
        "and overall composition) exactly as they appear in the original image. "
        "The result must be a photorealistic image that looks like a natural "
        "photograph, with consistent lighting, shadows, perspective, and "
        "visual style relative to the original."
    )
    return enhanced_prompt


# ============================================================================
# Remote Inference Function
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={VOLUME_MOUNT: volume},
    timeout=21600,  # 6 hours
    memory=65536,   # 64 GB RAM
)
def run_pretrained_inference():
    """
    Run inference with the original pre-trained Flux 2 Klein pipeline on all
    test cases found in the uploaded test data folder.

    Expects the volume to already contain:
        /data/test_data/<case_id>/original_image.jpg
        /data/test_data/<case_id>/edit_instruction.txt

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

    # ---- Validate volume contents ----
    if not test_data_dir.exists():
        raise FileNotFoundError(
            f"Test data directory not found at {test_data_dir}. "
            "Ensure the local entrypoint uploaded it correctly."
        )

    # ---- Load pipeline (pre-trained, no LoRA) ----
    print("=" * 70)
    print("Loading pre-trained Flux 2 Klein pipeline...")
    print(f"  Base model: {BASE_MODEL_ID}")
    print("  Mode: pre-trained (no finetuning / LoRA)")
    print("=" * 70)

    diffusers_mod = importlib.import_module("diffusers")
    Flux2KleinPipeline = getattr(diffusers_mod, "Flux2KleinPipeline")

    pipeline = Flux2KleinPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    pipeline.set_progress_bar_config(disable=False)
    print("Pipeline loaded successfully.\n")

    # ---- Discover test cases ----
    test_case_dirs = sorted(
        [d for d in test_data_dir.iterdir() if d.is_dir()],
        key=lambda x: int(x.name) if x.name.isdigit() else x.name,
    )
    print(f"Found {len(test_case_dirs)} test case(s) in {test_data_dir}\n")

    processed = 0
    skipped = 0

    for case_dir in test_case_dirs:
        case_name = case_dir.name

        # Required files
        edit_instruction_path = case_dir / "edit_instruction.txt"
        output_path = case_dir / "edited_image.jpg"

        # Find original_image with any supported extension
        original_image_path = None
        for ext in ("jpg", "jpeg", "png"):
            candidate = case_dir / f"original_image.{ext}"
            if candidate.exists():
                original_image_path = candidate
                break

        # Check all required files exist
        missing = []
        if original_image_path is None:
            missing.append("original_image.{jpg,jpeg,png}")
        if not edit_instruction_path.exists():
            missing.append("edit_instruction.txt")

        if missing:
            print(f"[SKIP] Test case {case_name}: missing {', '.join(missing)}")
            skipped += 1
            continue

        print(f"[{processed + 1}] Processing test case {case_name}...")

        # Load inputs
        original = Image.open(original_image_path).convert("RGB")
        edit_instruction = edit_instruction_path.read_text().strip()

        print(f"    Edit instruction: {edit_instruction[:120]}{'...' if len(edit_instruction) > 120 else ''}")
        print(f"    Image size: {original.size}")

        # Build enhanced prompt
        enhanced_prompt = build_enhanced_prompt(edit_instruction)

        generator = torch.Generator(device="cuda").manual_seed(SEED)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = pipeline(
                prompt=enhanced_prompt,
                image=original,
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

    # ---- Commit volume so the local entrypoint can download results ----
    print(f"\n{'=' * 70}")
    print(f"Inference complete. Processed: {processed}, Skipped: {skipped}")
    print(f"{'=' * 70}")
    print("Committing volume...")
    volume.commit()

    return {
        "status": "success",
        "processed": processed,
        "skipped": skipped,
        "total": len(test_case_dirs),
    }


# ============================================================================
# Local Entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    data_dir: str = "dummy_test_data",
):
    """
    Run pre-trained Flux 2 Klein inference on a local test data folder.

    Steps:
        1. Upload test data folder to Modal volume
        2. Run inference on GPU (all test cases processed in one call)
        3. Download processed test data (with edited_image.jpg) back locally
        4. Delete the Modal volume (prevents stale data on future runs)

    Args:
        data_dir: Path to test data directory (each subdirectory is a test case
                  containing original_image.jpg and edit_instruction.txt)
    """
    import subprocess
    import sys
    import shutil
    import time

    data_dir_abs = os.path.abspath(data_dir)
    data_dir_parent = os.path.dirname(data_dir_abs)
    data_dir_basename = os.path.basename(data_dir_abs)

    # Validate local path
    if not os.path.isdir(data_dir_abs):
        raise FileNotFoundError(f"Test data directory not found: {data_dir_abs}")

    # Count test cases
    test_cases = [
        d for d in os.listdir(data_dir_abs)
        if os.path.isdir(os.path.join(data_dir_abs, d))
    ]
    print("=" * 70)
    print("Flux 2 Klein Pre-trained Image Editing Inference")
    print("=" * 70)
    print(f"  Test data dir:   {data_dir_abs}")
    print(f"  Test cases:      {len(test_cases)}")
    print(f"  Volume:          {VOLUME_NAME}")
    print(f"  GPU:             {GPU_CONFIG}")
    print(f"  Model:           {BASE_MODEL_ID}")
    print("=" * 70)

    modal_cmd_base = [sys.executable, "-m", "modal"]

    # ==================================================================
    # Step 1: Upload test data to Modal volume
    # ==================================================================
    print(f"\n[Step 1/4] Uploading test data to Modal volume ({VOLUME_NAME})...")
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
    # Step 2: Run inference on Modal GPU
    # ==================================================================
    print(f"\n[Step 2/4] Running pre-trained inference on Modal ({GPU_CONFIG})...")
    t0 = time.time()
    inference_result = run_pretrained_inference.remote()
    elapsed = time.time() - t0
    print(f"\n  Inference finished ({elapsed:.1f}s)")
    print(f"  Result: {inference_result}")

    if inference_result.get("status") != "success":
        raise RuntimeError(f"Inference failed: {inference_result}")

    # ==================================================================
    # Step 3: Download processed test data back locally
    # ==================================================================
    print(f"\n[Step 3/4] Downloading processed test data from Modal volume...")
    t0 = time.time()

    # Back up the local test data directory, then replace it
    backup_dir = data_dir_abs + ".backup"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.move(data_dir_abs, backup_dir)
    print(f"  Backed up local test data to {backup_dir}")

    try:
        result = subprocess.run(
            modal_cmd_base + [
                "volume", "get", "--force",
                VOLUME_NAME,
                "test_data",
                data_dir_parent,
            ],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  ERROR downloading (exit {result.returncode}, {elapsed:.1f}s)")
            if result.stderr:
                print(f"  stderr: {result.stderr.strip()}")
            # Restore backup
            if os.path.exists(backup_dir):
                shutil.move(backup_dir, data_dir_abs)
                print(f"  Restored backup to {data_dir_abs}")
            raise RuntimeError("Failed to download processed test data from Modal volume.")

        # modal volume get downloads to <parent>/test_data
        # Rename to the original directory name if different
        downloaded_path = os.path.join(data_dir_parent, "test_data")
        if os.path.exists(downloaded_path) and downloaded_path != data_dir_abs:
            if os.path.exists(data_dir_abs):
                shutil.rmtree(data_dir_abs)
            shutil.move(downloaded_path, data_dir_abs)
            print(f"  Renamed downloaded test_data -> {data_dir_basename}")
        elif not os.path.exists(data_dir_abs):
            # Fallback: if the download didn't create the expected directory
            if os.path.exists(backup_dir):
                shutil.move(backup_dir, data_dir_abs)
                print(f"  WARNING: Download didn't produce expected directory. Restored backup.")
                raise RuntimeError("Download produced unexpected directory structure.")

        print(f"  Download complete ({elapsed:.1f}s)")

        # Remove backup
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
            print(f"  Removed backup")

    except subprocess.TimeoutExpired:
        print("  ERROR: Download timed out (30 min)")
        if os.path.exists(backup_dir):
            shutil.move(backup_dir, data_dir_abs)
            print(f"  Restored backup to {data_dir_abs}")
        raise

    # ==================================================================
    # Step 4: Delete the Modal volume
    # ==================================================================
    print(f"\n[Step 4/4] Deleting Modal volume ({VOLUME_NAME})...")
    t0 = time.time()
    result = subprocess.run(
        modal_cmd_base + ["volume", "delete", "--yes", VOLUME_NAME],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        # Non-fatal: volume deletion failure shouldn't fail the whole run
        print(f"  WARNING: Volume deletion failed (exit {result.returncode}, {elapsed:.1f}s)")
        if result.stderr:
            print(f"  stderr: {result.stderr.strip()}")
        print("  You can manually delete it with:")
        print(f"    modal volume delete --yes {VOLUME_NAME}")
    else:
        print(f"  Volume deleted ({elapsed:.1f}s)")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")
    print(f"  Processed: {inference_result.get('processed', '?')} test cases")
    print(f"  Skipped:   {inference_result.get('skipped', '?')} test cases")
    print(f"  Output:    {data_dir_abs}")
    print(f"  Each test case directory now contains edited_image.jpg")
    print(f"{'=' * 70}")

    return inference_result
