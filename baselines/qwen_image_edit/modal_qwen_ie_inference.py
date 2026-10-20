"""
Modal script for running pretrained Qwen Image Edit inference on test data.

This script:
1. Uploads the entire test data folder to a Modal volume
2. Runs image editing inference using the pretrained Qwen Image Edit pipeline on a GPU
3. Saves edited_image.jpg in each test case subdirectory
4. Downloads the processed test data folder back locally (replacing the original)
5. Deletes the Modal volume to prevent stale data corruption on future runs

Each test case subdirectory must contain:
    - original_image.jpg
    - edit_instruction.txt

The script is preemption-resilient: if a run is interrupted and restarted,
it resumes from where it left off by skipping test cases that already have
an edited_image.jpg file.

Usage:
    modal run modal_qwen_ie_inference.py --data-dir dummy_test_data

Environment variables (optional):
    QIE_MODAL_VOLUME_NAME  – Override volume name (useful for parallel runs)
    QIE_MODAL_APP_NAME     – Override Modal app name (useful for parallel runs)
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
VOLUME_NAME = os.environ.get("QIE_MODAL_VOLUME_NAME", "qwen-ie-inference-volume")
VOLUME_MOUNT = "/data"
REMOTE_TEST_DATA_DIR = "/data/test_data"

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 1.0
TRUE_CFG_SCALE = 4.0
SEED = int(os.environ.get("SGE_GEN_SEED", "0"))

# ============================================================================
# CUDA / Image Configuration
# ============================================================================
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App(os.environ.get("QIE_MODAL_APP_NAME", "qwen-ie-inference"))

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
        "transformers>=4.49.0",
        "peft",
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
    # Install diffusers from git for Qwen Image Edit support
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
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/data/.cache",
        }
    )
)

# HuggingFace secret for gated model access
huggingface_secret = modal.Secret.from_name("huggingface-secret")


# ============================================================================
# Enhanced Prompt Builder
# ============================================================================
def build_enhanced_prompt(edit_instruction: str) -> str:
    """
    Build an enhanced prompt for the Qwen Image Edit pipeline.

    The raw edit instruction is placed at the very top (first characters)
    to minimise chances of truncation due to token limits. The auxiliary
    enhancement text follows and provides additional context.
    """
    enhanced_prompt = f"""{edit_instruction}

You are given an original image that needs to be edited according to the instruction above. Please perform the specified edit on the image and generate the resulting edited scene.

Guidelines:
- Apply the edit described in the instruction precisely and faithfully
- Preserve all parts of the image that are not affected by the edit
- Maintain consistent lighting, shadows, style, and visual quality with the original image
- The edited image should look natural and photorealistic
- Do not introduce artifacts or distortions in unaffected regions"""

    return enhanced_prompt


# ============================================================================
# Remote Inference Function
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={VOLUME_MOUNT: volume},
    timeout=86400,  # 24 hours
    memory=65536,   # 64 GB RAM
)
def run_image_editing_inference():
    """
    Run image editing inference on all test cases in the uploaded test data folder.

    Expects the volume to already contain:
        /data/test_data/<case_id>/original_image.jpg
        /data/test_data/<case_id>/edit_instruction.txt

    Produces:
        /data/test_data/<case_id>/edited_image.jpg

    Preemption-resilient: skips test cases where edited_image.jpg already exists.
    """
    import importlib
    import inspect
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

    # ---- Load pipeline ----
    print("=" * 70)
    print("Loading Qwen Image Edit pipeline...")
    print(f"  Model: {MODEL_ID}")
    print("=" * 70)

    import diffusers

    # Try QwenImageEditPlusPipeline first (image editing), fall back to QwenImagePipeline
    QwenImageEditPlusPipeline = getattr(diffusers, "QwenImageEditPlusPipeline", None)
    if QwenImageEditPlusPipeline is None:
        try:
            QwenImageEditPlusPipeline = getattr(
                importlib.import_module(
                    "diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus"
                ),
                "QwenImageEditPlusPipeline",
            )
        except Exception:
            QwenImageEditPlusPipeline = None

    QwenImagePipeline = getattr(diffusers, "QwenImagePipeline", None)
    if QwenImagePipeline is None:
        try:
            QwenImagePipeline = getattr(
                importlib.import_module(
                    "diffusers.pipelines.qwenimage.pipeline_qwenimage"
                ),
                "QwenImagePipeline",
            )
        except Exception:
            QwenImagePipeline = None

    pipeline_cls = QwenImageEditPlusPipeline or QwenImagePipeline
    if pipeline_cls is None:
        raise ImportError(
            "Neither QwenImageEditPlusPipeline nor QwenImagePipeline could be imported. "
            "Install latest diffusers from main branch."
        )

    print(f"  Pipeline class: {pipeline_cls.__name__}")

    pipe = pipeline_cls.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=False)
    print("Pipeline loaded successfully (CPU offload enabled).\n")

    # Inspect call signature once to know what params are accepted
    call_params = set(inspect.signature(pipe.__call__).parameters.keys())
    print(f"Pipeline call params: {sorted(call_params)}\n")

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

        # Required files – support both .jpg and .png original images
        original_image_path = None
        for ext in ("jpg", "png"):
            candidate = case_dir / f"original_image.{ext}"
            if candidate.exists():
                original_image_path = candidate
                break
        edit_instruction_path = case_dir / "edit_instruction.txt"
        output_path = case_dir / "edited_image.jpg"

        # Check all required files exist
        missing = []
        if original_image_path is None:
            missing.append("original_image.jpg/.png")
        if not edit_instruction_path.exists():
            missing.append("edit_instruction.txt")

        if missing:
            print(f"[SKIP] Test case {case_name}: missing {', '.join(missing)}")
            skipped += 1
            continue

        # Skip if already processed (preemption resilience / resume support)
        if output_path.exists():
            print(f"[SKIP] Test case {case_name}: edited_image.jpg already exists (resume)")
            skipped += 1
            continue

        print(f"[{processed + 1}] Processing test case {case_name}...")

        # Load inputs
        original = Image.open(original_image_path).convert("RGB")
        edit_instruction = edit_instruction_path.read_text().strip()
        original_size = original.size  # (width, height)

        print(f"    Edit instruction: {edit_instruction[:100]}{'...' if len(edit_instruction) > 100 else ''}")
        print(f"    Image size: {original_size}")

        # Build enhanced prompt
        enhanced_prompt = build_enhanced_prompt(edit_instruction)

        # Build call kwargs
        call_kwargs = {
            "prompt": enhanced_prompt,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "num_images_per_prompt": 1,
        }
        if "true_cfg_scale" in call_params:
            call_kwargs["true_cfg_scale"] = TRUE_CFG_SCALE
        if "negative_prompt" in call_params:
            call_kwargs["negative_prompt"] = " "
        if "image" in call_params:
            call_kwargs["image"] = original
        # NOTE: We intentionally do NOT pass height/width to the pipeline.
        # Forcing small dimensions (e.g. 208x208, 128x128) causes the
        # diffusion model to hallucinate completely unrelated scenes.
        # Instead we let the model generate at its native resolution and
        # resize the output afterwards to match the original image.

        generator = torch.Generator(device="cpu").manual_seed(SEED)
        if "generator" in call_params:
            call_kwargs["generator"] = generator

        with torch.no_grad():
            result = pipe(**call_kwargs)

        # Save edited image
        edited_image = result.images[0]

        # Resize output to match original dimensions if they differ
        if edited_image.size != original_size:
            print(f"    Resizing output {edited_image.size} -> {original_size}")
            edited_image = edited_image.resize(original_size, Image.LANCZOS)

        edited_image.save(str(output_path), quality=95)
        print(f"    Saved: {output_path}")
        processed += 1

        # Periodic volume commit so progress survives container preemption
        if processed % 25 == 0:
            print(f"    [Checkpoint] Committing volume after {processed} cases ...")
            volume.commit()

    # ---- Final commit to persist any remaining results ----
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
    Run pretrained Qwen Image Edit inference on a local test data folder.

    Steps:
        1. Upload test data folder to Modal volume
        2. Run inference on GPU (all test cases processed in one function call)
        3. Download processed test data (with edited_image.jpg) back locally
        4. Delete the Modal volume (prevents stale data on future runs)

    Args:
        data_dir: Path to test data directory (each subdirectory is a test case)
    """
    import subprocess
    import sys
    import shutil
    import time

    data_dir_abs = os.path.abspath(data_dir)
    data_dir_parent = os.path.dirname(data_dir_abs)
    data_dir_basename = os.path.basename(data_dir_abs)

    # Validate local paths
    if not os.path.isdir(data_dir_abs):
        raise FileNotFoundError(f"Test data directory not found: {data_dir_abs}")

    # Count test cases
    test_cases = [
        d for d in os.listdir(data_dir_abs)
        if os.path.isdir(os.path.join(data_dir_abs, d))
    ]
    print("=" * 70)
    print("Qwen Image Edit – Pretrained Pipeline Inference")
    print("=" * 70)
    print(f"  Test data dir:   {data_dir_abs}")
    print(f"  Test cases:      {len(test_cases)}")
    print(f"  Volume:          {VOLUME_NAME}")
    print(f"  Model:           {MODEL_ID}")
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
    print(f"\n[Step 2/4] Running image editing inference on Modal ({GPU_CONFIG})...")
    inference_succeeded = False
    inference_result = {}
    t0 = time.time()
    try:
        inference_result = run_image_editing_inference.remote()
        elapsed = time.time() - t0
        print(f"\n  Inference finished ({elapsed:.1f}s)")
        print(f"  Result: {inference_result}")
        if inference_result.get("status") == "success":
            inference_succeeded = True
        else:
            print("  WARNING: Inference returned non-success status.")
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  WARNING: Inference interrupted or failed after {elapsed:.1f}s: {exc}")
        print("  Will attempt to download partial results from the volume...")

    # ==================================================================
    # Step 3: Download processed test data back locally
    # ==================================================================
    # Always attempt download – even on failure/timeout, the volume may
    # contain partial results (edited_image.jpg files) that we want to
    # preserve locally so that the next run can resume from where we left off.
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
            timeout=86400,  # 24 hours
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
            if inference_succeeded:
                raise RuntimeError("Failed to download processed test data from Modal volume.")
            else:
                print("  Could not download partial results. Local data restored from backup.")
        else:
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
                    if inference_succeeded:
                        raise RuntimeError("Download produced unexpected directory structure.")

            print(f"  Download complete ({elapsed:.1f}s)")

            # Remove backup
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
                print(f"  Removed backup")

    except subprocess.TimeoutExpired:
        print("  ERROR: Download timed out")
        if os.path.exists(backup_dir):
            shutil.move(backup_dir, data_dir_abs)
            print(f"  Restored backup to {data_dir_abs}")
        if inference_succeeded:
            raise

    # ==================================================================
    # Step 4: Delete the Modal volume
    # ==================================================================
    # Only delete the volume if inference fully succeeded.
    # On partial completion, keep the volume so the next run can resume
    # without re-uploading already-processed results.
    if inference_succeeded:
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
    else:
        print(f"\n[Step 4/4] Keeping Modal volume ({VOLUME_NAME}) – inference was partial.")
        print("  The volume contains partial results. On the next run, the upload")
        print("  step will overwrite it with local data (which now includes the")
        print("  partial results), and inference will resume from where it stopped.")

    # ==================================================================
    # Summary
    # ==================================================================
    # Count how many edited images we actually have locally now
    edited_count = sum(
        1 for d in os.listdir(data_dir_abs)
        if os.path.isdir(os.path.join(data_dir_abs, d))
        and os.path.isfile(os.path.join(data_dir_abs, d, "edited_image.jpg"))
    ) if os.path.isdir(data_dir_abs) else 0

    total_cases = len([
        d for d in os.listdir(data_dir_abs)
        if os.path.isdir(os.path.join(data_dir_abs, d))
    ]) if os.path.isdir(data_dir_abs) else 0

    print(f"\n{'=' * 70}")
    if inference_succeeded:
        print("DONE")
    else:
        print("PARTIAL – inference did not fully complete (see warnings above)")
        print("Re-run the same command to resume from where it left off.")
    print(f"{'=' * 70}")
    print(f"  Processed this run: {inference_result.get('processed', '?')} test cases")
    print(f"  Skipped this run:   {inference_result.get('skipped', '?')} test cases")
    print(f"  Total with edits:   {edited_count}/{total_cases} test cases have edited_image.jpg")
    print(f"  Output:             {data_dir_abs}")
    print(f"{'=' * 70}")

    return inference_result
