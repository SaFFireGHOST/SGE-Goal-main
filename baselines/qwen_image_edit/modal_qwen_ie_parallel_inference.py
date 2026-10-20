"""
Modal script for parallel Qwen Image Edit inference with N GPU workers.

Splits test cases into N contiguous partitions, each processed by an
independent GPU worker container running simultaneously.  All workers
share a single Modal volume for reading inputs and writing outputs.

Usage:
    QIE_MODAL_APP_NAME=qwen-ie-calvin-parallel \
    QIE_MODAL_VOLUME_NAME=qwen-ie-calvin-parallel-volume \
    modal run modal_qwen_ie_parallel_inference.py \
        --data-dir calvin_abcd_test_data --num-workers 7

Environment variables:
    QIE_MODAL_APP_NAME     – Modal app name  (default: qwen-ie-parallel)
    QIE_MODAL_VOLUME_NAME  – Volume name     (default: qwen-ie-parallel-volume)
"""

import modal
import os

# ============================================================================
# Configuration
# ============================================================================
GPU_CONFIG = "L40S:1"

VOLUME_NAME = os.environ.get("QIE_MODAL_VOLUME_NAME", "qwen-ie-parallel-volume")
VOLUME_MOUNT = "/data"
REMOTE_TEST_DATA_DIR = "/data/test_data"

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 1.0
TRUE_CFG_SCALE = 4.0
SEED = 0

CUDA_TAG = "12.8.1-devel-ubuntu24.04"

# ============================================================================
# Modal App & Volume
# ============================================================================
app = modal.App(os.environ.get("QIE_MODAL_APP_NAME", "qwen-ie-parallel"))
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ============================================================================
# Container Image
# ============================================================================
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .uv_pip_install("torch==2.8.0", "torchvision", "torchaudio")
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
    .run_commands("apt-get update && apt-get install -y git")
    .run_commands("pip install --upgrade pip setuptools wheel")
    .run_commands("pip install git+https://github.com/huggingface/diffusers.git")
    .run_commands("pip install flash-attn --no-build-isolation")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Container-local cache – avoids write conflicts when multiple
            # workers download the model simultaneously to the shared volume.
            "HF_HOME": "/root/.cache",
        }
    )
)

huggingface_secret = modal.Secret.from_name("huggingface-secret")


# ============================================================================
# Prompt Builder
# ============================================================================
def build_enhanced_prompt(edit_instruction: str) -> str:
    """Build enhanced prompt with the raw edit instruction first."""
    return f"""{edit_instruction}

You are given an original image that needs to be edited according to the instruction above. Please perform the specified edit on the image and generate the resulting edited scene.

Guidelines:
- Apply the edit described in the instruction precisely and faithfully
- Preserve all parts of the image that are not affected by the edit
- Maintain consistent lighting, shadows, style, and visual quality with the original image
- The edited image should look natural and photorealistic
- Do not introduce artifacts or distortions in unaffected regions"""


# ============================================================================
# GPU Worker Function
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={VOLUME_MOUNT: volume},
    timeout=86400,  # 24 hours
    memory=65536,   # 64 GB RAM
)
def run_partition(case_ids: list[str]) -> dict:
    """
    Process a partition of test cases on a single GPU.

    Args:
        case_ids: List of case directory names (e.g. ["1", "2", "3", ...])

    Returns:
        dict with processed / skipped / total counts.
    """
    import importlib
    import inspect
    import torch
    from PIL import Image
    from pathlib import Path

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    test_data_dir = Path(REMOTE_TEST_DATA_DIR)

    # Reload volume to see the newly‑uploaded data
    volume.reload()

    # ---- Load pipeline ------------------------------------------------
    print("=" * 70)
    print(f"Worker starting – {len(case_ids)} cases assigned")
    if case_ids:
        print(f"  Range: {case_ids[0]} .. {case_ids[-1]}")
    print("=" * 70)

    import diffusers

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
            "Neither QwenImageEditPlusPipeline nor QwenImagePipeline found. "
            "Install latest diffusers from main branch."
        )

    print(f"  Pipeline class: {pipeline_cls.__name__}")
    pipe = pipeline_cls.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=False)
    print("  Pipeline loaded (CPU offload enabled).\n")

    call_params = set(inspect.signature(pipe.__call__).parameters.keys())

    # ---- Process assigned cases ----------------------------------------
    processed = 0
    skipped = 0

    for case_id in case_ids:
        case_dir = test_data_dir / case_id
        if not case_dir.is_dir():
            print(f"[WARN] Case dir not found on volume: {case_id}")
            skipped += 1
            continue

        # Find original image (.jpg or .png)
        original_image_path = None
        for ext in ("jpg", "png"):
            candidate = case_dir / f"original_image.{ext}"
            if candidate.exists():
                original_image_path = candidate
                break

        edit_instruction_path = case_dir / "edit_instruction.txt"
        output_path = case_dir / "edited_image.jpg"

        # Validate required files
        missing = []
        if original_image_path is None:
            missing.append("original_image.jpg/.png")
        if not edit_instruction_path.exists():
            missing.append("edit_instruction.txt")
        if missing:
            print(f"[SKIP] {case_id}: missing {', '.join(missing)}")
            skipped += 1
            continue

        # Resume support – skip already processed
        if output_path.exists():
            print(f"[SKIP] {case_id}: already done (resume)")
            skipped += 1
            continue

        print(f"[{processed + 1}] Processing {case_id}...")

        original = Image.open(original_image_path).convert("RGB")
        original_size = original.size
        edit_instruction = edit_instruction_path.read_text().strip()

        print(f"    Edit: {edit_instruction[:80]}{'...' if len(edit_instruction) > 80 else ''}")
        print(f"    Size: {original_size}")

        enhanced_prompt = build_enhanced_prompt(edit_instruction)

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
        # NOTE: height/width NOT passed – small dims cause hallucinations.
        # Output is resized to match original after generation.

        generator = torch.Generator(device="cpu").manual_seed(SEED)
        if "generator" in call_params:
            call_kwargs["generator"] = generator

        with torch.no_grad():
            result = pipe(**call_kwargs)

        edited_image = result.images[0]
        if edited_image.size != original_size:
            print(f"    Resizing {edited_image.size} -> {original_size}")
            edited_image = edited_image.resize(original_size, Image.LANCZOS)

        edited_image.save(str(output_path), quality=95)
        print(f"    Saved: {output_path}")
        processed += 1

        # Commit every 10 cases to persist progress
        if processed % 10 == 0:
            print(f"    [Checkpoint] Committing after {processed} cases...")
            volume.commit()

    # ---- Final commit --------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Worker done. Processed: {processed}, Skipped: {skipped}")
    print(f"{'=' * 70}")
    volume.commit()

    return {
        "status": "success",
        "processed": processed,
        "skipped": skipped,
        "total": len(case_ids),
    }


# ============================================================================
# Local Entrypoint
# ============================================================================
@app.local_entrypoint()
def main(data_dir: str = "test_data", num_workers: int = 4):
    """
    Upload test data, partition into N chunks, run parallel GPU workers.

    This entrypoint blocks until ALL workers complete.  Run via nohup
    for overnight execution.  The downloader script handles syncing
    results back to local while inference runs.

    Args:
        data_dir:    Path to test data directory
        num_workers: Number of parallel GPU workers
    """
    import subprocess
    import sys
    import time

    data_dir_abs = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir_abs):
        raise FileNotFoundError(f"Not found: {data_dir_abs}")

    # Get sorted case IDs
    case_ids = sorted(
        [
            d
            for d in os.listdir(data_dir_abs)
            if os.path.isdir(os.path.join(data_dir_abs, d))
        ],
        key=lambda x: int(x) if x.isdigit() else x,
    )

    print("=" * 70)
    print("Qwen Image Edit – Parallel Inference")
    print("=" * 70)
    print(f"  Data dir:    {data_dir_abs}")
    print(f"  Test cases:  {len(case_ids)}")
    print(f"  Workers:     {num_workers}")
    print(f"  Volume:      {VOLUME_NAME}")
    print(f"  GPU:         {GPU_CONFIG}")
    print(f"  Model:       {MODEL_ID}")
    print("=" * 70)

    # ---- Partition cases (contiguous chunks) ----------------------------
    partitions = []
    chunk = len(case_ids) // num_workers
    remainder = len(case_ids) % num_workers
    start = 0
    for i in range(num_workers):
        end = start + chunk + (1 if i < remainder else 0)
        partitions.append(case_ids[start:end])
        start = end

    for i, p in enumerate(partitions):
        print(f"  Worker {i}: {len(p)} cases ({p[0]} .. {p[-1]})")

    # ---- Step 1: Upload -------------------------------------------------
    modal_cmd = [sys.executable, "-m", "modal"]
    print(f"\n[Step 1/2] Uploading test data to volume ({VOLUME_NAME})...")
    t0 = time.time()
    result = subprocess.run(
        modal_cmd
        + ["volume", "put", "--force", VOLUME_NAME, data_dir_abs, "test_data"],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr.strip()}")
        raise RuntimeError("Upload failed")
    print(f"  Upload complete ({elapsed:.1f}s)")

    # ---- Step 2: Run parallel workers -----------------------------------
    print(f"\n[Step 2/2] Launching {num_workers} parallel GPU workers...")
    t0 = time.time()

    results = list(run_partition.map(partitions, order_outputs=False))

    elapsed = time.time() - t0
    total_processed = sum(r["processed"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)

    print(f"\n{'=' * 70}")
    print(f"ALL {num_workers} WORKERS COMPLETE  ({elapsed:.1f}s / {elapsed / 60:.1f}m)")
    print(f"{'=' * 70}")
    print(f"  Total processed: {total_processed}")
    print(f"  Total skipped:   {total_skipped}")
    print(f"  Total cases:     {len(case_ids)}")
    print(f"  Volume:          {VOLUME_NAME}  (results on volume)")
    print(f"{'=' * 70}")

    return results
