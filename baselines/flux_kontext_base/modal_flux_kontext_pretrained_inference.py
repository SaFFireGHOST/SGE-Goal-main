"""
Modal script to run image editing inference using the pre-trained
FLUX.1-Kontext-dev pipeline (no finetuning / LoRA — just the original model).

Each test case sub-directory must contain:
  - original_image.{jpg,jpeg,png}   — the image to edit
  - edit_instruction.txt             — first line is the edit instruction

The script will save "edited_image.jpg" inside each test case sub-directory.

Workflow:
  1. Uploads the local test data folder to a temporary Modal volume.
  2. Runs inference on a Modal GPU: loads the pre-trained FLUX.1-Kontext-dev
     pipeline, iterates over every test-case sub-directory, and saves
     "edited_image.jpg" in each.
  3. Downloads the processed test data folder back to local, replacing the original.
  4. Deletes the temporary Modal volume so future runs start fresh.

Usage:
    conda activate lgie
    modal run modal_flux_kontext_pretrained_inference.py \
        --data-dir dummy_test_data
"""

import modal
import os

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App("flux-kontext-pretrained-bridge-remaining-inference")

VOLUME_NAME = "flux-kontext-pretrained-bridge-remaining-inference-volume"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# GPU configuration
GPU_TYPE = "L40S"
GPU_CONFIG = f"{GPU_TYPE}:1"

# CUDA Configuration
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Paths on the Modal volume (mounted at /data)
REMOTE_DATA_DIR = "/data/test_data"

# ============================================================================
# Modal Image with all required dependencies
# ============================================================================
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
        "pillow",
        "numpy",
        "huggingface-hub",
        "hf_transfer",
        "tqdm",
        "packaging",
        "wheel",
        "sentencepiece",
    )
    .run_commands("pip install flash-attn --no-build-isolation")
    .env(
        {
            "SGE_GEN_SEED": os.environ.get("SGE_GEN_SEED", "0"),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/data/.cache",
        }
    )
)

huggingface_secret = modal.Secret.from_name("huggingface-secret")


def build_enhanced_prompt(edit_instruction: str) -> str:
    """
    Build an enhanced prompt for the pre-trained FLUX.1-Kontext-dev pipeline.

    The raw edit instruction is placed at the very start of the prompt so that
    it falls within CLIP's 77-token window and is never truncated.  The rest
    of the prompt supplies auxiliary context that helps the model understand
    the task.

    Args:
        edit_instruction: The short, natural-language edit instruction
                          (e.g. "put the broccoli in the pot").

    Returns:
        An enhanced prompt string.
    """
    enhanced_prompt = (
        f"{edit_instruction}\n\n"
        "You are given the original image above that needs to be edited. "
        "The edit instruction above describes what change should be made to the image. "
        "Please apply the described edit to the original image and produce the resulting edited scene.\n\n"
        "Guidelines:\n"
        "- Perform only the edit described in the instruction; do not alter anything else.\n"
        "- Preserve the original scene layout, lighting, colours, and all unaffected elements exactly.\n"
        "- The edited image should look photorealistic, physically plausible, and visually consistent "
        "with the original."
    )
    return enhanced_prompt


# ============================================================================
# Remote: Upload files to volume
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,
)
def upload_files_to_volume(file_data: list[dict]):
    """
    Upload a batch of files to the Modal volume.
    file_data: list of {"remote_path": str, "content": bytes}
    """
    for item in file_data:
        remote_path = item["remote_path"]
        content = item["content"]
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        with open(remote_path, "wb") as f:
            f.write(content)
    volume.commit()
    print(f"Uploaded {len(file_data)} files to volume.")
    return {"status": "success", "files": len(file_data)}


# ============================================================================
# Remote: Run inference on all test cases
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=14400,  # 4 hours
    memory=65536,   # 64 GB RAM
)
def run_inference():
    """
    Load the pre-trained FLUX.1-Kontext-dev pipeline and run image editing
    inference on every test-case sub-directory inside REMOTE_DATA_DIR.

    Each test case needs:
      - original_image.{jpg,jpeg,png}
      - edit_instruction.txt  (first line = edit instruction)

    Saves "edited_image.jpg" in each test case directory.
    """
    import torch
    from PIL import Image
    from tqdm import tqdm
    from diffusers import FluxKontextPipeline

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    device = "cuda"
    dtype = torch.bfloat16

    # ------------------------------------------------------------------
    # 1. Load pre-trained FLUX.1-Kontext-dev pipeline (no LoRA / finetuning)
    # ------------------------------------------------------------------
    print("Loading pre-trained FLUX.1-Kontext-dev pipeline ...")
    pipeline = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=dtype,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    print("Pipeline loaded successfully.")

    # Reproducible generation
    seed = int(os.environ.get("SGE_GEN_SEED", "42"))
    generator = torch.Generator(device=device).manual_seed(seed)

    # ------------------------------------------------------------------
    # 2. Discover test cases
    # ------------------------------------------------------------------
    test_case_dirs = sorted(
        [
            d
            for d in os.listdir(REMOTE_DATA_DIR)
            if os.path.isdir(os.path.join(REMOTE_DATA_DIR, d))
        ],
        key=lambda x: int(x) if x.isdigit() else x,
    )
    print(f"Found {len(test_case_dirs)} test case directories.")

    # ------------------------------------------------------------------
    # 3. Helper to find image files
    # ------------------------------------------------------------------
    def find_image_file(directory: str, base_name: str):
        """Find base_name.{jpg,jpeg,png} in directory. Returns path or None."""
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(directory, base_name + ext)
            if os.path.exists(p):
                return p
        return None

    # ------------------------------------------------------------------
    # 4. Run inference for each test case
    # ------------------------------------------------------------------
    num_inference_steps = 28  # default for Kontext-dev

    for tc_name in tqdm(test_case_dirs, desc="Inference"):
        tc_dir = os.path.join(REMOTE_DATA_DIR, tc_name)

        # Locate required files
        original_path = find_image_file(tc_dir, "original_image")
        instruction_path = os.path.join(tc_dir, "edit_instruction.txt")

        # Validate
        missing = []
        if original_path is None:
            missing.append("original_image.*")
        if not os.path.exists(instruction_path):
            missing.append("edit_instruction.txt")
        if missing:
            print(f"  SKIPPING {tc_name}: missing {missing}")
            continue

        # Load inputs
        orig_image = Image.open(original_path).convert("RGB")
        with open(instruction_path, "r", encoding="utf-8") as f:
            edit_instruction = f.readline().strip()

        print(f"  [{tc_name}] Instruction: {edit_instruction}")

        # Build enhanced prompt (edit instruction first, then auxiliary context)
        prompt = build_enhanced_prompt(edit_instruction)

        # Get original image dimensions for generation output
        orig_w, orig_h = orig_image.size

        # Run the pre-trained pipeline with just image + prompt
        # NOTE: no torch.autocast — the pipeline handles dtype internally (bfloat16)
        out = pipeline(
            image=orig_image,
            prompt=prompt,
            height=orig_h,
            width=orig_w,
            num_inference_steps=num_inference_steps,
            generator=generator,
            _auto_resize=False,           # don't snap to preferred resolutions
            max_area=orig_h * orig_w,     # don't rescale to fit within a cap
        )

        # Extract predicted image
        if hasattr(out, "images") and out.images:
            pred_img = out.images[0]
        elif isinstance(out, (list, tuple)):
            candidate = out[0]
            pred_img = candidate[0] if isinstance(candidate, (list, tuple)) else candidate
        else:
            pred_img = out

        # Save
        output_path = os.path.join(tc_dir, "edited_image.jpg")
        if pred_img is not None and isinstance(pred_img, Image.Image):
            pred_img.save(output_path, "JPEG", quality=95)
            print(f"  Saved: {output_path}")
        else:
            print(f"  WARNING: Pipeline returned None for {tc_name}")

    # Commit volume so the caller can download the results
    volume.commit()
    print("All inference complete. Volume committed.")
    return {"status": "success", "test_cases": len(test_case_dirs)}


# ============================================================================
# Remote: List all files under the test data directory on the volume
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
)
def list_volume_files():
    """Return a list of all file paths (relative to REMOTE_DATA_DIR) on the volume."""
    results = []
    for root, _dirs, files in os.walk(REMOTE_DATA_DIR):
        for fname in files:
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, REMOTE_DATA_DIR)
            results.append(rel)
    return results


# ============================================================================
# Remote: Download a single file from the volume
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
)
def download_file_from_volume(relative_path: str):
    """Read a single file from the volume and return its bytes."""
    full = os.path.join(REMOTE_DATA_DIR, relative_path)
    with open(full, "rb") as f:
        return f.read()


# ============================================================================
# Local entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    data_dir: str = "../dummy_test_data",
):
    """
    Local entry point.

    Parameters
    ----------
    data_dir : str
        Path to the local test data folder. Each sub-folder is a test case
        containing original_image.{jpg,jpeg,png} and edit_instruction.txt.
    """
    import shutil
    import time

    data_dir = os.path.abspath(data_dir)
    print(f"Data directory: {data_dir}")

    # ------------------------------------------------------------------
    # 1. Collect all files to upload (test data only — no checkpoint needed)
    # ------------------------------------------------------------------
    print("\n[1/4] Collecting files for upload ...")
    upload_batch: list[dict] = []

    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, data_dir)
            remote_path = os.path.join(REMOTE_DATA_DIR, rel)
            with open(local_path, "rb") as f:
                content = f.read()
            upload_batch.append({"remote_path": remote_path, "content": content})

    total_bytes = sum(len(item["content"]) for item in upload_batch)
    print(f"  {len(upload_batch)} files, {total_bytes / 1e6:.1f} MB total")

    # ------------------------------------------------------------------
    # 2. Upload in batches (Modal has message-size limits ~50 MB)
    # ------------------------------------------------------------------
    print("\n[2/4] Uploading files to Modal volume ...")
    BATCH_SIZE_BYTES = 40 * 1024 * 1024  # 40 MB per batch
    batch: list[dict] = []
    batch_bytes = 0

    for item in upload_batch:
        item_size = len(item["content"])
        if batch and (batch_bytes + item_size > BATCH_SIZE_BYTES):
            upload_files_to_volume.remote(batch)
            batch = []
            batch_bytes = 0
        batch.append(item)
        batch_bytes += item_size

    if batch:
        upload_files_to_volume.remote(batch)

    print("  Upload complete.")

    # ------------------------------------------------------------------
    # 3. Run inference remotely
    # ------------------------------------------------------------------
    print("\n[3/4] Running inference on Modal GPU ...")
    t0 = time.time()
    result = run_inference.remote()
    elapsed = time.time() - t0
    print(f"  Inference finished in {elapsed:.1f}s — {result}")

    # ------------------------------------------------------------------
    # 4. Download processed test data and replace local copy
    # ------------------------------------------------------------------
    print("\n[4/4] Downloading processed test data from Modal volume ...")

    remote_files = list_volume_files.remote()
    print(f"  {len(remote_files)} files on volume")

    # Remove existing local data dir and recreate
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    for rel_path in remote_files:
        content = download_file_from_volume.remote(rel_path)
        local_path = os.path.join(data_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(content)

    print(f"  Downloaded and replaced {data_dir}")

    # ------------------------------------------------------------------
    # 5. Delete the temporary Modal volume
    # ------------------------------------------------------------------
    print("\nCleaning up: deleting temporary Modal volume ...")
    try:
        modal.Volume.delete(VOLUME_NAME)
        print(f"  Volume '{VOLUME_NAME}' deleted.")
    except Exception as e:
        print(f"  Warning: Could not delete volume: {e}")
        print(f"  You can delete it manually: modal volume rm {VOLUME_NAME}")

    print("\nDone! Edited images saved to each test-case sub-directory as 'edited_image.jpg'.")
