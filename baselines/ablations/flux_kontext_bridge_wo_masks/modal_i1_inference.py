"""
Modal inference script for Flux Kontext LoRA image editing.

Uploads a local test data folder and checkpoint to a temporary Modal volume,
runs the image editing pipeline on every test case, saves edited_image.jpg
back into each subdirectory, downloads the results locally, and deletes
the volume.

Usage:
    # Default: uses dummy_test_data/ and latest checkpoint
    python modal_i1_inference.py

    # Custom data dir and checkpoint dir
    python modal_i1_inference.py --data_dir /path/to/test_data --checkpoint_dir /path/to/checkpoint-epoch-13

Environment:
    Uses MODAL_TOKEN_ID / MODAL_TOKEN_SECRET env vars so the global
    ~/.modal.toml is never touched.
"""

import modal
import os
import sys
import argparse

# ============================================================================
# Configuration
# ============================================================================
VOLUME_NAME = "flux-kontext-i1-inference-tmp"
GPU_TYPE = "B200"
APP_NAME = "flux-kontext-i1-inference"

# Remote paths on the volume
REMOTE_DATA_DIR = "/data/test_data"
REMOTE_CHECKPOINT_DIR = "/data/checkpoint"

# ============================================================================
# Modal App & Image
# ============================================================================
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

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
        "sentencepiece",
        "tqdm",
        "packaging",
        "wheel",
    )
    .run_commands("pip install flash-attn --no-build-isolation")
    .env(
        {
            "SGE_GEN_SEED": os.environ.get("SGE_GEN_SEED", "0"),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/data/.cache",
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
            "dummy_test_data",
            "dummy_test_data/**",
            "*.safetensors",
            "*.bin",
            "*.pkl",
        ],
    )
)

huggingface_secret = modal.Secret.from_name("huggingface-secret")

app = modal.App(APP_NAME)


# ============================================================================
# Remote inference function
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_TYPE,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=7200,  # 2 hours
    memory=65536,
)
def run_inference_on_volume():
    """
    Load pipeline + LoRA weights, iterate over every test case on the volume,
    run the editing pipeline, and save edited_image.jpg in each subdirectory.
    """
    import os
    import sys
    import torch
    from PIL import Image
    from tqdm import tqdm

    sys.path.insert(0, "/root/flux_finetuning")

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline
    from flux_kontext_eval_utils import build_enhanced_prompt

    data_dir = REMOTE_DATA_DIR
    checkpoint_dir = REMOTE_CHECKPOINT_DIR

    # ----------------------------------------------------------------
    # Discover test cases: subdirectories with original_image.* and edit_instruction.txt
    # ----------------------------------------------------------------
    test_cases = []
    for entry in sorted(os.listdir(data_dir)):
        case_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(case_dir):
            continue
        # Find original image (jpg/png/jpeg)
        orig_path = None
        for fname in os.listdir(case_dir):
            if fname.lower().startswith("original_image"):
                orig_path = os.path.join(case_dir, fname)
                break
        instr_path = os.path.join(case_dir, "edit_instruction.txt")
        if orig_path and os.path.isfile(instr_path):
            test_cases.append({
                "case_dir": case_dir,
                "case_name": entry,
                "original_image_path": orig_path,
                "instruction_path": instr_path,
            })
    
    print(f"Found {len(test_cases)} test cases.")
    if len(test_cases) == 0:
        return {"status": "error", "message": "No test cases found"}

    # ----------------------------------------------------------------
    # Load pipeline
    # ----------------------------------------------------------------
    print("Loading Flux Kontext pipeline...")
    model_name = "black-forest-labs/FLUX.1-Kontext-dev"
    pipeline = FluxKontextPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Load LoRA weights from checkpoint
    lora_path = os.path.join(checkpoint_dir, "pytorch_lora_weights.safetensors")
    if os.path.isfile(lora_path):
        print(f"Loading LoRA weights from {lora_path}")
        pipeline.load_lora_weights(checkpoint_dir)
    else:
        print(f"WARNING: No LoRA weights found at {lora_path}. Running with base model.")

    pipeline.set_progress_bar_config(disable=True)

    # Deterministic generation
    generator = torch.Generator(device="cuda").manual_seed(int(os.environ.get("SGE_GEN_SEED", "0")))

    # ----------------------------------------------------------------
    # Process each test case
    # ----------------------------------------------------------------
    num_inference_steps = 20
    results = []

    for tc in tqdm(test_cases, desc="Running inference"):
        case_dir = tc["case_dir"]
        case_name = tc["case_name"]

        # Load original image
        orig_image = Image.open(tc["original_image_path"]).convert("RGB")
        orig_w, orig_h = orig_image.size

        # Load edit instruction
        with open(tc["instruction_path"], "r", encoding="utf-8") as f:
            edit_instruction = f.read().strip()

        # Build enhanced prompt (same as training/val)
        prompt = build_enhanced_prompt(edit_instruction)

        # Encode prompt
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt, prompt_2=None
            )
            prompt_embeds = prompt_embeds.to("cuda")
            pooled_prompt_embeds = pooled_prompt_embeds.to("cuda")

        # Call pipeline (match validation logic from eval_utils)
        call_args = {
            "original_image": orig_image,
            "num_inference_steps": num_inference_steps,
            "height": orig_h,
            "width": orig_w,
            "_auto_resize": False,
            "max_area": orig_h * orig_w,
        }

        with torch.autocast("cuda"):
            out = pipeline(
                **call_args,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                generator=generator,
            )

        # Extract predicted image
        if hasattr(out, "images"):
            pred_img = out.images[0]
        elif isinstance(out, (list, tuple)):
            candidate = out[0]
            pred_img = candidate[0] if isinstance(candidate, (list, tuple)) else candidate
        else:
            pred_img = out

        # Resize to original dimensions if needed
        if isinstance(pred_img, Image.Image):
            pred_w, pred_h = pred_img.size
            if (pred_w, pred_h) != (orig_w, orig_h):
                pred_img = pred_img.resize((orig_w, orig_h), resample=Image.BILINEAR)
        
        # Save as edited_image.jpg
        save_path = os.path.join(case_dir, "edited_image.jpg")
        pred_img.save(save_path, "JPEG", quality=95)
        print(f"  [{case_name}] Saved {save_path}")
        results.append({"case": case_name, "status": "ok"})

    # Commit volume so the edited images are persisted
    volume.commit()
    print(f"\nDone! Processed {len(results)} test cases. Volume committed.")
    return {"status": "success", "processed": len(results)}


# ============================================================================
# Local entrypoint: upload → run → download → delete volume
# ============================================================================
@app.local_entrypoint()
def main(
    data_dir: str = "dummy_test_data",
    checkpoint_dir: str = "models/kontext-i2i/checkpoint-epoch-13",
):
    """
    Local entrypoint: upload test data + checkpoint to volume, run remote
    inference, download results, and delete the volume.

    Args:
        data_dir: Local path to test data folder (each subdirectory has
                  original_image.* and edit_instruction.txt).
        checkpoint_dir: Local path to checkpoint directory containing
                        pytorch_lora_weights.safetensors.
    """
    import shutil
    import subprocess

    # Resolve to absolute paths
    data_dir = os.path.abspath(data_dir)
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    print(f"Data dir:       {data_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory does not exist: {data_dir}")
        sys.exit(1)
    if not os.path.isdir(checkpoint_dir):
        print(f"ERROR: Checkpoint directory does not exist: {checkpoint_dir}")
        sys.exit(1)

    # Detect modal binary
    modal_bin = shutil.which("modal")
    if modal_bin is None:
        modal_bin = "modal"

    # ------------------------------------------------------------------
    # Step 1: Upload test data to volume
    # ------------------------------------------------------------------
    print("\n[1/5] Uploading test data to Modal volume...")
    _run_modal_cmd(modal_bin, [
        "volume", "put", VOLUME_NAME, data_dir, "test_data",
    ])
    print("      Test data uploaded.")

    # ------------------------------------------------------------------
    # Step 2: Upload checkpoint to volume
    # ------------------------------------------------------------------
    print("\n[2/5] Uploading checkpoint to Modal volume...")
    _run_modal_cmd(modal_bin, [
        "volume", "put", VOLUME_NAME, checkpoint_dir, "checkpoint",
    ])
    print("      Checkpoint uploaded.")

    # ------------------------------------------------------------------
    # Step 3: Run remote inference
    # ------------------------------------------------------------------
    print("\n[3/5] Running remote inference...")
    result = run_inference_on_volume.remote()
    print(f"      Remote result: {result}")

    if result.get("status") != "success":
        print("ERROR: Remote inference failed.")
        _delete_volume(modal_bin)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Download processed test data back (replace local folder)
    # ------------------------------------------------------------------
    print("\n[4/5] Downloading processed test data...")
    # Download to a temp dir first, then replace
    tmp_download = data_dir + "_modal_tmp"
    if os.path.exists(tmp_download):
        shutil.rmtree(tmp_download)
    os.makedirs(tmp_download, exist_ok=True)

    _run_modal_cmd(modal_bin, [
        "volume", "get", VOLUME_NAME, "test_data/", tmp_download,
    ])

    # The downloaded structure might be nested (test_data/test_data/...) or
    # flat depending on modal volume get behavior. Detect and fix.
    downloaded_contents = os.listdir(tmp_download)
    actual_data_root = tmp_download

    # If modal created a nested test_data/ dir inside tmp_download
    nested_candidate = os.path.join(tmp_download, "test_data")
    if len(downloaded_contents) == 1 and os.path.isdir(nested_candidate):
        actual_data_root = nested_candidate

    # Replace local data_dir with downloaded version
    backup_dir = data_dir + "_backup"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.move(data_dir, backup_dir)

    shutil.move(actual_data_root, data_dir)

    # Clean up temp and backup
    if os.path.exists(tmp_download):
        shutil.rmtree(tmp_download)
    shutil.rmtree(backup_dir)

    print(f"      Local data dir replaced: {data_dir}")

    # Verify edited images exist
    edited_count = 0
    for entry in os.listdir(data_dir):
        case_dir = os.path.join(data_dir, entry)
        if os.path.isdir(case_dir) and os.path.isfile(os.path.join(case_dir, "edited_image.jpg")):
            edited_count += 1
    print(f"      Verified: {edited_count} test cases now have edited_image.jpg")

    # ------------------------------------------------------------------
    # Step 5: Delete the temporary volume
    # ------------------------------------------------------------------
    print("\n[5/5] Deleting temporary Modal volume...")
    _delete_volume(modal_bin)
    print("      Volume deleted.")

    print(f"\nAll done! {edited_count} edited images saved in {data_dir}/")


def _run_modal_cmd(modal_bin: str, args: list):
    """Run a modal CLI command, raising on failure."""
    import subprocess
    cmd = [modal_bin] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"Modal command failed: {' '.join(cmd)}")


def _delete_volume(modal_bin: str):
    """Delete the temporary inference volume."""
    import subprocess
    cmd = [modal_bin, "volume", "delete", VOLUME_NAME, "--yes"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: Could not delete volume. Manual cleanup needed.")
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
