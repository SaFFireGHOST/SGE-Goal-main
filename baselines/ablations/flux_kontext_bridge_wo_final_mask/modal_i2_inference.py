"""
Modal script for running Flux Kontext i2 ablation inference on a test data folder.

Workflow:
1. Uploads the local test data folder to a Modal volume.
2. Runs the image editing pipeline on Modal GPU for every test case.
3. Saves "edited_image.jpg" inside each test-case subdirectory on the volume.
4. Downloads the processed folder back to the local machine, replacing the
   original local folder.
5. Deletes the Modal volume so future runs always start fresh.

Usage:
    MODAL_TOKEN_ID=<id> MODAL_TOKEN_SECRET=<secret> \\
      /path/to/modal run modal_i2_inference.py \\
        --data-dir dummy_test_data \\
        --checkpoint-dir models/kontext-i2i/checkpoint-epoch-9
"""

import modal
import os

# ---------------------------------------------------------------------------
# Modal configuration
# ---------------------------------------------------------------------------
APP_NAME = "flux-kontext-i2-inference"
VOLUME_NAME = "flux-kontext-i2-inference-volume"
GPU_TYPE = "B200"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# CUDA image (same stack as training)
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
            "SGE_GEN_SEED": os.environ.get("SGE_GEN_SEED", "0"),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/data/.cache",
        }
    )
    # Ship pipeline + eval utils (small .py files)
    .add_local_dir(
        ".",
        remote_path="/root/flux_finetuning",
        ignore=[
            "*.git*", "__pycache__", "*.pyc", "*.pyo", ".env", ".venv",
            "*.log", ".DS_Store", "*.egg-info", ".pytest_cache",
            "models", "models/**", "*.safetensors", "*.bin", "*.pkl",
            "dummy_test_data", "dummy_test_data/**",
            "bridge_data_v2_tel_ier_fer_gt_test_data",
            "bridge_data_v2_tel_ier_fer_gt_test_data/**",
        ],
    )
)

huggingface_secret = modal.Secret.from_name("huggingface-secret")

app = modal.App(APP_NAME)

# ---------------------------------------------------------------------------
# Remote GPU function – runs inference over every test case on the volume
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": volume},
    secrets=[huggingface_secret],
    timeout=86400,
    memory=65536,
)
def run_inference(
    data_dir_volume_path: str,
    checkpoint_volume_path: str,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    seed: int = 0,
):
    """
    For every sub-directory in *data_dir_volume_path* that contains
    ``original_image.jpg``, ``initial_edit_region_mask.jpg``, and
    ``edit_instruction.txt``, run the Flux Kontext i2 ablation pipeline
    and save the result as ``edited_image.jpg``.
    """
    import sys
    sys.path.insert(0, "/root/flux_finetuning")

    import torch
    from PIL import Image
    from tqdm import tqdm
    from peft import PeftModel

    from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline
    from flux_kontext_eval_utils import build_enhanced_prompt

    data_root = data_dir_volume_path   # absolute path on volume, e.g. /data/test_data/...
    ckpt_root = checkpoint_volume_path  # absolute path on volume, e.g. /data/checkpoint/...

    # ------------------------------------------------------------------
    # 1. Discover test cases
    # ------------------------------------------------------------------
    cases = sorted(
        [
            d
            for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ],
        key=lambda x: int(x) if x.isdigit() else x,
    )
    print(f"Found {len(cases)} test-case directories in {data_root}")

    if not cases:
        return {"status": "error", "message": "No test cases found"}

    # ------------------------------------------------------------------
    # 2. Load base pipeline
    # ------------------------------------------------------------------
    weight_dtype = torch.bfloat16
    print("Loading base Flux Kontext pipeline …")
    pipeline = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=weight_dtype,
    )

    # ------------------------------------------------------------------
    # 3. Load LoRA weights from the checkpoint
    # ------------------------------------------------------------------
    lora_weights_path = os.path.join(ckpt_root, "pytorch_lora_weights.safetensors")
    if os.path.isfile(lora_weights_path):
        print(f"Loading LoRA weights from {lora_weights_path}")
        pipeline.load_lora_weights(ckpt_root)
    else:
        print(f"WARNING: No LoRA weights found at {lora_weights_path}; using base model.")

    pipeline = pipeline.to("cuda", dtype=weight_dtype)
    pipeline.set_progress_bar_config(disable=False)

    generator = torch.Generator(device="cuda").manual_seed(int(os.environ.get("SGE_GEN_SEED", str(seed))))

    # ------------------------------------------------------------------
    # 4. Run inference on each test case (skip already-processed ones)
    # ------------------------------------------------------------------
    COMMIT_EVERY = 50  # commit volume every N new predictions to save progress
    processed_count = 0
    skipped_count = 0
    new_count = 0

    for case_name in tqdm(cases, desc="Inference"):
        case_dir = os.path.join(data_root, case_name)
        out_path = os.path.join(case_dir, "edited_image.jpg")

        # Skip if already processed
        if os.path.isfile(out_path):
            skipped_count += 1
            continue

        orig_path = os.path.join(case_dir, "original_image.jpg")
        mask_path = os.path.join(case_dir, "initial_edit_region_mask.jpg")
        inst_path = os.path.join(case_dir, "edit_instruction.txt")

        # Validate required files
        if not os.path.isfile(orig_path):
            print(f"  [{case_name}] Skipping – missing original_image.jpg")
            continue
        if not os.path.isfile(mask_path):
            print(f"  [{case_name}] Skipping – missing initial_edit_region_mask.jpg")
            continue
        if not os.path.isfile(inst_path):
            print(f"  [{case_name}] Skipping – missing edit_instruction.txt")
            continue

        # Load inputs
        orig_image = Image.open(orig_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("RGB")  # pipeline expects RGB
        with open(inst_path, "r") as f:
            edit_instruction = f.read().strip()

        prompt = build_enhanced_prompt(edit_instruction)

        # Encode prompt once for this sample
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt, prompt_2=None
            )
            prompt_embeds = prompt_embeds.to("cuda")
            pooled_prompt_embeds = pooled_prompt_embeds.to("cuda")

        orig_w, orig_h = orig_image.size

        # Run pipeline
        with torch.autocast("cuda"):
            output = pipeline(
                original_image=orig_image,
                initial_edit_region_mask=mask_image,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=orig_h,
                width=orig_w,
                _auto_resize=False,
                max_area=orig_h * orig_w,
            )

        # Extract predicted image
        if hasattr(output, "images"):
            pred_img = output.images[0]
        elif isinstance(output, (list, tuple)):
            pred_img = output[0] if not isinstance(output[0], (list, tuple)) else output[0][0]
        else:
            pred_img = output

        if not isinstance(pred_img, Image.Image):
            import numpy as np
            pred_img = Image.fromarray(np.asarray(pred_img).astype("uint8")).convert("RGB")

        # Resize to original dimensions if pipeline changed the size
        if pred_img.size != (orig_w, orig_h):
            pred_img = pred_img.resize((orig_w, orig_h), resample=Image.BILINEAR)

        # Save
        pred_img.save(out_path, quality=95)
        new_count += 1
        processed_count += 1
        print(f"  [{case_name}] Saved edited_image.jpg ({pred_img.size})")

        # Periodically commit to save progress
        if new_count % COMMIT_EVERY == 0:
            volume.commit()
            print(f"  >>> Volume committed (checkpoint at {new_count} new, {skipped_count} skipped)")

    # Final commit
    volume.commit()
    print(f"Volume committed. new={new_count}, skipped={skipped_count}, total_dirs={len(cases)}")
    return {"status": "success", "cases_processed": new_count, "skipped": skipped_count}


# ---------------------------------------------------------------------------
# Upload helper – push a local folder to the volume
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={"/data": volume}, timeout=600)
def upload_to_volume(file_data: list[dict]):
    """Write a list of {path, content_bytes} dicts into the volume."""
    for item in file_data:
        dest = item["path"]
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(item["content_bytes"])
    volume.commit()
    print(f"Uploaded {len(file_data)} files to volume.")


# ---------------------------------------------------------------------------
# Remote helpers for downloading from volume (avoids local-entrypoint SDK limits)
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={"/data": volume}, timeout=300)
def list_volume_files(prefix: str) -> list[str]:
    """List all file paths under *prefix* on the volume (recursive)."""
    import os
    root = os.path.join("/data", prefix)
    result = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            # Return volume-relative path (strip "/data/")
            result.append(os.path.relpath(full, "/data"))
    return result


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def download_volume_files(paths: list[str]) -> list[tuple]:
    """Read a batch of files from the volume and return (path, bytes) pairs."""
    result = []
    for p in paths:
        full = os.path.join("/data", p)
        with open(full, "rb") as f:
            result.append((p, f.read()))
    return result


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    data_dir: str = "dummy_test_data",
    checkpoint_dir: str = "models/kontext-i2i/checkpoint-epoch-9",
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    seed: int = 0,
):
    """
    1. Upload test data + checkpoint to a temporary Modal volume.
    2. Run GPU inference remotely.
    3. Download processed test data (with edited_image.jpg) back.
    4. Replace local data_dir.
    5. Delete the Modal volume.
    """
    import shutil
    import time

    def _fprint(msg: str):
        print(msg, flush=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_data_dir = os.path.join(script_dir, data_dir) if not os.path.isabs(data_dir) else data_dir
    local_ckpt_dir = os.path.join(script_dir, checkpoint_dir) if not os.path.isabs(checkpoint_dir) else checkpoint_dir

    if not os.path.isdir(local_data_dir):
        raise FileNotFoundError(f"Data directory not found: {local_data_dir}")
    if not os.path.isdir(local_ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {local_ckpt_dir}")

    # Volume-mount paths
    vol_data_dir = "/data/test_data"
    vol_ckpt_dir = "/data/checkpoint"

    # ======================================================================
    # Step 1: Upload test data + checkpoint to volume (skip if resuming)
    # ======================================================================
    _fprint("\n[Step 1/4] Checking if data already exists on Modal volume …")

    # Quick check: see if the volume already has test_data with subdirectories
    # by trying to list via a remote helper
    existing_count = 0
    try:
        existing_files = list_volume_files.remote("test_data")
        existing_count = len(existing_files)
    except Exception:
        existing_count = 0

    if existing_count > 0:
        _fprint(f"  Volume already has {existing_count} files under test_data/. Skipping upload (resume mode).")
    else:
        _fprint("  No existing data on volume. Uploading test data and checkpoint …")

        def _collect_files(local_root: str, vol_root: str) -> list[dict]:
            items = []
            for dirpath, _, filenames in os.walk(local_root):
                for fname in filenames:
                    local_path = os.path.join(dirpath, fname)
                    rel = os.path.relpath(local_path, local_root)
                    vol_path = os.path.join(vol_root, rel)
                    with open(local_path, "rb") as f:
                        items.append({"path": vol_path, "content_bytes": f.read()})
            return items

        data_files = _collect_files(local_data_dir, vol_data_dir)
        ckpt_files = _collect_files(local_ckpt_dir, vol_ckpt_dir)
        all_files = data_files + ckpt_files
        _fprint(f"  Collected {len(data_files)} data files + {len(ckpt_files)} checkpoint files = {len(all_files)} total")

        # Upload in batches to avoid hitting gRPC message size limits
        BATCH_SIZE = 50
        for batch_start in range(0, len(all_files), BATCH_SIZE):
            batch = all_files[batch_start : batch_start + BATCH_SIZE]
            upload_to_volume.remote(batch)
            _fprint(f"  Uploaded batch {batch_start // BATCH_SIZE + 1} ({len(batch)} files)")

        _fprint("  Upload complete.")

    # ======================================================================
    # Step 2: Run inference on GPU
    # ======================================================================
    _fprint("\n[Step 2/4] Running inference on Modal GPU …")
    result = run_inference.remote(
        data_dir_volume_path=vol_data_dir,
        checkpoint_volume_path=vol_ckpt_dir,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    _fprint(f"  Inference result: {result}")

    if result.get("status") != "success":
        _fprint("  ERROR: Inference did not complete successfully. Aborting.")
        return

    # ======================================================================
    # Step 3: Download processed data folder from volume via CLI
    # ======================================================================
    _fprint("\n[Step 3/4] Downloading processed test data from Modal volume …")
    import subprocess

    # Download into a temporary directory, then swap
    tmp_download = local_data_dir + "_modal_tmp"
    if os.path.exists(tmp_download):
        shutil.rmtree(tmp_download)
    os.makedirs(tmp_download, exist_ok=True)

    # Use `modal volume get` CLI — inherits MODAL_TOKEN_* from environment so
    # it uses the correct account without touching the global profile.
    modal_bin = shutil.which("modal") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".conda/envs/lgie/bin/modal",
    )
    # Determine the modal binary that's currently running us
    import sys
    modal_bin = os.path.join(os.path.dirname(sys.executable), "modal")
    if not os.path.isfile(modal_bin):
        modal_bin = shutil.which("modal") or "modal"

    cmd = [
        modal_bin, "volume", "get",
        VOLUME_NAME,
        "test_data/",       # source on volume
        tmp_download + "/",  # local destination
    ]
    _fprint(f"  Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        _fprint(f"  STDERR: {proc.stderr}")
        _fprint(f"  ERROR: modal volume get failed (exit code {proc.returncode})")
        # Clean up temp dir
        if os.path.exists(tmp_download):
            shutil.rmtree(tmp_download)
        return
    _fprint(f"  {proc.stdout.strip()}")

    # The CLI may create a nested test_data/ inside tmp_download.
    # Detect and flatten: if tmp_download/test_data/* exists, move contents up.
    nested = os.path.join(tmp_download, "test_data")
    if os.path.isdir(nested):
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
            dst = os.path.join(tmp_download, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        shutil.rmtree(nested)

    _fprint(f"  Downloaded to {tmp_download}.")

    # Replace local data dir
    backup_dir = local_data_dir + "_backup"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    os.rename(local_data_dir, backup_dir)
    os.rename(tmp_download, local_data_dir)
    shutil.rmtree(backup_dir)
    _fprint(f"  Replaced local {data_dir} with processed version.")

    # ======================================================================
    # Step 4: Delete the Modal volume via CLI
    # ======================================================================
    _fprint("\n[Step 4/4] Deleting temporary Modal volume …")
    try:
        del_proc = subprocess.run(
            [modal_bin, "volume", "delete", VOLUME_NAME, "--yes"],
            capture_output=True, text=True, timeout=60,
        )
        if del_proc.returncode == 0:
            _fprint(f"  Volume '{VOLUME_NAME}' deleted.")
        else:
            _fprint(f"  WARNING: Could not delete volume: {del_proc.stderr.strip()}")
            _fprint(f"  You can delete it manually: modal volume delete {VOLUME_NAME} --yes")
    except Exception as e:
        _fprint(f"  WARNING: Could not delete volume: {e}")
        _fprint(f"  You can delete it manually: modal volume delete {VOLUME_NAME} --yes")

    _fprint("\nDone! Edited images saved as 'edited_image.jpg' in each test case sub-directory.")
