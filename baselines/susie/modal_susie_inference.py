"""
Modal script for running SuSIE image-editing inference on test data.

Each test case subdirectory must contain:
    - original_image.{jpg,jpeg,png}  (the scene before manipulation)
    - edit_instruction.txt           (the textual manipulation instruction)

The script produces:
    - edited_image.jpg               (the predicted final scene) in each test case dir

Usage:
    modal run modal_susie_inference.py --data-dir dummy_test_data
"""

import modal
import os

# ============================================================================
# GPU Configuration — L40S is the cheapest 48 GB option on Modal
# ============================================================================
GPU_TYPE = "L40S"
GPU_CONFIG = f"{GPU_TYPE}:1"

# ============================================================================
# Volume Configuration
# ============================================================================
VOLUME_NAME = "susie-bridge-inference-volume"
VOLUME_MOUNT = "/data"
REMOTE_TEST_DATA_DIR = "/data/test_data"

# ============================================================================
# SuSIE Model Configuration (paper defaults)
# ============================================================================
SUSIE_MODEL_PATH = "kvablack/susie"
PRETRAINED_PATH = "runwayml/stable-diffusion-v1-5"
NUM_ITERATIONS = 5
NUM_TIMESTEPS = 50
PROMPT_W = 7.5
CONTEXT_W = 2.5
ETA = 0.0

# ============================================================================
# CUDA / Image Configuration
# ============================================================================
CUDA_VERSION = "11.8.0"
CUDA_FLAVOR = "cudnn8-devel"
CUDA_OS = "ubuntu22.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App("susie-bridge-inference")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    # ---- Core scientific stack (pinned to match working local env) ----
    .pip_install(
        "numpy==1.24.3",
        "scipy==1.10.1",
        "Pillow",
    )
    # ---- JAX with CUDA 11 support (must match jaxlib cuda version) ----
    .pip_install(
        "jax==0.4.11",
        "jaxlib==0.4.11+cuda11.cudnn86",
        extra_options="-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    )
    # ---- Flax, diffusers, transformers and friends ----
    .pip_install(
        "flax==0.7.0",
        "diffusers==0.18.2",
        "transformers==4.33.1",
        "einops==0.6.1",
        "orbax-checkpoint==0.4.4",
        "optax==0.1.5",
        "chex==0.1.7",
        "ml-dtypes==0.2.0",
        "ml_collections",
        "wandb==0.15.5",
        "huggingface-hub==0.23.5",
        "tokenizers",
    )
    # ---- PyTorch CPU (needed for PyTorch→Flax weight conversion) ----
    .pip_install(
        "torch",
        extra_options="--index-url https://download.pytorch.org/whl/cpu",
    )
    # ---- Environment variables ----
    .env(
        {
            "HF_HOME": "/data/.cache",
        }
    )
    # ---- Bundle the SuSIE package source into the image ----
    .add_local_dir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "susie", "susie"),
        remote_path="/root/susie_pkg/susie",
    )
)

# HuggingFace secret for model access
huggingface_secret = modal.Secret.from_name("huggingface-secret")


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
def run_susie_inference():
    """
    Run SuSIE iterative subgoal generation on all test cases in the volume.

    Expects:
        /data/test_data/<case_id>/original_image.{jpg,jpeg,png}
        /data/test_data/<case_id>/edit_instruction.txt

    Produces:
        /data/test_data/<case_id>/edited_image.jpg
    """
    import sys
    import time
    from pathlib import Path

    # Make the mounted SuSIE package importable
    sys.path.insert(0, "/root/susie_pkg")

    import numpy as np
    from PIL import Image

    # Set HF token
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # ---- Import SuSIE modules ----
    from susie.model import create_sample_fn
    from susie.jax_utils import initialize_compilation_cache
    import jax

    test_data_dir = Path(REMOTE_TEST_DATA_DIR)
    if not test_data_dir.exists():
        raise FileNotFoundError(
            f"Test data directory not found at {test_data_dir}. "
            "Ensure the local entrypoint uploaded it correctly."
        )

    print("=" * 70)
    print("SuSIE Batch Inference (Modal)")
    print("=" * 70)
    print(f"  JAX devices    : {jax.devices()}")
    print(f"  Model          : {SUSIE_MODEL_PATH}")
    print(f"  Pretrained     : {PRETRAINED_PATH}")
    print(f"  Iterations     : {NUM_ITERATIONS}")
    print(f"  Timesteps      : {NUM_TIMESTEPS}")
    print(f"  prompt_w       : {PROMPT_W}")
    print(f"  context_w      : {CONTEXT_W}")
    print("=" * 70)

    # ---- Discover test cases ----
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

    def find_original_image(d: Path):
        for ext in IMAGE_EXTENSIONS:
            c = d / f"original_image{ext}"
            if c.exists():
                return c
        return None

    def read_instruction(d: Path):
        txt = d / "edit_instruction.txt"
        if not txt.exists():
            return None
        with open(txt) as f:
            for line in f:
                line = line.strip()
                if line and not line.lower().startswith("confidence"):
                    return line
        return None

    cases = []
    for entry in test_data_dir.iterdir():
        if not entry.is_dir():
            continue
        # Accept any case-dir name (e.g. combined-staging "<task>__ep000"),
        # not just pure-integer names.
        case_id = entry.name
        img = find_original_image(entry)
        instr = read_instruction(entry)
        if img is None or instr is None:
            print(f"  [SKIP] Test case {case_id}: missing image or instruction")
            continue
        cases.append({"id": case_id, "dir": entry, "image": img, "instruction": instr})

    cases.sort(key=lambda c: c["id"])

    # ---- Check which cases are already done (resume support) ----
    pending_cases = []
    already_done = 0
    for case in cases:
        output_path = case["dir"] / "edited_image.jpg"
        if output_path.exists():
            already_done += 1
        else:
            pending_cases.append(case)

    print(f"\nFound {len(cases)} test case(s) total.")
    if already_done > 0:
        print(f"  Already processed (from previous run): {already_done}")
    print(f"  Remaining to process: {len(pending_cases)}\n")

    if not pending_cases:
        print("All test cases already processed — nothing to do.")
        volume.commit()
        return {
            "status": "success",
            "processed": 0,
            "already_done": already_done,
            "failed": 0,
            "total": len(cases),
        }

    # ---- Load model once ----
    print("Loading SuSIE model (one-time)...")
    t_model = time.time()
    initialize_compilation_cache()
    sample_fn = create_sample_fn(
        path=SUSIE_MODEL_PATH,
        num_timesteps=NUM_TIMESTEPS,
        prompt_w=PROMPT_W,
        context_w=CONTEXT_W,
        eta=ETA,
        pretrained_path=PRETRAINED_PATH,
    )
    print(f"Model loaded in {time.time() - t_model:.1f}s\n")

    # ---- Run inference (only on pending cases) ----
    total_pending = len(pending_cases)
    succeeded = 0
    failed_ids = []
    COMMIT_EVERY = 10  # commit to volume every N cases to survive container kills
    since_last_commit = 0

    for idx, case in enumerate(pending_cases, 1):
        case_id = case["id"]
        output_path = case["dir"] / "edited_image.jpg"
        print(f"[{idx}/{total_pending}] Test case {case_id}: \"{case['instruction']}\"")

        t0 = time.time()
        try:
            img = Image.open(case["image"]).convert("RGB")
            if img.size != (256, 256):
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
            current_image = np.array(img)

            for it in range(1, NUM_ITERATIONS + 1):
                current_image = sample_fn(current_image, case["instruction"])

            Image.fromarray(current_image).save(str(output_path))
            elapsed = time.time() - t0
            print(f"         -> Saved {output_path}  ({elapsed:.1f}s)")
            succeeded += 1
            since_last_commit += 1

            # Periodically commit so progress survives container restarts
            if since_last_commit >= COMMIT_EVERY:
                print(f"         -> Committing volume (checkpoint)...")
                volume.commit()
                since_last_commit = 0

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"         -> FAILED after {elapsed:.1f}s: {exc}")
            import traceback
            traceback.print_exc()
            failed_ids.append(case_id)

    # ---- Final commit ----
    print(f"\n{'=' * 70}")
    print(f"Inference complete.")
    print(f"  Already done (previous run) : {already_done}")
    print(f"  Processed (this run)        : {succeeded}")
    print(f"  Failed (this run)           : {len(failed_ids)}")
    if failed_ids:
        print(f"  Failed IDs: {failed_ids}")
    print(f"  Total test cases            : {len(cases)}")
    print(f"{'=' * 70}")
    print("Committing volume (final)...")
    volume.commit()

    return {
        "status": "success",
        "processed": succeeded,
        "already_done": already_done,
        "failed": len(failed_ids),
        "total": len(cases),
    }


# ============================================================================
# Local Entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    data_dir: str = "dummy_test_data",
):
    """
    Run SuSIE inference on a local test data folder via Modal.

    Steps:
        1. Upload test data folder to Modal volume
        2. Run inference on Modal GPU
        3. Download processed test data (with edited_image.jpg) back locally
        4. Delete the Modal volume

    Args:
        data_dir: Path to test data directory containing numbered sub-folders.
    """
    import shutil
    import time

    data_dir_abs = os.path.abspath(data_dir)

    if not os.path.isdir(data_dir_abs):
        raise FileNotFoundError(f"Test data directory not found: {data_dir_abs}")

    # Count test cases
    test_cases = [
        d for d in os.listdir(data_dir_abs)
        if os.path.isdir(os.path.join(data_dir_abs, d))
    ]
    print("=" * 70)
    print("SuSIE Image Editing Inference (Modal)")
    print("=" * 70)
    print(f"  Test data dir : {data_dir_abs}")
    print(f"  Test cases    : {len(test_cases)}")
    print(f"  Volume        : {VOLUME_NAME}")
    print(f"  GPU           : {GPU_CONFIG}")
    print(f"  Model         : {SUSIE_MODEL_PATH}")
    print(f"  Iterations    : {NUM_ITERATIONS}")
    print("=" * 70)

    # ==================================================================
    # Step 1: Upload test data to Modal volume (SDK — auth-safe)
    #         Skip if data is already present (container restart / resume)
    # ==================================================================
    print(f"\n[Step 1/4] Checking Modal volume for existing test data...")
    t0 = time.time()

    data_already_on_volume = False
    try:
        existing = volume.listdir("test_data")
        if len(existing) > 0:
            data_already_on_volume = True
    except Exception:
        # listdir throws if the path doesn't exist — that's fine
        pass

    if data_already_on_volume:
        elapsed = time.time() - t0
        print(f"  Test data already present on volume — skipping upload (resume mode)")
        print(f"  ({elapsed:.1f}s)")
    else:
        print(f"  No existing data found — uploading to volume ({VOLUME_NAME})...")
        try:
            with volume.batch_upload(force=True) as batch:
                batch.put_directory(data_dir_abs, "test_data")
            elapsed = time.time() - t0
            print(f"  Upload complete ({elapsed:.1f}s)")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  ERROR uploading test data ({elapsed:.1f}s): {exc}")
            raise RuntimeError("Failed to upload test data to Modal volume.") from exc

    # ==================================================================
    # Step 2: Run inference on Modal GPU
    # ==================================================================
    print(f"\n[Step 2/4] Running SuSIE inference on Modal ({GPU_CONFIG})...")
    t0 = time.time()
    inference_result = run_susie_inference.remote()
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

    backup_dir = data_dir_abs + ".backup"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.move(data_dir_abs, backup_dir)
    print(f"  Backed up local test data to {backup_dir}")

    import subprocess
    import sys
    import tempfile

    try:
        # Use 'modal volume get' CLI for fast bulk download
        # (SDK read_file_into_fileobj is too slow file-by-file)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "-m", "modal", "volume", "get", "--force",
                 VOLUME_NAME, "test_data", tmpdir],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"modal volume get failed (exit {result.returncode}): "
                    f"{result.stderr.strip()}"
                )

            # modal volume get puts contents under <tmpdir>/test_data/
            downloaded_path = os.path.join(tmpdir, "test_data")
            if not os.path.isdir(downloaded_path):
                raise RuntimeError(
                    f"Expected directory {downloaded_path} not found after download"
                )

            # Move to final location
            if os.path.exists(data_dir_abs):
                shutil.rmtree(data_dir_abs)
            shutil.move(downloaded_path, data_dir_abs)

        elapsed = time.time() - t0
        edited_count = sum(
            1 for d in os.listdir(data_dir_abs)
            if os.path.isdir(os.path.join(data_dir_abs, d))
            and os.path.exists(os.path.join(data_dir_abs, d, "edited_image.jpg"))
        )
        print(f"  Download complete ({elapsed:.1f}s)")
        print(f"  Test cases with edited_image.jpg: {edited_count}")

        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
            print(f"  Removed backup")

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ERROR downloading ({elapsed:.1f}s): {exc}")
        if os.path.exists(backup_dir) and not os.path.exists(data_dir_abs):
            shutil.move(backup_dir, data_dir_abs)
            print(f"  Restored backup to {data_dir_abs}")
        raise

    # ==================================================================
    # Step 4: Delete the Modal volume (SDK — auth-safe)
    # ==================================================================
    print(f"\n[Step 4/4] Deleting Modal volume ({VOLUME_NAME})...")
    t0 = time.time()
    try:
        modal.Volume.delete(VOLUME_NAME)
        elapsed = time.time() - t0
        print(f"  Volume deleted ({elapsed:.1f}s)")
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  WARNING: Volume deletion failed ({elapsed:.1f}s): {exc}")
        print(f"  Manually delete: modal volume delete --yes {VOLUME_NAME}")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")
    print(f"  Already done (prev run) : {inference_result.get('already_done', '?')}")
    print(f"  Processed (this run)    : {inference_result.get('processed', '?')}")
    print(f"  Failed                  : {inference_result.get('failed', '?')}")
    print(f"  Total test cases        : {inference_result.get('total', '?')}")
    print(f"  Output    : {data_dir_abs}")
    print(f"  Each test case directory now contains edited_image.jpg")
    print(f"{'=' * 70}")

    return inference_result
