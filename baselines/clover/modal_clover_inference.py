"""
Modal script for running CLOVER Visual Planner inference on test data.

Each test case subdirectory must contain:
    - original_image.{jpg,jpeg,png}  (the scene before manipulation)
    - edit_instruction.txt           (the natural-language manipulation instruction)

The script produces:
    - edited_image.jpg               (the predicted final scene) in each test case dir

Usage:
    modal run modal_clover_inference.py --data-dir dummy_test_data
"""

import modal
import os

# ============================================================================
# GPU Configuration — L40S (48 GB VRAM, cheapest 48 GB option)
# ============================================================================
GPU_TYPE = "T4"
GPU_CONFIG = f"{GPU_TYPE}:1"

# ============================================================================
# Volume Configuration
# ============================================================================
VOLUME_NAME = "clover-calvin-inference-volume"
VOLUME_MOUNT = "/data"
REMOTE_TEST_DATA_DIR = "/data/test_data"
REMOTE_CODE_DIR = "/data/code"
REMOTE_CHECKPOINT_DIR = "/data/checkpoint"
PROGRESS_FILE = "/data/_clover_progress.json"  # tracks completed case numbers

# ============================================================================
# Model Configuration
# ============================================================================
HF_REPO_ID = "qwbu/CLOVER"
CHECKPOINT_FILENAME = "visual_planner.pt"
NUM_CHAINS = 10
CONVERGENCE_THRESHOLD = 0.001

# ============================================================================
# CUDA / Image Configuration
# ============================================================================
CUDA_VERSION = "12.1.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu22.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App("clover-calvin-inference")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    # System dependencies
    .run_commands(
        "apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0"
    )
    # Install PyTorch 2.4.1+cu121
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Install CLOVER dependencies
    .pip_install(
        "huggingface-hub",
        "transformers==4.34.0",
        "diffusers==0.23.1",
        "accelerate==0.23.0",
        "einops==0.7.0",
        "einops-exts==0.0.4",
        "ema-pytorch==0.2.3",
        "rotary-embedding-torch==0.5.3",
        "tqdm==4.66.1",
        "opencv-python==4.9.0.80",
        "imageio[ffmpeg]",
        "matplotlib==3.7.5",
        "scipy==1.10.1",
        "pynvml==11.5.0",
        "pytorch-fid==0.3.0",
        "peft==0.6.2",
        "timm",
        "pillow",
        "numpy",
    )
    # Environment variables
    .env(
        {
            "HF_HOME": "/data/.cache",
        }
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
    memory=65536,  # 64 GB RAM
)
def run_clover_inference():
    """
    Run CLOVER Visual Planner inference on all test cases in the volume.

    Expects the volume to contain:
        /data/test_data/<case_id>/original_image.{jpg,jpeg,png}
        /data/test_data/<case_id>/edit_instruction.txt
        /data/code/  (CLOVER codebase files)

    Produces:
        /data/test_data/<case_id>/edited_image.jpg
    """
    import json
    import sys
    import glob
    import torch
    import numpy as np
    from PIL import Image
    from pathlib import Path

    # Set HF token
    hf_token = os.environ.get("HF_TOKEN", "")
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    test_data_dir = Path(REMOTE_TEST_DATA_DIR)
    code_dir = Path(REMOTE_CODE_DIR)

    # Validate
    if not test_data_dir.exists():
        raise FileNotFoundError(f"Test data directory not found at {test_data_dir}")
    if not code_dir.exists():
        raise FileNotFoundError(f"Code directory not found at {code_dir}")

    # Add code paths to sys.path so imports work
    sys.path.insert(0, str(code_dir))
    sys.path.insert(0, str(code_dir / "CLOVER" / "visual_planner"))
    sys.path.insert(0, str(code_dir / "CLOVER"))

    # Download checkpoint from HuggingFace if not already cached
    checkpoint_dir = Path(REMOTE_CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILENAME

    if not checkpoint_path.exists():
        print(f"Downloading checkpoint from HuggingFace ({HF_REPO_ID})...")
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            HF_REPO_ID,
            CHECKPOINT_FILENAME,
            local_dir=str(checkpoint_dir),
            token=hf_token if hf_token else None,
        )
        print(f"  Checkpoint downloaded to: {downloaded}")
        # hf_hub_download may place it in a subdirectory; ensure it's at expected path
        if not checkpoint_path.exists():
            import shutil
            shutil.move(downloaded, str(checkpoint_path))
    else:
        print(f"Checkpoint already cached at {checkpoint_path}")

    # Now import our inference code
    from clover_visual_planner import CLOVERVisualPlannerInference

    # ---- Discover test cases ----
    test_case_dirs = []
    for entry in os.listdir(str(test_data_dir)):
        entry_path = test_data_dir / entry
        if entry_path.is_dir():
            # Accept any case-dir name (e.g. combined-staging "<task>__ep000"),
            # not just pure-integer names. case_num is the dir name (string);
            # the progress manifest below stores these names directly.
            test_case_dirs.append((entry, entry_path))
    test_case_dirs.sort(key=lambda x: x[0])

    print("=" * 60)
    print("CLOVER Visual Planner — Modal Inference")
    print("=" * 60)
    print(f"Test cases     : {len(test_case_dirs)}")
    print(f"Checkpoint     : {checkpoint_path}")
    print(f"Device         : cuda")
    print(f"Max chains     : {NUM_CHAINS}")
    print(f"Conv. threshold: {CONVERGENCE_THRESHOLD}")
    print("=" * 60)

    # ---- Initialize planner ----
    print("\nInitializing CLOVER Visual Planner...\n")
    planner = CLOVERVisualPlannerInference(
        checkpoint_path=str(checkpoint_path),
        device="cuda",
        with_depth=True,
    )

    # ---- Load progress manifest (resume support) ----
    # We track completed case numbers in a JSON file on the volume
    # rather than checking for edited_image.jpg, because the uploaded
    # test data may already contain stale edited_image.jpg files from
    # a prior experiment.
    progress_path = Path(PROGRESS_FILE)
    completed_cases: set = set()
    if progress_path.exists():
        try:
            completed_cases = set(json.loads(progress_path.read_text()))
        except (json.JSONDecodeError, ValueError):
            completed_cases = set()

    already_done = 0
    pending_dirs = []
    for case_num, case_dir in test_case_dirs:
        if case_num in completed_cases:
            already_done += 1
        else:
            pending_dirs.append((case_num, case_dir))

    if already_done > 0:
        print(f"\n  RESUME DETECTED: {already_done} test case(s) already processed")
        print(f"  Remaining to process: {len(pending_dirs)}")

    if not pending_dirs:
        print("\n  All test cases already processed — nothing to do.")
        volume.commit()
        return {
            "status": "success",
            "processed": 0,
            "already_done": already_done,
            "skipped": 0,
            "failed": 0,
            "total": len(test_case_dirs),
        }

    # ---- Process remaining test cases ----
    processed = 0
    skipped = 0
    failed = 0

    for idx, (case_num, case_dir) in enumerate(pending_dirs):
        print(f"\n{'='*60}")
        print(f"Test case {case_num} ({idx+1}/{len(pending_dirs)}, "
              f"overall {already_done + idx + 1}/{len(test_case_dirs)})")
        print(f"{'='*60}")

        # Find original image
        image_path = None
        for ext in ["png", "jpg", "jpeg"]:
            candidates = list(case_dir.glob(f"original_image.{ext}"))
            if candidates:
                image_path = candidates[0]
                break

        if image_path is None:
            print(f"  SKIP: No original_image found")
            skipped += 1
            continue

        # Read instruction
        instruction_path = case_dir / "edit_instruction.txt"
        if not instruction_path.exists():
            print(f"  SKIP: No edit_instruction.txt found")
            skipped += 1
            continue

        with open(str(instruction_path), "r") as f:
            instruction = f.readline().strip()
        if not instruction:
            print(f"  SKIP: Empty instruction")
            skipped += 1
            continue

        print(f"  Image      : {image_path.name}")
        print(f"  Instruction: {instruction}")

        try:
            # Preprocess
            condition_image = planner.preprocess_image(str(image_path))

            # Generate chained visual plan
            generated_frames, success, num_attempts, chain_labels = \
                planner.generate_chained_visual_plan(
                    condition_image=condition_image,
                    text_instruction=instruction,
                    num_chains=NUM_CHAINS,
                    convergence_threshold=CONVERGENCE_THRESHOLD,
                    auto_stop=True,
                )

            # Extract the final frame
            final_frame = generated_frames[0, -1]  # (3, H, W)
            final_frame_np = (final_frame.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # Save as edited_image.jpg
            output_path = case_dir / "edited_image.jpg"
            Image.fromarray(final_frame_np).save(str(output_path), quality=95)

            total_frames = generated_frames.shape[1]
            print(f"  Result     : {'SUCCESS' if success else 'FALLBACK'} "
                  f"({total_frames} frames, {num_attempts} attempt(s))")
            print(f"  Saved      : {output_path}")
            processed += 1

            # Record this case in the progress manifest and commit
            # so progress survives container restarts.
            completed_cases.add(case_num)
            progress_path.write_text(json.dumps(sorted(completed_cases)))
            volume.commit()

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # ---- Final commit ----
    print(f"\n{'='*60}")
    print(f"Inference complete.")
    print(f"  Already done : {already_done}")
    print(f"  Newly processed: {processed}")
    print(f"  Skipped      : {skipped}")
    print(f"  Failed       : {failed}")
    print(f"{'='*60}")
    print("Committing volume...")
    volume.commit()

    return {
        "status": "success" if failed == 0 else "partial_failure",
        "processed": processed,
        "already_done": already_done,
        "skipped": skipped,
        "failed": failed,
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
    Run CLOVER Visual Planner inference on a local test data folder via Modal.

    Steps:
        1. Upload test data + codebase to Modal volume
        2. Run inference on Modal GPU (L40S)
        3. Download processed test data (with edited_image.jpg) back locally
        4. Delete the Modal volume

    Args:
        data_dir: Path to test data directory (each subdirectory is a test case
                  containing original_image.{jpg,jpeg,png} and edit_instruction.txt)
    """
    import subprocess
    import sys
    import shutil
    import time

    data_dir_abs = os.path.abspath(data_dir)
    data_dir_parent = os.path.dirname(data_dir_abs)
    data_dir_basename = os.path.basename(data_dir_abs)
    # Project root is where this script lives
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Validate local path
    if not os.path.isdir(data_dir_abs):
        raise FileNotFoundError(f"Test data directory not found: {data_dir_abs}")

    # Count test cases
    test_cases = [
        d for d in os.listdir(data_dir_abs)
        if os.path.isdir(os.path.join(data_dir_abs, d))
    ]

    print("=" * 70)
    print("CLOVER Visual Planner — Modal Cloud Inference")
    print("=" * 70)
    print(f"  Test data dir:   {data_dir_abs}")
    print(f"  Test cases:      {len(test_cases)}")
    print(f"  Volume:          {VOLUME_NAME}")
    print(f"  GPU:             {GPU_CONFIG}")
    print(f"  Checkpoint:      {HF_REPO_ID}/{CHECKPOINT_FILENAME}")
    print("=" * 70)

    modal_cmd_base = [sys.executable, "-m", "modal"]

    # ==================================================================
    # Helper: check if a path already exists on the Modal volume
    # ==================================================================
    def volume_path_exists(remote_path: str) -> bool:
        """Return True if *remote_path* already exists on the volume."""
        probe = subprocess.run(
            modal_cmd_base + ["volume", "ls", VOLUME_NAME, remote_path],
            capture_output=True,
            text=True,
        )
        return probe.returncode == 0

    # ==================================================================
    # Step 1: Upload test data to Modal volume (skip if already present)
    # ==================================================================
    is_resume = volume_path_exists("test_data")

    if is_resume:
        print(f"\n[Step 1/5] Test data already on volume — skipping upload (resume)")
    else:
        print(f"\n[Step 1/5] Uploading test data to Modal volume...")
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
    # Step 2: Upload codebase to Modal volume (skip if already present)
    # ==================================================================
    code_exists = volume_path_exists("code")

    if code_exists:
        print(f"\n[Step 2/5] Codebase already on volume — skipping upload (resume)")
    else:
        print(f"\n[Step 2/5] Uploading codebase to Modal volume...")
        t0 = time.time()

        # We need to upload:
        #   clover_visual_planner.py  -> /data/code/clover_visual_planner.py
        #   run_inference.py          -> /data/code/run_inference.py
        #   CLOVER/                   -> /data/code/CLOVER/
        # Upload clover_visual_planner.py
        result = subprocess.run(
            modal_cmd_base + [
                "volume", "put", "--force", VOLUME_NAME,
                os.path.join(project_root, "clover_visual_planner.py"),
                "code/clover_visual_planner.py",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR uploading clover_visual_planner.py: {result.stderr.strip()}")
            raise RuntimeError("Failed to upload clover_visual_planner.py")

        # Upload run_inference.py
        result = subprocess.run(
            modal_cmd_base + [
                "volume", "put", "--force", VOLUME_NAME,
                os.path.join(project_root, "run_inference.py"),
                "code/run_inference.py",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR uploading run_inference.py: {result.stderr.strip()}")
            raise RuntimeError("Failed to upload run_inference.py")

        # Upload CLOVER directory
        result = subprocess.run(
            modal_cmd_base + [
                "volume", "put", "--force", VOLUME_NAME,
                os.path.join(project_root, "CLOVER"),
                "code/CLOVER",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR uploading CLOVER/: {result.stderr.strip()}")
            raise RuntimeError("Failed to upload CLOVER directory")

        elapsed = time.time() - t0
        print(f"  Upload complete ({elapsed:.1f}s)")

    # ==================================================================
    # Step 3: Run inference on Modal GPU
    # ==================================================================
    print(f"\n[Step 3/5] Running inference on Modal ({GPU_CONFIG})...")
    t0 = time.time()
    inference_result = run_clover_inference.remote()
    elapsed = time.time() - t0
    print(f"\n  Inference finished ({elapsed:.1f}s)")
    print(f"  Result: {inference_result}")

    newly = inference_result.get("processed", 0)
    done_before = inference_result.get("already_done", 0)
    if newly == 0 and done_before == 0 and inference_result.get("total", 0) > 0:
        raise RuntimeError(f"No test cases were processed: {inference_result}")

    # ==================================================================
    # Step 4: Download processed test data back locally
    # ==================================================================
    print(f"\n[Step 4/5] Downloading processed test data from Modal volume...")
    t0 = time.time()

    # Back up local test data
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
            timeout=1800,
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
            raise RuntimeError("Failed to download processed test data.")

        # modal volume get downloads to <parent>/test_data
        # Rename to the original directory name if different
        downloaded_path = os.path.join(data_dir_parent, "test_data")
        if os.path.exists(downloaded_path) and downloaded_path != data_dir_abs:
            if os.path.exists(data_dir_abs):
                shutil.rmtree(data_dir_abs)
            shutil.move(downloaded_path, data_dir_abs)
            print(f"  Renamed downloaded test_data -> {data_dir_basename}")
        elif not os.path.exists(data_dir_abs):
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
    # Step 5: Delete the Modal volume
    # ==================================================================
    print(f"\n[Step 5/5] Deleting Modal volume ({VOLUME_NAME})...")
    t0 = time.time()
    result = subprocess.run(
        modal_cmd_base + ["volume", "delete", "--yes", VOLUME_NAME],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  WARNING: Volume deletion failed (exit {result.returncode}, {elapsed:.1f}s)")
        if result.stderr:
            print(f"  stderr: {result.stderr.strip()}")
        print(f"  You can manually delete it with: modal volume delete --yes {VOLUME_NAME}")
    else:
        print(f"  Volume deleted ({elapsed:.1f}s)")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    print(f"  Already done:    {inference_result.get('already_done', '?')} test cases")
    print(f"  Newly processed: {inference_result.get('processed', '?')} test cases")
    print(f"  Skipped:         {inference_result.get('skipped', '?')} test cases")
    print(f"  Failed:          {inference_result.get('failed', '?')} test cases")
    print(f"  Output:    {data_dir_abs}")
    print(f"  Each test case directory now contains edited_image.jpg")
    print(f"{'='*70}")

    return inference_result
