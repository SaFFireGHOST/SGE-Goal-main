#!/usr/bin/env python3
"""
modal_lgie_eval_partitioned.py — Partitioned LGIE evaluator for max parallelism.

Spawns N_WORKERS parallel GPU containers, each processing a disjoint subset
of test cases for a single phase.  All workers write to the SAME volume
(non-overlapping case directories) with the SAME result suffix.

Env vars:
    LGIE_APP_NAME       — Modal app name (must be unique per launch)
    LGIE_VOLUME_NAME    — Data volume name (test cases live here)
    N_WORKERS           — Number of parallel GPU workers (default: 5)
    EVAL_PHASE          — Phase number to run (1/2/3/4)
    RESULT_SUFFIX       — File suffix (e.g. "p1" → evaluation_results_p1.json)
    LGIE_SHORT_NAME     — Dataset short name
    LGIE_REL_PATH       — Dataset relative path for local sync

Usage:
    LGIE_APP_NAME=eval-i1-p1 LGIE_VOLUME_NAME=data-i1 N_WORKERS=5 \\
        EVAL_PHASE=1 RESULT_SUFFIX=p1 \\
        modal run -d modal_lgie_eval_partitioned.py
"""

import modal
import os
import sys
import time
from pathlib import Path

# ============================================================================
# Configuration from env
# ============================================================================
APP_NAME = os.environ.get("LGIE_APP_NAME", "lgie-eval-part")
VOLUME_NAME = os.environ.get("LGIE_VOLUME_NAME", "lgie-data")
N_WORKERS = int(os.environ.get("N_WORKERS", "5"))
EVAL_PHASE = int(os.environ.get("EVAL_PHASE", "1"))
RESULT_SUFFIX = os.environ.get("RESULT_SUFFIX", "p1")
SHORT_NAME = os.environ.get("LGIE_SHORT_NAME", "unknown")
REL_PATH = os.environ.get("LGIE_REL_PATH", "")
COMMIT_INTERVAL = 25

# Mount points
WEIGHTS_MOUNT = "/weights"
DATA_MOUNT = "/eval-data"
EVAL_DATA_MOUNT = "/eval-images"
APP_MOUNT = "/app"

CUDA_VERSION = "12.4.0"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu22.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# ============================================================================
# Modal App + Volumes
# ============================================================================
app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
weights_volume = modal.Volume.from_name("lgie-eval-weights", create_if_missing=True)
eval_data_volume = modal.Volume.from_name("lgie-eval-data", create_if_missing=True)
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

# ============================================================================
# Container Image (same as modal_lgie_eval.py)
# ============================================================================
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    .apt_install(
        "git", "ffmpeg", "libsm6", "libxext6",
        "libgl1-mesa-glx", "libglib2.0-0",
    )
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.40.1",
        "accelerate==1.11.0",
        "huggingface-hub==0.36.0",
        "hf_transfer",
        "opencv-python==4.10.0.84",
        "opencv-python-headless",
        "pillow==12.0.0",
        "imageio==2.37.0",
        "imageio-ffmpeg",
        "av",
        "decord==0.6.0",
        "numpy==1.26.4",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "timm==0.4.12",
        "natsort==8.4.0",
        "einops==0.4.1",
        "kornia==0.8.1",
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
        "progressbar2",
        "gdown",
        "gitpython",
        "hickle==5.0.3",
        "tensorboard",
        "git+https://github.com/cheind/py-thin-plate-spline",
        "sentencepiece",
        "tqdm",
        "requests",
        "pyyaml",
        "packaging",
        "wheel",
        "setuptools",
        "ninja",
        "yacs==0.1.8",
        "bitsandbytes>=0.43.0",
        "google-genai>=1.0.0",
        "matplotlib",
        "shapely",
        "ftfy",
        "regex",
    )
    .run_commands(
        "pip install flash-attn==2.7.3 --no-build-isolation 2>&1 || "
        "pip install flash-attn --no-build-isolation || "
        "echo 'flash-attn installation failed — will use eager attention'",
    )
    .env({
        "HF_HOME": f"{WEIGHTS_MOUNT}/.cache/huggingface",
        "TRANSFORMERS_CACHE": f"{WEIGHTS_MOUNT}/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path=APP_MOUNT,
        ignore=[
            "*.git*", ".git", ".git/", "**/.git", "**/.git/**",
            "__pycache__", "*.pyc", "*.pyo", ".env", ".venv",
            "*.log", ".DS_Store", "*.egg-info", ".pytest_cache",
            "sam_vit_h_4b8939.pth",
            "XMem/saves/*.pth", "XMem/scripts/saves/*.pth",
            "loftr_weights/*.ckpt",
            "lgie-eval-data/", "lgie-eval-data-backup/",
            "sge-goal-evaluator-human-alignment-test/",
            "human-alignment-test-data/", "data/",
            "scripts-backup/", "enhanced-scripts-backup/",
            "enhanced-scripts-backup-2/",
            "R2_Output/", "R3_Output/",
            "object_change_detection_logs/",
            "FINAL-EVAL-RUNS/", "FINAL-EVAL-RUNS-*/",
            "EXTRA-EVAL-RUNS/",
            "CALVIN-ABLATION-RUNS/",
            "chat.json", "test_pack3.json", "aggregated-eval-results/",
            "*.packed.json", ".xpacked_*.json", ".cpacked_*.json",
            "FINAL-EVAL-RUNS-SGE-GOAL-EVALUATOR-ORIG-BACKUP-DO-NOT-DELETE/",
            "eval-logs/",
            "launch_all.py", "resume_all.py", "volume_sync.py",
            "volume_sync_resume.py", "setup_shared_volumes.py",
            "setup_account.py", "modal_eval_runner.py",
            "create_data_folder.py", "run_with_sync.py",
            "launch_dual.sh", "launch_detached.py", "launch_fast.sh",
            "modal_sync_watcher.py", "sge_goal_eval_aggregator.py",
            "test_7b_failures.py", "test_awq.py", "upload_weights.py",
            "verify_integrity.py", "watchdog_429.py", "watchdog_gemini.py",
            "MODAL_README.md", "environment.yml", "nohup.out",
            "FINAL-EVAL-RUNS-SGE-GOAL-EVALUATOR/",
            "watchdog_state.json", "watchdog_gemini.pid",
            "watchdog_429.pid", "watchdog_429.log",
            "watchdog_gemini.log", "*.pid",
        ],
    )
)


# ============================================================================
# Symlink setup (same as original)
# ============================================================================
def _setup_symlinks():
    """Create symlinks so R-scripts find weights + eval-data."""
    sam_src = Path(f"{WEIGHTS_MOUNT}/sam_vit_h_4b8939.pth")
    sam_dst = Path(f"{APP_MOUNT}/sam_vit_h_4b8939.pth")
    if sam_src.exists() and not sam_dst.exists():
        os.symlink(sam_src, sam_dst)

    xmem_dst_dir = Path(f"{APP_MOUNT}/XMem/saves")
    xmem_dst_dir.mkdir(parents=True, exist_ok=True)
    xmem_src_dir = Path(f"{WEIGHTS_MOUNT}/XMem")
    if xmem_src_dir.exists():
        for pth in xmem_src_dir.glob("*.pth"):
            dst = xmem_dst_dir / pth.name
            if not dst.exists():
                os.symlink(pth, dst)

    loftr_dst_dir = Path(f"{APP_MOUNT}/loftr_weights")
    loftr_dst_dir.mkdir(parents=True, exist_ok=True)
    loftr_src_dir = Path(f"{WEIGHTS_MOUNT}/loftr_weights")
    if loftr_src_dir.exists():
        for ckpt in loftr_src_dir.glob("*.ckpt"):
            dst = loftr_dst_dir / ckpt.name
            if not dst.exists():
                os.symlink(ckpt, dst)

    data_dst = Path(f"{APP_MOUNT}/lgie-eval-data")
    data_src = Path(f"{EVAL_DATA_MOUNT}/lgie-eval-data")
    if data_src.exists() and not data_dst.exists():
        os.symlink(data_src, data_dst)
    elif not data_dst.exists() and Path(EVAL_DATA_MOUNT).exists():
        if any(Path(EVAL_DATA_MOUNT).glob("original_image/*")):
            os.symlink(Path(EVAL_DATA_MOUNT), data_dst)


# ============================================================================
# GPU Function: evaluate a partition of test cases for a single phase
# ============================================================================
@app.function(
    image=image,
    gpu="l40s",
    secrets=[huggingface_secret],
    volumes={
        WEIGHTS_MOUNT: weights_volume,
        DATA_MOUNT: data_volume,
        EVAL_DATA_MOUNT: eval_data_volume,
    },
    timeout=86400,
    memory=65536,
)
def evaluate_partition(
    worker_idx: int,
    n_workers: int,
    phase: int,
    result_suffix: str,
) -> str:
    """Evaluate a partition of test cases for a single phase."""
    _setup_symlinks()

    if APP_MOUNT not in sys.path:
        sys.path.insert(0, APP_MOUNT)
    os.chdir(APP_MOUNT)

    from lgie_evaluator import list_test_cases, evaluate_all_phased, _ensure_paths
    _ensure_paths()

    # List and partition cases
    all_cases = list_test_cases(DATA_MOUNT)
    total_cases = len(all_cases)
    chunk = (total_cases + n_workers - 1) // n_workers
    start = worker_idx * chunk
    end = min(start + chunk, total_cases)
    my_cases = all_cases[start:end]

    if not my_cases:
        print(f"[Worker {worker_idx}] No cases assigned — exiting")
        return f"Worker {worker_idx}: no cases"

    output_filename = f"evaluation_results_{result_suffix}.json"
    print(f"[Worker {worker_idx}/{n_workers}] Phase {phase}")
    print(f"  Cases: {my_cases[0]}-{my_cases[-1]} ({len(my_cases)} of {total_cases})")
    print(f"  Output: {output_filename}")
    sys.stdout.flush()

    # Commit callback
    _counter = [0]

    def on_case_complete(case_num, done, total):
        _counter[0] += 1
        if _counter[0] % COMMIT_INTERVAL == 0:
            data_volume.commit()
            print(f"  [Worker {worker_idx}] Committed ({done}/{total} in phase, "
                  f"{_counter[0]} total)")
            sys.stdout.flush()

    def on_phase_complete(phase_num):
        data_volume.commit()
        print(f"  [Worker {worker_idx}] ✓ Phase {phase_num} commit")
        sys.stdout.flush()

    try:
        evaluate_all_phased(
            data_dir=DATA_MOUNT,
            cases=my_cases,
            skip_existing=True,
            phases=[phase],
            output_filename=output_filename,
            on_case_complete=on_case_complete,
            on_phase_complete=on_phase_complete,
        )
    except Exception as e:
        import traceback
        print(f"  [Worker {worker_idx}] ERROR: {e}")
        traceback.print_exc()
        data_volume.commit()
        return f"Worker {worker_idx}: ERROR {e}"

    data_volume.commit()
    print(f"  [Worker {worker_idx}] ✓ Complete — {_counter[0]} cases processed")
    return f"Worker {worker_idx}: done ({_counter[0]} cases)"


# ============================================================================
# Upload helper (same as original)
# ============================================================================
def _do_upload(local_data_dir: str):
    """Upload local data folder to the Modal volume."""
    from modal.volume import FileEntryType
    t0 = time.time()

    if not os.path.isdir(local_data_dir):
        print(f"ERROR: local data directory not found: {local_data_dir}")
        sys.exit(1)

    # Check if data already exists
    try:
        existing = [
            e for e in data_volume.listdir("/")
            if e.type == FileEntryType.DIRECTORY
            and e.path.strip("/").isdigit()
        ]
        if len(existing) > 0:
            print(f"Volume already has {len(existing)} test case folders — skipping upload")
            return
    except Exception:
        pass

    entries = os.listdir(local_data_dir)
    dirs = sorted([e for e in entries if os.path.isdir(
        os.path.join(local_data_dir, e)) and e.isdigit()], key=lambda x: int(x))
    files = [e for e in entries if os.path.isfile(
        os.path.join(local_data_dir, e))]
    print(f"Uploading {len(dirs)} test-case folders to volume '{VOLUME_NAME}'...")

    with data_volume.batch_upload(force=True) as batch:
        for fname in files:
            batch.put_file(os.path.join(local_data_dir, fname), f"/{fname}")
        for d in dirs:
            batch.put_directory(os.path.join(local_data_dir, d), f"/{d}")

    elapsed = time.time() - t0
    print(f"Upload complete — {len(dirs)} folders in {elapsed:.1f}s")


# ============================================================================
# Local entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    upload_dir: str = "",
):
    """Spawn N_WORKERS parallel GPU workers for a single phase."""
    # Optional upload step
    if upload_dir:
        _do_upload(upload_dir)

    print(f"\n{'='*60}")
    print(f"Spawning {N_WORKERS} workers for Phase {EVAL_PHASE}")
    print(f"  App: {APP_NAME}")
    print(f"  Volume: {VOLUME_NAME}")
    print(f"  Suffix: {RESULT_SUFFIX}")
    print(f"{'='*60}\n")

    handles = []
    for i in range(N_WORKERS):
        h = evaluate_partition.spawn(
            worker_idx=i,
            n_workers=N_WORKERS,
            phase=EVAL_PHASE,
            result_suffix=RESULT_SUFFIX,
        )
        handles.append(h)
        print(f"  ✓ Spawned worker {i}/{N_WORKERS}")

    print(f"\nAll {N_WORKERS} workers spawned — running in detached mode.")
    print(f"Workers will continue processing on Modal infrastructure.")

    # In detached mode, don't block waiting for results.
    # The workers will commit to the volume as they progress.
    # Use the realtime puller to monitor and download results.
