#!/usr/bin/env python3
"""
modal_lgie_eval.py — Unified Modal wrapper for LGIE Evaluation.

Architecture:
  1. Upload: copy the local data folder onto a Modal volume
  2. Evaluate: run lgie_evaluator.py inside the Modal container
               (R-scripts read/write to the volume-backed data dir)
  3. Download: pull the evaluated data back to the local machine

Preemption resilience:
  - Volume is committed after EVERY phase and every COMMIT_INTERVAL cases
  - skip_existing is ALWAYS enabled — restarts auto-resume where they left off
  - Upload checks if data already exists on volume; skips if so

Usage:
    # Full pipeline (upload → evaluate → download)
    modal run modal_lgie_eval.py --action run

    # Individual steps
    modal run modal_lgie_eval.py --action upload
    modal run modal_lgie_eval.py --action evaluate
    modal run modal_lgie_eval.py --action download

    # Evaluate just one test case (for debugging)
    modal run modal_lgie_eval.py --action run --test-case 1

    # Force re-upload even if data exists on volume
    modal run modal_lgie_eval.py --action upload --force-upload

    # Full run with nohup
    nohup modal run modal_lgie_eval.py --action run > sge_goal_evaluator_human_alignment.log 2>&1 &
"""

import modal
import os
import sys
import time
from pathlib import Path

# ============================================================================
# Configuration — parameterised via env vars for parallel runs
# ============================================================================
APP_NAME = os.environ.get(
    "LGIE_APP_NAME", "sge-goal-evaluator"
)
VOLUME_NAME = os.environ.get(
    "LGIE_VOLUME_NAME", "sge-goal-evaluator-data"
)
SYNC_VOLUME_NAME = os.environ.get(
    "LGIE_SYNC_VOLUME", "lgie-sync-queue"
)

# Dataset identity — used for sync queue entries
LGIE_SHORT_NAME = os.environ.get("LGIE_SHORT_NAME", "unknown")
LGIE_REL_PATH = os.environ.get("LGIE_REL_PATH", "")

# How often to commit the volume inside a phase (every N test cases)
COMMIT_INTERVAL = 25

# Local data path — override with LGIE_LOCAL_DATA_DIR for parallel runs
_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sge-goal-evaluator-human-alignment-test",
)
LOCAL_DATA_DIR = os.environ.get("LGIE_LOCAL_DATA_DIR", _DEFAULT_DATA_DIR)

# Modal mount points
WEIGHTS_MOUNT = "/weights"
DATA_MOUNT = "/eval-data"
EVAL_DATA_MOUNT = "/eval-images"  # lgie-eval-data (original scene images)
SYNC_MOUNT = "/sync-queue"        # sync queue for detached-mode result streaming
APP_MOUNT = "/app"

CUDA_VERSION = "12.4.0"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu22.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# ============================================================================
# Modal App + Volumes
# ============================================================================
app = modal.App(APP_NAME)

# Volume for test-case data (images + instructions + results)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Volume for model weights (SAM, XMem, LoFTR, HuggingFace cache)
weights_volume = modal.Volume.from_name("lgie-eval-weights", create_if_missing=True)

# Volume for lgie-eval-data (original scene images used by R scripts)
eval_data_volume = modal.Volume.from_name("lgie-eval-data", create_if_missing=True)

# Volume for sync queue — detached eval jobs write results here,
# sync watcher reads them and streams to local
sync_volume = modal.Volume.from_name(SYNC_VOLUME_NAME, create_if_missing=True)

# HuggingFace secret for gated models
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

# ============================================================================
# Container Image
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
            # large files that live on the weights volume
            "sam_vit_h_4b8939.pth",
            "XMem/saves/*.pth",
            "XMem/scripts/saves/*.pth",
            "loftr_weights/*.ckpt",
            # data dirs — live on volumes
            "lgie-eval-data/",
            "lgie-eval-data-backup/",
            "sge-goal-evaluator-human-alignment-test/",
            "human-alignment-test-data/",
            "data/",
            # backups / outputs — not needed in container
            "scripts-backup/",
            "enhanced-scripts-backup/",
            "enhanced-scripts-backup-2/",
            "R2_Output/",
            "R3_Output/",
            "object_change_detection_logs/",
            # parallel eval runs — data lives on per-run volumes
            "FINAL-EVAL-RUNS/",
            "FINAL-EVAL-RUNS-*/",
            "FINAL-EVAL-RUNS-SGE-GOAL-EVALUATOR-ORIG-BACKUP-DO-NOT-DELETE/",
            "eval-logs/",
            "lgie-eval-data-backup/",
            # launch / sync / setup / test scripts — not needed in container
            "launch_all.py",
            "resume_all.py",
            "volume_sync.py",
            "volume_sync_resume.py",
            "setup_shared_volumes.py",
            "setup_account.py",
            "modal_eval_runner.py",
            "create_data_folder.py",
            "run_with_sync.py",
            "launch_dual.sh",
            "launch_detached.py",
            "launch_fast.sh",
            "modal_sync_watcher.py",
            "sge_goal_eval_aggregator.py",
            "test_7b_failures.py",
            "test_awq.py",
            "upload_weights.py",
            "verify_integrity.py",
            "watchdog_429.py",
            "watchdog_gemini.py",
            "MODAL_README.md",
            "environment.yml",
            "nohup.out",
            # FINAL-EVAL-RUNS-* backup dirs
            "FINAL-EVAL-RUNS-SGE-GOAL-EVALUATOR/",
            # watchdog transient files — must NOT be mounted (race conditions)
            "watchdog_state.json",
            "watchdog_gemini.pid",
            "watchdog_429.pid",
            "watchdog_429.log",
            "watchdog_gemini.log",
            "*.pid",
        ],
    )
)


# ============================================================================
# Helper: set up symlinks so R-scripts find weights + eval-data at expected paths
# ============================================================================
def _setup_symlinks():
    """Create symlinks from volume mount-points to the locations R-scripts expect."""
    import shutil

    # SAM weights
    sam_src = Path(f"{WEIGHTS_MOUNT}/sam_vit_h_4b8939.pth")
    sam_dst = Path(f"{APP_MOUNT}/sam_vit_h_4b8939.pth")
    if sam_src.exists() and not sam_dst.exists():
        os.symlink(sam_src, sam_dst)

    # XMem weights
    xmem_dst_dir = Path(f"{APP_MOUNT}/XMem/saves")
    xmem_dst_dir.mkdir(parents=True, exist_ok=True)
    xmem_src_dir = Path(f"{WEIGHTS_MOUNT}/XMem")
    if xmem_src_dir.exists():
        for pth in xmem_src_dir.glob("*.pth"):
            dst = xmem_dst_dir / pth.name
            if not dst.exists():
                os.symlink(pth, dst)

    # LoFTR weights
    loftr_dst_dir = Path(f"{APP_MOUNT}/loftr_weights")
    loftr_dst_dir.mkdir(parents=True, exist_ok=True)
    loftr_src_dir = Path(f"{WEIGHTS_MOUNT}/loftr_weights")
    if loftr_src_dir.exists():
        for ckpt in loftr_src_dir.glob("*.ckpt"):
            dst = loftr_dst_dir / ckpt.name
            if not dst.exists():
                os.symlink(ckpt, dst)

    # lgie-eval-data (original scene images)
    data_dst = Path(f"{APP_MOUNT}/lgie-eval-data")
    data_src = Path(f"{EVAL_DATA_MOUNT}/lgie-eval-data")
    if data_src.exists() and not data_dst.exists():
        os.symlink(data_src, data_dst)
    # Also try if the volume root IS the data
    elif not data_dst.exists() and Path(EVAL_DATA_MOUNT).exists():
        # Check if images are directly in the volume
        if any(Path(EVAL_DATA_MOUNT).glob("original_image/*")):
            os.symlink(Path(EVAL_DATA_MOUNT), data_dst)


# ============================================================================
# Modal function: Phases 1-3 on L40S GPU (VideoChat-Flash, SAM+LoFTR+BLIP-2)
# ============================================================================
@app.function(
    image=image,
    gpu="l40s",
    secrets=[huggingface_secret],
    volumes={
        WEIGHTS_MOUNT: weights_volume,
        DATA_MOUNT: data_volume,
        EVAL_DATA_MOUNT: eval_data_volume,
        SYNC_MOUNT: sync_volume,
    },
    timeout=86400,    # 24 hours
    memory=65536,     # 64 GB RAM
)
def run_phases_1_2_3(
    test_case: int = None,
    verbose: bool = False,
    clean_gemini_phases: bool = False,
    phases: list = None,
    result_suffix: str = None,
    short_name: str = "unknown",
    rel_path: str = "",
) -> str:
    """
    Run Phases 1-3 (or a subset) on an L40S GPU.

    Phase 1: VideoChat-Flash → R1, R4, R8-9, R10-12, R13, R14-15
    Phase 2: SAM + LoFTR + BLIP-2 → R2
    Phase 3: SAM + BLIP-2 → R3
    """
    if phases is None:
        phases = [1, 2, 3]
    return _run_evaluation_inner(
        phases=phases,
        test_case=test_case,
        verbose=verbose,
        clean_gemini_phases=clean_gemini_phases,
        result_suffix=result_suffix,
        short_name=short_name,
        rel_path=rel_path,
    )


# ============================================================================
# Modal function: Phase 4 on L40S GPU (SAM + XMem + VideoChat-Flash → R5-7)
# ============================================================================
@app.function(
    image=image,
    gpu="l40s",
    secrets=[huggingface_secret],
    volumes={
        WEIGHTS_MOUNT: weights_volume,
        DATA_MOUNT: data_volume,
        EVAL_DATA_MOUNT: eval_data_volume,
        SYNC_MOUNT: sync_volume,
    },
    timeout=86400,    # 24 hours
    memory=65536,     # 64 GB RAM
)
def run_phase_4(
    test_case: int = None,
    verbose: bool = False,
    result_suffix: str = None,
    short_name: str = "unknown",
    rel_path: str = "",
) -> str:
    """
    Run Phase 4 on an L40S GPU (needed for XMem + VideoChat-Flash together).

    Phase 4: SAM + XMem + VideoChat-Flash → R5, R6, R7
    """
    return _run_evaluation_inner(
        phases=[4],
        test_case=test_case,
        verbose=verbose,
        clean_gemini_phases=False,
        result_suffix=result_suffix,
        short_name=short_name,
        rel_path=rel_path,
    )


# ============================================================================
# Shared evaluation logic (called by both GPU functions)
# ============================================================================
def _run_evaluation_inner(
    phases: list,
    test_case: int = None,
    verbose: bool = False,
    clean_gemini_phases: bool = False,
    result_suffix: str = None,
    short_name: str = "unknown",
    rel_path: str = "",
) -> str:
    """
    Shared evaluation body used by both run_phases_1_2_3 and run_phase_4.

    Key features:
      - Imports lgie_evaluator functions directly (no subprocess)
      - Always uses skip_existing=True so restarts auto-resume
      - Commits volume after EACH phase boundary
      - Commits volume every COMMIT_INTERVAL cases within a phase
    """
    # Set up weight and data symlinks
    _setup_symlinks()

    # Add the app mount to sys.path so we can import from R-scripts
    if APP_MOUNT not in sys.path:
        sys.path.insert(0, APP_MOUNT)
    os.chdir(APP_MOUNT)

    # Import evaluator functions
    from lgie_evaluator import (
        list_test_cases, evaluate_all_phased, _ensure_paths,
        PHASE2_KEYS, PHASE3_KEYS, PHASE4_KEYS,
    )

    # --- Optional cleanup: strip Phase 2/3/4 keys to force re-eval ---
    if clean_gemini_phases:
        import json as _json
        keys_to_remove = set(PHASE2_KEYS + PHASE3_KEYS + PHASE4_KEYS)
        data_path = Path(DATA_MOUNT)
        cleaned = 0
        for results_file in sorted(data_path.rglob("evaluation_results.json")):
            try:
                with open(results_file, "r") as f:
                    results = _json.load(f)
                removed = [k for k in keys_to_remove if k in results]
                if removed:
                    for k in removed:
                        del results[k]
                    with open(results_file, "w") as f:
                        _json.dump(results, f, indent=2)
                    cleaned += 1
            except Exception as e:
                print(f"[clean] Warning: could not process {results_file}: {e}")
        print(f"[clean] Stripped Phase 2/3/4 keys from {cleaned} results files")
        data_volume.commit()
        print("[clean] Volume committed after cleanup")
        sys.stdout.flush()

    # List what's in the data dir
    data_path = Path(DATA_MOUNT)
    if data_path.exists():
        dirs = sorted([
            d.name for d in data_path.iterdir()
            if d.is_dir() and d.name.isdigit()
        ], key=lambda x: int(x))
        print(f"[modal] Found {len(dirs)} test case folders on volume")
    else:
        print(f"[modal] WARNING: data dir does not exist: {DATA_MOUNT}")
        return "ERROR: data dir not found on volume"

    # Determine cases to evaluate
    if test_case is not None:
        cases = [test_case]
    else:
        cases = list_test_cases(DATA_MOUNT)

    phase_label = f"Phases {phases}"
    gpu_label = "L40S"
    print(f"[modal] Evaluating {len(cases)} test cases — {phase_label} on {gpu_label}")
    print(f"[modal] skip_existing=True (always), commit_interval={COMMIT_INTERVAL}")
    sys.stdout.flush()

    _ensure_paths()

    # --- Derive output filename from suffix ---
    output_filename = (
        f"evaluation_results_{result_suffix}.json"
        if result_suffix
        else "evaluation_results.json"
    )
    print(f"[modal] output_filename={output_filename}")
    sys.stdout.flush()

    # --- Sync queue helper: write result entry to sync volume ---
    _job_id = f"{short_name}"
    if result_suffix:
        _job_id += f"-{result_suffix}"
    _sync_dir = os.path.join(SYNC_MOUNT, _job_id)
    _sync_log_path = os.path.join(_sync_dir, "eval.log")
    os.makedirs(_sync_dir, exist_ok=True)

    def _write_sync_entry(case_num, results_data):
        """Write a sync entry JSON to the sync volume for the watcher to pick up."""
        try:
            import json as _json
            entry = {
                "short_name": short_name,
                "rel_path": rel_path,
                "case_num": str(case_num),
                "suffix": result_suffix or "",
                "results": results_data,
            }
            entry_path = os.path.join(_sync_dir, f"{case_num}.json")
            with open(entry_path, "w") as f:
                _json.dump(entry, f)
        except Exception as e:
            print(f"[sync] Warning: could not write sync entry for case {case_num}: {e}")

    def _append_log(msg):
        """Append a log line to the sync volume log file."""
        try:
            with open(_sync_log_path, "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        except Exception:
            pass

    _append_log(f"Job started: phases={phases} suffix={result_suffix} output={output_filename}")
    _append_log(f"Cases to evaluate: {len(cases)}")

    # --- Callbacks for periodic volume commits ---
    _commit_counter = [0]  # mutable counter in closure
    _sync_commit_counter = [0]

    def on_case_complete(case_num, done, total):
        """Called after each test case's results are saved to disk."""
        _commit_counter[0] += 1
        _sync_commit_counter[0] += 1

        # Commit data volume periodically
        if _commit_counter[0] % COMMIT_INTERVAL == 0:
            data_volume.commit()
            print(f"[modal] Data volume committed ({done}/{total} in current phase, "
                  f"{_commit_counter[0]} total since start)")
            sys.stdout.flush()

        # Write sync entry to sync volume
        try:
            import json as _json
            _tc_path = os.path.join(
                DATA_MOUNT, str(case_num), output_filename
            )
            if os.path.exists(_tc_path):
                with open(_tc_path) as _f:
                    _data = _json.load(_f)
                _write_sync_entry(case_num, _data)
        except Exception:
            pass  # never let sync break evaluation

        # Commit sync volume every 5 cases (lightweight JSON files)
        if _sync_commit_counter[0] % 5 == 0:
            try:
                sync_volume.commit()
            except Exception:
                pass

        # Log progress
        if _commit_counter[0] % 25 == 0:
            _append_log(f"Progress: {done}/{total} in current phase, "
                        f"{_commit_counter[0]} total cases processed")

    def on_phase_complete(phase_num):
        """Called after each phase finishes."""
        data_volume.commit()
        sync_volume.commit()
        _append_log(f"Phase {phase_num} complete")
        print(f"[modal] ✓ Volumes committed after Phase {phase_num}")
        sys.stdout.flush()

    # --- Run selected phases ---
    try:
        evaluate_all_phased(
            data_dir=DATA_MOUNT,
            cases=cases,
            skip_existing=True,  # ALWAYS skip existing for preemption resilience
            verbose=verbose,
            on_case_complete=on_case_complete,
            on_phase_complete=on_phase_complete,
            phases=phases,
            output_filename=output_filename,
        )
    except Exception as e:
        # Even on error, commit whatever progress was made
        print(f"[modal] ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        data_volume.commit()
        _append_log(f"ERROR: {e}")
        try:
            sync_volume.commit()
        except Exception:
            pass
        print("[modal] Volumes committed (partial progress saved after error)")
        return f"ERROR: {e}"

    # Final commit (in case last phase had < COMMIT_INTERVAL cases)
    data_volume.commit()
    _append_log(f"Job complete. Total cases processed: {_commit_counter[0]}")
    # Mark job as finished in sync volume
    try:
        import json as _json
        done_marker = os.path.join(_sync_dir, "_DONE.json")
        with open(done_marker, "w") as f:
            _json.dump({"status": "complete", "total_processed": _commit_counter[0],
                         "phases": phases, "suffix": result_suffix or ""}, f)
        sync_volume.commit()
    except Exception:
        pass
    print(f"[modal] ✓ Final volume commit. Total cases processed: {_commit_counter[0]}")

    return "exit_code=0"


# ============================================================================
# Local entrypoint (called by `modal run modal_lgie_eval.py`)
# ============================================================================
@app.local_entrypoint()
def main(
    action: str = "run",
    test_case: int = None,
    force_upload: bool = False,
    verbose: bool = False,
    clean_phases: bool = False,
    phases: str = None,
    result_suffix: str = None,
    short_name: str = "unknown",
    rel_path: str = "",
):
    """
    Orchestrate upload → evaluate → download.

    --action run   : full pipeline (upload + evaluate + download)
    --action upload   : upload local data folder to Modal volume
    --action evaluate : run evaluation on Modal (data must already be on volume)
    --action download : download results from Modal volume to local folder

    --force-upload : re-upload even if data already exists on volume
    --clean-phases : strip Phase 2/3/4 keys from all results before evaluation
                     (forces re-evaluation of Gemini-dependent phases)
    --phases : comma-separated phase numbers (e.g. "3" or "2,3" or "1,2,3,4")
    --result-suffix : suffix for output filename (e.g. "p3" -> evaluation_results_p3.json)
                      Used for parallel phase execution to avoid write conflicts.
    --short-name : dataset short name (for sync queue entry identification)
    --rel-path : dataset relative path (for sync queue local write target)
    """
    actions = action.lower().split(",")

    if "run" in actions:
        actions = ["upload", "evaluate", "download"]

    # Parse --phases flag
    phase_list = None
    if phases:
        phase_list = [int(p.strip()) for p in phases.split(",")]

    for act in actions:
        if act == "upload":
            _do_upload(test_case=test_case, force=force_upload)
        elif act == "evaluate":
            _do_evaluate(
                test_case=test_case,
                verbose=verbose,
                clean_gemini_phases=clean_phases,
                phase_list=phase_list,
                result_suffix=result_suffix,
                short_name=short_name,
                rel_path=rel_path,
            )
        elif act == "download":
            _do_download(test_case=test_case)
        else:
            print(f"Unknown action: {act}")
            sys.exit(1)


# ============================================================================
# Upload local data to volume (smart — skips if data already exists)
# ============================================================================
def _do_upload(test_case: int = None, force: bool = False):
    """Upload local data folder to the Modal volume using batch_upload.

    If data already exists on the volume, the upload is skipped unless
    --force-upload is passed.  This prevents overwriting evaluation_results.json
    files that were committed to the volume during a previous (possibly preempted)
    run.
    """
    from modal.volume import FileEntryType

    t0 = time.time()

    if not os.path.isdir(LOCAL_DATA_DIR):
        print(f"ERROR: local data directory not found: {LOCAL_DATA_DIR}")
        sys.exit(1)

    if test_case is not None:
        # --- Single test case ---
        tc_dir = os.path.join(LOCAL_DATA_DIR, str(test_case))
        if not os.path.isdir(tc_dir):
            print(f"ERROR: test case dir not found: {tc_dir}")
            sys.exit(1)

        if not force:
            # Check if this test case already exists on the volume
            try:
                existing = {e.path.strip("/") for e in data_volume.listdir("/")}
                if str(test_case) in existing:
                    print(f"Test case {test_case} already on volume — skipping upload "
                          f"(use --force-upload to overwrite)")
                    return
            except Exception:
                pass  # If listdir fails, proceed with upload

        print(f"Uploading test case {test_case} to volume '{VOLUME_NAME}' …")
        with data_volume.batch_upload(force=True) as batch:
            batch.put_directory(tc_dir, f"/{test_case}")
        print(f"Uploaded test case {test_case}.")
    else:
        # --- All test cases ---
        if not force:
            # Check if data already exists on the volume
            try:
                existing = [
                    e for e in data_volume.listdir("/")
                    if e.type == FileEntryType.DIRECTORY
                    and e.path.strip("/").isdigit()
                ]
                if len(existing) > 0:
                    print(f"Volume already has {len(existing)} test case folders — "
                          f"skipping upload (use --force-upload to overwrite)")
                    return
            except Exception:
                pass  # If listdir fails, proceed with upload

        print(f"Uploading {LOCAL_DATA_DIR} to volume '{VOLUME_NAME}' …")
        entries = os.listdir(LOCAL_DATA_DIR)
        dirs = sorted([e for e in entries if os.path.isdir(
            os.path.join(LOCAL_DATA_DIR, e)) and e.isdigit()],
            key=lambda x: int(x))
        files = [e for e in entries if os.path.isfile(
            os.path.join(LOCAL_DATA_DIR, e))]
        print(f"  {len(dirs)} test-case folders + {len(files)} root files")

        with data_volume.batch_upload(force=True) as batch:
            # Root-level files (manifest.json etc.)
            for fname in files:
                local_path = os.path.join(LOCAL_DATA_DIR, fname)
                batch.put_file(local_path, f"/{fname}")

            # Each test-case folder
            for i, d in enumerate(dirs):
                tc_path = os.path.join(LOCAL_DATA_DIR, d)
                batch.put_directory(tc_path, f"/{d}")

        print(f"Uploaded {len(dirs)} folders + {len(files)} files.")

    elapsed = time.time() - t0
    print(f"Upload complete in {elapsed:.1f}s.")


# ============================================================================
# Run evaluation on Modal
# ============================================================================
def _do_evaluate(test_case: int = None, verbose: bool = False,
                 clean_gemini_phases: bool = False, phase_list: list = None,
                 result_suffix: str = None, short_name: str = "unknown",
                 rel_path: str = ""):
    """Trigger the Modal GPU functions: all phases on L40S."""
    if phase_list is None:
        phase_list = [1, 2, 3, 4]

    l40s_phases_123 = [p for p in phase_list if p in (1, 2, 3)]
    need_phase4 = 4 in phase_list

    if l40s_phases_123:
        print(f"Starting Phases {l40s_phases_123} on Modal (GPU=L40S) …")
        if test_case:
            print(f"  test case: {test_case}")
        else:
            print("  mode: ALL test cases")
        print("  skip_existing: always enabled (preemption resilience)")
        if clean_gemini_phases:
            print("  clean_phases: YES — will strip Phase 2/3/4 keys before eval")
        print(f"  commit_interval: every {COMMIT_INTERVAL} cases")

        result_123 = run_phases_1_2_3.remote(
            test_case=test_case,
            verbose=verbose,
            clean_gemini_phases=clean_gemini_phases,
            phases=l40s_phases_123,
            result_suffix=result_suffix,
            short_name=short_name,
            rel_path=rel_path,
        )
        print(f"Phases {l40s_phases_123} result: {result_123}")

    if need_phase4:
        print(f"\nStarting Phase 4 on Modal (GPU=L40S) …")
        if test_case:
            print(f"  test case: {test_case}")
        else:
            print("  mode: ALL test cases")
        print("  skip_existing: always enabled (preemption resilience)")
        print(f"  commit_interval: every {COMMIT_INTERVAL} cases")

        result_4 = run_phase_4.remote(
            test_case=test_case,
            verbose=verbose,
            result_suffix=result_suffix,
            short_name=short_name,
            rel_path=rel_path,
        )
        print(f"Phase 4 result: {result_4}")


# ============================================================================
# Download results from volume back to local
# ============================================================================
def _do_download(test_case: int = None):
    """Download evaluated data from the volume back to local filesystem."""
    from modal.volume import FileEntryType

    t0 = time.time()
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

    print(f"Downloading from volume '{VOLUME_NAME}' → {LOCAL_DATA_DIR} …")

    count = 0
    for entry in data_volume.listdir("/"):
        name = entry.path.strip("/")

        if test_case is not None and name != str(test_case):
            # If we only want one test case, skip non-matching dirs
            if entry.type == FileEntryType.DIRECTORY:
                continue

        if entry.type == FileEntryType.FILE:
            # Root-level file (e.g. manifest.json)
            local_path = os.path.join(LOCAL_DATA_DIR, name)
            _download_file(name, local_path)
            count += 1
        elif entry.type == FileEntryType.DIRECTORY:
            # Test-case folder
            _download_folder(name, os.path.join(LOCAL_DATA_DIR, name))
            count += 1

    elapsed = time.time() - t0
    print(f"Download complete — {count} items in {elapsed:.1f}s.")


def _download_file(remote_path: str, local_path: str):
    """Download a single file from the volume."""
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    data = b""
    for chunk in data_volume.read_file(remote_path):
        data += chunk
    with open(local_path, "wb") as f:
        f.write(data)


def _download_folder(remote_dir: str, local_dir: str):
    """Download all files from a volume directory to a local directory."""
    from modal.volume import FileEntryType

    os.makedirs(local_dir, exist_ok=True)
    try:
        for entry in data_volume.listdir(f"/{remote_dir}"):
            fname = entry.path.strip("/").split("/")[-1]
            if entry.type == FileEntryType.FILE:
                remote_path = entry.path.lstrip("/")
                local_path = os.path.join(local_dir, fname)
                _download_file(remote_path, local_path)
    except Exception as e:
        print(f"  Warning downloading folder {remote_dir}: {e}")


if __name__ == "__main__":
    print("Use: modal run modal_lgie_eval.py --action run")
