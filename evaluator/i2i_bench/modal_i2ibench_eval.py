"""
Modal script to run I2I-Bench evaluation on cloud GPUs.

This script:
1. Uploads the local i2ibench-human-alignment-test data folder to a Modal volume
2. Sets up a container with all required dependencies (matching the local i2ibench env)
3. Runs the evaluation (subprocess-per-test-case) on an L40S GPU
4. Downloads the results (evaluation_results.json files) back locally

Usage:
    # Full run: upload data, evaluate all test cases, download results
    modal run modal_i2ibench_eval.py

    # Test with a single test case first
    modal run modal_i2ibench_eval.py --test-cases 1

    # Resume after preemption/crash (keeps existing results on volume)
    modal run modal_i2ibench_eval.py --resume

    # Use a custom data directory
    modal run modal_i2ibench_eval.py --data-dir /path/to/test_data

    # Upload data only
    modal run modal_i2ibench_eval.py --action upload

    # Download results only (after a previous run)
    modal run modal_i2ibench_eval.py --action download

    # Run with custom app/volume names (for parallel independent runs):
    MODAL_APP_NAME=my-app MODAL_VOLUME_NAME=my-vol modal run modal_i2ibench_eval.py --data-dir /path/to/data
"""

import modal
import os
import shutil

# ============================================================================
# Configuration
# ============================================================================
GPU_TYPE = "A10G"
NUM_GPUS = 1
GPU_CONFIG = f"{GPU_TYPE}:{NUM_GPUS}"
MAX_PARALLEL_WORKERS = int(os.environ.get("MODAL_MAX_WORKERS", "20"))

# These can be overridden via environment variables for parallel independent runs.
APP_NAME = os.environ.get("MODAL_APP_NAME", "i2ibench-evaluation")
VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", "i2ibench-human-alignment-test")

LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i2ibench-human-alignment-test")
# Internal directory name inside the volume (same for all runs since volumes are isolated)
VOLUME_DATA_PREFIX = "test-data"
REMOTE_DATA_DIR = f"/data/{VOLUME_DATA_PREFIX}"
REMOTE_CODE_DIR = "/root/i2ibench"

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App(APP_NAME)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

huggingface_secret = modal.Secret.from_name("huggingface-secret")

# ============================================================================
# Container Image
# ============================================================================
# Use CUDA 12.4 to match the local environment (torch 2.6.0+cu124)
CUDA_TAG = "12.4.1-devel-ubuntu22.04"

_project_dir = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    # Install PyTorch matching the local env
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Install all dependencies matching local env versions
    .pip_install(
        "transformers==5.1.0",
        "accelerate==1.12.0",
        "peft==0.18.1",
        "timm==1.0.24",
        "numpy==2.2.6",
        "Pillow==12.0.0",
        "scipy==1.15.3",
        "opencv-python==4.13.0.92",
        "scikit-image==0.25.2",
        "qwen-vl-utils==0.0.14",
        "tiktoken==0.12.0",
        "verovio==6.0.1",
        "sentencepiece==0.2.1",
        "protobuf==6.33.5",
        "einops==0.8.2",
        "tqdm==4.67.3",
        "huggingface-hub",
        "icecream",
    )
    # Create flash_attn stub directory
    .run_commands(
        "mkdir -p /usr/local/lib/python3.10/site-packages/flash_attn"
    )
    # Add flash_attn stub files (copy=True so we can add more files after)
    .add_local_file(
        os.path.join(_project_dir, "_modal_stubs", "flash_attn_init.py"),
        remote_path="/usr/local/lib/python3.10/site-packages/flash_attn/__init__.py",
        copy=True,
    )
    .add_local_file(
        os.path.join(_project_dir, "_modal_stubs", "flash_attn_interface.py"),
        remote_path="/usr/local/lib/python3.10/site-packages/flash_attn/flash_attn_interface.py",
        copy=True,
    )
    .add_local_file(
        os.path.join(_project_dir, "_modal_stubs", "flash_attn_bert_padding.py"),
        remote_path="/usr/local/lib/python3.10/site-packages/flash_attn/bert_padding.py",
        copy=True,
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    # Add the I2I-Bench code last (no copy=True so they mount at startup for fast iteration)
    .add_local_file(
        os.path.join(_project_dir, "i2i_bench_evaluator.py"),
        remote_path=f"{REMOTE_CODE_DIR}/i2i_bench_evaluator.py",
    )
    .add_local_file(
        os.path.join(_project_dir, "runner.py"),
        remote_path=f"{REMOTE_CODE_DIR}/runner.py",
    )
    .add_local_file(
        os.path.join(_project_dir, "phase_runner.py"),
        remote_path=f"{REMOTE_CODE_DIR}/phase_runner.py",
    )
    .add_local_file(
        os.path.join(_project_dir, "q_insight_scorer.py"),
        remote_path=f"{REMOTE_CODE_DIR}/q_insight_scorer.py",
    )
)


# ============================================================================
# Upload data to Modal volume
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,
)
def upload_data():
    """Upload is handled by the local entrypoint using volume.put_file. This is a no-op."""
    import os
    # Just verify the data is there
    if os.path.exists(REMOTE_DATA_DIR):
        entries = os.listdir(REMOTE_DATA_DIR)
        # Count only numbered directories (test cases)
        tc_dirs = [e for e in entries if os.path.isdir(os.path.join(REMOTE_DATA_DIR, e)) and e.isdigit()]
        print(f"Data directory exists with {len(tc_dirs)} test case directories")
        return {"status": "success", "test_cases": len(tc_dirs)}
    else:
        print("Data directory not found on volume!")
        return {"status": "error", "message": "Data not found"}


# ============================================================================
# Run evaluation
# ============================================================================
# How often (in seconds) to commit the volume during evaluation.
# This ensures that results are persisted even if the container is preempted.
VOLUME_COMMIT_INTERVAL_SECONDS = 120  # every 2 minutes


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=21600,  # 6 hours (was 24h, reduced to cap max GPU cost per worker)
    memory=65536,   # 64 GB RAM
    # Modal will retry on preemption — the resume logic skips already-done dims
    retries=modal.Retries(
        max_retries=3,
        initial_delay=10.0,
        backoff_coefficient=2.0,
    ),
)
def run_evaluation(test_cases: list[str] = None, phases: list[int] = None):
    """
    Run I2I-Bench phase-based evaluation on the data in the Modal volume.
    Each model group loads once, processes ALL test cases, then unloads.

    Preemption-resilient: a background thread commits the volume every
    VOLUME_COMMIT_INTERVAL_SECONDS so that results written by the phase
    runner are persisted even if the container is killed mid-run.
    On restart (retry), the phase runner's _dim_already_done() logic
    skips dimensions that already have a score in evaluation_results.json.
    """
    import subprocess
    import sys
    import os
    import threading
    import time
    import json
    import traceback

    def _mark_stuck_tc(data_dir, tc_id):
        """Mark all Phase 2 dimensions as failed for a stuck TC so skip logic works."""
        results_file = os.path.join(data_dir, str(tc_id), "evaluation_results.json")
        try:
            if os.path.exists(results_file):
                with open(results_file) as f:
                    results = json.load(f)
            else:
                results = {}
            for dim in ["non_edited_region_fidelity", "subject_identity_fidelity"]:
                if dim not in results or "score" not in results.get(dim, {}):
                    results[dim] = {
                        "score": 0.0,
                        "details": {"error": f"Stuck TC killed by watchdog"},
                    }
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[watchdog] Marked TC {tc_id} Phase 2 dims as failed")
        except Exception as exc:
            print(f"[watchdog] Failed to mark TC {tc_id}: {exc}")

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    # Verify data exists
    if not os.path.exists(REMOTE_DATA_DIR):
        raise RuntimeError(f"Data directory {REMOTE_DATA_DIR} not found on volume!")

    entries = os.listdir(REMOTE_DATA_DIR)
    tc_dirs = sorted(
        [e for e in entries if os.path.isdir(os.path.join(REMOTE_DATA_DIR, e)) and e.isdigit()],
        key=lambda x: int(x)
    )
    print(f"Found {len(tc_dirs)} test cases in {REMOTE_DATA_DIR}")

    # --- Count already-completed dimensions (for resume reporting) ---
    target_tcs = test_cases if test_cases else tc_dirs
    pre_existing = 0
    for tc_id in target_tcs:
        rpath = os.path.join(REMOTE_DATA_DIR, tc_id, "evaluation_results.json")
        if os.path.exists(rpath):
            import json
            try:
                with open(rpath) as _f:
                    _res = json.load(_f)
                pre_existing += sum(
                    1 for v in _res.values()
                    if isinstance(v, dict) and "error" not in v.get("details", {})
                )
            except Exception:
                pass
    if pre_existing:
        print(f"  Resuming: {pre_existing} dimension-results already present from a previous run")

    # --- Background thread: periodic volume commit ---
    _stop_commit = threading.Event()

    def _periodic_commit():
        while not _stop_commit.is_set():
            _stop_commit.wait(VOLUME_COMMIT_INTERVAL_SECONDS)
            if _stop_commit.is_set():
                break
            try:
                volume.commit()
                print("  [checkpoint] Volume committed")
            except Exception as exc:
                print(f"  [checkpoint] Volume commit failed (non-fatal): {exc}")

    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()

    # --- Run the phase runner with stuck-TC watchdog ---
    # sa2va.segment() can hang indefinitely in CUDA C extensions where Python-level
    # timeouts (signal.alarm, threading.join(timeout)) don't fire. We monitor the
    # subprocess output for progress and SIGKILL+restart when it appears stuck.
    STUCK_TIMEOUT = 180  # seconds without new TC = stuck (180s to catch hangs faster)
    MAX_RESTARTS = 5     # max restart attempts (was 30, reduced to avoid credit waste)

    cmd = [
        sys.executable, "-u",  # -u for unbuffered output
        f"{REMOTE_CODE_DIR}/phase_runner.py",
        "--data-dir", REMOTE_DATA_DIR,
    ]
    if test_cases:
        cmd.extend(["--test-cases"] + test_cases)
    if phases:
        cmd.extend(["--phases"] + [str(p) for p in phases])

    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)

    import re as _re
    import time as _time
    import signal as _signal

    final_returncode = 0
    for restart_num in range(MAX_RESTARTS + 1):
        if restart_num > 0:
            print(f"\n{'='*60}")
            print(f"[watchdog] Restarting phase_runner (attempt {restart_num}/{MAX_RESTARTS})")
            print(f"{'='*60}")

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=REMOTE_CODE_DIR, text=True, bufsize=1,
        )
        last_progress_time = _time.monotonic()
        stuck_tc = None
        finished_normally = False

        try:
            import selectors
            sel = selectors.DefaultSelector()
            sel.register(proc.stdout, selectors.EVENT_READ)

            while proc.poll() is None:
                events = sel.select(timeout=10)  # check every 10s
                for key, _ in events:
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    print(line, end='', flush=True)
                    # Detect progress: new TC, phase boundary, or model loading
                    if _re.search(r'\[Phase \d\].*TC \d+', line):
                        last_progress_time = _time.monotonic()
                        m = _re.search(r'TC (\d+)', line)
                        if m:
                            stuck_tc = m.group(1)
                    elif 'Phase' in line and 'complete' in line:
                        last_progress_time = _time.monotonic()
                        stuck_tc = None
                    elif any(kw in line for kw in [
                        'Loading', 'Fetching', 'Downloading',
                        'Loading weights', 'Loading Sa2VA', 'Loading LMM',
                        'Loading DINOv3', 'Loading QInsight',
                        'checkpoint', 'skip (done)',
                        'Materializing', 'generation_config',
                        'huggingface.co', 'new version',
                        'PHASE', 'Phase', 'Sa2VA', 'DINOv',
                    ]):
                        # Model loading / skipping / checkpoints count as progress
                        last_progress_time = _time.monotonic()

                # Check if stuck
                elapsed = _time.monotonic() - last_progress_time
                if elapsed > STUCK_TIMEOUT:
                    print(f"\n[watchdog] No progress for {elapsed:.0f}s (stuck on TC {stuck_tc}). Killing subprocess.")
                    proc.kill()
                    proc.wait()
                    # Mark the stuck TC dimensions as failed so skip logic works on restart
                    if stuck_tc:
                        _mark_stuck_tc(REMOTE_DATA_DIR, stuck_tc)
                    break

            sel.unregister(proc.stdout)
            sel.close()

            # Drain remaining output
            for line in proc.stdout:
                print(line, end='', flush=True)

            if proc.returncode is not None and proc.returncode >= 0:
                final_returncode = proc.returncode
                finished_normally = True
            elif proc.returncode is not None and proc.returncode < 0:
                # Killed by signal (our kill or OOM)
                final_returncode = proc.returncode
        except Exception as exc:
            print(f"[watchdog] Error monitoring subprocess: {exc}")
            traceback.print_exc()
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass

        if finished_normally:
            break
    else:
        print(f"[watchdog] Exhausted {MAX_RESTARTS} restarts. Some TCs may be incomplete.")

    class _FakeResult:
        def __init__(self, rc):
            self.returncode = rc
    result = _FakeResult(final_returncode)

    if result.returncode != 0:
        print(f"WARNING: Runner exited with code {result.returncode}")

    # Stop the background commit thread and do a final commit
    _stop_commit.set()
    commit_thread.join(timeout=5)

    print("\nCommitting volume to persist results...")
    volume.commit()

    # Count successful results
    successful = 0
    for tc_id in target_tcs:
        results_file = os.path.join(REMOTE_DATA_DIR, tc_id, "evaluation_results.json")
        if os.path.exists(results_file):
            successful += 1

    print(f"\nResults: {successful}/{len(target_tcs)} test cases have evaluation_results.json")

    return {
        "status": "success" if result.returncode == 0 else "partial",
        "total_test_cases": len(target_tcs),
        "successful": successful,
        "return_code": result.returncode,
    }


# ============================================================================
# Local Entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    action: str = "run",
    test_cases: str = None,
    phases: str = None,
    resume: bool = False,
    data_dir: str = None,
    workers: int = 0,
    no_wait: bool = False,
):
    """
    Local entrypoint for I2I-Bench evaluation on Modal.

    Args:
        action: "run" (upload+evaluate+download), "upload", "evaluate", "download"
        test_cases: Comma-separated test case IDs (e.g. "1,2,3"). None = all.
        phases: Comma-separated phase IDs (e.g. "1,2,3,4"). None = all.
        resume: If True, keep existing results on the volume (don't clean before upload).
                Also re-uploads code but preserves evaluation_results.json files.
                Use after a preempted/crashed run to continue where it left off.
        data_dir: Path to the local test-data directory. Overrides LOCAL_DATA_DIR.
        workers: Number of parallel GPU workers (0 = auto, up to MAX_PARALLEL_WORKERS).
                 Each worker evaluates a non-overlapping subset of test cases on its own GPU.
        no_wait: If True, spawn workers and exit without waiting for results or downloading.
                 Use with --detach for fire-and-forget launches where a separate sync script
                 handles downloading results from the volume.
    """
    effective_data_dir = data_dir if data_dir else LOCAL_DATA_DIR
    tc_list = test_cases.split(",") if test_cases else None
    phase_list = [int(p) for p in phases.split(",")] if phases else None

    print("=" * 60)
    print(f"I2I-Bench Modal Evaluation - Action: {action}")
    print(f"  App: {APP_NAME}  |  Volume: {VOLUME_NAME}")
    print(f"  Data dir: {effective_data_dir}")
    print("=" * 60)

    if action in ("run", "upload"):
        # Upload data to volume
        print(f"\nUploading data from {effective_data_dir} to Modal volume...")
        if not os.path.exists(effective_data_dir):
            raise FileNotFoundError(f"Local data directory not found: {effective_data_dir}")

        if resume:
            print("  RESUME mode: keeping existing evaluation results on volume")
        else:
            # Clean old evaluation results and intermediates from volume before uploading
            print("  Cleaning old evaluation results from volume...")
            cleaned = 0
            try:
                for entry in volume.listdir(VOLUME_DATA_PREFIX, recursive=True):
                    if entry.path.endswith("evaluation_results.json") or entry.path.endswith("_phase_intermediates.json"):
                        volume.remove_file(entry.path)
                        cleaned += 1
            except Exception:
                pass  # Volume may be empty
            if cleaned:
                print(f"  Removed {cleaned} old result/intermediate files")

        # Batch-upload all files to the volume
        # In resume mode, skip evaluation_results.json and _phase_intermediates.json
        # to avoid overwriting completed results with stale local copies.
        _skip_suffixes = {"evaluation_results.json", "_phase_intermediates.json"} if resume else set()
        file_count = 0
        skipped = 0
        with volume.batch_upload(force=True) as batch:
            for root, dirs, files in os.walk(effective_data_dir):
                for fname in files:
                    if fname in _skip_suffixes:
                        skipped += 1
                        continue
                    local_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(local_path, effective_data_dir)
                    remote_path = f"{VOLUME_DATA_PREFIX}/{rel_path}"
                    batch.put_file(local_path, remote_path)
                    file_count += 1
                    if file_count % 500 == 0:
                        print(f"  Queued {file_count} files...")

        upload_msg = f"  Uploaded {file_count} total files to Modal volume"
        if skipped:
            upload_msg += f" (skipped {skipped} result files in resume mode)"
        print(upload_msg)

        # Verify upload
        result = upload_data.remote()
        print(f"  Verification: {result}")

        if action == "upload":
            return

    if action in ("run", "evaluate"):
        # Discover all TC IDs for chunking
        if tc_list:
            all_tcs = tc_list
        else:
            all_tcs = sorted(
                [e for e in os.listdir(effective_data_dir)
                 if os.path.isdir(os.path.join(effective_data_dir, e)) and e.isdigit()],
                key=lambda x: int(x)
            )

        # Determine number of parallel workers
        num_workers = workers if workers > 0 else MAX_PARALLEL_WORKERS
        num_workers = min(num_workers, len(all_tcs))  # no more workers than TCs
        num_workers = max(1, num_workers)

        if num_workers == 1:
            # Single worker (original behaviour)
            print(f"\nStarting evaluation on {GPU_TYPE} GPU (single worker, {len(all_tcs)} TCs)...")
            if no_wait:
                handle = run_evaluation.spawn(test_cases=all_tcs, phases=phase_list)
                print(f"  --no-wait: worker spawned, exiting. Use sync script to download results.")
                print("\n" + "=" * 60)
                print("Done! (fire-and-forget)")
                print("=" * 60)
                return
            result = run_evaluation.remote(test_cases=all_tcs, phases=phase_list)
            print(f"\nEvaluation result: {result}")
        else:
            # ── Parallel mode ──────────────────────────────────────────
            # Split TCs into roughly-equal, non-overlapping chunks
            k, m = divmod(len(all_tcs), num_workers)
            chunks = [
                all_tcs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
                for i in range(num_workers)
            ]
            chunks = [c for c in chunks if c]  # drop empties
            num_workers = len(chunks)

            print(f"\nStarting PARALLEL evaluation: {num_workers} workers x {GPU_TYPE} GPU")
            print(f"  Total TCs: {len(all_tcs)}  |  ~{len(all_tcs) // num_workers} TCs/worker")
            for i, chunk in enumerate(chunks):
                print(f"  Worker {i:2d}: TCs {chunk[0]:>4s}..{chunk[-1]:<4s}  ({len(chunk)} TCs)")

            # Spawn all workers (non-blocking)
            handles = []
            for i, chunk in enumerate(chunks):
                handle = run_evaluation.spawn(test_cases=chunk, phases=phase_list)
                handles.append(handle)
            print(f"\n  All {num_workers} workers spawned on Modal!")
            print(f"  Monitor per-worker logs on the Modal dashboard.")

            if no_wait:
                print(f"\n  --no-wait: exiting without waiting for workers or downloading.")
                print(f"  Workers continue running on Modal. Use sync script to download results.")
                print("\n" + "=" * 60)
                print("Done! (fire-and-forget)")
                print("=" * 60)
                return

            print(f"  Waiting for all workers to complete...\n")

            # Wait for workers, reporting completions as they arrive
            import concurrent.futures as _cf

            results_arr = [None] * num_workers
            done_count = 0

            def _await_worker(idx_handle):
                idx, h = idx_handle
                return idx, h.get()

            with _cf.ThreadPoolExecutor(max_workers=num_workers) as pool:
                future_map = {
                    pool.submit(_await_worker, (i, h)): i
                    for i, h in enumerate(handles)
                }
                for future in _cf.as_completed(future_map):
                    idx = future_map[future]
                    try:
                        _, result = future.result()
                        results_arr[idx] = result
                    except Exception as exc:
                        print(f"  Worker {idx} FAILED: {exc}")
                        results_arr[idx] = {
                            "status": "error", "successful": 0,
                            "total_test_cases": len(chunks[idx]),
                        }
                    done_count += 1
                    r = results_arr[idx]
                    print(f"  [{done_count}/{num_workers}] Worker {idx} -> "
                          f"{r.get('successful', '?')}/{r.get('total_test_cases', '?')} TCs "
                          f"({r.get('status', '?')})")

            # Aggregate
            total_ok = sum(r.get("successful", 0) for r in results_arr if r)
            total_n = sum(r.get("total_test_cases", 0) for r in results_arr if r)
            all_ok = all(r and r.get("status") == "success" for r in results_arr)
            print(f"\n  All workers done: {total_ok}/{total_n} TCs with results")
            if not all_ok:
                print("  WARNING: some workers had errors or partial results")

        if action == "evaluate":
            return

    if action in ("run", "download"):
        # Download results back locally
        print(f"\nDownloading results from Modal volume to {effective_data_dir}...")

        download_count = 0
        for entry in volume.listdir(VOLUME_DATA_PREFIX, recursive=True):
            remote_path = entry.path  # e.g. "test-data/1/evaluation_results.json"
            # Compute local path
            rel_path = os.path.relpath(remote_path, VOLUME_DATA_PREFIX)
            local_path = os.path.join(effective_data_dir, rel_path)

            if entry.type.name == "DIRECTORY":
                os.makedirs(local_path, exist_ok=True)
                continue

            # Download file
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as local_f:
                for chunk in volume.read_file(remote_path):
                    local_f.write(chunk)
            download_count += 1
            if download_count % 100 == 0:
                print(f"  Downloaded {download_count} files...")

        print(f"  Downloaded {download_count} total files")

        # Count evaluation_results.json files
        results_count = 0
        for entry_name in os.listdir(effective_data_dir):
            entry_path = os.path.join(effective_data_dir, entry_name)
            if os.path.isdir(entry_path):
                if os.path.exists(os.path.join(entry_path, "evaluation_results.json")):
                    results_count += 1
        print(f"  {results_count} test cases now have evaluation_results.json locally")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
