"""
Modal script to launch Flux 2 Klein LoRA finetuning for ERM mask prediction.

Usage:
    # Run training
    modal run modal_flux_2_klein_erm_train.py --action train

    # Run debug (short run)
    modal run modal_flux_2_klein_erm_train.py --action debug

    # Run inference only (after training)
    modal run modal_flux_2_klein_erm_train.py --action inference

    # Download dataset first (optional, for caching)
    modal run modal_flux_2_klein_erm_train.py --action download
"""

import modal
import os
import shutil
from PIL import Image

# ============================================================================
# GPU Configuration
# ============================================================================
NUM_GPUS = 1
GPU_TYPE = "B200"
GPU_CONFIG = f"{GPU_TYPE}:{NUM_GPUS}"

ENABLE_GPU_MONITORING = False
GPU_MONITORING_INTERVAL = 60

# ============================================================================
# Training Configuration
# ============================================================================
TRAINING_CONFIG = {
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-klein-base-9B",
    "revision": None,
    "variant": None,
    "dataset_name": "RiddhiCh/LGRM_BridgeDataV2_Teleoperated",
    "dataset_config_name": None,
    "edited_image_column": "final_edit_region_mask",
    "original_image_column": "original_image",
    "edit_instruction_column": "edit_instruction",
    "initial_edit_region_mask_column": "initial_edit_region_mask",
    "final_edit_region_mask_column": "final_edit_region_mask",
    "output_dir": "/data/models/flux-2-klein-erm",
    "logging_dir": "/data/logs/flux-2-klein-erm",
    "cache_dir": "/data/datasets/.cache",
    "resolution": 256,
    "train_batch_size": 160,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 75,
    "learning_rate": 1e-4,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 200,
    "optimizer": "adamw",
    "rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "gradient_checkpointing": True,
    "mixed_precision": "bf16",
    "use_8bit_adam": True,
    "checkpointing_epochs": 1,
    "checkpoints_total_limit": 75,
    "validation_epochs": 1,
    "resume_from_checkpoint": None,
    "seed": 0,
    "max_sequence_length": 512,
    "guidance_scale": 4.0,
    "weighting_scheme": "none",
    "dataloader_num_workers": 4,
    "train_text_encoder": False,
    "report_to": "tensorboard",
    "t_rec_max": 0.35,
    "p_rec_fraction": 1.0,
    "lambda_channel": 25.0,
    "max_val_samples": 15,
    "max_test_samples": None,
    "eval_viz_num": 10,
    "val_batch_size": 150,
    "test_batch_size": 150,
    "run_pre_training_validation": True,
}

DEBUG_TRAINING_OVERRIDES = {
    "num_train_epochs": 1,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_val_samples": 2,
    "max_test_samples": 2,
    "eval_viz_num": 2,
    "val_batch_size": 1,
    "test_batch_size": 1,
}

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App("flux2-klein-bridge-fer-mask-prediction")

VOLUME_NAME = "flux2-klein-bridge-fer-mask-prediction-volume"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

LOCAL_DIR = os.path.dirname(__file__)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision",
        "torchaudio",
    )
    .uv_pip_install(
        "accelerate",
        "transformers",
        "peft",
        "datasets",
        "pillow",
        "numpy",
        "scipy",
        "huggingface-hub",
        "hf_transfer",
        "wandb",
        "bitsandbytes",
        "prodigyopt",
        "scikit-image",
        "lpips",
        "tqdm",
        "packaging",
        "wheel",
        "sentencepiece",
        "tensorboard",
    )
    .run_commands("apt-get update && apt-get install -y git")
    .run_commands("pip install --upgrade pip setuptools wheel")
    .run_commands("pip install git+https://github.com/huggingface/diffusers.git")
    .run_commands("pip install flash-attn --no-build-isolation")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/data/datasets/.cache",
        }
    )
    .add_local_file(
        os.path.join(LOCAL_DIR, "flux_2_klein_erm_finetuning.py"),
        remote_path="/root/flux_2_klein_erm/flux_2_klein_erm_finetuning.py",
    )
    .add_local_file(
        os.path.join(LOCAL_DIR, "modal_flux_2_klein_erm_train.py"),
        remote_path="/root/flux_2_klein_erm/modal_flux_2_klein_erm_train.py",
    )
)

huggingface_secret = modal.Secret.from_name("huggingface-secret")


# ============================================================================
# Utilities
# ============================================================================

def start_gpu_monitoring(interval: int = 60):
    import subprocess
    import threading
    import time

    stop_monitoring = threading.Event()

    def monitor_gpu_utilization():
        while not stop_monitoring.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPU Utilization:")
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 8:
                        (
                            gpu_idx,
                            name,
                            gpu_util,
                            mem_util,
                            mem_used,
                            mem_total,
                            temp,
                            power,
                        ) = parts
                        print(
                            f"  GPU{gpu_idx} ({name}): "
                            f"GPU={gpu_util}% | Mem={mem_util}% ({mem_used}/{mem_total}MB) | "
                            f"Temp={temp}C | Power={power}W"
                        )

            except Exception as e:
                print(f"Error monitoring GPU: {e}")

            stop_monitoring.wait(interval)

    monitor_thread = threading.Thread(target=monitor_gpu_utilization, daemon=True)
    monitor_thread.start()

    return stop_monitoring, monitor_thread


def verify_hf_access(model_id: str, token: str | None) -> None:
    if not token:
        print("HF token NOT found. Cannot verify gated model access.")
        return

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
    except Exception:
        print("huggingface_hub not available to verify access.")
        return

    try:
        hf_hub_download(
            repo_id=model_id,
            filename="tokenizer/tokenizer_config.json",
            token=token,
            local_files_only=False,
        )
    except GatedRepoError as e:
        raise RuntimeError(
            "HF token does not have access to the gated repo. "
            "Request access and update the Modal secret 'huggingface-secret'."
        ) from e


def build_training_command(config: dict, num_gpus: int) -> list:
    candidates = [
        "/root/flux_2_klein_erm/flux_2_klein_erm_finetuning.py",
        "/root/flux_2_klein_erm_finetuning.py",
        os.path.join(os.getcwd(), "flux_2_klein_erm_finetuning.py"),
    ]

    script_path = None
    for c in candidates:
        if os.path.exists(c):
            script_path = c
            break

    if script_path is None:
        for root_dir, _, files in os.walk("/root"):
            if "flux_2_klein_erm_finetuning.py" in files:
                script_path = os.path.join(root_dir, "flux_2_klein_erm_finetuning.py")
                break

    if script_path is None:
        root_list = []
        try:
            root_list = os.listdir("/root")
        except Exception:
            pass
        raise FileNotFoundError(
            f"Training script not found. Searched: {candidates}. /root contents: {root_list}"
        )

    if num_gpus > 1:
        cmd = [
            "accelerate",
            "launch",
            "--multi_gpu",
            f"--num_processes={num_gpus}",
            "--num_machines=1",
            f"--mixed_precision={config.get('mixed_precision', 'bf16')}",
            "--dynamo_backend=no",
            script_path,
        ]
    else:
        cmd = ["python", "-u", script_path]

    arg_mapping = {
        "pretrained_model_name_or_path": "--pretrained_model_name_or_path",
        "revision": "--revision",
        "variant": "--variant",
        "dataset_name": "--dataset_name",
        "dataset_config_name": "--dataset_config_name",
        "edited_image_column": "--edited_image_column",
        "original_image_column": "--original_image_column",
        "edit_instruction_column": "--edit_instruction_column",
        "initial_edit_region_mask_column": "--initial_edit_region_mask_column",
        "final_edit_region_mask_column": "--final_edit_region_mask_column",
        "output_dir": "--output_dir",
        "logging_dir": "--logging_dir",
        "cache_dir": "--cache_dir",
        "resolution": "--resolution",
        "train_batch_size": "--train_batch_size",
        "gradient_accumulation_steps": "--gradient_accumulation_steps",
        "num_train_epochs": "--num_train_epochs",
        "max_train_steps": "--max_train_steps",
        "learning_rate": "--learning_rate",
        "lr_scheduler": "--lr_scheduler",
        "lr_warmup_steps": "--lr_warmup_steps",
        "optimizer": "--optimizer",
        "rank": "--rank",
        "lora_alpha": "--lora_alpha",
        "lora_dropout": "--lora_dropout",
        "checkpointing_epochs": "--checkpointing_epochs",
        "checkpoints_total_limit": "--checkpoints_total_limit",
        "validation_epochs": "--validation_epochs",
        "resume_from_checkpoint": "--resume_from_checkpoint",
        "seed": "--seed",
        "max_sequence_length": "--max_sequence_length",
        "guidance_scale": "--guidance_scale",
        "weighting_scheme": "--weighting_scheme",
        "dataloader_num_workers": "--dataloader_num_workers",
        "max_val_samples": "--max_val_samples",
        "max_test_samples": "--max_test_samples",
        "eval_viz_num": "--eval_viz_num",
        "val_batch_size": "--val_batch_size",
        "test_batch_size": "--test_batch_size",
        "mixed_precision": "--mixed_precision",
        "t_rec_max": "--t_rec_max",
        "p_rec_fraction": "--p_rec_fraction",
        "lambda_channel": "--lambda_channel",
    }

    for key, arg in arg_mapping.items():
        value = config.get(key)
        if value is not None:
            cmd.extend([arg, str(value)])

    if config.get("gradient_checkpointing"):
        cmd.append("--gradient_checkpointing")
    if config.get("cache_latents"):
        cmd.append("--cache_latents")
    if config.get("use_8bit_adam"):
        cmd.append("--use_8bit_adam")
    if config.get("train_text_encoder"):
        cmd.append("--train_text_encoder")
    if config.get("run_pre_training_validation"):
        cmd.append("--run_pre_training_validation")

    cmd.append("--allow_tf32")

    report_to = config.get("report_to", "tensorboard")
    cmd.extend(["--report_to", report_to])

    return cmd


def find_latest_checkpoint(output_dir: str) -> str | None:
    if not os.path.exists(output_dir):
        return None
    dirs = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not dirs:
        return None

    def _sort_key(name: str) -> int:
        parts = name.split("-")
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return 0

    dirs = sorted(dirs, key=_sort_key)
    return dirs[-1]


# ============================================================================
# Local <-> Volume Sync Helpers
# ============================================================================

def _is_volume_directory(entry) -> bool:
    """Check if a Modal Volume FileEntry represents a directory."""
    t = getattr(entry, "type", None)
    if t is None:
        return False
    try:
        return int(t) == 2  # FileEntryType.DIRECTORY = 2
    except (ValueError, TypeError):
        return "dir" in str(t).lower()


def _collect_volume_files(base_path: str) -> list:
    """Recursively collect all file entries under a volume directory."""
    all_files = []
    try:
        entries = list(volume.listdir(base_path))
    except Exception:
        return all_files
    for entry in entries:
        if _is_volume_directory(entry):
            all_files.extend(_collect_volume_files(entry.path))
        else:
            all_files.append(entry)
    return all_files


def upload_local_checkpoints_to_volume():
    """Upload local model checkpoints to Modal volume (if any exist).

    Scans LOCAL_DIR/models/<output_subdir>/ for checkpoint-epoch-N directories
    and uploads ones not already present on the volume.
    """
    output_subdir = os.path.basename(TRAINING_CONFIG["output_dir"])  # "flux-2-klein-erm"
    local_ckpt_base = os.path.join(LOCAL_DIR, "models", output_subdir)

    if not os.path.isdir(local_ckpt_base):
        print(f"[LOCAL] No local model directory at {local_ckpt_base}. Skipping upload.")
        return

    def _ckpt_epoch(name: str) -> int:
        parts = name.split("-")
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return 0

    ckpt_dirs = sorted(
        [d for d in os.listdir(local_ckpt_base)
         if d.startswith("checkpoint") and os.path.isdir(os.path.join(local_ckpt_base, d))],
        key=_ckpt_epoch,
    )

    if not ckpt_dirs:
        print("[LOCAL] No local checkpoint directories found. Skipping upload.")
        return

    print(f"[LOCAL] Found {len(ckpt_dirs)} local checkpoint(s): {ckpt_dirs}")

    # Volume-relative path (strip /data/ mount prefix)
    vol_output_path = TRAINING_CONFIG["output_dir"].removeprefix("/data/")  # "models/flux-2-klein-erm"

    # Check what already exists on the volume
    existing_on_volume = set()
    try:
        volume.reload()
    except Exception:
        pass  # reload() only works inside containers; skip from local
    try:
        for entry in volume.listdir(vol_output_path):
            name = entry.path.rstrip("/").split("/")[-1]
            existing_on_volume.add(name)
    except Exception:
        pass  # Directory may not exist on volume yet

    to_upload = [d for d in ckpt_dirs if d not in existing_on_volume]

    if not to_upload:
        print("[LOCAL] All local checkpoints already present on volume. Nothing to upload.")
        return

    print(f"[LOCAL] Uploading {len(to_upload)} checkpoint(s) to Modal volume...")
    with volume.batch_upload() as batch:
        for ckpt_dir in to_upload:
            local_path = os.path.join(local_ckpt_base, ckpt_dir)
            remote_path = f"{vol_output_path}/{ckpt_dir}"
            print(f"  {ckpt_dir} -> /{remote_path}")
            batch.put_directory(local_path, remote_path)
    print("[LOCAL] Checkpoint upload complete.")


def download_models_from_volume():
    """Download the models directory from Modal volume to the local machine.

    Uses the `modal volume get` CLI command as a subprocess to avoid the
    Modal Python SDK's high memory consumption (~18 GB RSS), which causes
    OOM on machines with ≤32 GB RAM.

    Downloads to LOCAL_DIR/models/<output_subdir>/.
    """
    import subprocess as sp

    vol_output_path = TRAINING_CONFIG["output_dir"].removeprefix("/data/")  # "models/flux-2-klein-erm"
    output_subdir = os.path.basename(TRAINING_CONFIG["output_dir"])  # "flux-2-klein-erm"
    local_dest = os.path.join(LOCAL_DIR, "models", output_subdir)
    # modal volume get places contents into <local>/<basename(remote)>/, so
    # pass the *parent* directory so files land inside local_dest.
    local_parent = os.path.dirname(local_dest)

    print(f"\n>>> [LOCAL] Syncing volume:/{vol_output_path} -> {local_dest}")
    os.makedirs(local_dest, exist_ok=True)

    modal_bin = os.environ.get("MODAL_BIN", shutil.which("modal") or "modal")
    volume_name = VOLUME_NAME  # e.g. "flux2-klein-bridge-fer-mask-prediction-volume"
    cmd = [
        modal_bin, "volume", "get",
        volume_name,
        vol_output_path,
        local_parent,
        "--force",
    ]
    print(f">>> [LOCAL] Running: {' '.join(cmd)}")
    result = sp.run(cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode == 0:
        print(f">>> [LOCAL] Download complete.")
        if result.stdout.strip():
            # Print last few lines of output (file summary)
            lines = result.stdout.strip().splitlines()
            for line in lines[-10:]:
                print(f"  {line}")
    else:
        print(f">>> [LOCAL] Download failed (exit code {result.returncode})")
        if result.stderr.strip():
            print(f"  stderr: {result.stderr.strip()[:500]}")


# ============================================================================
# Dataset Download
# ============================================================================
@app.function(
    image=image,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=3600,
)
def download_dataset():
    import os
    from datasets import load_dataset

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    print(f"Downloading dataset: {TRAINING_CONFIG['dataset_name']}")

    dataset = load_dataset(
        TRAINING_CONFIG["dataset_name"],
        TRAINING_CONFIG.get("dataset_config_name"),
        cache_dir=TRAINING_CONFIG["cache_dir"],
    )

    print("Dataset downloaded successfully!")
    try:
        if hasattr(dataset, "keys"):
            train_size = getattr(dataset["train"], "num_rows", None) if "train" in dataset else None
            val_size = getattr(dataset["validation"], "num_rows", None) if "validation" in dataset else None
            test_size = getattr(dataset["test"], "num_rows", None) if "test" in dataset else None
            print(f"  Train size: {train_size if train_size is not None else 'unknown'}")
            print(f"  Validation size: {val_size if val_size is not None else 'unknown'}")
            print(f"  Test size: {test_size if test_size is not None else 'unknown'}")
        else:
            size = getattr(dataset, "num_rows", None)
            print(f"  Train size: {size if size is not None else 'unknown'}")
    except Exception:
        print("  Dataset size info unavailable")

    volume.commit()

    return {
        "status": "success",
        "dataset_name": TRAINING_CONFIG["dataset_name"],
        "message": "Dataset downloaded and cached successfully",
    }


# ============================================================================
# Training (multi-GPU)
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=86400,
    memory=65536,
)
def train_flux_2_klein_erm():
    import collections
    import os
    import subprocess
    import sys

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["logging_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["cache_dir"], exist_ok=True)

    print(f"Starting Flux 2 Klein ERM training with {NUM_GPUS} {GPU_TYPE} GPUs...")
    print(f"Output directory: {TRAINING_CONFIG['output_dir']}")

    # Reload volume to see any freshly uploaded checkpoints from local
    try:
        volume.reload()
    except Exception:
        pass

    latest_ckpt = find_latest_checkpoint(TRAINING_CONFIG["output_dir"])
    config = dict(TRAINING_CONFIG)
    if latest_ckpt:
        print(f"\n>>> Auto-detected checkpoint on volume: {latest_ckpt}")
        config["resume_from_checkpoint"] = "latest"
    else:
        config["resume_from_checkpoint"] = None

    stop_monitoring = None
    monitor_thread = None

    if ENABLE_GPU_MONITORING:
        stop_monitoring, monitor_thread = start_gpu_monitoring(interval=GPU_MONITORING_INTERVAL)
        print(f"GPU monitoring started (logging every {GPU_MONITORING_INTERVAL} seconds)\n")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        try:
            print(f"HF token present (len={len(hf_token)}): {hf_token[:4]}... (masked)")
        except Exception:
            print("HF token present (masked)")
    else:
        print("HF token NOT found in environment. Ensure Modal secret 'huggingface-secret' defines HF_TOKEN.")

    verify_hf_access(TRAINING_CONFIG["pretrained_model_name_or_path"], hf_token)

    cmd = build_training_command(config, NUM_GPUS)

    print(f"Running command:\n{' '.join(cmd)}\n")
    print("=" * 80)

    train_log_path = os.path.join(TRAINING_CONFIG["output_dir"], "train_output.log")
    PERIODIC_COMMIT_INTERVAL = 300  # 5 minutes

    try:
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            cmd,
            cwd="/root/flux_2_klein_erm",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        if process.stdout is None:
            raise RuntimeError("Training process did not provide stdout stream.")
        last_lines = collections.deque(maxlen=200)
        # Write subprocess output to a log file instead of stdout to avoid
        # the Modal SDK accumulating it all in memory on the local client
        # (which causes OOM on machines with ≤32 GB RAM).  Only signal
        # lines and periodic summaries are printed to stdout.
        line_count = 0
        import time as _time
        _last_periodic_commit = _time.time()
        with open(train_log_path, "a", buffering=1) as log_file:
            for line in process.stdout:
                log_file.write(line)
                last_lines.append(line.rstrip())
                line_count += 1

                # Print every 500th line so the user can see progress
                if line_count % 500 == 0:
                    print(f"[train log] {line_count} lines ... {line.rstrip()[-120:]}")

                if "###CHECKPOINT_SAVED###" in line:
                    print("\n>>> [MODAL] Checkpoint saved detected. Committing volume...")
                    try:
                        volume.commit()
                        print(">>> [MODAL] Volume committed successfully after checkpoint save.")
                    except Exception as e:
                        print(f">>> [MODAL] Warning: Volume commit failed: {e}")

                elif "###PRE_TRAINING_VALIDATION_COMPLETE###" in line:
                    print("\n>>> [MODAL] Pre-training validation complete. Committing volume...")
                    try:
                        volume.commit()
                        print(">>> [MODAL] Volume committed successfully after pre-training validation.")
                    except Exception as e:
                        print(f">>> [MODAL] Warning: Volume commit failed: {e}")

                elif "###VALIDATION_COMPLETE###" in line:
                    print("\n>>> [MODAL] Validation complete detected. Committing volume...")
                    try:
                        volume.commit()
                        print(">>> [MODAL] Volume committed successfully after validation.")
                    except Exception as e:
                        print(f">>> [MODAL] Warning: Volume commit failed: {e}")

                elif "###TEST_COMPLETE###" in line:
                    print("\n>>> [MODAL] Final test complete detected. Committing volume...")
                    try:
                        volume.commit()
                        print(">>> [MODAL] Volume committed successfully after final test.")
                    except Exception as e:
                        print(f">>> [MODAL] Warning: Volume commit failed: {e}")

                # Print epoch-level and loss lines so the user can track progress
                elif any(kw in line for kw in ("Starting epoch", "Epoch ", "completed. Mean train loss", "INFO - __main__")):
                    print(line.rstrip())

                # Periodic volume commit so train_output.log stays fresh
                # for the sync daemon to download
                _now = _time.time()
                if _now - _last_periodic_commit > PERIODIC_COMMIT_INTERVAL:
                    try:
                        volume.commit()
                        _last_periodic_commit = _now
                    except Exception:
                        pass

        remaining = process.stdout.read()
        if remaining:
            with open(train_log_path, "a") as log_file:
                log_file.write(remaining)
            for tail_line in remaining.splitlines():
                last_lines.append(tail_line.rstrip())

        process.wait()
        print(f"\n[train log] Total lines: {line_count}. Log file: {train_log_path}")
        if process.returncode != 0:
            if last_lines:
                print("\n----- TRAIN LOGS (tail) -----")
                print("\n".join(last_lines))
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        print(">>> [MODAL] Committing volume to preserve partial progress...")
        try:
            volume.commit()
        except Exception:
            pass
        raise
    finally:
        if ENABLE_GPU_MONITORING and stop_monitoring is not None:
            print("\nStopping GPU monitoring...")
            stop_monitoring.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=5)

    print("\nCommitting volume to persist trained model...")
    volume.commit()

    return {
        "signal": "training_complete",
        "status": "success",
        "output_dir": TRAINING_CONFIG["output_dir"],
        "num_gpus": NUM_GPUS,
        "gpu_type": GPU_TYPE,
        "resumed_from": latest_ckpt,
        "message": f"Flux 2 Klein ERM training completed on {NUM_GPUS} {GPU_TYPE} GPUs",
    }


# ============================================================================
# Training (single-GPU)
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=86400,
    memory=65536,
)
def train_flux_2_klein_erm_single_gpu(config_override: dict | None = None):
    import collections
    import os
    import subprocess
    import sys

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["logging_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["cache_dir"], exist_ok=True)

    print("Starting Flux 2 Klein ERM training on single GPU...")

    # Reload volume to see any freshly uploaded checkpoints from local
    try:
        volume.reload()
    except Exception:
        pass

    latest_ckpt = find_latest_checkpoint(TRAINING_CONFIG["output_dir"])
    config = dict(TRAINING_CONFIG)
    if config_override:
        config.update(config_override)
    if latest_ckpt:
        print(f"\n>>> Auto-detected checkpoint on volume: {latest_ckpt}")
        config["resume_from_checkpoint"] = "latest"
    else:
        config["resume_from_checkpoint"] = None

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        try:
            print(f"HF token present (len={len(hf_token)}): {hf_token[:4]}... (masked)")
        except Exception:
            print("HF token present (masked)")
    else:
        print("HF token NOT found in environment. Ensure Modal secret 'huggingface-secret' defines HF_TOKEN.")

    verify_hf_access(TRAINING_CONFIG["pretrained_model_name_or_path"], hf_token)

    cmd = build_training_command(config, 1)

    print(f"Running command:\n{' '.join(cmd)}\n")

    train_log_path = os.path.join(TRAINING_CONFIG["output_dir"], "train_output.log")
    PERIODIC_COMMIT_INTERVAL = 300  # 5 minutes

    try:
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            cmd,
            cwd="/root/flux_2_klein_erm",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        if process.stdout is None:
            raise RuntimeError("Training process did not provide stdout stream.")
        last_lines = collections.deque(maxlen=200)
        line_count = 0
        import time as _time
        _last_periodic_commit = _time.time()
        with open(train_log_path, "a", buffering=1) as log_file:
            for line in process.stdout:
                log_file.write(line)
                last_lines.append(line.rstrip())
                line_count += 1

                if line_count % 500 == 0:
                    print(f"[train log] {line_count} lines ... {line.rstrip()[-120:]}")

                if "###CHECKPOINT_SAVED###" in line:
                    print("\n>>> [MODAL] Checkpoint saved detected. Committing volume...")
                    try:
                        volume.commit()
                        print(">>> [MODAL] Volume committed successfully after checkpoint save.")
                    except Exception as e:
                        print(f">>> [MODAL] Warning: Volume commit failed: {e}")

                elif "###PRE_TRAINING_VALIDATION_COMPLETE###" in line:
                    print("\n>>> [MODAL] Pre-training validation complete. Committing volume...")
                    try:
                        volume.commit()
                        print(">>> [MODAL] Volume committed successfully after pre-training validation.")
                    except Exception as e:
                        print(f">>> [MODAL] Warning: Volume commit failed: {e}")

                elif "###VALIDATION_COMPLETE###" in line:
                    print("\n>>> [MODAL] Validation complete detected. Committing volume...")
                    try:
                        volume.commit()
                        print(">>> [MODAL] Volume committed successfully after validation.")
                    except Exception as e:
                        print(f">>> [MODAL] Warning: Volume commit failed: {e}")

                elif "###TEST_COMPLETE###" in line:
                    print("\n>>> [MODAL] Final test complete detected. Committing volume...")
                    try:
                        volume.commit()
                        print(">>> [MODAL] Volume committed successfully after final test.")
                    except Exception as e:
                        print(f">>> [MODAL] Warning: Volume commit failed: {e}")

                elif any(kw in line for kw in ("Starting epoch", "Epoch ", "completed. Mean train loss", "INFO - __main__")):
                    print(line.rstrip())

                # Periodic volume commit so train_output.log stays fresh
                _now = _time.time()
                if _now - _last_periodic_commit > PERIODIC_COMMIT_INTERVAL:
                    try:
                        volume.commit()
                        _last_periodic_commit = _now
                    except Exception:
                        pass

        remaining = process.stdout.read()
        if remaining:
            with open(train_log_path, "a") as log_file:
                log_file.write(remaining)
            for tail_line in remaining.splitlines():
                last_lines.append(tail_line.rstrip())

        process.wait()
        print(f"\n[train log] Total lines: {line_count}. Log file: {train_log_path}")
        if process.returncode != 0:
            if last_lines:
                print("\n----- TRAIN LOGS (tail) -----")
                print("\n".join(last_lines))
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        print(">>> [MODAL] Committing volume to preserve partial progress...")
        try:
            volume.commit()
        except Exception:
            pass
        raise

    volume.commit()

    return {
        "signal": "training_complete",
        "status": "success",
        "output_dir": TRAINING_CONFIG["output_dir"],
        "resumed_from": latest_ckpt,
        "message": "Flux 2 Klein ERM training completed on single GPU",
    }


# ============================================================================
# Inference
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=3600,
)
def run_inference(
    lora_path: str | None = None,
    prompt: str = "Move the object to the right",
    input_image_url: str | None = None,
    init_mask_url: str | None = None,
):
    import os
    import sys
    import torch
    from diffusers.utils import load_image

    sys.path.insert(0, "/root/flux_2_klein_erm")

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

    lora_path = lora_path or TRAINING_CONFIG["output_dir"]

    print("Loading Flux 2 Klein pipeline...")
    pipeline = Flux2KleinPipeline.from_pretrained(
        TRAINING_CONFIG["pretrained_model_name_or_path"],
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    print(f"Loading LoRA weights from {lora_path}...")
    pipeline.load_lora_weights(lora_path)

    if input_image_url:
        print(f"Loading input image from {input_image_url}...")
        input_image = load_image(input_image_url)
    else:
        print("No input image provided, creating a test image...")
        input_image = Image.new("RGB", (512, 512), color="white")

    if init_mask_url:
        print(f"Loading initial mask from {init_mask_url}...")
        init_mask = load_image(init_mask_url)
    else:
        print("No initial mask provided, using blank mask.")
        init_mask = Image.new("RGB", input_image.size, color="black")

    blank = Image.new("RGB", input_image.size, color="black")
    cond_images = [input_image, init_mask, blank]

    print(f"Running inference with prompt: {prompt}")
    output = pipeline(
        prompt=prompt,
        image=cond_images,
        guidance_scale=TRAINING_CONFIG.get("guidance_scale", 1.0),
        num_inference_steps=50,
    )

    output_path = "/data/inference_output_erm.png"
    output.images[0].save(output_path)
    print(f"Output saved to {output_path}")

    volume.commit()

    return {
        "status": "success",
        "output_path": output_path,
        "prompt": prompt,
    }


# ============================================================================
# Local Entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    action: str = "train",
    prompt: str = "Move the object to the right",
    image_url: str | None = None,
    init_mask_url: str | None = None,
    single_gpu: bool = True,
):
    print("=" * 60)
    print(f"Flux 2 Klein ERM Finetuning - Action: {action}")
    print("=" * 60)

    if action in ("train", "debug"):
        # ------------------------------------------------------------------
        # Step 1: Upload any local checkpoints to the Modal volume so the
        #         remote training function can auto-resume from them.
        # ------------------------------------------------------------------
        upload_local_checkpoints_to_volume()

        # ------------------------------------------------------------------
        # Step 2: Launch training via .spawn() (fire-and-forget).
        #
        # The training function runs on Modal independently. With
        # `modal run -d` (detached mode), it persists even after this
        # local process exits. Volume commits happen on the remote side
        # after each checkpoint, validation, and test step.
        #
        # Continuous downloading of checkpoints/visualizations is handled
        # by the separate sync_volume.py script (launched by
        # launch_training.sh). A watchdog.py script monitors the app
        # and restarts it if Modal stops it unexpectedly.
        # ------------------------------------------------------------------
        try:
            if action == "debug":
                print("Spawning short debug training on Modal...")
                fc = train_flux_2_klein_erm_single_gpu.spawn(
                    config_override=DEBUG_TRAINING_OVERRIDES
                )
            elif single_gpu:
                print("Spawning single-GPU training on Modal...")
                fc = train_flux_2_klein_erm_single_gpu.spawn()
            else:
                print("Spawning multi-GPU training on Modal...")
                fc = train_flux_2_klein_erm.spawn()

            print(f"\n>>> [LOCAL] Training launched on Modal!")
            print(f">>> [LOCAL] Function call ID: {fc.object_id}")
            print(f">>> [LOCAL] Use sync_volume.py and watchdog.py for continuous syncing and monitoring.")
        except Exception as e:
            print(f"\n>>> [LOCAL] Failed to launch training: {e}")
            raise

        result = {"status": "launched", "message": f"{action.capitalize()} launched on Modal"}

    elif action == "inference":
        print("Running inference...")
        result = run_inference.remote(
            prompt=prompt,
            input_image_url=image_url,
            init_mask_url=init_mask_url,
        )
        print("\nInference Result:")
        print(f"  Status: {result['status']}")
        print(f"  Output: {result['output_path']}")

    elif action == "download":
        print("Downloading and caching dataset...")
        result = download_dataset.remote()
        print("\nDownload Result:")
        print(f"  Status: {result['status']}")
        print(f"  Message: {result['message']}")

    else:
        raise ValueError(
            f"Invalid action: {action}. Use 'train', 'debug', 'inference', or 'download'."
        )

    print("=" * 60)
    return result
