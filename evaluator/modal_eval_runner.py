"""
Modal wrapper script to run LGIE evaluation scripts on Modal's serverless infrastructure.

This script provides a unified interface to run any of the evaluation scripts (R1-R15)
on Modal GPUs with all required dependencies pre-installed.

Usage:
    # Run a specific evaluation script
    modal run modal_eval_runner.py --script R1.py
    modal run modal_eval_runner.py --script R2.py
    modal run modal_eval_runner.py --script R3.py
    modal run modal_eval_runner.py --script R4.py
    modal run modal_eval_runner.py --script R5-6-7.py
    modal run modal_eval_runner.py --script R8-9.py
    modal run modal_eval_runner.py --script R10-11-12.py
    modal run modal_eval_runner.py --script R13.py
    modal run modal_eval_runner.py --script R14-15.py

    # Run with verbose output
    modal run modal_eval_runner.py --script R1.py --verbose

    # List available scripts
    modal run modal_eval_runner.py --action list

    # Download outputs from Modal volume
    modal run modal_eval_runner.py --action download

Requirements:
    - Modal CLI installed and authenticated: pip install modal && modal token new
    - Hugging Face token (for model downloads): modal secret create huggingface-secret HF_TOKEN=your_token
    - Gemini API key (for R2, R3): Already hardcoded in the scripts

Note:
    Large files (SAM checkpoint ~2.4GB, XMem weights, LoFTR weights) are uploaded to a
    Modal volume on first run and cached for subsequent runs.
"""

import modal
import os
from pathlib import Path

# ============================================================================
# GPU Configuration
# ============================================================================
# Default GPU type (can be overridden per-script)
DEFAULT_GPU_TYPE = "l4"  # Options: "t4", "l4", "a10g", "a100", "a100-80gb", "h100", "l40s"

# GPU type mapping per script
# L4: 24GB VRAM, L40S: 48GB VRAM
GPU_CONFIG = {
    "R1.py": "l4",
    "R2.py": "l4",
    "R3.py": "l4",
    "R4.py": "l4",
    "R5-6-7.py": "l40s",  # Requires more VRAM for VideoChat-Flash
    "R8-9.py": "l4",
    "R10-11-12.py": "l4",
    "R13.py": "l4",
    "R14-15.py": "l4",
}

# ============================================================================
# Valid evaluation scripts
# ============================================================================
VALID_SCRIPTS = [
    "R1.py",
    "R2.py", 
    "R3.py",
    "R4.py",
    "R5-6-7.py",
    "R8-9.py",
    "R10-11-12.py",
    "R13.py",
    "R14-15.py",
]

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App("lgie-eval-runner")

# Create volumes for persistent storage
# - weights_volume: stores large model weights (SAM, XMem, LoFTR)
# - data_volume: stores evaluation data and outputs
weights_volume = modal.Volume.from_name("lgie-eval-weights", create_if_missing=True)
data_volume = modal.Volume.from_name("lgie-eval-data", create_if_missing=True)

# CUDA Configuration
CUDA_VERSION = "12.4.0"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu22.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# ============================================================================
# Define the Modal image with all required dependencies
# ============================================================================
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    # Install system dependencies
    .apt_install(
        "git",
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libgl1-mesa-glx",
        "libglib2.0-0",
    )
    # Install PyTorch first (CUDA 12.4 compatible - matching lgie_dp environment)
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0", 
        "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # Install main dependencies for all evaluation scripts (versions match lgie_dp)
    .pip_install(
        # Core ML libraries
        "transformers==4.40.1",
        "accelerate==1.11.0",
        "huggingface-hub==0.36.0",
        "hf_transfer",
        # Vision/Video processing
        "opencv-python==4.10.0.84",
        "opencv-python-headless",
        "pillow==12.0.0",
        "imageio==2.37.0",
        "imageio-ffmpeg",
        "av",
        "decord==0.6.0",
        # Scientific computing
        "numpy==1.26.4",
        "scipy>=1.10.0",  # Compatible with Python 3.10
        "scikit-learn>=1.3.0",  # Compatible with Python 3.10
        "scikit-image>=0.21.0",  # Compatible with Python 3.10
        # Model-specific
        "timm==0.4.12",
        "natsort==8.4.0",
        "einops==0.4.1",
        "kornia==0.8.1",
        # SAM (Segment Anything)
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
        # XMem dependencies
        "progressbar2",
        "gdown",
        "gitpython",
        "hickle==5.0.3",
        "tensorboard",
        # Thin plate spline for XMem
        "git+https://github.com/cheind/py-thin-plate-spline",
        # BLIP2 dependencies
        "sentencepiece",
        # Utilities
        "tqdm",
        "requests",
        "pyyaml",
        "packaging",
        # Flash-attn build dependencies
        "wheel",
        "setuptools",
        "ninja",
        # LoFTR dependencies
        "yacs==0.1.8",
        # Gemini API (for Gemini-based evaluations)
        "google-genai>=1.0.0",
        # Visualization and geometry
        "matplotlib",
        "shapely",
        # Text processing
        "ftfy",
        "regex",
    )
    # Install flash-attn (required for VideoChat model)
    # Use version 2.7.3 which is compatible with PyTorch 2.x and CUDA 12.4
    .run_commands(
        # Try installing flash-attn with specific version first
        "pip install flash-attn==2.7.3 --no-build-isolation 2>&1 || pip install flash-attn --no-build-isolation || echo 'flash-attn installation failed'",
        # Verify flash_attn can be imported
        "python -c 'import flash_attn; print(f\"flash_attn version: {flash_attn.__version__}\")' || echo 'WARNING: flash_attn import failed'",
    )
    # Set environment variables
    .env(
        {
            "HF_HOME": "/weights/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/weights/.cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    # Add the project directory (only scripts and source code, not data)
    .add_local_dir(
        ".",
        remote_path="/app",
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
            # Exclude large files - they go to volume
            "sam_vit_h_4b8939.pth",
            "XMem/saves/*.pth",
            "loftr_weights/*.ckpt",
            # Exclude data directories - they're too large for image
            "lgie-eval-data/",
            "lgie-eval-data-backup/",
            "scripts-backup/",
            # Exclude output directories
            "R2_Output/",
            "R3_Output/",
            "object_change_detection_logs/",
        ],
    )
)

# ============================================================================
# Alternative: Lighter image without local dirs (for faster builds)
# Use this if you want to upload data separately to volume
# ============================================================================
image_light = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    .apt_install(
        "git",
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libgl1-mesa-glx",
        "libglib2.0-0",
    )
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0", 
        "torchaudio==2.6.0",
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
        "scipy==1.16.3",
        "scikit-learn==1.7.2",
        "scikit-image==0.25.2",
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
    )
    .run_commands(
        "pip install flash-attn --no-build-isolation || echo 'flash-attn installation failed, continuing without it'"
    )
    .env(
        {
            "HF_HOME": "/weights/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/weights/.cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

# HuggingFace secret for model downloads
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)


# ============================================================================
# Weight and Data Upload Functions
# ============================================================================
# Note: Due to Modal's deprecation of Mounts, we use a simpler approach:
# - Weights and data are uploaded using Modal's volume put commands from CLI
# - Or by running upload functions that copy from local mounts

# Simple base image for upload operations
image_upload = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    .apt_install("git")
    .add_local_file(
        "sam_vit_h_4b8939.pth",
        remote_path="/upload/sam_vit_h_4b8939.pth",
    )
    .add_local_dir(
        "XMem/saves",
        remote_path="/upload/XMem",
        ignore=["*.git*"],
    )
    .add_local_dir(
        "loftr_weights",
        remote_path="/upload/loftr_weights",
        ignore=["*.git*"],
    )
)

# Image for uploading evaluation data (separate to avoid timeout)
image_data_upload = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    .add_local_dir(
        "lgie-eval-data",
        remote_path="/upload/lgie-eval-data",
        ignore=["*.git*", "__pycache__"],
    )
)


@app.function(
    image=image_upload,
    volumes={"/weights": weights_volume},
    timeout=3600,
)
def upload_weights():
    """
    Upload large model weights to Modal volume.
    Run this once before running evaluations.
    """
    import shutil
    
    print("Uploading model weights to Modal volume...")
    
    weights_dir = Path("/weights")
    weights_dir.mkdir(exist_ok=True)
    
    # Track what was uploaded
    uploaded = []
    
    # SAM checkpoint
    sam_src = Path("/upload/sam_vit_h_4b8939.pth")
    sam_dst = weights_dir / "sam_vit_h_4b8939.pth"
    if sam_src.exists() and not sam_dst.exists():
        print(f"Uploading SAM checkpoint ({sam_src.stat().st_size / 1e9:.2f} GB)...")
        shutil.copy2(sam_src, sam_dst)
        uploaded.append("SAM checkpoint")
    elif sam_dst.exists():
        print("SAM checkpoint already exists in volume")
    else:
        print("WARNING: SAM checkpoint not found at /upload/sam_vit_h_4b8939.pth")
    
    # XMem weights
    xmem_src_dir = Path("/upload/XMem")
    xmem_dst_dir = weights_dir / "XMem"
    xmem_dst_dir.mkdir(exist_ok=True)
    
    if xmem_src_dir.exists():
        for pth_file in xmem_src_dir.glob("*.pth"):
            dst_file = xmem_dst_dir / pth_file.name
            if not dst_file.exists():
                print(f"Uploading XMem weight: {pth_file.name}")
                shutil.copy2(pth_file, dst_file)
                uploaded.append(f"XMem/{pth_file.name}")
            else:
                print(f"XMem weight already exists: {pth_file.name}")
    else:
        print("WARNING: XMem weights directory not found at /upload/XMem")
    
    # LoFTR weights
    loftr_src_dir = Path("/upload/loftr_weights")
    loftr_dst_dir = weights_dir / "loftr_weights"
    loftr_dst_dir.mkdir(exist_ok=True)
    
    if loftr_src_dir.exists():
        for ckpt_file in loftr_src_dir.glob("*.ckpt"):
            dst_file = loftr_dst_dir / ckpt_file.name
            if not dst_file.exists():
                print(f"Uploading LoFTR weight: {ckpt_file.name}")
                shutil.copy2(ckpt_file, dst_file)
                uploaded.append(f"loftr_weights/{ckpt_file.name}")
            else:
                print(f"LoFTR weight already exists: {ckpt_file.name}")
    else:
        print("WARNING: LoFTR weights directory not found at /upload/loftr_weights")
    
    weights_volume.commit()
    
    print("\n" + "=" * 60)
    print("Weight upload complete!")
    print(f"Uploaded: {uploaded if uploaded else 'Nothing new'}")
    print("=" * 60)
    
    return {"status": "success", "uploaded": uploaded}


@app.function(
    image=image_data_upload,
    volumes={"/data": data_volume},
    timeout=3600,
)
def upload_data():
    """
    Upload evaluation data to Modal volume.
    Run this once before running evaluations.
    """
    import shutil
    
    print("Uploading evaluation data to Modal volume...")
    
    data_dir = Path("/data")
    data_dir.mkdir(exist_ok=True)
    
    # Track what was uploaded
    uploaded = []
    
    # Copy lgie-eval-data
    src_dir = Path("/upload/lgie-eval-data")
    dst_dir = data_dir / "lgie-eval-data"
    
    if src_dir.exists():
        if dst_dir.exists():
            print("Evaluation data already exists in volume, updating...")
            shutil.rmtree(dst_dir)
        print(f"Copying evaluation data ({sum(1 for _ in src_dir.rglob('*') if _.is_file())} files)...")
        shutil.copytree(src_dir, dst_dir)
        uploaded.append("lgie-eval-data")
    else:
        print("WARNING: Evaluation data not found at /upload/lgie-eval-data")
    
    data_volume.commit()
    
    print("\n" + "=" * 60)
    print("Data upload complete!")
    print(f"Uploaded: {uploaded if uploaded else 'Nothing new'}")
    print("=" * 60)
    
    return {"status": "success", "uploaded": uploaded}


@app.function(
    image=image,
    volumes={"/weights": weights_volume},
    timeout=300,
)
def check_weights():
    """Check what weights are available in the Modal volume."""
    weights_dir = Path("/weights")
    
    if not weights_dir.exists():
        return {"status": "empty", "message": "Weights volume is empty. Run --action upload first."}
    
    files = []
    for f in weights_dir.rglob("*"):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            files.append({"path": str(f.relative_to(weights_dir)), "size_mb": round(size_mb, 2)})
    
    return {"status": "success", "files": files}


# ============================================================================
# Main Evaluation Runner
# ============================================================================
@app.function(
    image=image,
    gpu=DEFAULT_GPU_TYPE,  # Default GPU, can be overridden via with_options()
    secrets=[huggingface_secret],
    volumes={
        "/weights": weights_volume,
        "/data": data_volume,
    },
    timeout=86400,  # 24 hour timeout for long evaluations
    memory=65536,   # 64 GB RAM
)
def run_evaluation(script_name: str, verbose: bool = False, extra_args: str = ""):
    """
    Run an evaluation script on Modal.
    
    Args:
        script_name: Name of the script to run (e.g., "R1.py")
        verbose: Enable verbose output
        extra_args: Additional command-line arguments to pass to the script
    """
    import subprocess
    import sys
    
    # Set up environment
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")
    
    # Validate script name
    if script_name not in VALID_SCRIPTS:
        raise ValueError(
            f"Invalid script: {script_name}. "
            f"Valid scripts: {VALID_SCRIPTS}"
        )
    
    # Set up symlinks for weights (so scripts can find them at expected paths)
    _setup_weight_symlinks()
    
    # Set up PYTHONPATH for XMem imports
    xmem_path = "/app/XMem"
    loftr_path = "/app"
    
    env = os.environ.copy()
    pythonpath = f"{xmem_path}:{loftr_path}"
    if "PYTHONPATH" in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    
    # Build the command
    script_path = f"/app/{script_name}"
    cmd = [sys.executable, script_path]
    
    if extra_args:
        cmd.extend(extra_args.split())
    
    print("=" * 60)
    print(f"Running: {script_name}")
    print(f"Python: {sys.executable}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    print("=" * 60)
    
    # Show GPU info
    try:
        gpu_info = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        print(f"GPU: {gpu_info.stdout.strip()}")
    except Exception:
        pass
    
    print("=" * 60 + "\n")
    
    # Run the script, streaming output to terminal
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd="/app",
        bufsize=1,
        universal_newlines=True,
    )
    
    # Stream output line by line
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)
        output_lines.append(line)
    
    process.wait()
    
    print("\n" + "=" * 60)
    print(f"Script {script_name} completed with exit code: {process.returncode}")
    print("=" * 60)
    
    # Commit the data volume to persist outputs
    data_volume.commit()
    
    return {
        "status": "success" if process.returncode == 0 else "failed",
        "exit_code": process.returncode,
        "script": script_name,
    }


def _setup_weight_symlinks():
    """Create symlinks so scripts can find weights at expected paths."""
    import os
    
    # SAM checkpoint
    sam_src = Path("/weights/sam_vit_h_4b8939.pth")
    sam_dst = Path("/app/sam_vit_h_4b8939.pth")
    if sam_src.exists() and not sam_dst.exists():
        os.symlink(sam_src, sam_dst)
        print(f"Linked: {sam_dst} -> {sam_src}")
    
    # XMem weights
    xmem_src_dir = Path("/weights/XMem")
    xmem_dst_dir = Path("/app/XMem/saves")
    xmem_dst_dir.mkdir(parents=True, exist_ok=True)
    
    if xmem_src_dir.exists():
        for pth_file in xmem_src_dir.glob("*.pth"):
            dst_file = xmem_dst_dir / pth_file.name
            if not dst_file.exists():
                os.symlink(pth_file, dst_file)
                print(f"Linked: {dst_file} -> {pth_file}")
    
    # LoFTR weights
    loftr_src_dir = Path("/weights/loftr_weights")
    loftr_dst_dir = Path("/app/loftr_weights")
    loftr_dst_dir.mkdir(parents=True, exist_ok=True)
    
    if loftr_src_dir.exists():
        for ckpt_file in loftr_src_dir.glob("*.ckpt"):
            dst_file = loftr_dst_dir / ckpt_file.name
            if not dst_file.exists():
                os.symlink(ckpt_file, dst_file)
                print(f"Linked: {dst_file} -> {ckpt_file}")
    
    # Evaluation data
    data_src = Path("/data/lgie-eval-data")
    data_dst = Path("/app/lgie-eval-data")
    if data_src.exists() and not data_dst.exists():
        os.symlink(data_src, data_dst)
        print(f"Linked: {data_dst} -> {data_src}")
    elif not data_src.exists():
        print("WARNING: Evaluation data not found in volume. Run --action upload_data first.")


# ============================================================================
# Output Download Function
# ============================================================================
@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=3600,
)
def list_outputs():
    """List all output files in the data volume."""
    data_dir = Path("/data")
    
    if not data_dir.exists():
        return {"status": "empty", "message": "Data volume is empty"}
    
    files = []
    for f in data_dir.rglob("*"):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            files.append({
                "path": str(f.relative_to(data_dir)),
                "size_mb": round(size_mb, 2)
            })
    
    return {"status": "success", "files": files}


# ============================================================================
# Local Entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    script: str = None,
    action: str = "run",
    verbose: bool = False,
    extra_args: str = "",
):
    """
    Run LGIE evaluation scripts on Modal.
    
    Args:
        script: Name of the evaluation script to run (e.g., R1.py, R2.py)
        action: Action to perform: "run", "list", "upload", "upload_data", "check_weights", "list_outputs"
        verbose: Enable verbose output
        extra_args: Additional arguments to pass to the script
    """
    print("=" * 70)
    print("LGIE Evaluation Runner - Modal Wrapper")
    print("=" * 70)
    
    if action == "list":
        print("\nAvailable evaluation scripts:")
        for s in VALID_SCRIPTS:
            print(f"  - {s}")
        print("\nUsage: modal run modal_eval_runner.py --script <script_name>")
        return
    
    elif action == "upload":
        print("\nUploading model weights to Modal volume...")
        result = upload_weights.remote()
        print(f"\nResult: {result}")
        return
    
    elif action == "upload_data":
        print("\nUploading evaluation data to Modal volume...")
        result = upload_data.remote()
        print(f"\nResult: {result}")
        return
    
    elif action == "check_weights":
        print("\nChecking weights in Modal volume...")
        result = check_weights.remote()
        if result["status"] == "success":
            print("\nWeights available:")
            for f in result["files"]:
                print(f"  {f['path']} ({f['size_mb']} MB)")
        else:
            print(f"\n{result['message']}")
        return
    
    elif action == "list_outputs":
        print("\nListing outputs in Modal data volume...")
        result = list_outputs.remote()
        if result["status"] == "success":
            print("\nOutput files:")
            for f in result["files"]:
                print(f"  {f['path']} ({f['size_mb']} MB)")
        else:
            print(f"\n{result['message']}")
        return
    
    elif action == "run":
        if not script:
            print("\nError: --script is required for 'run' action")
            print(f"Valid scripts: {VALID_SCRIPTS}")
            print("\nExample: modal run modal_eval_runner.py --script R1.py")
            return
        
        # Determine GPU type for this script
        gpu_type = GPU_CONFIG.get(script, DEFAULT_GPU_TYPE)
        
        print(f"\nRunning evaluation script: {script}")
        print(f"GPU Type: {gpu_type}")
        print(f"Verbose: {verbose}")
        if extra_args:
            print(f"Extra args: {extra_args}")
        print("-" * 70 + "\n")
        
        # Use with_options to set the appropriate GPU for this script
        result = run_evaluation.with_options(gpu=gpu_type).remote(
            script_name=script,
            verbose=verbose,
            extra_args=extra_args,
        )
        
        print("\n" + "-" * 70)
        print(f"Final Result: {result}")
        return result
    
    else:
        print(f"\nError: Unknown action '{action}'")
        print("Valid actions: run, list, upload, upload_data, check_weights, list_outputs")
        return


# ============================================================================
# Deployment for continuous running (optional)
# ============================================================================
@app.function(
    image=image,
    gpu=DEFAULT_GPU_TYPE,  # Default GPU, can be overridden via with_options()
    secrets=[huggingface_secret],
    volumes={
        "/weights": weights_volume,
        "/data": data_volume,
    },
    timeout=86400,
    memory=65536,
    min_containers=1,  # Keep one instance warm for faster cold starts (renamed from keep_warm)
)
def run_evaluation_warm(script_name: str, verbose: bool = False, extra_args: str = ""):
    """
    Same as run_evaluation but with min_containers=1 for faster subsequent runs.
    Use this for development/debugging when running many evaluations.
    """
    return run_evaluation.local(script_name, verbose, extra_args)
