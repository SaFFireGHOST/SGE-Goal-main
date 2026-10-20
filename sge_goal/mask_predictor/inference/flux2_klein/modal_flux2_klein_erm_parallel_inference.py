"""
Parallel Modal script for Flux 2 Klein ERM (Final Edit Region) mask prediction.

Partitions 2000 test cases into 50 groups of 40, each running on a separate
B200 GPU container.  Every container has resume support – it skips cases whose
final_edit_region_mask.jpg already exists on the shared Modal volume.

Usage:
    modal run fer-mask-prediction-code/modal_flux2_klein_erm_parallel_inference.py \
        --data-dir ./bridge_data_v2_tel_test_data \
        --checkpoint-dir ./bridge-fer-mask-prediction-checkpoint/klein-fer-checkpoint-epoch-17
"""

import modal
import os
import sys
import time

# ============================================================================
# Pipeline / Model Configuration
# ============================================================================
PRETRAINED_MODEL = "black-forest-labs/FLUX.2-klein-base-9B"
GUIDANCE_SCALE = 4.0
NUM_INFERENCE_STEPS = 50
SEED = 0

LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "proj_mlp",
]

# ============================================================================
# Parallelism Configuration
# ============================================================================
NUM_WORKERS = 50

# Hardcoded partition: 2000 test cases (numbered 1..2000), 50 groups of 40.
# WORKER_ASSIGNMENTS[i] is the sorted list of test-case directory names for
# worker i  (i = 0 … 49).
_ALL_CASES = list(range(1, 2001))
WORKER_ASSIGNMENTS = []
_GROUP_SIZE = len(_ALL_CASES) // NUM_WORKERS          # 40
for _i in range(NUM_WORKERS):
    _start = _i * _GROUP_SIZE
    _end = _start + _GROUP_SIZE if _i < NUM_WORKERS - 1 else len(_ALL_CASES)
    WORKER_ASSIGNMENTS.append([str(c) for c in _ALL_CASES[_start:_end]])

# ============================================================================
# GPU Configuration
# ============================================================================
GPU_TYPE = "B200"
GPU_CONFIG = f"{GPU_TYPE}:1"

# ============================================================================
# Modal App Setup
# ============================================================================
APP_NAME = "klein-opt-fer-parallel-inference"
VOLUME_NAME = "klein-opt-fer-parallel-inference-vol"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

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
    .run_commands("apt-get update && apt-get install -y git")
    .run_commands("pip install --upgrade pip setuptools wheel")
    .run_commands("pip install git+https://github.com/huggingface/diffusers.git")
    .run_commands("pip install flash-attn --no-build-isolation")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/data/.cache",
        }
    )
)

huggingface_secret = modal.Secret.from_name("huggingface-secret")

# ============================================================================
# ERM Prompt
# ============================================================================
DEFAULT_ERM_INSTRUCTION = """TASK: Edit Region Mask Prediction for Robotic Manipulation

You are provided with the following inputs:
1. ORIGINAL IMAGE: A scene containing objects, including a robotic arm and dynamic objects that can be manipulated.
2. INITIAL EDIT REGION MASK: A binary mask (white=1, black=0) showing the current locations of dynamic objects in the original image. White pixels indicate regions occupied by objects that will be manipulated/moved.
3. EDIT INSTRUCTION (above): A text description of the manipulation/edit to perform on the dynamic objects.

YOUR TASK: Generate a FINAL EDIT REGION MASK - a binary mask (white=1, black=0) predicting where the dynamic objects should appear AFTER the edit instruction is executed.
- White pixels (1): Regions where dynamic objects will be located after manipulation
- Black pixels (0): Background regions that remain unchanged

IMPORTANT:
- Focus ONLY on predicting object locations, not colors or textures
- The mask should be consistent across all RGB channels
- Use the initial mask to understand which objects are dynamic
- Use the edit instruction to determine the target locations
- Output should be a clean binary mask showing final object positions"""


def build_erm_prompt(edit_instruction: str) -> str:
    return f"{edit_instruction}\n\n{DEFAULT_ERM_INSTRUCTION.strip()}"


# ============================================================================
# Per-Worker Remote Function
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=86400,
    memory=65536,
)
def run_worker(worker_id: int):
    """Run FER inference for a specific partition of test cases."""

    import importlib
    import torch
    from PIL import Image as PILImage
    from peft import LoraConfig, set_peft_model_state_dict
    from safetensors.torch import load_file as st_load
    import diffusers

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    case_names = WORKER_ASSIGNMENTS[worker_id]
    tag = f"[Worker {worker_id:02d}]"
    print(f"{tag} Assigned {len(case_names)} cases: {case_names[0]}..{case_names[-1]}")

    # Dynamic import
    def _import_or_get(module, name):
        try:
            return getattr(importlib.import_module(module), name)
        except Exception:
            return None

    Flux2KleinPipeline = (
        getattr(diffusers, "Flux2KleinPipeline", None)
        or _import_or_get("diffusers.pipelines", "Flux2KleinPipeline")
        or _import_or_get("diffusers.pipelines.flux_2_klein", "Flux2KleinPipeline")
    )
    if Flux2KleinPipeline is None:
        raise ImportError("Flux2KleinPipeline not found in diffusers")

    data_dir = "/data/test_data"
    checkpoint_file = "/data/checkpoint/transformer_lora.safetensors"

    try:
        volume.reload()
    except Exception:
        pass

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Test data directory not found: {data_dir}")
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    # Load pipeline
    print(f"{tag} Loading base pipeline …")
    t0 = time.time()
    pipeline = Flux2KleinPipeline.from_pretrained(
        PRETRAINED_MODEL, torch_dtype=torch.bfloat16,
    )
    print(f"{tag} Base pipeline loaded in {time.time() - t0:.1f}s")

    # Attach LoRA
    print(f"{tag} Adding LoRA adapter …")
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        init_lora_weights="gaussian", target_modules=LORA_TARGET_MODULES,
    )
    pipeline.transformer.add_adapter(lora_config)
    lora_state = st_load(checkpoint_file, device="cpu")
    incompatible = set_peft_model_state_dict(
        pipeline.transformer, lora_state, adapter_name="default",
    )
    if incompatible and getattr(incompatible, "unexpected_keys", None):
        print(f"{tag} Warning – unexpected keys: {incompatible.unexpected_keys}")
    print(f"{tag} LoRA loaded.")

    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    processed = 0
    skipped_resume = 0
    skipped_missing = 0

    for case_idx, subdir in enumerate(case_names):
        case_dir = os.path.join(data_dir, subdir)
        if not os.path.isdir(case_dir):
            print(f"{tag} [{subdir}] SKIP – directory missing")
            skipped_missing += 1
            continue

        original_path = os.path.join(case_dir, "original_image.jpg")
        init_mask_path = os.path.join(case_dir, "initial_edit_region_mask.jpg")
        instruction_path = os.path.join(case_dir, "edit_instruction.txt")

        missing = [
            name for path, name in [
                (original_path, "original_image.jpg"),
                (init_mask_path, "initial_edit_region_mask.jpg"),
                (instruction_path, "edit_instruction.txt"),
            ] if not os.path.exists(path)
        ]
        if missing:
            print(f"{tag} [{subdir}] SKIP – missing: {', '.join(missing)}")
            skipped_missing += 1
            continue

        output_path = os.path.join(case_dir, "final_edit_region_mask.jpg")
        if os.path.exists(output_path):
            print(f"{tag} [{subdir}] SKIP – already exists (resume)")
            skipped_resume += 1
            continue

        # Load inputs
        original = PILImage.open(original_path).convert("RGB")
        init_mask = PILImage.open(init_mask_path).convert("RGB")
        with open(instruction_path, "r") as f:
            edit_instruction = f.read().strip()

        prompt = build_erm_prompt(edit_instruction)

        orig_w, orig_h = original.size
        gen_h = (orig_h // 16) * 16
        gen_w = (orig_w // 16) * 16

        original = original.resize((gen_w, gen_h), resample=PILImage.BILINEAR)
        init_mask = init_mask.resize((gen_w, gen_h), resample=PILImage.NEAREST)

        blank = PILImage.new("RGB", (gen_w, gen_h), color=(0, 0, 0))
        cond_images = [original, init_mask, blank]

        seed_val = SEED + (int(subdir) if subdir.isdigit() else case_idx)
        generator = torch.Generator(device="cuda").manual_seed(seed_val)

        t1 = time.time()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = pipeline(
                prompt=prompt,
                image=cond_images,
                height=gen_h,
                width=gen_w,
                num_inference_steps=NUM_INFERENCE_STEPS,
                generator=generator,
                guidance_scale=GUIDANCE_SCALE,
            ).images[0]
        elapsed = time.time() - t1

        pred.save(output_path, quality=95)
        processed += 1
        print(f"{tag} [{subdir}] Saved final_edit_region_mask.jpg ({gen_w}x{gen_h}, {elapsed:.1f}s)")

        # Commit every 5 cases so sync script can pick them up in real-time
        if processed % 5 == 0:
            print(f"{tag} Committing volume ({processed} new so far) …")
            volume.commit()

    # Final commit
    volume.commit()
    summary = {
        "worker_id": worker_id,
        "processed": processed,
        "skipped_resume": skipped_resume,
        "skipped_missing": skipped_missing,
        "total_assigned": len(case_names),
    }
    print(f"{tag} Done. {summary}")
    return summary


# ============================================================================
# Local Entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    data_dir: str = "./bridge_data_v2_tel_test_data",
    checkpoint_dir: str = "./bridge-fer-mask-prediction-checkpoint/klein-fer-checkpoint-epoch-17",
):
    import subprocess as sp

    data_dir = os.path.abspath(data_dir)
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Resolve checkpoint file
    ckpt_file = os.path.join(checkpoint_dir, "transformer_lora.safetensors")
    if not os.path.isfile(ckpt_file):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")
    print(f"Checkpoint : {checkpoint_dir}")

    # Validate test data
    subdirs = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))],
        key=lambda x: (int(x) if x.isdigit() else float("inf"), x),
    )
    print(f"Test data  : {data_dir}  ({len(subdirs)} sub-directories)")

    # Upload to volume
    print("\n>>> Uploading test data and checkpoint to Modal volume …")
    t0 = time.time()
    modal_cmd = [sys.executable, "-m", "modal"]
    for label, local_path, remote_path in [
        ("test data", data_dir, "test_data"),
        ("checkpoint", checkpoint_dir, "checkpoint"),
    ]:
        print(f"    Uploading {label}: {local_path} → {remote_path}")
        up = sp.run(
            modal_cmd + ["volume", "put", "--force", VOLUME_NAME, local_path, remote_path],
            capture_output=True, text=True,
        )
        if up.returncode != 0:
            raise RuntimeError(
                f"Upload {label} failed (exit {up.returncode}): "
                f"{(up.stderr or '').strip()[:500]}"
            )
    print(f">>> Upload complete ({time.time() - t0:.1f}s)")

    # Launch all 50 workers in parallel
    print(f"\n>>> Launching {NUM_WORKERS} parallel workers …")
    t0 = time.time()
    handles = []
    for wid in range(NUM_WORKERS):
        handles.append(run_worker.spawn(wid))
    print(f">>> All {NUM_WORKERS} workers spawned.")

    # Collect results
    results = []
    for h in handles:
        results.append(h.get())
    elapsed = time.time() - t0

    total_processed = sum(r["processed"] for r in results)
    total_skipped_resume = sum(r["skipped_resume"] for r in results)
    total_skipped_missing = sum(r["skipped_missing"] for r in results)

    print(f"\n{'='*60}")
    print(f"  PARALLEL FER INFERENCE COMPLETE  ({elapsed:.0f}s)")
    print(f"  Workers        : {NUM_WORKERS}")
    print(f"  Processed      : {total_processed}")
    print(f"  Skipped (resume): {total_skipped_resume}")
    print(f"  Skipped (missing): {total_skipped_missing}")
    print(f"{'='*60}")
