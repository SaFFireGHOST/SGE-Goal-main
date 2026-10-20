"""
Modal script for Flux 2 Klein ERM (Final Edit Region) mask prediction inference.

Runs the trained LoRA mask-prediction pipeline on a folder of test cases,
saving the predicted final_edit_region_mask.jpg into each test-case sub-directory.

Usage (from workspace root, with the lgie conda environment activated):

    modal run fer-mask-prediction-code/modal_flux2_klein_erm_inference.py \
        --data-dir ./dummy_test_data \
        --checkpoint-dir ./fer-mask-prediction-checkpoint

Workflow:
  1. Uploads the test-data folder and the best checkpoint to a Modal volume.
  2. Runs batch inference on a GPU container (all test cases in one call).
  3. Downloads the processed test-data folder (now containing predictions)
     and replaces the local copy.
  4. Deletes the ephemeral Modal volume so future runs start fresh.
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
SEED = int(os.environ.get("SGE_GEN_SEED", "0"))

# LoRA hyper-parameters (must match training)
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
# GPU Configuration
# ============================================================================
GPU_TYPE = "L40S"
GPU_CONFIG = f"{GPU_TYPE}:1"

# ============================================================================
# Modal App Setup
# ============================================================================
app = modal.App(os.environ.get("FER_MODAL_APP_NAME", "flux2-klein-fer-erm-inference"))

VOLUME_NAME = os.environ.get("FER_MODAL_VOLUME_NAME", "flux2-klein-fer-erm-inference-volume")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    # pin torch stack to the CUDA-12.8 build matching the base image; unpinned
    # torchvision/torchaudio otherwise pull CUDA-13 wheels -> `libcudart.so.13`
    # import error.
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
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
            "SGE_GEN_SEED": os.environ.get("SGE_GEN_SEED", "0"),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/data/.cache",
        }
    )
)

huggingface_secret = modal.Secret.from_name("huggingface-secret")

# ============================================================================
# ERM Prompt  (must match training script exactly)
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
    """Build a mask-prediction prompt with a fixed ERM instruction suffix."""
    return f"{edit_instruction}\n\n{DEFAULT_ERM_INSTRUCTION.strip()}"


# ============================================================================
# Remote Batch Inference
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=86400,
    memory=65536,
)
def run_batch_inference():
    """Run the FER mask prediction pipeline on every test case in /data/test_data/."""

    import importlib
    import torch
    from PIL import Image as PILImage
    from peft import LoraConfig, set_peft_model_state_dict
    from safetensors.torch import load_file as st_load
    import diffusers

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    # ------------------------------------------------------------------
    #  Dynamic import of Flux2KleinPipeline (name may live in different
    #  sub‑modules depending on the diffusers build).
    # ------------------------------------------------------------------
    def _import_or_get(module: str, name: str):
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
        raise ImportError(
            "Flux2KleinPipeline not found in the installed diffusers build. "
            "Make sure you are using git+https://github.com/huggingface/diffusers.git"
        )

    # ------------------------------------------------------------------
    #  Paths on the Modal volume (mounted at /data)
    # ------------------------------------------------------------------
    data_dir = "/data/test_data"
    checkpoint_file = "/data/checkpoint/transformer_lora.safetensors"

    # Reload volume to pick up freshly-uploaded data
    try:
        volume.reload()
    except Exception:
        pass

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Test data directory not found on volume: {data_dir}")
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"LoRA checkpoint not found on volume: {checkpoint_file}")

    # ------------------------------------------------------------------
    #  Load base pipeline
    # ------------------------------------------------------------------
    print("Loading Flux 2 Klein base pipeline …")
    t0 = time.time()
    pipeline = Flux2KleinPipeline.from_pretrained(
        PRETRAINED_MODEL,
        torch_dtype=torch.bfloat16,
    )
    print(f"  Base pipeline loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    #  Attach LoRA adapter (must match training config exactly)
    # ------------------------------------------------------------------
    print("Adding LoRA adapter …")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        init_lora_weights="gaussian",
        target_modules=LORA_TARGET_MODULES,
    )
    pipeline.transformer.add_adapter(lora_config)

    print(f"Loading LoRA weights from {checkpoint_file} …")
    lora_state = st_load(checkpoint_file, device="cpu")
    incompatible = set_peft_model_state_dict(
        pipeline.transformer, lora_state, adapter_name="default"
    )
    if incompatible and getattr(incompatible, "unexpected_keys", None):
        print(f"  Warning – unexpected LoRA keys: {incompatible.unexpected_keys}")
    print("  LoRA weights loaded.")

    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    # ------------------------------------------------------------------
    #  Discover test cases
    # ------------------------------------------------------------------
    subdirs = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))],
        key=lambda x: (int(x) if x.isdigit() else float("inf"), x),
    )
    print(f"Found {len(subdirs)} test-case sub-directory(ies): {subdirs}")

    processed = 0
    skipped = 0

    for idx, subdir in enumerate(subdirs):
        case_dir = os.path.join(data_dir, subdir)

        original_path = os.path.join(case_dir, "original_image.jpg")
        init_mask_path = os.path.join(case_dir, "initial_edit_region_mask.jpg")
        instruction_path = os.path.join(case_dir, "edit_instruction.txt")

        # Check required files
        missing = [
            name
            for path, name in [
                (original_path, "original_image.jpg"),
                (init_mask_path, "initial_edit_region_mask.jpg"),
                (instruction_path, "edit_instruction.txt"),
            ]
            if not os.path.exists(path)
        ]
        if missing:
            print(f"  [{subdir}] SKIP – missing: {', '.join(missing)}")
            skipped += 1
            continue

        # Skip if already processed (resume support)
        output_path = os.path.join(case_dir, "final_edit_region_mask.jpg")
        if os.path.exists(output_path):
            print(f"  [{subdir}] SKIP – final_edit_region_mask.jpg already exists (resume)")
            skipped += 1
            continue

        # Load inputs
        original = PILImage.open(original_path).convert("RGB")
        init_mask = PILImage.open(init_mask_path).convert("RGB")
        with open(instruction_path, "r") as f:
            edit_instruction = f.read().strip()

        prompt = build_erm_prompt(edit_instruction)

        # Match validation code: use native resolution rounded down to
        # the nearest multiple of 16 (VAE scale_factor=8 × patchify=2).
        orig_w, orig_h = original.size
        gen_h = (orig_h // 16) * 16
        gen_w = (orig_w // 16) * 16

        original = original.resize((gen_w, gen_h), resample=PILImage.BILINEAR)
        init_mask = init_mask.resize((gen_w, gen_h), resample=PILImage.NEAREST)

        blank = PILImage.new("RGB", (gen_w, gen_h), color=(0, 0, 0))
        cond_images = [original, init_mask, blank]

        seed_val = SEED + (int(subdir) if subdir.isdigit() else idx)
        generator = torch.Generator(device="cuda").manual_seed(seed_val)

        # Run pipeline (matches validation / test visualisation path)
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

        output_path = os.path.join(case_dir, "final_edit_region_mask.jpg")
        pred.save(output_path, quality=95)
        processed += 1
        print(f"  [{subdir}] Saved final_edit_region_mask.jpg  "
              f"({gen_w}×{gen_h}, {elapsed:.1f}s)")

        # Periodic volume commit so progress survives container kills
        if processed % 25 == 0:
            print(f"  [Checkpoint] Committing volume after {processed} cases …")
            volume.commit()

    print(f"\nBatch inference complete.  Processed: {processed}  Skipped: {skipped}")

    # Final commit to persist any remaining results
    volume.commit()

    return {
        "status": "success",
        "processed": processed,
        "skipped": skipped,
        "total": len(subdirs),
    }


# ============================================================================
# Helpers
# ============================================================================

def _find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Return the path to the most recent checkpoint in *checkpoint_dir*.

    Handles three layouts:
      • The directory itself contains ``transformer_lora.safetensors``.
      • The directory contains ``checkpoint-epoch-N/`` sub-directories.
      • The directory contains arbitrarily-named sub-directories with
        ``transformer_lora.safetensors`` inside them.
    """
    direct = os.path.join(checkpoint_dir, "transformer_lora.safetensors")
    if os.path.isfile(direct):
        return checkpoint_dir

    # Look for ANY subdirectory that contains the weights file
    ckpt_subdirs = sorted(
        [
            d
            for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d))
            and os.path.isfile(os.path.join(checkpoint_dir, d, "transformer_lora.safetensors"))
        ],
    )
    if not ckpt_subdirs:
        raise FileNotFoundError(
            f"No sub-directories containing transformer_lora.safetensors "
            f"found in {checkpoint_dir}"
        )

    latest = ckpt_subdirs[-1]
    ckpt_path = os.path.join(checkpoint_dir, latest)
    return ckpt_path


def _validate_test_data(data_dir: str) -> tuple[int, int]:
    """Print per-case readiness summary.  Returns (ready, not_ready) counts."""
    subdirs = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))],
        key=lambda x: (int(x) if x.isdigit() else float("inf"), x),
    )

    ready = 0
    not_ready = 0
    for sd in subdirs:
        sd_path = os.path.join(data_dir, sd)
        has_orig = os.path.isfile(os.path.join(sd_path, "original_image.jpg"))
        has_mask = os.path.isfile(os.path.join(sd_path, "initial_edit_region_mask.jpg"))
        has_instr = os.path.isfile(os.path.join(sd_path, "edit_instruction.txt"))
        if has_orig and has_mask and has_instr:
            ready += 1
        else:
            not_ready += 1
            missing = []
            if not has_orig:
                missing.append("original_image.jpg")
            if not has_mask:
                missing.append("initial_edit_region_mask.jpg")
            if not has_instr:
                missing.append("edit_instruction.txt")
            print(f"  ⚠  {sd}/ missing: {', '.join(missing)}")

    return ready, not_ready


# ============================================================================
# Local Entrypoint
# ============================================================================
@app.local_entrypoint()
def main(
    data_dir: str = "./dummy_test_data",
    checkpoint_dir: str = "./fer-mask-prediction-checkpoint",
):
    """
    Run FER mask prediction on every test case in *data_dir*.

    Args:
        data_dir:       Path to the test-data folder (containing numbered
                        sub-directories, each with original_image.jpg,
                        initial_edit_region_mask.jpg, edit_instruction.txt).
        checkpoint_dir: Path to the checkpoint folder.  May contain
                        ``checkpoint-epoch-N/`` sub-directories or a single
                        ``transformer_lora.safetensors`` file directly.
    """
    import shutil
    import subprocess as sp

    data_dir = os.path.abspath(data_dir)
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # ------------------------------------------------------------------
    #  Resolve checkpoint
    # ------------------------------------------------------------------
    ckpt_path = _find_latest_checkpoint(checkpoint_dir)
    ckpt_name = os.path.basename(ckpt_path) if ckpt_path != checkpoint_dir else "(root)"
    print(f"Checkpoint : {ckpt_path}  ({ckpt_name})")

    # ------------------------------------------------------------------
    #  Validate test data
    # ------------------------------------------------------------------
    print(f"\nTest data  : {data_dir}")
    ready, not_ready = _validate_test_data(data_dir)
    total = ready + not_ready
    print(f"  {ready}/{total} test cases ready for inference")
    if ready == 0:
        raise RuntimeError(
            "No test cases are ready (all missing at least one required file). "
            "Ensure each sub-directory has original_image.jpg, "
            "initial_edit_region_mask.jpg, and edit_instruction.txt."
        )

    # ------------------------------------------------------------------
    #  Upload to Modal volume (--force to handle re-uploads on retry)
    # ------------------------------------------------------------------
    print("\n>>> Uploading test data and checkpoint to Modal volume …")
    t0 = time.time()
    modal_cmd = [sys.executable, "-m", "modal"]
    for label, local_path, remote_path in [
        ("test data", data_dir, "test_data"),
        ("checkpoint", ckpt_path, "checkpoint"),
    ]:
        print(f"    Uploading {label}: {local_path} → {remote_path}")
        up = sp.run(
            modal_cmd + ["volume", "put", "--force", VOLUME_NAME, local_path, remote_path],
            capture_output=True, text=True,
        )
        if up.returncode != 0:
            raise RuntimeError(
                f"Failed to upload {label} (exit {up.returncode}): "
                f"{(up.stderr or '').strip()[:500]}"
            )
    print(f">>> Upload complete ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    #  Run remote inference
    # ------------------------------------------------------------------
    print("\n>>> Launching batch inference on Modal …")
    t0 = time.time()
    result = run_batch_inference.remote()
    print(f">>> Inference finished ({time.time() - t0:.1f}s)")
    print(f"    Result: {result}")

    # ------------------------------------------------------------------
    #  Download processed test data from volume
    # ------------------------------------------------------------------
    print("\n>>> Downloading processed test data from Modal volume …")
    parent_dir = os.path.dirname(data_dir)
    tmp_dir = os.path.join(parent_dir, f".tmp_erm_download_{int(time.time())}")
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        "modal", "volume", "get",
        VOLUME_NAME,
        "test_data",
        tmp_dir,
        "--force",
    ]
    print(f"    {' '.join(cmd)}")
    dl = sp.run(cmd, capture_output=True, text=True, timeout=86400)
    if dl.returncode != 0:
        stderr_msg = dl.stderr.strip()[:500] if dl.stderr else "(no stderr)"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(
            f"modal volume get failed (exit {dl.returncode}): {stderr_msg}"
        )
    print(">>> Download complete.")

    # ------------------------------------------------------------------
    #  Determine the actual download root
    #
    #  `modal volume get` may put contents directly in tmp_dir, or may
    #  create a nested `test_data/` sub-directory.  Handle both.
    # ------------------------------------------------------------------
    nested_candidate = os.path.join(tmp_dir, "test_data")
    if os.path.isdir(nested_candidate):
        # Contents were placed inside a "test_data" sub-folder
        actual_download = nested_candidate
    else:
        actual_download = tmp_dir

    # Quick sanity-check: the download root should contain at least one
    # of the expected sub-directories.
    subdirs_in_download = [
        d for d in os.listdir(actual_download)
        if os.path.isdir(os.path.join(actual_download, d))
    ]
    if not subdirs_in_download:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(
            "Downloaded data appears empty – no sub-directories found.  "
            "Check the Modal volume contents manually."
        )
    print(f"    Downloaded {len(subdirs_in_download)} sub-directory(ies)")

    # ------------------------------------------------------------------
    #  Replace local data directory
    # ------------------------------------------------------------------
    print(f"\n>>> Replacing local data directory: {data_dir}")
    backup_dir = f"{data_dir}.backup_{int(time.time())}"
    try:
        os.rename(data_dir, backup_dir)
        os.rename(actual_download, data_dir)
        # Clean up the (now possibly-empty) tmp_dir wrapper if the
        # actual download was a nested sub-directory.
        if actual_download != tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        # Remove backup
        shutil.rmtree(backup_dir, ignore_errors=True)
        print(">>> Local data directory replaced successfully.")
    except Exception as exc:
        # Best-effort rollback
        if os.path.isdir(backup_dir) and not os.path.isdir(data_dir):
            os.rename(backup_dir, data_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to replace local data directory: {exc}") from exc

    # ------------------------------------------------------------------
    #  Delete the ephemeral Modal volume
    # ------------------------------------------------------------------
    print("\n>>> Deleting Modal volume …")
    del_result = sp.run(
        ["modal", "volume", "delete", VOLUME_NAME, "--yes"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if del_result.returncode == 0:
        print(">>> Modal volume deleted.")
    else:
        # Try alternate CLI verb
        del_result2 = sp.run(
            ["modal", "volume", "rm", VOLUME_NAME, "--yes"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if del_result2.returncode == 0:
            print(">>> Modal volume deleted (via 'rm').")
        else:
            print(
                f">>> Warning: could not delete volume "
                f"(exit codes: delete={del_result.returncode}, rm={del_result2.returncode}).  "
                f"Delete manually with: modal volume delete {VOLUME_NAME}"
            )

    # ------------------------------------------------------------------
    #  Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  FER MASK PREDICTION INFERENCE COMPLETE")
    print(f"  Processed : {result.get('processed', '?')}")
    print(f"  Skipped   : {result.get('skipped', '?')}")
    print(f"  Output    : {data_dir}")
    print("=" * 60)
