#!/usr/bin/env python3
"""
Batch Inference Script for SuSIE Image Editing Pipeline

Runs SuSIE inference on a structured test data directory.
Each test case subdirectory should contain:
  - original_image.{png,jpg,jpeg}: The starting scene image
  - edit_instruction.txt: The textual manipulation instruction (first line used)

Outputs:
  - edited_image.jpg: The predicted final scene after manipulation,
    saved in each test case's subdirectory.

Usage:
    python run_inference.py --data-dir dummy_test_data
    python run_inference.py --data-dir dummy_test_data --num-iterations 3
    python run_inference.py --data-dir dummy_test_data --use-gpu
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# JAX backend must be configured BEFORE importing JAX or any JAX-dependent code
# ---------------------------------------------------------------------------


def configure_jax_backend(use_gpu: bool):
    if not use_gpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def find_original_image(test_case_dir: Path) -> Path | None:
    """Return the path to the original image in a test case directory."""
    for ext in IMAGE_EXTENSIONS:
        candidate = test_case_dir / f"original_image{ext}"
        if candidate.exists():
            return candidate
    return None


def read_instruction(test_case_dir: Path) -> str | None:
    """Read the first non-empty line of edit_instruction.txt."""
    txt_path = test_case_dir / "edit_instruction.txt"
    if not txt_path.exists():
        return None
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.lower().startswith("confidence"):
                return line
    return None


def discover_test_cases(data_dir: Path) -> list[dict]:
    """
    Discover all valid test-case subdirectories under *data_dir*.

    Returns a list of dicts sorted by natural numeric order:
        [{"id": 1, "dir": Path, "image": Path, "instruction": str}, ...]
    """
    cases = []
    for entry in data_dir.iterdir():
        if not entry.is_dir():
            continue
        # Only consider directories whose name is a positive integer
        try:
            case_id = int(entry.name)
        except ValueError:
            continue

        image_path = find_original_image(entry)
        instruction = read_instruction(entry)

        if image_path is None:
            print(f"  [SKIP] Test case {case_id}: no original_image found in {entry}")
            continue
        if instruction is None:
            print(f"  [SKIP] Test case {case_id}: no valid instruction in {entry}")
            continue

        cases.append(
            {
                "id": case_id,
                "dir": entry,
                "image": image_path,
                "instruction": instruction,
            }
        )

    cases.sort(key=lambda c: c["id"])
    return cases


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------


def run_single_case(
    sample_fn,
    image_path: Path,
    instruction: str,
    output_path: Path,
    num_iterations: int,
):
    """Run SuSIE iterative subgoal generation for one test case."""
    import numpy as np
    from PIL import Image

    # Load and resize to 256x256 (model requirement)
    image = Image.open(image_path).convert("RGB")
    if image.size != (256, 256):
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
    current_image = np.array(image)

    for it in range(1, num_iterations + 1):
        current_image = sample_fn(current_image, instruction)

    # Save final result
    Image.fromarray(current_image).save(str(output_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Batch SuSIE inference on a structured test-data directory."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the test data directory containing numbered sub-folders.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of iterative subgoal generation steps (default: 5).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for inference (default: CPU).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="kvablack/susie",
        help="HuggingFace or local path to SuSIE UNet weights.",
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained Stable Diffusion (VAE + text encoder).",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=50,
        help="Number of DDIM sampling timesteps (default: 50).",
    )
    parser.add_argument(
        "--prompt-w",
        type=float,
        default=7.5,
        help="Text guidance weight (default: 7.5).",
    )
    parser.add_argument(
        "--context-w",
        type=float,
        default=2.5,
        help="Image guidance weight (default: 2.5).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM eta (0.0 = deterministic, default: 0.0).",
    )
    args = parser.parse_args()

    # ---- Configure backend ------------------------------------------------
    configure_jax_backend(args.use_gpu)

    # ---- Delayed imports (JAX must not be imported before backend config) --
    sys.path.insert(0, str(Path(__file__).parent / "susie"))
    from susie.model import create_sample_fn
    from susie.jax_utils import initialize_compilation_cache
    import jax

    # ---- Discover test cases ----------------------------------------------
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    print("=" * 60)
    print("SuSIE Batch Inference")
    print("=" * 60)
    print(f"Data directory : {data_dir.resolve()}")
    print(f"Iterations     : {args.num_iterations}")
    print(f"Backend        : {'GPU' if args.use_gpu else 'CPU'}")
    print(f"JAX devices    : {jax.devices()}")
    print()

    cases = discover_test_cases(data_dir)
    if not cases:
        print("No valid test cases found. Exiting.")
        sys.exit(1)

    print(f"\nFound {len(cases)} test case(s).\n")
    for c in cases:
        print(f"  [{c['id']:>3}] {c['image'].name}  |  \"{c['instruction']}\"")
    print()

    # ---- Load model once --------------------------------------------------
    print("Loading SuSIE model (one-time)...")
    initialize_compilation_cache()
    sample_fn = create_sample_fn(
        path=args.model_path,
        num_timesteps=args.num_timesteps,
        prompt_w=args.prompt_w,
        context_w=args.context_w,
        eta=args.eta,
        pretrained_path=args.pretrained_path,
    )
    print("Model loaded.\n")

    # ---- Run inference per test case --------------------------------------
    total = len(cases)
    succeeded = 0
    failed_ids = []

    for idx, case in enumerate(cases, 1):
        case_id = case["id"]
        output_path = case["dir"] / "edited_image.jpg"
        print(f"[{idx}/{total}] Test case {case_id}: \"{case['instruction']}\"")

        t0 = time.time()
        try:
            run_single_case(
                sample_fn=sample_fn,
                image_path=case["image"],
                instruction=case["instruction"],
                output_path=output_path,
                num_iterations=args.num_iterations,
            )
            elapsed = time.time() - t0
            print(f"         -> Saved {output_path}  ({elapsed:.1f}s)")
            succeeded += 1
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"         -> FAILED after {elapsed:.1f}s: {exc}")
            failed_ids.append(case_id)

    # ---- Summary ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total test cases : {total}")
    print(f"Succeeded        : {succeeded}")
    print(f"Failed           : {len(failed_ids)}")
    if failed_ids:
        print(f"Failed IDs       : {failed_ids}")
    print("Done.")


if __name__ == "__main__":
    main()
