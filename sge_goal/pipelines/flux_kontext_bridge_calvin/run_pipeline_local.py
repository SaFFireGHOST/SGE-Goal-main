#!/usr/bin/env python3
"""Local (no Modal) version of run_pipeline.py.

Runs IER mask prediction, FER mask prediction and image editing on the local GPU.
``run_pipeline.py`` is the Modal version and is left untouched.
"""

import argparse
import os
import subprocess
import sys


def run_command(command, cwd):
    """Run a command in the given directory, streaming output to stdout/stderr."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(command)}")
    print(f"In:      {cwd}")
    print(f"{'='*60}\n")
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        print(f"\nCommand failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run the full inference pipeline locally.")
    parser.add_argument("--data-dir", required=True, help="Absolute path to the test data folder")
    parser.add_argument("--fer-checkpoint-dir", required=True, help="Absolute path to the FER mask prediction checkpoint folder")
    parser.add_argument("--image-editing-checkpoint-dir", required=True, help="Absolute path to the image editing checkpoint folder")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip test cases whose output already exists.")
    parser.add_argument("--low-vram", action="store_true",
                        help="Steps 2-3: encode prompts once and keep only the transformer + VAE on the GPU "
                             "(for a shared GPU with less than ~36 GB free).")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    resume = ["--skip-existing"] if args.resume else []
    low_vram = ["--low-vram"] if args.low_vram else []

    # Step 1: IER mask prediction
    run_command(
        [sys.executable, "local_ier_mask_prediction.py",
         "--data-dir", args.data_dir, "--device", args.device] + resume,
        cwd=os.path.join(base_dir, "..", "..", "grounding", "flux_kontext_bridge_calvin"),
    )

    # Step 2: FER mask prediction
    run_command(
        [sys.executable, "local_fer_mask_inference.py",
         "--data-dir", args.data_dir,
         "--checkpoint-dir", args.fer_checkpoint_dir,
         "--device", args.device] + resume + low_vram,
        cwd=os.path.join(base_dir, "..", "..", "mask_predictor", "inference", "flux_kontext_bridge_calvin"),
    )

    # Step 3: Image editing
    run_command(
        [sys.executable, "local_flux_kontext_inference.py",
         "--data-dir", args.data_dir,
         "--checkpoint-dir", args.image_editing_checkpoint_dir,
         "--device", args.device] + resume + low_vram,
        cwd=os.path.join(base_dir, "..", "..", "synthesis", "inference", "flux_kontext_bridge_calvin"),
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
