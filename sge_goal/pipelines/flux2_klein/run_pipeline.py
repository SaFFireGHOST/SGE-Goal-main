#!/usr/bin/env python3
"""
Pipeline script that sequentially runs:
1. IER mask prediction
2. FER mask prediction
3. Image editing inference

Each phase is idempotent — if every test-case sub-directory already contains
the expected output file for a phase, that phase is skipped entirely.  Inside
each phase the individual scripts also have per-case resume/skip logic and
periodic volume commits, so even a partial run is never lost.

If Modal kills a container mid-phase, the pipeline automatically retries (up
to ``MAX_RETRIES`` times per phase).  Thanks to the per-case skip logic and
periodic volume commits inside each phase, retries pick up where the last
successful commit left off — at most ~25 cases of work are ever repeated.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

MAX_RETRIES = 50  # per phase — generous to survive many Modal restarts
RETRY_DELAY_SECS = 30  # cool-down between retries


def run_command_in_dir(command: list[str], directory: str) -> None:
    """Run a command in the given directory, retrying on failure."""
    dir_name = os.path.basename(directory)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'='*60}")
        if attempt > 1:
            print(f"RETRY {attempt}/{MAX_RETRIES} in {dir_name}/")
        print(f"Running in {dir_name}/: {' '.join(command)}")
        print(f"{'='*60}\n")

        result = subprocess.run(command, cwd=directory)

        if result.returncode == 0:
            print(f"\nCompleted successfully in {dir_name}/")
            return

        print(f"\nWARNING: Command failed with exit code {result.returncode} "
              f"in {dir_name}/ (attempt {attempt}/{MAX_RETRIES})")
        if attempt < MAX_RETRIES:
            print(f"Waiting {RETRY_DELAY_SECS}s before retry …")
            time.sleep(RETRY_DELAY_SECS)

    print(f"\nERROR: All {MAX_RETRIES} attempts failed in {dir_name}/. Giving up.")
    sys.exit(1)

def _phase_complete(data_dir: str, output_filename: str) -> bool:
    """Return True if *every* test-case sub-dir already has *output_filename*."""
    if not os.path.isdir(data_dir):
        return False
    subdirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ]
    if not subdirs:
        return False
    for sd in subdirs:
        if not os.path.isfile(os.path.join(data_dir, sd, output_filename)):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the full inference pipeline.")
    parser.add_argument("--data-dir", required=True, help="Absolute path to the test data folder")
    parser.add_argument("--fer-checkpoint-dir", required=True, help="Absolute path to the FER mask prediction checkpoint folder")
    parser.add_argument("--ie-checkpoint-dir", required=True, help="Absolute path to the image editing checkpoint folder")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    ier_dir = os.path.join(base_dir, os.path.join("..", "..", "grounding", "flux2_klein"))
    fer_dir = os.path.join(base_dir, os.path.join("..", "..", "mask_predictor", "inference", "flux2_klein"))
    ie_dir = os.path.join(base_dir, os.path.join("..", "..", "synthesis", "inference", "flux2_klein"))

    # Step 1: IER mask prediction
    if _phase_complete(args.data_dir, "initial_edit_region_mask.jpg"):
        print(f"\n{'='*60}")
        print("Step 1 SKIP — all test cases already have initial_edit_region_mask.jpg")
        print(f"{'='*60}")
    else:
        run_command_in_dir(
            ["modal", "run", "modal_ier_mask_prediction.py",
             "--data-dir", args.data_dir],
            ier_dir,
        )

    # Step 2: FER mask prediction
    if _phase_complete(args.data_dir, "final_edit_region_mask.jpg"):
        print(f"\n{'='*60}")
        print("Step 2 SKIP — all test cases already have final_edit_region_mask.jpg")
        print(f"{'='*60}")
    else:
        run_command_in_dir(
            ["modal", "run", "modal_flux2_klein_erm_inference.py",
             "--data-dir", args.data_dir,
             "--checkpoint-dir", args.fer_checkpoint_dir],
            fer_dir,
        )

    # Step 3: Image editing inference
    if _phase_complete(args.data_dir, "edited_image.jpg"):
        print(f"\n{'='*60}")
        print("Step 3 SKIP — all test cases already have edited_image.jpg")
        print(f"{'='*60}")
    else:
        run_command_in_dir(
            ["modal", "run", "modal_flux2_klein_ie_inference.py",
             "--data-dir", args.data_dir,
             "--checkpoint-dir", args.ie_checkpoint_dir],
            ie_dir,
        )

    print(f"\n{'='*60}")
    print("All steps completed successfully.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
