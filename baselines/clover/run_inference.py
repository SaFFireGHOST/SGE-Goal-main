#!/usr/bin/env python
"""
Batch Inference Script for CLOVER Visual Planner

Runs the CLOVER visual planner on structured test data directories.
Each test case subdirectory should contain:
  - original_image.{png,jpg,jpeg}  : the input scene before manipulation
  - edit_instruction.txt           : the textual manipulation instruction (first line used)

The script produces:
  - edited_image.jpg               : the predicted final scene after manipulation

Usage:
    python run_inference.py --data-dir dummy_test_data
    python run_inference.py --data-dir /path/to/test_data --num-chains 5
"""

import os
import sys
import glob
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional

# Import the CLOVER Visual Planner inference class
from clover_visual_planner import CLOVERVisualPlannerInference


def find_original_image(test_case_dir: str) -> Optional[str]:
    """Find the original image file in a test case directory."""
    for ext in ["png", "jpg", "jpeg"]:
        candidates = glob.glob(os.path.join(test_case_dir, f"original_image.{ext}"))
        if candidates:
            return candidates[0]
    return None


def read_instruction(test_case_dir: str) -> Optional[str]:
    """Read the manipulation instruction from edit_instruction.txt (first line only)."""
    instruction_path = os.path.join(test_case_dir, "edit_instruction.txt")
    if not os.path.exists(instruction_path):
        return None
    with open(instruction_path, "r") as f:
        # Only use the first line as the instruction (skip metadata like 'confidence:')
        instruction = f.readline().strip()
    return instruction if instruction else None


def discover_test_cases(data_dir: str) -> List[Tuple[int, str]]:
    """
    Discover all natural-numbered test case subdirectories.
    
    Returns a sorted list of (number, path) tuples.
    """
    test_cases = []
    for entry in os.listdir(data_dir):
        entry_path = os.path.join(data_dir, entry)
        if os.path.isdir(entry_path):
            try:
                num = int(entry)
                test_cases.append((num, entry_path))
            except ValueError:
                continue  # Skip non-numeric directories
    test_cases.sort(key=lambda x: x[0])
    return test_cases


def main():
    parser = argparse.ArgumentParser(
        description="Run CLOVER Visual Planner inference on structured test data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the test data directory containing numbered subdirectories"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/visual_planner.pt",
        help="Path to visual planner checkpoint (default: checkpoints/visual_planner.pt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=10,
        help="Max number of planning chains (default: 10)"
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.001,
        help="Early stopping convergence threshold (default: 0.001)"
    )
    args = parser.parse_args()

    # Validate data directory
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Discover test cases
    test_cases = discover_test_cases(args.data_dir)
    if not test_cases:
        print(f"Error: No numbered test case subdirectories found in {args.data_dir}")
        sys.exit(1)

    print("=" * 60)
    print("CLOVER Visual Planner — Batch Inference")
    print("=" * 60)
    print(f"Data directory : {args.data_dir}")
    print(f"Test cases     : {len(test_cases)}")
    print(f"Checkpoint     : {args.checkpoint}")
    print(f"Device         : {args.device}")
    print(f"Max chains     : {args.num_chains}")
    print(f"Conv. threshold: {args.convergence_threshold}")
    print("=" * 60)

    # Initialize the planner once (expensive: loads MiDaS, CLIP, diffusion model)
    print("\nInitializing CLOVER Visual Planner (one-time setup)...\n")
    planner = CLOVERVisualPlannerInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        with_depth=True,
    )

    # Process each test case
    results = {"success": [], "skipped": [], "failed": []}

    for idx, (case_num, case_dir) in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test case {case_num} ({idx+1}/{len(test_cases)}): {case_dir}")
        print(f"{'='*60}")

        # Find original image
        image_path = find_original_image(case_dir)
        if image_path is None:
            print(f"  SKIP: No original_image.{{png,jpg,jpeg}} found")
            results["skipped"].append(case_num)
            continue

        # Read instruction
        instruction = read_instruction(case_dir)
        if instruction is None:
            print(f"  SKIP: No edit_instruction.txt found or file is empty")
            results["skipped"].append(case_num)
            continue

        print(f"  Image      : {os.path.basename(image_path)}")
        print(f"  Instruction: {instruction}")

        try:
            # Preprocess the input image
            condition_image = planner.preprocess_image(image_path)

            # Generate visual plan with chaining
            if args.num_chains > 1:
                generated_frames, success, num_attempts, chain_labels = \
                    planner.generate_chained_visual_plan(
                        condition_image=condition_image,
                        text_instruction=instruction,
                        num_chains=args.num_chains,
                        convergence_threshold=args.convergence_threshold,
                        auto_stop=True,
                    )
            else:
                generated_frames, success, num_attempts = planner.generate_visual_plan(
                    condition_image=condition_image,
                    text_instruction=instruction,
                )

            # Extract the FINAL frame — the "after manipulation" scene
            # generated_frames shape: (1, K, 3, H, W) in [0, 1] range
            final_frame = generated_frames[0, -1]  # (3, H, W)
            final_frame_np = (final_frame.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # Save as edited_image.jpg
            output_path = os.path.join(case_dir, "edited_image.jpg")
            Image.fromarray(final_frame_np).save(output_path, quality=95)

            total_frames = generated_frames.shape[1]
            print(f"  Result     : {'SUCCESS' if success else 'FALLBACK'} "
                  f"({total_frames} frames generated, {num_attempts} attempt(s))")
            print(f"  Saved      : {output_path}")
            results["success"].append(case_num)

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results["failed"].append(case_num)

    # Summary
    print(f"\n{'='*60}")
    print("BATCH INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total test cases : {len(test_cases)}")
    print(f"Successful       : {len(results['success'])}  {results['success']}")
    print(f"Skipped          : {len(results['skipped'])}  {results['skipped']}")
    print(f"Failed           : {len(results['failed'])}  {results['failed']}")
    print(f"{'='*60}")

    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
