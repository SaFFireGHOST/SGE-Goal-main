#!/usr/bin/env python3
"""
I2I-Bench Runner Script

This script executes all 15 single-reference evaluation dimensions from I2I-Bench
on test cases in the data folder and saves results appropriately.

Each test case is evaluated in a separate subprocess to guarantee complete GPU
memory cleanup between test cases (avoids OOM from PyTorch CUDA memory fragmentation).

Usage:
    python runner.py [--data-dir DATA_DIR] [--dimensions DIM1 DIM2 ...] [--test-cases 1 2 3 ...]
    
Examples:
    python runner.py                                    # Run all dimensions on all test cases
    python runner.py --test-cases 1 2                   # Run on specific test cases
    python runner.py --dimensions blending_naturalness  # Run specific dimension
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from typing import Dict, List, Optional, Any
from pathlib import Path
from PIL import Image
import traceback

# Import the evaluator module
from i2i_bench_evaluator import (
    I2IBenchSingleReferenceEvaluator,
    EvaluatorConfig,
    find_image_file,
    clear_gpu_memory
)

import torch


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def discover_test_cases(data_dir: str) -> List[str]:
    """
    Discover all test case directories in the data folder.
    Test cases are expected to be numbered directories (1, 2, 3, ...).
    
    Returns:
        Sorted list of test case directory names
    """
    test_cases = []
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return test_cases
    
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # Check if it's a valid test case (contains required files)
            has_instruction = os.path.exists(os.path.join(item_path, "edit_instruction.txt"))
            has_original = find_image_file(item_path, "original_image") is not None
            has_edited = find_image_file(item_path, "edited_image") is not None
            
            if has_instruction and has_original and has_edited:
                test_cases.append(item)
            elif has_instruction or has_original or has_edited:
                print(f"Warning: Incomplete test case {item} - missing some required files")
    
    # Sort numerically if possible
    try:
        test_cases.sort(key=lambda x: int(x))
    except ValueError:
        test_cases.sort()
    
    return test_cases


def load_test_case(data_dir: str, test_case_id: str) -> Dict[str, Any]:
    """
    Load a test case's data (images and instruction).
    
    Returns:
        Dictionary containing 'original_image', 'edited_image', 'instruction', and paths
    """
    test_case_dir = os.path.join(data_dir, test_case_id)
    
    # Load instruction
    instruction_path = os.path.join(test_case_dir, "edit_instruction.txt")
    with open(instruction_path, 'r', encoding='utf-8') as f:
        instruction = f.read().strip()
    
    # Find and load images
    original_path = find_image_file(test_case_dir, "original_image")
    edited_path = find_image_file(test_case_dir, "edited_image")
    
    if original_path is None:
        raise FileNotFoundError(f"Original image not found in {test_case_dir}")
    if edited_path is None:
        raise FileNotFoundError(f"Edited image not found in {test_case_dir}")
    
    original_image = Image.open(original_path).convert("RGB")
    edited_image = Image.open(edited_path).convert("RGB")
    
    return {
        "original_image": original_image,
        "edited_image": edited_image,
        "instruction": instruction,
        "original_path": original_path,
        "edited_path": edited_path,
        "instruction_path": instruction_path,
        "test_case_dir": test_case_dir
    }





# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def run_evaluation(data_dir: str, 
                   test_cases: Optional[List[str]] = None,
                   dimensions: Optional[List[str]] = None,
                   verbose: bool = True):
    """
    Run evaluation on specified test cases and dimensions.
    
    Each test case is run in a separate subprocess to guarantee GPU memory
    is fully freed between test cases (prevents OOM from CUDA fragmentation).
    
    Args:
        data_dir: Path to data directory
        test_cases: List of test case IDs to evaluate (all if None)
        dimensions: List of dimensions to evaluate (all if None)
        verbose: Print progress information
    """
    # Initialize evaluator (in main process) just for validation
    print("Initializing I2I-Bench Single-Reference Evaluator...")
    config = EvaluatorConfig()
    evaluator = I2IBenchSingleReferenceEvaluator(config)
    
    # Discover test cases
    if test_cases is None:
        test_cases = discover_test_cases(data_dir)
    
    if not test_cases:
        print("No test cases found!")
        return
    
    # Use all dimensions if not specified
    if dimensions is None:
        dimensions = evaluator.list_dimensions()
    else:
        # Validate dimensions
        valid_dims = set(evaluator.list_dimensions())
        invalid = set(dimensions) - valid_dims
        if invalid:
            print(f"Warning: Invalid dimensions will be skipped: {invalid}")
            dimensions = [d for d in dimensions if d in valid_dims]
    
    # Free evaluator from main process - only subprocesses need GPU
    del evaluator
    
    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Test cases: {len(test_cases)}")
    print(f"  Dimensions: {len(dimensions)}")
    print(f"  Total evaluations: {len(test_cases) * len(dimensions)}")
    print(f"  Mode: Subprocess isolation per test case")
    print("\n" + "=" * 60)
    
    # Run each test case in a separate subprocess
    for tc_idx, tc_id in enumerate(test_cases, 1):
        print(f"\n[{tc_idx}/{len(test_cases)}] Processing Test Case: {tc_id} (in subprocess)")
        print("-" * 40)
        
        # The subprocess writes evaluation_results.json directly into the TC dir
        # Use a temp file for IPC to detect success/failure
        result_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', prefix=f'i2ibench_tc{tc_id}_', 
            delete=False, dir=data_dir
        )
        result_file.close()
        result_path = result_file.name
        
        try:
            # Build subprocess command
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--_internal_subprocess",
                "--data-dir", data_dir,
                "--test-cases", tc_id,
                "--_result_file", result_path
            ]
            if dimensions:
                cmd.extend(["--dimensions"] + dimensions)
            if not verbose:
                cmd.append("--quiet")
            
            # Run subprocess
            env = os.environ.copy()
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            proc = subprocess.run(
                cmd,
                env=env,
                timeout=7200,  # 2-hour timeout per test case
                capture_output=False  # Show output in real-time
            )
            
            if proc.returncode != 0:
                print(f"  WARNING: Subprocess for TC {tc_id} exited with code {proc.returncode}")
            
            # Read results from temp file for summary
            if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                with open(result_path, 'r') as f:
                    tc_results = json.load(f)
                
                if tc_results:
                    scores = [r.get("score", 0.0) for r in tc_results.values()]
                    avg_score = sum(scores) / len(scores) if scores else 0.0
                    successful = sum(1 for r in tc_results.values() if "error" not in r.get("details", {}))
                    print(f"  Result: {successful}/{len(tc_results)} dimensions successful, Average: {avg_score:.4f}")
            else:
                print(f"  ERROR: No results file produced for TC {tc_id}")
                
        except subprocess.TimeoutExpired:
            print(f"  ERROR: Subprocess for TC {tc_id} timed out after 2 hours")
        except Exception as e:
            print(f"  ERROR: Subprocess for TC {tc_id} failed: {e}")
            if verbose:
                traceback.print_exc()
        finally:
            # Clean up temp file
            try:
                if os.path.exists(result_path):
                    os.unlink(result_path)
            except:
                pass
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")


def run_single_tc_evaluation(data_dir: str, tc_id: str,
                              dimensions: List[str], result_file: str,
                              verbose: bool = True):
    """
    Run evaluation for a single test case (called in subprocess).
    This function runs all dimensions for one test case, saves results,
    then exits - allowing CUDA memory to be fully freed.
    """
    import gc
    
    print(f"  [Subprocess] Evaluating TC {tc_id} with {len(dimensions)} dimensions")
    
    # Initialize evaluator fresh in this subprocess
    config = EvaluatorConfig()
    evaluator = I2IBenchSingleReferenceEvaluator(config)
    
    try:
        # Load test case data
        test_data = load_test_case(data_dir, tc_id)
        
        if verbose:
            print(f"  Instruction: {test_data['instruction'][:80]}...")
        
        tc_results = {}
        
        # Evaluate each dimension
        for dim_idx, dim in enumerate(dimensions, 1):
            if verbose:
                print(f"  [{dim_idx}/{len(dimensions)}] Evaluating: {dim}...", end=" ", flush=True)
            
            try:
                result = evaluator.evaluate_dimension(
                    dim,
                    test_data["original_image"],
                    test_data["edited_image"],
                    test_data["instruction"]
                )
                
                # Convert score to plain float for JSON serialization
                score = result.get("score", 0.0)
                if hasattr(score, 'item'):
                    score = score.item()
                score = float(score)
                
                tc_results[dim] = {
                    "score": score,
                    "details": _make_json_serializable(result.get("details", {}))
                }
                
                if verbose:
                    print(f"Score: {score:.4f}")
                
            except Exception as e:
                print(f"Error: {e}")
                if verbose:
                    traceback.print_exc()
                tc_results[dim] = {"score": 0.0, "details": {"error": str(e)}}
            
            # Clear GPU memory between dimensions
            clear_gpu_memory()
        
        # Print test case summary
        if tc_results:
            scores = [r.get("score", 0.0) for r in tc_results.values()]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            print(f"  [Subprocess] TC {tc_id} Average Score: {avg_score:.4f}")
        
        # Save evaluation_results.json in the test case directory
        eval_results_path = os.path.join(test_data["test_case_dir"], "evaluation_results.json")
        with open(eval_results_path, 'w') as f:
            json.dump(tc_results, f, indent=2, default=str)
        print(f"  Saved: {eval_results_path}")
        
        # Also write to temp IPC file for parent process
        with open(result_file, 'w') as f:
            json.dump(tc_results, f, indent=2, default=str)
        
    except Exception as e:
        print(f"  [Subprocess] Error evaluating TC {tc_id}: {e}")
        traceback.print_exc()
        # Save error result
        with open(result_file, 'w') as f:
            json.dump({"_error": str(e)}, f)
    
    finally:
        # Cleanup
        del evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _make_json_serializable(obj):
    """Recursively convert non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif hasattr(obj, 'item'):  # numpy/torch scalar
        return obj.item()
    elif isinstance(obj, float):
        return obj
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        return str(obj)


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="I2I-Bench Single-Reference Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py                                    # Run all dimensions on all test cases
  python runner.py --test-cases 1 2 3                 # Run on specific test cases
  python runner.py --dimensions blending_naturalness  # Run specific dimension(s)
  python runner.py --data-dir /path/to/data           # Use custom data directory
  
Available Dimensions:
  - overall_image_quality        (Specialist: Q-Insight)
  - aesthetic_quality            (Specialist: ArtiMuse)
  - blending_naturalness         (LMM VQA-5Level)
  - editing_artifacts            (LMM VQA-5Level)
  - instruction_following_macro  (LMM VQA-5Level)
  - non_edited_region_fidelity   (Hybrid: Sa2VA + DINOv3)
  - physical_plausibility        (LMM Multi-Question VQA)
  - object_manipulation_accuracy (LMM VQA-2Level)
  - local_attribute_accuracy     (LMM VQA-2Level)
  - action_state_change_accuracy (LMM VQA-2Level)
  - spatial_accuracy             (LMM VQA-2Level)
  - text_content_style_accuracy  (Hybrid: GOT-OCR + LMM)
  - world_knowledge_reasoning    (LMM VQA-2Level)
  - subject_identity_fidelity    (Hybrid: Sa2VA + DINOv3)
  - composition_interaction      (LMM Multi-Question VQA)
        """
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Path to data directory (default: {DEFAULT_DATA_DIR})"
    )
    
    parser.add_argument(
        "--test-cases", "-t",
        nargs="+",
        type=str,
        help="Specific test case IDs to evaluate (e.g., 1 2 3)"
    )
    
    parser.add_argument(
        "--dimensions", "-dim",
        nargs="+",
        type=str,
        help="Specific dimensions to evaluate"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        "--list-dimensions",
        action="store_true",
        help="List all available dimensions and exit"
    )
    
    parser.add_argument(
        "--list-test-cases",
        action="store_true",
        help="List all discovered test cases and exit"
    )
    
    # Internal arguments for subprocess mode (not for user use)
    parser.add_argument(
        "--_internal_subprocess",
        action="store_true",
        help=argparse.SUPPRESS  # Hidden from help
    )
    parser.add_argument(
        "--_result_file",
        type=str,
        help=argparse.SUPPRESS  # Hidden from help
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle list options
    if args.list_dimensions:
        print("Available Evaluation Dimensions:")
        print("-" * 40)
        dims = I2IBenchSingleReferenceEvaluator.list_dimensions()
        for i, dim in enumerate(dims, 1):
            print(f"  {i:2d}. {dim}")
        return
    
    if args.list_test_cases:
        test_cases = discover_test_cases(args.data_dir)
        print(f"Discovered Test Cases in {args.data_dir}:")
        print("-" * 40)
        for tc in test_cases:
            print(f"  - {tc}")
        print(f"\nTotal: {len(test_cases)} test cases")
        return
    
    # Internal subprocess mode: evaluate a single test case and exit
    if args._internal_subprocess:
        if not args.test_cases or len(args.test_cases) != 1:
            print("Error: subprocess mode requires exactly one test case")
            sys.exit(1)
        if not args._result_file:
            print("Error: subprocess mode requires --_result_file")
            sys.exit(1)
        
        # Validate dimensions
        config = EvaluatorConfig()
        evaluator = I2IBenchSingleReferenceEvaluator(config)
        valid_dims = evaluator.list_dimensions()
        del evaluator
        
        dimensions = args.dimensions if args.dimensions else valid_dims
        dimensions = [d for d in dimensions if d in valid_dims]
        
        run_single_tc_evaluation(
            data_dir=args.data_dir,
            tc_id=args.test_cases[0],
            dimensions=dimensions,
            result_file=args._result_file,
            verbose=not args.quiet
        )
        return
    
    # Normal mode: orchestrate evaluation with subprocess per test case
    print("\n" + "=" * 60)
    print("I2I-Bench Single-Reference Evaluation")
    print("Based on: 'I2I-Bench: A Comprehensive Benchmark Suite")
    print("          for Image-to-Image Editing Models'")
    print("=" * 60)
    
    run_evaluation(
        data_dir=args.data_dir,
        test_cases=args.test_cases,
        dimensions=args.dimensions,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
