#!/usr/bin/env python3
"""
I2I-Bench Phase-Based Runner Script

Evaluates ALL test cases in phases organized by model, so each model loads
once, processes every test case, then unloads before the next model loads.

Phases:
  1. LMM (Qwen3-VL-8B) + GOT-OCR  →  10 LMM dims + text_content_style + seg command intermediates
  2. Sa2VA + DINOv3                 →  non_edited_region_fidelity, subject_identity_fidelity
  3. QInsight (OneAlign)            →  overall_image_quality
  4. ArtiMuse                       →  aesthetic_quality

Usage:
    python phase_runner.py --data-dir /path/to/data
    python phase_runner.py --data-dir /path/to/data --test-cases 1 2 3
    python phase_runner.py --data-dir /path/to/data --phases 3 4   # run only phases 3 & 4
"""

import os
import sys
import json
import gc
import re
import argparse
import subprocess
import traceback
import numpy as np
from typing import Dict, List, Optional, Any
from PIL import Image

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INTERMEDIATES_FILE = "_phase_intermediates.json"

PHASE_NAMES = {
    1: "LMM (Qwen3-VL-8B) + GOT-OCR",
    2: "Sa2VA + DINOv3",
    3: "QInsight (OneAlign)",
    4: "ArtiMuse",
}

PHASE_DIMS = {
    1: [
        "blending_naturalness", "editing_artifacts", "instruction_following_macro",
        "physical_plausibility", "object_manipulation_accuracy", "local_attribute_accuracy",
        "action_state_change_accuracy", "spatial_accuracy", "world_knowledge_reasoning",
        "composition_interaction", "text_content_style_accuracy",
    ],
    2: ["non_edited_region_fidelity", "subject_identity_fidelity"],
    3: ["overall_image_quality"],
    4: ["aesthetic_quality"],
}


# ==============================================================================
# HELPERS
# ==============================================================================

def discover_test_cases(data_dir: str) -> List[str]:
    """Discover all valid test case directories."""
    from i2i_bench_evaluator import find_image_file

    test_cases = []
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return test_cases

    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            has_instruction = os.path.exists(os.path.join(item_path, "edit_instruction.txt"))
            has_original = find_image_file(item_path, "original_image") is not None
            has_edited = find_image_file(item_path, "edited_image") is not None
            if has_instruction and has_original and has_edited:
                test_cases.append(item)

    try:
        test_cases.sort(key=lambda x: int(x))
    except ValueError:
        test_cases.sort()
    return test_cases


def load_test_case(data_dir: str, tc_id: str) -> Dict[str, Any]:
    """Load images and instruction for a test case."""
    from i2i_bench_evaluator import find_image_file

    tc_dir = os.path.join(data_dir, tc_id)
    with open(os.path.join(tc_dir, "edit_instruction.txt"), "r", encoding="utf-8") as f:
        instruction = f.read().strip()

    original_path = find_image_file(tc_dir, "original_image")
    edited_path = find_image_file(tc_dir, "edited_image")
    if original_path is None:
        raise FileNotFoundError(f"Original image not found in {tc_dir}")
    if edited_path is None:
        raise FileNotFoundError(f"Edited image not found in {tc_dir}")

    return {
        "original_image": Image.open(original_path).convert("RGB"),
        "edited_image": Image.open(edited_path).convert("RGB"),
        "instruction": instruction,
        "test_case_dir": tc_dir,
    }


def load_existing_results(tc_dir: str) -> dict:
    path = os.path.join(tc_dir, "evaluation_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_results(tc_dir: str, results: dict):
    path = os.path.join(tc_dir, "evaluation_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_intermediates(tc_dir: str) -> dict:
    path = os.path.join(tc_dir, INTERMEDIATES_FILE)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_intermediates(tc_dir: str, intermediates: dict):
    path = os.path.join(tc_dir, INTERMEDIATES_FILE)
    with open(path, "w") as f:
        json.dump(intermediates, f, indent=2, default=str)


def _dim_already_done(results: dict, dim: str) -> bool:
    """Check if a dimension was already successfully evaluated."""
    if dim not in results:
        return False
    details = results[dim].get("details", {})
    return "error" not in details


def _make_json_serializable(obj):
    """Recursively convert non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    elif isinstance(obj, float):
        return obj
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def _parse_seg_commands(response: str) -> List[str]:
    """Parse segmentation commands from LMM response."""
    try:
        json_match = re.search(r"\[([^\]]+)\]", response)
        if json_match:
            items = re.findall(r'"([^"]+)"', json_match.group())
            cmds = [item for item in items if item.lower().startswith("please segment")]
            if cmds:
                return cmds
    except Exception:
        pass
    commands = re.findall(r"Please segment [^\n\"]+", response)
    return commands if commands else []


# ==============================================================================
# PHASE 1: LMM (Qwen3-VL-8B) + GOT-OCR
# ==============================================================================

def run_phase_1(data_dir: str, test_cases: List[str], verbose: bool = True):
    """
    Load LMM + GOT-OCR once. For every test case:
      - Evaluate 10 pure LMM dimensions
      - Evaluate text_content_style_accuracy (LMM + GOT-OCR)
      - Generate & store seg commands for Phase 2 hybrid dims
    """
    import torch
    from i2i_bench_evaluator import (
        EvaluatorConfig, LMMHandler, Prompts, clear_gpu_memory,
        BlendingNaturalnessEvaluator, EditingArtifactsEvaluator,
        InstructionFollowingMacroEvaluator, PhysicalPlausibilityEvaluator,
        ObjectManipulationAccuracyEvaluator, LocalAttributeAccuracyEvaluator,
        ActionStateChangeAccuracyEvaluator, SpatialAccuracyEvaluator,
        WorldKnowledgeReasoningEvaluator, CompositionInteractionEvaluator,
        TextContentStyleAccuracyEvaluator,
    )

    config = EvaluatorConfig()

    # Build evaluator dict – singletons ensure LMM loads only once
    evaluators = {
        "blending_naturalness": BlendingNaturalnessEvaluator(config),
        "editing_artifacts": EditingArtifactsEvaluator(config),
        "instruction_following_macro": InstructionFollowingMacroEvaluator(config),
        "physical_plausibility": PhysicalPlausibilityEvaluator(config),
        "object_manipulation_accuracy": ObjectManipulationAccuracyEvaluator(config),
        "local_attribute_accuracy": LocalAttributeAccuracyEvaluator(config),
        "action_state_change_accuracy": ActionStateChangeAccuracyEvaluator(config),
        "spatial_accuracy": SpatialAccuracyEvaluator(config),
        "world_knowledge_reasoning": WorldKnowledgeReasoningEvaluator(config),
        "composition_interaction": CompositionInteractionEvaluator(config),
        "text_content_style_accuracy": TextContentStyleAccuracyEvaluator(config),
    }

    # Pre-initialize LMM so we can also use it for intermediate generation
    lmm = LMMHandler(config)
    lmm.initialize()

    for tc_idx, tc_id in enumerate(test_cases, 1):
        tc_dir = os.path.join(data_dir, tc_id)
        print(f"\n[Phase 1] [{tc_idx}/{len(test_cases)}] TC {tc_id}")

        try:
            test_data = load_test_case(data_dir, tc_id)
            results = load_existing_results(tc_dir)
            intermediates = load_intermediates(tc_dir)

            # ---- evaluate direct dimensions ----
            for dim_name, evaluator in evaluators.items():
                if _dim_already_done(results, dim_name):
                    if verbose:
                        print(f"  {dim_name}: skip (done)")
                    continue
                try:
                    if verbose:
                        print(f"  {dim_name}...", end=" ", flush=True)
                    result = evaluator.evaluate(
                        test_data["original_image"],
                        test_data["edited_image"],
                        test_data["instruction"],
                    )
                    score = float(result.get("score", 0.0))
                    results[dim_name] = {
                        "score": score,
                        "details": _make_json_serializable(result.get("details", {})),
                    }
                    if verbose:
                        print(f"{score:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    if verbose:
                        traceback.print_exc()
                    results[dim_name] = {"score": 0.0, "details": {"error": str(e)}}

            # ---- generate intermediates for Phase 2 ----
            if "non_edited_region_fidelity" not in intermediates:
                try:
                    seg_prompt = Prompts.NON_EDITED_REGION_SEGMENTATION.format(
                        edit_instruction=test_data["instruction"]
                    )
                    seg_command = lmm.generate(
                        images=[test_data["original_image"], test_data["edited_image"]],
                        system_prompt="You are a segmentation assistant.",
                        user_prompt=seg_prompt,
                        max_tokens=100,
                    )
                    intermediates["non_edited_region_fidelity"] = {"seg_command": seg_command}
                    if verbose:
                        print(f"  → seg command for NERF generated")
                except Exception as e:
                    print(f"  → NERF intermediate error: {e}")
                    intermediates["non_edited_region_fidelity"] = {"error": str(e)}

            if "subject_identity_fidelity" not in intermediates:
                try:
                    seg_prompt = (
                        Prompts.SUBJECT_IDENTITY_SEGMENTATION
                        + f"\n\nEditing Instruction: {test_data['instruction']}\n\nOutput only the JSON list."
                    )
                    seg_response = lmm.generate(
                        images=[test_data["original_image"], test_data["edited_image"]],
                        system_prompt="You are an expert in image editing analysis.",
                        user_prompt=seg_prompt,
                        max_tokens=256,
                    )
                    seg_commands = _parse_seg_commands(seg_response)
                    if not seg_commands:
                        seg_commands = ["Please segment the main subject"]
                    intermediates["subject_identity_fidelity"] = {
                        "seg_commands": seg_commands,
                        "raw_response": seg_response,
                    }
                    if verbose:
                        print(f"  → {len(seg_commands)} seg cmds for SIF generated")
                except Exception as e:
                    print(f"  → SIF intermediate error: {e}")
                    intermediates["subject_identity_fidelity"] = {"error": str(e)}

            # Persist
            save_results(tc_dir, results)
            save_intermediates(tc_dir, intermediates)

        except Exception as e:
            print(f"  TC {tc_id} FAILED: {e}")
            traceback.print_exc()

        # Free activations / KV cache between TCs
        torch.cuda.empty_cache()

    print(f"\nPhase 1 complete.")


# ==============================================================================
# PHASE 2: Sa2VA + DINOv3
# ==============================================================================

def run_phase_2(data_dir: str, test_cases: List[str], verbose: bool = True):
    """
    Load Sa2VA + DINOv3 once. For every test case:
      - non_edited_region_fidelity  (using stored seg_command from Phase 1)
      - subject_identity_fidelity   (using stored seg_commands from Phase 1)
    """
    import torch
    from i2i_bench_evaluator import EvaluatorConfig, Sa2VAHandler, DINOv3Handler

    config = EvaluatorConfig()
    sa2va = Sa2VAHandler(config)
    dino = DINOv3Handler(config)

    # Eagerly load both models - if either fails, both dims will fail for all TCs
    try:
        sa2va.initialize()
    except Exception as e:
        print(f"FATAL: Sa2VA initialization failed: {e}")
        traceback.print_exc()
        # Mark all TCs as failed
        for tc_id in test_cases:
            tc_dir = os.path.join(data_dir, tc_id)
            results = load_existing_results(tc_dir)
            for dim in ["non_edited_region_fidelity", "subject_identity_fidelity"]:
                results[dim] = {"score": 0.0, "details": {"error": f"Sa2VA init failed: {e}"}}
            save_results(tc_dir, results)
        return

    try:
        dino.initialize()
    except Exception as e:
        print(f"FATAL: DINOv3 initialization failed: {e}")
        traceback.print_exc()
        for tc_id in test_cases:
            tc_dir = os.path.join(data_dir, tc_id)
            results = load_existing_results(tc_dir)
            for dim in ["non_edited_region_fidelity", "subject_identity_fidelity"]:
                results[dim] = {"score": 0.0, "details": {"error": f"DINOv3 init failed: {e}"}}
            save_results(tc_dir, results)
        return

    for tc_idx, tc_id in enumerate(test_cases, 1):
        tc_dir = os.path.join(data_dir, tc_id)
        print(f"\n[Phase 2] [{tc_idx}/{len(test_cases)}] TC {tc_id}")

        try:
            test_data = load_test_case(data_dir, tc_id)
            results = load_existing_results(tc_dir)
            intermediates = load_intermediates(tc_dir)
            orig_img = test_data["original_image"]
            edit_img = test_data["edited_image"]
            orig_size = (orig_img.width, orig_img.height)
            edit_size = (edit_img.width, edit_img.height)

            # ---- non_edited_region_fidelity ----
            if not _dim_already_done(results, "non_edited_region_fidelity"):
                try:
                    nerf_inter = intermediates.get("non_edited_region_fidelity", {})
                    if "error" in nerf_inter:
                        raise RuntimeError(f"Phase 1 error: {nerf_inter['error']}")
                    seg_command = nerf_inter.get("seg_command", "Please segment the edited region")

                    mask_edit = sa2va.segment(edit_img, seg_command)
                    mask_non_edit = 1.0 - mask_edit

                    # Resize masks for each image
                    if mask_non_edit.shape[:2] != (orig_img.height, orig_img.width):
                        mask_for_orig = np.array(
                            Image.fromarray((mask_non_edit * 255).astype(np.uint8)).resize(
                                orig_size, Image.Resampling.NEAREST
                            )
                        ) / 255.0
                    else:
                        mask_for_orig = mask_non_edit

                    if mask_non_edit.shape[:2] != (edit_img.height, edit_img.width):
                        mask_for_edit = np.array(
                            Image.fromarray((mask_non_edit * 255).astype(np.uint8)).resize(
                                edit_size, Image.Resampling.NEAREST
                            )
                        ) / 255.0
                    else:
                        mask_for_edit = mask_non_edit

                    features_orig = dino.extract_features(orig_img, mask_for_orig)
                    features_edit = dino.extract_features(edit_img, mask_for_edit)
                    score = float(dino.compute_similarity(features_orig, features_edit))

                    results["non_edited_region_fidelity"] = {
                        "score": score,
                        "details": {
                            "method": "Hybrid LMM-Specialist (Sa2VA + DINOv3)",
                            "segmentation_command": seg_command,
                            "description": "Cosine similarity of non-edited regions",
                        },
                    }
                    if verbose:
                        print(f"  non_edited_region_fidelity: {score:.4f}")
                except Exception as e:
                    print(f"  non_edited_region_fidelity ERROR: {e}")
                    if verbose:
                        traceback.print_exc()
                    results["non_edited_region_fidelity"] = {
                        "score": 0.0,
                        "details": {"error": str(e)},
                    }
            else:
                if verbose:
                    print(f"  non_edited_region_fidelity: skip (done)")

            # ---- subject_identity_fidelity ----
            if not _dim_already_done(results, "subject_identity_fidelity"):
                try:
                    sif_inter = intermediates.get("subject_identity_fidelity", {})
                    if "error" in sif_inter:
                        raise RuntimeError(f"Phase 1 error: {sif_inter['error']}")
                    seg_commands = sif_inter.get("seg_commands", ["Please segment the main subject"])

                    similarities = []
                    seg_details = []
                    for cmd in seg_commands:
                        mask_orig = sa2va.segment(orig_img, cmd)
                        mask_edit = sa2va.segment(edit_img, cmd)

                        if mask_orig.shape[:2] != (orig_img.height, orig_img.width):
                            mask_orig = np.array(
                                Image.fromarray((mask_orig * 255).astype(np.uint8)).resize(
                                    orig_size, Image.Resampling.NEAREST
                                )
                            ) / 255.0
                        if mask_edit.shape[:2] != (edit_img.height, edit_img.width):
                            mask_edit = np.array(
                                Image.fromarray((mask_edit * 255).astype(np.uint8)).resize(
                                    edit_size, Image.Resampling.NEAREST
                                )
                            ) / 255.0

                        feat_orig = dino.extract_features(orig_img, mask_orig)
                        feat_edit = dino.extract_features(edit_img, mask_edit)
                        sim = float(dino.compute_similarity(feat_orig, feat_edit))
                        similarities.append(sim)
                        seg_details.append({"command": cmd, "similarity": sim})

                    score = float(np.mean(similarities)) if similarities else 0.0
                    results["subject_identity_fidelity"] = {
                        "score": score,
                        "details": {
                            "method": "Hybrid LMM-Specialist (Sa2VA + DINOv3)",
                            "segmentation_details": seg_details,
                            "num_subjects": len(seg_commands),
                        },
                    }
                    if verbose:
                        print(f"  subject_identity_fidelity: {score:.4f}")
                except Exception as e:
                    print(f"  subject_identity_fidelity ERROR: {e}")
                    if verbose:
                        traceback.print_exc()
                    results["subject_identity_fidelity"] = {
                        "score": 0.0,
                        "details": {"error": str(e)},
                    }
            else:
                if verbose:
                    print(f"  subject_identity_fidelity: skip (done)")

            save_results(tc_dir, results)

        except Exception as e:
            print(f"  TC {tc_id} FAILED: {e}")
            traceback.print_exc()

        torch.cuda.empty_cache()

    print(f"\nPhase 2 complete.")


# ==============================================================================
# PHASE 3: QInsight (OneAlign)
# ==============================================================================

def run_phase_3(data_dir: str, test_cases: List[str], verbose: bool = True):
    """Load QInsight once. Evaluate overall_image_quality for every test case."""
    import torch
    from i2i_bench_evaluator import EvaluatorConfig, OverallImageQualityEvaluator

    config = EvaluatorConfig()
    evaluator = OverallImageQualityEvaluator(config)

    for tc_idx, tc_id in enumerate(test_cases, 1):
        tc_dir = os.path.join(data_dir, tc_id)
        print(f"\n[Phase 3] [{tc_idx}/{len(test_cases)}] TC {tc_id}")

        try:
            results = load_existing_results(tc_dir)
            if _dim_already_done(results, "overall_image_quality"):
                if verbose:
                    print(f"  overall_image_quality: skip (done)")
                continue

            test_data = load_test_case(data_dir, tc_id)
            result = evaluator.evaluate(
                test_data["original_image"],
                test_data["edited_image"],
                test_data["instruction"],
            )
            score = float(result.get("score", 0.0))
            results["overall_image_quality"] = {
                "score": score,
                "details": _make_json_serializable(result.get("details", {})),
            }
            if verbose:
                print(f"  overall_image_quality: {score:.4f}")
            save_results(tc_dir, results)

        except Exception as e:
            print(f"  overall_image_quality ERROR: {e}")
            if verbose:
                traceback.print_exc()
            results = load_existing_results(tc_dir)
            results["overall_image_quality"] = {"score": 0.0, "details": {"error": str(e)}}
            save_results(tc_dir, results)

        torch.cuda.empty_cache()

    print(f"\nPhase 3 complete.")


# ==============================================================================
# PHASE 4: ArtiMuse
# ==============================================================================

def run_phase_4(data_dir: str, test_cases: List[str], verbose: bool = True):
    """Load ArtiMuse once. Evaluate aesthetic_quality for every test case."""
    import torch
    from i2i_bench_evaluator import EvaluatorConfig, AestheticQualityEvaluator

    config = EvaluatorConfig()
    evaluator = AestheticQualityEvaluator(config)

    for tc_idx, tc_id in enumerate(test_cases, 1):
        tc_dir = os.path.join(data_dir, tc_id)
        print(f"\n[Phase 4] [{tc_idx}/{len(test_cases)}] TC {tc_id}")

        try:
            results = load_existing_results(tc_dir)
            if _dim_already_done(results, "aesthetic_quality"):
                if verbose:
                    print(f"  aesthetic_quality: skip (done)")
                continue

            test_data = load_test_case(data_dir, tc_id)
            result = evaluator.evaluate(
                test_data["original_image"],
                test_data["edited_image"],
                test_data["instruction"],
            )
            score = float(result.get("score", 0.0))
            results["aesthetic_quality"] = {
                "score": score,
                "details": _make_json_serializable(result.get("details", {})),
            }
            if verbose:
                print(f"  aesthetic_quality: {score:.4f}")
            save_results(tc_dir, results)

        except Exception as e:
            print(f"  aesthetic_quality ERROR: {e}")
            if verbose:
                traceback.print_exc()
            results = load_existing_results(tc_dir)
            results["aesthetic_quality"] = {"score": 0.0, "details": {"error": str(e)}}
            save_results(tc_dir, results)

        torch.cuda.empty_cache()

    print(f"\nPhase 4 complete.")


# ==============================================================================
# PHASE DISPATCH
# ==============================================================================

PHASE_FUNCTIONS = {
    1: run_phase_1,
    2: run_phase_2,
    3: run_phase_3,
    4: run_phase_4,
}


# ==============================================================================
# ORCHESTRATOR  (main process – spawns one subprocess per phase)
# ==============================================================================

def orchestrate(data_dir: str, test_cases: Optional[List[str]] = None,
                phases: Optional[List[int]] = None, verbose: bool = True):
    """Run phases sequentially via subprocesses for clean GPU isolation."""
    if test_cases is None:
        test_cases = discover_test_cases(data_dir)
    if not test_cases:
        print("No test cases found!")
        return
    if phases is None:
        phases = [1, 2, 3, 4]

    print(f"Test cases : {len(test_cases)}")
    print(f"Phases     : {phases}")
    for p in phases:
        print(f"  Phase {p} ({PHASE_NAMES[p]}): {', '.join(PHASE_DIMS[p])}")

    for phase_id in phases:
        print(f"\n{'=' * 60}")
        print(f"PHASE {phase_id}: {PHASE_NAMES[phase_id]}")
        print(f"{'=' * 60}")

        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--_internal_phase", str(phase_id),
            "--data-dir", data_dir,
            "--test-cases", *test_cases,
        ]
        if not verbose:
            cmd.append("--quiet")

        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        proc = subprocess.run(cmd, env=env, timeout=86400)
        if proc.returncode != 0:
            print(f"WARNING: Phase {phase_id} exited with code {proc.returncode}")

    # Cleanup intermediates
    for tc_id in test_cases:
        inter_path = os.path.join(data_dir, tc_id, INTERMEDIATES_FILE)
        if os.path.exists(inter_path):
            os.remove(inter_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    for tc_id in test_cases:
        tc_dir = os.path.join(data_dir, tc_id)
        results = load_existing_results(tc_dir)
        n_success = sum(1 for r in results.values() if "error" not in r.get("details", {}))
        scores = [r.get("score", 0.0) for r in results.values() if "error" not in r.get("details", {})]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"  TC {tc_id}: {n_success}/{len(results)} dims ok, avg {avg:.4f}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Phase-based I2I-Bench Runner")
    parser.add_argument("--data-dir", "-d", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--test-cases", "-t", nargs="+", type=str)
    parser.add_argument("--phases", nargs="+", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--quiet", "-q", action="store_true")
    # Hidden: subprocess-internal
    parser.add_argument("--_internal_phase", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()

    if args._internal_phase:
        # Subprocess mode – run one phase directly
        tcs = args.test_cases or discover_test_cases(args.data_dir)
        PHASE_FUNCTIONS[args._internal_phase](
            args.data_dir, tcs, verbose=not args.quiet
        )
        return

    # Orchestrator mode
    print("\n" + "=" * 60)
    print("I2I-Bench Phase-Based Evaluation")
    print("=" * 60)

    orchestrate(
        data_dir=args.data_dir,
        test_cases=args.test_cases,
        phases=args.phases,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
