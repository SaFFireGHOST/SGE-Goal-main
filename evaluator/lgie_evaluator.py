#!/usr/bin/env python3
"""
lgie_evaluator.py — Unified LGIE Evaluation Driver (phased, local-only)

Evaluates test cases across all 15 evaluation dimensions (R1–R15).
Models are loaded ONCE per phase and reused across ALL test cases,
avoiding the massive overhead of per-test-case model loading.

Phase 1: VideoChat-Flash  →  R1, R4, R8-9, R10-12, R13, R14-15  (all cases)
Phase 2: R2 models (SAM + LoFTR + BLIP-2)  →  R2               (all cases)
Phase 3: R3 models (SAM + BLIP-2)          →  R3               (all cases)
Phase 4: R5-6-7 models (SAM + XMem + VCF)  →  R5-6-7           (all cases)

Results are saved incrementally after each test case so that a crashed /
interrupted run can be resumed with --skip-existing.

Usage:
    python lgie_evaluator.py --all --data_dir /path/to/data
    python lgie_evaluator.py --all --data_dir /path/to/data --skip-existing
    python lgie_evaluator.py --test_case 1 --data_dir /path/to/data
    python lgie_evaluator.py --list --data_dir /path/to/data
"""

import os
import sys
import json
import argparse
import time
import gc
import importlib

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# All 19 evaluation-result keys
EVAL_DIM_KEYS = [
    "R1_unmentioned_stationary_objects_count",
    "R1_unmentioned_stationary_objects_list",
    "R2_unmentioned_moved_within_scene_count",
    "R2_unmentioned_moved_within_scene_list",
    "R3_unmentioned_removed_from_scene_count",
    "R3_unmentioned_removed_from_scene_list",
    "R4_hallucinated_objects_count",
    "R4_hallucinated_objects_list",
    "R5_unrelated_color_change",
    "R6_unrelated_texture_change",
    "R7_unrelated_shape_change",
    "R8_scene_darker",
    "R9_scene_brighter",
    "R10_camera_zoom",
    "R11_camera_crop",
    "R12_camera_perspective",
    "R13_photorealism",
    "R14_source_adherence",
    "R15_destination_adherence",
]

# Per-phase key groups — used to check whether a phase is already done
PHASE1_KEYS = [
    "R1_unmentioned_stationary_objects_count",
    "R1_unmentioned_stationary_objects_list",
    "R4_hallucinated_objects_count",
    "R4_hallucinated_objects_list",
    "R8_scene_darker", "R9_scene_brighter",
    "R10_camera_zoom", "R11_camera_crop", "R12_camera_perspective",
    "R13_photorealism",
    "R14_source_adherence", "R15_destination_adherence",
]
PHASE2_KEYS = [
    "R2_unmentioned_moved_within_scene_count",
    "R2_unmentioned_moved_within_scene_list",
]
PHASE3_KEYS = [
    "R3_unmentioned_removed_from_scene_count",
    "R3_unmentioned_removed_from_scene_list",
]
PHASE4_KEYS = [
    "R5_unrelated_color_change",
    "R6_unrelated_texture_change",
    "R7_unrelated_shape_change",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def find_image(directory: str, prefix: str) -> str:
    """Find an image file with the given prefix in a directory."""
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = os.path.join(directory, prefix + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"No image found with prefix '{prefix}' in {directory}"
    )


def load_test_case(test_case_dir: str):
    """Load a test-case from its numbered subfolder."""
    original_img = find_image(test_case_dir, "original_image")
    edited_img = find_image(test_case_dir, "edited_image")
    instr_path = os.path.join(test_case_dir, "edit_instruction.txt")
    with open(instr_path, "r") as f:
        instruction = f.read().strip()
    return original_img, edited_img, instruction


def list_test_cases(data_dir: str):
    """List all numbered test-case subfolders in data_dir."""
    entries = []
    for name in os.listdir(data_dir):
        full = os.path.join(data_dir, name)
        if os.path.isdir(full) and name.isdigit():
            entries.append(int(name))
    return sorted(entries)


def _load_results(tc_dir: str, filename: str = "evaluation_results.json") -> dict:
    """Load existing partial/full results, or return empty dict."""
    path = os.path.join(tc_dir, filename)
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            print(f"[WARN] Corrupt JSON at {path} — deleting and starting fresh")
            os.remove(path)
    return {}


def _save_results(tc_dir: str, results: dict, filename: str = "evaluation_results.json"):
    """Save results to disk (overwrites)."""
    path = os.path.join(tc_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _phase_complete(results: dict, phase_keys: list) -> bool:
    """Check whether all keys for a phase are already present."""
    return all(k in results for k in phase_keys)


def _is_phase_done(tc_dir: str, phase_keys: list,
                   output_filename: str = "evaluation_results.json") -> bool:
    """Check if phase is done in the primary file or the specified output file."""
    primary = _load_results(tc_dir, "evaluation_results.json")
    if _phase_complete(primary, phase_keys):
        return True
    if output_filename != "evaluation_results.json":
        temp = _load_results(tc_dir, output_filename)
        if _phase_complete(temp, phase_keys):
            return True
    return False


def _ensure_paths():
    """Ensure XMem and BASE_DIR are on sys.path."""
    xmem_dir = os.path.join(BASE_DIR, "XMem")
    if xmem_dir not in sys.path:
        sys.path.insert(0, xmem_dir)
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)


def _free_vram():
    """Aggressively release GPU memory between phases."""
    import torch
    # Force garbage collection first
    gc.collect()
    gc.collect()  # second pass catches reference cycles
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        # Log VRAM state for debugging
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  [VRAM] allocated={alloc:.2f}GB, reserved={reserved:.2f}GB")
        sys.stdout.flush()


# ===================================================================
# Phased evaluation — models loaded ONCE, reused for all test cases
# ===================================================================

def evaluate_all_phased(data_dir: str, cases: list,
                        skip_existing: bool = False,
                        verbose: bool = False,
                        on_case_complete=None,
                        on_phase_complete=None,
                        phases=None,
                        output_filename="evaluation_results.json"):
    """Run selected phases across the given test cases.

    Args:
        on_case_complete: Optional callback(case_num, done, total) called
            after each test case's results are saved.  Useful for periodic
            volume commits in Modal.
        on_phase_complete: Optional callback(phase_num) called after each
            phase finishes.  Useful for volume commits between phases.
        phases: Optional list of phase numbers to run, e.g. [1,2,3] or [4].
                If None, runs all 4 phases (default behaviour).
        output_filename: Filename for saving results (default: evaluation_results.json).
                         Use a different name when running phases in parallel to avoid
                         race conditions on the same file.
    """
    _ensure_paths()

    if phases is None:
        phases = [1, 2, 3, 4]

    t0 = time.time()
    total = len(cases)
    print(f"\nPhased evaluation of {total} test cases")
    print(f"Data dir: {data_dir}")
    print(f"Skip existing: {skip_existing}")
    print(f"Phases to run: {phases}\n")
    sys.stdout.flush()

    if 1 in phases:
        _run_phase1(data_dir, cases, skip_existing, verbose, on_case_complete, output_filename)
        if on_phase_complete:
            on_phase_complete(1)
    if 2 in phases:
        _run_phase2(data_dir, cases, skip_existing, verbose, on_case_complete, output_filename)
        if on_phase_complete:
            on_phase_complete(2)
    if 3 in phases:
        _run_phase3(data_dir, cases, skip_existing, verbose, on_case_complete, output_filename)
        if on_phase_complete:
            on_phase_complete(3)
    if 4 in phases:
        _run_phase4(data_dir, cases, skip_existing, verbose, on_case_complete, output_filename)
        if on_phase_complete:
            on_phase_complete(4)

    elapsed = time.time() - t0
    phase_label = "ALL PHASES" if phases == [1, 2, 3, 4] else f"PHASES {phases}"
    print(f"\n{'='*70}")
    print(f"{phase_label} COMPLETE — {total} test cases in {elapsed/60:.1f} min")
    print(f"{'='*70}")


# -------------------------------------------------------------------
# Phase 1: VideoChat-Flash → R1, R4, R8-9, R10-12, R13, R14-15
# -------------------------------------------------------------------
def _run_phase1(data_dir, cases, skip_existing, verbose, on_case_complete=None, output_filename="evaluation_results.json"):
    import torch

    print(f"\n{'='*70}")
    print("PHASE 1/4: VideoChat-Flash → R1, R4, R8-9, R10-12, R13, R14-15")
    print(f"{'='*70}")

    # Determine which cases still need this phase
    if skip_existing:
        todo = []
        for c in cases:
            if not _is_phase_done(os.path.join(data_dir, str(c)), PHASE1_KEYS, output_filename):
                todo.append(c)
        skipped = len(cases) - len(todo)
        if skipped:
            print(f"  Skipping {skipped} cases already done for Phase 1")
    else:
        todo = list(cases)

    if not todo:
        print("  Phase 1: nothing to do.")
        return

    # Import R-script functions
    from R1 import (
        num_stationary_instruction_targets as r1_eval,
        load_videochat_model_and_tokenizer as load_vcf,
    )
    from R4 import num_unmentioned_hallucinations as r4_eval
    from R13 import check_photorealism as r13_eval
    r89_mod = importlib.import_module("R8-9")
    r101112_mod = importlib.import_module("R10-11-12")
    r1415_mod = importlib.import_module("R14-15")

    # Load VideoChat-Flash ONCE
    print("  Loading VideoChat-Flash …")
    tok, mdl = load_vcf()
    print(f"  VideoChat-Flash loaded. Processing {len(todo)} cases …")
    sys.stdout.flush()

    t0 = time.time()
    for i, c in enumerate(todo):
        tc_dir = os.path.join(data_dir, str(c))
        results = _load_results(tc_dir, output_filename)
        try:
            original_img, edited_img, instruction = load_test_case(tc_dir)

            # R1
            cnt1, objs1 = r1_eval(original_img, edited_img, instruction,
                                   mdl, tok, verbose=verbose)
            results["R1_unmentioned_stationary_objects_count"] = cnt1
            results["R1_unmentioned_stationary_objects_list"] = objs1

            # R4
            cnt4, objs4 = r4_eval(original_img, edited_img, instruction,
                                   mdl, tok, verbose=verbose)
            results["R4_hallucinated_objects_count"] = cnt4
            results["R4_hallucinated_objects_list"] = objs4

            # R8-9
            lighting = r89_mod.detect_lighting_changes(
                original_img, edited_img, instruction, mdl, tok,
                verbose=verbose)
            results["R8_scene_darker"] = lighting.get("Darker", "No")
            results["R9_scene_brighter"] = lighting.get("Brighter", "No")

            # R10-11-12
            cam = r101112_mod.detect_camera_transforms(
                original_img, edited_img, instruction, mdl, tok,
                verbose=verbose)
            results["R10_camera_zoom"] = cam.get("Zoom", "No")
            results["R11_camera_crop"] = cam.get("Crop", "No")
            results["R12_camera_perspective"] = cam.get("Perspective", "No")

            # R13
            results["R13_photorealism"] = r13_eval(
                original_img, edited_img, mdl, tok, verbose=verbose)

            # R14-15
            adh = r1415_mod.check_instruction_adherence(
                original_img, edited_img, instruction, mdl, tok,
                verbose=verbose)
            results["R14_source_adherence"] = adh.get("Source_Adherence", "No")
            results["R15_destination_adherence"] = adh.get(
                "Destination_Adherence", "No")

            _save_results(tc_dir, results, output_filename)
            if on_case_complete:
                on_case_complete(c, i + 1, len(todo))
            print(f"  [{i+1}/{len(todo)}] Case {c}: "
                  f"R1={cnt1} R4={cnt4} "
                  f"R13={results['R13_photorealism']} "
                  f"R14={results['R14_source_adherence']} "
                  f"R15={results['R15_destination_adherence']}")
        except Exception as e:
            print(f"  ERROR Phase 1, case {c}: {e}")
            import traceback
            traceback.print_exc()
        sys.stdout.flush()

    # Unload VideoChat-Flash
    del mdl, tok
    _free_vram()
    elapsed = time.time() - t0
    print(f"  Phase 1 complete ({elapsed/60:.1f} min) — "
          f"VideoChat-Flash unloaded.\n")


# -------------------------------------------------------------------
# Phase 2: R2 models (SAM + LoFTR + BLIP-2) → R2
# -------------------------------------------------------------------
def _run_phase2(data_dir, cases, skip_existing, verbose, on_case_complete=None, output_filename="evaluation_results.json"):
    import torch

    print(f"\n{'='*70}")
    print("PHASE 2/4: R2 models (SAM + LoFTR + BLIP-2) → R2")
    print(f"{'='*70}")

    if skip_existing:
        todo = []
        for c in cases:
            if not _is_phase_done(os.path.join(data_dir, str(c)), PHASE2_KEYS, output_filename):
                todo.append(c)
        skipped = len(cases) - len(todo)
        if skipped:
            print(f"  Skipping {skipped} cases already done for Phase 2")
    else:
        todo = list(cases)

    if not todo:
        print("  Phase 2: nothing to do.")
        return

    from R2 import (
        load_all_models as r2_load_all,
        num_unmentioned_moved_within_scene as r2_eval,
    )

    # Load R2 models ONCE
    print("  Loading R2 models (SAM + LoFTR + BLIP-2) …")
    r2_models = r2_load_all()
    print(f"  R2 models loaded. Processing {len(todo)} cases …")
    sys.stdout.flush()

    t0 = time.time()
    for i, c in enumerate(todo):
        tc_dir = os.path.join(data_dir, str(c))
        results = _load_results(tc_dir, output_filename)
        try:
            original_img, edited_img, instruction = load_test_case(tc_dir)

            cnt2, objs2 = r2_eval(
                original_img, edited_img, instruction,
                r2_models["video_chat_model"],
                r2_models["video_chat_tokenizer"],
                r2_models["sam_mask_generator"],
                r2_models["loftr_matcher"],
                r2_models["blip_processor"],
                r2_models["blip_model"],
                r2_models["blip_device"],
                verbose=verbose,
            )
            results["R2_unmentioned_moved_within_scene_count"] = cnt2
            results["R2_unmentioned_moved_within_scene_list"] = objs2

            _save_results(tc_dir, results, output_filename)
            if on_case_complete:
                on_case_complete(c, i + 1, len(todo))
            print(f"  [{i+1}/{len(todo)}] Case {c}: R2={cnt2}")
        except Exception as e:
            print(f"  ERROR Phase 2, case {c}: {e}")
            import traceback
            traceback.print_exc()
        sys.stdout.flush()

    del r2_models
    _free_vram()
    elapsed = time.time() - t0
    print(f"  Phase 2 complete ({elapsed/60:.1f} min) — "
          f"R2 models unloaded.\n")


# -------------------------------------------------------------------
# Phase 3: R3 models (SAM + BLIP-2) → R3
# -------------------------------------------------------------------
def _run_phase3(data_dir, cases, skip_existing, verbose, on_case_complete=None, output_filename="evaluation_results.json"):
    import torch

    print(f"\n{'='*70}")
    print("PHASE 3/4: R3 models (SAM + BLIP-2) → R3")
    print(f"{'='*70}")

    if skip_existing:
        todo = []
        for c in cases:
            if not _is_phase_done(os.path.join(data_dir, str(c)), PHASE3_KEYS, output_filename):
                todo.append(c)
        skipped = len(cases) - len(todo)
        if skipped:
            print(f"  Skipping {skipped} cases already done for Phase 3")
    else:
        todo = list(cases)

    if not todo:
        print("  Phase 3: nothing to do.")
        return

    from R3 import (
        load_all_models as r3_load_all,
        num_unmentioned_removed_from_scene as r3_eval,
    )

    # Load R3 models ONCE
    print("  Loading R3 models (SAM + BLIP-2) …")
    r3_models = r3_load_all()
    print(f"  R3 models loaded. Processing {len(todo)} cases …")
    sys.stdout.flush()

    t0 = time.time()
    for i, c in enumerate(todo):
        tc_dir = os.path.join(data_dir, str(c))
        results = _load_results(tc_dir, output_filename)
        try:
            original_img, edited_img, instruction = load_test_case(tc_dir)

            cnt3, objs3 = r3_eval(
                original_img, edited_img, instruction,
                r3_models["video_chat_model"],
                r3_models["video_chat_tokenizer"],
                r3_models["sam_mask_generator"],
                r3_models["blip_processor"],
                r3_models["blip_model"],
                r3_models["blip_device"],
                verbose=verbose,
            )
            results["R3_unmentioned_removed_from_scene_count"] = cnt3
            results["R3_unmentioned_removed_from_scene_list"] = objs3

            _save_results(tc_dir, results, output_filename)
            if on_case_complete:
                on_case_complete(c, i + 1, len(todo))
            print(f"  [{i+1}/{len(todo)}] Case {c}: R3={cnt3}")
        except Exception as e:
            print(f"  ERROR Phase 3, case {c}: {e}")
            import traceback
            traceback.print_exc()
        sys.stdout.flush()

    del r3_models
    _free_vram()
    elapsed = time.time() - t0
    print(f"  Phase 3 complete ({elapsed/60:.1f} min) — "
          f"R3 models unloaded.\n")


# -------------------------------------------------------------------
# Phase 4: R5-6-7 (SAM + XMem + VideoChat-Flash loaded together)
# -------------------------------------------------------------------
def _run_phase4(data_dir, cases, skip_existing, verbose, on_case_complete=None, output_filename="evaluation_results.json"):
    import torch
    import cv2
    import numpy as np

    print(f"\n{'='*70}")
    print("PHASE 4/4: R5-6-7 models (SAM + XMem + VideoChat-Flash)")
    print(f"{'='*70}")

    if skip_existing:
        todo = []
        for c in cases:
            if not _is_phase_done(os.path.join(data_dir, str(c)), PHASE4_KEYS, output_filename):
                todo.append(c)
        skipped = len(cases) - len(todo)
        if skipped:
            print(f"  Skipping {skipped} cases already done for Phase 4")
    else:
        todo = list(cases)

    if not todo:
        print("  Phase 4: nothing to do.")
        return

    r567_mod = importlib.import_module("R5-6-7")

    # --- Sub-phase 4a: SAM → generate masks for ALL test cases ---
    print("  [4a] Loading SAM …")
    sam_mask_gen, sam_model = r567_mod.init_sam_model()
    print(f"  [4a] SAM loaded. Generating masks for {len(todo)} cases …")
    sys.stdout.flush()

    all_masks_f1 = {}      # case_num → masks
    all_img_before = {}    # case_num → rgb array
    all_img_after = {}     # case_num → rgb array

    for i, c in enumerate(todo):
        tc_dir = os.path.join(data_dir, str(c))
        try:
            original_img, edited_img, _ = load_test_case(tc_dir)

            img_before = cv2.imread(original_img)
            img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
            img_after = cv2.imread(edited_img)
            img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)

            # Resize to common resolution (max 1024)
            h, w = img_before.shape[:2]
            h2, w2 = img_after.shape[:2]
            target_h = min(max(h, h2), 1024)
            target_w = min(max(w, w2), 1024)
            scale = min(target_h / h, target_w / w)
            new_h, new_w = int(h * scale), int(w * scale)
            if (new_h, new_w) != img_before.shape[:2]:
                img_before = cv2.resize(img_before, (new_w, new_h),
                                        interpolation=cv2.INTER_LANCZOS4)
            if img_after.shape[:2] != (new_h, new_w):
                img_after = cv2.resize(img_after, (new_w, new_h),
                                       interpolation=cv2.INTER_LANCZOS4)

            masks_f1 = r567_mod.run_sam_inference(img_before, sam_mask_gen)
            all_masks_f1[c] = masks_f1
            all_img_before[c] = img_before
            all_img_after[c] = img_after

            if (i + 1) % 50 == 0 or (i + 1) == len(todo):
                print(f"    SAM: {i+1}/{len(todo)} done")
        except Exception as e:
            print(f"    SAM ERROR case {c}: {e}")
            import traceback
            traceback.print_exc()
        sys.stdout.flush()

    # Unload SAM
    r567_mod.unload_sam_model(sam_model, sam_mask_gen)
    del sam_model, sam_mask_gen
    _free_vram()
    print("  [4a] SAM unloaded.")

    # --- Sub-phase 4b: XMem → track masks for ALL test cases ---
    print("  [4b] Loading XMem …")
    xmem_net, xmem_cfg = r567_mod.init_xmem_model()
    print(f"  [4b] XMem loaded. Tracking masks for {len(all_masks_f1)} cases …")
    sys.stdout.flush()

    all_masks_f2 = {}  # case_num → tracked masks
    processed = 0

    for c in todo:
        if c not in all_masks_f1:
            continue
        try:
            masks_f2 = r567_mod.track_masks_with_xmem(
                all_img_before[c], all_img_after[c],
                all_masks_f1[c], xmem_net, xmem_cfg)
            all_masks_f2[c] = masks_f2
            processed += 1
            if processed % 50 == 0 or processed == len(all_masks_f1):
                print(f"    XMem: {processed}/{len(all_masks_f1)} done")
        except Exception as e:
            print(f"    XMem ERROR case {c}: {e}")
            import traceback
            traceback.print_exc()
        sys.stdout.flush()

    # Unload XMem
    r567_mod.unload_xmem_model(xmem_net)
    del xmem_net
    _free_vram()
    print("  [4b] XMem unloaded.")

    # --- Sub-phase 4c: VideoChat-Flash → analyse changes for ALL cases ---
    print("  [4c] Loading VideoChat-Flash …")
    tok, mdl = r567_mod.load_videochat_model_and_tokenizer()
    print(f"  [4c] VideoChat-Flash loaded. Analysing {len(all_masks_f2)} cases …")
    sys.stdout.flush()

    t0 = time.time()
    analysed = 0

    for c in todo:
        if c not in all_masks_f2:
            continue
        tc_dir = os.path.join(data_dir, str(c))
        results = _load_results(tc_dir, output_filename)
        try:
            _, _, instruction = load_test_case(tc_dir)

            preds = r567_mod.analyze_object_changes_vlm(
                all_img_before[c], all_img_after[c],
                all_masks_f1[c], all_masks_f2[c],
                instruction, mdl, tok, verbose=verbose,
            )
            results["R5_unrelated_color_change"] = preds.get("Color", "No")
            results["R6_unrelated_texture_change"] = preds.get("Texture", "No")
            results["R7_unrelated_shape_change"] = preds.get("Shape", "No")

            _save_results(tc_dir, results, output_filename)
            analysed += 1
            if on_case_complete:
                on_case_complete(c, analysed, len(all_masks_f2))
            print(f"  [{analysed}/{len(all_masks_f2)}] Case {c}: "
                  f"R5={preds.get('Color','?')} "
                  f"R6={preds.get('Texture','?')} "
                  f"R7={preds.get('Shape','?')}")
        except Exception as e:
            print(f"  ERROR Phase 4c, case {c}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: save default "No" so the case isn't permanently skipped
            results["R5_unrelated_color_change"] = "No"
            results["R6_unrelated_texture_change"] = "No"
            results["R7_unrelated_shape_change"] = "No"
            _save_results(tc_dir, results, output_filename)
            analysed += 1
            if on_case_complete:
                on_case_complete(c, analysed, len(all_masks_f2))
            print(f"  [{analysed}/{len(all_masks_f2)}] Case {c}: "
                  f"R5=No R6=No R7=No (fallback — chat failed)")
        sys.stdout.flush()

    # Unload VCF + free all intermediaries
    del mdl, tok
    del all_masks_f1, all_masks_f2, all_img_before, all_img_after
    _free_vram()
    elapsed = time.time() - t0
    print(f"  Phase 4 complete ({elapsed/60:.1f} min) — "
          f"all R5-6-7 models unloaded.\n")


# ===================================================================
# Main CLI Entry
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="LGIE Evaluation Driver — phased, models loaded once"
    )
    parser.add_argument(
        "--test_case", type=int, default=None,
        help="Single test-case number to evaluate",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate ALL test cases",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to data folder with numbered test-case subfolders",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_cases",
        help="List available test cases and exit",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip phases already completed for a test case",
    )

    args = parser.parse_args()

    # --- List mode ---
    if args.list_cases:
        cases = list_test_cases(args.data_dir)
        print(f"Found {len(cases)} test cases in {args.data_dir}:")
        for c in cases:
            tc_dir = os.path.join(args.data_dir, str(c))
            desc_path = os.path.join(tc_dir, "test_description.txt")
            desc = ""
            if os.path.exists(desc_path):
                with open(desc_path) as f:
                    desc = f.read().strip()[:80]
            r = _load_results(tc_dir)
            phases = sum([
                _phase_complete(r, PHASE1_KEYS),
                _phase_complete(r, PHASE2_KEYS),
                _phase_complete(r, PHASE3_KEYS),
                _phase_complete(r, PHASE4_KEYS),
            ])
            status = f"{phases}/4" if phases < 4 else "done"
            print(f"  [{status:>4s}] {c:4d}: {desc}")
        return

    # --- Evaluate ALL ---
    if args.all:
        cases = list_test_cases(args.data_dir)
        if not cases:
            print("No test cases found.")
            return
        t0 = time.time()
        evaluate_all_phased(
            args.data_dir, cases,
            skip_existing=args.skip_existing,
            verbose=args.verbose,
        )
        elapsed = time.time() - t0
        print(f"\nTotal wall time: {elapsed/60:.1f} min")
        return

    # --- Evaluate SINGLE ---
    if args.test_case is not None:
        tc_dir = os.path.join(args.data_dir, str(args.test_case))
        if not os.path.isdir(tc_dir):
            print(f"Error: test case directory not found: {tc_dir}")
            sys.exit(1)

        print(f"Evaluating test case {args.test_case} (all 4 phases)")
        evaluate_all_phased(
            args.data_dir, [args.test_case],
            skip_existing=args.skip_existing,
            verbose=args.verbose,
        )
        r = _load_results(tc_dir)
        print("\nFinal Results:")
        print(json.dumps(r, indent=2))
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
