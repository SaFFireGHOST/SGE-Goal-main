#!/usr/bin/env python3
"""
verify_integrity.py — Thorough integrity check of all 24 test data folders.

For each dataset:
1. Verify all test case folders exist with required files (original_image, edited_image, edit_instruction.txt)
2. Count evaluation_results.json files
3. Parse corresponding log to determine which phases completed and how many cases were processed
4. Verify each evaluation_results.json has the expected keys for the phases that were completed
5. Report any mismatches or corruption
"""

import os
import json
import re
import sys
from pathlib import Path

# Evaluation workspace containing FINAL-EVAL-RUNS/ and eval-logs/.
BASE = os.environ.get("SGE_EVAL_ROOT", os.path.dirname(os.path.abspath(__file__)))
FINAL_DIR = os.path.join(BASE, "FINAL-EVAL-RUNS")
LOG_DIR = os.path.join(BASE, "eval-logs")

PHASE1_KEYS = [
    "R1_unmentioned_stationary_objects_count",
    "R1_unmentioned_stationary_objects_list",
    "R4_hallucinated_objects_count",
    "R4_hallucinated_objects_list",
    "R8_scene_darker",
    "R9_scene_brighter",
    "R10_camera_zoom",
    "R11_camera_crop",
    "R12_camera_perspective",
    "R13_photorealism",
    "R14_source_adherence",
    "R15_destination_adherence",
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

DATASETS = [
    ("clover-calvin", "clover/calvin_abcd_test_data", 1087),
    ("fkd-calvin", "flux-kontext-dev/calvin_abcd_test_data", 1087),
    ("fkd-bridge", "flux-kontext-dev/bridge_data_v2_tel_test_data", 2000),
    ("fkd-libero", "flux-kontext-dev/libero_10_90_combi_test_data", 376),
    ("fkdft-bridge", "flux-kontext-dev-ft/bridge_data_v2_tel_ier_fer_gt_test_data", 2000),
    ("fkdft-calvin", "flux-kontext-dev-ft/calvin_abcd_ier_fer_gt_test_data", 1087),
    ("fkdft-libero", "flux-kontext-dev-ft/libero_10_90_combi_ier_fer_gt_test_data", 376),
    ("f2k-bridge", "flux2-klein/bridge_data_v2_tel_test_data", 2000),
    ("f2k-calvin", "flux2-klein/calvin_abcd_test_data", 1087),
    ("f2k-libero", "flux2-klein/libero_10_90_combi_test_data", 376),
    ("f2kft-bridge", "flux2-klein-ft/bridge_data_v2_tel_ier_fer_gt_test_data", 2000),
    ("f2kft-calvin", "flux2-klein-ft/calvin_abcd_ier_fer_gt_test_data", 1087),
    ("f2kft-libero", "flux2-klein-ft/libero_10_90_combi_ier_fer_gt_test_data", 376),
    ("gpt-bridge", "gpt-image-1.5/bridge_data_v2_tel_test_data", 2000),
    ("gpt-calvin", "gpt-image-1.5/calvin_abcd_test_data", 1087),
    ("gpt-libero", "gpt-image-1.5/libero_10_90_combi_test_data", 376),
    ("qwen-bridge", "qwen-image-edit/bridge_data_v2_tel_test_data", 2000),
    ("qwen-calvin", "qwen-image-edit/calvin_abcd_test_data", 1087),
    ("qwen-libero", "qwen-image-edit/libero_10_90_combi_test_data", 376),
    ("susie-bridge", "susie/bridge_data_v2_tel_test_data", 2000),
    ("susie-calvin", "susie/calvin_abcd_test_data", 1087),
    ("susie-libero", "susie/libero_10_90_combi_test_data", 376),
]


def parse_log(short_name):
    """Parse the eval log file to determine what was processed."""
    log_path = os.path.join(LOG_DIR, f"{short_name}.log")
    if not os.path.exists(log_path):
        return {"exists": False}

    with open(log_path, "r", errors="replace") as f:
        content = f.read()

    info = {"exists": True, "phases_completed": [], "phase_progress": {}}

    # Check which phases completed
    for phase_num in [1, 2, 3, 4]:
        # Look for "Phase X complete" or "Volume committed after Phase X"
        if re.search(rf"Phase {phase_num}.*complete", content) or \
           re.search(rf"Volume committed after Phase {phase_num}", content):
            info["phases_completed"].append(phase_num)

    # Parse per-phase progress: [X/Y] Case Z
    # Phase 1 log pattern: PHASE 1/4 ... then [X/Y] Case Z: R1=... R4=... R13=...
    # Phase 2 log pattern: PHASE 2/4 ... then [X/Y] Case Z: R2=...
    # Phase 3 log pattern: PHASE 3/4 ... then [X/Y] Case Z: R3=...
    # Phase 4 log pattern: PHASE 4/4 ... then [X/Y] Case Z: R5=...

    # Find last progress line for each phase
    # Split by phase headers
    phase_sections = {}
    phase_starts = list(re.finditer(r"PHASE (\d)/4:", content))
    for i, m in enumerate(phase_starts):
        phase_num = int(m.group(1))
        start = m.start()
        end = phase_starts[i + 1].start() if i + 1 < len(phase_starts) else len(content)
        phase_sections[phase_num] = content[start:end]

    for phase_num, section in phase_sections.items():
        # Find all [X/Y] progress lines
        progress = re.findall(r"\[(\d+)/(\d+)\] Case (\d+)", section)
        if progress:
            last = progress[-1]
            info["phase_progress"][phase_num] = {
                "done": int(last[0]),
                "total": int(last[1]),
                "last_case": int(last[2]),
            }

    # Check for 429 errors
    info["had_429"] = "429" in content or "RATE-LIMIT" in content

    # Check for "ALL PHASES COMPLETE" or "PHASES [1, 2, 3, 4] COMPLETE"
    info["all_done"] = "ALL PHASES COMPLETE" in content or "PHASES [1, 2, 3, 4] COMPLETE" in content

    # Check for "Evaluation done"
    info["eval_done_logged"] = "Evaluation done" in content or "exit_code=0" in content

    return info


def verify_dataset(short_name, rel_path, expected_total):
    """Verify a single dataset's integrity."""
    data_dir = os.path.join(FINAL_DIR, rel_path)
    issues = []

    # 1. Check directory exists
    if not os.path.isdir(data_dir):
        return [f"CRITICAL: data dir not found: {data_dir}"]

    # 2. Check all test case folders exist with required files
    missing_dirs = []
    missing_orig = []
    missing_edit = []
    missing_inst = []
    for i in range(1, expected_total + 1):
        tc_dir = os.path.join(data_dir, str(i))
        if not os.path.isdir(tc_dir):
            missing_dirs.append(i)
            continue
        files = os.listdir(tc_dir)
        has_orig = any(f.startswith("original_image") for f in files)
        has_edit = any(f.startswith("edited_image") for f in files)
        has_inst = "edit_instruction.txt" in files
        if not has_orig:
            missing_orig.append(i)
        if not has_edit:
            missing_edit.append(i)
        if not has_inst:
            missing_inst.append(i)

    if missing_dirs:
        issues.append(f"CRITICAL: {len(missing_dirs)} missing test case dirs: {missing_dirs[:5]}...")
    if missing_orig:
        issues.append(f"CRITICAL: {len(missing_orig)} missing original images: {missing_orig[:5]}...")
    if missing_edit:
        issues.append(f"CRITICAL: {len(missing_edit)} missing edited images: {missing_edit[:5]}...")
    if missing_inst:
        issues.append(f"CRITICAL: {len(missing_inst)} missing edit_instruction.txt: {missing_inst[:5]}...")

    # 3. Count and validate evaluation_results.json
    results_count = 0
    corrupt_results = []
    wrong_keys = []
    cases_with_results = []

    # Parse log to know what keys to expect
    log_info = parse_log(short_name)

    for i in range(1, expected_total + 1):
        results_file = os.path.join(data_dir, str(i), "evaluation_results.json")
        if os.path.exists(results_file):
            results_count += 1
            cases_with_results.append(i)
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    corrupt_results.append(i)
                    continue

                # Check keys based on what phases completed
                keys = set(data.keys())

                # Phase 1 keys should always be present if the file exists
                missing_p1 = set(PHASE1_KEYS) - keys
                if missing_p1:
                    wrong_keys.append((i, f"missing Phase1 keys: {missing_p1}"))

                # Phase 2 keys: present if Phase 2 completed for this case
                if 2 in log_info.get("phases_completed", []):
                    missing_p2 = set(PHASE2_KEYS) - keys
                    # Not all cases may have Phase 2 if it was interrupted
                    # Only flag if case number <= phase2 progress
                    p2_prog = log_info.get("phase_progress", {}).get(2, {})
                    if p2_prog and i <= p2_prog.get("done", 0):
                        if missing_p2:
                            wrong_keys.append((i, f"missing Phase2 keys: {missing_p2}"))

                # Phase 3 keys
                if 3 in log_info.get("phases_completed", []):
                    missing_p3 = set(PHASE3_KEYS) - keys
                    p3_prog = log_info.get("phase_progress", {}).get(3, {})
                    if p3_prog and i <= p3_prog.get("done", 0):
                        if missing_p3:
                            wrong_keys.append((i, f"missing Phase3 keys: {missing_p3}"))

                # Phase 4 keys
                if 4 in log_info.get("phases_completed", []):
                    missing_p4 = set(PHASE4_KEYS) - keys
                    p4_prog = log_info.get("phase_progress", {}).get(4, {})
                    if p4_prog and i <= p4_prog.get("done", 0):
                        if missing_p4:
                            wrong_keys.append((i, f"missing Phase4 keys: {missing_p4}"))

            except json.JSONDecodeError:
                corrupt_results.append(i)
            except Exception as e:
                corrupt_results.append(i)

    if corrupt_results:
        issues.append(f"CRITICAL: {len(corrupt_results)} corrupt evaluation_results.json: {corrupt_results[:5]}...")
    if wrong_keys:
        issues.append(f"WARNING: {len(wrong_keys)} cases with unexpected keys: {wrong_keys[:3]}...")

    # 4. Cross-reference with log
    # For Phase 1: if Phase 1 completed, all cases should have results
    if 1 in log_info.get("phases_completed", []):
        if results_count < expected_total:
            # Phase 1 completed but not all cases have results — might be download issue
            issues.append(f"WARNING: Phase 1 completed per log but only {results_count}/{expected_total} have results")
    elif 1 in log_info.get("phase_progress", {}):
        p1 = log_info["phase_progress"][1]
        expected_p1 = p1["done"]
        if results_count < expected_p1:
            issues.append(f"WARNING: Phase 1 processed {expected_p1} cases per log but only {results_count} have results locally")

    # Determine what's needed
    phases_done = log_info.get("phases_completed", [])
    all_done = log_info.get("all_done", False)

    return {
        "issues": issues,
        "total_cases": expected_total,
        "results_count": results_count,
        "phases_completed": phases_done,
        "phase_progress": log_info.get("phase_progress", {}),
        "all_done": all_done,
        "had_429": log_info.get("had_429", False),
        "log_exists": log_info.get("exists", False),
        "corrupt_count": len(corrupt_results),
        "wrong_keys_count": len(wrong_keys),
    }


def main():
    print("=" * 80)
    print("  THOROUGH INTEGRITY CHECK — 24 LGIE Test Data Folders")
    print("=" * 80)
    print()

    all_ok = True
    needs_resume = []

    for short, rel, total in DATASETS:
        result = verify_dataset(short, rel, total)
        status_char = "✓" if not result["issues"] else "✗"
        done_char = "DONE" if result["all_done"] else "PARTIAL"

        phase_str = ",".join(str(p) for p in result["phases_completed"]) or "none"
        progress_parts = []
        for pn in sorted(result["phase_progress"].keys()):
            pp = result["phase_progress"][pn]
            progress_parts.append(f"P{pn}:{pp['done']}/{pp['total']}")
        progress_str = " ".join(progress_parts) if progress_parts else "no progress data"

        print(f"{status_char} {short:20s}  results={result['results_count']:>5}/{result['total_cases']:<5}  "
              f"phases_done=[{phase_str}]  {done_char}  429={result['had_429']}")
        print(f"    progress: {progress_str}")

        if result["issues"]:
            all_ok = False
            for issue in result["issues"]:
                print(f"    ⚠ {issue}")

        if not result["all_done"]:
            needs_resume.append((short, rel, total, result))

        print()

    print("=" * 80)
    if all_ok:
        print("  ALL CHECKS PASSED — no critical issues found")
    else:
        print("  ISSUES FOUND — see above")
    print(f"  Datasets needing resume: {len(needs_resume)}")
    for short, rel, total, r in needs_resume:
        print(f"    - {short}: {r['results_count']}/{total} ({','.join(str(p) for p in r['phases_completed'])} done)")
    print("=" * 80)


if __name__ == "__main__":
    main()
