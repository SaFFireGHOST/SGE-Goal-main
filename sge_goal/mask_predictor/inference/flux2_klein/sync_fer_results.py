#!/usr/bin/env python3
"""
Real-time sync: Continuously downloads newly-generated final_edit_region_mask.jpg
files from the Modal volume to the local test data directory.

Only downloads individual mask files (not the entire volume), so each poll
cycle is fast.

Usage:
    python fer-mask-prediction-code/sync_fer_results.py \
        --local-data-dir ./bridge_data_v2_tel_test_data \
        --poll-interval 30
"""

import argparse
import os
import shutil
import subprocess
import sys
import time


VOLUME_NAME = "klein-opt-fer-parallel-inference-vol"
REMOTE_ROOT = "test_data"
TARGET_FILE = "final_edit_region_mask.jpg"
# All 2000 test case directory names
ALL_CASES = [str(i) for i in range(1, 2001)]


def get_local_existing(local_data_dir: str) -> set[str]:
    """Return set of case directory names that already have the target file locally."""
    existing = set()
    for d in ALL_CASES:
        fp = os.path.join(local_data_dir, d, TARGET_FILE)
        if os.path.isfile(fp):
            existing.add(d)
    return existing


def sync_once(local_data_dir: str, modal_cmd: list[str], known_local: set[str]) -> set[str]:
    """Download entire volume to tmp, extract only new mask files."""
    tmp_dir = os.path.join(os.path.dirname(local_data_dir),
                           f".tmp_fer_sync_{os.getpid()}")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        dl = subprocess.run(
            modal_cmd + ["volume", "get", VOLUME_NAME, REMOTE_ROOT, tmp_dir, "--force"],
            capture_output=True, text=True, timeout=600,
        )
        if dl.returncode != 0:
            print(f"  [sync] volume get failed (exit {dl.returncode}): "
                  f"{(dl.stderr or '').strip()[:200]}", flush=True)
            return set()

        # modal volume get may nest as test_data/
        nested = os.path.join(tmp_dir, "test_data")
        actual = nested if os.path.isdir(nested) else tmp_dir

        newly_synced = set()
        for d in os.listdir(actual):
            if d in known_local:
                continue
            mask_src = os.path.join(actual, d, TARGET_FILE)
            if not os.path.isfile(mask_src):
                continue
            local_case = os.path.join(local_data_dir, d)
            if os.path.isdir(local_case):
                os.replace(mask_src, os.path.join(local_case, TARGET_FILE))
                newly_synced.add(d)

        return newly_synced
    except subprocess.TimeoutExpired:
        print("  [sync] volume get timed out", flush=True)
        return set()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Real-time FER mask sync from Modal volume")
    parser.add_argument("--local-data-dir", required=True, help="Local test data directory")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between sync polls")
    args = parser.parse_args()

    local_data_dir = os.path.abspath(args.local_data_dir)
    if not os.path.isdir(local_data_dir):
        raise FileNotFoundError(f"Local data dir not found: {local_data_dir}")

    total_cases = len(ALL_CASES)
    modal_cmd = [sys.executable, "-m", "modal"]

    print(f"[sync] Monitoring Modal volume '{VOLUME_NAME}' for new FER masks", flush=True)
    print(f"[sync] Local dir: {local_data_dir}  ({total_cases} total cases)", flush=True)
    print(f"[sync] Poll interval: {args.poll_interval}s", flush=True)
    print(flush=True)

    known_local = get_local_existing(local_data_dir)
    print(f"[sync] Already have {len(known_local)}/{total_cases} masks locally", flush=True)

    while len(known_local) < total_cases:
        time.sleep(args.poll_interval)

        # Only check cases we don't have locally yet
        missing = [c for c in ALL_CASES if c not in known_local]
        if not missing:
            break

        newly = sync_once(local_data_dir, modal_cmd, known_local)
        if newly:
            known_local.update(newly)
            sample = sorted(newly, key=lambda x: int(x))[:10]
            print(f"[sync] +{len(newly)} new → {len(known_local)}/{total_cases}  "
                  f"(e.g. {', '.join(sample)}{'…' if len(newly) > 10 else ''})", flush=True)
        else:
            print(f"[sync] No new masks. {len(known_local)}/{total_cases}. Waiting…", flush=True)

    print(f"\n[sync] All {total_cases} masks synced! Done.", flush=True)


if __name__ == "__main__":
    main()
