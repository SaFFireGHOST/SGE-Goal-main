#!/usr/bin/env python3
"""
Continuously sync Modal volume 'models/' directory to local machine.

Runs as a background daemon alongside the training process. Downloads new
checkpoints and always re-syncs non-checkpoint content (validation/test
visualizations) since they update every epoch.

Usage:
    python sync_modal_volume.py [--interval SECONDS]
"""

import argparse
import io
import os
import re
import signal
import sys
import time
import traceback
from datetime import datetime

import modal
from modal.volume import FileEntryType

# Must match modal_flux_kontext_train.py settings
VOLUME_NAME = "flux-kontext-dev-bridge-image-editing-volume"
REMOTE_MODELS_DIR = "models"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

DEFAULT_SYNC_INTERVAL = 120  # seconds

# Graceful shutdown
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    log("Received shutdown signal. Will exit after current sync pass.")


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGHUP, signal.SIG_IGN)  # Ignore terminal hangup


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[SYNC {ts}] {msg}", flush=True)


def get_local_checkpoints(local_models_path):
    """Return set of checkpoint directory names already present locally."""
    ckpts = set()
    if not os.path.isdir(local_models_path):
        return ckpts
    for root, dirs, _ in os.walk(local_models_path):
        for d in dirs:
            if re.match(r"checkpoint-epoch-\d+$", d) or re.match(r"checkpoint-\d+$", d):
                ckpts.add(d)
    return ckpts


def sync_once(downloaded_checkpoints, downloaded_files):
    """Perform one sync pass. Returns number of new files downloaded.
    
    Strategy:
    - Checkpoint dirs already downloaded locally: skip entirely
    - Non-checkpoint files (validation visualizations etc): only re-download
      when a new checkpoint is detected (they only change after an epoch)
    - New checkpoint dirs: download fully
    """
    checkpoint_re = re.compile(r"(?:checkpoint-epoch-\d+|checkpoint-\d+)")

    # Fresh handle each pass — volume.reload() only works inside Modal containers,
    # so we get a new Volume reference to see the latest files.
    volume = modal.Volume.from_name(VOLUME_NAME)

    try:
        entries = list(volume.listdir(REMOTE_MODELS_DIR, recursive=True))
    except Exception as e:
        log(f"Could not list volume directory '{REMOTE_MODELS_DIR}': {e}")
        return 0

    file_entries = [e for e in entries if e.type == FileEntryType.FILE]
    if not file_entries:
        return 0

    # First pass: detect new checkpoints
    new_checkpoint_dirs = set()
    for entry in file_entries:
        for p in entry.path.split("/"):
            if checkpoint_re.fullmatch(p) and p not in downloaded_checkpoints:
                new_checkpoint_dirs.add(p)

    has_new_checkpoints = len(new_checkpoint_dirs) > 0

    files_to_download = []
    skipped = 0

    for entry in file_entries:
        parts = entry.path.split("/")
        ckpt_dir = None
        for p in parts:
            if checkpoint_re.fullmatch(p):
                ckpt_dir = p
                break

        # Skip files inside already-downloaded checkpoint dirs
        if ckpt_dir and ckpt_dir in downloaded_checkpoints:
            skipped += 1
            continue

        # For non-checkpoint files: only re-download when new checkpoints appear
        if ckpt_dir is None and entry.path in downloaded_files and not has_new_checkpoints:
            skipped += 1
            continue

        files_to_download.append(entry)

    if not files_to_download:
        return 0

    reason = "new checkpoint detected" if has_new_checkpoints else "initial sync"
    log(
        f"Downloading {len(files_to_download)} files "
        f"({len(new_checkpoint_dirs)} new checkpoints, "
        f"skipped {skipped} already-synced, reason: {reason})..."
    )

    file_count = 0
    total_bytes = 0
    for entry in files_to_download:
        local_file_path = os.path.join(SCRIPT_DIR, entry.path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        try:
            with open(local_file_path, "wb") as f:
                bytes_written = volume.read_file_into_fileobj(entry.path, f)
                total_bytes += bytes_written
            file_count += 1
        except Exception as e:
            log(f"  Failed to download {entry.path}: {e}")

    # Mark new checkpoint dirs as downloaded so we won't re-download them
    downloaded_checkpoints.update(new_checkpoint_dirs)
    # Track all successfully downloaded file paths
    for entry in files_to_download:
        if os.path.exists(os.path.join(SCRIPT_DIR, entry.path)):
            downloaded_files.add(entry.path)

    total_mb = total_bytes / (1024 * 1024)
    log(
        f"Downloaded {file_count} files ({total_mb:.1f} MB). "
        f"Total known checkpoints: {len(downloaded_checkpoints)}"
    )
    return file_count


def main():
    parser = argparse.ArgumentParser(description="Sync Modal volume to local machine")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_SYNC_INTERVAL,
        help="Sync interval in seconds (default: 120)",
    )
    args = parser.parse_args()

    log(f"Starting volume sync daemon (interval={args.interval}s)")
    log(f"Volume: {VOLUME_NAME}")
    log(f"Remote dir: {REMOTE_MODELS_DIR}")
    log(f"Local dir: {LOCAL_MODELS_DIR}")

    # Seed with existing local checkpoints
    downloaded_checkpoints = get_local_checkpoints(LOCAL_MODELS_DIR)
    if downloaded_checkpoints:
        log(f"Seeded with {len(downloaded_checkpoints)} existing local checkpoints: {sorted(downloaded_checkpoints)}")

    # Track all downloaded non-checkpoint files to avoid re-downloading
    downloaded_files = set()

    consecutive_errors = 0
    max_consecutive_errors = 30

    while not _shutdown:
        try:
            new_files = sync_once(downloaded_checkpoints, downloaded_files)
            if new_files > 0:
                log(f"Sync pass complete: {new_files} new files")
            else:
                log("Sync pass: nothing new.")
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            log(f"Sync error ({consecutive_errors}/{max_consecutive_errors}): {type(e).__name__}: {e}")
            if consecutive_errors >= max_consecutive_errors:
                log("Too many consecutive errors. Exiting.")
                sys.exit(1)

        # Sleep in small increments to allow graceful shutdown
        for _ in range(args.interval):
            if _shutdown:
                break
            time.sleep(1)

    log("Sync daemon shutting down.")


if __name__ == "__main__":
    main()
