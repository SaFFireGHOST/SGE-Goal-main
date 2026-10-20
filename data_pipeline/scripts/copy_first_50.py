#!/usr/bin/env python3
"""Copy first N files from given source directories into a destination folder,
preserving each source directory's relative folder structure.

Example usage:
  python scripts/copy_first_50.py \
    --sources \
      teleoperated_BridgeData_V2_Structured_Subset/EditInstructions \
      teleoperated_BridgeData_V2_Structured_Subset/JPEGImages \
    --dest Test_Data

Defaults: copies first 50 files (non-recursive), preserves folder path relative to the workspace root (cwd).
"""

import argparse
import logging
import os
import shutil
import sys
from typing import List, Dict


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def list_files(src: str, count: int, recursive: bool) -> List[str]:
    src = os.path.abspath(src)
    files = []
    if not os.path.exists(src):
        raise FileNotFoundError(src)

    if recursive:
        for root, _, filenames in os.walk(src):
            filenames = sorted(filenames)
            for fn in filenames:
                files.append(os.path.join(root, fn))
                if len(files) >= count:
                    return files
    else:
        entries = sorted(os.listdir(src))
        for fn in entries:
            full = os.path.join(src, fn)
            if os.path.isfile(full):
                files.append(full)
            if len(files) >= count:
                break

    return files


def compute_dest_subdir(src: str, dest_root: str, workspace_root: str) -> str:
    """Compute the destination subdirectory under dest_root.

    By default place the source's basename directly under dest_root (e.g.,
    Test_Data/JPEGImages and Test_Data/EditInstructions). This avoids an
    extra parent folder (like the long dataset folder) being included.
    """
    src = os.path.abspath(src)
    base = os.path.basename(src.rstrip(os.sep))
    return os.path.join(dest_root, base)


def copy_files(src: str, dest_root: str, count: int, workspace_root: str, recursive: bool, overwrite: bool, dry_run: bool) -> int:
    files = list_files(src, count, recursive)
    if not files:
        logging.warning("No files found in %s", src)
        return 0

    dest_subdir = compute_dest_subdir(src, dest_root, workspace_root)
    copied = 0
    for fpath in files:
        if recursive:
            rel_fp = os.path.relpath(fpath, start=src)
            target = os.path.join(dest_subdir, rel_fp)
        else:
            target = os.path.join(dest_subdir, os.path.basename(fpath))

        target_dir = os.path.dirname(target)
        if not dry_run:
            os.makedirs(target_dir, exist_ok=True)

        if os.path.exists(target) and not overwrite:
            logging.info("Skipping existing: %s", target)
            continue

        logging.info("Copying: %s -> %s", fpath, target)
        if not dry_run:
            shutil.copy2(fpath, target)
        copied += 1

    return copied


def gather_keys(src: str, recursive: bool) -> Dict[str, List[str]]:
    """Return a dict mapping a "key" -> list of absolute paths found under src.

    Key rules:
      - For directories, the key is the directory basename (name of the folder).
      - For files, the key is the filename WITHOUT extension (so 'foo.jpg' -> 'foo').

    This allows matching directory names in one source to files (without ext) in another.
    """
    src = os.path.abspath(src)
    keys_map: Dict[str, List[str]] = {}

    if not os.path.exists(src):
        return keys_map

    if recursive:
        for root, dirnames, filenames in os.walk(src):
            # directories (use directory basename as key)
            for d in sorted(dirnames):
                full = os.path.join(root, d)
                key = os.path.basename(full)
                keys_map.setdefault(key, []).append(full)
            # files (use filename without extension)
            for fn in sorted(filenames):
                full = os.path.join(root, fn)
                if os.path.isfile(full):
                    key = os.path.splitext(fn)[0]
                    keys_map.setdefault(key, []).append(full)
    else:
        for entry in sorted(os.listdir(src)):
            full = os.path.join(src, entry)
            if os.path.isdir(full):
                key = os.path.basename(full)
                keys_map.setdefault(key, []).append(full)
            elif os.path.isfile(full):
                key = os.path.splitext(entry)[0]
                keys_map.setdefault(key, []).append(full)

    return keys_map


def copy_paired_files(sources: List[str], dest_root: str, count: int, workspace_root: str, recursive: bool, overwrite: bool, dry_run: bool) -> int:
    """Copy items whose keys are present in ALL sources. Copies up to `count` keys.

    Keys may refer to directories (copied recursively) or files (copied as-is).
    """
    if len(sources) < 2:
        raise ValueError("--pair-sources requires at least two source directories")

    maps = [gather_keys(s, recursive) for s in sources]
    sets = [set(m.keys()) for m in maps]

    common = set.intersection(*sets) if sets else set()
    if not common:
        logging.warning("No common keys found across sources")
        return 0

    keys = sorted(common)[:count]
    copied = 0

    logging.info("Found %d common keys; copying up to %d", len(common), count)

    for key in keys:
        for idx, src in enumerate(sources):
            candidates = sorted(maps[idx][key])
            src_path = candidates[0]

            dest_subdir = compute_dest_subdir(src, dest_root, workspace_root)

            if recursive:
                rel_fp = os.path.relpath(src_path, start=src)
                target = os.path.join(dest_subdir, rel_fp)
            else:
                target = os.path.join(dest_subdir, os.path.basename(src_path))

            # Ensure parent directory exists for files, or parent exists for copytree target
            target_parent = os.path.dirname(target)
            if not dry_run:
                os.makedirs(target_parent, exist_ok=True)

            if os.path.isdir(src_path):
                # Copy directory tree
                if os.path.exists(target):
                    if not overwrite:
                        logging.info("Skipping existing directory: %s", target)
                        continue
                    if not dry_run:
                        shutil.rmtree(target)
                logging.info("Copying paired directory: %s -> %s", src_path, target)
                if not dry_run:
                    shutil.copytree(src_path, target)
                copied += 1
            else:
                # Copy a single file (keep original filename)
                if os.path.exists(target) and not overwrite:
                    logging.info("Skipping existing file: %s", target)
                    continue
                logging.info("Copying paired file: %s -> %s", src_path, target)
                if not dry_run:
                    shutil.copy2(src_path, target)
                copied += 1

    return copied


def main():
    p = argparse.ArgumentParser(description="Copy first N files from sources into dest preserving folder structure")
    p.add_argument("--sources", nargs="+", default=[
        "teleoperated_BridgeData_V2_Structured_Subset/EditInstructions",
        "teleoperated_BridgeData_V2_Structured_Subset/JPEGImages",
    ], help="Source directories to copy from (default: the two BridgeData dirs)")
    p.add_argument("--dest", required=False, default="Test_Data", help="Destination root (default: Test_Data)")
    p.add_argument("--count", type=int, default=50, help="Number of files/pairs to copy (default: 50)")
    p.add_argument("--recursive", action="store_true", help="If set, search files recursively inside each source and preserve subdirs")
    p.add_argument("--pair-sources", dest="pair_sources", action="store_true", default=True, help="Copy only files whose basenames exist in ALL sources (paired across sources) (default: True)")
    p.add_argument("--no-pair-sources", dest="pair_sources", action="store_false", help="Disable pairing mode")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files in destination")
    p.add_argument("--dry-run", action="store_true", help="Print actions without copying files")
    args = p.parse_args()

    workspace_root = os.getcwd()

    dest_root = os.path.abspath(args.dest)
    logging.info("Workspace root: %s", workspace_root)
    logging.info("Destination root: %s", dest_root)

    total = 0
    if args.pair_sources:
        try:
            copied = copy_paired_files(args.sources, dest_root, args.count, workspace_root, args.recursive, args.overwrite, args.dry_run)
            total += copied
        except Exception as e:
            logging.error("Failed to copy paired files: %s", e)
    else:
        for src in args.sources:
            if not os.path.exists(src):
                logging.warning("Source not found: %s", src)
                continue
            try:
                copied = copy_files(src, dest_root, args.count, workspace_root, args.recursive, args.overwrite, args.dry_run)
                total += copied
            except Exception as e:
                logging.error("Failed to copy from %s: %s", src, e)

    logging.info("Done — total files copied: %d", total)


if __name__ == "__main__":
    main()
