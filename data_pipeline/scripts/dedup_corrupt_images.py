#!/usr/bin/env python3
"""Deduplicate a newline-separated list of file paths while preserving order.

Usage:
  python dedup_corrupt_images.py [INPUT] [-o OUTPUT] [--inplace]

Defaults:
  INPUT: corrupt_images.txt
  OUTPUT: corrupt_images.dedup.txt

By default the original file is left unchanged. Use --inplace to replace
INPUT with the deduplicated contents (a backup INPUT.bak will be created).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Tuple


def dedup_file(input_path: str, output_path: str) -> Tuple[int, int]:
    """Deduplicate lines from input_path and write to output_path.

    Returns: (total_lines, unique_lines)
    """
    seen = set()
    total = 0
    unique = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for raw in fin:
            total += 1
            line = raw.rstrip("\n")
            if line in seen:
                continue
            seen.add(line)
            fout.write(line + "\n")
            unique += 1

    return total, unique


def safe_replace(original: str, temp: str) -> None:
    backup = original + ".bak"
    shutil.copy2(original, backup)
    os.replace(temp, original)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remove duplicate lines from a file")
    p.add_argument(
        "input",
        nargs="?",
        default="corrupt_images.txt",
        help="Input file (default: corrupt_images.txt)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file (default: input + '.dedup.txt')",
    )
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input file (creates input.bak first)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2

    output_path = args.output or (os.path.splitext(input_path)[0] + ".dedup.txt")

    # If user asked for inplace we'll write to a temp file then replace.
    if args.inplace:
        temp_path = input_path + ".tmp.dedup"
        total, unique = dedup_file(input_path, temp_path)
        # create backup and replace
        safe_replace(input_path, temp_path)
        print(
            f"Wrote {unique} unique lines (from {total}) into {input_path} (backup: {input_path}.bak)"
        )
    else:
        total, unique = dedup_file(input_path, output_path)
        print(f"Wrote {unique} unique lines (from {total}) into {output_path}")

    if total - unique > 0:
        print(f"Removed {total - unique} duplicate lines.")
    else:
        print("No duplicates found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
