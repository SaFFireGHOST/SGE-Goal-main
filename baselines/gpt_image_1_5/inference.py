#!/usr/bin/env python3
"""
GPT-Image-1.5 Instruction-Guided Image Editing Inference Script.

For each test case (numbered subdirectory), reads an original image and an
edit instruction, calls the OpenAI gpt-image-1.5 model to produce an edited
image, resizes the output to match the original dimensions, and saves it as
edited_image.jpg inside the same subdirectory.
"""

from __future__ import annotations
import argparse
import base64
import io
import os
import re
import sys
import time
import glob
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from PIL import Image

# ── API Key (hardcoded) ──────────────────────────────────────────────────────
API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Constants ────────────────────────────────────────────────────────────────
MODEL = "gpt-image-1.5"
QUALITY = "medium"
SIZE = "auto"            # let the model pick the best output size
RETRY_BACKOFF = 2        # exponential backoff base (seconds)
MAX_RETRY_WAIT = 120     # cap per-retry wait at 120s
DEFAULT_WORKERS = 5      # concurrent API requests

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def build_prompt(edit_instruction: str) -> str:
    """
    Build the enhanced prompt for image editing.

    The raw edit instruction is placed at the very start (first characters)
    to minimise any risk of truncation — it is the most important part.
    The auxiliary context that follows tells the model *how* to apply it.
    """
    prompt = (
        f"{edit_instruction}\n\n"
        "You are given the original image of a scene that needs to be edited. "
        "The edit instruction above describes the specific change to apply. "
        "Perform the described edit faithfully and precisely on the original image. "
        "Preserve every other aspect of the original image that is not explicitly "
        "mentioned in the edit instruction — including background, lighting, "
        "camera angle, colors, object positions, textures, and overall scene "
        "composition. The output should look like the same scene after the "
        "described edit has been carried out, and nothing else should change."
    )
    return prompt


def find_original_image(case_dir: str) -> str | None:
    """Find original_image with any supported extension."""
    for ext in ("jpg", "jpeg", "png"):
        pattern = os.path.join(case_dir, f"original_image.{ext}")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def read_edit_instruction(case_dir: str) -> str:
    """Read edit_instruction.txt and strip any 'confidence: ...' line."""
    path = os.path.join(case_dir, "edit_instruction.txt")
    with open(path, "r") as f:
        lines = f.read().strip().splitlines()
    # Filter out confidence lines
    instruction_lines = [
        line for line in lines if not re.match(r"^\s*confidence\s*:", line, re.IGNORECASE)
    ]
    return "\n".join(instruction_lines).strip()


def process_test_case(client: OpenAI, case_dir: str, case_id: str) -> bool:
    """
    Process a single test case: read image + instruction, call API,
    resize output to original dimensions, save edited_image.jpg.

    Returns True on success, False on failure.
    """
    # ── Find input image ─────────────────────────────────────────────────
    image_path = find_original_image(case_dir)
    if image_path is None:
        log.warning(f"[Case {case_id}] No original_image found — skipping.")
        return False

    # ── Read instruction ─────────────────────────────────────────────────
    try:
        edit_instruction = read_edit_instruction(case_dir)
    except FileNotFoundError:
        log.warning(f"[Case {case_id}] edit_instruction.txt missing — skipping.")
        return False

    if not edit_instruction:
        log.warning(f"[Case {case_id}] Edit instruction is empty — skipping.")
        return False

    # ── Get original dimensions ──────────────────────────────────────────
    with Image.open(image_path) as img:
        orig_width, orig_height = img.size

    # ── Build prompt ─────────────────────────────────────────────────────
    prompt = build_prompt(edit_instruction)

    # ── Call API with infinite retries (never skip) ──────────────────────
    attempt = 0
    while True:
        attempt += 1
        try:
            with open(image_path, "rb") as img_file:
                result = client.images.edit(
                    model=MODEL,
                    image=img_file,
                    prompt=prompt,
                    quality=QUALITY,
                    size=SIZE,
                )

            # ── Decode response ──────────────────────────────────────────
            image_b64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_b64)

            # ── Resize to original dimensions ────────────────────────────
            edited_img = Image.open(io.BytesIO(image_bytes))
            if edited_img.size != (orig_width, orig_height):
                edited_img = edited_img.resize(
                    (orig_width, orig_height), Image.LANCZOS
                )

            # ── Convert to RGB (JPEG doesn't support alpha) ─────────────
            if edited_img.mode in ("RGBA", "P", "LA"):
                edited_img = edited_img.convert("RGB")

            # ── Save ─────────────────────────────────────────────────────
            output_path = os.path.join(case_dir, "edited_image.jpg")
            edited_img.save(output_path, "JPEG", quality=95)

            log.info(
                f"[Case {case_id}] ✓ Saved ({orig_width}x{orig_height}) — "
                f"instruction: \"{edit_instruction[:60]}...\""
                if len(edit_instruction) > 60
                else f"[Case {case_id}] ✓ Saved ({orig_width}x{orig_height}) — "
                     f"instruction: \"{edit_instruction}\""
            )
            return True

        except Exception as e:
            wait = min(RETRY_BACKOFF ** attempt, MAX_RETRY_WAIT)
            # For rate-limit errors, parse the suggested wait time
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                import re as _re
                m = _re.search(r"try again in (\d+)", err_str)
                if m:
                    wait = max(wait, int(m.group(1)) + 1)
            log.error(
                f"[Case {case_id}] Attempt {attempt} failed: {e}"
            )
            log.info(f"[Case {case_id}] Retrying in {wait}s …")
            time.sleep(wait)


def discover_test_cases(data_dir: str) -> list[tuple[str, str]]:
    """
    Return sorted list of (case_dir, case_id) for every natural-numbered
    subdirectory in data_dir.
    """
    cases = []
    for entry in os.listdir(data_dir):
        full_path = os.path.join(data_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if entry.isdigit() or os.path.exists(os.path.join(full_path, "edit_instruction.txt")):
            # numeric (000) or combined-staging (<task>__ep<NNN>) case dirs
            cases.append((full_path, entry))
    # Sort: numeric by value first, otherwise lexicographically
    cases.sort(key=lambda x: (0, int(x[1])) if x[1].isdigit() else (1, x[1]))
    return cases


def main():
    parser = argparse.ArgumentParser(
        description="GPT-Image-1.5 instruction-guided image editing inference."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to test data directory containing numbered subdirectories.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of concurrent API workers (default: {DEFAULT_WORKERS}).",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        log.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)

    # ── Discover test cases ──────────────────────────────────────────────
    cases = discover_test_cases(data_dir)
    if not cases:
        log.error("No numbered subdirectories found.")
        sys.exit(1)

    # ── Skip already-completed cases ─────────────────────────────────────
    remaining = [
        (d, cid) for d, cid in cases
        if not os.path.exists(os.path.join(d, "edited_image.jpg"))
    ]
    done_count = len(cases) - len(remaining)

    log.info(f"Dataset       : {data_dir}")
    log.info(f"Total cases   : {len(cases)}")
    log.info(f"Already done  : {done_count}")
    log.info(f"To process    : {len(remaining)}")
    log.info(f"Workers       : {args.workers}")
    log.info(f"Model         : {MODEL}  |  Quality: {QUALITY}  |  Size: {SIZE}")
    log.info("=" * 70)

    if not remaining:
        log.info("All test cases already processed. Nothing to do.")
        return

    # ── Create client ────────────────────────────────────────────────────
    client = OpenAI(api_key=API_KEY)

    # ── Run inference ────────────────────────────────────────────────────
    success = 0
    failure = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_test_case, client, case_dir, case_id): case_id
            for case_dir, case_id in remaining
        }
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                if future.result():
                    success += 1
                else:
                    failure += 1
            except Exception as e:
                log.error(f"[Case {case_id}] Unexpected error: {e}")
                failure += 1

            total_done = success + failure
            if total_done % 50 == 0 or total_done == len(remaining):
                elapsed = time.time() - start_time
                rate = total_done / elapsed if elapsed > 0 else 0
                log.info(
                    f"Progress: {total_done}/{len(remaining)} "
                    f"({success} ok, {failure} fail) — "
                    f"{elapsed:.0f}s elapsed, {rate:.1f} cases/s"
                )

    elapsed = time.time() - start_time
    log.info("=" * 70)
    log.info(
        f"DONE — {success} succeeded, {failure} failed, "
        f"{elapsed:.1f}s total ({elapsed/max(success+failure,1):.1f}s/case avg)"
    )


if __name__ == "__main__":
    main()
