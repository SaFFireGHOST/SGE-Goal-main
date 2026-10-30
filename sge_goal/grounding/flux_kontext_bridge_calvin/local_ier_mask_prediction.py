"""
Local (no Modal) Initial Edit Region (IER) mask prediction.

Same predictor as ``modal_ier_mask_prediction.py`` (Florence-2 + Grounding DINO +
BLIP-2 + SAM), run on the local GPU, with an optional resume that skips test cases
whose ``initial_edit_region_mask.jpg`` already exists.

Usage:
    python local_ier_mask_prediction.py --data-dir /abs/path/to/test_data [--skip-existing]
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_initial_edit_region_mask import InitialEditRegionMaskPredictor  # noqa: E402

OUTPUT_FILENAME = "initial_edit_region_mask.jpg"


def find_image(directory, base_name):
    for ext in ("jpg", "jpeg", "png", "bmp", "webp", "tiff"):
        path = os.path.join(directory, f"{base_name}.{ext}")
        if os.path.isfile(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Local IER mask prediction.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip test cases that already have initial_edit_region_mask.jpg (resume).")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    test_cases = sorted(
        (d for d in os.listdir(data_dir)
         if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    pending = [
        c for c in test_cases
        if not (args.skip_existing and os.path.isfile(os.path.join(data_dir, c, OUTPUT_FILENAME)))
    ]
    print(f"{len(test_cases)} test cases, {len(pending)} to process.")
    if not pending:
        return

    predictor = InitialEditRegionMaskPredictor(device=args.device)
    for n, case_id in enumerate(pending, 1):
        case_dir = os.path.join(data_dir, case_id)
        instr_path = os.path.join(case_dir, "edit_instruction.txt")
        image_path = find_image(case_dir, "original_image")
        if image_path is None or not os.path.isfile(instr_path):
            print(f"[{case_id}] missing original_image or edit_instruction.txt, skipping")
            continue
        with open(instr_path, "r", encoding="utf-8") as f:
            instruction = f.read().strip()
        image = Image.open(image_path).convert("RGB")
        mask = predictor.predict_mask(image, instruction)
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
            os.path.join(case_dir, OUTPUT_FILENAME), quality=95
        )
        print(f"[{case_id}] IER done ({n}/{len(pending)})", flush=True)


if __name__ == "__main__":
    main()
