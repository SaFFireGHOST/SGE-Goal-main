"""
Image sheet of counterfactual pairs (cf_pairs.json) for a visual check before training. Per pair, one row:
  image A | image B | M_i A on image A | M_f A on image A (T_A) | M_f B on image A (T_B) | M_f A vs B
The question is whether M_f B is a sensible answer for image A under T_B (the block it marks is where A's block is).

    ~/.conda/envs/sgegen/bin/python run/final_mask/sheet_cf_pairs.py --split train --l1-max 3 --n 12 --out ...
"""

import argparse
import glob
import io
import json
import os
import random

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RUN = os.path.join(REPO, "run")
COLS = ["id", "original_image", "initial_edit_region_mask", "final_edit_region_mask"]
S = 208


def overlay(img, m, color):
    a = np.asarray(img, np.float32).copy()
    a[m] = 0.45 * a[m] + 0.55 * np.array(color, np.float32)
    return Image.fromarray(a.astype(np.uint8))


def both(ma, mb):
    a = np.zeros((S, S, 3), np.uint8)
    a[ma & ~mb] = (0, 200, 0)
    a[mb & ~ma] = (220, 40, 40)
    a[ma & mb] = (230, 230, 230)
    return Image.fromarray(a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=f"{RUN}/final_mask/cf_pairs.json")
    p.add_argument("--split", default="train")
    p.add_argument("--l1-min", type=float, default=0.0)
    p.add_argument("--l1-max", type=float, default=99.0)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    pairs = [q for q in json.load(open(args.pairs))["pairs"]
             if q["split"] == args.split and args.l1_min <= q["scene_l1"] < args.l1_max]
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[:args.n]
    hf = "validation" if args.split == "val" else "train"
    rows = []
    for f in sorted(glob.glob(f"{RUN}/calvin/hf_dataset/data/{hf}-*.parquet")):
        rows += pq.read_table(f, columns=COLS).to_pylist()

    def im(r, k):
        return Image.open(io.BytesIO(r[k]["bytes"])).convert("RGB" if k == "original_image" else "L").resize((S, S))

    head, gap = 34, 6
    sheet = Image.new("RGB", (6 * S + 5 * gap, len(pairs) * (S + head + gap)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for k, q in enumerate(pairs):
        a, b = rows[q["a"]], rows[q["b"]]
        ia, ib = im(a, "original_image"), im(b, "original_image")
        mia = np.asarray(im(a, "initial_edit_region_mask")) > 127
        mfa = np.asarray(im(a, "final_edit_region_mask")) > 127
        mfb = np.asarray(im(b, "final_edit_region_mask")) > 127
        tiles = [ia, ib, overlay(ia, mia, (255, 220, 0)), overlay(ia, mfa, (0, 200, 0)), overlay(ia, mfb, (220, 40, 40)),
                 both(mfa, mfb)]
        y = k * (S + head + gap)
        d.text((2, y + 2), f"A {q['id_a']}: {q['instr_a']}   [L1 {q['scene_l1']:.2f}, M_f IoU {q['mf_iou']:.2f}, "
                           f"M_i IoU {q['mi_iou']:.2f}]", fill=(0, 120, 0))
        d.text((2, y + 17), f"B {q['id_b']}: {q['instr_b']}", fill=(180, 20, 20))
        for c, t in enumerate(tiles):
            sheet.paste(t, (c * (S + gap), y + head))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sheet.save(args.out)
    print(args.out, len(pairs))


if __name__ == "__main__":
    main()
