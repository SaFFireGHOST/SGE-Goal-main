"""Build a CALVIN-EVALS folder whose initial masks are the ground-truth M_i from the HF test parquet.

Copies original_image + edit_instruction.txt from --src-dir (case N = parquet row N-1) and writes the
GT mask as lossless initial_edit_region_mask.png. Steps 2-3 then run on it unchanged.

Usage: python run/make_gt_ier_folder.py --src-dir ... --out-dir ... [--parquet "run/calvin/hf_dataset/data/test-*.parquet"]
"""
import argparse
import glob
import io
import os
import shutil

import pyarrow.parquet as pq
from PIL import Image

RUN = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--src-dir", required=True)
p.add_argument("--out-dir", required=True)
p.add_argument("--parquet", default=f"{RUN}/calvin/hf_dataset/data/test-*.parquet")
args = p.parse_args()

col = pq.read_table(sorted(glob.glob(args.parquet))[0], columns=["initial_edit_region_mask"])["initial_edit_region_mask"].to_pylist()
cases = sorted((d for d in os.listdir(args.src_dir) if d.isdigit()), key=int)
for case in cases:
    src, dst = os.path.join(args.src_dir, case), os.path.join(args.out_dir, case)
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name.startswith("original_image.") or name == "edit_instruction.txt":
            shutil.copy2(os.path.join(src, name), dst)
    mask = Image.open(io.BytesIO(col[int(case) - 1]["bytes"])).convert("L")
    mask.point(lambda v: 255 if v > 127 else 0).save(os.path.join(dst, "initial_edit_region_mask.png"))
print(f"wrote {len(cases)} cases with GT initial masks to {args.out_dir}")
