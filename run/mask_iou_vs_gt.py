"""IoU of a CALVIN-EVALS folder's masks against the ground-truth masks in the HF test parquet.

Case folder N is compared with parquet row N-1. Masks are binarized at 127 (Step 2 outputs are grey-level).

Usage: python run/mask_iou_vs_gt.py --mask final --dirs name1=path1 name2=path2 [--out results.json]
"""
import argparse
import glob
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

RUN = os.path.dirname(os.path.abspath(__file__))
FILES = {"initial": "initial_edit_region_mask", "final": "final_edit_region_mask"}

p = argparse.ArgumentParser()
p.add_argument("--mask", choices=list(FILES), required=True)
p.add_argument("--dirs", nargs="+", required=True, help="name=path pairs")
p.add_argument("--parquet", default=f"{RUN}/calvin/hf_dataset/data/test-*.parquet")
p.add_argument("--out", default=None)
args = p.parse_args()

stem = FILES[args.mask]
gt_col = pq.read_table(sorted(glob.glob(args.parquet))[0], columns=[stem])[stem].to_pylist()
gts = [np.asarray(Image.open(io.BytesIO(c["bytes"])).convert("L")) > 127 for c in gt_col]


def find(case_dir):
    for ext in ("jpg", "jpeg", "png"):
        path = os.path.join(case_dir, f"{stem}.{ext}")
        if os.path.isfile(path):
            return path
    return None


results = {}
for spec in args.dirs:
    name, path = spec.split("=", 1)
    ious, cov, missing = [], [], 0
    for i, gt in enumerate(gts):
        f = find(os.path.join(path, str(i + 1)))
        if f is None:
            missing += 1
            continue
        pred = np.asarray(Image.open(f).convert("L").resize(gt.shape[::-1], Image.BILINEAR)) > 127
        union = (gt | pred).sum()
        ious.append((gt & pred).sum() / union if union else 1.0)
        cov.append(pred.mean())
    ious, cov = np.array(ious), np.array(cov)
    results[name] = {"n": len(ious), "missing": missing, "mean_iou": float(ious.mean()),
                     "median_iou": float(np.median(ious)), "iou_lt_0.1": float((ious < .1).mean()),
                     "iou_gt_0.5": float((ious > .5).mean()), "mean_coverage": float(cov.mean()),
                     "gt_mean_coverage": float(np.mean([g.mean() for g in gts]))}

print(f"{args.mask} mask IoU vs ground truth")
for name, r in results.items():
    print(f"  {name:22s} n={r['n']:4d} missing={r['missing']:4d}  mean IoU {r['mean_iou']:.3f}  median {r['median_iou']:.3f}"
          f"  IoU>0.5 {r['iou_gt_0.5']:.0%}  IoU<0.1 {r['iou_lt_0.1']:.0%}  coverage {r['mean_coverage']:.3f}"
          f" (GT {r['gt_mean_coverage']:.3f})")
if args.out:
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
