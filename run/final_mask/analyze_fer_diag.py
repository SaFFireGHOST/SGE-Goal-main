"""Step 2 diagnostics (run/final_mask/run_diag.sh): why are predicted final masks poor?

On 256 HF validation tuples with learned M_i, 8 Step 2 samples each (seeds 1000 s + case index):
  mean      IoU of a single sample (vs ground-truth final mask)
  best@N    oracle: best of the first N samples  -> is the right mask in Step 2's distribution?
  medoid@N  sample with the highest mean IoU to the other N-1  -> label-free selection
  vote@N    pixel majority of the first N samples              -> label-free consensus
  pairwise  mean IoU between two samples of the same tuple     -> sample diversity
Plus Step 2 with ground-truth M_i on 256 train vs 256 validation tuples (overfitting vs ambiguity).
Masks binarized at 127 (as run/mask_iou_vs_gt.py). Writes run/final_mask/fer_diag_summary.json.
"""
import glob
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = f"{RUN}/calvin/hf_dataset/data"
D = f"{RUN}/CALVIN-EVALS/fer_diag"
SUB = "calvin_abcd_test_data"
index = json.load(open(f"{RUN}/final_mask/fer_diag_index.json"))
gt = {}
for split in ("validation", "train"):
    t = pq.read_table(sorted(glob.glob(f"{HF}/{split}-*.parquet")), columns=["final_edit_region_mask"])
    gt[split] = t["final_edit_region_mask"].to_pylist()


def gt_mask(e):
    return np.asarray(Image.open(io.BytesIO(gt[e["split"]][e["row"]]["bytes"])).convert("L")) > 127


def pred_mask(set_name, k, shape):
    im = Image.open(f"{D}/{set_name}/{SUB}/{k}/final_edit_region_mask.jpg").convert("L").resize(shape[::-1])
    return np.asarray(im) > 127


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


out = {}
seeds = sorted(s for s in index if s.startswith("val_pred_s"))
cases = sorted(index[seeds[0]], key=int)
ious = np.zeros((len(cases), len(seeds)))
stats = {f"{m}@{n}": [] for m in ("best", "medoid", "vote") for n in (2, 4, 8)}
pairwise, empty = [], []
for i, k in enumerate(cases):
    g = gt_mask(index[seeds[0]][k])
    ps = [pred_mask(s, k, g.shape) for s in seeds]
    ious[i] = [iou(p, g) for p in ps]
    empty.append(np.mean([p.sum() == 0 for p in ps]))
    pw = np.array([[iou(a, b) for b in ps] for a in ps])
    pairwise.append(pw[np.triu_indices(len(ps), 1)].mean())
    for n in (2, 4, 8):
        stats[f"best@{n}"].append(ious[i, :n].max())
        stats[f"medoid@{n}"].append(ious[i, :n][(pw[:n, :n].sum(1) - 1).argmax()])
        stats[f"vote@{n}"].append(iou(np.mean(ps[:n], 0) >= 0.5, g))
out["val_learned_mi"] = {"n": len(cases), "mean": float(ious.mean()),
                         "frac_iou_ge_0.5_single": float((ious >= 0.5).mean()),
                         "pairwise_iou_between_samples": float(np.mean(pairwise)),
                         "frac_empty_masks": float(np.mean(empty)),
                         **{k: float(np.mean(v)) for k, v in stats.items()},
                         **{f"frac_ge_0.5_{k}": float(np.mean(np.array(v) >= 0.5)) for k, v in stats.items()}}
for name in ("val_gt_s0", "train_gt_s0"):
    v = [iou(pred_mask(name, k, (g := gt_mask(e)).shape), g) for k, e in index[name].items()]
    out[name] = {"n": len(v), "mean": float(np.mean(v)), "frac_iou_ge_0.5": float(np.mean(np.array(v) >= 0.5))}

for k, v in out.items():
    print(k)
    for m, x in v.items():
        print(f"  {m:32s} {x:.3f}" if isinstance(x, float) else f"  {m:32s} {x}")
json.dump(out, open(f"{RUN}/final_mask/fer_diag_summary.json", "w"), indent=1)
