"""Does initial-mask quality predict downstream success? Per-row IoU (pred vs GT initial mask) vs SGE-Goal / GT-goal success.

Also writes a side-by-side image (original | GT initial | pred initial | pred final | SGE goal) for sample rows.
"""
import glob
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

RUN = os.path.dirname(os.path.abspath(__file__))
D = f"{RUN}/CALVIN-EVALS/sge_goal/calvin_abcd_test_data"
t = pq.read_table(sorted(glob.glob(f"{RUN}/calvin/hf_dataset/data/test-*.parquet"))[0],
                  columns=["initial_edit_region_mask", "final_edit_region_mask"])
gt_i = t["initial_edit_region_mask"].to_pylist()
gt_f = t["final_edit_region_mask"].to_pylist()


def dec(cell):
    return Image.open(io.BytesIO(cell["bytes"])).convert("L")


def success(folder):
    out = {}
    for f in glob.glob(f"{RUN}/{folder}/per_sample_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            out[r["hf_row"]] = (int(r["success"]), r["task"])
    return out


sge, gtg = success("eval_sge_goal"), success("eval_ground_truth")

iou_i, iou_f_gt_vs_pred, cov = [], [], []
for i, cell in enumerate(gt_i):
    g = np.asarray(dec(cell)) > 127
    p = np.asarray(Image.open(f"{D}/{i + 1}/initial_edit_region_mask.jpg").convert("L").resize(g.shape[::-1])) > 127
    iou_i.append((g & p).sum() / max((g | p).sum(), 1))
    cov.append(p.mean())
    gf = np.asarray(dec(gt_f[i])) > 127
    pf = np.asarray(Image.open(f"{D}/{i + 1}/final_edit_region_mask.jpg").convert("L").resize(gf.shape[::-1])) > 127
    iou_f_gt_vs_pred.append((gf & pf).sum() / max((gf | pf).sum(), 1))
iou_i, iou_f, cov = map(np.array, (iou_i, iou_f_gt_vs_pred, cov))
s = np.array([sge[i][0] for i in range(1087)])
g = np.array([gtg[i][0] for i in range(1087)])
task = np.array([sge[i][1] for i in range(1087)])

print(f"overall: SGE-Goal {s.mean():.3f}   GT goals {g.mean():.3f}")
print(f"mean IoU initial mask {iou_i.mean():.3f}   mean IoU FINAL mask {iou_f.mean():.3f}")
print(f"corr(IoU_initial, success) = {np.corrcoef(iou_i, s)[0, 1]:.3f}   corr(IoU_final, success) = {np.corrcoef(iou_f, s)[0, 1]:.3f}")
print("\nSGE-Goal success by initial-mask IoU bucket:")
for lo, hi in [(0, .05), (.05, .2), (.2, .5), (.5, 1.01)]:
    m = (iou_i >= lo) & (iou_i < hi)
    print(f"  IoU [{lo:.2f},{hi:.2f})  n={m.sum():4d}  SGE-Goal {s[m].mean():.3f}   GT goals {g[m].mean():.3f}")
print("\nSGE-Goal success by predicted initial-mask coverage:")
for name, m in [("<1% white", cov < .01), ("1-50% white", (cov >= .01) & (cov <= .5)), (">50% white", cov > .5)]:
    print(f"  {name:12s} n={m.sum():4d}  SGE-Goal {s[m].mean():.3f}   GT goals {g[m].mean():.3f}")
print("\nPer task (n, GT-goal SR, SGE-goal SR, mean IoU_init):")
for tk in sorted(set(task)):
    m = task == tk
    print(f"  {tk:28s} n={m.sum():3d}  GT {g[m].mean():.2f}  SGE {s[m].mean():.2f}  IoU {iou_i[m].mean():.2f}")

rows = [int(r) for r in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 40, 200, 500, 800]
W = Image.new("RGB", (160 * 5, 160 * len(rows)))
for r, i in enumerate(rows):
    ims = [Image.open(f"{D}/{i + 1}/original_image.png").convert("RGB"), dec(gt_i[i]).convert("RGB"),
           Image.open(f"{D}/{i + 1}/initial_edit_region_mask.jpg").convert("RGB"),
           Image.open(f"{D}/{i + 1}/final_edit_region_mask.jpg").convert("RGB"),
           Image.open(f"{D}/{i + 1}/edited_image.jpg").convert("RGB")]
    for c, im in enumerate(ims):
        W.paste(im.resize((160, 160)), (160 * c, 160 * r))
    print(f"row {i}: {task[i]}  IoU_init={iou_i[r if False else i]:.2f}  SGE success={s[i]}  GT success={g[i]}")
W.save(f"{RUN}/ier_examples.png")
