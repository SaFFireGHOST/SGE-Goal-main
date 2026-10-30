"""Is the final mask predictable from (I_o, T, M_i)? Nearest-neighbour (retrieval) baseline vs Step 2.

Retrieval: among HF train tuples with the same instruction (neighbours from the same episode excluded),
rank by IoU of initial masks minus 0.1 x standardized 52x52 image distance; the predicted final mask is the
pixel majority vote of the top 5 neighbours' ground-truth final masks (nn5_vote; nn1 = top neighbour only).

  python run/final_mask/nn_baseline.py              # 256 val tuples of fer_diag_index.json (ground-truth M_i)
                                                    # + all 1087 test rows (learned M_i from sge_goal_union_ier)
                                                    # -> run/final_mask/nn_baseline.json
  python run/final_mask/nn_baseline.py --write-test-folder
      also writes CALVIN-EVALS/sge_goal_union_nnmf/: sge_goal_union_ier's inputs (learned M_i) with the
      nn5_vote mask as final_edit_region_mask.png (input to Step 3, see run_nn_eval.sh).
"""
import argparse
import glob
import io
import json
import os
import shutil
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = f"{RUN}/calvin/hf_dataset/data"
UNION = f"{RUN}/CALVIN-EVALS/sge_goal_union_ier/calvin_abcd_test_data"
NNMF = f"{RUN}/CALVIN-EVALS/sge_goal_union_nnmf/calvin_abcd_test_data"
LOW, K = 52, 5


def episode(i):
    return i.rsplit("_atomic_", 1)[0]


def load(split):
    return pq.read_table(sorted(glob.glob(f"{HF}/{split}-*.parquet")),
                         columns=["id", "original_image", "initial_edit_region_mask", "final_edit_region_mask",
                                  "edit_instruction"]).to_pylist()


def mask(cell, size=None):
    im = Image.open(io.BytesIO(cell["bytes"])).convert("L")
    return np.asarray(im.resize((size, size), Image.BILINEAR) if size else im) > 127


def file_mask(path):
    return np.asarray(Image.open(path).convert("L").resize((208, 208))) > 127


def low(m):
    return (np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((LOW, LOW), Image.BILINEAR)) > 127).ravel()


def img(cell):
    return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB").resize((LOW, LOW), Image.BILINEAR),
                      dtype=np.float32).ravel() / 255


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


class Retriever:
    def __init__(self, train):
        self.train = train
        self.by_instr = defaultdict(list)
        for j, r in enumerate(train):
            self.by_instr[r["edit_instruction"].strip().lower()].append(j)
        self.mi = np.stack([mask(r["initial_edit_region_mask"], LOW).ravel() for r in train])
        self.img = np.stack([img(r["original_image"]) for r in train])
        self.ep = np.array([episode(r["id"]) for r in train])

    def __call__(self, row, mi_full):
        """Ground-truth final masks of the K nearest train tuples, nearest first."""
        cand = np.array(self.by_instr.get(row["edit_instruction"].strip().lower(), []), dtype=int)
        cand = cand[self.ep[cand] != episode(row["id"])]
        if not len(cand):
            cand = np.flatnonzero(self.ep != episode(row["id"]))
        mi = low(mi_full)
        score = (self.mi[cand] & mi).sum(1) / np.maximum((self.mi[cand] | mi).sum(1), 1)
        dist = np.abs(self.img[cand] - img(row["original_image"])).mean(1)
        score = score - dist / max(dist.std(), 1e-6) * 0.1
        return [mask(self.train[j]["final_edit_region_mask"]) for j in cand[np.argsort(-score)[:K]]]


def summarize(res):
    out = {k: float(np.mean(v)) for k, v in res.items()}
    out.update({f"frac_ge_0.5_{k}": float(np.mean(np.array(v) >= 0.5)) for k, v in res.items()})
    for k, v in out.items():
        print(f"  {k:36s} {v:.3f}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write-test-folder", action="store_true")
    args = p.parse_args()

    retrieve = Retriever(load("train"))
    out = {}

    val = load("validation")
    index = json.load(open(f"{RUN}/final_mask/fer_diag_index.json"))["val_gt_s0"]
    res = defaultdict(list)
    for k, e in index.items():
        r = val[e["row"]]
        g, mi = mask(r["final_edit_region_mask"]), mask(r["initial_edit_region_mask"])
        nn = retrieve(r, mi)
        res["nn1"].append(iou(nn[0], g))
        res["nn5_vote"].append(iou(np.mean(nn, 0) >= 0.5, g))
        res["nn5_best"].append(max(iou(m, g) for m in nn))
        res["copy_initial_mask"].append(iou(mi, g))
    print("validation (ground-truth M_i)")
    out["validation_gt_mi"] = summarize(res)

    test = load("test")
    res = defaultdict(list)
    for i, r in enumerate(test):
        g, mi = mask(r["final_edit_region_mask"]), file_mask(f"{UNION}/{i + 1}/initial_edit_region_mask.jpg")
        nn = retrieve(r, mi)
        vote = np.mean(nn, 0) >= 0.5
        res["nn1"].append(iou(nn[0], g))
        res["nn5_vote"].append(iou(vote, g))
        res["step2_current"].append(iou(file_mask(f"{UNION}/{i + 1}/final_edit_region_mask.jpg"), g))
        res["copy_initial_mask"].append(iou(mi, g))
        if args.write_test_folder:
            d = f"{NNMF}/{i + 1}"
            os.makedirs(d, exist_ok=True)
            for name in ("original_image.png", "edit_instruction.txt", "initial_edit_region_mask.jpg"):
                shutil.copy2(f"{UNION}/{i + 1}/{name}", f"{d}/{name}")
            Image.fromarray(vote.astype(np.uint8) * 255).save(f"{d}/final_edit_region_mask.png")
    print("test (learned M_i)")
    out["test_learned_mi"] = summarize(res)
    json.dump(out, open(f"{RUN}/final_mask/nn_baseline.json", "w"), indent=1)


if __name__ == "__main__":
    main()
