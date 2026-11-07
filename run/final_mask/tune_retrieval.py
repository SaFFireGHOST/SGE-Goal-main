"""Tune the retrieval (nearest-neighbour) final-mask predictor of nn_baseline.py on the HF validation tuples with
learned M_i (as at test time), then evaluate the chosen configuration once on the 1087 test rows.

A query's neighbours are train tuples from other episodes, either with the same instruction (nn_baseline.py) or
the same CALVIN task (instruction -> task from ~/calvin/calvin_models/conf/annotations, 34 tasks), ranked by
IoU of 52x52 initial masks minus alpha x standardized 52x52 image L1. The prediction aggregates the top-K
neighbours' ground-truth final masks (pixel vote at threshold tau, or the medoid), optionally after shifting each
by the offset between the query's and the neighbour's initial-mask centroids.

Staged search (val mean IoU at 208 px): 1. ranking (pool, neighbour M_i ground truth / learned, alpha) with the
nn_baseline aggregation; 2. aggregation (K, vote tau / medoid, shift) for the best rankings.

    python run/final_mask/tune_retrieval.py                       # -> run/final_mask/retrieval_tuning.json
    python run/final_mask/tune_retrieval.py --write-test-folder   # + CALVIN-EVALS/sge_goal_union_nnmf2/
"""
import argparse
import glob
import io
import itertools
import json
import os
import shutil

import numpy as np
import pyarrow.parquet as pq
import yaml
from PIL import Image

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = f"{RUN}/calvin/hf_dataset"
UNION = f"{RUN}/CALVIN-EVALS/sge_goal_union_ier/calvin_abcd_test_data"
OUT_DIR = f"{RUN}/CALVIN-EVALS/sge_goal_union_nnmf2/calvin_abcd_test_data"
ANN = os.path.expanduser("~/calvin/calvin_models/conf/annotations")
LOW, FULL = 52, 208
BASELINE = {"pool": "instr", "cand_mi": "gt", "alpha": 0.1, "k": 5, "agg": "vote0.5", "shift": False}


def episode(i):
    return i.rsplit("_atomic_", 1)[0]


def task_map():
    m = {}
    for f in ("new_playtable.yaml", "new_playtable_validation.yaml"):
        for task, ins in yaml.safe_load(open(f"{ANN}/{f}")).items():
            m.update({s.strip().lower(): task for s in ins})
    return m


def full_mask(b):
    im = Image.open(io.BytesIO(b)).convert("L")
    return np.asarray(im if im.size == (FULL, FULL) else im.resize((FULL, FULL), Image.BILINEAR)) > 127


def low(m):
    return (np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((LOW, LOW), Image.BILINEAR)) > 127).ravel()


def img(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((LOW, LOW), Image.BILINEAR),
                      dtype=np.float32).ravel() / 255


def centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([ys.mean(), xs.mean()]) if len(ys) else np.array([FULL / 2, FULL / 2])


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def shifted(m, dy, dx):
    out = np.zeros_like(m)
    dy, dx = int(round(dy)), int(round(dx))
    ys, yd = (slice(0, FULL - dy), slice(dy, FULL)) if dy >= 0 else (slice(-dy, FULL), slice(0, FULL + dy))
    xs, xd = (slice(0, FULL - dx), slice(dx, FULL)) if dx >= 0 else (slice(-dx, FULL), slice(0, FULL + dx))
    out[yd, xd] = m[ys, xs]
    return out


class Split:
    """Per-tuple features. mi: dict source -> full-res initial masks (ground truth and / or learned)."""

    def __init__(self, split, tasks, pred_mi=None, test_mi_dir=None):
        cols = ["id", "original_image", "initial_edit_region_mask", "final_edit_region_mask", "edit_instruction"]
        t = pq.read_table(sorted(glob.glob(f"{HF}/data/{split}-*.parquet")), columns=cols).to_pydict()
        self.ids = t["id"]
        self.ep = np.array([episode(i) for i in self.ids])
        self.instr = [s.strip().lower() for s in t["edit_instruction"]]
        self.task = [tasks.get(s, s) for s in self.instr]
        self.img = np.stack([img(c["bytes"]) for c in t["original_image"]])
        self.mf = np.stack([full_mask(c["bytes"]) for c in t["final_edit_region_mask"]])
        self.mi = {"gt": [full_mask(c["bytes"]) for c in t["initial_edit_region_mask"]]}
        if pred_mi:
            p = pq.read_table(pred_mi).to_pydict()
            pm = dict(zip(p["id"], p["pred_initial_mask"]))
            self.mi["pred"] = [full_mask(pm[i]) for i in self.ids]
        if test_mi_dir:  # learned M_i exactly as the test pipeline used it (folder i + 1)
            self.mi["pred"] = [np.asarray(Image.open(f"{test_mi_dir}/{i + 1}/initial_edit_region_mask.jpg")
                                          .convert("L").resize((FULL, FULL))) > 127 for i in range(len(self.ids))]
        self.mi_low = {k: np.stack([low(m) for m in v]) for k, v in self.mi.items()}
        self.cent = {k: np.stack([centroid(m) for m in v]) for k, v in self.mi.items()}
        self.originals = t["original_image"]


class Retriever:
    def __init__(self, train):
        self.tr = train
        self.by = {"instr": {}, "task": {}}
        for j in range(len(train.ids)):
            self.by["instr"].setdefault(train.instr[j], []).append(j)
            self.by["task"].setdefault(train.task[j], []).append(j)
        self.by = {k: {g: np.array(v) for g, v in d.items()} for k, d in self.by.items()}

    def rank(self, q, i, pool, cand_mi, alpha, kmax):
        """Indices of the kmax nearest train tuples to query tuple i of split q (its learned M_i)."""
        key = q.instr[i] if pool == "instr" else q.task[i]
        cand = self.by[pool].get(key, np.arange(0))
        cand = cand[self.tr.ep[cand] != q.ep[i]]
        if not len(cand):
            cand = np.flatnonzero(self.tr.ep != q.ep[i])
        mi = q.mi_low["pred"][i]
        cm = self.tr.mi_low[cand_mi][cand]
        score = (cm & mi).sum(1) / np.maximum((cm | mi).sum(1), 1)
        if alpha:
            dist = np.abs(self.tr.img[cand] - q.img[i]).mean(1)
            score = score - alpha * dist / max(dist.std(), 1e-6)
        return cand[np.argsort(-score, kind="stable")[:kmax]]

    def predict(self, q, i, nbrs, cand_mi, k, agg, shift):
        masks = self.tr.mf[nbrs[:k]]
        if shift:
            off = q.cent["pred"][i] - self.tr.cent[cand_mi][nbrs[:k]]
            masks = np.stack([shifted(m, dy, dx) for m, (dy, dx) in zip(masks, off)])
        if agg == "medoid":
            if len(masks) <= 2:
                return masks[0]
            flat = masks.reshape(len(masks), -1)
            inter = flat.astype(np.float32) @ flat.T.astype(np.float32)
            area = flat.sum(1)
            pair = inter / np.maximum(area[:, None] + area[None] - inter, 1)
            return masks[int(np.argmax(pair.sum(1)))]
        return masks.mean(0) >= float(agg[len("vote"):])


def evaluate(ret, q, cfgs):
    """Mean IoU and fraction >= 0.5 for each config (configs sharing a ranking reuse it)."""
    res = {}
    for rk, group in itertools.groupby(sorted(cfgs, key=lambda c: (c["pool"], c["cand_mi"], c["alpha"])),
                                       key=lambda c: (c["pool"], c["cand_mi"], c["alpha"])):
        group = list(group)
        kmax = max(c["k"] for c in group)
        scores = {json.dumps(c, sort_keys=True): [] for c in group}
        for i in range(len(q.ids)):
            nbrs = ret.rank(q, i, *rk, kmax)
            for c in group:
                pred = ret.predict(q, i, nbrs, c["cand_mi"], c["k"], c["agg"], c["shift"])
                scores[json.dumps(c, sort_keys=True)].append(iou(pred, q.mf[i]))
        for key, v in scores.items():
            res[key] = {"cfg": json.loads(key), "iou": float(np.mean(v)), "frac_ge_0.5": float(np.mean(np.array(v) >= 0.5)),
                        "per_row": v}
    return res


def show(title, res, top=None):
    print(f"\n{title}")
    rows = sorted(res.values(), key=lambda r: -r["iou"])[:top]
    for r in rows:
        c = r["cfg"]
        print(f"  {r['iou']:.4f}  ({r['frac_ge_0.5']:.1%} >= 0.5)  pool={c['pool']:5s} cand_mi={c['cand_mi']:4s} "
              f"alpha={c['alpha']:<4} k={c['k']:<3} agg={c['agg']:8s} shift={c['shift']}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write-test-folder", action="store_true")
    args = p.parse_args()

    tasks = task_map()
    train = Split("train", tasks, pred_mi=f"{HF}/pred_ier/train.parquet")
    val = Split("validation", tasks, pred_mi=f"{HF}/pred_ier/validation.parquet")
    ret = Retriever(train)
    print(f"train {len(train.ids)}, validation {len(val.ids)} (learned M_i)", flush=True)
    copy = float(np.mean([iou(val.mi["pred"][i], val.mf[i]) for i in range(len(val.ids))]))
    print(f"copy learned M_i: {copy:.4f}")

    # 1. ranking, with the nn_baseline aggregation
    s1 = [dict(BASELINE, pool=pool, cand_mi=cm, alpha=a) for pool in ("instr", "task") for cm in ("gt", "pred")
          for a in (0, 0.1, 0.3, 1, 3, 10)]
    r1 = evaluate(ret, val, s1)
    show("stage 1: ranking (k=5, vote 0.5, no shift)", r1)
    base = r1[json.dumps(BASELINE, sort_keys=True)]["iou"]

    # 2. aggregation, for the 3 best rankings
    best_rk = [r["cfg"] for r in sorted(r1.values(), key=lambda r: -r["iou"])[:3]]
    s2 = [dict(c, k=k, agg=agg, shift=sh) for c in best_rk for k in (1, 3, 5, 9, 15, 25)
          for agg in ("vote0.3", "vote0.4", "vote0.5", "medoid") for sh in (False, True)]
    r2 = evaluate(ret, val, s2)
    rows = show("stage 2: aggregation (top 15)", r2, top=15)
    best = rows[0]
    print(f"\nbaseline (nn_baseline nn5_vote) {base:.4f} -> best {best['iou']:.4f} on validation")

    # oracle headroom: best single neighbour among the top 25 of the best ranking
    c = best["cfg"]
    orc = [max(iou(train.mf[j], val.mf[i]) for j in ret.rank(val, i, c["pool"], c["cand_mi"], c["alpha"], 25))
           for i in range(len(val.ids))]
    print(f"oracle best-of-25 neighbours (same ranking): {np.mean(orc):.4f}")

    # test, once: baseline vs chosen config
    test = Split("test", tasks, test_mi_dir=UNION)
    rt = evaluate(ret, test, [BASELINE, c])
    show("test (learned M_i), baseline vs chosen", rt)

    strip = lambda r: {k: v for k, v in r.items() if k != "per_row"}
    out = {"validation": {"copy_learned_mi": copy, "stage1": [strip(r) for r in r1.values()],
                          "stage2": [strip(r) for r in r2.values()], "baseline": base, "best": strip(best),
                          "oracle_best_of_25": float(np.mean(orc))},
           "test": [strip(r) for r in rt.values()]}
    json.dump(out, open(f"{RUN}/final_mask/retrieval_tuning.json", "w"), indent=1)

    if args.write_test_folder:
        for i in range(len(test.ids)):
            nbrs = ret.rank(test, i, c["pool"], c["cand_mi"], c["alpha"], c["k"])
            pred = ret.predict(test, i, nbrs, c["cand_mi"], c["k"], c["agg"], c["shift"])
            d = f"{OUT_DIR}/{i + 1}"
            os.makedirs(d, exist_ok=True)
            for name in ("original_image.png", "edit_instruction.txt", "initial_edit_region_mask.jpg"):
                shutil.copy2(f"{UNION}/{i + 1}/{name}", f"{d}/{name}")
            Image.fromarray(pred.astype(np.uint8) * 255).save(f"{d}/final_edit_region_mask.png")
        print(f"wrote {len(test.ids)} rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
