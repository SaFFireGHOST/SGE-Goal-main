"""
Counterfactual pairs for Step 2 Flow-DPO (plan_final_mask.md §5.1): two HF rows A, B with nearly the same scene, the
same manipulated object and different CALVIN tasks. Training conditions both directions on row A's image and M_i and
prefers M_A under T_A, M_B under T_B.

Scene distance: mean absolute difference of 32x32 RGB images (0-255), as in §3.6. Object: the block colour for the
coloured-block tasks (push / rotate / lift), else drawer / slider / led / lightbulb. Tasks whose object is not named
(stack, unstack, place_in_*, push_into_drawer) are left out.

Splits. The HF split is not by video (every validation and test episode also appears in train), so the released model
has seen every video and "held out" can only mean held out from our fine-tuning:
  train          pairs inside HF train, both rows from training episodes, L1 < --train-l1
  heldout        pairs inside HF train, both rows from --heldout-frac of episodes (by hash), L1 < --heldout-l1.
                 These episodes should also be left out of the fine-tuning's ordinary rows (listed in the output).
  val            pairs inside HF validation (rows no model has trained on), L1 < --heldout-l1
Pairs that cross the train / held-out episode boundary are dropped. Each row keeps at most --per-row nearest partners.
Held-out at L1 < 3 gives too few pairs, so held-out and val use L1 < 6 by default; summaries also report L1 < 3.

Articulated objects (LED, lightbulb, slider; drawer has no pairs) only give opposite-direction pairs whose scenes differ in
exactly that state (LED on vs off), so the other row's instruction is not a valid one for row A's image. They are left out
unless --articulated.

Same place: a 4-6 L1 match is often a different block layout (the block on the shelf in one image, on the table in the
other), so B's mask is wrong for A's image and the image alone rejects it. Blocks are flat pure colours, so each block is
segmented by colour on the 208x208 image and a pair needs IoU >= --block-iou between A's and B's block (both visible,
>= 30 px). Blocks hidden in a drawer or behind the arm drop out. Label check: M_i is arm + objects that will move, so each
row's ground-truth M_i must cover >= --mi-cover of its block; rows where the authors' pipeline missed the block drop out.

    ~/.conda/envs/sgegen/bin/python run/final_mask/build_cf_pairs.py
"""

import argparse
import glob
import hashlib
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_retrieval import episode, task_map  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RUN = os.path.join(REPO, "run")
DATA = f"{RUN}/calvin/hf_dataset/data"
COLS = ["id", "original_image", "initial_edit_region_mask", "final_edit_region_mask", "edit_instruction"]
SIDE = 32


def obj(task):
    for c in ("red", "blue", "pink"):
        if f"_{c}_block" in task:
            return c
    for o in ("drawer", "slider", "led", "lightbulb"):
        if task in (f"open_{o}", f"close_{o}", f"move_{o}_left", f"move_{o}_right", f"turn_on_{o}", f"turn_off_{o}"):
            return o
    return None


def load(split):
    rows = []
    for f in sorted(glob.glob(f"{DATA}/{split}-*.parquet")):
        rows += pq.read_table(f, columns=COLS).to_pylist()
    return rows


def small(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((SIDE, SIDE), Image.BILINEAR), np.float32).ravel()


def mask(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("L")) > 127


def block(b, colour):
    a = np.asarray(Image.open(io.BytesIO(b)).convert("RGB")).astype(np.int16)
    r, g, bl = a[..., 0], a[..., 1], a[..., 2]
    if colour == "red":
        return (r > 140) & (g < 70) & (bl < 70)
    if colour == "blue":
        return (bl > 110) & (r < 70) & (g < 70)
    return (r > 150) & (bl > 150) & (g < 90)


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def l1_matrix(x, device):
    x = torch.from_numpy(x).to(device)
    out = torch.empty(len(x), len(x))
    for s in range(0, len(x), 512):
        out[s:s + 512] = torch.cdist(x[s:s + 512], x, p=1).cpu() / x.shape[1]
    return out.numpy()


def heldout_episode(ep, frac):
    return int(hashlib.md5(ep.encode()).hexdigest(), 16) % 1000 < frac * 1000


def mine(rows, tasks, d, thr, per_row, keep_row, articulated=False, block_iou=0.5, mi_cover=0.3):
    """Unordered pairs (i, j), same object, different task, L1 < thr, block in the same place; each row keeps its
    per_row nearest partners."""
    t = [tasks.get(r["edit_instruction"].strip().lower()) for r in rows]
    o = [obj(x) if x else None for x in t]
    bm = {}

    def bmask(i):
        if i not in bm:
            bm[i] = block(rows[i]["original_image"]["bytes"], o[i]) if o[i] in ("red", "blue", "pink") else None
        return bm[i]

    cover = {}

    def covered(i):
        if i not in cover:
            m, bk = mask(rows[i]["initial_edit_region_mask"]["bytes"]), bmask(i)
            cover[i] = (m & bk).sum() / max(bk.sum(), 1) >= mi_cover
        return cover[i]

    def same_place(i, j):
        a, b = bmask(i), bmask(j)
        if a is None:
            return True
        return a.sum() >= 30 and b.sum() >= 30 and iou(a, b) >= block_iou and covered(i) and covered(j)

    pairs = set()
    for i in range(len(rows)):
        if o[i] is None or not keep_row(i) or (not articulated and o[i] not in ("red", "blue", "pink")):
            continue
        near = np.nonzero(d[i] < thr)[0]
        near = near[np.argsort(d[i, near])]
        cand = [j for j in near if j != i and o[j] == o[i] and t[j] != t[i] and keep_row(j) and same_place(i, j)]
        for j in cand[:per_row]:
            pairs.add((int(min(i, j)), int(max(i, j))))
    return sorted(pairs), t, o, bmask


def describe(rows, pairs, t, o, d, split, bmask):
    out = []
    for i, j in pairs:
        a, b = rows[i], rows[j]
        out.append({"split": split, "a": i, "b": j, "id_a": a["id"], "id_b": b["id"],
                    "instr_a": a["edit_instruction"].strip(), "instr_b": b["edit_instruction"].strip(),
                    "task_a": t[i], "task_b": t[j], "object": o[i], "scene_l1": float(d[i, j]),
                    "same_video": episode(a["id"]) == episode(b["id"]),
                    "block_iou": iou(bmask(i), bmask(j)) if bmask(i) is not None else None,
                    "mf_iou": iou(mask(a["final_edit_region_mask"]["bytes"]), mask(b["final_edit_region_mask"]["bytes"])),
                    "mi_iou": iou(mask(a["initial_edit_region_mask"]["bytes"]), mask(b["initial_edit_region_mask"]["bytes"]))})
    return out


def summarise(pairs):
    s = {"n": len(pairs), "rows": len({p["id_a"] for p in pairs} | {p["id_b"] for p in pairs}),
         "n_l1_lt_3": sum(p["scene_l1"] < 3 for p in pairs)}
    if pairs:
        s["mean_mf_iou"] = float(np.mean([p["mf_iou"] for p in pairs]))
        s["mean_mi_iou"] = float(np.mean([p["mi_iou"] for p in pairs]))
        s["same_video"] = float(np.mean([p["same_video"] for p in pairs]))
        s["by_object"] = {}
        for p in pairs:
            s["by_object"][p["object"]] = s["by_object"].get(p["object"], 0) + 1
        kinds = {}
        for p in pairs:
            k = " vs ".join(sorted((p["task_a"].replace(f"_{p['object']}", ""), p["task_b"].replace(f"_{p['object']}", ""))))
            kinds[k] = kinds.get(k, 0) + 1
        s["by_task_pair"] = dict(sorted(kinds.items(), key=lambda kv: -kv[1]))
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-l1", type=float, default=6.0)
    p.add_argument("--heldout-l1", type=float, default=6.0)
    p.add_argument("--heldout-frac", type=float, default=0.25)
    p.add_argument("--per-row", type=int, default=3)
    p.add_argument("--articulated", action="store_true", help="Keep LED / lightbulb / slider pairs.")
    p.add_argument("--block-iou", type=float, default=0.5)
    p.add_argument("--mi-cover", type=float, default=0.3)
    p.add_argument("--out", default=f"{RUN}/final_mask/cf_pairs.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    tasks = task_map()

    train = load("train")
    x = np.stack([small(r["original_image"]["bytes"]) for r in train])
    d = l1_matrix(x, args.device)
    np.fill_diagonal(d, np.inf)
    t = [tasks.get(r["edit_instruction"].strip().lower()) for r in train]
    known = np.array([x is not None for x in t])
    calib = {}
    tt = np.array([x or "" for x in t])
    oo = np.array([obj(x) if x else "" for x in t])
    for thr in (1.5, 3.0, 6.0):
        any_diff = same_obj = same_task = 0
        for i in np.nonzero(known)[0]:
            near = np.nonzero(d[i] < thr)[0]
            near = near[known[near]]
            dt = near[tt[near] != tt[i]]
            any_diff += len(dt) > 0
            same_obj += oo[i] != "" and bool((oo[dt] == oo[i]).any())
            same_task += bool((tt[near] == tt[i]).any())
        calib[thr] = {"rows_with_diff_task_match": int(any_diff), "rows_with_same_object_diff_task_match": int(same_obj),
                      "rows_with_same_task_match": int(same_task)}
    print("calibration (plan §3.6: 95 / 920 / 14,139 diff-task rows; 51 / 325 / 5,438 same-object):",
          json.dumps(calib, indent=1), flush=True)

    held_eps = sorted({episode(r["id"]) for r in train if heldout_episode(episode(r["id"]), args.heldout_frac)})
    is_held = np.array([episode(r["id"]) in set(held_eps) for r in train])
    tr_pairs, t, o, bm = mine(train, tasks, d, args.train_l1, args.per_row, lambda i: not is_held[i], args.articulated,
                              args.block_iou, args.mi_cover)
    ho_pairs, _, _, _ = mine(train, tasks, d, args.heldout_l1, args.per_row, lambda i: is_held[i], args.articulated,
                             args.block_iou, args.mi_cover)
    pairs = describe(train, tr_pairs, t, o, d, "train", bm) + describe(train, ho_pairs, t, o, d, "heldout", bm)

    val = load("validation")
    xv = np.stack([small(r["original_image"]["bytes"]) for r in val])
    dv = l1_matrix(xv, args.device)
    np.fill_diagonal(dv, np.inf)
    va_pairs, tv, ov, bv = mine(val, tasks, dv, args.heldout_l1, args.per_row, lambda i: True, args.articulated,
                                args.block_iou, args.mi_cover)
    pairs += describe(val, va_pairs, tv, ov, dv, "val", bv)

    summary = {"args": vars(args), "calibration": calib, "heldout_episodes": held_eps,
               "heldout_rows": int(is_held.sum()),
               "splits": {s: summarise([q for q in pairs if q["split"] == s]) for s in ("train", "heldout", "val")}}
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "pairs": pairs}, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "heldout_episodes"}, indent=1))


if __name__ == "__main__":
    main()
