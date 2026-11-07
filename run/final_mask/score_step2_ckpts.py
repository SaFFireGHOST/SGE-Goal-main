"""
Step 2 checkpoints scored on the HF validation tuples NOT used to select them: the complement of the trainer's 256
val_mask_iou rows (TupleDataset max_samples=256, seed=1). Learned M_i, 10 steps, same seed per row for every
checkpoint, so the comparison is paired. 95% CI by bootstrap over rows. Per-row IoU is saved.

    python run/final_mask/score_step2_ckpts.py --out run/final_mask/ioudpo_val/heldout_iou.json \
        --ckpt released=<dir> --ckpt ioudpo=<dir>
"""

import argparse
import glob
import io
import json
import os
import random
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_onestep import RUN, COLUMNS, INFERENCE_CONFIG, iou, load_pipe, prompt  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", action="append", required=True, help="name=dir; the first (or --ref) is the reference.")
    p.add_argument("--ref", default=None, help="name=json: reference per-row IoU from an earlier run of this script.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    t = pq.read_table(sorted(glob.glob(f"{RUN}/calvin/hf_dataset/data/validation-*.parquet")), columns=COLUMNS).to_pylist()
    used = set(random.Random(1).sample(range(len(t)), 256))
    rows = [t[i] for i in range(len(t)) if i not in used]
    pred = pq.read_table(f"{RUN}/calvin/hf_dataset/pred_ier/validation.parquet").to_pydict()
    pred = dict(zip(pred["id"], pred["pred_initial_mask"]))
    print(f"{len(rows)} held-out rows", flush=True)

    scores = {}
    if args.ref:
        name, path = args.ref.split("=", 1)
        with open(path) as f:
            scores[name] = json.load(f)["per_row"][name]
    for item in args.ckpt:
        name, ckpt = item.split("=", 1)
        pipe, cache = load_pipe(ckpt, [prompt(r["edit_instruction"]) for r in rows], args.device)
        t0, out = time.perf_counter(), []
        for b in range(0, len(rows), args.batch):
            chunk = rows[b:b + args.batch]
            orig = [Image.open(io.BytesIO(r["original_image"]["bytes"])).convert("RGB") for r in chunk]
            w, h = orig[0].size
            tw, th = w // 32 * 32, h // 32 * 32
            with torch.no_grad(), torch.autocast("cuda"):
                ims = pipe(original_image=orig,
                           initial_edit_region_mask=[Image.open(io.BytesIO(pred[r["id"]])).convert("RGB") for r in chunk],
                           prompt_embeds=torch.cat([cache[prompt(r["edit_instruction"])][0] for r in chunk]).to(args.device),
                           pooled_prompt_embeds=torch.cat([cache[prompt(r["edit_instruction"])][1] for r in chunk]).to(args.device),
                           generator=[torch.Generator(device=args.device).manual_seed(args.seed + b + i)
                                      for i in range(len(chunk))],
                           height=th, width=tw, guidance_scale=1.0, num_inference_steps=10,
                           max_sequence_length=INFERENCE_CONFIG["max_sequence_length"],
                           _auto_resize=False, max_area=tw * th).images
            for im, r in zip(ims, chunk):
                gt = np.asarray(Image.open(io.BytesIO(r["final_edit_region_mask"]["bytes"])).convert("L")) > 127
                out.append(iou(np.asarray(im.resize((w, h), Image.BILINEAR).convert("L")) > 127, gt))
        scores[name] = out
        print(f"{name}: IoU {np.mean(out):.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        del pipe, cache
        torch.cuda.empty_cache()

    names = list(scores)
    res = {"n": len(rows), "iou": {k: float(np.mean(v)) for k, v in scores.items()}, "per_row": scores}
    boot = np.random.default_rng(0).integers(0, len(rows), size=(10000, len(rows)))
    for k in names[1:]:
        d = np.array(scores[k]) - np.array(scores[names[0]])
        bs = d[boot].mean(1)
        res[f"{k}_minus_{names[0]}"] = {"diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                         float(np.percentile(bs, 97.5))]}
        print(k, res[f"{k}_minus_{names[0]}"], flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print("HELDOUT_IOU_DONE", flush=True)


if __name__ == "__main__":
    main()
