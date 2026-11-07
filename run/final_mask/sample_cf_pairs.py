"""
Sampled (10-step) Step 2 masks on counterfactual pairs: does the output actually follow the instruction?

For each conditioning row c of a pair (c, o) (both orientations), sample M_f from row c's image and ground-truth M_i
under T_c (output P_c) and under T_o (output P_o), with the same seed for both, so only the instruction differs.
Per case:
  right       IoU(P_c, M_c), IoU(P_o, M_o)          the asked-for mask
  stuck       IoU(P_o, M_c)                          T_o ignored: still draws row c's own mask
  cf_hit      IoU(P_o, M_o) > IoU(P_o, M_c)          P_o is closer to the other row's mask
  own_hit     IoU(P_c, M_c) > IoU(P_c, M_o)
  text_shift  1 - IoU(P_c, P_o)                      how much the instruction moves the output
  seed_shift  1 - IoU(P_c, P_c')                     same instruction, another seed (the floor for text_shift)
Sets: "overfit" = the 16 train pairs of the overfit runs (random.Random(0).sample of the train pairs, as
train_step2_cfdpo.py --n-train-pairs 16), --seeds seeds each; "heldout" = --n-held held-out pairs, one seed.

Guidance (--setting name=G:C, default g1=1:1): G = embedded guidance (FLUX-dev distilled guidance input; 1.0 is what
Step 2 was trained and released with), C = true classifier-free guidance scale with the instruction-free prompt
(empty instruction + the fixed paragraph) as the negative; C > 1 doubles the cost. Results are keyed "ckpt@setting".

    python run/final_mask/sample_cf_pairs.py --ckpt midrop=<dir> --ckpt cfsft=<dir> --out <json>
    python run/final_mask/sample_cf_pairs.py --ckpt midrop=<dir> --setting g35=3.5:1 --setting cfg4=1:4 --out <json>
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
from predict_onestep import RUN, INFERENCE_CONFIG, iou, load_pipe, prompt  # noqa: E402

COLS = ["id", "original_image", "initial_edit_region_mask", "final_edit_region_mask", "edit_instruction"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True, help="name=dir")
    p.add_argument("--pairs", default=f"{RUN}/final_mask/cf_pairs.json")
    p.add_argument("--n-overfit", type=int, default=16)
    p.add_argument("--n-held", type=int, default=32)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--setting", action="append", default=None, help="name=G:C (embedded guidance : true CFG scale)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cf = json.load(open(args.pairs))["pairs"]
    sets = {"overfit": (random.Random(0).sample([q for q in cf if q["split"] == "train"], args.n_overfit), args.seeds),
            "heldout": (random.Random(0).sample([q for q in cf if q["split"] == "heldout"], args.n_held), 1)}
    rows = []
    for f in sorted(glob.glob(f"{RUN}/calvin/hf_dataset/data/train-*.parquet")):
        rows += pq.read_table(f, columns=COLS).to_pylist()
    for q in cf:
        if q["split"] != "val":
            assert rows[q["a"]]["id"] == q["id_a"] and rows[q["b"]]["id"] == q["id_b"]

    def img(r, k, mode):
        return Image.open(io.BytesIO(r[k]["bytes"])).convert(mode)

    def gt(r):
        return np.asarray(img(r, "final_edit_region_mask", "L")) > 127

    # jobs: (set, case index, c, o, which instruction, seed)
    jobs, cases, masks = [], {}, {}
    for s, (pairs, n_seed) in sets.items():
        cases[s] = []
        for q in pairs:
            for c, o in ((q["a"], q["b"]), (q["b"], q["a"])):
                k = len(cases[s])
                masks[(s, k)] = (gt(rows[c]), gt(rows[o]))
                cases[s].append({"c": rows[c]["id"], "o": rows[o]["id"], "instr_c": rows[c]["edit_instruction"].strip(),
                                 "instr_o": rows[o]["edit_instruction"].strip()})
                for seed in range(n_seed):
                    for which in ("c", "o"):
                        jobs.append((s, k, c, o, which, 1000 * k + seed))
    prompts = sorted({prompt(rows[j[2] if j[4] == "c" else j[3]]["edit_instruction"]) for j in jobs} | {prompt("")})
    settings = [(n, *map(float, gc.split(":"))) for n, gc in (x.split("=", 1) for x in (args.setting or ["g1=1:1"]))]
    print(f"{len(jobs)} samples per checkpoint", flush=True)

    res = {"args": vars(args), "summary": {}, "per_case": {}}
    for item in args.ckpt:
      cname, ckpt = item.split("=", 1)
      pipe, cache = load_pipe(ckpt, prompts, args.device)
      neg = cache[prompt("")]
      for sname, g_emb, g_cfg in settings:
        name = cname if len(settings) == 1 and sname == "g1" else f"{cname}@{sname}"
        t0, outs = time.perf_counter(), {}
        for b in range(0, len(jobs), args.batch):
            chunk = jobs[b:b + args.batch]
            orig = [img(rows[j[2]], "original_image", "RGB") for j in chunk]
            w, h = orig[0].size
            tw, th = w // 32 * 32, h // 32 * 32
            pr = [prompt(rows[j[2] if j[4] == "c" else j[3]]["edit_instruction"]) for j in chunk]
            with torch.no_grad(), torch.autocast("cuda"):
                ims = pipe(original_image=orig,
                           initial_edit_region_mask=[img(rows[j[2]], "initial_edit_region_mask", "RGB") for j in chunk],
                           prompt_embeds=torch.cat([cache[x][0] for x in pr]).to(args.device),
                           pooled_prompt_embeds=torch.cat([cache[x][1] for x in pr]).to(args.device),
                           generator=[torch.Generator(device=args.device).manual_seed(j[5]) for j in chunk],
                           negative_prompt_embeds=neg[0].repeat(len(chunk), 1, 1).to(args.device) if g_cfg > 1 else None,
                           negative_pooled_prompt_embeds=neg[1].repeat(len(chunk), 1).to(args.device)
                           if g_cfg > 1 else None, true_cfg_scale=g_cfg,
                           height=th, width=tw, guidance_scale=g_emb, num_inference_steps=10,
                           max_sequence_length=INFERENCE_CONFIG["max_sequence_length"],
                           _auto_resize=False, max_area=tw * th).images
            for im, j in zip(ims, chunk):
                outs[(j[0], j[1], j[4], j[5])] = np.asarray(im.resize((w, h), Image.BILINEAR).convert("L")) > 127
        print(f"{name}: sampled in {time.perf_counter() - t0:.0f}s", flush=True)

        res["summary"][name], res["per_case"][name] = {}, {}
        for s, (pairs, n_seed) in sets.items():
            recs = []
            for k in range(len(cases[s])):
                mc, mo = masks[(s, k)]
                for seed in range(n_seed):
                    sd = 1000 * k + seed
                    pc, po = outs[(s, k, "c", sd)], outs[(s, k, "o", sd)]
                    r = {"right_c": iou(pc, mc), "right_o": iou(po, mo), "stuck": iou(po, mc),
                         "cf_hit": float(iou(po, mo) > iou(po, mc)), "own_hit": float(iou(pc, mc) > iou(pc, mo)),
                         "text_shift": 1 - iou(pc, po), "gt_shift": 1 - iou(mc, mo)}
                    if n_seed > 1:
                        r["seed_shift"] = 1 - iou(pc, outs[(s, k, "c", 1000 * k + (seed + 1) % n_seed)])
                    recs.append(r)
            res["summary"][name][s] = {key: float(np.mean([r[key] for r in recs if key in r]))
                                       for key in recs[0]}
            res["per_case"][name][s] = [{**cases[s][i // n_seed], **r} for i, r in enumerate(recs)]
        print(name, json.dumps(res["summary"][name]), flush=True)
        with open(args.out, "w") as f:
            json.dump(res, f)
      del pipe, cache
      torch.cuda.empty_cache()
    print("SAMPLE_CF_DONE", flush=True)


if __name__ == "__main__":
    main()
