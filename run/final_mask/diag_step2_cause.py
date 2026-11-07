"""
Why is the released Step 2 (FLUX-Kontext LoRA) M_f IoU low? Separates the candidate causes on HF validation tuples
(learned M_i, as at test time; sampler as sample_step2.py):

  vae_roundtrip   ground-truth M_f through the VAE and back: the ceiling the representation allows
  s10 / s10b      10 steps, two seeds: baseline IoU and seed-to-seed agreement
  s1              1 step from pure noise = the model's estimate of the conditional mean mask
  s20             20 steps: is 10 too few?
  shuf_text       another row's instruction: does the model use T?
  zero_mi         black M_i: does the model use M_i?
  shuf_image      another row's I_o (M_i kept): does the model use I_o?

    python run/final_mask/diag_step2_cause.py --n 128 --out run/final_mask/step2_cause.json
"""

import argparse
import glob
import io
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FER_DIR = os.path.join(REPO, "sge_goal", "mask_predictor", "inference", "flux_kontext_bridge_calvin")
sys.path.insert(0, FER_DIR)
from local_fer_mask_inference import DEFAULT_INSTRUCTION, INFERENCE_CONFIG, encode_prompts_once  # noqa: E402
from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline  # noqa: E402

RUN = os.path.join(REPO, "run")
COLUMNS = ["id", "original_image", "final_edit_region_mask", "edit_instruction"]
THRESHOLDS = [64, 96, 127, 160]


def binary(im, thr=127):
    return np.asarray(im.convert("L")) > thr


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", default=f"{RUN}/final_mask/step2_cause.json")
    p.add_argument("--checkpoint-dir", default=f"{RUN}/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin/step2_final_mask")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    table = pq.read_table(sorted(glob.glob(f"{RUN}/calvin/hf_dataset/data/validation-*.parquet")), columns=COLUMNS)
    pred = pq.read_table(f"{RUN}/calvin/hf_dataset/pred_ier/validation.parquet").to_pydict()
    pred = dict(zip(pred["id"], pred["pred_initial_mask"]))
    order = np.random.default_rng(0).permutation(table.num_rows)[:args.n]
    rows = []
    for r in order:
        t = table.slice(int(r), 1).to_pylist()[0]
        rows.append({"instr": t["edit_instruction"].strip(),
                     "orig": Image.open(io.BytesIO(t["original_image"]["bytes"])).convert("RGB"),
                     "mi": Image.open(io.BytesIO(pred[t["id"]])).convert("RGB"),
                     "gt": Image.open(io.BytesIO(t["final_edit_region_mask"]["bytes"]))})
    # A fixed derangement: every row borrows another row's instruction / image.
    shift = np.roll(np.arange(len(rows)), len(rows) // 3)

    pipe = FluxKontextPipeline.from_pretrained(INFERENCE_CONFIG["pretrained_model_name_or_path"],
                                               torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(args.checkpoint_dir, weight_name="pytorch_lora_weights.safetensors")
    pipe.to("cpu")
    pipe.set_progress_bar_config(disable=True)
    prompt = lambda instr: instr + "\n\n" + DEFAULT_INSTRUCTION.strip()
    cache = encode_prompts_once(pipe, [prompt(r["instr"]) for r in rows], args.device)
    pipe.transformer.to(args.device)
    pipe.vae.to(args.device)
    device = args.device
    pipe.__class__ = type(pipe.__class__.__name__, (pipe.__class__,),
                          {"_execution_device": property(lambda self: torch.device(device))})

    w, h = rows[0]["orig"].size
    tw, th = w // 32 * 32, h // 32 * 32
    black = Image.new("RGB", (w, h))

    def run(name, steps=10, seed=0, text=None, image=None, mi=None):
        t0, out = time.perf_counter(), []
        for b in range(0, len(rows), args.batch):
            idx = list(range(b, min(b + args.batch, len(rows))))
            pe = torch.cat([cache[prompt(rows[text[i]]["instr"] if text is not None else rows[i]["instr"])][0] for i in idx])
            pp = torch.cat([cache[prompt(rows[text[i]]["instr"] if text is not None else rows[i]["instr"])][1] for i in idx])
            with torch.no_grad(), torch.autocast("cuda"):
                ims = pipe(original_image=[rows[image[i]]["orig"] if image is not None else rows[i]["orig"] for i in idx],
                           initial_edit_region_mask=[black if mi == "zero" else rows[i]["mi"] for i in idx],
                           prompt_embeds=pe.to(device), pooled_prompt_embeds=pp.to(device),
                           generator=[torch.Generator(device=device).manual_seed(seed + i) for i in idx],
                           height=th, width=tw, guidance_scale=1.0, num_inference_steps=steps,
                           max_sequence_length=INFERENCE_CONFIG["max_sequence_length"],
                           _auto_resize=False, max_area=tw * th).images
            out += [im.resize((w, h), Image.BILINEAR) for im in ims]
        print(f"{name}: {time.perf_counter() - t0:.0f}s", flush=True)
        return out

    res = {}
    gts = [binary(r["gt"]) for r in rows]
    mis = [binary(r["mi"]) for r in rows]

    def score(name, ims, ref=None):
        masks = [binary(im) for im in ims]
        s = {"iou": float(np.mean([iou(m, g) for m, g in zip(masks, gts)])),
             "area": float(np.mean([m.mean() for m in masks])),
             "iou_with_input_mi": float(np.mean([iou(m, i) for m, i in zip(masks, mis)])),
             "gray_frac": float(np.mean([((np.asarray(im.convert("L")) > 32) & (np.asarray(im.convert("L")) < 223)).mean()
                                         for im in ims]))}
        if ref is not None:
            s["agreement_with_s10"] = float(np.mean([iou(m, r) for m, r in zip(masks, ref)]))
        res[name] = s
        print(name, json.dumps(s), flush=True)
        return masks

    res["gt_area"] = float(np.mean([g.mean() for g in gts]))
    res["copy_learned_mi_iou"] = float(np.mean([iou(i, g) for i, g in zip(mis, gts)]))

    # VAE ceiling
    vae_iou = []
    with torch.no_grad():
        for r, g in zip(rows, gts):
            x = torch.from_numpy(np.asarray(r["gt"].convert("RGB").resize((tw, th), Image.BILINEAR)).copy()).float()
            x = (x / 127.5 - 1).permute(2, 0, 1)[None].to(device, torch.bfloat16)
            rec = pipe.vae.decode(pipe.vae.encode(x).latent_dist.mode()).sample[0].float().mean(0).cpu().numpy()
            rec = Image.fromarray(((rec + 1) * 127.5).clip(0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
            vae_iou.append(iou(binary(rec), g))
    res["vae_roundtrip_iou"] = float(np.mean(vae_iou))
    print("vae_roundtrip_iou", res["vae_roundtrip_iou"], flush=True)

    base = score("s10", run("s10"))
    score("s10b", run("s10b", seed=10_000), ref=base)
    s1 = run("s1", steps=1)
    score("s1", s1, ref=base)
    res["s1_iou_by_threshold"] = {str(t): float(np.mean([iou(binary(im, t), g) for im, g in zip(s1, gts)]))
                                  for t in THRESHOLDS}
    print("s1_iou_by_threshold", res["s1_iou_by_threshold"], flush=True)
    score("s20", run("s20", steps=20), ref=base)
    score("shuf_text", run("shuf_text", text=shift), ref=base)
    score("zero_mi", run("zero_mi", mi="zero"), ref=base)
    score("shuf_image", run("shuf_image", image=shift), ref=base)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("STEP2_CAUSE_DONE", flush=True)


if __name__ == "__main__":
    main()
