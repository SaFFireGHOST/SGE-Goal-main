"""
Controls for diag_step2_cause.py, on fresh HF validation tuples (the next rows of the same permutation):

  s10             baseline, 10 steps
  shuf_text       another row's instruction, same seed (repeat on new rows)
  shuf_mi         another row's learned M_i, same I_o, T and seed: does the output follow M_i?
  iou_with_donor  for shuf_mi, overlap with the donor's M_i and with the donor's ground-truth M_f
  s1 / s1_mean4   one step from noise (one seed / mean of 4 seeds), thresholds chosen on the earlier rows
  s1_shuf_text    one step, another row's instruction: is the mean language-blind too?

    python run/final_mask/diag_step2_controls.py --start 128 --n 64 --out run/final_mask/step2_controls.json
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
THRESHOLDS = [64, 96, 127]


def binary(im, thr=127):
    return np.asarray(im.convert("L")) > thr


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=128)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", default=f"{RUN}/final_mask/step2_controls.json")
    p.add_argument("--checkpoint-dir", default=f"{RUN}/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin/step2_final_mask")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    table = pq.read_table(sorted(glob.glob(f"{RUN}/calvin/hf_dataset/data/validation-*.parquet")), columns=COLUMNS)
    pred = pq.read_table(f"{RUN}/calvin/hf_dataset/pred_ier/validation.parquet").to_pydict()
    pred = dict(zip(pred["id"], pred["pred_initial_mask"]))
    order = np.random.default_rng(0).permutation(table.num_rows)[args.start:args.start + args.n]
    rows = []
    for r in order:
        t = table.slice(int(r), 1).to_pylist()[0]
        rows.append({"instr": t["edit_instruction"].strip(),
                     "orig": Image.open(io.BytesIO(t["original_image"]["bytes"])).convert("RGB"),
                     "mi": Image.open(io.BytesIO(pred[t["id"]])).convert("RGB"),
                     "gt": Image.open(io.BytesIO(t["final_edit_region_mask"]["bytes"]))})
    shift = np.roll(np.arange(len(rows)), len(rows) // 3)
    res = {"n": len(rows), "distinct_instructions": len({r["instr"] for r in rows}),
           "shifted_instruction_differs": float(np.mean([rows[i]["instr"] != rows[shift[i]]["instr"]
                                                         for i in range(len(rows))]))}

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

    def run(name, steps=10, seed=0, text=None, mi=None):
        t0, out = time.perf_counter(), []
        for b in range(0, len(rows), args.batch):
            idx = list(range(b, min(b + args.batch, len(rows))))
            who = [rows[text[i]]["instr"] if text is not None else rows[i]["instr"] for i in idx]
            with torch.no_grad(), torch.autocast("cuda"):
                ims = pipe(original_image=[rows[i]["orig"] for i in idx],
                           initial_edit_region_mask=[rows[mi[i]]["mi"] if mi is not None else rows[i]["mi"] for i in idx],
                           prompt_embeds=torch.cat([cache[prompt(s)][0] for s in who]).to(device),
                           pooled_prompt_embeds=torch.cat([cache[prompt(s)][1] for s in who]).to(device),
                           generator=[torch.Generator(device=device).manual_seed(seed + i) for i in idx],
                           height=th, width=tw, guidance_scale=1.0, num_inference_steps=steps,
                           max_sequence_length=INFERENCE_CONFIG["max_sequence_length"],
                           _auto_resize=False, max_area=tw * th).images
            out += [im.resize((w, h), Image.BILINEAR) for im in ims]
        print(f"{name}: {time.perf_counter() - t0:.0f}s", flush=True)
        return out

    gts = [binary(r["gt"]) for r in rows]
    mis = [binary(r["mi"]) for r in rows]

    def m_iou(a, b):
        return float(np.mean([iou(x, y) for x, y in zip(a, b)]))

    res["copy_learned_mi"] = m_iou(mis, gts)
    res["seed_noise_floor"] = None
    base_im = run("s10")
    base = [binary(im) for im in base_im]
    res["s10"] = m_iou(base, gts)
    other = [binary(im) for im in run("s10_seed2", seed=10_000)]
    res["seed_noise_floor"] = m_iou(other, base)

    t = [binary(im) for im in run("shuf_text", text=shift)]
    res["shuf_text"] = {"iou": m_iou(t, gts), "agreement_with_s10": m_iou(t, base)}

    m = [binary(im) for im in run("shuf_mi", mi=shift)]
    res["shuf_mi"] = {"iou": m_iou(m, gts), "agreement_with_s10": m_iou(m, base),
                      "iou_with_donor_mi": m_iou(m, [mis[i] for i in shift]),
                      "iou_with_own_mi": m_iou(m, mis),
                      "iou_with_donor_gt_mf": m_iou(m, [gts[i] for i in shift])}

    s1 = [run(f"s1_seed{k}", steps=1, seed=k * 10_000) for k in range(4)]
    arr = [np.stack([np.asarray(s[i].convert("L"), dtype=np.float32) for s in s1]) for i in range(len(rows))]
    res["s1"] = {str(thr): m_iou([binary(im, thr) for im in s1[0]], gts) for thr in THRESHOLDS}
    res["s1_mean4"] = {str(thr): m_iou([a.mean(0) > thr for a in arr], gts) for thr in THRESHOLDS}
    res["s1_two_seed_agreement_thr64"] = m_iou([binary(im, 64) for im in s1[0]], [binary(im, 64) for im in s1[1]])
    st = run("s1_shuf_text", steps=1, text=shift)
    res["s1_shuf_text"] = {"iou_thr64": m_iou([binary(im, 64) for im in st], gts),
                           "agreement_with_s1_thr64": m_iou([binary(im, 64) for im in st],
                                                            [binary(im, 64) for im in s1[0]])}

    print(json.dumps(res, indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("STEP2_CONTROLS_DONE", flush=True)


if __name__ == "__main__":
    main()
