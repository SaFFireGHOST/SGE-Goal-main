"""
One-step final masks from the released Step 2: a single denoising step from pure noise (the model's estimate of the
conditional mean), binarised at --threshold. No training. Threshold 64 was chosen on 128 validation rows and
confirmed on 64 fresh ones (diag_step2_cause.py, diag_step2_controls.py).

  --validation OUT.json   IoU on all HF validation tuples (learned M_i) at several thresholds
  --data-dir DIR          CALVIN-EVALS-style cases: writes final_edit_region_mask.png (binary), seed + case index as
                          local_fer_mask_inference.py. Skips cases that already have one.

Sampler otherwise as local_fer_mask_inference.py (size rounded down to a multiple of 32, guidance 1.0, bf16).
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
from local_fer_mask_inference import DEFAULT_INSTRUCTION, INFERENCE_CONFIG, encode_prompts_once, find_image  # noqa: E402
from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline  # noqa: E402

RUN = os.path.join(REPO, "run")
COLUMNS = ["id", "original_image", "final_edit_region_mask", "edit_instruction"]


def prompt(instr):
    return instr.strip() + "\n\n" + DEFAULT_INSTRUCTION.strip()


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def load_pipe(checkpoint_dir, prompts, device):
    pipe = FluxKontextPipeline.from_pretrained(INFERENCE_CONFIG["pretrained_model_name_or_path"],
                                               torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(checkpoint_dir, weight_name="pytorch_lora_weights.safetensors")
    pipe.to("cpu")
    pipe.set_progress_bar_config(disable=True)
    cache = encode_prompts_once(pipe, prompts, device)
    pipe.transformer.to(device)
    pipe.vae.to(device)
    pipe.__class__ = type(pipe.__class__.__name__, (pipe.__class__,),
                          {"_execution_device": property(lambda self: torch.device(device))})
    return pipe, cache


def one_step(pipe, cache, batch, device):
    """batch: list of (orig RGB, M_i RGB, instruction, seed). Returns 'L' images at each input's size."""
    w, h = batch[0][0].size
    tw, th = w // 32 * 32, h // 32 * 32
    with torch.no_grad(), torch.autocast("cuda"):
        ims = pipe(original_image=[b[0] for b in batch], initial_edit_region_mask=[b[1] for b in batch],
                   prompt_embeds=torch.cat([cache[prompt(b[2])][0] for b in batch]).to(device),
                   pooled_prompt_embeds=torch.cat([cache[prompt(b[2])][1] for b in batch]).to(device),
                   generator=[torch.Generator(device=device).manual_seed(b[3]) for b in batch],
                   height=th, width=tw, guidance_scale=INFERENCE_CONFIG["guidance_scale"], num_inference_steps=1,
                   max_sequence_length=INFERENCE_CONFIG["max_sequence_length"],
                   _auto_resize=False, max_area=tw * th).images
    return [im.resize((w, h), Image.BILINEAR).convert("L") for im in ims]


def validation(args):
    t = pq.read_table(sorted(glob.glob(f"{RUN}/calvin/hf_dataset/data/validation-*.parquet")), columns=COLUMNS).to_pylist()
    pred = pq.read_table(f"{RUN}/calvin/hf_dataset/pred_ier/validation.parquet").to_pydict()
    pred = dict(zip(pred["id"], pred["pred_initial_mask"]))
    pipe, cache = load_pipe(args.checkpoint_dir, [prompt(r["edit_instruction"]) for r in t], args.device)
    thresholds = [48, 64, 80, 96, 127]
    scores = {thr: [] for thr in thresholds}
    t0 = time.perf_counter()
    for b in range(0, len(t), args.batch):
        rows = t[b:b + args.batch]
        batch = [(Image.open(io.BytesIO(r["original_image"]["bytes"])).convert("RGB"),
                  Image.open(io.BytesIO(pred[r["id"]])).convert("RGB"), r["edit_instruction"], args.seed + b + i)
                 for i, r in enumerate(rows)]
        for im, r in zip(one_step(pipe, cache, batch, args.device), rows):
            gt = np.asarray(Image.open(io.BytesIO(r["final_edit_region_mask"]["bytes"])).convert("L")) > 127
            arr = np.asarray(im)
            for thr in thresholds:
                scores[thr].append(iou(arr > thr, gt))
    res = {"n": len(t), "seconds": time.perf_counter() - t0, "iou": {str(k): float(np.mean(v)) for k, v in scores.items()},
           "frac_ge_0.5": {str(k): float(np.mean(np.array(v) >= 0.5)) for k, v in scores.items()}}
    print(json.dumps(res, indent=1), flush=True)
    with open(args.validation, "w") as f:
        json.dump(res, f, indent=1)


def folder(args):
    cases = sorted((d for d in os.listdir(args.data_dir) if d.isdigit()), key=int)
    todo = []
    for idx, c in enumerate(cases):
        d = os.path.join(args.data_dir, c)
        if not os.path.exists(f"{d}/final_edit_region_mask.png"):
            todo.append((idx, d, open(f"{d}/edit_instruction.txt", encoding="utf-8").read().strip()))
    print(f"{len(todo)} of {len(cases)} cases to do", flush=True)
    if not todo:
        return
    pipe, cache = load_pipe(args.checkpoint_dir, [prompt(x[2]) for x in todo], args.device)
    for b in range(0, len(todo), args.batch):
        rows = todo[b:b + args.batch]
        batch = [(Image.open(find_image(d, "original_image")).convert("RGB"),
                  Image.open(find_image(d, "initial_edit_region_mask")).convert("RGB"), instr, args.seed + idx)
                 for idx, d, instr in rows]
        for im, (_, d, _) in zip(one_step(pipe, cache, batch, args.device), rows):
            Image.fromarray(((np.asarray(im) > args.threshold) * 255).astype(np.uint8)).convert("RGB").save(
                f"{d}/final_edit_region_mask.png")
    print("ONESTEP_FOLDER_DONE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validation", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--threshold", type=int, default=64)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--checkpoint-dir", default=f"{RUN}/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin/step2_final_mask")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.validation:
        validation(args)
    if args.data_dir:
        folder(args)


if __name__ == "__main__":
    main()
