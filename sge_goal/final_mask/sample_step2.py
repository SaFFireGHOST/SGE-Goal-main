"""
Sample the released Step 2 (final-mask) model K times per HF CALVIN train tuple and score every sample by IoU
against the ground-truth final mask: the raw material for IoU-ranked preference pairs (Flow-DPO on Step 2).

Conditions match test time: original image + learned M_i (run/calvin/hf_dataset/pred_ier/train.parquet,
EVF-SAM union; ``--mi gt`` for the ground-truth M_i) + instruction. Sampler as local_fer_mask_inference.py
(10 steps, guidance 1.0, size rounded down to a multiple of 32, bf16, prompts encoded once). Rows x samples are
batched in one pipeline call, each sample with its own generator (seed + K * row + k), so a row's samples do not
depend on the batch layout. Rows are a fixed permutation of the train split (--row-seed), so a short run is a
prefix of a longer one.

Output (--out-dir): rows/<row>/{original.png, m_i.jpg, s<k>.png} (s<k> = raw RGB model output at the original
size, lossless) and samples.jsonl, one line per row: {row, id, instruction, iou: [K], copy_mi_iou, ...}.
Resumable (rows already in samples.jsonl are skipped).

    python sge_goal/final_mask/sample_step2.py --out-dir run/final_mask/step2_samples --num-rows 2000
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
COLUMNS = ["id", "original_image", "initial_edit_region_mask", "final_edit_region_mask", "edit_instruction"]


def binary(im):
    return np.asarray(im.convert("L")) > 127


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def load_rows(args):
    files = sorted(glob.glob(args.train_parquet))
    table = pq.read_table(files, columns=COLUMNS)
    order = np.random.default_rng(args.row_seed).permutation(table.num_rows)[:args.num_rows]
    pred = {}
    if args.mi == "pred":
        p = pq.read_table(args.pred_ier).to_pydict()
        pred = dict(zip(p["id"], p["pred_initial_mask"]))
    rows = []
    for r in order:
        t = table.slice(int(r), 1).to_pylist()[0]
        mi = pred[t["id"]] if args.mi == "pred" else t["initial_edit_region_mask"]["bytes"]
        rows.append({"row": int(r), "id": t["id"], "instruction": t["edit_instruction"].strip(),
                     "original": t["original_image"]["bytes"], "m_i": mi,
                     "m_f": t["final_edit_region_mask"]["bytes"]})
    return rows


def target_size(w, h, resolution=INFERENCE_CONFIG["resolution"], multiple_of=32):
    """As local_fer_mask_inference.py: round down to a multiple of 32, area capped at resolution^2."""
    tw, th = w // multiple_of * multiple_of, h // multiple_of * multiple_of
    if tw * th > resolution ** 2:
        aspect = tw / th
        tw = int((resolution ** 2 * aspect) ** 0.5) // multiple_of * multiple_of
        th = int((resolution ** 2 / aspect) ** 0.5) // multiple_of * multiple_of
    return tw, th


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-rows", type=int, default=2000)
    p.add_argument("--k", type=int, default=4, help="Samples per row.")
    p.add_argument("--batch-rows", type=int, default=8, help="Rows per pipeline call (batch = batch_rows * k).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--row-seed", type=int, default=0)
    p.add_argument("--mi", choices=["pred", "gt"], default="pred")
    p.add_argument("--train-parquet", default=f"{RUN}/calvin/hf_dataset/data/train-*.parquet")
    p.add_argument("--pred-ier", default=f"{RUN}/calvin/hf_dataset/pred_ier/train.parquet")
    p.add_argument("--checkpoint-dir", default=f"{RUN}/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin/step2_final_mask")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.makedirs(f"{args.out_dir}/rows", exist_ok=True)
    out_jsonl = f"{args.out_dir}/samples.jsonl"
    done = set()
    if os.path.exists(out_jsonl):
        done = {json.loads(l)["row"] for l in open(out_jsonl)}
    t0 = time.perf_counter()
    rows = [r for r in load_rows(args) if r["row"] not in done]
    print(f"{len(rows)} rows to sample ({len(done)} done), k={args.k}, M_i={args.mi}; "
          f"data {time.perf_counter() - t0:.0f}s", flush=True)
    if not rows:
        return

    t0 = time.perf_counter()
    pipe = FluxKontextPipeline.from_pretrained(INFERENCE_CONFIG["pretrained_model_name_or_path"],
                                               torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(args.checkpoint_dir, weight_name="pytorch_lora_weights.safetensors")
    pipe.to("cpu")
    pipe.set_progress_bar_config(disable=True)
    prompt = lambda instr: instr + "\n\n" + DEFAULT_INSTRUCTION.strip()
    cache = encode_prompts_once(pipe, [prompt(r["instruction"]) for r in rows], args.device)
    pipe.transformer.to(args.device)
    pipe.vae.to(args.device)
    device = args.device
    pipe.__class__ = type(pipe.__class__.__name__, (pipe.__class__,),
                          {"_execution_device": property(lambda self: torch.device(device))})
    print(f"model + {len(cache)} prompts ready in {time.perf_counter() - t0:.0f}s", flush=True)

    t_start, n_samples, all_ious, all_best = time.perf_counter(), 0, [], []
    with open(out_jsonl, "a") as fout:
        for b in range(0, len(rows), args.batch_rows):
            batch = rows[b:b + args.batch_rows]
            origs, mis, pes, pps, gens = [], [], [], [], []
            for r in batch:
                r["orig_im"] = Image.open(io.BytesIO(r["original"])).convert("RGB")
                r["mi_im"] = Image.open(io.BytesIO(r["m_i"])).convert("RGB")
                pe, pp = cache[prompt(r["instruction"])]
                for k in range(args.k):
                    origs.append(r["orig_im"])
                    mis.append(r["mi_im"])
                    pes.append(pe)
                    pps.append(pp)
                    gens.append(torch.Generator(device=device).manual_seed(args.seed + args.k * r["row"] + k))
            w, h = batch[0]["orig_im"].size
            tw, th = target_size(w, h)
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda"):
                images = pipe(original_image=origs, initial_edit_region_mask=mis,
                              prompt_embeds=torch.cat(pes).to(device), pooled_prompt_embeds=torch.cat(pps).to(device),
                              generator=gens, height=th, width=tw,
                              guidance_scale=INFERENCE_CONFIG["guidance_scale"],
                              num_inference_steps=INFERENCE_CONFIG["num_inference_steps"],
                              max_sequence_length=INFERENCE_CONFIG["max_sequence_length"],
                              _auto_resize=False, max_area=tw * th).images
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            for i, r in enumerate(batch):
                d = f"{args.out_dir}/rows/{r['row']}"
                os.makedirs(d, exist_ok=True)
                open(f"{d}/original.png", "wb").write(r["original"])
                open(f"{d}/m_i.jpg" if args.mi == "pred" else f"{d}/m_i.png", "wb").write(r["m_i"])
                gt = binary(Image.open(io.BytesIO(r["m_f"])))
                ious = []
                for k in range(args.k):
                    im = images[i * args.k + k]
                    if im.size != (w, h):
                        im = im.resize((w, h), Image.BILINEAR)
                    im.save(f"{d}/s{k}.png")
                    ious.append(iou(binary(im), gt))
                all_ious += ious
                all_best.append(max(ious))
                fout.write(json.dumps({"row": r["row"], "id": r["id"], "instruction": r["instruction"],
                                       "iou": ious, "copy_mi_iou": iou(binary(r["mi_im"]), gt),
                                       "gt_empty": bool(not gt.any())}) + "\n")
            fout.flush()
            n_samples += len(images)
            print(f"batch {b // args.batch_rows}: {len(images)} samples in {dt:.1f}s ({dt / len(images):.2f} s/sample); "
                  f"{n_samples} total, {(time.perf_counter() - t_start) / n_samples:.2f} s/sample overall; "
                  f"running mean IoU {np.mean(all_ious):.3f}, best-of-{args.k} {np.mean(all_best):.3f}", flush=True)
    print(f"SAMPLING_DONE {n_samples} samples in {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
