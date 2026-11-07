"""
Does the fixed Step 2 prompt paragraph wash out the instruction before the transformer sees it? Text encoders only
(frozen base CLIP-L and T5-XXL of FLUX.1-Kontext-dev; the Step 2 LoRA touches only the transformer).

Uses the 96 rows and replacement instructions of the language test (language_test.json: orig vs close = opposite task,
para = other wording of the same task, rand = task from another group). Three ways to encode:

  current     instruction + paragraph to both encoders (what Step 2 was trained with)
  instr_only  instruction alone to both encoders
  split       instruction alone to CLIP (pooled vector), instruction + paragraph to T5

Per pair (orig vs alternative), per encoding:
  clip    1 - cosine between CLIP pooled vectors
  t5      ||E_a - E_b||_F / ||E_a||_F over the full 512-token T5 sequence (padding included, as the pipeline feeds it)
  t5_ins  mean 1 - cosine over the instruction's own T5 tokens (positions 0..n-1 of the original instruction)

Also the share of rows where the opposite instruction moves the embedding more than a paraphrase does (close > para):
a meaning-sensitive encoder should be near 1.

    ~/.conda/envs/sgegen/bin/python run/final_mask/diag_text_embed.py
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FER_DIR = os.path.join(REPO, "sge_goal", "mask_predictor", "inference", "flux_kontext_bridge_calvin")
sys.path.insert(0, FER_DIR)
from local_fer_mask_inference import DEFAULT_INSTRUCTION, INFERENCE_CONFIG  # noqa: E402
from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline  # noqa: E402

RUN = os.path.join(REPO, "run")
ALTS = ["close", "para", "rand"]
MODES = ["current", "instr_only", "split"]


def full(instr):
    return instr.strip() + "\n\n" + DEFAULT_INSTRUCTION.strip()


def texts(instr, mode):
    """(CLIP text, T5 text) for one instruction."""
    if mode == "current":
        return full(instr), full(instr)
    if mode == "instr_only":
        return instr.strip(), instr.strip()
    return instr.strip(), full(instr)


def ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default=f"{RUN}/final_mask/language_test.json")
    p.add_argument("--out", default=f"{RUN}/final_mask/text_embed_test.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    rows = json.load(open(args.rows))["per_row"]
    pipe = FluxKontextPipeline.from_pretrained(INFERENCE_CONFIG["pretrained_model_name_or_path"], transformer=None,
                                               vae=None, torch_dtype=torch.bfloat16)
    pipe.text_encoder.to(args.device)
    pipe.text_encoder_2.to(args.device)
    tok2 = pipe.tokenizer_2

    cache = {}

    def enc(clip_text, t5_text):
        key = (clip_text, t5_text)
        if key not in cache:
            with torch.no_grad():
                e, pooled, _ = pipe.encode_prompt(clip_text, prompt_2=t5_text, device=args.device,
                                                  max_sequence_length=INFERENCE_CONFIG["max_sequence_length"])
            cache[key] = (e[0].float().cpu(), pooled[0].float().cpu())
        return cache[key]

    clip_tokens = [len(pipe.tokenizer(full(r["instr"]))["input_ids"]) for r in rows[:1]]
    per_row = []
    for r in rows:
        n_ins = len(tok2(r["instr"].strip(), add_special_tokens=False)["input_ids"])
        rec = {"id": r["id"], "group": r["group"], "instr": r["instr"], "n_t5_instr_tokens": n_ins}
        for mode in MODES:
            ea, pa = enc(*texts(r["instr"], mode))
            for alt in ALTS:
                eb, pb = enc(*texts(r["alt"][alt], mode))
                rec[f"{mode}_{alt}_clip"] = float(1 - torch.cosine_similarity(pa, pb, dim=0))
                rec[f"{mode}_{alt}_t5"] = float((ea - eb).norm() / ea.norm())
                n = min(n_ins, len(tok2(r["alt"][alt].strip(), add_special_tokens=False)["input_ids"]))
                rec[f"{mode}_{alt}_t5_ins"] = float((1 - torch.cosine_similarity(ea[:n], eb[:n], dim=1)).mean())
        per_row.append(rec)

    summary = {"n_rows": len(rows), "clip_tokens_full_prompt_example": clip_tokens[0], "by_mode": {}}
    for mode in MODES:
        s = {}
        for m in ("clip", "t5", "t5_ins"):
            for alt in ALTS:
                s[f"{alt}_{m}"] = ci([x[f"{mode}_{alt}_{m}"] for x in per_row])
            s[f"close_over_rand_{m}"] = float(np.mean([x[f"{mode}_close_{m}"] for x in per_row]) /
                                              np.mean([x[f"{mode}_rand_{m}"] for x in per_row]))
            s[f"share_close_gt_para_{m}"] = float(np.mean([x[f"{mode}_close_{m}"] > x[f"{mode}_para_{m}"]
                                                           for x in per_row]))
        summary["by_mode"][mode] = s
    by_group = {}
    for g in sorted({x["group"] for x in per_row}):
        xs = [x for x in per_row if x["group"] == g]
        by_group[g] = {f"{mode}_close_clip": float(np.mean([x[f"{mode}_close_clip"] for x in xs])) for mode in MODES}
    summary["close_clip_by_group"] = by_group

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_row": per_row}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
