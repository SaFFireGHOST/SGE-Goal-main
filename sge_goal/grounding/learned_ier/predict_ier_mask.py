"""
Predict initial edit region masks with a trained learned-IER checkpoint.

Reads a CALVIN-EVALS style folder (``<data-dir>/<N>/original_image.* +
edit_instruction.txt``) and writes a new folder with the same layout plus
``initial_edit_region_mask.jpg``, ready for the local Step 2 / Step 3 scripts.
The input folder is never modified.

Usage (via run/predict_ier.sh):
    bash run/predict_ier.sh --checkpoint run/ier_training/evf_union/best.pt \
        --data-dir run/CALVIN-EVALS/sge_goal/calvin_abcd_test_data \
        --out-dir run/CALVIN-EVALS/sge_goal_learned_ier/calvin_abcd_test_data \
        --gt-parquet "run/calvin/hf_dataset/data/test-*.parquet"
"""

import argparse
import glob
import io
import json
import os
import shutil
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ier_model import DEFAULT_EVF_ROOT, LearnedIerModel, mask_metrics, union_prob  # noqa: E402

OUTPUT_FILENAME = "initial_edit_region_mask.jpg"


def find_image(directory, base_name):
    for ext in ("jpg", "jpeg", "png", "bmp", "webp"):
        path = os.path.join(directory, f"{base_name}.{ext}")
        if os.path.isfile(path):
            return path
    return None


def main():
    p = argparse.ArgumentParser(description="Predict M_i with a learned-IER checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--evf-root", default=DEFAULT_EVF_ROOT)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--gt-parquet", default="",
                   help="HF parquet whose row i is case folder i+1; reports IoU vs its initial masks.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    data_dir, out_dir = os.path.abspath(args.data_dir), os.path.abspath(args.out_dir)
    if data_dir == out_dir:
        raise SystemExit("--out-dir must differ from --data-dir")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(ckpt["evf_version"], padding_side="right", use_fast=False)
    model = LearnedIerModel.from_pretrained(ckpt["evf_version"], args.evf_root,
                                            heads=ckpt["heads"], arm_prompt=ckpt["arm_prompt"])
    model.load_trainable_state_dict(ckpt["trainable"])
    model.sam.image_encoder.to(torch.bfloat16)
    model.to(args.device).eval()

    cases = sorted((d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))),
                   key=lambda x: int(x) if x.isdigit() else x)
    pending = []
    for case in cases:
        src, dst = os.path.join(data_dir, case), os.path.join(out_dir, case)
        if args.skip_existing and os.path.isfile(os.path.join(dst, OUTPUT_FILENAME)):
            continue
        image_path, instr_path = find_image(src, "original_image"), os.path.join(src, "edit_instruction.txt")
        if image_path is None or not os.path.isfile(instr_path):
            print(f"[{case}] missing original_image or edit_instruction.txt, skipping")
            continue
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(image_path, dst)
        shutil.copy2(instr_path, dst)
        pending.append((case, image_path, instr_path))
    print(f"{len(cases)} cases, {len(pending)} to predict.")

    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start:start + args.batch_size]
        images = [np.asarray(Image.open(path).convert("RGB")) for _, path, _ in chunk]
        instructions = [open(path, encoding="utf-8").read().strip() for _, _, path in chunk]
        sizes = {im.shape for im in images}
        batches = [list(range(len(chunk)))] if len(sizes) == 1 else [[i] for i in range(len(chunk))]
        for idxs in batches:
            x = torch.from_numpy(np.stack([images[i] for i in idxs])).permute(0, 3, 1, 2).to(args.device)
            tok = tokenizer(model.prompt_texts([instructions[i] for i in idxs]), padding=True,
                            truncation=True, max_length=64, return_tensors="pt")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, tok["input_ids"].to(args.device), tok["attention_mask"].to(args.device))
            masks = (union_prob(logits.float()) > args.threshold).cpu().numpy()
            for i, mask in zip(idxs, masks):
                case = chunk[i][0]
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
                    os.path.join(out_dir, case, OUTPUT_FILENAME), quality=95)
        print(f"predicted {min(start + args.batch_size, len(pending))}/{len(pending)}", flush=True)

    if args.gt_parquet:
        path = sorted(glob.glob(os.path.expanduser(args.gt_parquet)))[0]
        gt_col = pq.read_table(path, columns=["initial_edit_region_mask"])["initial_edit_region_mask"].to_pylist()
        preds, gts = [], []
        for case in cases:
            pred_path = os.path.join(out_dir, case, OUTPUT_FILENAME)
            if not (case.isdigit() and os.path.isfile(pred_path)) or int(case) - 1 >= len(gt_col):
                continue
            gt = np.asarray(Image.open(io.BytesIO(gt_col[int(case) - 1]["bytes"])).convert("L")) > 127
            pred = Image.open(pred_path).convert("L").resize(gt.shape[::-1], Image.NEAREST)
            preds.append(torch.from_numpy((np.asarray(pred) > 127).astype(np.float32)))
            gts.append(torch.from_numpy(gt.astype(np.float32)))
        m = mask_metrics(torch.stack(preds), torch.stack(gts), threshold=0.5)
        summary = {"n": len(preds), "mean_iou": m["iou"].mean().item(), "median_iou": m["iou"].median().item(),
                   "mean_dice": m["dice"].mean().item(), "iou_lt_0.1": (m["iou"] < 0.1).float().mean().item(),
                   "iou_gt_0.5": (m["iou"] > 0.5).float().mean().item(), "mean_coverage": m["coverage"].mean().item()}
        print(json.dumps(summary, indent=2))
        with open(os.path.join(os.path.dirname(out_dir), "ier_iou_vs_gt.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
