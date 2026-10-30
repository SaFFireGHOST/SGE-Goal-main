"""Is the learned IER model's M_i on its own training images cleaner than on unseen images?

A-sup / Variant A feed Step 2 predicted M_i for the 22k HF training tuples. If those predictions are
much better than on held-out images (memorisation), Step 2 would train on unrealistically clean
inputs. Compares IoU vs GT on a random training subset against the validation split.

Usage: bash run/predict_ier.sh is not used; run with the same env as run/train_ier.sh:
    PYTHONPATH=run/pydeps:run/third_party/EVF-SAM python run/ier_train_vs_val_iou.py
"""
import glob
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

RUN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(RUN), "sge_goal", "grounding", "learned_ier"))
from ier_data import IerParquetDataset, collate  # noqa: E402
from ier_model import LearnedIerModel, mask_metrics, union_prob  # noqa: E402

CKPT = f"{RUN}/ier_training/evf_union/best.pt"
DATA = f"{RUN}/calvin/hf_dataset/data"
N = 966  # same size as the validation split

ckpt = torch.load(CKPT, map_location="cpu")
tok = AutoTokenizer.from_pretrained(ckpt["evf_version"], padding_side="right", use_fast=False)
model = LearnedIerModel.from_pretrained(ckpt["evf_version"], heads=ckpt["heads"], arm_prompt=ckpt["arm_prompt"])
model.load_trainable_state_dict(ckpt["trainable"])
model.sam.image_encoder.to(torch.bfloat16)
model.to("cuda").eval()

out = {}
for split, pattern, n in [("train_subset", "train-*.parquet", N), ("validation", "validation-*.parquet", None)]:
    ds = IerParquetDataset(sorted(glob.glob(f"{DATA}/{pattern}")), max_samples=n, seed=1)
    ious = []
    for batch in DataLoader(ds, batch_size=8, num_workers=4, collate_fn=collate):
        t = tok(model.prompt_texts(batch["instruction"]), padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch["image"].cuda(), t["input_ids"].cuda(), t["attention_mask"].cuda())
        ious.append(mask_metrics(union_prob(logits.float()), batch["mask"].cuda())["iou"].cpu())
    iou = torch.cat(ious)
    out[split] = {"n": len(iou), "mean_iou": iou.mean().item(), "median_iou": iou.median().item(),
                  "iou_gt_0.5": (iou > 0.5).float().mean().item(), "iou_lt_0.1": (iou < 0.1).float().mean().item()}
    print(split, json.dumps(out[split]))
with open(f"{RUN}/ier_results/ier_train_vs_val_iou.json", "w") as f:
    json.dump(out, f, indent=2)
