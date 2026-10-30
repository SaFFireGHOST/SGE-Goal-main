"""Learned-IER M_i for the HF CALVIN train / validation tuples (inputs for A-sup and Variant A anchors).

Masks are stored exactly as the inference pipeline writes them (binary, mode L, JPEG quality 95),
so Step 2 trains on the same input encoding it sees at test time.
Output: run/calvin/hf_dataset/pred_ier/<split>.parquet with columns id, pred_initial_mask (JPEG bytes).
Resumable per split. Launch with the learned-IER env (see run/train_ier.sh):
    PYTHONPATH=run/pydeps:run/third_party/EVF-SAM python run/variant_a/predict_hf_ier.py
"""
import glob
import io
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(RUN), "sge_goal", "grounding", "learned_ier"))
from ier_data import IerParquetDataset, collate  # noqa: E402
from ier_model import LearnedIerModel, union_prob  # noqa: E402

CKPT = f"{RUN}/ier_training/evf_union/best.pt"
DATA = f"{RUN}/calvin/hf_dataset/data"
OUT = f"{RUN}/calvin/hf_dataset/pred_ier"


def main():
    os.makedirs(OUT, exist_ok=True)
    ckpt = torch.load(CKPT, map_location="cpu")
    tok = AutoTokenizer.from_pretrained(ckpt["evf_version"], padding_side="right", use_fast=False)
    model = LearnedIerModel.from_pretrained(ckpt["evf_version"], heads=ckpt["heads"], arm_prompt=ckpt["arm_prompt"])
    model.load_trainable_state_dict(ckpt["trainable"])
    model.sam.image_encoder.to(torch.bfloat16)
    model.to("cuda").eval()

    for split, pattern in [("validation", "validation-*.parquet"), ("train", "train-*.parquet")]:
        path = f"{OUT}/{split}.parquet"
        if os.path.isfile(path):
            print(f"{split}: done, skipping")
            continue
        ds = IerParquetDataset(sorted(glob.glob(f"{DATA}/{pattern}")))
        ids, masks, t0 = [], [], time.time()
        for b, batch in enumerate(DataLoader(ds, batch_size=16, num_workers=6, collate_fn=collate)):
            t = tok(model.prompt_texts(batch["instruction"]), padding=True, truncation=True, max_length=64, return_tensors="pt")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch["image"].cuda(), t["input_ids"].cuda(), t["attention_mask"].cuda())
            for i, m in zip(batch["id"], (union_prob(logits.float()) > 0.5).cpu().numpy()):
                buf = io.BytesIO()
                Image.fromarray((m * 255).astype(np.uint8), mode="L").save(buf, format="JPEG", quality=95)
                ids.append(i)
                masks.append(buf.getvalue())
            if b % 100 == 0:
                print(f"{split}: {len(ids)}/{len(ds)}  {time.time() - t0:.0f}s", flush=True)
        pq.write_table(pa.table({"id": ids, "pred_initial_mask": masks}), path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"{split}: wrote {len(ids)} masks in {time.time() - t0:.0f}s", flush=True)
    print("PRED_DONE")


if __name__ == "__main__":
    main()
