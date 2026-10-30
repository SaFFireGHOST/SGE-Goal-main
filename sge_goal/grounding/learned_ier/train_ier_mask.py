"""
Train the learned Initial Edit Region (IER) mask predictor on CALVIN.

Fine-tunes EVF-SAM (frozen SAM ViT-H encoder; BEiT-3-L, projection and mask
decoder trained) to predict ``M_i`` from ``(I_o, T)``, selects the checkpoint
on validation IoU and, optionally, reports the best checkpoint on the test
split.

Usage (via run/train_ier.sh, which sets PYTHONPATH for EVF-SAM's dependencies):
    bash run/train_ier.sh --name evf_union --epochs 5
    bash run/train_ier.sh --name smoke --max-train-samples 64 --max-val-samples 32 --max-steps 20

Outputs in --out-dir/<name>/: best.pt, last.pt (trainable weights only),
metrics.jsonl (train/val log), args.json, test_metrics.json.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ier_data import IerParquetDataset, collate, resolve_parquets  # noqa: E402
from ier_model import (  # noqa: E402
    DEFAULT_ARM_PROMPT, DEFAULT_EVF_ROOT, DEFAULT_EVF_VERSION, REPO_ROOT,
    LearnedIerModel, mask_loss, mask_metrics, union_prob,
)

HF_DATA = os.path.join(REPO_ROOT, "run", "calvin", "hf_dataset", "data")


def parse_args():
    p = argparse.ArgumentParser(description="Train the learned IER mask predictor (EVF-SAM).")
    # Data
    p.add_argument("--train-parquet", default=os.path.join(HF_DATA, "train-*.parquet"))
    p.add_argument("--val-parquet", default=os.path.join(HF_DATA, "validation-*.parquet"))
    p.add_argument("--test-parquet", default=os.path.join(HF_DATA, "test-*.parquet"),
                   help="Evaluated once with the best checkpoint at the end; '' to skip.")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--augment", action="store_true", help="Photometric jitter on training images.")
    p.add_argument("--num-workers", type=int, default=6)
    # Model
    p.add_argument("--evf-version", default=DEFAULT_EVF_VERSION)
    p.add_argument("--evf-root", default=DEFAULT_EVF_ROOT)
    p.add_argument("--heads", choices=["union", "arm_object"], default="union")
    p.add_argument("--arm-prompt", default=DEFAULT_ARM_PROMPT)
    p.add_argument("--freeze-beit", action="store_true", help="Train only the projection + mask decoder.")
    # Optimisation
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr-beit", type=float, default=2e-5)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Loss
    p.add_argument("--bce-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=0.0, help=">0 turns BCE into focal loss.")
    p.add_argument("--tversky-weight", type=float, default=0.0)
    # Bookkeeping
    p.add_argument("--name", default="evf_union")
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "run", "ier_training"))
    p.add_argument("--eval-every", type=int, default=0, help="Validate every N optimizer steps (0 = per epoch).")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--resume", default=None,
                   help="last.pt to continue from (restores weights and step; optimizer moments restart).")
    p.add_argument("--save-fp32", action="store_true", help="Save checkpoints in fp32 instead of bf16.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(tokenizer, texts, device):
    tok = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    return tok["input_ids"].to(device), tok["attention_mask"].to(device)


def run_batch(model, tokenizer, batch, device):
    images = batch["image"].to(device, non_blocking=True)
    input_ids, attn = tokenize(tokenizer, model.prompt_texts(batch["instruction"]), device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(images, input_ids, attn)


@torch.no_grad()
def evaluate(model, tokenizer, loader, args, device):
    model.eval()
    sums, n = {}, 0
    for batch in loader:
        logits = run_batch(model, tokenizer, batch, device).float()
        target = batch["mask"].to(device)
        loss, _ = mask_loss(logits, target, args.bce_weight, args.dice_weight, args.focal_gamma, args.tversky_weight)
        m = mask_metrics(union_prob(logits), target, args.threshold)
        k = target.shape[0]
        sums["loss"] = sums.get("loss", 0.0) + loss.item() * k
        for key, v in m.items():
            sums[key] = sums.get(key, 0.0) + v.sum().item()
        sums["iou_lt_0.1"] = sums.get("iou_lt_0.1", 0.0) + (m["iou"] < 0.1).sum().item()
        sums["iou_gt_0.5"] = sums.get("iou_gt_0.5", 0.0) + (m["iou"] > 0.5).sum().item()
        n += k
    model.train()
    return {key: v / max(n, 1) for key, v in sums.items()} | {"n": n}


def lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def save_checkpoint(path, model, args, step, epoch, val):
    torch.save({
        "trainable": model.trainable_state_dict(torch.float32 if args.save_fp32 else torch.bfloat16),
        "evf_version": args.evf_version, "heads": args.heads, "arm_prompt": args.arm_prompt,
        "step": step, "epoch": epoch, "val": val, "args": vars(args),
    }, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    args = parse_args()
    seed_all(args.seed)
    device = args.device
    out_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log_path = os.path.join(out_dir, "metrics.jsonl")

    def log(record):
        print(json.dumps(record), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # Data
    train_set = IerParquetDataset(resolve_parquets(args.train_parquet), augment=args.augment,
                                  max_samples=args.max_train_samples, seed=args.seed)
    val_set = IerParquetDataset(resolve_parquets(args.val_parquet), max_samples=args.max_val_samples, seed=args.seed)
    print(f"train={len(train_set)} val={len(val_set)}", flush=True)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate,
                     pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)

    # Model
    tokenizer = AutoTokenizer.from_pretrained(args.evf_version, padding_side="right", use_fast=False)
    model = LearnedIerModel.from_pretrained(args.evf_version, args.evf_root, heads=args.heads, arm_prompt=args.arm_prompt)
    model.configure_trainable(train_beit=not args.freeze_beit)
    model.to(device).train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.1f}M", flush=True)

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = min(args.epochs * steps_per_epoch, args.max_steps or float("inf"))
    optimizer = torch.optim.AdamW(model.param_groups(args.lr_beit, args.lr_head, args.weight_decay))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, args.warmup_steps, total_steps))

    step, start_epoch, best_iou = 0, 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_trainable_state_dict(ckpt["trainable"])
        step, start_epoch = ckpt["step"], ckpt["epoch"] + 1
        best_iou = (ckpt.get("val") or {}).get("iou", -1.0)
        for _ in range(step):
            scheduler.step()
        print(f"resumed from {args.resume} at step {step}, epoch {start_epoch}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    t0, done = time.time(), False

    def validate(epoch):
        nonlocal best_iou
        val = evaluate(model, tokenizer, val_loader, args, device)
        log({"type": "val", "step": step, "epoch": epoch, **val})
        save_checkpoint(os.path.join(out_dir, "last.pt"), model, args, step, epoch, val)
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            save_checkpoint(os.path.join(out_dir, "best.pt"), model, args, step, epoch, val)
            print(f"new best val IoU {best_iou:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            logits = run_batch(model, tokenizer, batch, device)
            loss, terms = mask_loss(logits, batch["mask"].to(device), args.bce_weight, args.dice_weight,
                                    args.focal_gamma, args.tversky_weight)
            (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                with torch.no_grad():
                    iou = mask_metrics(union_prob(logits.float()), batch["mask"].to(device), args.threshold)["iou"]
                elapsed = time.time() - t0
                log({"type": "train", "step": step, "epoch": epoch, "loss": loss.item(), **terms,
                     "batch_iou": iou.mean().item(), "grad_norm": grad_norm.item(),
                     "lr_head": optimizer.param_groups[0]["lr"], "sec_per_step": elapsed / args.log_every,
                     "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9})
                t0 = time.time()
            if args.eval_every and step % args.eval_every == 0:
                validate(epoch)
            if step >= total_steps:
                done = True
                break
        if not args.eval_every or step % args.eval_every:
            validate(epoch)
        if done:
            break

    if args.test_parquet:
        best = torch.load(os.path.join(out_dir, "best.pt"), map_location="cpu")
        model.load_trainable_state_dict(best["trainable"])
        test_set = IerParquetDataset(resolve_parquets(args.test_parquet))
        test = evaluate(model, tokenizer, DataLoader(test_set, shuffle=False, **loader_kw), args, device)
        test |= {"best_step": best["step"], "best_epoch": best["epoch"]}
        log({"type": "test", **test})
        with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
            json.dump(test, f, indent=2)


if __name__ == "__main__":
    main()
