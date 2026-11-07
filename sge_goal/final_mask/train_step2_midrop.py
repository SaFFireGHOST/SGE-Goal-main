"""
Step 2 supervised training with M_i dropout, so the model cannot rely on copying M_i and has to read the image and
instruction. Per training example: black M_i with probability --mi-drop, a damaged M_i with probability --mi-damage
(one of shift / grow / shrink / cut, on average as far from the true mask as learned M_i, so it is harder to tell
from a real mask than a black one), otherwise unchanged. Validation is untouched. Otherwise train_flux_lora_lowmem.py, same arguments; also keeps ckpt_v<n> at each
validation (n-th "val" row of metrics.jsonl, v0 = before any update, not saved).

    python sge_goal/final_mask/train_step2_midrop.py --step 2 --name midrop_step2 --train-gt-mi ...
"""

import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_flux_lora_lowmem import trainer  # noqa: E402


def damage(m, rng):
    """m: 3 x H x W in [-1, 1]. One random edit of the binarised mask, same format."""
    x = (m[:1] > 0).float()[None]
    for op in rng.sample(("shift", "grow", "shrink", "cut"), 1):
        if op == "shift":
            dy, dx = rng.randint(-8, 8), rng.randint(-8, 8)
            x = F.pad(x, (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)))
            x = x[..., max(-dy, 0):x.shape[-2] - max(dy, 0), max(-dx, 0):x.shape[-1] - max(dx, 0)]
        elif op in ("grow", "shrink"):
            k = rng.choice(range(3, 12, 2))
            x = F.max_pool2d(x, k, 1, k // 2) if op == "grow" else 1 - F.max_pool2d(1 - x, k, 1, k // 2)
        else:
            on = x[0, 0].nonzero()
            for _ in range(rng.randint(1, 3) if len(on) else 0):
                cy, cx = on[rng.randrange(len(on))].tolist()
                r = rng.randint(6, 20)
                x[..., max(cy - r, 0):cy + r, max(cx - r, 0):cx + r] = 0
    return (x[0] * 2 - 1).expand_as(m)


def perturb(m_i, p_drop, p_damage, rng):
    out = m_i.clone()
    for k in range(len(out)):
        u = rng.random()
        if u < p_drop:
            out[k] = -1.0
        elif u < p_drop + p_damage:
            out[k] = damage(out[k], rng)
    return out


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mi-drop", type=float, default=0.25)
    p.add_argument("--mi-damage", type=float, default=0.25)
    own, rest = p.parse_known_args()
    sys.argv = sys.argv[:1] + rest
    a = trainer.parse_args()
    out = os.path.join(a.out_dir, a.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "midrop.json"), "w") as f:
        json.dump(vars(own), f, indent=2)
    rng = random.Random(a.seed)

    flow_and_aux_loss = trainer.KontextLoraTrainer.flow_and_aux_loss
    val_mask_iou = trainer.KontextLoraTrainer.val_mask_iou

    def patched_loss(self, batch, **kw):
        if self.transformer.training and torch.is_grad_enabled():
            batch = {**batch, "m_i": perturb(batch["m_i"], own.mi_drop, own.mi_damage, rng)}
        return flow_and_aux_loss(self, batch, **kw)

    def patched_iou(self, ds, n):
        r = val_mask_iou(self, ds, n)
        path = os.path.join(out, "metrics.jsonl")
        v = sum('"type": "val"' in line for line in open(path)) if os.path.isfile(path) else 0
        if v:
            self.save_lora(os.path.join(out, f"ckpt_v{v}"))
        return r

    trainer.KontextLoraTrainer.flow_and_aux_loss = patched_loss
    trainer.KontextLoraTrainer.val_mask_iou = patched_iou
    trainer.main()


if __name__ == "__main__":
    main()
