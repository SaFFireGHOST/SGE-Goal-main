"""
Lean LoRA trainer for SGE-Goal's FLUX.1-Kontext Steps 2-3 (Variant A).

Reproduces the original CALVIN trainers (``mask_predictor/train/flux_kontext_calvin`` and
``synthesis/train/flux_kontext_calvin``) step for step:
  * prompt: Step 2 = instruction + "\\n\\n" + DEFAULT_INSTRUCTION; Step 3 = build_enhanced_prompt(instruction)
  * images resized to resolution x resolution (bilinear), normalised to [-1, 1]
  * VAE latent_dist.mode(), (z - shift) * scale; position ids 0 = target, 1 = I_o, 2 = M_i, 3 = M_f
  * uniform timestep sampling (weighting "none") from the model's flow-matching scheduler, guidance 1.0
  * Step 2 loss: flow MSE + 25 * channel consistency of the decoded one-step estimate (sigma < 0.35)
  * Step 3 loss: flow MSE + 7 * static-scene loss outside max(M_i, M_f) (sigma < 0.35)
  * LoRA rank 16 / alpha 16 on the same 13 module types, gradient checkpointing, bf16 base weights

Differences, on purpose:
  * initialises from a released LoRA (``--init-lora``) instead of from scratch;
  * input M_i can come from the learned IER predictions (``--pred-ier``), as at test time;
  * LoRA parameters in fp32 with plain AdamW (bitsandbytes is not installed);
  * text embeddings precomputed once for the distinct prompts (390 in the CALVIN train split).

Modes:
  * ``sup`` (A-sup): the original losses on HF train tuples.
  * ``dpo`` (A-sparse, proposal Eq. 5-7): Flow-DPO on success preference pairs (``--pairs``, from
    run/variant_a/build_pilot_pairs.py) against a frozen copy of the initial LoRA as reference, plus the
    original losses on HF train tuples as anchor: L = L_sup + lambda_succ * L_DPO. Winner and loser share
    noise and sigma. Use ``--train-gt-mi`` so the anchor sees ground-truth M_i (supervised training on
    learned M_i lowered M_f IoU, plan.md §6.1) while validation keeps the learned M_i.
  * ``dense`` (A-dense, Step 3, Eq. 3 / 7): behaviour matching through the frozen GR-MG policy
    (sge_goal/variant_a/behavior_loss.py) on d_train demo windows, sigma drawn from [0, t_rec_max)
    (the indicator in Eq. 7), plus the original Step 3 loss on HF train tuples as anchor:
    L = L_edit + lambda_beh * L_beh. Conditions for each demo come from ``--beh-dir``.

Usage (via run/variant_a/train_flux_lora.sh):
    bash run/variant_a/train_flux_lora.sh --step 2 --name asup_step2 --epochs 1
    bash run/variant_a/train_flux_lora.sh --step 2 --mode dpo --name pilot_dpo_step2 --train-gt-mi --epochs 4
"""

import argparse
import copy
import glob
import io
import json
import math
import os
import random
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.training_utils import cast_training_params
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN = os.path.join(REPO_ROOT, "run")
HF_DATA = os.path.join(RUN, "calvin", "hf_dataset")
RELEASED = os.path.join(RUN, "ckpts", "sge-goal-checkpoints", "flux1-kontext-dev", "calvin")
STEP_DIRS = {
    2: os.path.join(REPO_ROOT, "sge_goal", "mask_predictor", "inference", "flux_kontext_bridge_calvin"),
    3: os.path.join(REPO_ROOT, "sge_goal", "synthesis", "inference", "flux_kontext_bridge_calvin"),
}
BASE_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
LORA_TARGETS = ["attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0", "attn.add_k_proj",
                "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out", "ff.net.0.proj", "ff.net.2",
                "ff_context.net.0.proj", "ff_context.net.2", "proj_mlp"]


def parse_args():
    p = argparse.ArgumentParser(description="LoRA trainer for SGE-Goal Steps 2-3 (Variant A).")
    p.add_argument("--step", type=int, choices=[2, 3], required=True)
    p.add_argument("--mode", choices=["sup", "dpo", "dense"], default="sup")
    p.add_argument("--name", required=True)
    p.add_argument("--out-dir", default=os.path.join(RUN, "variant_a", "training"))
    p.add_argument("--init-lora", default=None, help="Released LoRA dir (default: calvin/step{2,3}_*).")
    p.add_argument("--train-parquet", default=os.path.join(HF_DATA, "data", "train-*.parquet"))
    p.add_argument("--val-parquet", default=os.path.join(HF_DATA, "data", "validation-*.parquet"))
    p.add_argument("--pred-ier-dir", default=os.path.join(HF_DATA, "pred_ier"),
                   help="Learned-IER M_i per split (<dir>/<split>.parquet); '' = ground-truth M_i.")
    p.add_argument("--train-gt-mi", action="store_true",
                   help="Ground-truth M_i for the training / anchor tuples; validation keeps --pred-ier-dir.")
    p.add_argument("--pairs", default=os.path.join(RUN, "variant_a", "pilot", "pairs.json"), help="dpo mode.")
    p.add_argument("--pairs-batch", type=int, default=2, help="dpo mode: preference pairs per micro-step.")
    p.add_argument("--beta", type=float, default=2000.0, help="dpo mode: DPO temperature on per-example MSE.")
    p.add_argument("--lambda-succ", type=float, default=1.0, help="dpo mode: weight of L_DPO.")
    p.add_argument("--beh-dir", default=None, help="dense mode: CALVIN-EVALS-style folder with M_i / M_f per demo.")
    p.add_argument("--d-train-dir", default=os.path.join(RUN, "calvin", "d_train"))
    p.add_argument("--beh-batch", type=int, default=2, help="dense mode: demo windows per micro-step.")
    p.add_argument("--beh-val-frac", type=float, default=0.05, help="dense mode: held-out demos for val L_beh.")
    p.add_argument("--lambda-beh", type=float, default=10.0, help="dense mode; L_beh is ~1e-2 vs flow ~0.1-0.3.")
    p.add_argument("--beh-gripper-weight", type=float, default=0.1)
    p.add_argument("--grmg-ckpt", default=os.path.join(RUN, "eval_inputs", "ckpts", "gr_mg", "epoch=47-step=83712.ckpt"))
    p.add_argument("--grmg-config", default=os.path.join(REPO_ROOT, "downstream", "calvin", "gr_mg", "policy", "config", "train.json"))
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=float, default=1.0, help="Over HF tuples (sup) or preference pairs (dpo).")
    p.add_argument("--max-steps", type=int, default=None, help="Optimizer steps; overrides --epochs.")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--t-rec-max", type=float, default=0.35)
    p.add_argument("--lambda-chan", type=float, default=25.0)
    p.add_argument("--lambda-static", type=float, default=7.0)
    p.add_argument("--val-every", type=int, default=100, help="Optimizer steps; 0 = only at start/end.")
    p.add_argument("--val-loss-samples", type=int, default=128)
    p.add_argument("--val-iou-samples", type=int, default=64, help="Step 2 only: 10-step sampling, mask IoU.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="Continue from <out>/last_state.pt.")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_split(pattern, columns):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return pq.read_table(files, columns=columns).to_pydict()


class TupleDataset(Dataset):
    """HF CALVIN tuples -> normalised tensors. Step 2 target = M_f; Step 3 target = I_e (M_f is a condition)."""

    def __init__(self, pattern, step, pred_ier=None, max_samples=None, seed=0, resolution=256):
        cols = ["id", "original_image", "initial_edit_region_mask", "final_edit_region_mask", "edit_instruction"]
        if step == 3:
            cols.append("edited_image")
        d = read_split(pattern, cols)
        self.ids = d["id"]
        self.instructions = [s.strip() for s in d["edit_instruction"]]
        self.images = {k: [c["bytes"] for c in d[k]] for k in cols if k not in ("id", "edit_instruction")}
        if pred_ier:
            pred = pq.read_table(pred_ier).to_pydict()
            lookup = dict(zip(pred["id"], pred["pred_initial_mask"]))
            missing = [i for i in self.ids if i not in lookup]
            if missing:
                raise KeyError(f"{len(missing)} ids without a predicted M_i in {pred_ier}, e.g. {missing[:3]}")
            self.images["initial_edit_region_mask"] = [lookup[i] for i in self.ids]
        self.order = list(range(len(self.ids)))
        if max_samples is not None and max_samples < len(self.order):
            self.order = sorted(random.Random(seed).sample(self.order, max_samples))
        self.step = step
        self.tf = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.order)

    def _img(self, key, j):
        return self.tf(Image.open(io.BytesIO(self.images[key][j])).convert("RGB"))

    def __getitem__(self, i):
        j = self.order[i]
        item = {"original": self._img("original_image", j), "m_i": self._img("initial_edit_region_mask", j),
                "m_f": self._img("final_edit_region_mask", j), "instruction": self.instructions[j], "index": j}
        item["target"] = self._img("edited_image", j) if self.step == 3 else item["m_f"]
        return item


def collate(items):
    keys = [k for k in ("original", "m_i", "m_f", "target", "target_l") if k in items[0]]
    out = {k: torch.stack([it[k] for it in items]) for k in keys}
    out["instruction"] = [it["instruction"] for it in items]
    out["index"] = [it["index"] for it in items]
    return out


class PairDataset(Dataset):
    """Success preference pairs: target = winner, target_l = loser, same conditions (Step 3 adds M_f)."""

    def __init__(self, pairs, step, resolution=256):
        self.pairs, self.step = pairs, step
        self.tf = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.instructions = [q["instruction"] for q in pairs]

    def __len__(self):
        return len(self.pairs)

    def _img(self, path):
        return self.tf(Image.open(path).convert("RGB"))

    def __getitem__(self, i):
        q = self.pairs[i]
        item = {"original": self._img(q["original"]), "m_i": self._img(q["m_i"]), "instruction": q["instruction"],
                "target": self._img(q["win"]), "target_l": self._img(q["lose"]), "index": i}
        item["m_f"] = self._img(q["m_f"]) if self.step == 3 else item["target"]
        return item


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class KontextLoraTrainer:
    def __init__(self, args):
        self.args, self.step, self.device = args, args.step, args.device
        sys.path.insert(0, STEP_DIRS[args.step])
        from pipeline_flux_kontext_multi_imgs import FluxKontextPipeline  # step-specific copy
        self.Pipe = FluxKontextPipeline
        if args.step == 2:
            from local_fer_mask_inference import DEFAULT_INSTRUCTION
            self.make_prompt = lambda instr: instr.strip() + "\n\n" + DEFAULT_INSTRUCTION.strip()
        else:
            from flux_kontext_eval_utils import build_enhanced_prompt
            self.make_prompt = lambda instr: build_enhanced_prompt(instr.strip().splitlines()[0])

        dt = torch.bfloat16
        self.transformer = FluxTransformer2DModel.from_pretrained(BASE_MODEL, subfolder="transformer", torch_dtype=dt)
        self.vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", torch_dtype=dt)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.vsf = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.shift, self.scale = self.vae.config.shift_factor, self.vae.config.scaling_factor

        self.transformer.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, lora_dropout=0.0,
                                                init_lora_weights="gaussian", target_modules=LORA_TARGETS))
        init = args.init_lora or os.path.join(RELEASED, "step2_final_mask" if args.step == 2 else "step3_goal_image")
        self.load_lora(os.path.join(init, "pytorch_lora_weights.safetensors"))
        cast_training_params(self.transformer, dtype=torch.float32)
        if args.mode == "dpo":  # frozen reference = the initial LoRA (add_adapter activates it; switch back)
            self.transformer.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, lora_dropout=0.0,
                                                    target_modules=LORA_TARGETS), adapter_name="ref")
            self.load_lora(os.path.join(init, "pytorch_lora_weights.safetensors"), adapter="ref")
            self.transformer.set_adapter("default")
        self.transformer.enable_gradient_checkpointing()
        self.transformer.to(self.device)
        self.vae.to(self.device)
        self.lora_params = [p for p in self.transformer.parameters() if p.requires_grad]
        if args.mode == "dense":
            from behavior_loss import load_grmg
            self.policy = load_grmg(args.grmg_ckpt, args.grmg_config, self.device)
        print(f"LoRA from {init}: {sum(p.numel() for p in self.lora_params) / 1e6:.1f}M trainable params", flush=True)

    def load_lora(self, path, adapter="default"):
        sd = {k[len("transformer."):]: v for k, v in load_file(path).items() if k.startswith("transformer.")}
        res = set_peft_model_state_dict(self.transformer, sd, adapter_name=adapter)
        unexpected = getattr(res, "unexpected_keys", []) or []
        if unexpected:
            raise KeyError(f"LoRA keys not matched in the transformer, e.g. {unexpected[:3]}")
        n_lora = sum(1 for n, _ in self.transformer.named_parameters() if "lora_" in n and f".{adapter}." in n)
        if n_lora != len(sd):
            raise ValueError(f"{path} has {len(sd)} LoRA tensors but the transformer has {n_lora}")

    def save_lora(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        state = {k: v.to(torch.bfloat16) for k, v in get_peft_model_state_dict(self.transformer).items()}
        self.Pipe.save_lora_weights(out_dir, transformer_lora_layers=state,
                                    weight_name="pytorch_lora_weights.safetensors")

    # -- text ---------------------------------------------------------------
    def build_prompt_cache(self, instructions):
        """Encode every distinct prompt once (text encoders loaded on the GPU only for this)."""
        prompts = sorted({self.make_prompt(s) for s in instructions})
        pipe = self.Pipe.from_pretrained(BASE_MODEL, transformer=None, vae=None, torch_dtype=torch.bfloat16)
        pipe.text_encoder.to(self.device)
        pipe.text_encoder_2.to(self.device)
        cache = {}
        with torch.no_grad():
            for p in prompts:
                emb, pooled, text_ids = pipe.encode_prompt(p, prompt_2=None, device=self.device, max_sequence_length=512)
                cache[p] = (emb.cpu(), pooled.cpu())
        self.text_ids = text_ids.to(self.device)
        del pipe
        torch.cuda.empty_cache()
        self.prompt_cache = cache
        print(f"encoded {len(cache)} distinct prompts", flush=True)

    def text_inputs(self, instructions):
        pairs = [self.prompt_cache[self.make_prompt(s)] for s in instructions]
        emb = torch.cat([e for e, _ in pairs]).to(self.device)
        pooled = torch.cat([q for _, q in pairs]).to(self.device)
        return emb, pooled

    # -- latents ------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x):
        z = self.vae.encode(x.to(self.device, self.vae.dtype)).latent_dist.mode()
        return (z - self.shift) * self.scale

    def decode(self, z):
        return self.vae.decode((z / self.scale + self.shift).to(self.vae.dtype)).sample.float()

    def ids_for(self, latents_list):
        ids = []
        for k, z in enumerate(latents_list):
            i = self.Pipe._prepare_latent_image_ids(z.shape[0], z.shape[2] // 2, z.shape[3] // 2, self.device, z.dtype)
            i[..., 0] = k
            ids.append(i)
        return torch.cat(ids, dim=0)

    def pack(self, z):
        return self.Pipe._pack_latents(z, z.shape[0], z.shape[1], z.shape[2], z.shape[3])

    def flow_and_aux_loss(self, batch, generator=None, noise=None, idx=None, with_aux=True, return_x0=False):
        """Per-example flow loss + the step's auxiliary loss (as in the original trainers).
        ``noise`` / ``idx`` (timestep indices) can be given to share them across calls (DPO)."""
        x = self.encode(batch["target"])
        conds = [self.encode(batch["original"]), self.encode(batch["m_i"])]
        if self.step == 3:
            conds.append(self.encode(batch["m_f"]))
        emb, pooled = self.text_inputs(batch["instruction"])
        b = x.shape[0]
        if noise is None:
            noise = torch.randn(x.shape, generator=generator, device=self.device, dtype=x.dtype)
        if idx is None:
            u = torch.rand(b, generator=generator, device=self.device)
            idx = (u * self.scheduler.config.num_train_timesteps).long().cpu()
        timesteps = self.scheduler.timesteps[idx].to(self.device)
        sigmas = self.scheduler.sigmas[idx].to(self.device, x.dtype).view(b, 1, 1, 1)
        noisy = (1.0 - sigmas) * x + sigmas * noise
        hidden = torch.cat([self.pack(noisy)] + [self.pack(c) for c in conds], dim=1)
        n_tok = hidden.shape[1] // (1 + len(conds))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.transformer(hidden_states=hidden, timestep=timesteps / 1000,
                                    guidance=torch.full((b,), 1.0, device=self.device),
                                    pooled_projections=pooled, encoder_hidden_states=emb,
                                    txt_ids=self.text_ids, img_ids=self.ids_for([x] + conds), return_dict=False)[0]
        pred = self.Pipe._unpack_latents(pred[:, :n_tok], x.shape[2] * self.vsf, x.shape[3] * self.vsf, self.vsf)
        target = noise - x
        flow = ((pred.float() - target.float()) ** 2).reshape(b, -1).mean(1)
        if return_x0:  # one-step clean estimate (latent), as used by the auxiliary losses
            return flow, noise - pred

        aux = torch.zeros((), device=self.device)
        rec = (sigmas.view(-1) < self.args.t_rec_max).nonzero().view(-1)
        if with_aux and rec.numel():
            pred_img = self.decode((noise - pred)[rec])
            if self.step == 2:
                r, g, bl = pred_img[:, 0:1], pred_img[:, 1:2], pred_img[:, 2:3]
                aux = self.args.lambda_chan * ((r - g) ** 2 + (r - bl) ** 2 + (g - bl) ** 2).mean()
            else:
                with torch.no_grad():
                    orig = self.decode(conds[0][rec])
                    masks = [self.decode(c[rec]) for c in conds[1:3]]
                    norm = [((m - m.amin((1, 2, 3), keepdim=True)) /
                             (m.amax((1, 2, 3), keepdim=True) - m.amin((1, 2, 3), keepdim=True) + 1e-8)).clamp(0, 1)
                            for m in masks]
                    keep = 1.0 - torch.maximum(*norm).clamp(0, 1).mean(1, keepdim=True)
                aux = self.args.lambda_static * (((pred_img - orig) ** 2) * keep).mean()
        return flow, aux

    def dpo_loss(self, batch, generator=None):
        """Flow-DPO (Eq. 5): winner and loser share noise and sigma; reference = frozen initial LoRA."""
        n = batch["target"].shape[0]
        both = {"original": batch["original"].repeat(2, 1, 1, 1), "m_i": batch["m_i"].repeat(2, 1, 1, 1),
                "m_f": batch["m_f"].repeat(2, 1, 1, 1), "instruction": batch["instruction"] * 2,
                "target": torch.cat([batch["target"], batch["target_l"]])}
        r = self.args.resolution // self.vsf
        noise = torch.randn((n, self.vae.config.latent_channels, r, r), generator=generator,
                            device=self.device, dtype=self.vae.dtype).repeat(2, 1, 1, 1)
        u = torch.rand(n, generator=generator, device=self.device)
        idx = (u * self.scheduler.config.num_train_timesteps).long().cpu().repeat(2)
        with torch.no_grad():
            self.transformer.set_adapter("ref")
            ref = self.flow_and_aux_loss(both, noise=noise, idx=idx, with_aux=False)[0]
            self.transformer.set_adapter("default")
        pol = self.flow_and_aux_loss(both, noise=noise, idx=idx, with_aux=False)[0]
        d_w, d_l = pol[:n] - ref[:n], pol[n:] - ref[n:]
        logits = -self.args.beta * (d_w - d_l)
        return -F.logsigmoid(logits).mean(), (logits > 0).float().mean(), (d_l - d_w).mean()

    def beh_loss(self, batch, generator=None):
        """Eq. 3 / 7: decoded one-step estimate at sigma < t_rec_max -> frozen GR-MG action discrepancy."""
        from behavior_loss import behavior_loss
        b = batch["target"].shape[0]
        cand = (self.scheduler.sigmas[:-1] < self.args.t_rec_max).nonzero().view(-1)
        u = torch.rand(b, generator=generator, device=self.device).cpu()
        idx = cand[(u * len(cand)).long().clamp(max=len(cand) - 1)]
        flow, x0 = self.flow_and_aux_loss(batch, generator=generator, idx=idx, with_aux=False, return_x0=True)
        img01 = ((self.decode(x0) + 1) / 2).clamp(0, 1)
        per = behavior_loss(self.policy, img01, batch, self.device, generator, self.args.beh_gripper_weight)
        return per.mean(), flow.mean()

    @torch.no_grad()
    def val_beh(self, loader):
        self.transformer.eval()
        g = torch.Generator(device=self.device).manual_seed(1234)
        tot = n = 0.0
        for batch in loader:
            l, _ = self.beh_loss(batch, generator=g)
            tot, n = tot + l.item() * len(batch["index"]), n + len(batch["index"])
        self.transformer.train()
        return tot / max(n, 1)

    # -- validation ---------------------------------------------------------
    @torch.no_grad()
    def val_pairs(self, loader):
        """Held-out preference pairs: DPO loss and implicit preference accuracy (fixed noise)."""
        self.transformer.eval()
        g = torch.Generator(device=self.device).manual_seed(1234)
        loss = acc = n = 0.0
        for batch in loader:
            lo, ac, _ = self.dpo_loss(batch, generator=g)
            b = batch["target"].shape[0]
            loss, acc, n = loss + lo.item() * b, acc + ac.item() * b, n + b
        self.transformer.train()
        return loss / max(n, 1), acc / max(n, 1)

    @torch.no_grad()
    def val_loss(self, loader):
        self.transformer.eval()
        g = torch.Generator(device=self.device).manual_seed(1234)  # same noise / sigmas every time
        tot, n = 0.0, 0
        for batch in loader:
            flow, aux = self.flow_and_aux_loss(batch, generator=g)
            tot += (flow.sum() + aux * flow.shape[0]).item()
            n += flow.shape[0]
        self.transformer.train()
        return tot / max(n, 1)

    @torch.no_grad()
    def val_mask_iou(self, ds, n):
        """Step 2: 10-step sampling as local_fer_mask_inference.py, IoU of the binarised M_f vs ground truth."""
        self.transformer.eval()
        # Own scheduler copy: sampling calls set_timesteps(10), which would corrupt training's 1000-step table.
        pipe = self.Pipe(scheduler=copy.deepcopy(self.scheduler), vae=self.vae, text_encoder=None, tokenizer=None,
                         text_encoder_2=None, tokenizer_2=None, transformer=self.transformer)
        pipe.set_progress_bar_config(disable=True)
        device = self.device
        pipe.__class__ = type(pipe.__class__.__name__, (pipe.__class__,),
                              {"_execution_device": property(lambda self: torch.device(device))})
        ious = []
        for i in range(min(n, len(ds))):
            j = ds.order[i]
            orig = Image.open(io.BytesIO(ds.images["original_image"][j])).convert("RGB")
            m_i = Image.open(io.BytesIO(ds.images["initial_edit_region_mask"][j])).convert("RGB")
            gt = np.asarray(Image.open(io.BytesIO(ds.images["final_edit_region_mask"][j])).convert("L")) > 127
            w, h = (orig.size[0] // 32) * 32, (orig.size[1] // 32) * 32
            emb, pooled = self.text_inputs([ds.instructions[j]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = pipe(original_image=orig, initial_edit_region_mask=m_i, prompt_embeds=emb,
                           pooled_prompt_embeds=pooled, height=h, width=w, guidance_scale=1.0,
                           num_inference_steps=10, max_sequence_length=512, _auto_resize=False, max_area=w * h,
                           generator=torch.Generator(device=device).manual_seed(i)).images[0]
            pred = np.asarray(out.resize(orig.size, Image.BILINEAR).convert("L")) > 127
            union = (pred | gt).sum()
            ious.append((pred & gt).sum() / union if union else 1.0)
        self.transformer.train()
        return float(np.mean(ious))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = os.path.join(args.out_dir, args.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    def log(rec):
        print(json.dumps(rec), flush=True)
        with open(os.path.join(out, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    pred = lambda split: os.path.join(args.pred_ier_dir, f"{split}.parquet") if args.pred_ier_dir else None
    train_mi = None if args.train_gt_mi else pred("train")
    train_ds = TupleDataset(args.train_parquet, args.step, train_mi, args.max_train_samples, args.seed, args.resolution)
    val_ds = TupleDataset(args.val_parquet, args.step, pred("validation"), args.val_loss_samples, 0, args.resolution)
    iou_ds = TupleDataset(args.val_parquet, args.step, pred("validation"), args.val_iou_samples, 1, args.resolution)
    dpo = args.mode == "dpo"
    if dpo:
        with open(args.pairs) as f:
            pairs = json.load(f)[f"step{args.step}"]
        pair_ds, pair_val_ds = PairDataset(pairs["train"], args.step, args.resolution), PairDataset(pairs["val"], args.step, args.resolution)
        pair_val_loader = DataLoader(pair_val_ds, batch_size=args.pairs_batch, shuffle=False, num_workers=2, collate_fn=collate)
    dense = args.mode == "dense"
    if dense:
        assert args.step == 3 and args.beh_dir, "dense mode is Step 3 with --beh-dir"
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from behavior_loss import BehaviorWindows, collate_windows
        all_beh = BehaviorWindows(args.beh_dir, args.d_train_dir, resolution=args.resolution)
        eps = sorted(all_beh.ctx)
        val_eps = set(random.Random(args.seed).sample(eps, max(1, round(args.beh_val_frac * len(eps)))))
        beh_ds = torch.utils.data.Subset(all_beh, [i for i, (e, _) in enumerate(all_beh.items) if e not in val_eps])
        beh_val = torch.utils.data.Subset(all_beh, [i for i, (e, _) in enumerate(all_beh.items) if e in val_eps])
        beh_val_loader = DataLoader(beh_val, batch_size=args.beh_batch, shuffle=False, num_workers=2, collate_fn=collate_windows)
    # The epoch runs over the driver set: HF tuples (sup), preference pairs (dpo) or demo windows (dense);
    # in dpo / dense the HF tuples are the anchor.
    driver_ds, driver_bs = ((pair_ds, args.pairs_batch) if dpo else (beh_ds, args.beh_batch) if dense
                            else (train_ds, args.micro_batch))
    steps_per_epoch = len(driver_ds) // (driver_bs * args.grad_accum)
    assert steps_per_epoch > 0, "fewer training samples than one optimizer step"

    def epoch_loader(step, ds=driver_ds, bs=driver_bs, seed=args.seed):
        # Order is a fixed function of (seed, epoch), so --resume continues mid-epoch on unseen samples.
        spe = len(ds) // (bs * args.grad_accum)
        epoch, done_steps = divmod(step, spe)
        perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        return DataLoader(ds, batch_size=bs, sampler=perm[done_steps * bs * args.grad_accum:], drop_last=True,
                          num_workers=args.num_workers, collate_fn=collate_windows if dense and ds is beh_ds else collate)

    def anchor_batches():
        # Endless anchor stream over HF tuples (its own order; restarts from the top on --resume).
        spe, epoch = len(train_ds) // (args.micro_batch * args.grad_accum), 0
        while True:
            yield from epoch_loader(epoch * spe, train_ds, args.micro_batch, args.seed + 10_000)
            epoch += 1

    val_loader = DataLoader(val_ds, batch_size=args.micro_batch, shuffle=False, num_workers=2, collate_fn=collate)
    print(f"train={len(train_ds)} (M_i: {'ground truth' if train_mi is None else 'learned'}) val_loss={len(val_ds)} "
          f"val_iou={len(iou_ds)} (M_i: {'learned' if args.pred_ier_dir else 'ground truth'})"
          + (f" pairs train={len(pair_ds)} val={len(pair_val_ds)}" if dpo else "")
          + (f" windows train={len(beh_ds)} val={len(beh_val)} ({len(eps)} demos)" if dense else ""), flush=True)

    tr = KontextLoraTrainer(args)
    tr.build_prompt_cache(train_ds.instructions + val_ds.instructions
                          + (pair_ds.instructions + pair_val_ds.instructions if dpo else [])
                          + (all_beh.instructions if dense else []))

    total = args.max_steps or max(1, int(args.epochs * steps_per_epoch))
    opt = torch.optim.AdamW(tr.lora_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)))
    step, best = 0, None
    state_path = os.path.join(out, "last_state.pt")
    if args.resume and os.path.isfile(state_path):
        st = torch.load(state_path, map_location="cpu")
        tr.load_lora(os.path.join(out, "last", "pytorch_lora_weights.safetensors"))
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step, best = st["step"], st.get("best")
        print(f"resumed at step {step}", flush=True)

    def validate():
        nonlocal best
        rec = {"type": "val", "step": step, "val_loss": tr.val_loss(val_loader)}
        if args.step == 2 and args.val_iou_samples:
            rec["val_mask_iou"] = tr.val_mask_iou(iou_ds, args.val_iou_samples)
        if dpo:
            if len(pair_val_ds):
                rec["val_pair_loss"], rec["val_pair_acc"] = tr.val_pairs(pair_val_loader)
        if dense:
            rec["val_beh"] = tr.val_beh(beh_val_loader)
        if dpo or dense:
            tr.save_lora(os.path.join(out, f"ckpt_{step:05d}"))  # selection is downstream (rollouts)
        score = rec.get("val_mask_iou", -rec["val_loss"])
        if best is None or score > best:
            best = score
            tr.save_lora(os.path.join(out, "best"))
            rec["new_best"] = True
        log(rec)

    if step == 0:
        validate()  # released LoRA, before any update
    tr.transformer.train()
    t0, done = time.time(), step >= total
    torch.cuda.reset_peak_memory_stats()
    anchor = anchor_batches() if dpo or dense else None
    while not done:
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(epoch_loader(step)):
            extra = {}
            if dpo:  # backward per term, so only one graph is alive at a time
                l_dpo, acc, margin = tr.dpo_loss(batch)
                (args.lambda_succ * l_dpo / args.grad_accum).backward()
                extra = {"dpo": l_dpo.item(), "pair_acc": acc.item(), "margin": margin.item()}
                batch = next(anchor)
            elif dense:
                l_beh, beh_flow = tr.beh_loss(batch)
                (args.lambda_beh * l_beh / args.grad_accum).backward()
                extra = {"beh": l_beh.item(), "beh_flow": beh_flow.item()}
                batch = next(anchor)
            flow, aux = tr.flow_and_aux_loss(batch)
            loss = flow.mean() + aux
            (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum:
                continue
            gn = torch.nn.utils.clip_grad_norm_(tr.lora_params, args.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                dt = (time.time() - t0) / args.log_every
                log({"type": "train", "step": step, "loss": loss.item(), "flow": flow.mean().item(),
                     "aux": float(aux), **extra, "grad_norm": gn.item(), "lr": sched.get_last_lr()[0],
                     "sec_per_step": dt, "sec_per_sample": dt / (args.micro_batch * args.grad_accum),
                     "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9})
                t0 = time.time()
            if args.val_every and step % args.val_every == 0:
                tr.save_lora(os.path.join(out, "last"))
                torch.save({"step": step, "best": best, "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict()}, state_path)
                validate()
            if step >= total:
                done = True
                break
    tr.save_lora(os.path.join(out, "last"))
    if not args.val_every or step % args.val_every:
        validate()
    log({"type": "done", "step": step})


if __name__ == "__main__":
    main()
