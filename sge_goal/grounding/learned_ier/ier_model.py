"""
Learned Initial Edit Region (IER) mask predictor built on EVF-SAM.

EVF-SAM = SAM ViT-H image encoder + BEiT-3-Large text-image encoder + SAM mask
decoder (https://github.com/hustvl/EVF-SAM). The SAM image encoder stays frozen
(run in bf16 under no_grad); BEiT-3, the projection and the mask decoder are
fine-tuned to predict ``M_i`` from ``(I_o, T)``.

Heads:
  * ``union``      — one prompt (the instruction) predicts ``M_i`` directly.
  * ``arm_object`` — two prompts per image through the same decoder: a fixed
    arm prompt and the instruction. ``M_i`` = soft union
    ``1 - (1 - p_arm)(1 - p_obj)``. The dataset only has the union mask, so the
    arm/object split is not supervised separately; the prompts only give it an
    inductive bias.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_EVF_ROOT = os.path.join(REPO_ROOT, "run", "third_party", "EVF-SAM")
DEFAULT_EVF_VERSION = "YxZhang/evf-sam"
DEFAULT_ARM_PROMPT = "robot arm and gripper"

SAM_SIZE = 1024
BEIT_SIZE = 224
SAM_MEAN = (123.675, 116.28, 103.53)
SAM_STD = (58.395, 57.12, 57.375)


def import_evf(evf_root=DEFAULT_EVF_ROOT):
    """Make EVF-SAM's ``model`` package importable and return its model class."""
    if evf_root not in sys.path:
        sys.path.insert(0, evf_root)
    from model.evf_sam import EvfSamModel  # noqa: E402
    return EvfSamModel


def sam_preprocess(images):
    """uint8 [B, 3, H, W] -> SAM input [B, 3, 1024, 1024] (resize longest side, normalize, pad)."""
    _, _, h, w = images.shape
    scale = SAM_SIZE / max(h, w)
    nh, nw = int(h * scale + 0.5), int(w * scale + 0.5)
    x = F.interpolate(images.float(), size=(nh, nw), mode="bilinear", align_corners=False)
    mean = torch.tensor(SAM_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(SAM_STD, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return F.pad(x, (0, SAM_SIZE - nw, 0, SAM_SIZE - nh)), (nh, nw)


def beit_preprocess(images):
    """uint8 [B, 3, H, W] -> BEiT-3 input [B, 3, 224, 224] in [-1, 1]."""
    x = F.interpolate(images.float() / 255.0, size=(BEIT_SIZE, BEIT_SIZE),
                      mode="bicubic", align_corners=False).clamp(0, 1)
    return (x - 0.5) / 0.5


def union_prob(logits):
    """[B, P, H, W] per-prompt logits -> [B, H, W] probability of the union."""
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    return 1 - torch.prod(1 - torch.sigmoid(logits), dim=1)


def mask_loss(logits, target, bce_weight=1.0, dice_weight=1.0, focal_gamma=0.0,
              tversky_weight=0.0, tversky_alpha=0.3, tversky_beta=0.7):
    """BCE (or focal) + soft Dice (+ optional Tversky) on the union mask, averaged over the batch.

    Smoothing of 1 makes an empty prediction on an empty target cost nothing.
    """
    logits, target = logits.float(), target.float()
    if logits.shape[1] == 1:
        prob = torch.sigmoid(logits[:, 0])
        bce = F.binary_cross_entropy_with_logits(logits[:, 0], target, reduction="none")
    else:
        prob = union_prob(logits).clamp(1e-6, 1 - 1e-6)
        bce = F.binary_cross_entropy(prob, target, reduction="none")
    if focal_gamma > 0:
        p_t = prob * target + (1 - prob) * (1 - target)
        bce = bce * (1 - p_t) ** focal_gamma
    bce = bce.flatten(1).mean(1)

    p, t = prob.flatten(1), target.flatten(1)
    tp = (p * t).sum(1)
    dice = 1 - (2 * tp + 1) / (p.sum(1) + t.sum(1) + 1)
    total = bce_weight * bce + dice_weight * dice
    terms = {"bce": bce.mean().item(), "dice": dice.mean().item()}
    if tversky_weight > 0:
        fp, fn = (p * (1 - t)).sum(1), ((1 - p) * t).sum(1)
        tversky = 1 - (tp + 1) / (tp + tversky_alpha * fp + tversky_beta * fn + 1)
        total = total + tversky_weight * tversky
        terms["tversky"] = tversky.mean().item()
    return total.mean(), terms


@torch.no_grad()
def mask_metrics(prob, target, threshold=0.5):
    """Per-sample IoU / Dice / predicted coverage at ``threshold`` (empty vs empty counts as 1)."""
    pred = (prob > threshold).flatten(1)
    gt = target.flatten(1) > 0.5
    inter = (pred & gt).sum(1).float()
    union = (pred | gt).sum(1).float()
    size = pred.sum(1).float() + gt.sum(1).float()
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    dice = torch.where(size > 0, 2 * inter / size.clamp(min=1), torch.ones_like(size))
    return {"iou": iou, "dice": dice, "coverage": pred.float().mean(1)}


class LearnedIerModel(nn.Module):
    def __init__(self, evf, heads="union", arm_prompt=DEFAULT_ARM_PROMPT):
        super().__init__()
        if heads not in ("union", "arm_object"):
            raise ValueError(f"Unknown heads: {heads}")
        self.evf = evf
        self.heads = heads
        self.arm_prompt = arm_prompt

    @classmethod
    def from_pretrained(cls, version=DEFAULT_EVF_VERSION, evf_root=DEFAULT_EVF_ROOT, **kwargs):
        EvfSamModel = import_evf(evf_root)
        evf = EvfSamModel.from_pretrained(version, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        return cls(evf, **kwargs)

    @property
    def sam(self):
        return self.evf.visual_model

    @property
    def prompts_per_image(self):
        return 2 if self.heads == "arm_object" else 1

    def prompt_texts(self, instructions):
        """Texts in the order the model expects: [arm_0, obj_0, arm_1, obj_1, ...] or [obj_0, ...]."""
        if self.heads == "union":
            return list(instructions)
        return [t for instr in instructions for t in (self.arm_prompt, instr)]

    def configure_trainable(self, train_beit=True):
        """Freeze everything, then unfreeze the projection, mask decoder, no-mask embedding and optionally BEiT-3."""
        for p in self.evf.parameters():
            p.requires_grad_(False)
        for module in (self.evf.text_hidden_fcs, self.sam.mask_decoder, self.sam.prompt_encoder.no_mask_embed):
            for p in module.parameters():
                p.requires_grad_(True)
        if train_beit:
            for p in self.evf.mm_extractor.parameters():
                p.requires_grad_(True)
        self.sam.image_encoder.to(torch.bfloat16)

    def param_groups(self, lr_beit, lr_head, weight_decay):
        beit, head = [], []
        for name, p in self.evf.named_parameters():
            if p.requires_grad:
                (beit if name.startswith("mm_extractor.") else head).append(p)
        groups = [{"params": head, "lr": lr_head, "weight_decay": weight_decay}]
        if beit:
            groups.append({"params": beit, "lr": lr_beit, "weight_decay": weight_decay})
        return groups

    def trainable_state_dict(self, dtype=torch.bfloat16):
        names = {n for n, p in self.evf.named_parameters() if p.requires_grad}
        return {n: t.detach().to(dtype).cpu() for n, t in self.evf.state_dict().items() if n in names}

    def load_trainable_state_dict(self, state):
        missing = [n for n in state if n not in dict(self.evf.named_parameters())]
        if missing:
            raise KeyError(f"Checkpoint has unknown parameters, e.g. {missing[:3]}")
        own = dict(self.evf.named_parameters())
        with torch.no_grad():
            for n, t in state.items():
                own[n].copy_(t.to(own[n].dtype))

    def train(self, mode=True):
        super().train(mode)
        self.sam.image_encoder.eval()  # frozen encoder always in eval mode
        return self

    @torch.no_grad()
    def image_embeddings(self, sam_images, chunk=4):
        enc = self.sam.image_encoder
        dtype = next(enc.parameters()).dtype
        out = [enc(sam_images[i:i + chunk].to(dtype)) for i in range(0, sam_images.shape[0], chunk)]
        return torch.cat(out).float()

    def prompt_features(self, beit_images, input_ids, attention_mask):
        out = self.evf.mm_extractor.beit3(
            visual_tokens=beit_images,
            textual_tokens=input_ids,
            text_padding_position=~attention_mask.bool(),
        )
        return self.evf.text_hidden_fcs[0](out["encoder_out"][:, :1, :])  # [N, 1, 256]

    def forward(self, images, input_ids, attention_mask):
        """uint8 images [B, 3, H, W] + tokenized prompt_texts -> per-prompt logits [B, P, H, W]."""
        b, _, h, w = images.shape
        n_prompts = self.prompts_per_image
        sam_x, input_size = sam_preprocess(images)
        emb = self.image_embeddings(sam_x)
        feats = self.prompt_features(
            beit_preprocess(images).repeat_interleave(n_prompts, dim=0), input_ids, attention_mask
        )
        dense_pe = self.sam.prompt_encoder.get_dense_pe()
        logits = []
        for i in range(b):
            feat_i = feats[i * n_prompts:(i + 1) * n_prompts]
            sparse, dense = self.sam.prompt_encoder(points=None, boxes=None, masks=None, text_embeds=feat_i)
            low_res, _ = self.sam.mask_decoder(
                image_embeddings=emb[i:i + 1],
                image_pe=dense_pe,
                sparse_prompt_embeddings=sparse.to(feat_i.dtype),
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            logits.append(self.sam.postprocess_masks(low_res, input_size=input_size, original_size=(h, w))[:, 0])
        return torch.stack(logits)
