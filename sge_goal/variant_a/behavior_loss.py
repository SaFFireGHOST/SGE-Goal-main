"""Dense policy signal for Variant A (proposal Eq. 3): behaviour matching through the frozen GR-MG policy.

L_beh = E_{t, xi} || pi(o_t, xi(I_hat_e), T) - pi(o_t, xi(I_e), T) ||^2
  * I_hat_e: Step 3's one-step clean estimate (decoded, differentiable), I_e: the demo's end frame;
  * o_t: a 10-frame GR-MG history window from the same demo (run/calvin/d_train frames);
  * xi: one shared random differentiable augmentation (brightness / contrast, mild blur, small shift);
  * actions: the full (seq_len x act_len) arm-action chunk predicted with the style vector at 0, as at
    test time, plus gripper probabilities (weight --beh-gripper-weight).
The goal path mirrors evaluation: generated goal 256 -> 208 (saved size) -> 200 bicubic -> 224 bicubic.
Gradients reach the generator only through the goal image; GR-MG weights stay frozen (eval mode).
"""
import glob
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRMG_DIR = os.path.join(REPO_ROOT, "downstream", "calvin", "gr_mg")
GRMG_DEPS = os.path.join(REPO_ROOT, "run", "pydeps_grmg")  # flamingo-pytorch, openai CLIP, ftfy (--target install)
for p in (GRMG_DEPS, GRMG_DIR, os.path.join(GRMG_DIR, "evaluate")):
    if p not in sys.path:
        sys.path.insert(0, p)

CLIP_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SEQ_LEN = 10


def load_grmg(ckpt_path, config_path, device):
    """GR_MG built exactly as evaluate/wrapper/model_wrapper.py (without the CALVIN env wrapper), frozen."""
    import clip
    import policy.model.vision_transformer as vits
    from policy.model.model import GR_MG

    with open(config_path) as f:
        cfg = json.load(f)
    pc, ic, tc = cfg["policy"], cfg["input"], cfg["trainer"]
    target = [k for k in ("act_pred", "fwd_pred", "fwd_pred_hand", "progress_pred") if tc[k]]
    clip_model, _ = clip.load("ViT-B/32", device=device)
    model = GR_MG(state_dim=ic["state_dim"], act_dim=ic["act_dim"], act_len=pc["act_len"],
                  act_latent_dim=pc["act_latent_dim"], act_encoder_dim=pc["act_encoder_dim"],
                  act_decoder_dim=pc["act_decoder_dim"], progress_decoder_dim=pc["progress_decoder_dim"],
                  hidden_size=pc["embed_dim"], model_mae=vits.__dict__["vit_base"](patch_size=16, num_classes=0),
                  clip_model=clip_model, img_feat_dim=pc["img_feat_dim"], lang_feat_dim=pc["lang_feat_dim"],
                  patch_feat_dim=pc["patch_feat_dim"], resampler_params=pc["resampler_params"],
                  max_length=pc["seq_len"], training_target=target,
                  without_norm_pix_loss=tc["without_norm_pix_loss"], use_hand_rgb=ic["use_hand_rgb"],
                  use_state=ic["use_state"], use_resampler=pc["use_resampler"], n_layer=pc["n_layer"],
                  n_head=pc["n_head"], n_inner=4 * pc["embed_dim"], activation_function=pc["activation_function"],
                  n_positions=1024, resid_pdrop=pc["dropout"], attn_pdrop=pc["dropout"], device=device)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict({k[6:]: v for k, v in sd.items() if "model." in k}, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model


def clip_preprocess(x01):
    """[0, 1] images (b, 3, h, w) -> GR-MG input: 224 bicubic + ImageNet normalisation (differentiable)."""
    x = F.interpolate(x01, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    return (x - CLIP_MEAN.to(x)) / CLIP_STD.to(x)


def generated_goal_to_policy_res(img01):
    """Generated goal at the trainer resolution -> 208 (pipeline output) -> 200 bicubic (eval loader)."""
    x = F.interpolate(img01, size=(208, 208), mode="bilinear", align_corners=False, antialias=True)
    return F.interpolate(x, size=(200, 200), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)


def augment(xs, generator):
    """One random brightness / contrast / blur / shift, applied identically to every tensor in xs."""
    b, dev = xs[0].shape[0], xs[0].device
    r = lambda *s: torch.rand(*s, generator=generator, device=dev)
    bright = (0.9 + 0.2 * r(b)).view(b, 1, 1, 1)
    contrast = (0.9 + 0.2 * r(b)).view(b, 1, 1, 1)
    sigma = 0.1 + 0.7 * r(b)
    shift = (r(b, 2) - 0.5) * 2 * (4 / 100)  # up to +-4 px of 200, in [-1, 1] grid units
    k = torch.arange(-1, 2, device=dev, dtype=xs[0].dtype)
    g1 = torch.exp(-(k.view(1, 3) ** 2) / (2 * sigma.view(b, 1) ** 2))
    g1 = g1 / g1.sum(1, keepdim=True)
    kern = (g1.view(b, 3, 1) * g1.view(b, 1, 3)).view(b, 1, 3, 3).repeat_interleave(3, 0)
    theta = torch.zeros(b, 2, 3, device=dev, dtype=xs[0].dtype)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, :, 2] = shift
    out = []
    for x in xs:
        mean = x.mean((1, 2, 3), keepdim=True)
        y = ((x - mean) * contrast + mean) * bright
        y = F.pad(y, (1, 1, 1, 1), mode="replicate").reshape(1, b * 3, x.shape[2] + 2, x.shape[3] + 2)
        y = F.conv2d(y, kern, groups=b * 3).reshape(b, 3, *x.shape[2:])
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        out.append(F.grid_sample(y, grid, mode="bilinear", padding_mode="border", align_corners=False).clamp(0, 1))
    return out


def rel_states(robot_obs):
    """GR-MG relative proprioception for a window (CustomModel.compute_rel_state)."""
    from utils.utils import euler2rotm, rotm2euler

    first_xyz, first_rotm = robot_obs[0, :3], euler2rotm(robot_obs[0, 3:6])
    arm = np.zeros((len(robot_obs), 6))
    for i in range(1, len(robot_obs)):
        rotm = euler2rotm(robot_obs[i, 3:6])
        arm[i, :3] = first_rotm.T @ (robot_obs[i, :3] - first_xyz)
        arm[i, 3:6] = rotm2euler(first_rotm.T @ rotm)
    gripper = ((robot_obs[:, -1] + 1.0) / 2.0).astype(np.int64)
    return arm, gripper


def _rgb01(b):
    return torch.from_numpy(np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)


COND_FILES = {"original": "original_image", "m_i": "initial_edit_region_mask", "m_f": "final_edit_region_mask"}


def _find(d, stem):
    for ext in ("png", "jpg", "jpeg"):
        if os.path.isfile(os.path.join(d, f"{stem}.{ext}")):
            return os.path.join(d, f"{stem}.{ext}")
    return None


def load_conditions(src):
    """{episode: {instruction, original, m_i, m_f (encoded image bytes)}} from a CALVIN-EVALS folder or a parquet."""
    if src.endswith(".parquet"):
        return {r.pop("episode"): r for r in pq.read_table(src).to_pylist()}
    out = {}
    for c in os.listdir(src):
        d = os.path.join(src, c)
        paths = {k: _find(d, stem) for k, stem in COND_FILES.items()}
        if not c.isdigit() or None in paths.values():
            continue
        rec = {k: open(p, "rb").read() for k, p in paths.items()}
        rec["instruction"] = open(os.path.join(d, "edit_instruction.txt")).read().strip()
        out[int(c) - 1] = rec
    return out


def pack_conditions(src_dir, out_path):
    """CALVIN-EVALS condition folder -> one parquet (fast to upload / read)."""
    import pyarrow as pa
    cond = load_conditions(src_dir)
    rows = [{"episode": e, **r} for e, r in sorted(cond.items())]
    pq.write_table(pa.Table.from_pylist(rows), out_path)
    return len(rows)


class BehaviorWindows(Dataset):
    """d_train demo windows with the generator's Step 3 conditions for that demo.

    Conditions come from a CALVIN-EVALS-style folder (case <episode + 1>: original_image, edit_instruction.txt,
    initial_edit_region_mask.jpg, final_edit_region_mask.jpg) or from the same packed into one parquet
    (pack_conditions; columns episode, instruction, original, m_i, m_f); the target is the demo's end frame.
    One item = one (episode, window end) pair; only episodes with conditions are used.
    """

    def __init__(self, cond_dir, d_train_dir, episodes=None, resolution=256):
        from torchvision import transforms

        self.cond = load_conditions(cond_dir)
        cases = set(self.cond)
        if episodes is not None:
            cases &= set(episodes)
        ctx = pq.read_table(sorted(glob.glob(f"{d_train_dir}/contexts_part*.parquet")),
                            columns=["episode_index", "window_ends", "robot_obs", "end_image"]).to_pylist()
        self.ctx = {c["episode_index"]: c for c in ctx if c["episode_index"] in cases}
        need = {(e, f) for e, c in self.ctx.items() for t in c["window_ends"] for f in range(t - SEQ_LEN + 1, t + 1)}
        import pyarrow as pa
        import pyarrow.compute as pc
        self.frames = {}
        for f in sorted(glob.glob(f"{d_train_dir}/frames_part*.parquet")):
            t = pq.read_table(f)
            t = t.filter(pc.is_in(t["episode_index"], value_set=pa.array(sorted(self.ctx), pa.int64())))
            for r in t.to_pylist():
                if (r["episode_index"], r["frame"]) in need:
                    self.frames[(r["episode_index"], r["frame"])] = (r["image"], r["wrist"])
        self.items = [(e, t) for e in sorted(self.ctx) for t in self.ctx[e]["window_ends"]]
        self.tf = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.instructions = sorted({self.cond[e]["instruction"] for e in self.ctx})

    def __len__(self):
        return len(self.items)

    def _cond(self, e, key):
        return self.tf(Image.open(io.BytesIO(self.cond[e][key])).convert("RGB"))

    def __getitem__(self, i):
        e, t = self.items[i]
        c = self.ctx[e]
        fids = range(t - SEQ_LEN + 1, t + 1)
        obs = np.asarray(c["robot_obs"], dtype=np.float64).reshape(-1, 15)[list(fids)]
        arm, grip = rel_states(obs)
        end = Image.open(io.BytesIO(c["end_image"])).convert("RGB")
        return {
            "original": self._cond(e, "original"), "m_i": self._cond(e, "m_i"),
            "m_f": self._cond(e, "m_f"), "target": self.tf(end),
            "goal_gt01": torch.from_numpy(np.asarray(end.resize((200, 200), Image.BICUBIC), dtype=np.float32) / 255.0).permute(2, 0, 1),
            "instruction": self.cond[e]["instruction"],
            "rgb01": torch.stack([_rgb01(self.frames[(e, f)][0]) for f in fids]),
            "hand01": torch.stack([_rgb01(self.frames[(e, f)][1]) for f in fids]),
            "arm_state": torch.from_numpy(arm).float(), "gripper_state": torch.from_numpy(grip), "index": i,
        }


def collate_windows(items):
    out = {k: torch.stack([it[k] for it in items]) for k in
           ("original", "m_i", "m_f", "target", "goal_gt01", "rgb01", "hand01", "arm_state", "gripper_state")}
    out["instruction"] = [it["instruction"] for it in items]
    out["index"] = [it["index"] for it in items]
    return out


def policy_actions(policy, goal01, batch, device):
    """GR-MG action chunk (b, seq_len, act_len, 6) + gripper probs (b, seq_len, act_len) for goal images in [0, 1]."""
    b = goal01.shape[0]
    with torch.no_grad():
        rgb = clip_preprocess(batch["rgb01"].to(device).flatten(0, 1)).view(b, SEQ_LEN, 3, 224, 224)
        hand = clip_preprocess(batch["hand01"].to(device).flatten(0, 1)).view(b, SEQ_LEN, 3, 224, 224)
    act_len = policy.act_len
    inp = {"goal_rgb": clip_preprocess(goal01), "rgb": rgb, "hand_rgb": hand,
           "arm_state": batch["arm_state"].to(device),
           "gripper_state": F.one_hot(batch["gripper_state"].to(device), 2).float(),
           "arm_action": torch.zeros(b, SEQ_LEN, act_len, 6, device=device),
           "gripper_action": torch.zeros(b, SEQ_LEN, act_len, device=device),
           "attention_mask": torch.ones(b, SEQ_LEN, dtype=torch.long, device=device),
           "text": [batch["instruction"]], "progress": torch.zeros(b, SEQ_LEN, device=device)}
    pred = policy(inp, is_training=False)
    return pred["arm_action_preds"].float(), torch.sigmoid(pred["gripper_action_preds"].float().squeeze(-1))


def behavior_loss(policy, goal_hat01, batch, device, generator=None, gripper_weight=0.1):
    """Eq. 3 for a batch: goal_hat01 = decoded one-step estimate in [0, 1] at the trainer resolution."""
    hat = generated_goal_to_policy_res(goal_hat01.float())
    gt = batch["goal_gt01"].to(device)
    hat, gt = augment([hat, gt], generator)
    with torch.no_grad():
        arm_gt, grip_gt = policy_actions(policy, gt, batch, device)
    arm_hat, grip_hat = policy_actions(policy, hat, batch, device)
    per = ((arm_hat - arm_gt) ** 2).flatten(1).mean(1) + gripper_weight * ((grip_hat - grip_gt) ** 2).flatten(1).mean(1)
    return per
