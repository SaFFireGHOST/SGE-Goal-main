# Octo (JAX) -> CALVIN policy adapter.
#
# Mimics the interface of GR-MG's `CustomModel` so that
# `evaluate/eval_method.py`'s rollout loop is unchanged:
#
#     model = OctoModelWrapper(...)
#     model.reset()
#     action_torch7d, info = model.step(obs, [goal_rgb], [lang])
#
# * obs: same dict produced by CalvinEnvWrapperRaw
#       obs["rgb_obs"]["rgb_static"]   — (200, 200, 3) uint8
#       obs["rgb_obs"]["rgb_gripper"] — (84, 84, 3) uint8
#       obs["robot_obs"]              — (15,) float
# * action returned: torch.Tensor shape (7,) in CALVIN rel_actions space
#       [dx, dy, dz, drx, dry, drz, gripper]   gripper in {-1, +1}
#
# Octo-base is not natively trained on CALVIN; it WAS trained on
# `taco_play` from Open-X-Embodiment, which IS the CALVIN-D play data
# (recorded by KIT). So we ask the model to produce actions in the
# `taco_play` action space and convert to CALVIN's rel_actions.
#
# taco_play action format (per Octo's oxe_standardization_transforms.py):
#   [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]
#   gripper in {0, 1} (closed/open)  -> remap to {-1, +1}
# Magnitudes are in METERS (xyz) and RADIANS (rpy) per environment step.
#
# CALVIN's rel_actions are NOT metric: calvin_env/robot/robot.py
# relative_to_absolute() does
#     rel_pos *= max_rel_pos (0.02) * magic_scaling_factor_pos (1)
#     rel_orn *= max_rel_orn (0.05) * magic_scaling_factor_orn (1)
# so a CALVIN rel_action of 1.0 means 2 cm / 0.05 rad and the space is
# normalised to ~[-1, 1]. Passing taco_play's metre/radian deltas straight
# through therefore under-drives the arm by ~50x (xyz) and ~20x (rpy), which
# is why every Octo-CALVIN method scored ~1-3% at n=1 and 0% beyond.
# Convert explicitly: divide by the per-step maxima.

from __future__ import annotations

import os

# CALVIN relative-action scaling (calvin_env/robot/robot.py defaults).
CALVIN_MAX_REL_POS = 0.02   # metres per unit rel_action
CALVIN_MAX_REL_ORN = 0.05   # radians per unit rel_action

from typing import List, Optional

import numpy as np
from PIL import Image


def _resize_uint8(img: np.ndarray, size: int) -> np.ndarray:
    """Resize a (H, W, 3) uint8 image to (size, size, 3) uint8."""
    if img.shape[:2] == (size, size):
        return img.astype(np.uint8, copy=False)
    pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    pil = pil.resize((size, size), Image.BICUBIC)
    return np.asarray(pil, dtype=np.uint8)


class OctoModelWrapper:
    """Pretrained Octo wrapper presenting GR-MG's CustomModel interface."""

    def __init__(
        self,
        hf_repo: str = "hf://rail-berkeley/octo-base",
        action_head: str = "taco_play",
        action_space: str = None,   # "taco" (zero-shot) | "calvin" (finetuned)
        history_window: int = 2,
        pred_horizon: int = 4,
        device_id: int = 0,
    ):
        self.hf_repo = hf_repo
        self.action_head = action_head
        # Zero-shot octo-base emits taco_play-space actions; a CALVIN-finetuned
        # checkpoint emits CALVIN rel_actions directly. Default keeps the
        # zero-shot behaviour; override with OCTO_ACTION_SPACE=calvin.
        self.action_space = (action_space
                             or os.environ.get('OCTO_ACTION_SPACE', 'taco')).lower()

        self.history_window = int(history_window)
        self.pred_horizon = int(pred_horizon)
        self.device_id = int(device_id)
        # Octo expects 256x256 image_primary and 128x128 image_wrist.
        self.size_primary = 256
        self.size_wrist = 128

        import jax  # noqa: F401  (initialise JAX before model import)
        from octo.model.octo_model import OctoModel  # type: ignore

        print(f"[octo] loading {hf_repo}")
        self.model = OctoModel.load_pretrained(hf_repo)
        # Per-dataset un-normalisation statistics (Octo policy outputs
        # normalised actions which must be un-normalised back to physical units
        # using the dataset's mean/std).
        ds_stats = self.model.dataset_statistics
        if "action" in ds_stats:
            # A checkpoint finetuned on a SINGLE dataset stores its statistics
            # flat (keys: action, proprio, num_trajectories, num_transitions),
            # whereas the multi-dataset octo-base release keys them by dataset
            # name ("taco_play", "bridge_dataset", ...). For a finetune these
            # already ARE the target dataset's statistics.
            self.unnorm_stats = ds_stats
            action_head = "<finetuned:single-dataset>"
        elif action_head in ds_stats:
            self.unnorm_stats = ds_stats[action_head]
        else:
            raise KeyError(
                f"action_head={action_head!r} not in Octo dataset_statistics. "
                f"Available: {list(ds_stats.keys())}"
            )
        print(
            f"[octo] using action head: {action_head}  "
            f"action_dim={len(self.unnorm_stats['action']['mean'])}"
        )

        # Internal buffers (filled by reset()).
        self._obs_history: List[dict] = []   # rolling window of obs dicts
        self._action_chunk: Optional[np.ndarray] = None  # cached chunk
        self._chunk_idx: int = 0
        self._task = None  # JAX task pytree (set on first step / reset_with_task)

        import jax.random as jrandom  # type: ignore
        self._rng = jrandom.PRNGKey(0)

    # ------------------------------------------------------------------
    # GR-MG-compatible API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._obs_history = []
        self._action_chunk = None
        self._chunk_idx = 0
        self._task = None  # task is rebuilt lazily on first step(...)

    def step(self, obs, goal, text):
        """One inference step.

        Args:
            obs: dict from CalvinEnvWrapperRaw.
            goal: list of 1 element — the goal RGB as uint8 (200,200,3).
            text: list of 1 string — the language instruction.

        Returns:
            (action_torch7d, info_int) — info is a dummy progress placeholder.
        """
        import jax.random as jrandom  # type: ignore

        # ---- 1. Build / cache the task (goal_image and/or language) ---
        # OCTO_TASK_MODE selects what the policy is conditioned on:
        #   both      (default) goal image + language instruction
        #   goal_only goal image alone -- Octo pads/masks the language
        #   text_only language alone
        # This matters for measuring goal-image QUALITY: with a correct language
        # instruction always present, a policy finetuned with language dropout
        # can satisfy the task from language and largely ignore the goal image,
        # which collapses the spread between generators. create_tasks documents
        # "omit texts to run the goal-conditioned model".
        if self._task is None:
            mode = os.environ.get("OCTO_TASK_MODE", "both").lower()
            goal_img = _resize_uint8(goal[0], self.size_primary)[None]  # (1, H, W, 3)
            lang = text[0]
            if mode == "goal_only":
                self._task = self.model.create_tasks(
                    goals={"image_primary": goal_img})
            elif mode == "text_only":
                self._task = self.model.create_tasks(texts=[lang])
            else:
                self._task = self.model.create_tasks(
                    goals={"image_primary": goal_img}, texts=[lang])

        # ---- 2. Update rolling obs window ----------------------------
        primary = _resize_uint8(
            obs["rgb_obs"]["rgb_static"], self.size_primary
        )
        wrist = _resize_uint8(
            obs["rgb_obs"]["rgb_gripper"], self.size_wrist
        )
        self._obs_history.append({"image_primary": primary, "image_wrist": wrist})
        if len(self._obs_history) > self.history_window:
            self._obs_history.pop(0)

        # ---- 3. Resample chunk every pred_horizon steps --------------
        if self._action_chunk is None or self._chunk_idx >= self.pred_horizon:
            obs_batch = self._build_obs_batch()
            self._rng, sub = jrandom.split(self._rng)
            actions = self.model.sample_actions(
                obs_batch,
                self._task,
                unnormalization_statistics=self.unnorm_stats["action"],
                rng=sub,
            )
            # actions: (1, pred_horizon, action_dim)
            actions = np.asarray(actions[0])  # (pred_horizon, A)
            self._action_chunk = actions
            self._chunk_idx = 0

        a = self._action_chunk[self._chunk_idx]
        self._chunk_idx += 1

        # ---- 4. Map the model's action -> CALVIN rel_actions ----------
        a7 = np.zeros(7, dtype=np.float32)
        if self.action_space == "taco":
            # Zero-shot octo-base speaks the taco_play head's space:
            #   [dx, dy, dz, droll, dpitch, dyaw] in METRES / RADIANS,
            #   gripper in {0, 1}.
            # metres -> CALVIN rel units (1.0 == max_rel_pos == 0.02 m)
            a7[:3] = np.asarray(a[:3], dtype=np.float32) / CALVIN_MAX_REL_POS
            # radians -> CALVIN rel units (1.0 == max_rel_orn == 0.05 rad)
            a7[3:6] = np.asarray(a[3:6], dtype=np.float32) / CALVIN_MAX_REL_ORN
            g = float(a[6])
            a7[6] = -1.0 if g < 0.5 else 1.0
        else:
            # A checkpoint finetuned on CALVIN (see calvin_rlds/) already emits
            # CALVIN rel_actions: Octo un-normalises with the finetuning
            # dataset's own statistics, and the gripper is left unnormalised at
            # +-1 by action_normalization_mask. Rescaling here would be wrong.
            a7[:6] = np.asarray(a[:6], dtype=np.float32)
            g = float(a[6])
            a7[6] = -1.0 if g < 0.0 else 1.0
        # CALVIN's rel_actions live in [-1, 1]; clip so a large chunk value
        # cannot command a single implausible jump.
        a7[:6] = np.clip(a7[:6], -1.0, 1.0)

        import torch  # type: ignore
        return torch.from_numpy(a7), 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_obs_batch(self) -> dict:
        """Build the JAX-friendly obs batch with history padding."""
        H = self.history_window
        n = len(self._obs_history)
        # Pad with the oldest frame at the front.
        padded = [self._obs_history[0]] * (H - n) + self._obs_history if n < H else self._obs_history[-H:]
        primary = np.stack([f["image_primary"] for f in padded], axis=0)  # (H, h, w, 3)
        wrist = np.stack([f["image_wrist"] for f in padded], axis=0)

        timestep_pad_mask = np.ones((H,), dtype=np.bool_)
        if n < H:
            timestep_pad_mask[: H - n] = False

        return {
            "image_primary": primary[None],          # (1, H, h, w, 3) uint8
            "image_wrist": wrist[None],              # (1, H, h, w, 3) uint8
            "timestep_pad_mask": timestep_pad_mask[None],  # (1, H)
        }
