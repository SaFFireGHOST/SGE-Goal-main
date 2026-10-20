#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

import diffusers
import importlib


def _import_or_get(module: str, name: str) -> Any:
    """Import a symbol with fallbacks without hard import failures."""
    try:
        return getattr(importlib.import_module(module), name)
    except Exception:
        return None


AutoencoderKLFlux2 = getattr(diffusers, "AutoencoderKLFlux2", None) or _import_or_get(
    "diffusers.models", "AutoencoderKLFlux2"
)
FlowMatchEulerDiscreteScheduler = getattr(
    diffusers, "FlowMatchEulerDiscreteScheduler", None
) or _import_or_get("diffusers.schedulers", "FlowMatchEulerDiscreteScheduler") or _import_or_get(
    "diffusers.schedulers.scheduling_flow_match_euler_discrete", "FlowMatchEulerDiscreteScheduler"
)
Flux2KleinPipeline = getattr(diffusers, "Flux2KleinPipeline", None) or _import_or_get(
    "diffusers.pipelines", "Flux2KleinPipeline"
) or _import_or_get("diffusers.pipelines.flux_2_klein", "Flux2KleinPipeline")
Flux2Transformer2DModel = getattr(diffusers, "Flux2Transformer2DModel", None) or _import_or_get(
    "diffusers.models", "Flux2Transformer2DModel"
) or _import_or_get("diffusers.models.transformers", "Flux2Transformer2DModel")

_missing = [
    name
    for name, obj in {
        "AutoencoderKLFlux2": AutoencoderKLFlux2,
        "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
        "Flux2KleinPipeline": Flux2KleinPipeline,
        "Flux2Transformer2DModel": Flux2Transformer2DModel,
    }.items()
    if obj is None
]
if _missing:
    raise ImportError(
        "Missing Diffusers symbols: "
        + ", ".join(_missing)
        + ". Ensure you installed a Diffusers build that contains Flux 2 Klein."
        + " (Using git+https://github.com/huggingface/diffusers.git is recommended.)"
    )

from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# Optional imports for LoRA-only checkpoint hooks (available in recent diffusers)
try:
    from diffusers.training_utils import cast_training_params
except ImportError:
    cast_training_params = None

try:
    from diffusers.training_utils import _collate_lora_metadata
except ImportError:
    _collate_lora_metadata = None

try:
    from diffusers.loaders.peft import convert_unet_state_dict_to_peft
except ImportError:
    convert_unet_state_dict_to_peft = None
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_torch_npu_available, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "You are trying to load your data using the datasets library. If you wish to train using custom "
        "captions please install the datasets library: `pip install datasets`."
    )

if is_wandb_available():
    import wandb

check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

if is_torch_npu_available():
    npu = getattr(torch, "npu", None)
    if npu is not None:
        try:
            npu.config.allow_internal_format = False
        except Exception:
            pass


DEFAULT_ERM_INSTRUCTION = """TASK: Edit Region Mask Prediction for Robotic Manipulation

You are provided with the following inputs:
1. ORIGINAL IMAGE: A scene containing objects, including a robotic arm and dynamic objects that can be manipulated.
2. INITIAL EDIT REGION MASK: A binary mask (white=1, black=0) showing the current locations of dynamic objects in the original image. White pixels indicate regions occupied by objects that will be manipulated/moved.
3. EDIT INSTRUCTION (above): A text description of the manipulation/edit to perform on the dynamic objects.

YOUR TASK: Generate a FINAL EDIT REGION MASK - a binary mask (white=1, black=0) predicting where the dynamic objects should appear AFTER the edit instruction is executed.
- White pixels (1): Regions where dynamic objects will be located after manipulation
- Black pixels (0): Background regions that remain unchanged

IMPORTANT:
- Focus ONLY on predicting object locations, not colors or textures
- The mask should be consistent across all RGB channels
- Use the initial mask to understand which objects are dynamic
- Use the edit instruction to determine the target locations
- Output should be a clean binary mask showing final object positions"""


def build_erm_prompt(edit_instruction: str) -> str:
    """Build a mask-prediction prompt with a fixed ERM instruction suffix."""
    return f"{edit_instruction}\n\n{DEFAULT_ERM_INSTRUCTION.strip()}"


# -------------------------
# Latent packing helpers
# -------------------------

def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
    return latents


def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
    return latents


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
    return latents


def _unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch_size, _, channels = latents.shape
    h_patch = height // 2
    w_patch = width // 2
    latents = latents.permute(0, 2, 1).reshape(batch_size, channels, h_patch, w_patch)
    latents = _unpatchify_latents(latents)
    return latents


def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = latents.shape
    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.arange(1)
    latent_ids = torch.cartesian_prod(t, h, w, l)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
    return latent_ids


def _prepare_image_ids(image_latents: List[torch.Tensor], scale: int = 10) -> torch.Tensor:
    if not isinstance(image_latents, list):
        raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")

    t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
    t_coords = [t.view(-1) for t in t_coords]

    image_latent_ids = []
    for x, t in zip(image_latents, t_coords):
        _, height, width = x.shape[1], x.shape[2], x.shape[3]
        x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
        image_latent_ids.append(x_ids)

    image_latent_ids = torch.cat(image_latent_ids, dim=0)
    batch_size = image_latents[0].shape[0]
    image_latent_ids = image_latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

    return image_latent_ids


# -------------------------
# Metrics helpers
# -------------------------

def _pil_to_numpy(img: Optional[Image.Image]) -> Optional[np.ndarray]:
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        arr = img
    else:
        img = img.convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _binarize_mask_from_rgb(mask_img: Any, threshold: float = 0.5) -> Optional[np.ndarray]:
    if mask_img is None:
        return None
    if isinstance(mask_img, np.ndarray):
        m = mask_img.astype(np.float32)
    else:
        m = np.asarray(mask_img.convert("RGB")).astype(np.float32)
    if m.ndim == 3:
        m_single = np.max(m[..., :3], axis=2)
    else:
        m_single = m
    if m_single.max() > 1.0:
        m_single = m_single / 255.0
    bin_mask = (m_single > threshold).astype(np.float32)
    return bin_mask


def _compute_confusion_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[int, int, int, int]:
    pred_bool = pred_mask > 0.5
    gt_bool = gt_mask > 0.5
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, np.logical_not(gt_bool)).sum())
    fn = int(np.logical_and(np.logical_not(pred_bool), gt_bool).sum())
    tn = int(np.logical_and(np.logical_not(pred_bool), np.logical_not(gt_bool)).sum())
    return tp, fp, tn, fn


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def compute_mask_metrics(
    predicted_images: List[Optional[Image.Image]],
    ground_truth_images: List[Optional[Image.Image]],
    is_final_validation: bool,
) -> Dict[str, Any]:
    n = len(predicted_images)
    assert n == len(ground_truth_images), "Input lists must match length"

    per_sample: Dict[str, List[float]] = {
        "iou": [],
        "dice": [],
        "precision": [],
        "recall": [],
        "pixel_accuracy": [],
        "mae": [],
    }

    for i in range(n):
        pred_img = predicted_images[i]
        gt_img = ground_truth_images[i]

        pred_np = _pil_to_numpy(pred_img) if pred_img is not None else None
        gt_np = _pil_to_numpy(gt_img) if gt_img is not None else None

        if pred_np is None or gt_np is None:
            for k in per_sample:
                per_sample[k].append(float("nan"))
            continue

        if pred_np.shape[:2] != gt_np.shape[:2]:
            gt_h, gt_w = gt_np.shape[:2]
            pred_img = Image.fromarray((pred_np * 255).astype(np.uint8)).convert("RGB")
            pred_img = pred_img.resize((gt_w, gt_h), resample=Image.BILINEAR)
            pred_np = np.asarray(pred_img).astype(np.float32) / 255.0

        pred_mask = _binarize_mask_from_rgb((pred_np * 255).astype(np.uint8))
        gt_mask = _binarize_mask_from_rgb((gt_np * 255).astype(np.uint8))

        tp, fp, tn, fn = _compute_confusion_from_masks(pred_mask, gt_mask)
        total = tp + fp + tn + fn

        pixel_acc = _safe_div(tp + tn, total) if total > 0 else float("nan")
        precision = _safe_div(tp, (tp + fp))
        recall = _safe_div(tp, (tp + fn))
        dice = _safe_div(2 * tp, (2 * tp + fp + fn))
        iou = _safe_div(tp, (tp + fp + fn))
        mae = float(np.mean(np.abs(pred_mask - gt_mask)))

        per_sample["iou"].append(float(iou))
        per_sample["dice"].append(float(dice))
        per_sample["precision"].append(float(precision))
        per_sample["recall"].append(float(recall))
        per_sample["pixel_accuracy"].append(float(pixel_acc))
        per_sample["mae"].append(float(mae))

    def _safe_stats(vals: List[float]) -> Dict[str, float]:
        arr = np.array([x for x in vals if not (isinstance(x, float) and math.isnan(x))], dtype=float)
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

    aggregated: Dict[str, float] = {}
    for k, v in per_sample.items():
        stats = _safe_stats(v)
        aggregated[f"mask_{k}_mean"] = stats["mean"]
        aggregated[f"mask_{k}_std"] = stats["std"]

    prefix = "test" if is_final_validation else "val"
    logger.info("[%s] Mask-prediction aggregated metrics:", prefix.upper())
    for k, v in aggregated.items():
        logger.info("%s/%s: %s", prefix, k, v)

    return {"aggregated": aggregated, "per_sample": per_sample}


# -------------------------
# Dataset
# -------------------------

class KontextDataset(Dataset):
    def __init__(
        self,
        args,
        split: str = "train",
        repeats: int = 1,
        center_crop: bool = False,
    ):
        self.args = args
        self.center_crop = center_crop

        dataset: Any = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=split,
            cache_dir=args.cache_dir,
        )
        self.dataset = dataset

        self.edited_image_column = args.edited_image_column
        self.original_image_column = args.original_image_column
        self.initial_edit_region_mask_column = args.initial_edit_region_mask_column
        self.final_edit_region_mask_column = args.final_edit_region_mask_column
        self.edit_instruction_column = args.edit_instruction_column

        self.repeats = repeats
        self.num_instance_images = len(self.dataset)
        self._length = self.num_instance_images * repeats
        self.resolution = args.resolution

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        actual_index = index % self.num_instance_images

        target_mask = self.dataset[actual_index][self.edited_image_column]
        target_mask = exif_transpose(target_mask)
        if target_mask.mode != "RGB":
            target_mask = target_mask.convert("RGB")

        original_image = self.dataset[actual_index][self.original_image_column]
        original_image = exif_transpose(original_image)
        if original_image.mode != "RGB":
            original_image = original_image.convert("RGB")

        initial_edit_region_mask = self.dataset[actual_index][self.initial_edit_region_mask_column]
        initial_edit_region_mask = exif_transpose(initial_edit_region_mask)
        if initial_edit_region_mask.mode != "RGB":
            initial_edit_region_mask = initial_edit_region_mask.convert("RGB")

        final_edit_region_mask = self.dataset[actual_index][self.final_edit_region_mask_column]
        final_edit_region_mask = exif_transpose(final_edit_region_mask)
        if final_edit_region_mask.mode != "RGB":
            final_edit_region_mask = final_edit_region_mask.convert("RGB")

        size = (self.resolution, self.resolution)

        target_mask, original_image, initial_edit_region_mask, final_edit_region_mask = self.tuple_transform(
            target_mask,
            original_image=original_image,
            initial_edit_region_mask=initial_edit_region_mask,
            final_edit_region_mask=final_edit_region_mask,
            size=size,
            center_crop=self.center_crop,
            random_flip=self.args.random_flip,
        )

        edit_instruction = self.dataset[actual_index][self.edit_instruction_column]
        enhanced_prompt = build_erm_prompt(edit_instruction)

        example = {
            "instance_image": target_mask,
            "original_image": original_image,
            "initial_edit_region_mask": initial_edit_region_mask,
            "final_edit_region_mask": final_edit_region_mask,
            "instance_prompt": enhanced_prompt,
            "edit_instruction": edit_instruction,
        }

        return example

    def tuple_transform(
        self,
        image,
        original_image,
        initial_edit_region_mask,
        final_edit_region_mask,
        size=(224, 224),
        center_crop: bool = False,
        random_flip: bool = False,
    ):
        resize_image = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        resize_mask = transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST)

        image = resize_mask(image)
        original_image = resize_image(original_image)
        initial_edit_region_mask = resize_mask(initial_edit_region_mask)
        final_edit_region_mask = resize_mask(final_edit_region_mask)

        if center_crop:
            crop = transforms.CenterCrop(size)
            image = crop(image)
            original_image = crop(original_image)
            initial_edit_region_mask = crop(initial_edit_region_mask)
            final_edit_region_mask = crop(final_edit_region_mask)
        else:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=size)
            image = TF.crop(image, i, j, h, w)
            original_image = TF.crop(original_image, i, j, h, w)
            initial_edit_region_mask = TF.crop(initial_edit_region_mask, i, j, h, w)
            final_edit_region_mask = TF.crop(final_edit_region_mask, i, j, h, w)

        if random_flip:
            do_flip = random.random() < 0.5
            if do_flip:
                image = TF.hflip(image)
                original_image = TF.hflip(original_image)
                initial_edit_region_mask = TF.hflip(initial_edit_region_mask)
                final_edit_region_mask = TF.hflip(final_edit_region_mask)

        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        image = normalize(to_tensor(image))
        original_image = normalize(to_tensor(original_image))
        initial_edit_region_mask = normalize(to_tensor(initial_edit_region_mask))
        final_edit_region_mask = normalize(to_tensor(final_edit_region_mask))

        return image, original_image, initial_edit_region_mask, final_edit_region_mask


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_values = torch.stack([x["instance_image"] for x in examples]).to(
        memory_format=torch.contiguous_format
    ).float()
    original_image_pixel_values = torch.stack([x["original_image"] for x in examples]).to(
        memory_format=torch.contiguous_format
    ).float()
    initial_edit_region_mask_pixel_values = torch.stack([x["initial_edit_region_mask"] for x in examples]).to(
        memory_format=torch.contiguous_format
    ).float()
    final_edit_region_mask_pixel_values = torch.stack([x["final_edit_region_mask"] for x in examples]).to(
        memory_format=torch.contiguous_format
    ).float()
    prompts = [x["instance_prompt"] for x in examples]
    edit_instructions = [x["edit_instruction"] for x in examples]

    batch = {
        "pixel_values": pixel_values,
        "original_image_pixel_values": original_image_pixel_values,
        "initial_edit_region_mask_pixel_values": initial_edit_region_mask_pixel_values,
        "final_edit_region_mask_pixel_values": final_edit_region_mask_pixel_values,
        "prompts": prompts,
        "edit_instructions": edit_instructions,
    }
    return batch


# -------------------------
# Qwen encoding
# -------------------------

def _prepare_text_ids(prompt_embeds: torch.Tensor, device, dtype) -> torch.Tensor:
    """Prepare 4D position IDs for text tokens matching Flux2KleinPipeline._prepare_text_ids."""
    B, L, _ = prompt_embeds.shape
    out_ids = []
    for _i in range(B):
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(L)
        coords = torch.cartesian_prod(t, h, w, l)
        out_ids.append(coords)
    return torch.stack(out_ids).to(device=device, dtype=dtype)


def encode_prompt_qwen(tokenizer, text_encoder, prompt, max_sequence_length, device,
                       text_encoder_out_layers=(9, 18, 27)):
    """Encode prompt matching Flux2KleinPipeline._get_qwen3_prompt_embeds exactly."""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    all_input_ids = []
    all_attention_masks = []
    for single_prompt in prompt:
        # Use chat template matching pipeline
        messages = [{"role": "user", "content": single_prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            # Fallback if enable_thinking not supported
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
        all_input_ids.append(inputs["input_ids"])
        all_attention_masks.append(inputs["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0).to(device)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

    output = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    # Use specific intermediate layers matching pipeline (default: 9, 18, 27)
    hidden_states = output.hidden_states
    selected_layers = []
    for k in text_encoder_out_layers:
        if k < len(hidden_states):
            selected_layers.append(hidden_states[k])
        else:
            selected_layers.append(hidden_states[-1])

    dtype = text_encoder.dtype
    out = torch.stack(selected_layers, dim=1).to(dtype=dtype, device=device)  # [B, 3, L, D]
    B_out, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(B_out, seq_len, num_channels * hidden_dim)

    text_ids = _prepare_text_ids(prompt_embeds, device, dtype)

    return prompt_embeds, text_ids


# -------------------------
# Losses
# -------------------------

def compute_batch_losses(
    batch,
    args,
    accelerator,
    vae,
    text_encoder,
    tokenizer,
    transformer,
    noise_scheduler,
    weight_dtype,
):
    """Compute flow loss and mask-specific losses for a batch.

    VAE encoding/decoding matches Flux2KleinPipeline exactly:
      encode: vae.encode().mode() -> patchify -> batch_norm
      decode: batch_norm_denorm -> unpatchify -> vae.decode()
    """

    # --- Batch norm stats from VAE (Flux2 Klein uses BN on patchified latents) ---
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=accelerator.device, dtype=weight_dtype)
    bn_var = vae.bn.running_var.view(1, -1, 1, 1).to(device=accelerator.device, dtype=weight_dtype)
    bn_eps = getattr(vae.config, "batch_norm_eps", 1e-5)
    bn_std = torch.sqrt(bn_var + bn_eps)

    def encode_img(pixel_values):
        """Encode images matching Flux2KleinPipeline._encode_vae_image exactly.
        Returns patchified + BN-normalized latents: [B, C*4, H/2, W/2]
        """
        pixel_values = pixel_values.to(dtype=weight_dtype)
        # Use .mode() (argmax) matching pipeline's sample_mode="argmax"
        latents = vae.encode(pixel_values).latent_dist.mode()
        # Patchify: [B, C, H, W] -> [B, C*4, H/2, W/2]
        latents = _patchify_latents(latents)
        # Batch normalize (matches pipeline)
        latents = (latents - bn_mean) / bn_std
        return latents

    def decode_latents(patchified_latents):
        """Decode from BN-normalized patchified space matching pipeline's post-denoise path.
        Input: [B, C*4, H/2, W/2] in BN-normalized space
        Output: [B, 3, H_img, W_img] pixel space
        """
        # BN denormalize
        latents = patchified_latents * bn_std + bn_mean
        # Unpatchify: [B, C*4, H/2, W/2] -> [B, C, H, W]
        latents = _unpatchify_latents(latents)
        latents = latents.to(dtype=weight_dtype)
        return vae.decode(latents, return_dict=False)[0]

    # Encode all images - returns patchified + BN normalized [B, C*4, H/2, W/2]
    latents = encode_img(batch["pixel_values"])
    original_latents = encode_img(batch["original_image_pixel_values"])
    initial_mask_latents = encode_img(batch["initial_edit_region_mask_pixel_values"])

    # Null condition: encode a proper black image (not zeros in latent space)
    # In [-1,1] pixel space, black = -1.0
    black_pixels = torch.full_like(batch["pixel_values"], -1.0)
    null_cond_latents = encode_img(black_pixels)

    bsz, _, h_patch, w_patch = latents.shape

    # Pack latents (already patchified by encode_img, no separate patchify needed)
    packed_latents = _pack_latents(latents)
    packed_original = _pack_latents(original_latents)
    packed_initial = _pack_latents(initial_mask_latents)
    packed_null = _pack_latents(null_cond_latents)

    noise = torch.randn_like(packed_latents)
    bsz_packed = packed_latents.shape[0]

    u = compute_density_for_timestep_sampling(
        weighting_scheme=args.weighting_scheme,
        batch_size=bsz_packed,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        mode_scale=args.mode_scale,
    )
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    indices = indices.clamp(0, noise_scheduler.config.num_train_timesteps - 1)

    timesteps = noise_scheduler.timesteps[indices].to(accelerator.device)

    sigmas = noise_scheduler.sigmas[indices].to(accelerator.device)
    if sigmas.ndim == 1:
        sigmas = sigmas.view(-1, 1, 1)

    noisy_latents = (1.0 - sigmas) * packed_latents + sigmas * noise

    model_input = torch.cat([noisy_latents, packed_original, packed_initial, packed_null], dim=1)

    # Position IDs - latents already patchified, pass directly
    ids_noisy = _prepare_latent_ids(latents).to(device=accelerator.device, dtype=weight_dtype)
    cond_latents_list = [original_latents, initial_mask_latents, null_cond_latents]
    ids_cond = _prepare_image_ids(cond_latents_list).to(device=accelerator.device, dtype=weight_dtype)
    img_ids = torch.cat([ids_noisy, ids_cond], dim=1)

    prompt_embeds, text_ids = encode_prompt_qwen(
        tokenizer, text_encoder, batch["prompts"], args.max_sequence_length, accelerator.device
    )
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)

    # guidance=None for base (undistilled) model, matching pipeline
    model_pred = transformer(
        hidden_states=model_input,
        timestep=timesteps / 1000,
        guidance=None,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=img_ids,
        return_dict=False,
    )[0]

    model_pred_noise = model_pred[:, : packed_latents.shape[1], :]

    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
    target = noise - packed_latents
    flow_loss = torch.mean(
        (weighting.float() * (model_pred_noise.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    ).mean()

    # Reconstruct clean latents in packed space
    pred_packed = noisy_latents - sigmas * model_pred_noise

    t_rec_max = args.t_rec_max
    t_vals = sigmas.view(-1)
    rec_mask = t_vals < t_rec_max
    rec_idx = rec_mask.nonzero(as_tuple=False).view(-1)

    chan_consistency = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)

    if rec_idx.numel() > 0:
        keep_n = max(1, int(args.p_rec_fraction * rec_idx.numel()))
        if rec_idx.numel() > keep_n:
            rec_idx = rec_idx[:keep_n]

        # Select samples and unpack to patchified space: [n, seq, C] -> [n, C, h_p, w_p]
        n_sel = rec_idx.numel()
        pred_sel_packed = pred_packed[rec_idx]

        pred_sel_patch = pred_sel_packed.permute(0, 2, 1).reshape(n_sel, -1, h_patch, w_patch)

        # Decode through BN denorm + unpatchify + VAE decode
        pred_img = decode_latents(pred_sel_patch)
        pred_img = (pred_img + 1.0) / 2.0
        pred_img = torch.clamp(pred_img, 0.0, 1.0)

        # Channel consistency loss: encourages R=G=B (binary masks should be uniform across channels)
        c0 = pred_img[:, 0:1, ...]
        c1 = pred_img[:, 1:2, ...]
        c2 = pred_img[:, 2:3, ...]
        chan_consistency = (torch.abs(c0 - c1) + torch.abs(c1 - c2) + torch.abs(c0 - c2)).mean() / 3.0

    mask_loss = args.lambda_channel * chan_consistency

    total_loss = flow_loss + mask_loss

    return flow_loss, chan_consistency, total_loss


# -------------------------
# Validation / test
# -------------------------

def run_validation_or_test(
    transformer,
    scheduler,
    vae,
    text_encoder,
    tokenizer,
    args,
    accelerator,
    epoch,
    is_final_validation: bool,
    weight_dtype,
):
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return None

    split = "test" if is_final_validation else "validation"
    logger.info("Running %s evaluation...", split)

    eval_args = copy.copy(args)
    eval_args.random_flip = False

    try:
        dataset = KontextDataset(args=eval_args, split=split, repeats=1, center_crop=True)
    except Exception as e:
        logger.warning("No %s split available: %s", split, e)
        return None

    batch_size = args.test_batch_size if is_final_validation else args.val_batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    flow_losses = []
    chan_losses = []
    total_losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{split} loss", total=len(dataloader)):
            # Move batch tensors to accelerator device (dataloader is not prepared by accelerator)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(accelerator.device)
            flow_loss, chan_loss, total_loss = compute_batch_losses(
                batch,
                args,
                accelerator,
                vae,
                text_encoder,
                tokenizer,
                transformer,
                scheduler,
                weight_dtype,
            )
            flow_losses.append(flow_loss.item())
            chan_losses.append(chan_loss.item())
            total_losses.append(total_loss.item())

    mean_flow_loss = float(np.mean(flow_losses)) if flow_losses else float("nan")
    mean_chan_consistency = float(np.mean(chan_losses)) if chan_losses else float("nan")
    mean_total_loss = float(np.mean(total_losses)) if total_losses else float("nan")

    logger.info(
        "[%s] Losses: flow=%s chan=%s total=%s",
        split.upper(),
        mean_flow_loss,
        mean_chan_consistency,
        mean_total_loss,
    )

    # Metrics + visualization
    try:
        hf_dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=split,
            cache_dir=args.cache_dir,
        )
    except Exception as e:
        logger.warning("Unable to load %s split for metrics: %s", split, e)
        return {
            "mean_flow_loss": mean_flow_loss,
            "mean_chan_consistency": mean_chan_consistency,
            "mean_total_loss": mean_total_loss,
            "metrics": None,
        }

    max_samples = args.max_test_samples if is_final_validation else args.max_val_samples
    total_samples = getattr(hf_dataset, "num_rows", None)
    if total_samples is None:
        logger.warning("Dataset does not expose num_rows; skipping metrics/visualization.")
        return {
            "mean_flow_loss": mean_flow_loss,
            "mean_chan_consistency": mean_chan_consistency,
            "mean_total_loss": mean_total_loss,
            "metrics": None,
        }
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)

    viz_root = (
        os.path.join(args.output_dir, "test_visualization")
        if is_final_validation
        else os.path.join(args.output_dir, "validation_visualization")
    )
    if os.path.exists(viz_root):
        shutil.rmtree(viz_root, ignore_errors=True)
    os.makedirs(viz_root, exist_ok=True)

    # Use a deep copy of the scheduler so that pipeline inference (which calls
    # scheduler.set_timesteps(num_inference_steps)) does not corrupt the
    # training scheduler's sigmas/timesteps arrays.
    inference_scheduler = copy.deepcopy(scheduler)

    pipeline = Flux2KleinPipeline(
        scheduler=inference_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(transformer),
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    predicted_images = []
    ground_truth_images = []

    select_fn = getattr(hf_dataset, "select", None)
    if callable(select_fn):
        subset = select_fn(range(total_samples))
        sample_iter = cast(Any, subset)
    else:
        sample_iter = itertools.islice(hf_dataset, total_samples)

    for i, example in enumerate(tqdm(sample_iter, desc=f"{split} metrics", total=total_samples)):
        original = example[args.original_image_column].convert("RGB")
        init_mask = example[args.initial_edit_region_mask_column].convert("RGB")
        final_mask = example[args.final_edit_region_mask_column].convert("RGB")
        prompt = build_erm_prompt(example[args.edit_instruction_column])

        # Use the original image's native dimensions for pipeline generation.
        # This ensures predictions match input resolution (e.g. 640x480) instead
        # of being forced to training resolution (e.g. 256x256).
        orig_w, orig_h = original.size  # PIL: (width, height)

        # Ensure dimensions are divisible by 16 (VAE scale_factor=8 * patchify=2)
        gen_h = (orig_h // 16) * 16
        gen_w = (orig_w // 16) * 16

        # Resize condition images and ground truth to the generation dimensions
        original = original.resize((gen_w, gen_h), resample=Image.BILINEAR)
        init_mask = init_mask.resize((gen_w, gen_h), resample=Image.NEAREST)
        final_mask = final_mask.resize((gen_w, gen_h), resample=Image.NEAREST)

        blank = Image.new("RGB", (gen_w, gen_h), color=(0, 0, 0))
        cond_images = [original, init_mask, blank]

        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator = generator.manual_seed(args.seed + i)

        with torch.autocast(device_type="cuda", dtype=weight_dtype):
            pred = pipeline(
                prompt=prompt,
                image=cond_images,
                height=gen_h,
                width=gen_w,
                num_inference_steps=50,
                generator=generator,
                guidance_scale=args.guidance_scale,
            ).images[0]

        predicted_images.append(pred)
        ground_truth_images.append(final_mask)

        if i < args.eval_viz_num:
            pred.save(os.path.join(viz_root, f"{i}_pred.png"))
            final_mask.save(os.path.join(viz_root, f"{i}_gt.png"))
            original.save(os.path.join(viz_root, f"{i}_orig.png"))
            init_mask.save(os.path.join(viz_root, f"{i}_init_mask.png"))
            with open(os.path.join(viz_root, f"{i}_prompt.txt"), "w") as pf:
                pf.write(prompt)

    metrics = compute_mask_metrics(predicted_images, ground_truth_images, is_final_validation)

    # Print summary of losses and metrics
    logger.info("=" * 60)
    logger.info("[%s] EVALUATION SUMMARY (epoch %d)", split.upper(), epoch)
    logger.info("[%s] Losses   — flow=%.6f  chan=%.6f  total=%.6f", split.upper(), mean_flow_loss, mean_chan_consistency, mean_total_loss)
    if metrics and "aggregated" in metrics:
        for mk, mv in metrics["aggregated"].items():
            logger.info("[%s] Metric   — %s = %.6f", split.upper(), mk, mv)
    logger.info("=" * 60)

    accelerator.wait_for_everyone()

    return {
        "mean_flow_loss": mean_flow_loss,
        "mean_chan_consistency": mean_chan_consistency,
        "mean_total_loss": mean_total_loss,
        "metrics": metrics,
    }


# -------------------------
# Main
# -------------------------


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb to use wandb.")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    logger.info(accelerator.state)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.force_fp32:
        weight_dtype = torch.float32

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        token=hf_token,
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
        token=hf_token,
    )

    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        token=hf_token,
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
        token=hf_token,
    )

    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        token=hf_token,
    )

    if args.gradient_checkpointing:
        enable_gc = getattr(transformer, "enable_gradient_checkpointing", None)
        if callable(enable_gc):
            enable_gc()

    num_train_timesteps = None
    if hasattr(noise_scheduler, "config"):
        cfg = noise_scheduler.config
        num_train_timesteps = getattr(cfg, "num_train_timesteps", None)
        if num_train_timesteps is None and isinstance(cfg, dict):
            num_train_timesteps = cfg.get("num_train_timesteps")
    if num_train_timesteps is None:
        num_train_timesteps = 1000

    if noise_scheduler.sigmas is None or noise_scheduler.timesteps is None:
        noise_scheduler.set_timesteps(num_train_timesteps, device=accelerator.device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    if args.lora_layers:
        target_modules = [x.strip() for x in args.lora_layers.split(",")]
    else:
        target_modules = ["attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0", "proj_mlp"]

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    # --- Helper to unwrap compiled/DDP models ---
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # --- LoRA-only save/load hooks ---
    # These hooks intercept accelerator.save_state() / load_state() so that
    # only LoRA adapter weights are serialised — NOT the full 9B base model.
    # This reduces checkpoint size from ~36-50 GiB to ~100-200 MB.

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    lora_state = get_peft_model_state_dict(model)
                    # Save LoRA weights in safetensors format
                    from safetensors.torch import save_file as st_save
                    st_save(lora_state, os.path.join(output_dir, "transformer_lora.safetensors"))
                else:
                    logger.warning("Unexpected model type in save_model_hook: %s", type(model))
                # Pop weight so accelerator does NOT save the full model
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                lora_path = os.path.join(input_dir, "transformer_lora.safetensors")
                if os.path.exists(lora_path):
                    from safetensors.torch import load_file as st_load
                    lora_state = st_load(lora_path, device="cpu")
                    incompatible = set_peft_model_state_dict(model, lora_state, adapter_name="default")
                    if incompatible is not None:
                        unexpected = getattr(incompatible, "unexpected_keys", None)
                        if unexpected:
                            logger.warning("Unexpected keys loading LoRA: %s", unexpected)
                else:
                    logger.warning("No LoRA weights found at %s", lora_path)
            else:
                logger.warning("Unexpected model type in load_model_hook: %s", type(model))

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(params_to_optimize, lr=args.learning_rate)
        except Exception:
            logger.warning("bitsandbytes not available; falling back to AdamW")
            optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate)

    # Create dataset & dataloader first so we can compute total training steps
    train_dataset = KontextDataset(args=args, split="train", repeats=args.repeats)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # --- Epoch setting always takes precedence over max_train_steps ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    logger.info(
        "Training for %d epochs (%d update steps/epoch, %d total steps)",
        args.num_train_epochs, num_update_steps_per_epoch, args.max_train_steps,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Recompute after accelerator.prepare (distributed setups may change dataloader length)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    global_step = 0
    starting_epoch = 0

    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        if resume_path == "latest":
            latest_ckpt = find_latest_checkpoint(args.output_dir)
            if latest_ckpt is None:
                logger.warning("No checkpoint found to resume from; starting fresh.")
                resume_path = None
            else:
                resume_path = os.path.join(args.output_dir, latest_ckpt)

        if resume_path:
            logger.info("Resuming from checkpoint: %s", resume_path)
            accelerator.load_state(resume_path)
            state_path = os.path.join(resume_path, "training_state.json")
            if os.path.exists(state_path):
                with open(state_path, "r") as f:
                    state = json.load(f)
                starting_epoch = int(state.get("epoch", 0))
                global_step = int(state.get("global_step", starting_epoch * num_update_steps_per_epoch))
            else:
                ckpt_name = os.path.basename(resume_path)
                parts = ckpt_name.split("-")
                epoch_num = 0
                for p in reversed(parts):
                    if p.isdigit():
                        epoch_num = int(p)
                        break
                starting_epoch = epoch_num
                global_step = starting_epoch * num_update_steps_per_epoch

    if starting_epoch > 0:
        logger.info(
            "Resuming training from epoch %d/%d (global_step=%d). Epochs remaining: %d",
            starting_epoch + 1, args.num_train_epochs, global_step,
            args.num_train_epochs - starting_epoch,
        )

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )

    # --- Pre-training validation (before any training) ---
    if getattr(args, "run_pre_training_validation", False) and starting_epoch == 0:
        logger.info("Running pre-training validation (epoch 0)...")
        pre_val_result = run_validation_or_test(
            transformer,
            noise_scheduler,
            vae,
            text_encoder,
            tokenizer,
            args,
            accelerator,
            0,
            is_final_validation=False,
            weight_dtype=weight_dtype,
        )
        if accelerator.is_main_process:
            print("###PRE_TRAINING_VALIDATION_COMPLETE###")
        transformer.train()

    for epoch in range(starting_epoch, args.num_train_epochs):
        transformer.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        logger.info("Starting epoch %d/%d (global_step=%d)", epoch + 1, args.num_train_epochs, global_step)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                flow_loss, chan_loss, loss = compute_batch_losses(
                    batch,
                    args,
                    accelerator,
                    vae,
                    text_encoder,
                    tokenizer,
                    transformer,
                    noise_scheduler,
                    weight_dtype,
                )

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            flow_loss_val = flow_loss.detach().item()
            chan_loss_val = chan_loss.detach().item()
            loss_val = loss.detach().item()
            epoch_loss_sum += loss_val
            epoch_steps += 1

            if epoch_steps % 10 == 1 or epoch_steps <= 5:
                logger.info(
                    "  [Step %d] flow=%.6f  chan=%.6f  total=%.6f",
                    epoch_steps, flow_loss_val, chan_loss_val, loss_val,
                )

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

        mean_epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
        logger.info(
            "Epoch %d/%d completed. Mean train loss: %.6f (global_step=%d)",
            epoch + 1, args.num_train_epochs, mean_epoch_loss, global_step,
        )

        if accelerator.is_main_process:
            if (epoch + 1) % args.checkpointing_epochs == 0:
                save_checkpoint(
                    accelerator,
                    args.output_dir,
                    epoch + 1,
                    global_step,
                    args.checkpoints_total_limit,
                )

        if (epoch + 1) % args.validation_epochs == 0:
            _ = run_validation_or_test(
                transformer,
                noise_scheduler,
                vae,
                text_encoder,
                tokenizer,
                args,
                accelerator,
                epoch + 1,
                is_final_validation=False,
                weight_dtype=weight_dtype,
            )
            if accelerator.is_main_process:
                print(f"###VALIDATION_COMPLETE### epoch={epoch + 1}")

    _ = run_validation_or_test(
        transformer,
        noise_scheduler,
        vae,
        text_encoder,
        tokenizer,
        args,
        accelerator,
        args.num_train_epochs,
        is_final_validation=True,
        weight_dtype=weight_dtype,
    )
    if accelerator.is_main_process:
        print("###TEST_COMPLETE###")

    # --- Save standalone LoRA weights at end of training ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_transformer = unwrap_model(transformer)
        unwrapped_transformer = unwrapped_transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(unwrapped_transformer)
        from safetensors.torch import save_file as st_save_final
        lora_save_path = os.path.join(args.output_dir, "transformer_lora_final.safetensors")
        st_save_final(transformer_lora_layers, lora_save_path)
        logger.info("Saved final LoRA weights to %s", lora_save_path)

        # Memory cleanup
        del unwrapped_transformer, transformer_lora_layers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accelerator.end_training()


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.exists(output_dir):
        return None
    dirs = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not dirs:
        return None

    def _sort_key(name: str) -> int:
        parts = name.split("-")
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return 0

    dirs = sorted(dirs, key=_sort_key)
    return dirs[-1]


def prune_checkpoints(output_dir: str, limit: Optional[int]) -> None:
    if limit is None or limit <= 0:
        return
    checkpoints = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint") and os.path.isdir(os.path.join(output_dir, d))
    ]

    def _sort_key(name: str) -> int:
        parts = name.split("-")
        for p in reversed(parts):
            if p.isdigit():
                return int(p)
        return 0

    checkpoints = sorted(checkpoints, key=_sort_key)
    excess = len(checkpoints) - limit
    if excess <= 0:
        return

    for ckpt in checkpoints[:excess]:
        path = os.path.join(output_dir, ckpt)
        shutil.rmtree(path, ignore_errors=True)


def save_checkpoint(
    accelerator: Accelerator,
    output_dir: str,
    epoch: int,
    global_step: int,
    checkpoints_total_limit: Optional[int],
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
    accelerator.save_state(save_path)
    state_path = os.path.join(save_path, "training_state.json")
    with open(state_path, "w") as f:
        json.dump({"epoch": epoch, "global_step": global_step}, f)
    prune_checkpoints(output_dir, checkpoints_total_limit)
    print(f"###CHECKPOINT_SAVED### epoch={epoch} path={save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="flux-2-klein-erm")
    parser.add_argument("--edited_image_column", type=str, default="final_edit_region_mask")
    parser.add_argument("--original_image_column", type=str, default="original_image")
    parser.add_argument("--initial_edit_region_mask_column", type=str, default="initial_edit_region_mask")
    parser.add_argument("--final_edit_region_mask_column", type=str, default="final_edit_region_mask")
    parser.add_argument("--edit_instruction_column", type=str, default="edit_instruction")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpointing_epochs", type=int, default=1)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_layers", type=str, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--weighting_scheme", type=str, default="none")
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--validation_epochs", type=int, default=1)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--max_val_samples", type=int, default=25)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--eval_viz_num", type=int, default=10)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--allow_tf32", action="store_true")

    parser.add_argument("--debug_cuda", action="store_true")
    parser.add_argument("--force_fp32", action="store_true")
    parser.add_argument("--disable_8bit", action="store_true")
    parser.add_argument("--disable_tf32", action="store_true")

    parser.add_argument("--t_rec_max", type=float, default=0.35)
    parser.add_argument("--p_rec_fraction", type=float, default=1.0)
    parser.add_argument("--lambda_channel", type=float, default=25.0)
    parser.add_argument("--run_pre_training_validation", action="store_true")

    args = parser.parse_args()

    if args.debug_cuda:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if args.disable_8bit:
        args.use_8bit_adam = False

    main(args)
