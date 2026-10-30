"""
CALVIN data for the learned Initial Edit Region (IER) mask predictor.

Reads ``(original_image, edit_instruction, initial_edit_region_mask)`` from the
Hugging Face parquet shards (``RiddhiCh/Calvin-Dataset-Final``). The target
``M_i`` is the union of the robot-arm mask and the masks of the objects that
move during the step (paper §3.1), so the model learns to anticipate it from
``(I_o, T)`` alone.

No geometric augmentation: instructions refer to left/right and absolute
positions, so flips or large crops would change the label's meaning.
"""

import glob
import io
import os
import random

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

COLUMNS = ["id", "original_image", "initial_edit_region_mask", "edit_instruction"]


def resolve_parquets(spec):
    """Expand a comma-separated list of paths or glob patterns into sorted files."""
    files = []
    for part in (p for p in spec.split(",") if p):
        matched = sorted(glob.glob(os.path.expanduser(part)))
        if not matched:
            raise FileNotFoundError(f"No parquet files match: {part}")
        files.extend(matched)
    return files


def photometric_jitter(img, strength=0.2):
    """Random brightness / contrast / saturation in [1 - strength, 1 + strength]."""
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enhancer(img).enhance(random.uniform(1 - strength, 1 + strength))
    return img


class IerParquetDataset(Dataset):
    """One item per tuple: uint8 image [3, H, W], binary mask [H, W], instruction, id."""

    def __init__(self, parquet_files, augment=False, max_samples=None, seed=0):
        self.ids, self.images, self.masks, self.instructions = [], [], [], []
        for path in parquet_files:
            table = pq.read_table(path, columns=COLUMNS)
            self.ids += table["id"].to_pylist()
            self.images += [cell["bytes"] for cell in table["original_image"].to_pylist()]
            self.masks += [cell["bytes"] for cell in table["initial_edit_region_mask"].to_pylist()]
            self.instructions += [s.strip() for s in table["edit_instruction"].to_pylist()]
        if max_samples is not None and max_samples < len(self.ids):
            keep = sorted(random.Random(seed).sample(range(len(self.ids)), max_samples))
            self.ids, self.images, self.masks, self.instructions = (
                [lst[i] for i in keep] for lst in (self.ids, self.images, self.masks, self.instructions)
            )
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.images[idx])).convert("RGB")
        mask = Image.open(io.BytesIO(self.masks[idx])).convert("L")
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.NEAREST)
        if self.augment:
            img = photometric_jitter(img)
        return {
            "image": torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1),
            "mask": torch.from_numpy((np.asarray(mask) > 127).astype(np.float32)),
            "instruction": self.instructions[idx],
            "id": self.ids[idx],
        }


def collate(batch):
    sizes = {tuple(b["image"].shape) for b in batch}
    if len(sizes) != 1:
        raise ValueError(f"Mixed image sizes in one batch: {sizes}")
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "instruction": [b["instruction"] for b in batch],
        "id": [b["id"] for b in batch],
    }
