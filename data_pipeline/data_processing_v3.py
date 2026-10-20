"""
Command to run the file (lgie_dp conda env): PYTHONPATH=./XMem python3 data_processing_v3.py
"""

# ---------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------
import argparse
import ast
import base64
import concurrent.futures
import io
import json
import math
import colorsys
import multiprocessing
import os
import random
import re
import shutil
import sys
import tempfile
import textwrap
import time
import shelve
import hashlib
import gc
import threading

# Ensure local segment-anything and XMem are used
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SAM_PATH = os.path.join(_BASE_DIR, "segment-anything")
_XMEM_PATH = os.path.join(_BASE_DIR, "XMem")

for _p in [_SAM_PATH, _XMEM_PATH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from math import ceil
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from os import path
from typing import List, Optional

# ---------------------------------------------------------------------
# Scientific / Core Libraries
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Image and Visualization
# ---------------------------------------------------------------------
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, Image as PILImage

# ---------------------------------------------------------------------
# Progress, Sorting, and Utilities
# ---------------------------------------------------------------------
from natsort import natsorted
from progressbar import progressbar
from tqdm import tqdm

# ---------------------------------------------------------------------
# Corrupt Image Logging
# ---------------------------------------------------------------------
CORRUPT_IMAGES_FILE = "corrupt_images.txt"
corrupt_images_lock = threading.Lock()

def log_corrupt_image(image_path):
    with corrupt_images_lock:
        with open(CORRUPT_IMAGES_FILE, "a") as f:
            f.write(image_path + "\n")

# ---------------------------------------------------------------------
# Transformers / Hugging Face Core
# ---------------------------------------------------------------------
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    Blip2ForConditionalGeneration,
    Blip2Processor
)

# ---------------------------------------------------------------------
# Hugging Face Datasets and Hub
# ---------------------------------------------------------------------
import datasets
from datasets import Dataset, DatasetDict, Features, Value, Image as DatasetsImage
from datasets import load_from_disk, concatenate_datasets
from huggingface_hub import HfApi
import huggingface_hub

# ---------------------------------------------------------------------
# XMem / Project-Specific Imports
# ---------------------------------------------------------------------
from inference.data.mask_mapper import MaskMapper
from inference.data.test_datasets import (
    LongTestDataset,
    DAVISTestDataset,
    YouTubeVOSTestDataset
)
from inference.inference_core import InferenceCore
from model.network import XMem

# ---------------------------------------------------------------------
# SAM (Segment Anything Model)
# ---------------------------------------------------------------------
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ---------------------------------------------------------------------
# Clustering (KMeans)
# ---------------------------------------------------------------------
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------
# Torchvision / Preprocessing
# ---------------------------------------------------------------------
from torchvision import transforms
from torchvision.transforms.functional import (
    InterpolationMode,
    resize,
    to_pil_image
)
import torchvision.transforms.functional as TF
try:
    from torchvision.io import read_image, ImageReadMode
    _HAS_TVIO = True
except Exception:
    read_image = None
    ImageReadMode = None
    _HAS_TVIO = False

# ---------------------------------------------------------------------
# Misc / Helpers
# ---------------------------------------------------------------------
import warnings
import traceback


def _imread_rgb(path_str: str) -> Optional[np.ndarray]:
    """Read image as RGB uint8 HWC.

    Prefers torchvision.io (often libjpeg-turbo backed) when available.
    Falls back to OpenCV otherwise.
    """
    use_tvio = os.environ.get("DP_USE_TORCHVISION_IO", "1") == "1"
    if _HAS_TVIO and use_tvio and read_image is not None and ImageReadMode is not None:
        try:
            t = read_image(path_str, mode=ImageReadMode.RGB)  # uint8, CHW
            return t.permute(1, 2, 0).contiguous().numpy()
        except Exception:
            # fall back to cv2
            pass

    img_bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if img_bgr is None:
        if os.path.exists(path_str):
            log_corrupt_image(path_str)
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _imread_gray(path_str: str) -> Optional[np.ndarray]:
    """Read image as GRAY uint8 HW."""
    use_tvio = os.environ.get("DP_USE_TORCHVISION_IO", "1") == "1"
    if _HAS_TVIO and use_tvio and read_image is not None and ImageReadMode is not None:
        try:
            t = read_image(path_str, mode=ImageReadMode.GRAY)  # uint8, 1HW
            return t[0].contiguous().numpy()
        except Exception:
            pass

    img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    if img is None:
        if os.path.exists(path_str):
            log_corrupt_image(path_str)
    return img

def _imread_tensor(path_str: str, device="cuda") -> Optional[torch.Tensor]:
    """Read image as RGB uint8 CHW tensor on device."""
    use_tvio = os.environ.get("DP_USE_TORCHVISION_IO", "1") == "1"
    if _HAS_TVIO and use_tvio and read_image is not None and ImageReadMode is not None:
        try:
            t = read_image(path_str, mode=ImageReadMode.RGB)  # uint8, CHW
            return t.to(device=device)
        except Exception:
            pass

    img_bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if img_bgr is None:
        if os.path.exists(path_str):
            log_corrupt_image(path_str)
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img_rgb).permute(2, 0, 1).to(device=device)

# ---------------------------------------------------------------------
# Runtime / Performance Knobs
# ---------------------------------------------------------------------

def _configure_runtime_threads():
    """Configure CPU threading to balance throughput vs contention.

    Why this matters:
      - This script uses Python thread pools + OpenCV + PyTorch.
      - If OpenCV/PyTorch also spawn many internal threads, CPU oversubscription
        can make *more* Python workers slower (typical reason 4 workers < 2 workers).

    Defaults are conservative (good for GPU-bound inference).
    To intentionally use more CPU cores for decode/resize/etc, enable multi-core mode.

    Environment knobs:
      - DP_MULTICORE=1                Enable multi-core CPU usage (default: 0)
      - DP_DISABLE_OPENCV_THREADS=1   Force-disable OpenCV threads (default: 1 when DP_MULTICORE=0)
      - DP_OPENCV_NUM_THREADS=N       OpenCV threads when enabled (default: auto)
      - DP_TORCH_NUM_THREADS=N        PyTorch intra-op threads (default: 1 when DP_MULTICORE=0; auto when DP_MULTICORE=1)
      - DP_TORCH_NUM_INTEROP_THREADS=N PyTorch inter-op threads (default: 1 when DP_MULTICORE=0; 1-2 when DP_MULTICORE=1)

    Worker hints (used only for auto-scaling in DP_MULTICORE=1):
      - DP_EVF_WORKERS, DP_XMEM_NUM_WORKERS
    """

    cpu_count = os.cpu_count() or 1
    multicore = os.environ.get("DP_MULTICORE", "0") == "1"

    # Heuristic: if user runs multiple Python workers, scale internal threading down.
    # This helps use *multiple cores* without oversubscribing.
    try:
        worker_hint = 1
        worker_hint = max(worker_hint, int(os.environ.get("DP_EVF_WORKERS", "2")))
        worker_hint = max(worker_hint, int(os.environ.get("DP_XMEM_NUM_WORKERS", "2")))
        worker_hint = max(1, worker_hint)
    except Exception:
        worker_hint = 1

    # ---------------- OpenCV threads ----------------
    try:
        disable_cv_threads_default = "0" if multicore else "1"
        disable_cv_threads = os.environ.get("DP_DISABLE_OPENCV_THREADS", disable_cv_threads_default) != "0"
        if disable_cv_threads:
            cv2.setNumThreads(0)
        else:
            # Auto: spread cores across worker threads; cap to avoid runaway oversubscription
            auto_cv = max(1, min(8, cpu_count // worker_hint))
            cv_threads = int(os.environ.get("DP_OPENCV_NUM_THREADS", str(auto_cv)))
            cv2.setNumThreads(max(1, cv_threads))

        # OpenCL often adds overhead for this workload
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    # ---------------- PyTorch CPU threads ----------------
    try:
        if multicore:
            auto_torch = max(1, min(16, cpu_count // worker_hint))
            auto_interop = 1 if auto_torch <= 4 else 2
            torch_threads = int(os.environ.get("DP_TORCH_NUM_THREADS", str(auto_torch)))
            interop_threads = int(os.environ.get("DP_TORCH_NUM_INTEROP_THREADS", str(auto_interop)))
        else:
            torch_threads = int(os.environ.get("DP_TORCH_NUM_THREADS", "1"))
            interop_threads = int(os.environ.get("DP_TORCH_NUM_INTEROP_THREADS", "1"))

        torch.set_num_threads(max(1, torch_threads))
        torch.set_num_interop_threads(max(1, interop_threads))
    except Exception:
        pass

    # CUDNN autotune can improve throughput when shapes are somewhat stable
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def _configure_gpu_perf():
    """Optional GPU perf knobs.

    Environment knobs:
      - DP_TF32=1   Enable TF32 (default: 0)
      - DP_MATMUL_PRECISION=high|medium|highest (default: high when TF32 enabled)
    """
    if not torch.cuda.is_available():
        return

    enable_tf32 = os.environ.get("DP_TF32", "0") == "1"
    if enable_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        # PyTorch 2.x matmul precision hint
        try:
            prec = os.environ.get("DP_MATMUL_PRECISION", "high")
            torch.set_float32_matmul_precision(prec)
        except Exception:
            pass


_configure_runtime_threads()
_configure_gpu_perf()

# ----------------- Consolidated Argument Parsing -----------------
parser = ArgumentParser()
parser.add_argument('--dataset_name', default="Test_Data")
parser.add_argument('--batch_size', type=int, default=None, help="Batch size for processing episodes")
parser.add_argument('--resume', type=int, default=1, help="Batch index to resume from (1-indexed)")
parser.add_argument('--hf_repo_id', default="SuyashAjitChavan2002/LGRM_BridgeDataV2_Teleoperated")
parser.add_argument('--hf_token', default=os.environ.get("HF_TOKEN"), help="Defaults to $HF_TOKEN")
parser.add_argument('--cleanup', action='store_true', default=True, help="Cleanup batch files after processing")

# XMem options
parser.add_argument('--xmem_model', default=os.path.join(os.path.dirname(__file__), "XMem/saves/XMem.pth"))
parser.add_argument('--size', default=480, type=int, help='Resize the shorter side to this size. -1 to use original resolution.')
parser.add_argument('--disable_long_term', action='store_true')
parser.add_argument('--max_mid_term_frames', type=int, default=10)
parser.add_argument('--min_mid_term_frames', type=int, default=5)
parser.add_argument('--max_long_term_elements', type=int, default=10000)
parser.add_argument('--num_prototypes', type=int, default=128)
parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--mem_every', type=int, default=5)
parser.add_argument('--deep_update_every', type=int, default=-1)
parser.add_argument('--save_scores', action='store_true')
parser.add_argument('--flip', action='store_true')
parser.add_argument('--benchmark', action='store_true')
parser.add_argument('--save_all', action='store_true', default=True)

# LoFTR options
parser.add_argument('--weight_type', default="outdoor_ot", choices=["indoor_ds_new", "indoor_ot", "outdoor_ds", "outdoor_ot"])

args = parser.parse_args()
config = vars(args)
config['enable_long_term'] = not config['disable_long_term']

# ---------------------------- Define All Directories here ---------------------------------------
BASE_DIR = os.path.dirname(__file__)
dataset_name = args.dataset_name
DATA_DIR = os.path.join(BASE_DIR, dataset_name)
IMAGE_DIR = os.path.join(DATA_DIR, "JPEGImages")
ROBOTICMASK_DIR = os.path.join(DATA_DIR, "RoboticArmMasks")
ANNOTATION_DIR = os.path.join(DATA_DIR, "Annotations")
EDIT_INSTRUCTION_DIR = os.path.join(DATA_DIR, "EditInstructions")

XMEM_MODEL_PATH = args.xmem_model
XMEM_OUTPUT_DIR = os.path.join(DATA_DIR, "XMem_Output")

SAM_CHECKPOINT = os.path.join(BASE_DIR, "sam_vit_h_4b8939.pth")

EDIT_MASKS_BASE = os.path.join(BASE_DIR, "edit_masks")
EDIT_MASKS_DIR = os.path.join(EDIT_MASKS_BASE, dataset_name)

LOFTR_WEIGHTS_DIR = os.path.join(BASE_DIR, "loftr_weights")

HF_REPO_ID = args.hf_repo_id
HF_TOKEN = args.hf_token

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------- Color Configurations  ---------------------------------------
# 256 commonly used named colors
COLOR_PALETTE = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85), (255, 51, 51), (255, 102, 51),
    (255, 153, 51), (255, 204, 51), (204, 255, 51), (153, 255, 51), (102, 255, 51),
    (51, 255, 51), (51, 255, 102), (51, 255, 153), (51, 255, 204), (51, 204, 255),
    (51, 153, 255), (51, 102, 255), (51, 51, 255), (102, 51, 255), (153, 51, 255),
    (204, 51, 255), (255, 51, 204), (255, 51, 153), (255, 51, 102), (255, 102, 102),
    (255, 153, 102), (255, 204, 102), (204, 255, 102), (153, 255, 102), (102, 255, 102),
    (102, 255, 153), (102, 255, 204), (102, 204, 255), (102, 153, 255), (102, 102, 255),
    (153, 102, 255), (204, 102, 255), (255, 102, 204), (255, 102, 153), (255, 153, 153),
    (255, 204, 153), (204, 255, 153), (153, 255, 153), (153, 255, 204), (153, 204, 255),
    (153, 153, 255), (204, 153, 255), (255, 153, 204), (255, 204, 204), (204, 255, 204),
    (204, 204, 255), (255, 204, 255), (255, 64, 0), (255, 128, 0), (255, 191, 0),
    (223, 255, 0), (159, 255, 0), (96, 255, 0), (32, 255, 0), (0, 255, 32),
    (0, 255, 96), (0, 255, 159), (0, 255, 223), (0, 191, 255), (0, 128, 255),
    (0, 64, 255), (32, 0, 255), (96, 0, 255), (159, 0, 255), (223, 0, 255),
    (255, 0, 223), (255, 0, 159), (255, 0, 96), (255, 0, 32), (255, 64, 64),
    (255, 128, 64), (255, 191, 64), (223, 255, 64), (159, 255, 64), (96, 255, 64),
    (32, 255, 64), (64, 255, 96), (64, 255, 159), (64, 255, 223), (64, 223, 255),
    (64, 159, 255), (64, 96, 255), (64, 32, 255), (96, 64, 255), (159, 64, 255),
    (223, 64, 255), (255, 64, 223), (255, 64, 159), (255, 64, 96), (255, 128, 128),
    (255, 191, 128), (223, 255, 128), (159, 255, 128), (96, 255, 128), (128, 255, 159),
    (128, 255, 223), (128, 223, 255), (128, 159, 255), (128, 96, 255), (128, 32, 255),
    (191, 128, 255), (223, 128, 255), (255, 128, 223), (255, 128, 159), (255, 159, 159),
    (255, 223, 159), (191, 255, 159), (159, 255, 191), (159, 255, 255), (159, 191, 255),
    (159, 159, 255), (191, 159, 255), (223, 159, 255), (255, 159, 223), (255, 191, 191),
    (223, 255, 191), (191, 255, 223), (191, 223, 255), (191, 191, 255), (223, 191, 255),
    (255, 191, 223), (255, 32, 64), (255, 64, 128), (255, 96, 191), (255, 128, 255),
    (191, 128, 255), (128, 128, 255), (64, 128, 255), (0, 128, 255), (0, 191, 191),
    (0, 255, 128), (0, 255, 64), (0, 255, 0), (64, 255, 0), (128, 255, 0),
    (191, 255, 0), (255, 255, 0), (255, 191, 0), (255, 128, 0), (255, 64, 0),
    (255, 0, 0), (255, 0, 64), (255, 0, 128), (255, 0, 191), (255, 0, 255),
    (191, 0, 255), (128, 0, 255), (64, 0, 255), (0, 0, 255), (0, 64, 255),
    (0, 128, 255), (0, 191, 255), (0, 255, 255), (0, 255, 191), (0, 255, 128),
    (0, 255, 64), (0, 255, 0), (64, 255, 0), (128, 255, 0), (191, 255, 0),
    (255, 255, 0), (255, 191, 0), (255, 128, 0), (255, 64, 0), (255, 0, 0),
    (255, 64, 64), (255, 128, 128), (255, 191, 191), (191, 255, 255),
    (128, 255, 255), (64, 255, 255), (0, 255, 255), (0, 255, 191), (0, 255, 128),
    (0, 255, 64), (0, 255, 0), (64, 255, 0), (128, 255, 0), (191, 255, 0),
    (255, 255, 0), (255, 191, 0), (255, 128, 0), (255, 64, 0), (255, 0, 0),
    (191, 0, 0), (128, 0, 0), (64, 0, 0)
]
# print(len(COLOR_PALETTE))

def generate_color_palette(num_colors=256):
    """Generate a list of evenly spaced RGB colors excluding white and black."""
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        lightness = 0.5 + 0.4 * ((i % 2) - 0.5)  # alternate lightness a bit for contrast
        saturation = 0.9
        r, g, b = [int(x * 255) for x in colorsys.hls_to_rgb(hue, lightness, saturation)]
        # skip black and white
        if (r, g, b) not in [(0, 0, 0), (255, 255, 255)]:
            colors.append((r, g, b))

    # Flatten to a 768-length list (256 * 3), padded with zeros if needed
    flat_palette = [v for c in colors[:256] for v in c]
    flat_palette.extend([0] * (768 - len(flat_palette)))
    return flat_palette

# -------------------------- EVF-SAM Inference for Robotic Arm Mask -------------------------------------------
EVF_SAM_VERSION = "YxZhang/evf-sam"
EVF_SAM_PRECISION="fp16" 
EVF_SAM_MODEL_TYPE = "ori"
BLIP_MODEL = "Salesforce/blip2-flan-t5-xl"
ROBOT_OBJECT_DESCRIPTION = {
    "evfsam": "robotic arm and gripper",
    "blip": "robotic gripper"
}
confidence_threshold = 0.0

def init_models():
    tokenizer = AutoTokenizer.from_pretrained(
        EVF_SAM_VERSION,
        padding_side="right",
        use_fast=False,
    )

    # Precision handling
    torch_dtype = torch.float32
    if EVF_SAM_PRECISION == "bf16":
        torch_dtype = torch.bfloat16
    elif EVF_SAM_PRECISION == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}

    # Model selection
    if EVF_SAM_MODEL_TYPE == "ori":
        from evf_sam_model.evf_sam import EvfSamModel
        model = EvfSamModel.from_pretrained(EVF_SAM_VERSION, low_cpu_mem_usage=True, **kwargs)

    elif EVF_SAM_MODEL_TYPE == "effi":
        from evf_sam_model.evf_effisam import EvfEffiSamModel
        model = EvfEffiSamModel.from_pretrained(EVF_SAM_VERSION, low_cpu_mem_usage=True, **kwargs)

    elif EVF_SAM_MODEL_TYPE == "sam2":
        from evf_sam_model.evf_sam2 import EvfSam2Model
        model = EvfSam2Model.from_pretrained(EVF_SAM_VERSION, low_cpu_mem_usage=True, **kwargs)

    else:
        raise ValueError(f"Unknown EVF_SAM_MODEL_TYPE: {EVF_SAM_MODEL_TYPE}")

    model = model.cuda()
    model.eval()
    return tokenizer, model

def sam_preprocess_gpu(x_tensor, pixel_mean, pixel_std, img_size=1024, model_type="ori"):
    # x_tensor: (C, H, W), float, 0-255
    # pixel_mean/std should be on same device
    
    if model_type == "ori":
        h, w = x_tensor.shape[-2:]
        scale = img_size * 1.0 / max(h, w)
        newh, neww = int(h * scale + 0.5), int(w * scale + 0.5)
        
        x = F.interpolate(x_tensor.unsqueeze(0), size=(newh, neww), mode='bilinear', align_corners=False, antialias=True).squeeze(0)
        x = (x - pixel_mean.to(x.device)) / pixel_std.to(x.device)
        
        pad_h = img_size - newh
        pad_w = img_size - neww
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (newh, neww)
    else:
        # Fallback or implementation for other types if needed
        # For now assuming 'ori' as per original code
        return x_tensor, (x_tensor.shape[-2], x_tensor.shape[-1])

def beit3_preprocess_gpu(x_tensor, img_size=224):
    # x_tensor: (C, H, W), float, 0-255
    x = x_tensor / 255.0
    x = F.interpolate(x.unsqueeze(0), size=(img_size, img_size), mode='bicubic', align_corners=False, antialias=True).squeeze(0)
    x = (x - 0.5) / 0.5
    return x

class ResizeLongestSide:
    def __init__(self, target_length: int):
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        return np.array(resize(to_pil_image(image), target_size, antialias=None))

    def get_preprocess_shape(self, oldh, oldw, long_side_length):
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        return int(newh + 0.5), int(neww + 0.5)

def sam_preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
                   pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
                   img_size=1024, model_type="ori"):

    if model_type == "ori":
        x = ResizeLongestSide(img_size).apply_image(x)
        h, w = resize_shape = x.shape[:2]
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        x = (x - pixel_mean) / pixel_std
        x = F.pad(x, (0, img_size - w, 0, img_size - h))
    else:
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        x = F.interpolate(x.unsqueeze(0), (img_size, img_size),
                          mode="bilinear", align_corners=False).squeeze(0)
        x = (x - pixel_mean) / pixel_std
        resize_shape = None

    return x, resize_shape

def beit3_preprocess(x, img_size=224):
    beit_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size),
                          interpolation=InterpolationMode.BICUBIC, antialias=None),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])
    return beit_transform(x)

# --------------------------------------------------------
# Initialize BLIP-2 in fp16
# --------------------------------------------------------
def init_blip(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        processor = Blip2Processor.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True
        )
    except Exception:
        processor = Blip2Processor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return processor, model, device

# ------------------------------
# Helper: crop masked region and remove background
# ------------------------------
def crop_from_mask(image_rgb: np.ndarray, mask: np.ndarray) -> Optional[Image.Image]:
    """
    Crop bounding box around mask and zero-out background pixels inside the crop.
    Returns a PIL RGB image (object only, background black) or None if mask empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if x2 < x1 or y2 < y1:
        return None

    crop_img = image_rgb[y1:y2 + 1, x1:x2 + 1].copy()
    crop_mask = mask[y1:y2 + 1, x1:x2 + 1]
    crop_img[crop_mask == 0] = 0

    return Image.fromarray(crop_img)

def crop_from_mask_gpu(image_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> Optional[Image.Image]:
    """
    GPU version of crop_from_mask.
    image_tensor: (C, H, W) float 0-255
    mask_tensor: (H, W) bool or 0/1
    """
    # Find indices
    nonzero = torch.nonzero(mask_tensor)
    if nonzero.shape[0] == 0:
        return None
    
    y1 = nonzero[:, 0].min().item()
    y2 = nonzero[:, 0].max().item()
    x1 = nonzero[:, 1].min().item()
    x2 = nonzero[:, 1].max().item()
    
    # Crop
    crop_img = image_tensor[:, y1:y2+1, x1:x2+1].clone()
    crop_mask = mask_tensor[y1:y2+1, x1:x2+1]
    
    # Apply mask (set background to 0)
    crop_img = crop_img * crop_mask.unsqueeze(0)
    
    # Convert to PIL (move to CPU)
    crop_img_np = crop_img.permute(1, 2, 0).byte().cpu().numpy() # HWC
    return Image.fromarray(crop_img_np)


def crop_from_mask_gpu_bbox(image_rgb: np.ndarray, mask_tensor: torch.Tensor) -> Optional[Image.Image]:
    """Hybrid crop optimized for throughput.

    Computes bbox from a GPU mask, but crops from the original CPU image.
    Only the *bbox-region* mask is transferred GPU->CPU, which is typically much
    smaller than transferring the cropped image tensor.
    """
    nonzero = torch.nonzero(mask_tensor)
    if nonzero.shape[0] == 0:
        return None

    y1 = int(nonzero[:, 0].min().item())
    y2 = int(nonzero[:, 0].max().item())
    x1 = int(nonzero[:, 1].min().item())
    x2 = int(nonzero[:, 1].max().item())

    crop_img = image_rgb[y1:y2 + 1, x1:x2 + 1].copy()

    # Transfer only bbox mask region
    crop_mask = mask_tensor[y1:y2 + 1, x1:x2 + 1].detach().to("cpu").numpy().astype(bool)
    crop_img[~crop_mask] = 0

    return Image.fromarray(crop_img)

# ------------------------------
# BLIP-2 verification (VQA)
# ------------------------------
@torch.no_grad()
def verify_mask_with_blip(
    blip_processor,
    blip_model,
    blip_device,
    image_input, # numpy or tensor
    mask_input, # numpy or tensor
    text_prompt: str,
    use_gpu_crop=False
) -> bool:
    """
    Use BLIP-2 to verify if mask matches text description.
    """
    if use_gpu_crop:
        # Fast path: keep image on CPU, transfer only small bbox-mask slice from GPU
        if isinstance(image_input, np.ndarray) and isinstance(mask_input, torch.Tensor):
            crop = crop_from_mask_gpu_bbox(image_input, mask_input)
        else:
            crop = crop_from_mask_gpu(image_input, mask_input)
    else:
        crop = crop_from_mask(image_input, mask_input)
        
    if crop is None:
        return False

    # Debug/throughput knob: allow disabling BLIP gating entirely.
    # Useful to confirm EVF-SAM is producing reasonable masks.
    if os.environ.get("DP_DISABLE_BLIP_VERIFY", "0") == "1":
        return True

    question = f"Question: What object does the given image represent? Answer with yes or no. Is it a {text_prompt}?"

    # Prepare input
    inputs = blip_processor(images=crop, text=question, return_tensors="pt")

    # BatchEncoding supports .to(device); keep token dtypes intact, cast only floats.
    try:
        inputs = inputs.to(blip_device)
    except Exception:
        # Fallback for older transformer versions
        inputs = {k: (v.to(blip_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    if isinstance(inputs, dict):
        if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
    else:
        # BatchEncoding-like
        if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

    # Generate answer
    outputs = blip_model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False
    )
    answer = blip_processor.batch_decode(outputs, skip_special_tokens=True)[0].lower().strip()
    if os.environ.get("DP_BLIP_DEBUG", "0") == "1":
        print(f"[INFO] BLIP verification answer: {answer}")

    # Match v2 semantics: accept if BLIP answer contains "yes" anywhere.
    return "yes" in answer


@torch.no_grad()
def verify_masks_with_blip_batch(
    blip_processor,
    blip_model,
    blip_device,
    images_rgb: List[np.ndarray],
    masks_gpu: List[torch.Tensor],
    text_prompt: str,
) -> List[bool]:
    """Batch BLIP verification to reduce per-call overhead.

    images_rgb: list of RGB uint8 HWC
    masks_gpu: list of (H,W) bool/0-1 tensors on GPU
    """
    if len(images_rgb) != len(masks_gpu):
        raise ValueError("images_rgb and masks_gpu must have same length")

    question = (
        "Question: What object does the given image represent? Answer with yes or no. "
        f"Is it a {text_prompt}?"
    )

    crops: List[Optional[Image.Image]] = []
    valid_idx: List[int] = []
    for i, (img_rgb, msk) in enumerate(zip(images_rgb, masks_gpu)):
        crop = crop_from_mask_gpu_bbox(img_rgb, msk)
        crops.append(crop)
        if crop is not None:
            valid_idx.append(i)

    results = [False] * len(images_rgb)
    if not valid_idx:
        return results

    # Debug/throughput knob: allow disabling BLIP gating entirely.
    if os.environ.get("DP_DISABLE_BLIP_VERIFY", "0") == "1":
        for i in valid_idx:
            results[i] = True
        return results

    valid_crops = [crops[i] for i in valid_idx]
    texts = [question] * len(valid_crops)

    inputs = blip_processor(images=valid_crops, text=texts, return_tensors="pt", padding=True)
    try:
        inputs = inputs.to(blip_device)
    except Exception:
        inputs = {k: (v.to(blip_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

    outputs = blip_model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
    )
    answers = [a.lower().strip() for a in blip_processor.batch_decode(outputs, skip_special_tokens=True)]
    if os.environ.get("DP_BLIP_DEBUG", "0") == "1":
        for ans in answers[: min(8, len(answers))]:
            print(f"[INFO] BLIP batch answer: {ans}")

    # Match v2 semantics: accept if BLIP answer contains "yes" anywhere.
    for idx, ans in zip(valid_idx, answers):
        results[idx] = ("yes" in ans)

    return results

# ------------------------------
# EVF-SAM + BLIP verification
# ------------------------------
def extract_segmentation_mask(
    image_path: str,
    object_description: dict,
    output_path: str,
    evf_ctx: dict,
    blip_processor=None,
    blip_model=None,
    blip_device=None,
) -> float:
    """
    Run EVF-SAM inference and verify predicted mask with BLIP-2.
    """
    
    def _save_black_mask():
        """Saves an all-black mask to output_path.
        Falls back to default size (640x480) if the image is unreadable.
        """
        DEFAULT_SIZE = (640, 480)

        try:
            im_check = Image.open(image_path)
            im_check.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
            mask_size = im_check.size
        except Exception as e:
            log_corrupt_image(image_path)
            # print(f"[WARN] Could not read image '{image_path}'. "
            #     f"Using default mask size {DEFAULT_SIZE}. Error: {e}")
            mask_size = DEFAULT_SIZE

        empty_mask = Image.new("RGB", mask_size, (0, 0, 0))
        empty_mask.save(output_path)
        # print(f"[INFO] Saved empty mask at {output_path} with size {mask_size}")

    # Safety check
    if blip_processor is None or blip_model is None or blip_device is None:
        print("[ERROR] BLIP arguments missing.")
        _save_black_mask()
        return 0.0

    model = evf_ctx["model"]
    device = evf_ctx["device"]

    # ---------------- Load image ----------------
    image_rgb = _imread_rgb(image_path)
    if image_rgb is None:
        print(f"[ERROR] Could not read image: {image_path}")
        _save_black_mask()
        return 0.0

    original_size = image_rgb.shape[:2]

    # Move to GPU immediately (already RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).to(device=device, dtype=torch.float32)

    # ---------------- Preprocess (GPU) ----------------
    image_beit = evf_ctx["beit3_preprocess_gpu"](image_tensor, img_size=evf_ctx["img_size"]).to(
        dtype=model.dtype, device=device
    ).unsqueeze(0)

    image_sam, resize_shape = evf_ctx["sam_preprocess_gpu"](
        image_tensor,
        pixel_mean=evf_ctx["sam_pixel_mean"],
        pixel_std=evf_ctx["sam_pixel_std"],
        model_type=evf_ctx["model_type"],
    )
    image_sam = image_sam.to(dtype=model.dtype, device=device).unsqueeze(0)

    input_ids = evf_ctx["evfsam_input_ids"]

    # ---------------- Inference ----------------
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            pred_mask = model.inference(
                image_sam,
                image_beit,
                input_ids,
                resize_list=[resize_shape],
                original_size_list=[original_size],
            )
        # pred_mask is (1, H, W) tensor on GPU
        pred = pred_mask[0] > 0
    except Exception as e:
        print(f"[ERROR] EVF inference failed: {e}")
        _save_black_mask()
        return 0.0

    # ---------------- BLIP verification ----------------
    try:
        ok = verify_mask_with_blip(
            blip_processor,
            blip_model,
            blip_device,
            image_rgb,  # keep image on CPU; crop uses GPU mask bbox only
            pred,       # mask stays on GPU
            object_description["blip"],
            use_gpu_crop=True,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"[WARN] BLIP verification failed: {e}")
        _save_black_mask()
        return 0.0

    if not ok:
        # print("[WARN] Mask rejected: BLIP mismatch.")
        _save_black_mask()
        return 0.0

    # ---------------- Save mask ----------------
    mask_uint8 = (pred.byte().cpu().numpy() * 255).astype(np.uint8)
    out = Image.fromarray(mask_uint8, mode="L").convert("RGB")
    out.save(output_path)

    return 1.0


def extract_segmentation_masks_batch(
    image_paths: List[str],
    object_description: dict,
    output_paths: List[str],
    evf_ctx: dict,
    blip_processor=None,
    blip_model=None,
    blip_device=None,
) -> List[float]:
    """Run EVF-SAM inference in batches (and batch BLIP verification).

    Returns a list of confidences (1.0 accepted, 0.0 rejected/failure) aligned with inputs.
    """
    if len(image_paths) != len(output_paths):
        raise ValueError("image_paths and output_paths must have same length")

    if blip_processor is None or blip_model is None or blip_device is None:
        # keep behaviour consistent with single-image path
        return [0.0 for _ in image_paths]

    model = evf_ctx["model"]
    device = evf_ctx["device"]

    # Parallel image loading
    with ThreadPoolExecutor(max_workers=min(len(image_paths), 8)) as executor:
        images_rgb = list(executor.map(_imread_rgb, image_paths))

    original_sizes = [img.shape[:2] if img is not None else None for img in images_rgb]

    # Preprocess each image into fixed-size tensors, then stack
    image_sam_list = []
    image_beit_list = []
    resize_shapes = []
    valid_indices = []

    for i, img_rgb in enumerate(images_rgb):
        if img_rgb is None:
            continue
        image_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device=device, dtype=torch.float32)
        image_beit = evf_ctx["beit3_preprocess_gpu"](image_tensor, img_size=evf_ctx["img_size"]).to(
            dtype=model.dtype, device=device
        )
        image_sam, resize_shape = evf_ctx["sam_preprocess_gpu"](
            image_tensor,
            pixel_mean=evf_ctx["sam_pixel_mean"],
            pixel_std=evf_ctx["sam_pixel_std"],
            model_type=evf_ctx["model_type"],
        )
        image_sam_list.append(image_sam.to(dtype=model.dtype, device=device))
        image_beit_list.append(image_beit)
        resize_shapes.append(resize_shape)
        valid_indices.append(i)

    confs = [0.0] * len(image_paths)
    if not valid_indices:
        # nothing decoded
        return confs

    image_sam_batch = torch.stack(image_sam_list, dim=0)
    image_beit_batch = torch.stack(image_beit_list, dim=0)
    del image_sam_list, image_beit_list

    input_ids = evf_ctx["evfsam_input_ids"]
    if input_ids.dim() == 2 and input_ids.shape[0] == 1:
        input_ids = input_ids.expand(image_sam_batch.shape[0], -1)

    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            pred_masks = model.inference(
                image_sam_batch,
                image_beit_batch,
                input_ids,
                resize_list=resize_shapes,
                original_size_list=[original_sizes[i] for i in valid_indices],
            )
        
        del image_sam_batch, image_beit_batch

        # EVF-SAM implementations may return:
        #  - torch.Tensor of shape (B,H,W) or (B,1,H,W)
        #  - list/tuple of tensors (each HxW or 1xHxW)
        preds: List[Optional[torch.Tensor]] = []
        if isinstance(pred_masks, torch.Tensor):
            pm = pred_masks
            if pm.dim() == 4 and pm.shape[1] == 1:
                pm = pm[:, 0]
            if pm.dim() != 3:
                raise RuntimeError(f"Unexpected pred_masks tensor shape: {tuple(pm.shape)}")
            preds = [(pm[j] > 0) for j in range(pm.shape[0])]
        elif isinstance(pred_masks, (list, tuple)):
            for m in pred_masks:
                if isinstance(m, torch.Tensor):
                    mt = m
                    if mt.dim() == 3 and mt.shape[0] == 1:
                        mt = mt[0]
                    if mt.dim() != 2:
                        raise RuntimeError(f"Unexpected pred mask element shape: {tuple(mt.shape)}")
                    preds.append(mt > 0)
                else:
                    preds.append(None)
        else:
            raise RuntimeError(f"Unexpected pred_masks type: {type(pred_masks)}")

        del pred_masks

        # Align length to batch
        if len(preds) != len(valid_indices):
            if os.environ.get("DP_EVF_DEBUG", "0") == "1":
                print(f"[WARN] EVF batch output length mismatch: preds={len(preds)} valid={len(valid_indices)}")
            # best-effort pad/truncate
            preds = (preds + [None] * len(valid_indices))[: len(valid_indices)]

    except Exception as e:
        if os.environ.get("DP_EVF_DEBUG", "0") == "1":
            print(f"[WARN] EVF batch inference failed: {e}")
        preds = [None for _ in valid_indices]

    # BLIP batch verification
    to_verify_imgs = []
    to_verify_masks = []
    verify_map = []
    for local_j, global_i in enumerate(valid_indices):
        p = preds[local_j]
        if p is None:
            continue
        to_verify_imgs.append(images_rgb[global_i])
        to_verify_masks.append(p)
        verify_map.append(global_i)

    oks = []
    if verify_map:
        try:
            oks = verify_masks_with_blip_batch(
                blip_processor,
                blip_model,
                blip_device,
                images_rgb=[images_rgb[i] for i in verify_map],
                masks_gpu=to_verify_masks,
                text_prompt=object_description["blip"],
            )
        except Exception:
            oks = [False] * len(verify_map)

    ok_by_idx = {i: ok for i, ok in zip(verify_map, oks)}

    # Parallel mask saving
    def _save_mask_worker(args):
        local_j, global_i = args
        pred = preds[local_j]
        out_path = output_paths[global_i]
        img_rgb = images_rgb[global_i]
        is_ok = ok_by_idx.get(global_i, False)

        if pred is None or not is_ok:
            # black mask
            try:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                if img_rgb is not None:
                    h, w = img_rgb.shape[:2]
                else:
                    w, h = 640, 480
                Image.new("RGB", (w, h), (0, 0, 0)).save(out_path)
            except Exception:
                pass
            return 0.0
        else:
            try:
                mask_uint8 = (pred.byte().cpu().numpy() * 255).astype(np.uint8)
                out = Image.fromarray(mask_uint8, mode="L").convert("RGB")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                out.save(out_path)
                return 1.0
            except Exception:
                return 0.0

    with ThreadPoolExecutor(max_workers=min(len(valid_indices), 8)) as executor:
        results = list(executor.map(_save_mask_worker, [(lj, gi) for lj, gi in enumerate(valid_indices)]))

    for idx, global_i in enumerate(valid_indices):
        confs[global_i] = results[idx]

    return confs

def is_robotic_arm_detected(mask_confidence, confidence_threshold):
    return mask_confidence > confidence_threshold, mask_confidence

def process_video(item, evf_ctx, blip_processor, blip_model, blip_device):
    src_path = os.path.join(IMAGE_DIR, item)
    dest_path = os.path.join(ROBOTICMASK_DIR, item)
    os.makedirs(dest_path, exist_ok=True)

    all_files = [f for f in os.listdir(src_path) if f.lower().endswith(".jpg")]
    sorted_files = natsorted(all_files)
    subset_count = max(1, int(0.2 * len(sorted_files)))
    selected_files = sorted_files[:subset_count]

    detections = []

    # --- Pass 1: run detection for each frame ---
    # Performance default: batch EVF+BLIP calls to reduce overhead.
    # Override as needed for GPU memory.
    evf_batch = int(os.environ.get("DP_EVF_BATCH", "8"))
    if evf_batch <= 1:
        for filename in selected_files:
            image_path = os.path.join(src_path, filename)
            base_name = os.path.splitext(filename)[0]
            output_name = f"{base_name}.png"
            output_path = os.path.join(dest_path, output_name)
            conf = extract_segmentation_mask(
                image_path,
                ROBOT_OBJECT_DESCRIPTION,
                output_path,
                evf_ctx,
                blip_processor,
                blip_model,
                blip_device,
            )
            detected, conf_value = is_robotic_arm_detected(conf, confidence_threshold)
            detections.append((detected, round(conf_value, 4)))
    else:
        for j in range(0, len(selected_files), evf_batch):
            batch_files = selected_files[j:j + evf_batch]
            image_paths = [os.path.join(src_path, f) for f in batch_files]
            output_paths = [os.path.join(dest_path, f"{os.path.splitext(f)[0]}.png") for f in batch_files]
            confs = extract_segmentation_masks_batch(
                image_paths=image_paths,
                object_description=ROBOT_OBJECT_DESCRIPTION,
                output_paths=output_paths,
                evf_ctx=evf_ctx,
                blip_processor=blip_processor,
                blip_model=blip_model,
                blip_device=blip_device,
            )
            for conf in confs:
                detected, conf_value = is_robotic_arm_detected(conf, confidence_threshold)
                detections.append((detected, round(conf_value, 4)))

    # --- Pass 2: find first index i where arm detected in both i and i+1 ---
    detected_index = None
    if len(detections) == 1:
        # v2 intent: require consistency to avoid false positives.
        # For very short clips / tiny subsets, fall back to a single-hit decision.
        if detections[0][0]:
            detected_index = 0
    else:
        for i in range(len(detections) - 1):
            if detections[i][0] and detections[i + 1][0]:
                detected_index = i
                break

    # Optional: if batched mode produced no detections, retry with v2-like single-frame path.
    # This is a robustness fix: some transformer pipelines behave differently in batch vs single.
    if detected_index is None and evf_batch > 1 and os.environ.get("DP_EVF_RETRY_SINGLE_ON_FAIL", "0") == "1":
        detections = []
        for filename in selected_files:
            image_path = os.path.join(src_path, filename)
            base_name = os.path.splitext(filename)[0]
            output_name = f"{base_name}.png"
            output_path = os.path.join(dest_path, output_name)
            conf = extract_segmentation_mask(
                image_path,
                ROBOT_OBJECT_DESCRIPTION,
                output_path,
                evf_ctx,
                blip_processor,
                blip_model,
                blip_device,
            )
            detected, conf_value = is_robotic_arm_detected(conf, confidence_threshold)
            detections.append((detected, round(conf_value, 4)))

        # Recompute detected_index after retry
        detected_index = None
        if len(detections) == 1:
            if detections[0][0]:
                detected_index = 0
        else:
            for i in range(len(detections) - 1):
                if detections[i][0] and detections[i + 1][0]:
                    detected_index = i
                    break

    # --- Pass 3: handle results ---
    if detected_index is not None:
        filename = selected_files[detected_index]
        first_detected_indices[item] = {
            "frame": filename,
            "index": detected_index,
            "confidence": [detections[detected_index][1]],
        }
        for j in range(detected_index + 1, len(detections)):
            first_detected_indices[item]["confidence"].append(detections[j][1])
    else:
        # Never delete or modify the source dataset.
        # If detection fails, record it so you can inspect or handle downstream.
        try:
            videos_without_detection.append(item)
        except Exception:
            pass

# Load EVF-SAM and BLIP-2 Lazy Initializer
tokenizer = None
model = None
blip_processor = None
blip_model = None
blip_device = None
evf_ctx = None

# ---------------------- Memory and CUDA Utilities ----------------------------------

def _safe_to_cpu(obj, name=None):
    """Try to move model/tensor to CPU to release GPU memory gracefully."""
    if obj is None:
        return
    try:
        if hasattr(obj, "to"):
            obj.to("cpu")
        elif hasattr(obj, "cpu"):
            obj.cpu()
    except Exception:
        if name:
            print(f"[WARN] Couldn't move {name} to cpu cleanly.")

def _safe_del(varname, glbs):
    """Delete a global variable name if it exists by setting to None."""
    try:
        if varname in glbs: 
            glbs[varname] = None
    except Exception:
        pass

def run_cuda_cleanup():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

def init_evf_and_blip():
    global tokenizer, model, blip_processor, blip_model, blip_device, evf_ctx
    if model is None:
        tokenizer, model = init_models()
    
    if blip_model is None:
        print(f"[INFO] Loading BLIP-2 model: {BLIP_MODEL}")
        blip_processor, blip_model, blip_device = init_blip(BLIP_MODEL)
        
    if evf_ctx is None:
        evf_ctx = {
            "model": model,
            "tokenizer": tokenizer,
            "device": model.device,
            "img_size": 224,
            "model_type": EVF_SAM_MODEL_TYPE,
            "sam_preprocess_gpu": sam_preprocess_gpu,
            "beit3_preprocess_gpu": beit3_preprocess_gpu,
            # Cache constant tensors once (avoid per-frame allocations)
            "sam_pixel_mean": torch.tensor([123.675, 116.28, 103.53], device=model.device, dtype=torch.float32).view(-1, 1, 1),
            "sam_pixel_std": torch.tensor([58.395, 57.12, 57.375], device=model.device, dtype=torch.float32).view(-1, 1, 1),
            "evfsam_input_ids": tokenizer(ROBOT_OBJECT_DESCRIPTION["evfsam"], return_tensors="pt")["input_ids"].to(model.device),
        }
    return evf_ctx, blip_processor, blip_model, blip_device

def run_stage_robotic_arm_extraction(batch_subdirs):
    print(f"[INFO] Stage 0: Running robotic arm mask extraction for {len(batch_subdirs)} videos...")
    
    ctx, b_proc, b_mod, b_dev = init_evf_and_blip()
    
    global first_detected_indices, videos_without_detection
    first_detected_indices = {}
    videos_without_detection = []

    def process_video_wrapper(item):
        try:
            process_video(item, ctx, b_proc, b_mod, b_dev)
            return item, first_detected_indices.get(item)
        except Exception as e:
            print(f"[ERROR] Failed processing {item}: {e}")
            return item, None

    evf_workers = int(os.environ.get("DP_EVF_WORKERS", "1"))
    with ThreadPoolExecutor(max_workers=evf_workers) as executor:
        list(tqdm(executor.map(process_video_wrapper, batch_subdirs), 
                  total=len(batch_subdirs), 
                  desc="Finding robotic arm masks using EVF-SAM"))
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_stage_post_evf_cleanup():
    print("[INFO] >>> Cleaning up Stage 0 (EVF-SAM & BLIP) memory...")

    # try to move the EVF model
    if 'model' in globals():
        _safe_to_cpu(globals().get('model'), name='model')

    # BLIP
    if 'blip_model' in globals():
        _safe_to_cpu(globals().get('blip_model'), name='blip_model')

    # If you kept evf_ctx dict, try to move any model references inside it
    _ctx = globals().get('evf_ctx')
    if isinstance(_ctx, dict):
        for _, v in _ctx.items():
            try:
                if hasattr(v, "to"):
                    v.to("cpu")
            except Exception:
                pass

    # Delete references (globals) so Python can free them
    _globs = globals()
    for name in [
        "model",
        "tokenizer",
        "blip_model",
        "blip_processor",
        "blip_device",
        "evf_ctx"
    ]:
        _safe_del(name, _globs)

    run_cuda_cleanup()
    print("[INFO] >>> Stage 0 memory cleanup completed.")

def run_stage_post_sam_cleanup():
    print("[INFO] >>> Cleaning up Stage 1 (SAM) memory...")
    
    # SAM
    if 'sam_model' in globals():
        _safe_to_cpu(globals().get('sam_model'), name='sam_model')

    # SAM mask generator may hold a SAM model inside: try to move inner model to CPU
    if 'mask_generator' in globals():
        mg = globals().get('mask_generator')
        try:
            inner = getattr(mg, "model", None) or getattr(mg, "sam", None)
            if inner is not None:
                _safe_to_cpu(inner, name='mask_generator.inner_model')
        except Exception:
            pass

    # Clear thread local
    if '_sam_thread_local' in globals():
        tl = globals().get('_sam_thread_local')
        if hasattr(tl, 'mask_generator'):
            del tl.mask_generator

    _globs = globals()
    for name in ["sam_model", "mask_generator"]:
        _safe_del(name, _globs)

    run_cuda_cleanup()
    print("[INFO] >>> Stage 1 memory cleanup completed.")

def run_stage_post_xmem_cleanup():
    print("[INFO] >>> Cleaning up Stage 2 (XMem) memory...")
    
    if 'network' in globals():
        _safe_to_cpu(globals().get('network'), name='xmem_network')

    _globs = globals()
    for name in ["network"]:
        _safe_del(name, _globs)

    run_cuda_cleanup()
    print("[INFO] >>> Stage 2 memory cleanup completed.")

# -------------------------- SAM Inference on first frame of every video --------------------------------

def compute_mask_matrices_gpu(masks_tensor):
    # masks_tensor: (N, H, W), binary (0/1), float or half
    N, H, W = masks_tensor.shape
    # Use float16 for faster matrix multiplication and lower memory usage
    flat = masks_tensor.contiguous().reshape(N, -1).to(dtype=torch.float16)  # (N, HW)
    
    # Intersection: (N, N)
    inter = torch.mm(flat, flat.t())
    
    # Area: (N)
    area = flat.sum(dim=1)
    
    # Union: (N, N)
    # area[i] + area[j] - inter[i, j]
    area_i = area.unsqueeze(1) # (N, 1)
    area_j = area.unsqueeze(0) # (1, N)
    union = area_i + area_j - inter
    
    # IoU: (N, N)
    iou = inter / (union + 1e-6)
    
    # Containment (i is submask of j): inter[i, j] / area[i]
    # We want matrix C[i, j] = inter[i, j] / area[i]
    # If area[i] is 0, containment is 0
    containment = inter / (area_i + 1e-6)
    
    return iou, containment, area

def discard_submasks_gpu(masks_tensor, iou_thresh=0.82):
    """Match v2's discard_submasks semantics using GPU-computed containment.

    v2 builds a directed graph where edge i->j means j is a submask of i,
    then keeps only "master parents" found by DFS from masks sorted by area.
    """
    if masks_tensor.shape[0] == 0:
        return masks_tensor

    iou_mat, containment_mat, areas = compute_mask_matrices_gpu(masks_tensor)
    
    # Move to CPU for the DFS logic which is hard to vectorize
    containment_mat = containment_mat.cpu().numpy()
    areas = areas.cpu().numpy()

    n = masks_tensor.shape[0]
    visited = [False for _ in range(n)]
    is_master_parent = [False for _ in range(n)]

    # Build adjacency lists: i -> [j where j is submask of i]
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        # masks[j] is submask of masks[i] iff containment[j,i] >= thresh
        js = np.where(containment_mat[:, i] >= iou_thresh)[0].tolist()
        adj[i] = [j for j in js if j != i]

    sorted_indices = list(np.argsort(areas)[::-1])

    # Iterative DFS (avoids recursion depth issues)
    for root in sorted_indices:
        if visited[root]:
            continue
        is_master_parent[root] = True
        master_parent = root
        stack = [root]
        while stack:
            node = stack.pop()
            if visited[node]:
                # mimic v2 behavior: if we re-encounter a master-parent via another path, demote it
                if node != master_parent and is_master_parent[node]:
                    is_master_parent[node] = False
                continue
            visited[node] = True
            for nxt in adj[node]:
                if visited[nxt]:
                    if nxt != master_parent and is_master_parent[nxt]:
                        is_master_parent[nxt] = False
                else:
                    stack.append(nxt)

    return masks_tensor[is_master_parent]

def merge_similar_masks_gpu(masks_tensor, sim_thresh=0.82):
    """Match v2's merge_similar_masks semantics (complete-linkage) efficiently.

    Keeps one of the two masks (the larger by area) and deletes the smaller.
    Updates similarity to other masks via:
      IoU(k, keep) = min(IoU(k, keep), IoU(k, discard))
    """
    if masks_tensor.shape[0] <= 1:
        return masks_tensor

    iou_mat, _, areas = compute_mask_matrices_gpu(masks_tensor)
    
    # Move to CPU for the iterative merging logic
    iou = iou_mat.cpu().numpy().astype(np.float32, copy=True)
    areas = areas.cpu().numpy()
    n = masks_tensor.shape[0]

    np.fill_diagonal(iou, -1.0)
    deleted = np.zeros(n, dtype=bool)

    # Maintain deleted rows/cols at -1 so they never get selected.
    while True:
        tri = np.triu(iou, 1)
        flat_idx = int(np.argmax(tri))
        max_iou = float(tri.flat[flat_idx])
        if max_iou < sim_thresh:
            break

        i, j = np.unravel_index(flat_idx, tri.shape)
        if deleted[i] or deleted[j]:
            iou[i, j] = -1.0
            iou[j, i] = -1.0
            continue

        if areas[i] >= areas[j]:
            keep, discard = i, j
        else:
            keep, discard = j, i

        # Update only active k (excluding keep/discard), matching v2.
        active = ~deleted
        update_mask = active.copy()
        update_mask[keep] = False
        update_mask[discard] = False
        if np.any(update_mask):
            iou[update_mask, keep] = np.minimum(iou[update_mask, keep], iou[update_mask, discard])
            iou[keep, update_mask] = iou[update_mask, keep]

        deleted[discard] = True
        iou[discard, :] = -1.0
        iou[:, discard] = -1.0
        iou[keep, keep] = -1.0

    return masks_tensor[~deleted]

def init_sam_model(sam_checkpoint=SAM_CHECKPOINT, model_type="vit_h"):
    """Initialize SAM once."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    return sam

def SAM_inference(input_image_path, sam_mask_path, mask_generator):
    """Run SAM inference using a pre-initialized mask generator."""
    # Load image as tensor (RGB CHW uint8)
    image_tensor = _imread_tensor(input_image_path)
    if image_tensor is None:
        raise ValueError(f"Could not read image: {input_image_path}")

    # Generate SAM proposals (returns tensor of shape N, H, W)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        masks_tensor = mask_generator.generate(image_tensor)

    if masks_tensor is None or masks_tensor.shape[0] == 0:
        # Save empty mask
        h, w = image_tensor.shape[1:]
        cv2.imwrite(sam_mask_path, np.zeros((h, w, 3), dtype=np.uint8))
        return

    # Filtering (works on float tensors)
    # Using half() to save memory
    masks_tensor = masks_tensor.half()
    masks_tensor = discard_submasks_gpu(masks_tensor)
    masks_tensor = merge_similar_masks_gpu(masks_tensor)

    # Overlay visualization (GPU vectorized)
    H, W = image_tensor.shape[1:]
    overlay_ts = torch.zeros((H, W, 3), dtype=torch.uint8, device="cuda")
    
    if masks_tensor.shape[0] > 0:
        any_mask = torch.any(masks_tensor > 0, dim=0)
        if torch.any(any_mask):
            # Get the index of the last mask that is 1
            # We use .byte() to avoid issues with bool argmax in some torch versions
            last_mask_idx = (masks_tensor.shape[0] - 1) - torch.argmax(masks_tensor.flip(0).byte(), dim=0)
            palette = torch.tensor(COLOR_PALETTE, dtype=torch.uint8, device="cuda")
            color_indices = last_mask_idx % palette.shape[0]
            overlay_ts[any_mask] = palette[color_indices[any_mask]]

    overlay = overlay_ts.cpu().numpy()
    os.makedirs(os.path.dirname(sam_mask_path), exist_ok=True)
    cv2.imwrite(sam_mask_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Optional: intermediate cleanup
    del masks_tensor, overlay_ts, any_mask, image_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Main loop (reuse SAM model)
sam_model = None
_sam_thread_local = threading.local()

def init_sam():
    global sam_model
    if sam_model is None:
        sam_model = init_sam_model()
    return sam_model

def get_sam_generator():
    if not hasattr(_sam_thread_local, "mask_generator"):
        # Default to 64 points per batch for ViT-H to avoid OOM
        points_per_batch = int(os.environ.get("DP_SAM_POINTS_PER_BATCH", "64"))
        _sam_thread_local.mask_generator = SamAutomaticMaskGenerator(
            init_sam(), 
            points_per_batch=points_per_batch, 
            output_mode="torch_mask"
        )
    return _sam_thread_local.mask_generator

def run_stage_sam_inference(batch_subdirs):
    print(f"[INFO] Stage 1: Starting parallel SAM inference for {len(batch_subdirs)} videos...")
    
    # Initialize SAM model once globally before starting threads
    init_sam()
    
    def sam_inference_wrapper(item):
        src_path = os.path.join(IMAGE_DIR, item)
        if not os.path.isdir(src_path):
            return

        dest_path = os.path.join(ANNOTATION_DIR, item)
        os.makedirs(dest_path, exist_ok=True)

        # Use the first frame where the arm was detected, or fallback to 00000.jpg
        frame_name = "00000.jpg"
        if item in first_detected_indices:
            frame_name = first_detected_indices[item].get("frame", "00000.jpg")
        
        mask_name = os.path.splitext(frame_name)[0] + ".png"

        image_path = os.path.join(src_path, frame_name)
        sam_mask_path = os.path.join(dest_path, mask_name)
        
        try:
            mask_generator = get_sam_generator()
            SAM_inference(image_path, sam_mask_path, mask_generator)
        except Exception as e:
            print(f"[ERROR] SAM inference failed for {item}: {e}")

    sam_workers = int(os.environ.get("DP_SAM_WORKERS", "1"))
    with ThreadPoolExecutor(max_workers=sam_workers) as executor:
        list(tqdm(executor.map(sam_inference_wrapper, batch_subdirs), 
                  total=len(batch_subdirs), 
                  desc="Generating SAM Proposals"))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------------------- XMEM inference on SAM masks of 1st frame for Object tracking --------------------------------------

# Lazy-import hickle only when --save_scores is enabled.
hkl = None

network = None

def init_xmem():
    global network
    if network is None:
        network = XMem(config, XMEM_MODEL_PATH).cuda().eval()
        model_weights = torch.load(XMEM_MODEL_PATH)
        network.load_weights(model_weights, init_as_zero_if_needed=True)
    return network

def run_stage_xmem_tracking(batch_subdirs):
    print(f"[INFO] Stage 2: Starting parallel XMem tracking for {len(batch_subdirs)} videos...")
    
    # Use global network
    net = init_xmem()
    
    # We need to filter FastLongTestDataset for this batch
    class BatchedLongTestDataset(FastLongTestDataset):
        def __init__(self, data_root, vid_list, size=-1):
            super().__init__(data_root, size)
            self.vid_list = [v for v in vid_list if os.path.isdir(os.path.join(self.image_dir, v))]

    meta_dataset = BatchedLongTestDataset(DATA_DIR, batch_subdirs, size=args.size)
    
    torch.autograd.set_grad_enabled(False)
    mask_saver = MaskSaver(max_workers=16)
    num_video_workers = int(os.environ.get("DP_XMEM_VIDEO_WORKERS", "2"))

    with ThreadPoolExecutor(max_workers=num_video_workers) as executor:
        # Note: meta_dataset.get_datasets() yields readers for each video in self.vid_list
        futures = [executor.submit(process_xmem_video, vid_reader, net, config, args, mask_saver) 
                   for vid_reader in meta_dataset.get_datasets()]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="XMem Tracking"):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing video: {e}")

    mask_saver.shutdown()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ------------------------------ Distance metric for masks --------------------------------------

class FastVideoReader(torch.utils.data.Dataset):
    """
    Optimized VideoReader using _imread_tensor and faster processing.
    """
    def __init__(self, vid_name, image_dir, mask_dir, size=-1, to_save=None, use_all_mask=False, size_dir=None):
        self.vid_name = vid_name
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.to_save = to_save
        self.use_all_mask = use_all_mask
        self.size_dir = size_dir if size_dir is not None else image_dir

        self.frames = sorted(os.listdir(self.image_dir))
        
        # Get palette from first mask
        mask_files = sorted(os.listdir(mask_dir))
        if mask_files:
            first_mask_path = path.join(mask_dir, mask_files[0])
            try:
                with Image.open(first_mask_path) as img:
                    self.palette = img.getpalette()
                self.first_gt_path = first_mask_path
            except Exception:
                self.palette = None
                self.first_gt_path = None
        else:
            self.palette = None
            self.first_gt_path = None
            
        self.size = size

    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}
        info['frame'] = frame
        info['save'] = (self.to_save is None) or (frame[:-4] in self.to_save)

        im_path = path.join(self.image_dir, frame)
        data['is_corrupt'] = False
        data['im_path'] = im_path

        try:
            # Read as float32 [0, 1] on CPU
            img_tensor = _imread_tensor(im_path, device="cpu")
            if img_tensor is None:
                # Fallback to PIL if cv2 fails
                img_pil = Image.open(im_path).convert('RGB')
                img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)
            
            img_tensor = img_tensor.float() / 255.0
            
            # Normalization (XMem expects im_normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std

            h, w = img_tensor.shape[-2:]
            info['shape'] = (h, w)
            
            if self.size > 0:
                min_side = min(h, w)
                new_h = int(h / min_side * self.size)
                new_w = int(w / min_side * self.size)
                img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
                info['need_resize'] = True
            else:
                info['need_resize'] = False
            
            data['rgb'] = img_tensor
        except Exception:
            data['is_corrupt'] = True
            log_corrupt_image(im_path)
            data['rgb'] = torch.zeros((3, 1, 1)) # Dummy to avoid collation errors
            data['info'] = info
            return data

        gt_path = path.join(self.mask_dir, frame[:-4]+'.png')
        load_mask = self.use_all_mask or (gt_path == self.first_gt_path)
        if load_mask and path.exists(gt_path):
            try:
                mask = Image.open(gt_path).convert('P', dither=Image.NONE)
                mask = np.array(mask, dtype=np.uint8)
                data['mask'] = mask
            except Exception:
                data['is_corrupt'] = True
                data['im_path'] = gt_path
                log_corrupt_image(gt_path)
                return data

        data['info'] = info
        return data

    def resize_mask(self, mask):
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), mode='nearest')

    def get_palette(self):
        return self.palette

    def __len__(self):
        return len(self.frames)

class FastLongTestDataset:
    def __init__(self, data_root, size=-1):
        self.image_dir = path.join(data_root, 'JPEGImages')
        self.mask_dir = path.join(data_root, 'Annotations')
        self.size = size
        self.vid_list = sorted(os.listdir(self.image_dir))

    def get_datasets(self):
        for video in self.vid_list:
            yield FastVideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                to_save = [
                    name[:-4] for name in os.listdir(path.join(self.mask_dir, video))
                ],
                size=self.size,
            )

    def __len__(self):
        return len(self.vid_list)

class MaskSaver:
    def __init__(self, max_workers=8):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def save(self, mask_np, path, palette=None):
        self.executor.submit(self._save_impl, mask_np, path, palette)

    def _save_impl(self, mask_np, path, palette):
        img = Image.fromarray(mask_np)
        if palette:
            img.putpalette(palette)
        else:
            img.putpalette(generate_color_palette())
        img.save(path)

    def shutdown(self):
        self.executor.shutdown(wait=True)

def process_xmem_video(vid_reader, network, config, args, mask_saver):
    gc.collect()
    torch.cuda.empty_cache()

    vid_name = vid_reader.vid_name
    device = next(network.parameters()).device
    
    # Use 1 worker for DataLoader since we parallelize videos
    loader = DataLoader(
        vid_reader,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    
    vid_length = len(loader)
    local_config = config.copy()
    local_config['enable_long_term_count_usage'] = (
        local_config['enable_long_term'] and
        (vid_length
            / (local_config['max_mid_term_frames']-local_config['min_mid_term_frames'])
            * local_config['num_prototypes'])
        >= local_config['max_long_term_elements']
    )

    mapper = MaskMapper()
    processor = InferenceCore(network, config=local_config)
    first_mask_loaded = False
    
    global hkl

    for ti, data in enumerate(loader):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=not args.benchmark):
            # Check for corrupt frames and log them
            if data.get('is_corrupt', [False])[0]:
                continue

            rgb = data['rgb'].to(device, non_blocking=True)[0]
            msk = data.get('mask')
            info = data['info']
            frame = info['frame'][0]
            shape = (info['shape'][0].item(), info['shape'][1].item())
            need_resize = info['need_resize'][0]

            if not first_mask_loaded:
                if msk is not None:
                    first_mask_loaded = True
                else:
                    del rgb
                    continue

            if args.flip:
                rgb = torch.flip(rgb, dims=[-1])
                if msk is not None:
                    if isinstance(msk, torch.Tensor):
                        msk = torch.flip(msk, dims=[-1])
                    else:
                        msk = np.flip(msk, axis=-1).copy()

            msk_tensor = None
            labels = None
            if msk is not None:
                m0 = msk[0]
                if isinstance(m0, torch.Tensor):
                    m0_np = m0.detach().cpu().numpy()
                else:
                    m0_np = m0
                msk_tensor, labels = mapper.convert_mask(m0_np)
                msk_tensor = msk_tensor.to(device, non_blocking=True)
                if need_resize:
                    msk_tensor = vid_reader.resize_mask(msk_tensor.unsqueeze(0))[0]
                processor.set_all_labels(list(mapper.remappings.values()))

            try:
                prob = processor.step(rgb, msk_tensor, labels, end=(ti==vid_length-1))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                prob = processor.step(rgb, msk_tensor, labels, end=(ti==vid_length-1))

            if need_resize:
                prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

            if args.flip:
                prob = torch.flip(prob, dims=[-1])

            out_mask = torch.max(prob, dim=0).indices
            out_mask_np = (out_mask.detach().cpu().numpy()).astype(np.uint8)

            if args.save_all or info['save'][0]:
                this_out_path = path.join(XMEM_OUTPUT_DIR, vid_name)
                os.makedirs(this_out_path, exist_ok=True)
                out_mask_remapped = mapper.remap_index_mask(out_mask_np)
                
                save_path = path.join(this_out_path, frame[:-4]+'.png')
                mask_saver.save(out_mask_remapped, save_path, vid_reader.get_palette())

            if args.save_scores:
                if hkl is None:
                    try:
                        import hickle as hkl  # type: ignore
                    except Exception:
                        raise RuntimeError("--save_scores requires 'hickle' to be installed (pip install hickle)")
                np_path = path.join(XMEM_OUTPUT_DIR, 'Scores', vid_name)
                os.makedirs(np_path, exist_ok=True)
                if ti==len(loader)-1:
                    hkl.dump(mapper.remappings, path.join(np_path, f'backward.hkl'), mode='w')
                if args.save_all or info['save'][0]:
                    hkl.dump(prob.detach().cpu().numpy(), path.join(np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')
            
            # Explicitly delete large tensors to free VRAM
            del rgb, msk_tensor, prob, out_mask
    
    # Clear memory for this video
    del processor, mapper, loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return vid_length

# ------------------------------ Distance metric for masks --------------------------------------
def mask_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.mean(xs), np.mean(ys)

# Bounding box and mask utilities
def bbox_from_mask(mask: np.ndarray, pad: int = 16):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h, w = mask.shape[:2]
    x0 = max(0, x0 - pad)
    x1 = min(w - 1, x1 + pad)
    y0 = max(0, y0 - pad)
    y1 = min(h - 1, y1 + pad)
    return x0, y0, x1, y1

def crop_and_offset(img: np.ndarray, bbox):
    x0, y0, x1, y1 = bbox
    crop = img[y0:y1+1, x0:x1+1]
    return crop, (x0, y0)

def map_pts_to_original(pts: np.ndarray, offset: tuple):
    x0, y0 = offset
    return pts + np.array([x0, y0], dtype=np.float32)

def filter_pts_in_mask(pts: np.ndarray, mask: np.ndarray):
    h, w = mask.shape[:2]
    xs = np.round(pts[:, 0]).astype(int)
    ys = np.round(pts[:, 1]).astype(int)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & (mask[ys, xs] > 0)
    return valid

# LoFTR official model loading
def load_loftr_model(weight_type: str, weight_path: str, device: str = "cuda"):
    print(f"[PROFILE] Loading LoFTR model ({weight_type})...")
    start_time = time.perf_counter()
    from loftr_src.loftr import LoFTR, default_cfg
    if weight_type == "outdoor_ot":
        # Coarse matching
        default_cfg['match_coarse']['match_type'] = 'sinkhorn'
        default_cfg['match_coarse']['sparse_spvs'] = False
        default_cfg['match_coarse']['train_coarse_percent'] = 0.3
        # Trainer settings (nested under 'trainer' key)
        default_cfg['trainer'] = default_cfg.get('trainer', {})  # create if doesn't exist
        default_cfg['trainer']['canonical_lr'] = 8e-3
        default_cfg['trainer']['warmup_step'] = 1875
        default_cfg['trainer']['warmup_ratio'] = 0.1
        default_cfg['trainer']['mslr_milestones'] = [8, 12, 16, 20, 24]
        default_cfg['trainer']['ransac_pixel_thr'] = 0.5
        default_cfg['trainer']['optimizer'] = "adamw"
        default_cfg['trainer']['adamw_decay'] = 0.1
    elif weight_type == "indoor_ot":
        # Coarse matching type
        default_cfg['match_coarse']['match_type'] = 'sinkhorn'
        # Sparse supervision
        default_cfg['match_coarse']['sparse_spvs'] = False
        # Trainer milestones
        # Ensure 'trainer' exists
        default_cfg['trainer'] = default_cfg.get('trainer', {})
        default_cfg['trainer']['mslr_milestones'] = [3, 6, 9, 12, 17, 20, 23, 26, 29]
    matcher = LoFTR(config=default_cfg)
    ckpt = torch.load(weight_path, map_location=device)
    matcher.load_state_dict(ckpt['state_dict'])
    matcher = matcher.eval().to(device)
    print(f"[PROFILE] LoFTR model loaded in {time.perf_counter() - start_time:.4f}s")
    return matcher

# --- LoFTR Profiling Stats ---
LOFTR_STATS = {
    'load_data': 0.0,
    'arm_refinement': 0.0,
    'distance_computation': 0.0,
    'match_visible_crop': 0.0,
    'match_visible_prep': 0.0,
    'match_visible_inf': 0.0,
    'match_visible_post': 0.0,
    'match_visible_count': 0,
    'kmeans': 0.0,
    'edit_detection': 0.0,
    'save_results': 0.0,
    'total_subdirs': 0
}
stats_lock = threading.Lock()

# Core matching function
def match_visible_correspondences(matcher, frameA, frameB, maskA, maskB, device="cuda"):
    if frameA is None or frameB is None or maskA is None or maskB is None:
        return []
    
    t0 = time.perf_counter()
    bboxA = bbox_from_mask(maskA)
    bboxB = bbox_from_mask(maskB)
    if bboxA is None or bboxB is None:
        # print("Empty mask region.")
        return []

    cropA, offA = crop_and_offset(frameA, bboxA)
    cropB, offB = crop_and_offset(frameB, bboxB)
    _cropMaskA, _ = crop_and_offset(maskA, bboxA)
    _cropMaskB, _ = crop_and_offset(maskB, bboxB)
    dt_crop = time.perf_counter() - t0

    # --- Convert to grayscale, resize to 640x480 for LoFTR compatibility
    t1 = time.perf_counter()
    imgA = cv2.cvtColor(cropA, cv2.COLOR_RGB2GRAY)
    imgB = cv2.cvtColor(cropB, cv2.COLOR_RGB2GRAY)
    imgA = cv2.resize(imgA, (640, 480))
    imgB = cv2.resize(imgB, (640, 480))

    tA = torch.from_numpy(imgA)[None][None].float().to(device) / 255.0
    tB = torch.from_numpy(imgB)[None][None].float().to(device) / 255.0
    dt_prep = time.perf_counter() - t1

    # --- Inference
    t2 = time.perf_counter()
    batch = {"image0": tA, "image1": tB}
    with torch.no_grad():
        matcher(batch)
    dt_inf = time.perf_counter() - t2

    t3 = time.perf_counter()
    mkpts0 = batch["mkpts0_f"].cpu().numpy()
    mkpts1 = batch["mkpts1_f"].cpu().numpy()
    mconf = batch["mconf"].cpu().numpy()

    if len(mkpts0) == 0:
        # print("No keypoints found.")
        return []

    # --- Map back to full-image coordinates (adjust for resize + crop offsets)
    scaleA = cropA.shape[1] / 640, cropA.shape[0] / 480
    scaleB = cropB.shape[1] / 640, cropB.shape[0] / 480
    mkpts0_scaled = mkpts0 * np.array(scaleA)
    mkpts1_scaled = mkpts1 * np.array(scaleB)
    ptsA_full = map_pts_to_original(mkpts0_scaled, offA)
    ptsB_full = map_pts_to_original(mkpts1_scaled, offB)

    # --- Filter by mask
    keepA = filter_pts_in_mask(ptsA_full, maskA)
    keepB = filter_pts_in_mask(ptsB_full, maskB)
    keep = keepA & keepB
    ptsA_vis = ptsA_full[keep]
    ptsB_vis = ptsB_full[keep]
    mconf = mconf[keep]
    dt_post = time.perf_counter() - t3

    with stats_lock:
        LOFTR_STATS['match_visible_crop'] += dt_crop
        LOFTR_STATS['match_visible_prep'] += dt_prep
        LOFTR_STATS['match_visible_inf'] += dt_inf
        LOFTR_STATS['match_visible_post'] += dt_post
        LOFTR_STATS['match_visible_count'] += 1

    if len(ptsA_vis) == 0:
        # print("No matches inside mask.")
        return []

    matched_pairs = [(ptsA_vis[i].tolist(), ptsB_vis[i].tolist()) for i in range(len(ptsA_vis))]
    # print(f"Total matched pairs: {len(matched_pairs)}")
    return matched_pairs, ptsA_vis, ptsB_vis, mconf
    return matched_pairs, ptsA_vis, ptsB_vis, mconf

def get_distance(matcher, frameA, frameB, maskA, maskB):
    # Compute distance between both masks using the centroid based approach
    c0 = mask_centroid(maskA)
    cN = mask_centroid(maskB)
    if c0 is None or cN is None:
        return -1
    centroid_dist = np.linalg.norm(np.array(cN) - np.array(c0))
    
    # Compute distance between both masks using feature matching approach
    fm_dist = None
    matches, ptsA, ptsB, conf = [], None, None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        matches, ptsA, ptsB, conf = match_visible_correspondences(
            matcher, frameA, frameB, maskA, maskB, device=device
        )
    except Exception as e:
        fm_dist = -1
    if len(matches) == 0 or ptsA is None or ptsB is None:
        fm_dist = -1
    else:
        # Compute pairwise distances
        distances = np.linalg.norm(ptsA - ptsB, axis=1)
        # Return the average distance
        avg_distance = float(distances.mean())
        fm_dist = avg_distance
    
    if fm_dist == -1:
        return centroid_dist
    return min(centroid_dist, fm_dist)
                                                                    

# ------------------------------Edited Image identification and Final edit region ------------------------------------------

def mask_overlap_arm_mask(mask1, mask2):
    """Compute area overlap ratio of mask1 over mask2."""
    intersection = np.logical_and(mask1, mask2).sum()
    area1 = mask1.sum()
    if area1 == 0:
        return 0.0
    return intersection / area1

def mask_overlap(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    # print(inter, union)
    return inter / union if union > 0 else 0

def get_edit_masks_advanced_distance_metric(matcher, frame_dir, img_dir, robotic_arm_dir, out_dir="./edit_masks"):
    os.makedirs(out_dir, exist_ok=True)

    t_load_start = time.perf_counter()
    
    # --- Load XMem masks (which define the range of frames processed) ---
    mask_files = natsorted([f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg"))])
    if len(mask_files) < 2:
        return
        
    # Pick first and middle from the successfully tracked masks
    first_mask_name = mask_files[0]
    middle_mask_name = mask_files[len(mask_files)//2]
    
    # --- Load original frames corresponding to these masks ---
    first_frame_path = os.path.join(frame_dir, first_mask_name.replace(".png", ".jpg"))
    middle_frame_path = os.path.join(frame_dir, middle_mask_name.replace(".png", ".jpg"))
    
    first_frame = _imread_rgb(first_frame_path)
    middle_frame = _imread_rgb(middle_frame_path)
    if first_frame is None or middle_frame is None:
        return

    # --- Load robotic arm mask for the FIRST frame in the tracked sequence ---
    # Arm masks are aligned with original JPEG names
    arm_mask_path = os.path.join(robotic_arm_dir, first_mask_name)
    base_arm_first = _imread_gray(arm_mask_path)
    if base_arm_first is None:
        return
    base_arm_first = (base_arm_first > 0).astype(np.uint8)
    
    # --- Load XMem color images (masks with object IDs) ---
    first_rgb = _imread_rgb(os.path.join(img_dir, first_mask_name))
    middle_rgb = _imread_rgb(os.path.join(img_dir, middle_mask_name))
    if first_rgb is None or middle_rgb is None:
        return
    
    # For iteration below
    images = [os.path.join(img_dir, f) for f in mask_files]
    # We also need a way to map 'images' index back to original 'frames' paths
    frames = [os.path.join(frame_dir, f.replace(".png", ".jpg")) for f in mask_files]
    
    dt_load = time.perf_counter() - t_load_start

    t_arm_start = time.perf_counter()
    # --- Find unique colors (object IDs) ---
    unique_colors = np.unique(first_rgb.reshape(-1, 3), axis=0)
    unique_colors = [tuple(c) for c in unique_colors if not np.all(c == 0)]

    # --- First guess of robotic arm colors from the first frame ---
    robotic_arm_colors = []
    for color in unique_colors:
        mask_first = np.all(first_rgb == color, axis=-1).astype(np.uint8)
        overlap_first = mask_overlap_arm_mask(mask_first, base_arm_first)
        if overlap_first > 0.25:
            robotic_arm_colors.append(color)
        
    dt_arm = time.perf_counter() - t_arm_start

    t_dist_start = time.perf_counter()
    # --- Compute movement distances for non-robotic-arm colors ---
    current_frames = {}
    current_rgbs = {}
    moved_distances = []
    max_moved_distances = {}
    target_object_candidate_colors = list(set(unique_colors) - set(robotic_arm_colors))

    for color in target_object_candidate_colors:
        mask_first = np.all(first_rgb == color, axis=-1).astype(np.uint8)
        for i in range(len(images)-1, int((3*len(images))/4), -1): # choose lower limit from either --> int((3*len(images))/4) or int(len(images)/2) or 1 
            if i not in current_rgbs:
                current_rgbs[i] = _imread_rgb(images[i])
            mask_current = np.all(current_rgbs[i] == color, axis=-1).astype(np.uint8)
            
            # --- Area consistency check ---
            area_first = np.sum(mask_first)
            area_current = np.sum(mask_current)

            if area_first > 0 and area_current > 0:
                if i not in current_frames:
                    current_frames[i] = _imread_rgb(frames[i])
                dist = get_distance(matcher, first_frame, current_frames[i], mask_first, mask_current)
                if dist != -1:
                    if color in max_moved_distances:
                        max_moved_distances[color] = max(max_moved_distances[color], dist)
                    else:
                        max_moved_distances[color] = dist
        mask_middle = np.all(middle_rgb == color, axis=-1).astype(np.uint8)
        for i in range(len(images)-1, int((3*len(images))/4), -1): # choose lower limit from either --> int((3*len(images))/4) or int(len(images)/2) or 1 
            if i not in current_rgbs:
                current_rgbs[i] = _imread_rgb(images[i])
            mask_current = np.all(current_rgbs[i] == color, axis=-1).astype(np.uint8)
            
            # --- Area consistency check ---
            area_middle = np.sum(mask_middle)
            area_current = np.sum(mask_current)

            if area_middle > 0 and area_current > 0:
                if i not in current_frames:
                    current_frames[i] = _imread_rgb(frames[i])
                dist = get_distance(matcher, middle_frame, current_frames[i], mask_middle, mask_current)
                if dist != -1:
                    if color in max_moved_distances:
                        max_moved_distances[color] = max(max_moved_distances[color], dist)
                    else:
                        max_moved_distances[color] = dist
        if color in max_moved_distances:
            moved_distances.append((color, max_moved_distances[color]))
        else:
            moved_distances.append((color, 0.0))
    dt_dist = time.perf_counter() - t_dist_start
    
    t_cluster_start = time.perf_counter()
    # --- Sort by motion distance ---
    moved_distances = sorted(moved_distances, key=lambda x: x[1], reverse=True)
    # print("All detected mask movements:", moved_distances)
    
    # --- target object colors ---
    target_colors = []
    
    # Handle edge cases where clustering is not possible
    if len(moved_distances) == 0:
        # No masks detected - nothing to do
        pass
    elif len(moved_distances) == 1:
        # Only one mask - treat it as moved if it has any significant distance
        color, dist = moved_distances[0]
        if dist > 0:
            target_colors.append(color)
    elif len(moved_distances) < 3:
        # Too few samples for reliable clustering - use simple heuristic
        # Take the top mover(s) if they have significant movement
        for color, dist in moved_distances:
            if dist > 0:
                target_colors.append(color)
    else:
        # Enough samples for clustering
        # Extract just the distance values for clustering
        # Reshape is needed because sklearn expects a 2D array (list of lists)
        distances = np.array([item[1] for item in moved_distances]).reshape(-1, 1)

        # Fit K-Means with 2 clusters (Moved vs. Stationary)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(distances)

        # Identify which cluster represents the "moved" (higher distance) group
        # We look at the cluster centers to see which one is larger
        moved_cluster_label = np.argmax(kmeans.cluster_centers_)

        # Filter the original list to keep only items in that cluster
        moved = [
            item for i, item in enumerate(moved_distances) 
            if kmeans.labels_[i] == moved_cluster_label
        ]

        # print("Top moved masks:", moved)

        for color, _ in moved:
            target_colors.append(color)
    dt_cluster = time.perf_counter() - t_cluster_start

    t_edit_start = time.perf_counter()
    # --- Detect edit frame ---
    overlap_started = False
    edited_idx = None

    for i, img_path in enumerate(images):
        rgb = _imread_rgb(img_path)
        if rgb is None:
            continue

        # Construct arm and target masks using colors
        arm_mask = np.zeros_like(base_arm_first, dtype=np.uint8)
        for color in robotic_arm_colors:
            arm_mask |= np.all(rgb == color, axis=-1).astype(np.uint8)
        target_mask = np.zeros_like(arm_mask, dtype=np.uint8)
        for color in target_colors:
            target_mask |= np.all(rgb == color, axis=-1).astype(np.uint8)

        # Add dilation
        kernel = np.ones((5, 5), np.uint8)
        arm_mask = cv2.dilate(arm_mask, kernel, iterations=1)
        target_mask = cv2.dilate(target_mask, kernel, iterations=1)

        overlap = mask_overlap(arm_mask, target_mask)
        overlap_threshold = 0

        if not overlap_started and overlap > overlap_threshold:
            overlap_started = True # overlap starts
        elif overlap_started and overlap <= overlap_threshold:
            edited_idx = i-1  # overlap ends
            overlap_started = False
        elif overlap_started and overlap > overlap_threshold:
            edited_idx = i

    if edited_idx is None:
        edited_idx = len(images) - 1
        
    final_frame_num = os.path.splitext(os.path.basename(images[edited_idx]))[0]

    dt_edit = time.perf_counter() - t_edit_start

    t_save_start = time.perf_counter()
    # --- Save masks ---
    init_rgb = _imread_rgb(images[0])
    final_rgb = _imread_rgb(images[edited_idx])
    if init_rgb is None or final_rgb is None:
        return

    # Construct arm + target combined masks
    init_mask = np.zeros(init_rgb.shape[:2], dtype=np.uint8)
    final_mask = np.zeros(final_rgb.shape[:2], dtype=np.uint8)

    for color in robotic_arm_colors + target_colors:
        init_mask |= np.all(init_rgb == color, axis=-1).astype(np.uint8)
        final_mask |= np.all(final_rgb == color, axis=-1).astype(np.uint8)

    # --- Save binary masks and frame number ---
    cv2.imwrite(os.path.join(out_dir, "initial_edit_region_mask.png"), init_mask * 255)
    cv2.imwrite(os.path.join(out_dir, "final_edit_region_mask.png"), final_mask * 255)
    with open(os.path.join(out_dir, "edited_frame_number.txt"), "w") as f:
        f.write(str(final_frame_num))
    dt_save = time.perf_counter() - t_save_start
    
    with stats_lock:
        LOFTR_STATS['load_data'] += dt_load
        LOFTR_STATS['arm_refinement'] += dt_arm
        LOFTR_STATS['distance_computation'] += dt_dist
        LOFTR_STATS['kmeans'] += dt_cluster
        LOFTR_STATS['edit_detection'] += dt_edit
        LOFTR_STATS['save_results'] += dt_save
        LOFTR_STATS['total_subdirs'] += 1

loftr_matcher = None

def init_loftr():
    global loftr_matcher
    if loftr_matcher is None:
        weight_map = {
            "indoor_ds_new": os.path.join(LOFTR_WEIGHTS_DIR, "indoor_ds_new.ckpt"),
            "indoor_ot":     os.path.join(LOFTR_WEIGHTS_DIR, "indoor_ot.ckpt"),
            "outdoor_ds":    os.path.join(LOFTR_WEIGHTS_DIR, "outdoor_ds.ckpt"),
            "outdoor_ot":    os.path.join(LOFTR_WEIGHTS_DIR, "outdoor_ot.ckpt"),
        }
        weight_path = weight_map[args.weight_type]
        loftr_matcher = load_loftr_model(args.weight_type, weight_path, device)
    return loftr_matcher

def run_stage_loftr_masking(batch_subdirs):
    print(f"[INFO] Stage 3: Generating Edit Region Masks for {len(batch_subdirs)} videos...")
    
    # make sure base edit_masks dir exists
    os.makedirs(EDIT_MASKS_BASE, exist_ok=True)
    os.makedirs(EDIT_MASKS_DIR, exist_ok=True)

    matcher = init_loftr()

    def loftr_wrapper(subdir):
        track_dir = os.path.join(XMEM_OUTPUT_DIR, subdir)
        robotic_mask_dir = os.path.join(ROBOTICMASK_DIR, subdir)
        frame_dir = os.path.join(IMAGE_DIR, subdir)
        if os.path.isdir(track_dir):
            out_dir = os.path.join(EDIT_MASKS_DIR, subdir)
            os.makedirs(out_dir, exist_ok=True)
            try:
                get_edit_masks_advanced_distance_metric(matcher, frame_dir, track_dir, robotic_mask_dir, out_dir=out_dir)
            except Exception as e:
                print(f"[ERROR] LoFTR processing failed for {subdir}: {e}")

    num_workers = min(4, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(loftr_wrapper, batch_subdirs), 
                  total=len(batch_subdirs), 
                  desc="Generating LoFTR Masks"))

def run_stage_post_loftr_cleanup():
    print("[INFO] >>> Cleaning up Stage 3 (LoFTR) memory...")
    
    if 'loftr_matcher' in globals():
        _safe_to_cpu(globals().get('loftr_matcher'), name='loftr_matcher')

    _globs = globals()
    for name in ["loftr_matcher"]:
        _safe_del(name, _globs)

    run_cuda_cleanup()
    print("[INFO] >>> Stage 3 memory cleanup completed.")

def run_stage_hf_upload(batch_subdirs, batch_idx):
    print(f"[INFO] Stage 4: Uploading Batch {batch_idx} ({len(batch_subdirs)} videos) to HF...")
    
    # Predefine features for consistency
    features = Features({
        "id": Value("string"),
        "original_image": DatasetsImage(),
        "edited_image": DatasetsImage(),
        "initial_edit_region_mask": DatasetsImage(),
        "final_edit_region_mask": DatasetsImage(),
        "edit_instruction": Value("string"),
    })

    samples = []
    # img_dir, mask_dir, instruct_dir are global WORK_* dirs (or DATA_DIR if aliased)
    # val/test splits can be handled per batch or ignored if small
    
    # We'll use the existing process_single_subdir logic but collect in-memory for the batch
    # Since batch_size is small (e.g. 50), this should fit in RAM.
    
    num_workers = min(10, (os.cpu_count() or 1) + 4)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_subdir = {
            executor.submit(
                process_single_subdir, 
                subdir, IMAGE_DIR, EDIT_MASKS_DIR, EDIT_INSTRUCTION_DIR, 
                val_split=0.05, test_split=0.05, seed=42
            ): subdir for subdir in batch_subdirs
        }
        
        for future in tqdm(as_completed(future_to_subdir), total=len(batch_subdirs), desc="Preparing Samples"):
            result = future.result()
            if result:
                # result is (split_name, sample)
                _, sample = result
                samples.append(sample)

    if not samples:
        print("[WARN] No valid samples found for this batch.")
        return

    ds = Dataset.from_list(samples, features=features)
    
    # Push to hub. By default, push_to_hub with split='train' will create/append.
    # To get 'train-i', we can use the 'name' parameter or just let HF handle sharding.
    # However, user specifically asked for 'train-i'.
    # A reliable way is to upload to a specific path using HfApi.
    
    api = HfApi()
    
    # Create temp parquet
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        ds.to_parquet(tmp.name)
        tmp_path = tmp.name
    
    repo_id = HF_REPO_ID
    token = HF_TOKEN
    
    # Upload to data/train-<batch_idx>.parquet
    target_path = f"data/train-{batch_idx:05d}-of-unknown.parquet" 
    # (The 'of-unknown' suffix is optional but follows common naming)
    # Let's just use a simpler one:
    target_path = f"data/train-{batch_idx:05d}.parquet"

    print(f"[INFO] Uploading batch to {target_path} in {repo_id}...")
    
    while True:
        try:
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=target_path,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message=f"Upload batch {batch_idx} ({len(samples)} samples)"
            )
            break # Success
        except Exception as e:
            print(f"[ERROR] Upload failed for batch {batch_idx}: {e}")
            print(f"[INFO] Retrying in 30 seconds...")
            time.sleep(30)
    
    os.remove(tmp_path)
    print(f"[INFO] Batch {batch_idx} uploaded successfully.")

def run_stage_cleanup(batch_subdirs):
    if not args.cleanup:
        return
    print(f"[INFO] Stage 5: Cleaning up batch files to save disk space...")
    
    # Paths to cleanup for each subdir in batch
    dirs_to_clean = [
        ANNOTATION_DIR, XMEM_OUTPUT_DIR, 
        EDIT_MASKS_DIR, ROBOTICMASK_DIR
    ]
    
    for subdir in batch_subdirs:
        for d in dirs_to_clean:
            target = os.path.join(d, subdir)
            if os.path.exists(target):
                if os.path.islink(target):
                    try:
                        os.remove(target)
                    except Exception:
                        pass
                elif os.path.isdir(target):
                    shutil.rmtree(target, ignore_errors=True)
    
    print(f"[INFO] Cleanup completed for {len(batch_subdirs)} subdirectories.")

def process_single_subdir(subdir, img_dir, mask_dir, instruct_dir, val_split, test_split, seed):
    """
    Worker function to process a single subdirectory.
    """
    try:
        img_subdir_path = os.path.join(img_dir, subdir)
        mask_subdir_path = os.path.join(mask_dir, subdir)
        instruct_subdir_path = os.path.join(instruct_dir, subdir)

        # Use the first frame where arm was detected as the initial image
        frame_name = "00000.jpg"
        if subdir in first_detected_indices:
            frame_name = first_detected_indices[subdir].get("frame", "00000.jpg")
        
        init_img_path = os.path.join(img_subdir_path, frame_name)
        edited_frame_number_path = os.path.join(mask_subdir_path, "edited_frame_number.txt")
        instruction_path = os.path.join(instruct_subdir_path, "edit_instruction.txt")

        if not os.path.exists(edited_frame_number_path) or not os.path.exists(instruction_path):
            return None

        with open(edited_frame_number_path, "r") as f:
            try:
                frame_num = int(f.read().strip())
            except Exception:
                return None

        edited_img_path = os.path.join(img_subdir_path, f"{frame_num:05d}.jpg")
        init_mask_path = os.path.join(mask_subdir_path, "initial_edit_region_mask.png")
        final_mask_path = os.path.join(mask_subdir_path, "final_edit_region_mask.png")

        if not (
            os.path.exists(init_img_path)
            and os.path.exists(edited_img_path)
            and os.path.exists(init_mask_path)
            and os.path.exists(final_mask_path)
        ):
            return None

        # Verify images are not corrupt
        for p in [init_img_path, edited_img_path]:
            try:
                with Image.open(p) as img:
                    img.load() # This actually decodes and checks
            except Exception:
                log_corrupt_image(p)
                return None

        # read instruction
        with open(instruction_path, "r", encoding="utf-8") as f:
            instruction_text = f.read().strip()

        # Read files directly as bytes
        with open(init_img_path, "rb") as f:
            orig_bytes = f.read()
        with open(edited_img_path, "rb") as f:
            edited_bytes = f.read()
        with open(init_mask_path, "rb") as f:
            init_mask_bytes = f.read()
        with open(final_mask_path, "rb") as f:
            final_mask_bytes = f.read()

        # Assign split
        m = hashlib.md5()
        m.update(subdir.encode("utf-8"))
        m.update(str(seed).encode("utf-8"))
        hv = int.from_bytes(m.digest()[:8], "big")
        frac = hv / (2**64 - 1)
        if frac < test_split:
            split_name = "test"
        elif frac < test_split + val_split:
            split_name = "validation"
        else:
            split_name = "train"

        sample = {
            "id": subdir,
            "original_image": orig_bytes,
            "edited_image": edited_bytes,
            "initial_edit_region_mask": init_mask_bytes,
            "final_edit_region_mask": final_mask_bytes,
            "edit_instruction": instruction_text,
        }
        return (split_name, sample)

    except Exception:
        return None

def ensure_hf_repo_exists(repo_id, hf_token):
    """
    Ensures the repository exists on Hugging Face.
    """
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, token=hf_token, repo_type="dataset", exist_ok=True)
        print(f"[INFO] Verified HF repo: {repo_id}")
    except Exception as e:
        print(f"[WARN] Could not create/verify HF repo: {e}")

# -------------------------
# Main Execution Loop
# -------------------------
def main():
    # Initial setup
    ensure_hf_repo_exists(repo_id=HF_REPO_ID, hf_token=HF_TOKEN)
    
    # Get all episodes
    all_episodes = sorted([d for d in os.listdir(os.path.join(BASE_DIR, args.dataset_name, "JPEGImages")) 
                           if os.path.isdir(os.path.join(BASE_DIR, args.dataset_name, "JPEGImages", d))])
    
    total_episodes = len(all_episodes)
    batch_size = args.batch_size or total_episodes
    
    start_batch = args.resume - 1
    start_idx = start_batch * batch_size

    print(f"[INFO] Total episodes to process: {total_episodes}")
    print(f"[INFO] Using batch size: {batch_size}")
    if start_batch > 0:
        print(f"[INFO] Resuming from Batch {args.resume} (Starting index: {start_idx})")

    for i in range(start_idx, total_episodes, batch_size):
        batch_idx = i // batch_size
        batch_subdirs = all_episodes[i : i + batch_size]
        print(f"\n[INFO] Starting Batch {batch_idx+1} (Episodes {i} to {min(i + batch_size, total_episodes)})...")
        
        # Stage 0: Robotic arm mask extraction
        run_stage_robotic_arm_extraction(batch_subdirs)
        
        # Free memory before starting SAM
        run_stage_post_evf_cleanup()
        
        # Stage 1: SAM inference
        run_stage_sam_inference(batch_subdirs)
        
        # Free memory before starting XMem
        run_stage_post_sam_cleanup()
        
        # Stage 2: XMem tracking
        run_stage_xmem_tracking(batch_subdirs)
        
        # Free memory before starting LoFTR
        run_stage_post_xmem_cleanup()
        
        # Stage 3: LoFTR Masking
        run_stage_loftr_masking(batch_subdirs)
        
        # Free memory before starting Upload
        run_stage_post_loftr_cleanup()
        
        # Stage 4: Upload to HF
        run_stage_hf_upload(batch_subdirs, batch_idx)
        
        # Stage 5: Cleanup
        run_stage_cleanup(batch_subdirs)

    print("\n" + "="*50)
    print("ALL BATCHES PROCESSED SUCCESSFULLY")
    print("="*50)
    
    # --- LoFTR Profiling Report ---
    print("\n" + "="*50)
    print("LOFTR STAGE PROFILING REPORT")
    print("="*50)
    num_dirs = LOFTR_STATS['total_subdirs'] or 1
    total_match_calls = LOFTR_STATS['match_visible_count'] or 1

    print(f"Total Subdirectories Processed: {LOFTR_STATS['total_subdirs']}")
    print(f"Total match_visible_correspondences calls: {LOFTR_STATS['match_visible_count']}")

    print("\n--- Average Time Per Subdirectory ---")
    print(f"  Load Data:           {LOFTR_STATS['load_data']/num_dirs:.4f}s")
    print(f"  Arm Refinement:      {LOFTR_STATS['arm_refinement']/num_dirs:.4f}s")
    print(f"  Distance Computation: {LOFTR_STATS['distance_computation']/num_dirs:.4f}s")
    print(f"  KMeans Clustering:   {LOFTR_STATS['kmeans']/num_dirs:.4f}s")
    print(f"  Edit Detection:      {LOFTR_STATS['edit_detection']/num_dirs:.4f}s")
    print(f"  Save Results:        {LOFTR_STATS['save_results']/num_dirs:.4f}s")

    print("\n--- Average Time Per match_visible_correspondences Call ---")
    print(f"  Crop & Offset:       {LOFTR_STATS['match_visible_crop']/total_match_calls:.4f}s")
    print(f"  Pre-processing:      {LOFTR_STATS['match_visible_prep']/total_match_calls:.4f}s")
    print(f"  Model Inference:     {LOFTR_STATS['match_visible_inf']/total_match_calls:.4f}s")
    print(f"  Post-processing:     {LOFTR_STATS['match_visible_post']/total_match_calls:.4f}s")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
