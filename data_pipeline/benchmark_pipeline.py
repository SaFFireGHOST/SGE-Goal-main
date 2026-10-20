"""
Command to run the file (lgie_dp conda env): 
PYTHONPATH=./XMem python3 benchmark_pipeline.py --batch_size 100 --cleanup
"""

# ---------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------
import argparse
import concurrent.futures
import math
import colorsys
import os
import random
import shutil
import sys
import tempfile
import time
import hashlib
import gc
import threading
import json

# Ensure local segment-anything and XMem are used
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SAM_PATH = os.path.join(_BASE_DIR, "segment-anything")
_XMEM_PATH = os.path.join(_BASE_DIR, "XMem")

for _p in [_SAM_PATH, _XMEM_PATH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import path
from typing import List, Optional

# ---------------------------------------------------------------------
# Scientific / Core Libraries
# ---------------------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Image and Visualization
# ---------------------------------------------------------------------
import cv2
from PIL import Image, Image as PILImage

# ---------------------------------------------------------------------
# Progress, Sorting, and Utilities
# ---------------------------------------------------------------------
from natsort import natsorted
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
# Transformers Core
# ---------------------------------------------------------------------
from transformers import (
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor
)

# ---------------------------------------------------------------------
# XMem / Project-Specific Imports
# ---------------------------------------------------------------------
from inference.data.mask_mapper import MaskMapper
from inference.data.test_datasets import LongTestDataset
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
from torchvision.transforms.functional import InterpolationMode, resize, to_pil_image
try:
    from torchvision.io import read_image, ImageReadMode
    _HAS_TVIO = True
except Exception:
    read_image = None
    ImageReadMode = None
    _HAS_TVIO = False

import traceback

# =====================================================================
# Global Timing Stats Containers
# =====================================================================
stage0_times = []
stage1_times = []
stage2_times = []
stage3_times = []

def _imread_rgb(path_str: str) -> Optional[np.ndarray]:
    use_tvio = os.environ.get("DP_USE_TORCHVISION_IO", "1") == "1"
    if _HAS_TVIO and use_tvio and read_image is not None and ImageReadMode is not None:
        try:
            t = read_image(path_str, mode=ImageReadMode.RGB)
            return t.permute(1, 2, 0).contiguous().numpy()
        except Exception:
            pass

    img_bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if img_bgr is None:
        if os.path.exists(path_str):
            log_corrupt_image(path_str)
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _imread_gray(path_str: str) -> Optional[np.ndarray]:
    use_tvio = os.environ.get("DP_USE_TORCHVISION_IO", "1") == "1"
    if _HAS_TVIO and use_tvio and read_image is not None and ImageReadMode is not None:
        try:
            t = read_image(path_str, mode=ImageReadMode.GRAY)
            return t[0].contiguous().numpy()
        except Exception:
            pass

    img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    if img is None:
        if os.path.exists(path_str):
            log_corrupt_image(path_str)
    return img

def _imread_tensor(path_str: str, device="cuda") -> Optional[torch.Tensor]:
    use_tvio = os.environ.get("DP_USE_TORCHVISION_IO", "1") == "1"
    if _HAS_TVIO and use_tvio and read_image is not None and ImageReadMode is not None:
        try:
            t = read_image(path_str, mode=ImageReadMode.RGB)
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


def _configure_runtime_threads():
    cpu_count = os.cpu_count() or 1
    multicore = os.environ.get("DP_MULTICORE", "0") == "1"
    try:
        worker_hint = 1
        worker_hint = max(worker_hint, int(os.environ.get("DP_EVF_WORKERS", "2")))
        worker_hint = max(worker_hint, int(os.environ.get("DP_XMEM_NUM_WORKERS", "2")))
        worker_hint = max(1, worker_hint)
    except Exception:
        worker_hint = 1

    try:
        disable_cv_threads_default = "0" if multicore else "1"
        disable_cv_threads = os.environ.get("DP_DISABLE_OPENCV_THREADS", disable_cv_threads_default) != "0"
        if disable_cv_threads:
            cv2.setNumThreads(0)
        else:
            auto_cv = max(1, min(8, cpu_count // worker_hint))
            cv_threads = int(os.environ.get("DP_OPENCV_NUM_THREADS", str(auto_cv)))
            cv2.setNumThreads(max(1, cv_threads))
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

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
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def _configure_gpu_perf():
    if not torch.cuda.is_available():
        return
    enable_tf32 = os.environ.get("DP_TF32", "0") == "1"
    if enable_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
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
parser.add_argument('--cleanup', action='store_true', default=True, help="Cleanup batch files after processing")

# XMem options
parser.add_argument('--xmem_model', default=os.path.join(os.path.dirname(__file__), "XMem/saves/XMem.pth"))
parser.add_argument('--size', default=480, type=int, help='Resize the shorter side to this size.')
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

device = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_PALETTE = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)] * 45 

def generate_color_palette(num_colors=256):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        lightness = 0.5 + 0.4 * ((i % 2) - 0.5) 
        saturation = 0.9
        r, g, b = [int(x * 255) for x in colorsys.hls_to_rgb(hue, lightness, saturation)]
        if (r, g, b) not in [(0, 0, 0), (255, 255, 255)]:
            colors.append((r, g, b))
    flat_palette = [v for c in colors[:256] for v in c]
    flat_palette.extend([0] * (768 - len(flat_palette)))
    return flat_palette

# -------------------------- EVF-SAM Inference for Robotic Arm Mask -------------------------------------------
EVF_SAM_VERSION = "YxZhang/evf-sam"
EVF_SAM_PRECISION="fp16" 
EVF_SAM_MODEL_TYPE = "ori"
BLIP_MODEL = "Salesforce/blip2-flan-t5-xl"
ROBOT_OBJECT_DESCRIPTION = {"evfsam": "robotic arm and gripper", "blip": "robotic gripper"}
confidence_threshold = 0.0

def init_models():
    tokenizer = AutoTokenizer.from_pretrained(EVF_SAM_VERSION, padding_side="right", use_fast=False)
    torch_dtype = torch.float32
    if EVF_SAM_PRECISION == "bf16": torch_dtype = torch.bfloat16
    elif EVF_SAM_PRECISION == "fp16": torch_dtype = torch.half
    kwargs = {"torch_dtype": torch_dtype}

    from evf_sam_model.evf_sam import EvfSamModel
    model = EvfSamModel.from_pretrained(EVF_SAM_VERSION, low_cpu_mem_usage=True, **kwargs)
    model = model.cuda().eval()
    return tokenizer, model

def sam_preprocess_gpu(x_tensor, pixel_mean, pixel_std, img_size=1024, model_type="ori"):
    h, w = x_tensor.shape[-2:]
    scale = img_size * 1.0 / max(h, w)
    newh, neww = int(h * scale + 0.5), int(w * scale + 0.5)
    
    x = F.interpolate(x_tensor.unsqueeze(0), size=(newh, neww), mode='bilinear', align_corners=False, antialias=True).squeeze(0)
    x = (x - pixel_mean.to(x.device)) / pixel_std.to(x.device)
    
    pad_h = img_size - newh
    pad_w = img_size - neww
    x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (newh, neww)

def beit3_preprocess_gpu(x_tensor, img_size=224):
    x = x_tensor / 255.0
    x = F.interpolate(x.unsqueeze(0), size=(img_size, img_size), mode='bicubic', align_corners=False, antialias=True).squeeze(0)
    x = (x - 0.5) / 0.5
    return x

def init_blip(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Blip2Processor.from_pretrained(model_name, trust_remote_code=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    ).eval()
    return processor, model, device

def crop_from_mask_gpu(image_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> Optional[Image.Image]:
    nonzero = torch.nonzero(mask_tensor)
    if nonzero.shape[0] == 0: return None
    y1, y2 = nonzero[:, 0].min().item(), nonzero[:, 0].max().item()
    x1, x2 = nonzero[:, 1].min().item(), nonzero[:, 1].max().item()
    crop_img = image_tensor[:, y1:y2+1, x1:x2+1].clone()
    crop_mask = mask_tensor[y1:y2+1, x1:x2+1]
    crop_img = crop_img * crop_mask.unsqueeze(0)
    crop_img_np = crop_img.permute(1, 2, 0).byte().cpu().numpy()
    return Image.fromarray(crop_img_np)

def crop_from_mask_gpu_bbox(image_rgb: np.ndarray, mask_tensor: torch.Tensor) -> Optional[Image.Image]:
    nonzero = torch.nonzero(mask_tensor)
    if nonzero.shape[0] == 0: return None
    y1, y2 = int(nonzero[:, 0].min().item()), int(nonzero[:, 0].max().item())
    x1, x2 = int(nonzero[:, 1].min().item()), int(nonzero[:, 1].max().item())
    crop_img = image_rgb[y1:y2 + 1, x1:x2 + 1].copy()
    crop_mask = mask_tensor[y1:y2 + 1, x1:x2 + 1].detach().to("cpu").numpy().astype(bool)
    crop_img[~crop_mask] = 0
    return Image.fromarray(crop_img)

@torch.no_grad()
def verify_mask_with_blip(
    blip_processor, blip_model, blip_device, image_input, mask_input, text_prompt: str, use_gpu_crop=False
) -> bool:
    if use_gpu_crop:
        if isinstance(image_input, np.ndarray) and isinstance(mask_input, torch.Tensor):
            crop = crop_from_mask_gpu_bbox(image_input, mask_input)
        else: crop = crop_from_mask_gpu(image_input, mask_input)
    else: crop = None
        
    if crop is None: return False
    if os.environ.get("DP_DISABLE_BLIP_VERIFY", "0") == "1": return True

    question = f"Question: What object does the given image represent? Answer with yes or no. Is it a {text_prompt}?"
    inputs = blip_processor(images=crop, text=question, return_tensors="pt")
    inputs = {k: (v.to(blip_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

    outputs = blip_model.generate(**inputs, max_new_tokens=32, do_sample=False)
    answer = blip_processor.batch_decode(outputs, skip_special_tokens=True)[0].lower().strip()
    return "yes" in answer

@torch.no_grad()
def verify_masks_with_blip_batch(blip_processor, blip_model, blip_device, images_rgb, masks_gpu, text_prompt) -> List[bool]:
    question = f"Question: What object does the given image represent? Answer with yes or no. Is it a {text_prompt}?"
    crops, valid_idx = [], []
    for i, (img_rgb, msk) in enumerate(zip(images_rgb, masks_gpu)):
        crop = crop_from_mask_gpu_bbox(img_rgb, msk)
        crops.append(crop)
        if crop is not None: valid_idx.append(i)

    results = [False] * len(images_rgb)
    if not valid_idx: return results
    if os.environ.get("DP_DISABLE_BLIP_VERIFY", "0") == "1":
        for i in valid_idx: results[i] = True
        return results

    valid_crops = [crops[i] for i in valid_idx]
    texts = [question] * len(valid_crops)
    inputs = blip_processor(images=valid_crops, text=texts, return_tensors="pt", padding=True)
    inputs = {k: (v.to(blip_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    if "pixel_values" in inputs: inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

    outputs = blip_model.generate(**inputs, max_new_tokens=32, do_sample=False)
    answers = [a.lower().strip() for a in blip_processor.batch_decode(outputs, skip_special_tokens=True)]
    for idx, ans in zip(valid_idx, answers): results[idx] = ("yes" in ans)
    return results

def extract_segmentation_masks_batch(
    image_paths: List[str], object_description: dict, output_paths: List[str],
    evf_ctx: dict, blip_processor, blip_model, blip_device
) -> List[float]:
    model = evf_ctx["model"]
    device = evf_ctx["device"]
    with ThreadPoolExecutor(max_workers=min(len(image_paths), 8)) as executor:
        images_rgb = list(executor.map(_imread_rgb, image_paths))

    original_sizes = [img.shape[:2] if img is not None else None for img in images_rgb]
    image_sam_list, image_beit_list, resize_shapes, valid_indices = [], [], [], []

    for i, img_rgb in enumerate(images_rgb):
        if img_rgb is None: continue
        image_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device=device, dtype=torch.float32)
        image_beit = evf_ctx["beit3_preprocess_gpu"](image_tensor, img_size=evf_ctx["img_size"]).to(dtype=model.dtype, device=device)
        image_sam, resize_shape = evf_ctx["sam_preprocess_gpu"](
            image_tensor, pixel_mean=evf_ctx["sam_pixel_mean"], pixel_std=evf_ctx["sam_pixel_std"], model_type=evf_ctx["model_type"]
        )
        image_sam_list.append(image_sam.to(dtype=model.dtype, device=device))
        image_beit_list.append(image_beit)
        resize_shapes.append(resize_shape)
        valid_indices.append(i)

    confs = [0.0] * len(image_paths)
    if not valid_indices: return confs

    image_sam_batch = torch.stack(image_sam_list, dim=0)
    image_beit_batch = torch.stack(image_beit_list, dim=0)
    del image_sam_list, image_beit_list

    input_ids = evf_ctx["evfsam_input_ids"]
    if input_ids.dim() == 2 and input_ids.shape[0] == 1:
        input_ids = input_ids.expand(image_sam_batch.shape[0], -1)

    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            pred_masks = model.inference(image_sam_batch, image_beit_batch, input_ids, resize_list=resize_shapes, original_size_list=[original_sizes[i] for i in valid_indices])
        del image_sam_batch, image_beit_batch

        preds = []
        if isinstance(pred_masks, torch.Tensor):
            pm = pred_masks[:, 0] if pred_masks.dim() == 4 and pred_masks.shape[1] == 1 else pred_masks
            preds = [(pm[j] > 0) for j in range(pm.shape[0])]
        elif isinstance(pred_masks, (list, tuple)):
            for m in pred_masks:
                mt = m[0] if m.dim() == 3 and m.shape[0] == 1 else m
                preds.append(mt > 0)
        del pred_masks
        preds = (preds + [None] * len(valid_indices))[: len(valid_indices)]
    except Exception as e:
        preds = [None for _ in valid_indices]

    to_verify_imgs, to_verify_masks, verify_map = [], [], []
    for local_j, global_i in enumerate(valid_indices):
        if preds[local_j] is not None:
            to_verify_imgs.append(images_rgb[global_i])
            to_verify_masks.append(preds[local_j])
            verify_map.append(global_i)

    oks = []
    if verify_map:
        try:
            oks = verify_masks_with_blip_batch(
                blip_processor, blip_model, blip_device,
                images_rgb=[images_rgb[i] for i in verify_map],
                masks_gpu=to_verify_masks, text_prompt=object_description["blip"],
            )
        except Exception: oks = [False] * len(verify_map)

    ok_by_idx = {i: ok for i, ok in zip(verify_map, oks)}

    def _save_mask_worker(args):
        local_j, global_i = args
        pred, out_path, is_ok = preds[local_j], output_paths[global_i], ok_by_idx.get(global_i, False)
        if pred is None or not is_ok:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            Image.new("RGB", (640, 480), (0, 0, 0)).save(out_path)
            return 0.0
        else:
            mask_uint8 = (pred.byte().cpu().numpy() * 255).astype(np.uint8)
            out = Image.fromarray(mask_uint8, mode="L").convert("RGB")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            out.save(out_path)
            return 1.0

    with ThreadPoolExecutor(max_workers=min(len(valid_indices), 8)) as executor:
        results = list(executor.map(_save_mask_worker, [(lj, gi) for lj, gi in enumerate(valid_indices)]))

    for idx, global_i in enumerate(valid_indices): confs[global_i] = results[idx]
    return confs

def is_robotic_arm_detected(mask_confidence, confidence_threshold):
    return mask_confidence > confidence_threshold, mask_confidence

def process_video(item, evf_ctx, blip_processor, blip_model, blip_device):
    src_path = os.path.join(IMAGE_DIR, item)
    dest_path = os.path.join(ROBOTICMASK_DIR, item)
    os.makedirs(dest_path, exist_ok=True)

    sorted_files = natsorted([f for f in os.listdir(src_path) if f.lower().endswith(".jpg")])
    subset_count = max(1, int(0.2 * len(sorted_files)))
    selected_files = sorted_files[:subset_count]
    detections = []

    evf_batch = int(os.environ.get("DP_EVF_BATCH", "8"))
    for j in range(0, len(selected_files), evf_batch):
        batch_files = selected_files[j:j + evf_batch]
        image_paths = [os.path.join(src_path, f) for f in batch_files]
        output_paths = [os.path.join(dest_path, f"{os.path.splitext(f)[0]}.png") for f in batch_files]
        confs = extract_segmentation_masks_batch(
            image_paths, ROBOT_OBJECT_DESCRIPTION, output_paths,
            evf_ctx, blip_processor, blip_model, blip_device,
        )
        for conf in confs:
            det, val = is_robotic_arm_detected(conf, confidence_threshold)
            detections.append((det, round(val, 4)))

    detected_index = None
    if len(detections) == 1:
        if detections[0][0]: detected_index = 0
    else:
        for i in range(len(detections) - 1):
            if detections[i][0] and detections[i + 1][0]:
                detected_index = i
                break

    if detected_index is not None:
        filename = selected_files[detected_index]
        first_detected_indices[item] = {
            "frame": filename, "index": detected_index,
            "confidence": [detections[detected_index][1]],
        }
        for j in range(detected_index + 1, len(detections)):
            first_detected_indices[item]["confidence"].append(detections[j][1])
    else:
        videos_without_detection.append(item)

tokenizer, model, blip_processor, blip_model, blip_device, evf_ctx = None, None, None, None, None, None

def _safe_to_cpu(obj, name=None):
    if obj is None: return
    try: obj.to("cpu") if hasattr(obj, "to") else obj.cpu()
    except Exception: pass

def _safe_del(varname, glbs):
    if varname in glbs: glbs[varname] = None

def run_cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        try: torch.cuda.empty_cache()
        except Exception: pass

def init_evf_and_blip():
    global tokenizer, model, blip_processor, blip_model, blip_device, evf_ctx
    if model is None: tokenizer, model = init_models()
    if blip_model is None: blip_processor, blip_model, blip_device = init_blip(BLIP_MODEL)
    if evf_ctx is None:
        evf_ctx = {
            "model": model, "tokenizer": tokenizer, "device": model.device, "img_size": 224, "model_type": EVF_SAM_MODEL_TYPE,
            "sam_preprocess_gpu": sam_preprocess_gpu, "beit3_preprocess_gpu": beit3_preprocess_gpu,
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
        t0 = time.perf_counter()
        try:
            process_video(item, ctx, b_proc, b_mod, b_dev)
            elapsed = time.perf_counter() - t0
            return item, elapsed
        except Exception as e:
            print(f"[ERROR] Failed processing {item}: {e}")
            return item, None

    evf_workers = int(os.environ.get("DP_EVF_WORKERS", "1"))
    with ThreadPoolExecutor(max_workers=evf_workers) as executor:
        results = list(tqdm(executor.map(process_video_wrapper, batch_subdirs), total=len(batch_subdirs), desc="Finding robotic arm masks"))
        for _, t in results:
            if t is not None:
                stage0_times.append(t)
    
    run_cuda_cleanup()

def run_stage_post_evf_cleanup():
    _safe_to_cpu(globals().get('model'))
    _safe_to_cpu(globals().get('blip_model'))
    _globs = globals()
    for name in ["model", "tokenizer", "blip_model", "blip_processor", "blip_device", "evf_ctx"]: _safe_del(name, _globs)
    run_cuda_cleanup()

def run_stage_post_sam_cleanup():
    _safe_to_cpu(globals().get('sam_model'))
    _globs = globals()
    for name in ["sam_model", "mask_generator"]: _safe_del(name, _globs)
    run_cuda_cleanup()

def run_stage_post_xmem_cleanup():
    _safe_to_cpu(globals().get('network'))
    _safe_del("network", globals())
    run_cuda_cleanup()

# -------------------------- SAM Inference --------------------------------
def compute_mask_matrices_gpu(masks_tensor):
    N, H, W = masks_tensor.shape
    flat = masks_tensor.contiguous().reshape(N, -1).to(dtype=torch.float16)
    inter = torch.mm(flat, flat.t())
    area = flat.sum(dim=1)
    area_i = area.unsqueeze(1)
    area_j = area.unsqueeze(0)
    union = area_i + area_j - inter
    iou = inter / (union + 1e-6)
    containment = inter / (area_i + 1e-6)
    return iou, containment, area

def discard_submasks_gpu(masks_tensor, iou_thresh=0.82):
    if masks_tensor.shape[0] == 0: return masks_tensor
    iou_mat, containment_mat, areas = compute_mask_matrices_gpu(masks_tensor)
    containment_mat = containment_mat.cpu().numpy()
    areas = areas.cpu().numpy()
    n = masks_tensor.shape[0]
    visited = [False for _ in range(n)]
    is_master_parent = [False for _ in range(n)]

    adj = [[] for _ in range(n)]
    for i in range(n):
        js = np.where(containment_mat[:, i] >= iou_thresh)[0].tolist()
        adj[i] = [j for j in js if j != i]

    sorted_indices = list(np.argsort(areas)[::-1])
    for root in sorted_indices:
        if visited[root]: continue
        is_master_parent[root] = True
        master_parent = root
        stack = [root]
        while stack:
            node = stack.pop()
            if visited[node]:
                if node != master_parent and is_master_parent[node]: is_master_parent[node] = False
                continue
            visited[node] = True
            for nxt in adj[node]:
                if visited[nxt]:
                    if nxt != master_parent and is_master_parent[nxt]: is_master_parent[nxt] = False
                else: stack.append(nxt)
    return masks_tensor[is_master_parent]

def merge_similar_masks_gpu(masks_tensor, sim_thresh=0.82):
    if masks_tensor.shape[0] <= 1: return masks_tensor
    iou_mat, _, areas = compute_mask_matrices_gpu(masks_tensor)
    iou = iou_mat.cpu().numpy().astype(np.float32, copy=True)
    areas = areas.cpu().numpy()
    n = masks_tensor.shape[0]

    np.fill_diagonal(iou, -1.0)
    deleted = np.zeros(n, dtype=bool)

    while True:
        tri = np.triu(iou, 1)
        flat_idx = int(np.argmax(tri))
        max_iou = float(tri.flat[flat_idx])
        if max_iou < sim_thresh: break
        i, j = np.unravel_index(flat_idx, tri.shape)
        if deleted[i] or deleted[j]:
            iou[i, j] = iou[j, i] = -1.0
            continue
        keep, discard = (i, j) if areas[i] >= areas[j] else (j, i)
        active = ~deleted
        update_mask = active.copy()
        update_mask[keep] = update_mask[discard] = False
        if np.any(update_mask):
            iou[update_mask, keep] = np.minimum(iou[update_mask, keep], iou[update_mask, discard])
            iou[keep, update_mask] = iou[update_mask, keep]

        deleted[discard] = True
        iou[discard, :] = iou[:, discard] = -1.0
        iou[keep, keep] = -1.0

    return masks_tensor[~deleted]

def init_sam_model(sam_checkpoint=SAM_CHECKPOINT, model_type="vit_h"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device).eval()
    return sam

def SAM_inference(input_image_path, sam_mask_path, mask_generator):
    image_tensor = _imread_tensor(input_image_path)
    if image_tensor is None: return

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        masks_tensor = mask_generator.generate(image_tensor)

    if masks_tensor is None or masks_tensor.shape[0] == 0:
        h, w = image_tensor.shape[1:]
        cv2.imwrite(sam_mask_path, np.zeros((h, w, 3), dtype=np.uint8))
        return

    masks_tensor = discard_submasks_gpu(masks_tensor.half())
    masks_tensor = merge_similar_masks_gpu(masks_tensor)

    H, W = image_tensor.shape[1:]
    overlay_ts = torch.zeros((H, W, 3), dtype=torch.uint8, device="cuda")
    
    if masks_tensor.shape[0] > 0:
        any_mask = torch.any(masks_tensor > 0, dim=0)
        if torch.any(any_mask):
            last_mask_idx = (masks_tensor.shape[0] - 1) - torch.argmax(masks_tensor.flip(0).byte(), dim=0)
            palette = torch.tensor(COLOR_PALETTE, dtype=torch.uint8, device="cuda")
            color_indices = last_mask_idx % palette.shape[0]
            overlay_ts[any_mask] = palette[color_indices[any_mask]]

    overlay = overlay_ts.cpu().numpy()
    os.makedirs(os.path.dirname(sam_mask_path), exist_ok=True)
    cv2.imwrite(sam_mask_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

sam_model = None
_sam_thread_local = threading.local()

def init_sam():
    global sam_model
    if sam_model is None: sam_model = init_sam_model()
    return sam_model

def get_sam_generator():
    if not hasattr(_sam_thread_local, "mask_generator"):
        points_per_batch = int(os.environ.get("DP_SAM_POINTS_PER_BATCH", "64"))
        _sam_thread_local.mask_generator = SamAutomaticMaskGenerator(
            init_sam(), points_per_batch=points_per_batch, output_mode="torch_mask"
        )
    return _sam_thread_local.mask_generator

def run_stage_sam_inference(batch_subdirs):
    print(f"[INFO] Stage 1: Starting parallel SAM inference for {len(batch_subdirs)} videos...")
    init_sam()
    
    def sam_inference_wrapper(item):
        t0 = time.perf_counter()
        src_path = os.path.join(IMAGE_DIR, item)
        if not os.path.isdir(src_path): return None
        dest_path = os.path.join(ANNOTATION_DIR, item)
        os.makedirs(dest_path, exist_ok=True)
        frame_name = "00000.jpg"
        if item in first_detected_indices: frame_name = first_detected_indices[item].get("frame", "00000.jpg")
        mask_name = os.path.splitext(frame_name)[0] + ".png"
        image_path = os.path.join(src_path, frame_name)
        sam_mask_path = os.path.join(dest_path, mask_name)
        try:
            mask_generator = get_sam_generator()
            SAM_inference(image_path, sam_mask_path, mask_generator)
            return time.perf_counter() - t0
        except Exception as e:
            return None

    sam_workers = int(os.environ.get("DP_SAM_WORKERS", "1"))
    with ThreadPoolExecutor(max_workers=sam_workers) as executor:
        results = list(tqdm(executor.map(sam_inference_wrapper, batch_subdirs), total=len(batch_subdirs), desc="Generating SAM Proposals"))
        for t in results:
            if t is not None: stage1_times.append(t)
    run_cuda_cleanup()

# ---------------------- XMEM inference --------------------------------------
network = None

def init_xmem():
    global network
    if network is None:
        network = XMem(config, XMEM_MODEL_PATH).cuda().eval()
        network.load_weights(torch.load(XMEM_MODEL_PATH), init_as_zero_if_needed=True)
    return network

class FastVideoReader(torch.utils.data.Dataset):
    def __init__(self, vid_name, image_dir, mask_dir, size=-1, to_save=None, use_all_mask=False):
        self.vid_name, self.image_dir, self.mask_dir = vid_name, image_dir, mask_dir
        self.to_save, self.use_all_mask, self.size = to_save, use_all_mask, size
        self.frames = sorted(os.listdir(self.image_dir))
        mask_files = sorted(os.listdir(mask_dir))
        if mask_files:
            try:
                with Image.open(path.join(mask_dir, mask_files[0])) as img:
                    self.palette = img.getpalette()
                self.first_gt_path = path.join(mask_dir, mask_files[0])
            except Exception:
                self.palette, self.first_gt_path = None, None
        else: self.palette, self.first_gt_path = None, None

    def __getitem__(self, idx):
        frame = self.frames[idx]
        data, info = {'is_corrupt': False, 'im_path': path.join(self.image_dir, frame)}, {'frame': frame, 'save': (self.to_save is None) or (frame[:-4] in self.to_save)}
        try:
            img_tensor = _imread_tensor(data['im_path'], device="cpu")
            if img_tensor is None: img_tensor = torch.from_numpy(np.array(Image.open(data['im_path']).convert('RGB'))).permute(2, 0, 1)
            img_tensor = img_tensor.float() / 255.0
            mean, std = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1), torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            img_tensor = (img_tensor - mean) / std
            h, w = img_tensor.shape[-2:]
            info['shape'] = (h, w)
            if self.size > 0:
                new_h, new_w = int(h / min(h, w) * self.size), int(w / min(h, w) * self.size)
                img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
                info['need_resize'] = True
            else: info['need_resize'] = False
            data['rgb'] = img_tensor
        except Exception:
            data['is_corrupt'] = True
            data['rgb'] = torch.zeros((3, 1, 1))
            data['info'] = info
            return data

        gt_path = path.join(self.mask_dir, frame[:-4]+'.png')
        if (self.use_all_mask or gt_path == self.first_gt_path) and path.exists(gt_path):
            try: data['mask'] = np.array(Image.open(gt_path).convert('P', dither=Image.NONE), dtype=np.uint8)
            except Exception: data['is_corrupt'] = True
        data['info'] = info
        return data

    def resize_mask(self, mask):
        h, w = mask.shape[-2:]
        return F.interpolate(mask, (int(h/min(h, w)*self.size), int(w/min(h, w)*self.size)), mode='nearest')
    def get_palette(self): return self.palette
    def __len__(self): return len(self.frames)

class BatchedLongTestDataset:
    def __init__(self, data_root, vid_list, size=-1):
        self.image_dir, self.mask_dir, self.size = path.join(data_root, 'JPEGImages'), path.join(data_root, 'Annotations'), size
        self.vid_list = [v for v in vid_list if os.path.isdir(path.join(self.image_dir, v))]
    def get_datasets(self):
        for video in self.vid_list:
            yield FastVideoReader(video, path.join(self.image_dir, video), path.join(self.mask_dir, video),
                to_save=[name[:-4] for name in os.listdir(path.join(self.mask_dir, video))], size=self.size)

class MaskSaver:
    def __init__(self, max_workers=8): self.executor = ThreadPoolExecutor(max_workers=max_workers)
    def save(self, mask_np, path, palette=None): self.executor.submit(self._save_impl, mask_np, path, palette)
    def _save_impl(self, mask_np, path, palette):
        img = Image.fromarray(mask_np)
        img.putpalette(palette) if palette else img.putpalette(generate_color_palette())
        img.save(path)
    def shutdown(self): self.executor.shutdown(wait=True)

def process_xmem_video(vid_reader, network, config, args, mask_saver):
    t0 = time.perf_counter()
    device = next(network.parameters()).device
    loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    vid_length = len(loader)
    
    local_config = config.copy()
    local_config['enable_long_term_count_usage'] = (
        local_config['enable_long_term'] and
        (vid_length
            / (local_config['max_mid_term_frames'] - local_config['min_mid_term_frames'])
            * local_config['num_prototypes'])
        >= local_config['max_long_term_elements']
    )
    mapper, processor = MaskMapper(), InferenceCore(network, config=local_config)
    first_mask_loaded = False

    for ti, data in enumerate(loader):
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=not args.benchmark):
            if data.get('is_corrupt', [False])[0]: continue
            rgb = data['rgb'].to(device, non_blocking=True)[0]
            msk, info = data.get('mask'), data['info']
            frame, shape, need_resize = info['frame'][0], (info['shape'][0].item(), info['shape'][1].item()), info['need_resize'][0]

            if not first_mask_loaded:
                if msk is not None: first_mask_loaded = True
                else: continue

            msk_tensor, labels = None, None
            if msk is not None:
                m0_np = msk[0].detach().cpu().numpy() if isinstance(msk[0], torch.Tensor) else msk[0]
                msk_tensor, labels = mapper.convert_mask(m0_np)
                msk_tensor = msk_tensor.to(device, non_blocking=True)
                if need_resize: msk_tensor = vid_reader.resize_mask(msk_tensor.unsqueeze(0))[0]
                processor.set_all_labels(list(mapper.remappings.values()))

            prob = processor.step(rgb, msk_tensor, labels, end=(ti==vid_length-1))
            if need_resize: prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]
            out_mask_np = (torch.max(prob, dim=0).indices.detach().cpu().numpy()).astype(np.uint8)

            if args.save_all or info['save'][0]:
                this_out_path = path.join(XMEM_OUTPUT_DIR, vid_reader.vid_name)
                os.makedirs(this_out_path, exist_ok=True)
                out_mask_remapped = mapper.remap_index_mask(out_mask_np)
                mask_saver.save(out_mask_remapped, path.join(this_out_path, frame[:-4]+'.png'), vid_reader.get_palette())

    del processor, mapper, loader
    run_cuda_cleanup()
    return time.perf_counter() - t0

def run_stage_xmem_tracking(batch_subdirs):
    print(f"[INFO] Stage 2: Starting parallel XMem tracking for {len(batch_subdirs)} videos...")
    net = init_xmem()
    meta_dataset = BatchedLongTestDataset(DATA_DIR, batch_subdirs, size=args.size)
    torch.autograd.set_grad_enabled(False)
    mask_saver = MaskSaver(max_workers=16)
    
    with ThreadPoolExecutor(max_workers=int(os.environ.get("DP_XMEM_VIDEO_WORKERS", "2"))) as executor:
        futures = [executor.submit(process_xmem_video, vr, net, config, args, mask_saver) for vr in meta_dataset.get_datasets()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="XMem Tracking"):
            try:
                elapsed = future.result()
                if elapsed is not None: stage2_times.append(elapsed)
            except Exception as e:
                print(f"[ERROR] XMem tracking failed: {e}")
    mask_saver.shutdown()
    run_cuda_cleanup()

# ------------------------------ Distance metric for masks / LoFTR --------------------------------------
def mask_centroid(mask):
    ys, xs = np.where(mask > 0)
    return (np.mean(xs), np.mean(ys)) if len(xs) > 0 else (None, None)

def bbox_from_mask(mask: np.ndarray, pad: int = 16):
    ys, xs = np.where(mask)
    if len(xs) == 0: return None
    h, w = mask.shape[:2]
    return max(0, xs.min() - pad), max(0, ys.min() - pad), min(w - 1, xs.max() + pad), min(h - 1, ys.max() + pad)

def crop_and_offset(img, bbox):
    x0, y0, x1, y1 = bbox
    return img[y0:y1+1, x0:x1+1], (x0, y0)

def map_pts_to_original(pts, offset): return pts + np.array(offset, dtype=np.float32)

def filter_pts_in_mask(pts, mask):
    h, w = mask.shape[:2]
    xs, ys = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    return (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & (mask[ys, xs] > 0)

def load_loftr_model(weight_type, weight_path, device="cuda"):
    from loftr_src.loftr import LoFTR, default_cfg
    if weight_type == "outdoor_ot":
        default_cfg['match_coarse']['match_type'] = 'sinkhorn'
        default_cfg['match_coarse']['sparse_spvs'] = False
        default_cfg['trainer'] = default_cfg.get('trainer', {})
        default_cfg['trainer']['canonical_lr'] = 8e-3
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(weight_path, map_location=device)['state_dict'])
    return matcher.eval().to(device)

def match_visible_correspondences(matcher, frameA, frameB, maskA, maskB, device="cuda"):
    if None in [frameA, frameB, maskA, maskB]: return [], None, None, None
    bboxA, bboxB = bbox_from_mask(maskA), bbox_from_mask(maskB)
    if not bboxA or not bboxB: return [], None, None, None

    cropA, offA = crop_and_offset(frameA, bboxA)
    cropB, offB = crop_and_offset(frameB, bboxB)

    imgA = cv2.resize(cv2.cvtColor(cropA, cv2.COLOR_RGB2GRAY), (640, 480))
    imgB = cv2.resize(cv2.cvtColor(cropB, cv2.COLOR_RGB2GRAY), (640, 480))
    tA = torch.from_numpy(imgA)[None][None].float().to(device) / 255.0
    tB = torch.from_numpy(imgB)[None][None].float().to(device) / 255.0

    batch = {"image0": tA, "image1": tB}
    with torch.no_grad(): matcher(batch)
    mkpts0, mkpts1, mconf = batch["mkpts0_f"].cpu().numpy(), batch["mkpts1_f"].cpu().numpy(), batch["mconf"].cpu().numpy()

    if len(mkpts0) == 0: return [], None, None, None

    scaleA, scaleB = (cropA.shape[1] / 640, cropA.shape[0] / 480), (cropB.shape[1] / 640, cropB.shape[0] / 480)
    ptsA_full = map_pts_to_original(mkpts0 * np.array(scaleA), offA)
    ptsB_full = map_pts_to_original(mkpts1 * np.array(scaleB), offB)

    keep = filter_pts_in_mask(ptsA_full, maskA) & filter_pts_in_mask(ptsB_full, maskB)
    ptsA_vis, ptsB_vis, mconf = ptsA_full[keep], ptsB_full[keep], mconf[keep]

    if len(ptsA_vis) == 0: return [], None, None, None
    return [(ptsA_vis[i].tolist(), ptsB_vis[i].tolist()) for i in range(len(ptsA_vis))], ptsA_vis, ptsB_vis, mconf

def get_distance(matcher, frameA, frameB, maskA, maskB):
    c0, cN = mask_centroid(maskA), mask_centroid(maskB)
    if c0[0] is None or cN[0] is None: return -1
    centroid_dist = np.linalg.norm(np.array(cN) - np.array(c0))
    
    try: matches, ptsA, ptsB, _ = match_visible_correspondences(matcher, frameA, frameB, maskA, maskB)
    except Exception: fm_dist = -1
    else: fm_dist = float(np.linalg.norm(ptsA - ptsB, axis=1).mean()) if len(matches) > 0 else -1
    
    return centroid_dist if fm_dist == -1 else min(centroid_dist, fm_dist)

def mask_overlap_arm_mask(mask1, mask2):
    area1 = mask1.sum()
    return np.logical_and(mask1, mask2).sum() / area1 if area1 > 0 else 0.0

def mask_overlap(mask1, mask2):
    union = np.logical_or(mask1, mask2).sum()
    return np.logical_and(mask1, mask2).sum() / union if union > 0 else 0

def get_edit_masks_advanced_distance_metric(matcher, frame_dir, img_dir, robotic_arm_dir, out_dir="./edit_masks"):
    os.makedirs(out_dir, exist_ok=True)
    mask_files = natsorted([f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg"))])
    if len(mask_files) < 2: return
        
    first_mask_name, middle_mask_name = mask_files[0], mask_files[len(mask_files)//2]
    first_frame = _imread_rgb(os.path.join(frame_dir, first_mask_name.replace(".png", ".jpg")))
    middle_frame = _imread_rgb(os.path.join(frame_dir, middle_mask_name.replace(".png", ".jpg")))
    if first_frame is None or middle_frame is None: return

    base_arm_first = _imread_gray(os.path.join(robotic_arm_dir, first_mask_name))
    if base_arm_first is None: return
    base_arm_first = (base_arm_first > 0).astype(np.uint8)
    
    first_rgb = _imread_rgb(os.path.join(img_dir, first_mask_name))
    middle_rgb = _imread_rgb(os.path.join(img_dir, middle_mask_name))
    if first_rgb is None or middle_rgb is None: return
    
    images = [os.path.join(img_dir, f) for f in mask_files]
    frames = [os.path.join(frame_dir, f.replace(".png", ".jpg")) for f in mask_files]
    
    unique_colors = [tuple(c) for c in np.unique(first_rgb.reshape(-1, 3), axis=0) if not np.all(c == 0)]
    robotic_arm_colors = [c for c in unique_colors if mask_overlap_arm_mask(np.all(first_rgb == c, axis=-1).astype(np.uint8), base_arm_first) > 0.25]
    
    current_frames, current_rgbs, moved_distances, max_moved_distances = {}, {}, [], {}
    target_object_candidate_colors = list(set(unique_colors) - set(robotic_arm_colors))

    for color in target_object_candidate_colors:
        mask_first = np.all(first_rgb == color, axis=-1).astype(np.uint8)
        mask_middle = np.all(middle_rgb == color, axis=-1).astype(np.uint8)
        
        for i in range(len(images)-1, int((3*len(images))/4), -1):
            if i not in current_rgbs: current_rgbs[i] = _imread_rgb(images[i])
            mask_current = np.all(current_rgbs[i] == color, axis=-1).astype(np.uint8)
            
            if np.sum(mask_first) > 0 and np.sum(mask_current) > 0:
                if i not in current_frames: current_frames[i] = _imread_rgb(frames[i])
                dist = get_distance(matcher, first_frame, current_frames[i], mask_first, mask_current)
                if dist != -1: max_moved_distances[color] = max(max_moved_distances.get(color, 0), dist)

            if np.sum(mask_middle) > 0 and np.sum(mask_current) > 0:
                if i not in current_frames: current_frames[i] = _imread_rgb(frames[i])
                dist = get_distance(matcher, middle_frame, current_frames[i], mask_middle, mask_current)
                if dist != -1: max_moved_distances[color] = max(max_moved_distances.get(color, 0), dist)

        moved_distances.append((color, max_moved_distances.get(color, 0.0)))
    
    moved_distances = sorted(moved_distances, key=lambda x: x[1], reverse=True)
    target_colors = []
    
    if len(moved_distances) == 1 and moved_distances[0][1] > 0: target_colors.append(moved_distances[0][0])
    elif 1 < len(moved_distances) < 3: target_colors = [c for c, d in moved_distances if d > 0]
    elif len(moved_distances) >= 3:
        distances = np.array([item[1] for item in moved_distances]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(distances)
        moved_cluster_label = np.argmax(kmeans.cluster_centers_)
        target_colors = [item[0] for i, item in enumerate(moved_distances) if kmeans.labels_[i] == moved_cluster_label]

    overlap_started, edited_idx = False, None
    for i, img_path in enumerate(images):
        rgb = _imread_rgb(img_path)
        if rgb is None: continue

        arm_mask, target_mask = np.zeros_like(base_arm_first), np.zeros_like(base_arm_first)
        for c in robotic_arm_colors: arm_mask |= np.all(rgb == c, axis=-1).astype(np.uint8)
        for c in target_colors: target_mask |= np.all(rgb == c, axis=-1).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        arm_mask, target_mask = cv2.dilate(arm_mask, kernel, iterations=1), cv2.dilate(target_mask, kernel, iterations=1)
        overlap = mask_overlap(arm_mask, target_mask)

        if not overlap_started and overlap > 0: overlap_started = True
        elif overlap_started and overlap <= 0: edited_idx, overlap_started = i-1, False
        elif overlap_started and overlap > 0: edited_idx = i

    if edited_idx is None:
        edited_idx = len(images) - 1

    final_frame_num = os.path.splitext(os.path.basename(images[edited_idx]))[0]

    init_rgb = _imread_rgb(images[0])
    final_rgb = _imread_rgb(images[edited_idx])
    if init_rgb is None or final_rgb is None:
        return

    init_mask = np.zeros(init_rgb.shape[:2], dtype=np.uint8)
    final_mask = np.zeros(final_rgb.shape[:2], dtype=np.uint8)

    for c in robotic_arm_colors + target_colors:
        init_mask |= np.all(init_rgb == c, axis=-1).astype(np.uint8)
        final_mask |= np.all(final_rgb == c, axis=-1).astype(np.uint8)

    cv2.imwrite(os.path.join(out_dir, "initial_edit_region_mask.png"), init_mask * 255)
    cv2.imwrite(os.path.join(out_dir, "final_edit_region_mask.png"), final_mask * 255)
    with open(os.path.join(out_dir, "edited_frame_number.txt"), "w") as f:
        f.write(str(final_frame_num))

loftr_matcher = None
def init_loftr():
    global loftr_matcher
    if loftr_matcher is None:
        weight_map = {
            "indoor_ds_new": os.path.join(LOFTR_WEIGHTS_DIR, "indoor_ds_new.ckpt"),
            "indoor_ot": os.path.join(LOFTR_WEIGHTS_DIR, "indoor_ot.ckpt"),
            "outdoor_ds": os.path.join(LOFTR_WEIGHTS_DIR, "outdoor_ds.ckpt"),
            "outdoor_ot": os.path.join(LOFTR_WEIGHTS_DIR, "outdoor_ot.ckpt"),
        }
        weight_path = weight_map[args.weight_type]
        loftr_matcher = load_loftr_model(args.weight_type, weight_path, device)
    return loftr_matcher

def run_stage_loftr_masking(batch_subdirs):
    print(f"[INFO] Stage 3: Generating Edit Region Masks for {len(batch_subdirs)} videos...")
    os.makedirs(EDIT_MASKS_DIR, exist_ok=True)
    matcher = init_loftr()

    def loftr_wrapper(subdir):
        t0 = time.perf_counter()
        track_dir, robotic_mask_dir, frame_dir = os.path.join(XMEM_OUTPUT_DIR, subdir), os.path.join(ROBOTICMASK_DIR, subdir), os.path.join(IMAGE_DIR, subdir)
        if os.path.isdir(track_dir):
            try:
                get_edit_masks_advanced_distance_metric(matcher, frame_dir, track_dir, robotic_mask_dir, out_dir=os.path.join(EDIT_MASKS_DIR, subdir))
                return time.perf_counter() - t0
            except Exception: pass
        return None

    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 4)) as executor:
        results = list(tqdm(executor.map(loftr_wrapper, batch_subdirs), total=len(batch_subdirs), desc="Generating LoFTR Masks"))
        for t in results:
            if t is not None: stage3_times.append(t)

def run_stage_post_loftr_cleanup():
    _safe_to_cpu(globals().get('loftr_matcher'))
    _safe_del("loftr_matcher", globals())
    run_cuda_cleanup()

def run_stage_cleanup(batch_subdirs):
    if not args.cleanup: return
    dirs_to_clean = [ANNOTATION_DIR, XMEM_OUTPUT_DIR, EDIT_MASKS_DIR, ROBOTICMASK_DIR]
    for subdir in batch_subdirs:
        for d in dirs_to_clean:
            target = os.path.join(d, subdir)
            if os.path.exists(target):
                if os.path.islink(target):
                    try: os.remove(target)
                    except: pass
                elif os.path.isdir(target):
                    shutil.rmtree(target, ignore_errors=True)

# -------------------------
# Main Execution Loop
# -------------------------
def main():
    all_episodes = sorted([d for d in os.listdir(os.path.join(BASE_DIR, args.dataset_name, "JPEGImages")) 
                           if os.path.isdir(os.path.join(BASE_DIR, args.dataset_name, "JPEGImages", d))])
    
    total_episodes = len(all_episodes)
    batch_size = args.batch_size or total_episodes
    start_idx = (args.resume - 1) * batch_size

    print(f"[INFO] Total episodes to process: {total_episodes}")
    print(f"[INFO] Using batch size: {batch_size}")

    for i in range(start_idx, total_episodes, batch_size):
        batch_idx = i // batch_size
        batch_subdirs = all_episodes[i : i + batch_size]
        print(f"\n[INFO] Starting Batch {batch_idx+1} (Episodes {i} to {min(i + batch_size, total_episodes)})...")
        
        # Stages are run and timed
        run_stage_robotic_arm_extraction(batch_subdirs)
        run_stage_post_evf_cleanup()
        
        run_stage_sam_inference(batch_subdirs)
        run_stage_post_sam_cleanup()
        
        run_stage_xmem_tracking(batch_subdirs)
        run_stage_post_xmem_cleanup()
        
        run_stage_loftr_masking(batch_subdirs)
        run_stage_post_loftr_cleanup()
        
        run_stage_cleanup(batch_subdirs)

    # -----------------------------------------------------
    # Calculate and Print Statistics
    # -----------------------------------------------------
    print("\n" + "="*50)
    print("PIPELINE TIMING STATISTICS (PER VIDEO)")
    print("="*50)

    def print_stats(name, times):
        if not times:
            print(f"{name}: No data collected")
            return {"mean": 0, "std": 0, "raw": []}
        arr = np.array(times)
        print(f"{name}:")
        print(f"  Mean  : {arr.mean():.4f} seconds")
        print(f"  Std   : {arr.std():.4f} seconds")
        print(f"  Sum   : {arr.sum():.4f} seconds")
        print(f"  Count : {len(arr)}")
        return {"mean": arr.mean(), "std": arr.std(), "raw": times}

    stats = {
        "stage0": print_stats("Stage 0 (EVF-SAM + BLIP)", stage0_times),
        "stage1": print_stats("Stage 1 (SAM First Frame)", stage1_times),
        "stage2": print_stats("Stage 2 (XMem Tracking)", stage2_times),
        "stage3": print_stats("Stage 3 (LoFTR + KMeans)", stage3_times)
    }

    # Save logic solely for profiling timing output
    with open("timing_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print("\n[INFO] Benchmarking complete. Statistics saved to 'timing_stats.json'.")

if __name__ == "__main__":
    main()