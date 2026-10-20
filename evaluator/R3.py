"""
R3_SAM.py

Sophisticated algorithm to detect UNMENTIONED REMOVED-FROM-SCENE objects in manipulation videos.
This version uses SAM (Segment Anything Model) for mask generation.

Algorithm:
1) Extract mentioned objects from manipulation instruction using Gemini/VideoChat-Flash
2) Perform removed-from-scene object detection:
   a) SAM proposal generation on first frame
   b) XMem tracking to get masks in second frame
   c) Identify objects present in first frame but absent in second frame (removed from scene)
   d) BLIP2 verification to filter out mentioned objects from the removed set

Requirements:
    pip install transformers torch opencv-python segment-anything natsort
    pip install flash-attn --no-build-isolation (optional)
    
GPU Memory Optimization:
    For better memory management, set before running:
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
Command to run:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=./XMem python3 R3_SAM.py
"""

import os
import re
import gc
import ast
import json
import time
import atexit
import shutil
import tempfile
from typing import Tuple, List, Optional, Dict, Any
from argparse import Namespace
from contextlib import contextmanager

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image

from natsort import natsorted

# -------------------------
# GPU Memory Optimization Utilities
# -------------------------

def clear_cuda_cache():
    """Aggressively clear CUDA memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


@contextmanager
def gpu_memory_manager():
    """Context manager for GPU memory-intensive operations."""
    try:
        yield
    finally:
        clear_cuda_cache()


def move_model_to_cpu(model):
    """Move model to CPU and clear GPU cache."""
    if model is not None and hasattr(model, 'to'):
        model.to('cpu')
    clear_cuda_cache()


def move_model_to_gpu(model, device='cuda'):
    """Move model to GPU."""
    if model is not None and hasattr(model, 'to'):
        model.to(device)
    return model

from transformers import (
    AutoModel,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor
)

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# XMem imports (require PYTHONPATH=./XMem)
from inference.data.mask_mapper import MaskMapper
from inference.data.test_datasets import LongTestDataset
from inference.inference_core import InferenceCore
from model.network import XMem

# -------------------------
# Constants and Configs
# -------------------------
BASE_DIR = os.path.dirname(__file__)
SAM_CHECKPOINT = os.path.join(BASE_DIR, "sam_vit_h_4b8939.pth")
XMEM_MODEL_PATH = os.path.join(BASE_DIR, "XMem/saves/XMem.pth")
LOFTR_WEIGHTS_DIR = os.path.join(BASE_DIR, "loftr_weights")

# Output directory for all pipeline results
R3_OUTPUT_DIR = os.path.join(BASE_DIR, "R3_Output")

MODEL_NAME = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"
BLIP_MODEL = "Salesforce/blip2-flan-t5-xl"

# -------------------------
# Instruction Target Extraction Method Selection
# -------------------------
# Set to "local_llm" to use a local small LLM (default), or "videochat" to use VideoChatFlash
INSTRUCTION_TARGET_METHOD = "local_llm"

# -------------------------
# Local LLM Configuration (replaces Gemini API — no external API calls,
# no rate limits, no 429 errors, no watchdog-triggered account bans)
# -------------------------
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Module-level globals — loaded once by load_local_llm(), used throughout
_local_llm_model = None
_local_llm_tokenizer = None

# Reduced from 32 to 8 for memory efficiency (we only use 2-frame videos)
MAX_NUM_FRAMES = 8

# Generation configs for VideoChat-Flash (use greedy decoding for efficiency)
GEN_CONFIG_SHORT = dict(do_sample=False, max_new_tokens=100, num_beams=1, top_k=1)

# -------------------------
# R3-Specific Configuration
# -------------------------
# Minimum mask area as fraction of image area (filter out tiny fragments)
MIN_MASK_AREA_RATIO = 0.0007  # 0.07% of image (lowered to catch smaller objects)

# Minimum mask area for tracked masks (objects must maintain significant size)
MIN_TRACKED_MASK_AREA_RATIO = 0.0005  # 0.05% of image

# Noise object keywords to filter out from captions
NOISE_OBJECT_KEYWORDS = [
    'shadow', 'reflection',
    'background', 'table surface', 'floor', 'wall', 'ceiling',
    'finger', 'hand', 'wire', 'cable', 'blur',
    'artifact', 'unknown', 'nothing', 'n/a', 'none', 'part of',
    'piece of', 'portion', 'fragment',
]
# More precise robot filtering: filter single-word 'robot' or 'gripper'
# but allow compound phrases like 'robotic arm' to pass through
ROBOT_EXACT_KEYWORDS = ['robot', 'gripper']

# Keywords that indicate false positive captions (too generic/abstract)
GENERIC_CAPTION_KEYWORDS = [
    'cross', 'diamond', 'ring', 'sleeve', 'wing', 'sword', 'helmet',
    'tee', 'shirt', 'jacket', 'pair of wings', 'glass ring', 'silver ring',
]

# Color palette for mask visualization
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
    (153, 153, 255), (204, 153, 255), (255, 153, 204), (255, 191, 191),
]


def generate_color_palette(num_colors=256):
    """Generate a list of evenly spaced RGB colors excluding white and black."""
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        r = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (hue + 0.0))))
        g = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (hue + 0.33))))
        b = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (hue + 0.67))))
        colors.append((r, g, b))
    flat_palette = [v for c in colors[:256] for v in c]
    flat_palette.extend([0] * (768 - len(flat_palette)))
    return flat_palette


# =============================================================================
# Output Logging Utilities
# =============================================================================

def setup_test_output_dir(test_name: str) -> Dict[str, str]:
    """
    Create output directory structure for a test case.
    
    Returns:
        Dictionary with paths to all subdirectories
    """
    # Sanitize test name for folder creation
    safe_name = re.sub(r'[^\w\-_]', '_', test_name)[:50]
    test_dir = os.path.join(R3_OUTPUT_DIR, safe_name)
    
    subdirs = {
        'root': test_dir,
        'instruction_targets': os.path.join(test_dir, '1_instruction_targets'),
        'sam_masks': os.path.join(test_dir, '2_sam_masks'),
        'xmem_tracking': os.path.join(test_dir, '3_xmem_tracking'),
        'removed_objects': os.path.join(test_dir, '4_removed_objects'),
        'blip2_verification': os.path.join(test_dir, '5_blip2_verification'),
        'final_results': os.path.join(test_dir, '6_final_results'),
    }
    
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)
    
    return subdirs


def save_instruction_targets(output_dir: str, instruction: str, mentioned_objects: List[str], method: str):
    """Save instruction target extraction results."""
    result = {
        'instruction': instruction,
        'extraction_method': method,
        'mentioned_objects': mentioned_objects,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save as JSON
    with open(os.path.join(output_dir, 'extraction_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save as readable text
    with open(os.path.join(output_dir, 'extraction_result.txt'), 'w') as f:
        f.write(f"Instruction: {instruction}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Extracted Objects: {', '.join(mentioned_objects)}\n")


def save_sam_masks(output_dir: str, sam_masks: List[np.ndarray], original_image_path: str):
    """Save SAM mask proposals as images."""
    # Load original image for overlay
    original_img = cv2.imread(original_image_path)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_img.shape[:2]
    
    # Save individual masks
    for idx, mask in enumerate(sam_masks):
        # Binary mask
        mask_img = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f'mask_{idx:03d}.png'), mask_img)
        
        # Colored overlay
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        overlay = original_img_rgb.copy()
        overlay[mask == 1] = color
        blended = cv2.addWeighted(original_img_rgb, 0.5, overlay, 0.5, 0)
        cv2.imwrite(os.path.join(output_dir, f'mask_{idx:03d}_overlay.png'), 
                    cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    
    # Save combined visualization
    combined = original_img_rgb.copy()
    for idx, mask in enumerate(sam_masks):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        combined[mask == 1] = color
    blended_combined = cv2.addWeighted(original_img_rgb, 0.4, combined, 0.6, 0)
    cv2.imwrite(os.path.join(output_dir, 'all_masks_combined.png'),
                cv2.cvtColor(blended_combined, cv2.COLOR_RGB2BGR))
    
    # Save summary
    with open(os.path.join(output_dir, 'sam_summary.json'), 'w') as f:
        json.dump({
            'total_masks': len(sam_masks),
            'image_size': [h, w],
            'mask_areas': [int(np.sum(m)) for m in sam_masks]
        }, f, indent=2)


def save_xmem_tracking(output_dir: str, tracked_masks: Dict, color_to_idx: Dict, 
                       first_frame_path: str, second_frame_path: str):
    """Save XMem tracking results."""
    # Load second frame for overlay
    second_frame = cv2.imread(second_frame_path)
    second_frame_rgb = cv2.cvtColor(second_frame, cv2.COLOR_BGR2RGB)
    
    # Resize second frame to match tracked masks if needed
    if len(tracked_masks) > 0:
        sample_mask = list(tracked_masks.values())[0]
        mask_h, mask_w = sample_mask.shape[:2]
        frame_h, frame_w = second_frame_rgb.shape[:2]
        if mask_h != frame_h or mask_w != frame_w:
            second_frame_rgb = cv2.resize(second_frame_rgb, (mask_w, mask_h))
    
    # Create subdirectories for individual masks and overlays
    individual_masks_dir = os.path.join(output_dir, 'individual_masks')
    individual_overlays_dir = os.path.join(output_dir, 'individual_overlays')
    os.makedirs(individual_masks_dir, exist_ok=True)
    os.makedirs(individual_overlays_dir, exist_ok=True)
    
    # Save individual tracked masks with unique indices
    mask_info = []
    for enum_idx, (color, mask) in enumerate(tracked_masks.items()):
        # Use color_to_idx if available, otherwise use enumeration index
        idx = color_to_idx.get(color, enum_idx)
        
        # Binary mask
        mask_img = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(individual_masks_dir, f'tracked_mask_{idx:03d}.png'), mask_img)
        
        # Overlay on second frame
        overlay = second_frame_rgb.copy()
        overlay[mask == 1] = color
        blended = cv2.addWeighted(second_frame_rgb, 0.5, overlay, 0.5, 0)
        cv2.imwrite(os.path.join(individual_overlays_dir, f'tracked_mask_{idx:03d}_overlay.png'),
                    cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        
        # Track mask info
        mask_info.append({
            'index': int(idx),
            'color_rgb': [int(c) for c in color] if isinstance(color, tuple) else color,
            'area_pixels': int(np.sum(mask))
        })
    
    # Save combined tracked masks on second frame
    combined = second_frame_rgb.copy()
    for color, mask in tracked_masks.items():
        combined[mask == 1] = color
    blended_combined = cv2.addWeighted(second_frame_rgb, 0.4, combined, 0.6, 0)
    cv2.imwrite(os.path.join(output_dir, 'all_tracked_masks_combined.png'),
                cv2.cvtColor(blended_combined, cv2.COLOR_RGB2BGR))
    
    # Save summary
    with open(os.path.join(output_dir, 'xmem_summary.json'), 'w') as f:
        json.dump({
            'total_tracked_masks': len(tracked_masks),
            'color_to_index_mapping': {str(k): v for k, v in color_to_idx.items()},
            'mask_details': mask_info
        }, f, indent=2)


def save_blip2_verification(output_dir: str, verification_results: List[Dict],
                            first_frame_path: str, first_frame_masks: Dict):
    """Save BLIP2 verification results."""
    # Load first frame for cropped object images
    first_frame = cv2.imread(first_frame_path)
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if len(first_frame_masks) > 0:
        sample_mask = list(first_frame_masks.values())[0]
        mask_h, mask_w = sample_mask.shape[:2]
        frame_h, frame_w = first_frame_rgb.shape[:2]
        if mask_h != frame_h or mask_w != frame_w:
            first_frame_rgb = cv2.resize(first_frame_rgb, (mask_w, mask_h))
    
    first_frame_bgr = cv2.cvtColor(first_frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Save crops and results for each verified object
    for i, result in enumerate(verification_results):
        color = result.get('color')
        # Convert list to tuple for dict lookup
        if color:
            color_key = tuple(color) if isinstance(color, list) else color
            if color_key in first_frame_masks:
                mask = first_frame_masks[color_key]
                crop = crop_from_mask(first_frame_bgr, mask)
                if crop:
                    crop.save(os.path.join(output_dir, f'object_{i:03d}_crop.png'))
    
    # Save summary JSON
    with open(os.path.join(output_dir, 'verification_results.json'), 'w') as f:
        json.dump({
            'total_verified': len(verification_results),
            'results': verification_results
        }, f, indent=2)
    
    # Save readable summary
    with open(os.path.join(output_dir, 'verification_results.txt'), 'w') as f:
        f.write("BLIP-2 Verification Results\n")
        f.write("=" * 50 + "\n")
        for i, result in enumerate(verification_results):
            f.write(f"\nObject {i}:\n")
            f.write(f"  Color: {result.get('color')}\n")
            f.write(f"  Is mentioned: {result.get('is_mentioned', False)}\n")
            if result.get('matched_object'):
                f.write(f"  Matched with: {result.get('matched_object')}\n")
            if result.get('caption'):
                f.write(f"  Caption: {result.get('caption')}\n")


def save_final_results(output_dir: str, test_case: Dict, predicted_count: int, 
                       objects: List[str], score: float):
    """Save final test results."""
    result = {
        'test_description': test_case.get('description', ''),
        'instruction': test_case.get('instruction', ''),
        'before_image': test_case.get('before', ''),
        'after_image': test_case.get('after', ''),
        'expected_count': test_case.get('expected_count', 0),
        'predicted_count': predicted_count,
        'unmentioned_removed_objects': objects,
        'score': score,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'final_results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    
    with open(os.path.join(output_dir, 'final_results.txt'), 'w') as f:
        f.write("Final Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test: {result['test_description']}\n")
        f.write(f"Instruction: {result['instruction']}\n")
        f.write(f"\nExpected count: {result['expected_count']}\n")
        f.write(f"Predicted count: {result['predicted_count']}\n")
        f.write(f"Score: {result['score']:.4f}\n")
        f.write(f"\nUnmentioned removed objects:\n")
        for obj in objects:
            f.write(f"  - {obj}\n")


# =============================================================================
# SECTION 0: Local LLM for Instruction Target Extraction
# =============================================================================

LLM_INSTRUCTION_TARGET_PROMPT = """You are an AI assistant that extracts object names from manipulation instructions.

Manipulation Instruction:
"{instruction}"

Your task is to extract the names of objects that are INTENDED TO BE MOVED according to the manipulation instruction.

RULES:
1. Only extract objects that the instruction explicitly says should be moved/picked up/placed/put.
2. Do NOT include destination objects (where something is placed onto/into) unless they are also being moved.
3. Do NOT include the robot, robotic arm, gripper, or any agent performing the action.
4. Return object names as a simple comma-separated list.
5. If no objects are intended to be moved, return "none".

EXAMPLES:
- "The robotic arm moves the red cube to the right" -> "red cube"
- "The robotic arm picks up the apple and puts it on the plate" -> "apple" (plate is the destination, not moved)
- "The robotic arm removes the blue cup from the table" -> "blue cup"
- "The robotic arm moves the book to the top of the jar" -> "book" (jar is the destination, not moved)
- "Put the bottle in the drawer" -> "bottle" (drawer is the destination, not moved)
- "Move both the apple and the orange to the basket" -> "apple, orange" (basket is the destination)
- "The robotic arm places the cup on top of the saucer" -> "cup" (saucer is the destination surface)
- "The robotic arm puts the egg on top of the bread" -> "egg" (bread is the destination surface, not moved)
- "Turn the knob clockwise" -> "knob"

Now extract the objects from the given instruction. Output ONLY the comma-separated object names (or "none"):
"""


def load_local_llm():
    """Load the local LLM for text-only tasks (instruction parsing).
    
    Uses Qwen2.5-7B-Instruct with bitsandbytes 4-bit NF4 quantization (~5 GB VRAM).
    Includes a compatibility patch for transformers 4.40.x + accelerate bnb dispatch.
    Runs entirely on-device with no API calls.
    """
    global _local_llm_model, _local_llm_tokenizer
    if _local_llm_model is not None:
        return
    
    # ---- bnb / transformers 4.40.x compatibility patch ----
    import transformers.modeling_utils as _mu
    _orig_to = _mu.PreTrainedModel.to
    def _safe_to(self, *args, **kwargs):
        if getattr(self, 'is_quantized', False):
            return self
        return _orig_to(self, *args, **kwargs)
    _mu.PreTrainedModel.to = _safe_to
    # ---- end patch ----
    
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    print(f"[INFO] Loading local LLM: {LOCAL_LLM_MODEL} (4-bit NF4)")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    _local_llm_tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_LLM_MODEL, trust_remote_code=True
    )
    _local_llm_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_LLM_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    _local_llm_model.eval()
    
    # Move non-quantized buffers (rotary embeddings etc.) that stayed on CPU
    for _, buf in _local_llm_model.named_buffers():
        if buf.device.type == 'cpu':
            buf.data = buf.data.to('cuda')
    
    # Restore original .to() so BLIP-2 / SAM loading is unaffected
    _mu.PreTrainedModel.to = _orig_to
    
    print(f"[INFO] Local LLM loaded successfully ({LOCAL_LLM_MODEL}, 4-bit NF4)")


@torch.no_grad()
def call_local_llm(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 256,
) -> str:
    """
    Call the local LLM for text generation.
    Drop-in replacement for the old call_gemini_api().
    No external API calls — runs entirely on-device.
    """
    global _local_llm_model, _local_llm_tokenizer
    if _local_llm_model is None or _local_llm_tokenizer is None:
        load_local_llm()
    
    messages = [
        {"role": "system", "content": "You are a precise and concise assistant. Follow instructions exactly."},
        {"role": "user", "content": prompt},
    ]
    text = _local_llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _local_llm_tokenizer(text, return_tensors="pt").to(_local_llm_model.device)
    
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        pad_token_id=_local_llm_tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = 0.95
    
    outputs = _local_llm_model.generate(**inputs, **gen_kwargs)
    # Decode only the newly generated tokens
    response = _local_llm_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response


def extract_instruction_targets_local_llm(
    instruction: str,
    verbose: bool = False
) -> List[str]:
    """
    Extract the list of objects mentioned in the manipulation instruction
    using a local small LLM.
    
    This is a lightweight, rate-limit-free replacement for Gemini API that
    runs entirely on-device with no external API calls.
    """
    prompt = LLM_INSTRUCTION_TARGET_PROMPT.format(instruction=instruction)
    
    try:
        response = call_local_llm(prompt)
        response = response.strip()
        
        if verbose:
            print(f"Local LLM response: {response}")
        
        # Parse the response
        text = re.sub(r'[\*\[\]\-\(\)\"\'\.]', ' ', response).lower()
        text = text.replace("none", "").strip()
        
        if not text:
            return []
        
        # Split by common delimiters
        items = re.split(r",|\band\b|;", text)
        return [x.strip() for x in items if len(x.strip()) > 1]
        
    except Exception as e:
        print(f"[ERROR] Local LLM extraction failed: {e}")
        return []


# =============================================================================
# SECTION 1: VideoChat-Flash for Instruction Target Extraction
# =============================================================================

def two_images_to_video(img1_path: str, img2_path: str, out_path: str, fps: int = 1) -> str:
    """Create a minimal 2-frame video from two images."""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None:
        raise ValueError(f"Could not read: {img1_path}")
    if img2 is None:
        raise ValueError(f"Could not read: {img2_path}")

    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    out.write(img1)
    out.write(img2)
    out.release()
    return out_path


def _safe_chat_call(video_chat_model, verbose=False, **kwargs):
    """Safe wrapper for VideoChat-Flash model chat call with memory optimization."""
    call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if "video_path" not in call_kwargs:
        call_kwargs["video_path"] = ""
    
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        result = video_chat_model.chat(**call_kwargs)
    
    # Clear intermediate tensors
    clear_cuda_cache()
    
    if verbose and isinstance(result, tuple):
        print(f"[DEBUG] Chat output: {result[0]}")
    return result if not isinstance(result, tuple) else result[0]


def _clean_list(text: str) -> List[str]:
    """Clean and parse comma-separated object list from model output."""
    text = re.sub(r'[\*\[\]\-\(\)\"\']', ' ', text).lower()
    text = text.replace("none", "").replace(".", "")
    items = re.split(r",|\band\b|;", text)
    return [x.strip() for x in items if len(x.strip()) > 1]


def load_videochat_model():
    """Load VideoChat-Flash model and tokenizer with memory optimization."""
    print(f"[INFO] Loading VideoChat-Flash model: {MODEL_NAME}")
    clear_cuda_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Use auto for better memory distribution
        low_cpu_mem_usage=True,
    )
    model.config.mm_llm_compress = False
    model.eval()  # Ensure eval mode
    
    # Disable gradient computation globally for this model
    for param in model.parameters():
        param.requires_grad = False
    
    return tokenizer, model


def extract_instruction_targets(
    instruction: str,
    img_before_path: str,
    img_after_path: str,
    video_chat_model=None,
    video_chat_tokenizer=None,
    max_num_frames: int = MAX_NUM_FRAMES,
    verbose: bool = False,
    method: str = None
) -> List[str]:
    """
    Extract the list of objects mentioned in the manipulation instruction.
    
    Uses either Gemini API or VideoChat-Flash model based on the method parameter
    or the global INSTRUCTION_TARGET_METHOD setting.
    
    Args:
        instruction: The manipulation instruction text
        img_before_path: Path to the before image (used by VideoChatFlash)
        img_after_path: Path to the after image (used by VideoChatFlash)
        video_chat_model: VideoChat-Flash model (optional if using Gemini)
        video_chat_tokenizer: VideoChat-Flash tokenizer (optional if using Gemini)
        max_num_frames: Max frames for VideoChatFlash
        verbose: Print debug info
        method: "gemini" or "videochat" (overrides global setting if provided)
    """
    # Determine which method to use
    use_method = method if method is not None else INSTRUCTION_TARGET_METHOD
    
    if use_method in ("local_llm", "gemini"):
        if verbose:
            print("  Using local LLM for instruction target extraction")
        return extract_instruction_targets_local_llm(instruction, verbose=verbose)
    
    # Fall back to VideoChatFlash
    if verbose:
        print("  Using VideoChatFlash for instruction target extraction")
    
    if video_chat_model is None or video_chat_tokenizer is None:
        raise ValueError("VideoChatFlash model and tokenizer required when method='videochat'")
    # Create temp video file and ensure cleanup
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video_path = tmpf.name
    tmpf.close()
    atexit.register(lambda: os.path.exists(tmp_video_path) and os.remove(tmp_video_path))

    two_images_to_video(img_before_path, img_after_path, tmp_video_path, fps=1)

    # Q1: extract instruction targets
    EX1 = r"""
    EX1 (manipulation instruction -> targets intended to be moved solely as per the manipulation instruction)
    1) Manipulation Instruction: "The robotic arm moves the red cube to the right"          -> "red cube"
    2) Manipulation Instruction: "The robotic arm picks up the apple and puts it on the plate" -> "apple"
    3) Manipulation Instruction: "The robotic arm removes the blue cup from the table"      -> "blue cup"
    4) Manipulation Instruction: "The robotic arm moves the book to the top of the jar"   -> "book"
    """
    q1 = f"""
    TASK 1 — Instruction targets

    Manipulation Instruction:
    "{instruction}"
    
    Read the single-line manipulation instruction exactly as given above and LIST the object instance names
    that the instruction intends to move.
    
    NOTE: 
    - You should ignore the given video. The given manipulation instruction might differ from what actually happens in the given video. You should ignore whatever happens in the video and focus on what objects are intended to be moved by the given manipulation instruction only. 
    - Also, you should not consider the destination of the manipulation operation to be an object intended to be moved, unless the manipulation instruction explicitly states that the destination should also be moved.

    A generic example scenario: Say the given manipulation instruction intends to move object-x to the top of an object-y. But in the video you see object-x being moved to the top of object-y, but additionally object-z is also moved to some place. So here, object-y is just the manipulation destination and the manipulation instruction didn't mention any movement for object-y. As for object-z, there is no mention of it's movement in the manipulation instruction -- the video shows it's movement, but we are supposed to ignore the given video. So you should only output "object-x" since that's the only object intended to be moved by the given manipulation instruction.
    
    EXAMPLES:
    {EX1}

    OUTPUT: single short comma-separated line with only the names of the objects intended to be moved, as inferred solely from the given manipulation instruction. If the given manipulation instruction doesn't intend to move any object, then output "none".
    Manipulation Instruction:
    "{instruction}"
    """
    out1 = _safe_chat_call(
        video_chat_model,
        verbose=verbose,
        video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer,
        user_prompt=q1,
        return_history=False,
        max_num_frames=max_num_frames,
        generation_config=GEN_CONFIG_SHORT
    )
    instr_targets_line = out1.strip()
    if verbose:
        print("Q1 -> instruction targets:", instr_targets_line)

    # Clean up temp video
    if os.path.exists(tmp_video_path):
        os.remove(tmp_video_path)

    return _clean_list(instr_targets_line)


# =============================================================================
# SECTION 2: SAM Proposal Generation
# =============================================================================

def get_iou_between_masks(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.sum(mask1 * mask2)
    union = np.sum(np.maximum(mask1, mask2))
    return intersection / union if union > 0 else 0.0


def is_submask(mask1: np.ndarray, mask2: np.ndarray, iou_thresh: float) -> bool:
    """Returns True if mask1 is a submask of mask2."""
    if get_iou_between_masks(mask1, mask1 * mask2) >= iou_thresh:
        return True
    return False


def dfs(node, master_parent, visited, adj, is_master_parent):
    """DFS for submask detection."""
    visited[node] = True
    for i in range(len(adj[node])):
        if visited[adj[node][i]]:
            if adj[node][i] != master_parent and is_master_parent[adj[node][i]]:
                is_master_parent[adj[node][i]] = False
        else:
            dfs(adj[node][i], master_parent, visited, adj, is_master_parent)


def discard_submasks(masks: List[np.ndarray], iou_thresh: float = 0.82) -> List[np.ndarray]:
    """Discard masks that are submasks of other larger masks."""
    n = len(masks)
    visited = [False for _ in range(n)]
    is_master_parent = {i: False for i in range(n)}
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                if is_submask(masks[j], masks[i], iou_thresh):
                    adj[i].append(j)
    mask_sizes = [np.sum(mask) for mask in masks]
    sorted_indices = sorted(range(n), key=lambda i: mask_sizes[i], reverse=True)
    for i in sorted_indices:
        if not visited[i]:
            is_master_parent[i] = True
            dfs(i, i, visited, adj, is_master_parent)
    master_parent_masks = []
    for i in range(n):
        if is_master_parent[i]:
            master_parent_masks.append(masks[i])
    return master_parent_masks


def merge_similar_masks(masks: List[np.ndarray], sim_thresh: float = 0.82) -> List[np.ndarray]:
    """Merge masks with IoU >= sim_thresh."""
    n = len(masks)
    if n <= 1:
        return masks
    single_cluster = None
    pairwise_iou = {}
    for i in range(n):
        for j in range(n):
            if i < j:
                pairwise_iou[(i, j)] = get_iou_between_masks(masks[i], masks[j])
                pairwise_iou[(j, i)] = pairwise_iou[(i, j)]
    deleted = [False for _ in range(n)]
    key, max_iou = max(pairwise_iou.items(), key=lambda item: item[1])
    while max_iou >= sim_thresh:
        larger_node = key[0] if np.sum(masks[key[0]]) >= np.sum(masks[key[1]]) else key[1]
        smaller_node = key[1] if larger_node == key[0] else key[0]
        for i in range(n):
            if not deleted[i] and i != larger_node and i != smaller_node:
                pairwise_iou[(i, larger_node)] = min(pairwise_iou[(i, larger_node)], pairwise_iou[(i, smaller_node)])
                pairwise_iou[(larger_node, i)] = pairwise_iou[(i, larger_node)]
        for i in range(n):
            if not deleted[i] and i != smaller_node:
                del pairwise_iou[(i, smaller_node)]
                del pairwise_iou[(smaller_node, i)]
                deleted[smaller_node] = True
        if pairwise_iou:
            key, max_iou = max(pairwise_iou.items(), key=lambda item: item[1])
        else:
            single_cluster = larger_node
            break
    merged_masks = []
    if single_cluster is not None:
        merged_masks.append(masks[single_cluster])
    else:
        for i in range(n):
            if not deleted[i]:
                merged_masks.append(masks[i])
    return merged_masks


def init_sam_model(sam_checkpoint: str = SAM_CHECKPOINT, model_type: str = "vit_h"):
    """Initialize SAM model once and reuse its mask generator."""
    print(f"[INFO] Loading SAM model from {sam_checkpoint}")
    clear_cuda_cache()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    
    # Disable gradients for SAM
    for param in sam.parameters():
        param.requires_grad = False
    
    return SamAutomaticMaskGenerator(sam)


def generate_sam_proposals(
    image_path: str,
    mask_generator,
    return_overlay: bool = False
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """
    Run SAM inference on an image and return filtered masks.
    
    Returns:
        masks: List of binary masks (np.uint8)
        overlay: Optional visualization overlay
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_area = image.shape[0] * image.shape[1]
    min_area = int(image_area * MIN_MASK_AREA_RATIO)

    # Generate SAM proposals with memory optimization
    with torch.no_grad():
        sam_proposals = mask_generator.generate(image)
    
    sam_masks = [p["segmentation"].astype(np.uint8) for p in sam_proposals]
    
    # Clear GPU memory after inference
    clear_cuda_cache()

    # Filtering
    sam_masks = discard_submasks(sam_masks)
    sam_masks = merge_similar_masks(sam_masks)
    
    # Filter by minimum area to remove tiny fragments
    sam_masks = [m for m in sam_masks if np.sum(m) >= min_area]

    overlay = None
    if return_overlay:
        overlay = np.zeros(image.shape, dtype=np.uint8)
        for idx, mask in enumerate(sam_masks):
            color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
            overlay[mask == 1] = color

    return sam_masks, overlay


# =============================================================================
# SECTION 3: XMem Object Tracking
# =============================================================================

def run_xmem_tracking(
    first_frame_path: str,
    second_frame_path: str,
    sam_masks: List[np.ndarray],
    xmem_model_path: str = XMEM_MODEL_PATH,
    size: int = -1
) -> Dict[int, np.ndarray]:
    """
    Run XMem tracking on SAM masks from first frame to second frame.
    
    Args:
        first_frame_path: Path to first frame image
        second_frame_path: Path to second frame image
        sam_masks: List of SAM masks for first frame
        xmem_model_path: Path to XMem model weights
        size: Resize shorter side to this size (-1 for original resolution)
        
    Returns:
        tracked_masks: Dictionary mapping color tuple to mask in second frame
                       Only includes masks that are present in both frames
    """
    # Create temporary directory structure for XMem
    tmp_dir = tempfile.mkdtemp()
    jpeg_dir = os.path.join(tmp_dir, "JPEGImages", "video")
    annot_dir = os.path.join(tmp_dir, "Annotations", "video")
    output_dir = os.path.join(tmp_dir, "XMem_Output")
    
    os.makedirs(jpeg_dir, exist_ok=True)
    os.makedirs(annot_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy frames and ensure dimensions are divisible by 16
    img1 = cv2.imread(first_frame_path)
    img2 = cv2.imread(second_frame_path)
    
    # Store original dimensions for cropping later
    orig_h, orig_w = img1.shape[:2]
    
    # Ensure both images have the same dimensions
    h2, w2 = img2.shape[:2]
    if h2 != orig_h or w2 != orig_w:
        # Resize img2 to match img1
        img2 = cv2.resize(img2, (orig_w, orig_h))
    
    # Pad images to have dimensions divisible by 16
    h, w = orig_h, orig_w
    new_h = ((h + 15) // 16) * 16
    new_w = ((w + 15) // 16) * 16
    
    if new_h != h or new_w != w:
        # Pad with zeros (black)
        img1_padded = np.zeros((new_h, new_w, 3), dtype=img1.dtype)
        img2_padded = np.zeros((new_h, new_w, 3), dtype=img2.dtype)
        img1_padded[:h, :w] = img1
        img2_padded[:h, :w] = img2
        img1 = img1_padded
        img2 = img2_padded
        
        # Also pad SAM masks
        sam_masks_padded = []
        for mask in sam_masks:
            mask_padded = np.zeros((new_h, new_w), dtype=mask.dtype)
            mask_padded[:h, :w] = mask
            sam_masks_padded.append(mask_padded)
        sam_masks = sam_masks_padded
        h, w = new_h, new_w
    
    cv2.imwrite(os.path.join(jpeg_dir, "00000.jpg"), img1)
    cv2.imwrite(os.path.join(jpeg_dir, "00001.jpg"), img2)
    
    # Create annotation mask from SAM proposals as an INDEXED palette image
    # XMem expects a palette PNG where pixel values are object indices (1, 2, 3, ...)
    # Index 0 is reserved for background
    annotation_indexed = np.zeros((h, w), dtype=np.uint8)
    idx_to_color = {}  # Map index to color for later retrieval
    for idx, mask in enumerate(sam_masks):
        obj_idx = idx + 1  # Object indices start from 1 (0 is background)
        annotation_indexed[mask == 1] = obj_idx
        idx_to_color[obj_idx] = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
    
    # Save as palette PNG
    annot_img = Image.fromarray(annotation_indexed, mode='P')
    annot_img.putpalette(generate_color_palette())
    annot_img.save(os.path.join(annot_dir, "00000.png"))
    
    # XMem config
    config = {
        'model': xmem_model_path,
        'disable_long_term': False,
        'enable_long_term': True,
        'max_mid_term_frames': 10,
        'min_mid_term_frames': 5,
        'max_long_term_elements': 10000,
        'num_prototypes': 128,
        'top_k': 30,
        'mem_every': 5,
        'deep_update_every': -1,
        'size': size,
        'flip': False,
        'save_scores': False,
        'benchmark': False,
        'enable_long_term_count_usage': False,
    }
    
    # Load XMem with memory optimization
    clear_cuda_cache()
    torch.autograd.set_grad_enabled(False)
    network = XMem(config, xmem_model_path).cuda().eval()
    model_weights = torch.load(xmem_model_path, map_location='cuda')
    network.load_weights(model_weights, init_as_zero_if_needed=True)
    del model_weights  # Free the weights dict
    clear_cuda_cache()
    
    # Run inference
    meta_dataset = LongTestDataset(tmp_dir, size=size)
    meta_loader = meta_dataset.get_datasets()
    
    tracked_masks = {}
    
    for vid_reader in meta_loader:
        loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=0)
        mapper = MaskMapper()
        processor = InferenceCore(network, config=config)
        first_mask_loaded = False
        
        for ti, data in enumerate(loader):
            with torch.amp.autocast('cuda', enabled=True):
                rgb = data['rgb'].cuda()[0]
                msk = data.get('mask')
                info = data['info']
                frame = info['frame'][0]
                shape = info['shape']
                need_resize = info['need_resize'][0]
                
                if not first_mask_loaded:
                    if msk is not None:
                        first_mask_loaded = True
                    else:
                        continue
                
                # Map possibly non-continuous labels to continuous ones
                if msk is not None:
                    msk, labels = mapper.convert_mask(msk[0].numpy())
                    msk = torch.Tensor(msk).cuda()
                    if need_resize:
                        msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                    processor.set_all_labels(list(mapper.remappings.values()))
                else:
                    labels = None
                
                # Run the model
                prob = processor.step(rgb, msk, labels, end=(ti == len(loader) - 1))
                
                # Upsample to original size if needed
                if need_resize:
                    prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:, 0]
                
                # Probability mask -> index mask
                out_mask = torch.max(prob, dim=0).indices
                out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
                
                # Save second frame masks
                if ti == 1:  # Second frame
                    # Remap indices back to original object indices
                    out_mask = mapper.remap_index_mask(out_mask)
                    
                    # Crop back to original dimensions if padded
                    if out_mask.shape[0] != orig_h or out_mask.shape[1] != orig_w:
                        out_mask = out_mask[:orig_h, :orig_w]
                    
                    # Extract tracked masks by index (matching original SAM mask indices)
                    unique_indices = np.unique(out_mask)
                    for obj_idx in unique_indices:
                        if obj_idx == 0:  # Skip background
                            continue
                        # obj_idx corresponds to (SAM mask index + 1)
                        sam_idx = obj_idx - 1
                        if sam_idx < len(sam_masks):
                            color = idx_to_color.get(obj_idx, COLOR_PALETTE[sam_idx % len(COLOR_PALETTE)])
                            mask = (out_mask == obj_idx).astype(np.uint8)
                            if np.sum(mask) > 0:
                                tracked_masks[tuple(color)] = mask
    
    # Build color_to_idx mapping for consistency with rest of code
    color_to_idx = {tuple(color): (obj_idx - 1) for obj_idx, color in idx_to_color.items()}
    
    # Crop SAM masks back to original size if padded
    if sam_masks[0].shape[0] != orig_h or sam_masks[0].shape[1] != orig_w:
        sam_masks = [m[:orig_h, :orig_w] for m in sam_masks]
    
    # Cleanup - more aggressive
    del network, processor, mapper
    del meta_dataset, meta_loader
    clear_cuda_cache()
    
    # Remove temp directory
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return tracked_masks, color_to_idx, sam_masks


# =============================================================================
# SECTION 4: BLIP2 Verification and Captioning
# =============================================================================

def is_noise_caption(caption: str) -> bool:
    """
    Check if a caption represents a noise/artifact object that should be filtered out.
    """
    caption_lower = caption.lower().strip()
    
    # Check for empty or too short captions
    if len(caption_lower) < 2:
        return True
    
    # Check for noise keywords
    for keyword in NOISE_OBJECT_KEYWORDS:
        if keyword in caption_lower:
            return True
    
    # Robot/gripper: only filter exact/single-word matches, not compound phrases like "robotic arm"
    caption_words = caption_lower.split()
    if len(caption_words) <= 1:
        for keyword in ROBOT_EXACT_KEYWORDS:
            if keyword in caption_lower:
                return True
    elif caption_lower in ('robot arm', 'a robot', 'the robot', 'a gripper', 'the gripper'):
        return True
    
    # Check for generic/abstract captions that are likely false positives
    for keyword in GENERIC_CAPTION_KEYWORDS:
        if keyword in caption_lower:
            return True
    
    return False


def normalize_object_name(name: str) -> str:
    """Normalize an object name for comparison."""
    name = name.lower().strip()
    # Remove articles
    for article in ['a ', 'an ', 'the ', 'some ']:
        if name.startswith(article):
            name = name[len(article):]
    # Remove trailing punctuation
    name = name.rstrip('.,;:!?')
    return name.strip()


def objects_are_semantically_similar(obj1: str, obj2: str) -> bool:
    """
    Check if two object names are semantically similar.
    Uses a conservative approach - only match if clearly the same object.
    """
    n1 = normalize_object_name(obj1)
    n2 = normalize_object_name(obj2)
    
    # Exact match
    if n1 == n2:
        return True
    
    # Empty strings shouldn't match anything
    if not n1 or not n2:
        return False
    
    # EXPLICITLY DISSIMILAR objects - these should NEVER match each other
    # This is the most important check to avoid false positives
    dissimilar_pairs = [
        # Utensils - each is distinct
        ('fork', 'spoon'), ('fork', 'knife'), ('spoon', 'knife'),
        # Common kitchen objects vs "pot"
        ('pumpkin', 'pot'), ('corn', 'pot'), ('pepper', 'pot'),
        ('carrot', 'pot'), ('lettuce', 'pot'), ('broccoli', 'pot'),
        ('cabbage', 'pot'), ('tomato', 'pot'), ('cucumber', 'pot'),
        ('plate', 'pot'), ('bowl', 'pot'), ('hat', 'pot'),
        ('bottle', 'pot'), ('spoon', 'pot'), ('knife', 'pot'), ('fork', 'pot'),
        # Common kitchen objects vs "pan"
        ('pumpkin', 'pan'), ('corn', 'pan'), ('pepper', 'pan'),
        ('carrot', 'pan'), ('lettuce', 'pan'), ('broccoli', 'pan'),
        ('plate', 'pan'), ('bowl', 'pan'), ('hat', 'pan'),
        # Different vegetables
        ('carrot', 'pepper'), ('corn', 'pepper'), ('carrot', 'corn'),
        ('pumpkin', 'pepper'), ('pumpkin', 'corn'), ('pumpkin', 'carrot'),
        ('lettuce', 'pepper'), ('lettuce', 'corn'), ('lettuce', 'carrot'),
        ('broccoli', 'pepper'), ('broccoli', 'corn'), ('broccoli', 'carrot'),
        # Common objects vs "broccoli"
        ('bowl', 'broccoli'), ('container', 'broccoli'), ('plate', 'broccoli'),
        # Containers are distinct
        ('cup', 'bowl'), ('plate', 'bowl'), ('cup', 'plate'),
        # Different fruits
        ('apple', 'orange'), ('banana', 'apple'), ('banana', 'orange'),
        # Random dissimilar pairs
        ('hat', 'bowl'), ('hat', 'plate'), ('hat', 'container'),
        ('spoon', 'banana'), ('cube', 'spoon'), ('cube', 'banana'),
        ('board', 'carrot'), ('board', 'bowl'),
        # Lever shouldn't match other things
        ('lever', 'pot'), ('lever', 'pepper'), ('lever', 'vegetable'),
        # Microwave shouldn't match sushi
        ('microwave', 'sushi'),
    ]
    for pair in dissimilar_pairs:
        if (pair[0] in n1 and pair[1] in n2) or (pair[1] in n1 and pair[0] in n2):
            return False
    
    # Check for exact substring match where one is clearly part of the other
    # e.g., "bell pepper" contains "pepper", "corn cob" contains "corn"
    if n1 in n2 or n2 in n1:
        return True
    
    # Check for common word overlap (for compound nouns)
    words1 = set(n1.split())
    words2 = set(n2.split())
    
    # Significant words that indicate the same object
    significant_words = words1 & words2
    if significant_words:
        # Filter out color/size words
        non_modifiers = [w for w in significant_words if len(w) >= 3 and w not in 
                        {'red', 'blue', 'green', 'yellow', 'black', 'white', 'orange',
                         'big', 'small', 'large', 'tiny', 'the', 'and', 'plastic', 'metal'}]
        if non_modifiers:
            return True
    
    # Check for known synonyms - use exact matching within groups
    synonym_groups = [
        {'pot', 'saucepan', 'cooking pot'},
        {'pan', 'frying pan', 'skillet'},
        {'bowl', 'mixing bowl', 'salad bowl'},
        {'pepper', 'bell pepper', 'capsicum'},
        {'chili', 'chilly', 'chilli'},
        {'cloth', 'clothes', 'clothing', 'fabric'},
        {'corn', 'corn cob', 'maize'},
        {'pumpkin', 'squash'},
    ]
    
    for group in synonym_groups:
        n1_in_group = n1 in group
        n2_in_group = n2 in group
        if n1_in_group and n2_in_group:
            return True
    
    return False


def deduplicate_captions(captions: List[str]) -> List[str]:
    """
    Remove duplicate or near-duplicate captions from a list.
    """
    if not captions:
        return []
    
    unique = []
    for cap in captions:
        cap_norm = normalize_object_name(cap)
        is_dup = False
        for existing in unique:
            existing_norm = normalize_object_name(existing)
            if objects_are_semantically_similar(cap_norm, existing_norm):
                is_dup = True
                break
        if not is_dup:
            unique.append(cap)
    
    return unique


def init_blip(model_name: str = BLIP_MODEL):
    """Initialize BLIP-2 model for VQA and captioning with memory optimization."""
    print(f"[INFO] Loading BLIP-2 model: {model_name}")
    clear_cuda_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and image processor separately to avoid num_query_tokens issue
    from transformers import AutoTokenizer, BlipImageProcessor
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    image_processor = BlipImageProcessor.from_pretrained(model_name)
    processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    # Disable gradients for BLIP
    for param in model.parameters():
        param.requires_grad = False

    return processor, model, device


def crop_from_mask(image_bgr: np.ndarray, mask: np.ndarray) -> Optional[Image.Image]:
    """
    Crop bounding box around mask and zero-out background pixels.
    Returns a PIL RGB image (object only, background black) or None if mask empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if x2 < x1 or y2 < y1:
        return None

    # Skip tiny crops: when height or width <= 3, the numpy shape (e.g. (1,W,3))
    # causes transformers' infer_channel_dimension_format to misidentify the
    # spatial dimension as the channel dimension, breaking BLIP-2 normalize.
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    if w < 4 or h < 4:
        return None

    crop_img = image_bgr[y1:y2 + 1, x1:x2 + 1].copy()
    crop_mask = mask[y1:y2 + 1, x1:x2 + 1]
    crop_img[crop_mask == 0] = 0

    # Robustly ensure 3-channel uint8 RGB output
    if crop_img.ndim == 2:
        crop_rgb = np.stack([crop_img, crop_img, crop_img], axis=-1)
    elif crop_img.ndim == 3 and crop_img.shape[2] == 1:
        crop_rgb = np.concatenate([crop_img, crop_img, crop_img], axis=-1)
    elif crop_img.ndim == 3 and crop_img.shape[2] == 4:
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGRA2RGB)
    elif crop_img.ndim == 3 and crop_img.shape[2] == 3:
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    else:
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb.astype(np.uint8), mode='RGB')


@torch.no_grad()
def verify_blip2(
    obj_mask: np.ndarray,
    original_image_frame: np.ndarray,
    obj_description: str,
    blip_processor,
    blip_model,
    blip_device
) -> bool:
    """
    Ask BLIP-2 whether the crop of the original image frame as demarcated 
    by the obj_mask corresponds to the given obj_description.
    
    Uses caption-only verification to avoid BLIP2's tendency to answer "yes"
    to direct questions even when objects are only visually similar.
    """
    # Convert to BGR for cropping
    if original_image_frame.shape[2] == 3 and original_image_frame.dtype == np.uint8:
        # Check if RGB or BGR by looking at channel order
        image_bgr = cv2.cvtColor(original_image_frame, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = original_image_frame
    
    crop = crop_from_mask(image_bgr, obj_mask)
    if crop is None:
        return False

    # Ensure crop is RGB (BLIP-2 normalize expects 3 channels)
    if crop.mode != "RGB":
        crop = crop.convert("RGB")

    # Get a caption - this is more reliable than direct yes/no questions
    caption_prompt = "Question: What object is shown in this image? Answer with a single noun:"
    inputs = blip_processor(images=crop, text=caption_prompt, return_tensors="pt")
    moved_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.dtype.is_floating_point:
                moved_inputs[k] = v.to(blip_device, torch.float16)
            else:
                moved_inputs[k] = v.to(blip_device)
        else:
            moved_inputs[k] = v

    with torch.amp.autocast('cuda', dtype=torch.float16):
        outputs = blip_model.generate(
            **moved_inputs,
            max_new_tokens=20,
            do_sample=False
        )
    caption = blip_processor.batch_decode(outputs, skip_special_tokens=True)[0].lower().strip()
    
    # Clean up
    del inputs, moved_inputs, outputs
    
    # Check if caption is semantically similar to the description
    # This uses the dissimilar_pairs check which prevents incorrect matches
    return objects_are_semantically_similar(caption, obj_description)


@torch.no_grad()
def blip2_caption(
    obj_mask: np.ndarray,
    original_image_frame: np.ndarray,
    blip_processor,
    blip_model,
    blip_device
) -> str:
    """
    Ask BLIP-2 to predict a caption for the crop of the original image frame
    as demarcated by the obj_mask.
    """
    # Convert to BGR for cropping
    if original_image_frame.shape[2] == 3 and original_image_frame.dtype == np.uint8:
        image_bgr = cv2.cvtColor(original_image_frame, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = original_image_frame
    
    crop = crop_from_mask(image_bgr, obj_mask)
    if crop is None:
        return "unknown object"

    # Ensure crop is RGB (BLIP-2 normalize expects 3 channels)
    if crop.mode != "RGB":
        crop = crop.convert("RGB")

    prompt = "Question: What object is shown in this image? Answer:"

    inputs = blip_processor(images=crop, text=prompt, return_tensors="pt")
    moved_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.dtype.is_floating_point:
                moved_inputs[k] = v.to(blip_device, torch.float16)
            else:
                moved_inputs[k] = v.to(blip_device)
        else:
            moved_inputs[k] = v

    with torch.amp.autocast('cuda', dtype=torch.float16):
        outputs = blip_model.generate(
            **moved_inputs,
            max_new_tokens=50,
            do_sample=False
        )
    caption = blip_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    
    # Clean up
    del inputs, moved_inputs, outputs
    
    return caption


# =============================================================================
# SECTION 5: Main Algorithm
# =============================================================================

def num_unmentioned_removed_from_scene(
    img_before_path: str,
    img_after_path: str,
    instruction: str,
    video_chat_model,
    video_chat_tokenizer,
    sam_mask_generator,
    blip_processor,
    blip_model,
    blip_device,
    verbose: bool = False,
    output_dirs: Optional[Dict[str, str]] = None
) -> Tuple[int, List[str]]:
    """
    Compute number of UNMENTIONED objects that were REMOVED-FROM-SCENE.
    
    Algorithm:
    1) Extract mentioned objects from manipulation instruction using Gemini/VideoChat-Flash
    2) Perform removed-from-scene object detection:
       a) SAM proposal generation on first frame
       b) XMem tracking to get masks in second frame
       c) Identify objects present in first frame but absent in second (removed from scene)
       d) BLIP2 verification to filter out mentioned objects from the removed set
    
    Args:
        img_before_path: Path to first (before) frame
        img_after_path: Path to second (after) frame  
        instruction: Manipulation instruction text
        video_chat_model: Loaded VideoChat-Flash model (optional if using Gemini)
        video_chat_tokenizer: VideoChat-Flash tokenizer (optional if using Gemini)
        sam_mask_generator: SAM mask generator
        blip_processor: BLIP-2 processor
        blip_model: BLIP-2 model
        blip_device: Device for BLIP-2
        verbose: Print debug info
        output_dirs: Dictionary of output directory paths for logging
        
    Returns:
        count: Number of unmentioned removed-from-scene objects
        objects: List of object names/descriptions
    """
    
    # =========================================================================
    # STEP 1: Extract mentioned objects from instruction
    # =========================================================================
    if verbose:
        print("[STEP 1] Extracting instruction targets...")
    
    mentioned_objects = extract_instruction_targets(
        instruction=instruction,
        img_before_path=img_before_path,
        img_after_path=img_after_path,
        video_chat_model=video_chat_model,
        video_chat_tokenizer=video_chat_tokenizer,
        verbose=verbose
    )
    
    # Clear GPU memory after VideoChat-Flash inference
    clear_cuda_cache()
    
    if verbose:
        print(f"  Mentioned objects: {mentioned_objects}")
    
    # Save instruction targets
    if output_dirs:
        save_instruction_targets(
            output_dirs['instruction_targets'], 
            instruction, 
            mentioned_objects,
            INSTRUCTION_TARGET_METHOD
        )
    
    # =========================================================================
    # STEP 2a: SAM proposal generation on first frame
    # =========================================================================
    if verbose:
        print("[STEP 2a] Generating SAM proposals...")
    
    sam_masks, _ = generate_sam_proposals(img_before_path, sam_mask_generator)
    
    # Clear GPU memory after SAM inference
    clear_cuda_cache()
    
    if verbose:
        print(f"  Generated {len(sam_masks)} SAM masks")
    
    # Save SAM masks
    if output_dirs and len(sam_masks) > 0:
        save_sam_masks(output_dirs['sam_masks'], sam_masks, img_before_path)
    
    if len(sam_masks) == 0:
        return 0, []
    
    # =========================================================================
    # STEP 2b: XMem tracking to second frame
    # =========================================================================
    if verbose:
        print("[STEP 2b] Running XMem tracking...")
    
    tracked_masks, color_to_idx, original_masks = run_xmem_tracking(
        first_frame_path=img_before_path,
        second_frame_path=img_after_path,
        sam_masks=sam_masks
    )
    
    if verbose:
        print(f"  Tracked {len(tracked_masks)} masks to second frame")
    
    # Build mapping from color to first frame mask
    first_frame_masks_by_color = {}
    for idx, mask in enumerate(sam_masks):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        first_frame_masks_by_color[color] = mask
    
    # Identify "removed from scene" objects (present in SAM but not tracked)
    all_sam_colors = set(first_frame_masks_by_color.keys())
    tracked_colors = set(tracked_masks.keys())
    removed_from_scene_colors = list(all_sam_colors - tracked_colors)
    
    if verbose:
        print(f"  Detected {len(removed_from_scene_colors)} objects removed from scene")
    
    # Save XMem tracking results
    if output_dirs:
        save_xmem_tracking(
            output_dirs['xmem_tracking'], 
            tracked_masks, 
            color_to_idx,
            img_before_path,
            img_after_path
        )
    
    # =========================================================================
    # STEP 2c: Save removed-from-scene objects info (skip distance computation)
    # =========================================================================
    if verbose:
        print("[STEP 2c] Recording removed-from-scene objects...")
    
    first_frame = cv2.cvtColor(cv2.imread(img_before_path), cv2.COLOR_BGR2RGB)
    
    # For R3, we don't need LoFTR distances - we directly have the removed objects
    # Just save basic info about removed objects
    removed_info = []
    for color in removed_from_scene_colors:
        idx = color_to_idx.get(color, -1)
        removed_info.append({
            'mask_index': idx,
            'color_rgb': list(color) if isinstance(color, tuple) else color,
            'status': 'removed_from_scene'
        })
    
    # Save removed objects info
    if output_dirs:
        with open(os.path.join(output_dirs['removed_objects'], 'removed_objects.json'), 'w') as f:
            json.dump({
                'total_removed': len(removed_from_scene_colors),
                'removed_objects': removed_info
            }, f, indent=2)
        
        with open(os.path.join(output_dirs['removed_objects'], 'removed_objects.txt'), 'w') as f:
            f.write("Removed-from-Scene Objects:\n")
            f.write("=" * 50 + "\n")
            for r in removed_info:
                f.write(f"Mask {r['mask_index']:3d} | Color {r['color_rgb']} | Status: {r['status']}\n")
    
    # =========================================================================
    # STEP 2d: BLIP2 verification to filter out mentioned objects
    # =========================================================================
    if verbose:
        print("[STEP 2d] BLIP2 verification of removed-from-scene objects...")
    
    final_answer = []
    verification_results = []
    
    # Calculate minimum tracked mask area for filtering
    first_frame_img = cv2.imread(img_before_path)
    image_area = first_frame_img.shape[0] * first_frame_img.shape[1]
    min_tracked_area = int(image_area * MIN_TRACKED_MASK_AREA_RATIO)
    
    for idx, color in enumerate(removed_from_scene_colors):
        # Periodic cache clearing during BLIP verification
        if idx > 0 and idx % 3 == 0:
            clear_cuda_cache()
        
        # Get the mask from the first frame for captioning
        if color not in first_frame_masks_by_color:
            continue
        obj_mask = first_frame_masks_by_color[color]
        
        # Filter out masks that are too small (likely artifacts)
        mask_area = np.sum(obj_mask)
        if mask_area < min_tracked_area:
            if verbose:
                print(f"    Skipping small mask (area={mask_area}, min={min_tracked_area})")
            continue
        
        # Check if this removed object matches any mentioned (instruction target) object
        is_mentioned = False
        matched_object = None
        for men_obj in mentioned_objects:
            if verify_blip2(obj_mask, first_frame, men_obj, blip_processor, blip_model, blip_device):
                is_mentioned = True
                matched_object = men_obj
                if verbose:
                    print(f"    Removed object with color {color} matches mentioned object '{men_obj}'")
                break
        
        caption = None
        if not is_mentioned:
            # Get caption for this unmentioned removed object
            caption = blip2_caption(obj_mask, first_frame, blip_processor, blip_model, blip_device)
            
            # Filter out noise captions
            if not is_noise_caption(caption):
                final_answer.append(caption)
                if verbose:
                    print(f"    Unmentioned removed-from-scene object: {caption}")
            else:
                if verbose:
                    print(f"    Filtered noise caption: {caption}")
        
        # Store verification result for logging
        verification_results.append({
            'color': list(color) if isinstance(color, tuple) else color,
            'mask_index': color_to_idx.get(color, -1),
            'is_mentioned': is_mentioned,
            'matched_object': matched_object,
            'caption': caption,
            'mask_area': int(mask_area)
        })
    
    # Deduplicate captions
    final_answer = deduplicate_captions(final_answer)
    
    if verbose and len(final_answer) > 0:
        print(f"    Final deduplicated objects: {final_answer}")
    
    # Save BLIP2 verification results
    if output_dirs:
        save_blip2_verification(
            output_dirs['blip2_verification'],
            verification_results,
            img_before_path,
            first_frame_masks_by_color
        )
    
    # Final cleanup
    clear_cuda_cache()
    
    return len(final_answer), final_answer


# =============================================================================
# SECTION 6: Model Loading and Test Runner
# =============================================================================

def load_all_models():
    """Load all required models with memory optimization.
    
    Note: R3 does not require LoFTR since we don't compute movement distances.
    We only need to identify objects that are removed from the scene.
    """
    print("[INFO] Loading all models for R3 (removed-from-scene detection)...")
    print(f"[INFO] Instruction target extraction method: {INSTRUCTION_TARGET_METHOD}")
    
    # Set memory-efficient CUDA settings
    if torch.cuda.is_available():
        # Enable memory-efficient attention if available
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # VideoChat-Flash (largest model - load only if needed)
    video_chat_model = None
    video_chat_tokenizer = None
    if INSTRUCTION_TARGET_METHOD == "videochat":
        video_chat_tokenizer, video_chat_model = load_videochat_model()
        clear_cuda_cache()
    elif INSTRUCTION_TARGET_METHOD in ("local_llm", "gemini"):
        # Load the small local LLM for instruction target extraction
        load_local_llm()
        clear_cuda_cache()
        print("[INFO] Skipping VideoChatFlash model loading (using local LLM instead)")
    else:
        print("[INFO] Skipping VideoChatFlash model loading")
    
    # SAM
    sam_mask_generator = init_sam_model()
    clear_cuda_cache()
    
    # Note: LoFTR not needed for R3 - we skip distance computation
    print("[INFO] Skipping LoFTR model loading (not needed for removed-from-scene detection)")
    
    # BLIP-2
    blip_processor, blip_model, blip_device = init_blip()
    clear_cuda_cache()
    
    print("[INFO] All models loaded successfully!")
    
    return {
        'video_chat_model': video_chat_model,
        'video_chat_tokenizer': video_chat_tokenizer,
        'sam_mask_generator': sam_mask_generator,
        'blip_processor': blip_processor,
        'blip_model': blip_model,
        'blip_device': blip_device,
    }


def run_evaluation(models: Dict, test_cases: List[Dict], verbose: bool = False):
    """Run evaluation on test cases with memory optimization and comprehensive logging."""
    total = len(test_cases)
    score_sum = 0.0
    
    # Delete and recreate main output directory for fresh run
    if os.path.exists(R3_OUTPUT_DIR):
        shutil.rmtree(R3_OUTPUT_DIR)
    os.makedirs(R3_OUTPUT_DIR, exist_ok=True)
    
    # Save overall run info
    run_info = {
        'total_tests': total,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'instruction_target_method': INSTRUCTION_TARGET_METHOD,
        'task': 'unmentioned_removed_from_scene_detection',
    }
    with open(os.path.join(R3_OUTPUT_DIR, 'run_info.json'), 'w') as f:
        json.dump(run_info, f, indent=2)
    
    all_results = []
    
    for i, tc in enumerate(test_cases, start=1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{total}: {tc['description']}")
        print(f"{'='*60}")
        
        # Setup output directories for this test
        output_dirs = setup_test_output_dir(f"test_{i:02d}_{tc['description']}")
        
        # Copy input images to output directory for reference
        if os.path.exists(tc["before"]):
            shutil.copy(tc["before"], os.path.join(output_dirs['root'], 'input_before.jpg'))
        if os.path.exists(tc["after"]):
            shutil.copy(tc["after"], os.path.join(output_dirs['root'], 'input_after.png'))
        
        # Clear cache before each test
        clear_cuda_cache()
        
        predicted_count, objects = num_unmentioned_removed_from_scene(
            img_before_path=tc["before"],
            img_after_path=tc["after"],
            instruction=tc["instruction"],
            video_chat_model=models['video_chat_model'],
            video_chat_tokenizer=models['video_chat_tokenizer'],
            sam_mask_generator=models['sam_mask_generator'],
            blip_processor=models['blip_processor'],
            blip_model=models['blip_model'],
            blip_device=models['blip_device'],
            verbose=verbose,
            output_dirs=output_dirs
        )
        
        print(f"  Expected count: {tc['expected_count']}")
        print(f"  Predicted count: {predicted_count}")
        print(f"  Unmentioned removed-from-scene objects: {objects}")
        
        score = 2 ** (-abs(predicted_count - tc["expected_count"]))
        score_sum += score
        print(f"  Score: {score:.4f}")
        
        # Save final results for this test
        save_final_results(output_dirs['final_results'], tc, predicted_count, objects, score)
        
        # Track results for summary
        all_results.append({
            'test_index': i,
            'description': tc['description'],
            'expected': tc['expected_count'],
            'predicted': predicted_count,
            'score': score,
            'objects': objects
        })
        
        # Clear cache after each test
        clear_cuda_cache()
    
    success_score = score_sum / total
    print(f"\n{'='*60}")
    print(f"Final success_score = {success_score:.6f} ({success_score * 100:.2f}%)")
    print(f"{'='*60}")
    
    # Save overall summary
    summary = {
        'final_score': success_score,
        'final_score_percent': f"{success_score * 100:.2f}%",
        'total_tests': total,
        'task': 'unmentioned_removed_from_scene_detection',
        'results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(R3_OUTPUT_DIR, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save human-readable summary
    with open(os.path.join(R3_OUTPUT_DIR, 'evaluation_summary.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("R3 Evaluation Summary: Unmentioned Removed-from-Scene Objects\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Final Score: {success_score:.4f} ({success_score * 100:.2f}%)\n")
        f.write(f"Total Tests: {total}\n\n")
        f.write("-" * 60 + "\n")
        f.write("Individual Test Results:\n")
        f.write("-" * 60 + "\n\n")
        for r in all_results:
            f.write(f"Test {r['test_index']}: {r['description']}\n")
            f.write(f"  Expected: {r['expected']}, Predicted: {r['predicted']}, Score: {r['score']:.4f}\n")
            if r['objects']:
                f.write(f"  Detected removed objects: {', '.join(r['objects'])}\n")
            f.write("\n")
    
    print(f"\n[INFO] All outputs saved to: {R3_OUTPUT_DIR}")
    
    return success_score


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test cases for R3: Unmentioned Removed-from-Scene Objects Detection
    # 
    # R3 detects objects that:
    # 1. Were present in the first frame but absent in the second frame (removed from scene)
    # 2. Were NOT mentioned as the target of the manipulation instruction
    #
    test_cases = [
        {
            "before": "lgie-eval-data/input_2.jpg",
            "after": "lgie-eval-data/output_2.png",
            "instruction": "The robotic arm puts the omelette on top of the cabbage",
            "expected_count": 0,  # R3: No objects removed from scene
            "description": "Omelette on cabbage - no objects removed from scene"
        },
        {
            "before": "lgie-eval-data/input_1.jpg",
            "after": "lgie-eval-data/output_1_1.png",
            "instruction": "The robotic arm puts the cucumber on top of the blue cube",
            "expected_count": 0,  # R3: No objects removed from scene
            "description": "Cucumber on blue cube - no objects removed from scene"
        },
        {
            "before": "lgie-eval-data/input_5.jpg",
            "after": "lgie-eval-data/output_5_1.png",
            "instruction": "The robotic arm puts the sushi into the pot",
            "expected_count": 0,  # R3: No objects removed from scene (sushi goes into pot, still visible)
            "description": "Sushi into pot - no objects removed from scene"
        },
        {
            "before": "lgie-eval-data/input_5.jpg",
            "after": "lgie-eval-data/output_5_1.png",
            "instruction": "The robotic arm puts the carrot into the pot",
            "expected_count": 0,  # R3: No objects removed from scene
            "description": "Carrot into pot (wrong instruction) - no objects removed from scene"
        },
        {
            "before": "lgie-eval-data/input_4.jpg",
            "after": "lgie-eval-data/output_4_1.png",
            "instruction": "The robotic arm moves the blue spoon out of the scene",
            "expected_count": 0,  # R3: Blue spoon removed but it's mentioned, so 0 unmentioned
            "description": "Blue spoon removed from scene (mentioned) - 0 unmentioned removals"
        },
        {
            "before": "lgie-eval-data/input_4.jpg",
            "after": "lgie-eval-data/output_4_2.png",
            "instruction": "The robotic arm moves the blue spoon out of the scene",
            "expected_count": 2,  # R3: Blue spoon (mentioned), banana and blue cube (unmentioned) removed
            "description": "Blue spoon, banana, blue cube removed - 2 unmentioned (banana, blue cube)"
        },
          {
            "id": 0,
            "before": "lgie-eval-data/original_image/oi_0.png",
            "after": "lgie-eval-data/r3_gemini_output/out_0.png",
            "instruction": "put carrot on plate",
            "removed_objects": [
              "blue bowl"
            ],
            "expected_count": 1,
            "description": "The instruction mentions manipulation of carrot and plate, but the model removed blue bowl which is unmentioned."
          },
          {
            "id": 1,
            "before": "lgie-eval-data/original_image/oi_1.png",
            "after": "lgie-eval-data/r3_gemini_output/out_1.png",
            "instruction": "put broccoli in pot or pan",
            "removed_objects": [
              "carrot",
              "blue bowl",
              "green container",
              "red bell pepper"
            ],
            "expected_count": 4,
            "description": "The instruction mentions manipulation of broccoli, and pot (pan) but the model removed carrot, blue bowl, green container, and red bell pepper which are unmentioned."
          },
          {
            "id": 2,
            "before": "lgie-eval-data/original_image/oi_2.png",
            "after": "lgie-eval-data/r3_gemini_output/out_2.png",
            "instruction": "pick up pan from stove",
            "removed_objects": [
              "yellow bell pepper",
              "orange pumpkin",
              "green plate"
            ],
            "expected_count": 3,
            "description": "The instruction mentions manipulation of pan and stove, but the model removed yellow bell pepper, orange pumpkin, and green plate which are unmentioned."
          },
          {
            "id": 3,
            "before": "lgie-eval-data/original_image/oi_3.png",
            "after": "lgie-eval-data/r3_gemini_output/out_3.png",
            "instruction": "take sushi out of pan",
            "removed_objects": [
              "corn"
            ],
            "expected_count": 1,
            "description": "The instruction mentions manipulation of sushi and pan, but the model removed corn which is unmentioned."
          },
          {
            "id": 4,
            "before": "lgie-eval-data/original_image/oi_4.png",
            "after": "lgie-eval-data/r3_gemini_output/out_4.png",
            "instruction": "close white1fbox flap",
            "removed_objects": [
              "bean container"
            ],
            "expected_count": 1,
            "description": "The instruction mentions manipulation of white1fbox flap, but the model removed bean container which is unmentioned."
          },
          {
            "id": 5,
            "before": "lgie-eval-data/original_image/oi_5.png",
            "after": "lgie-eval-data/r3_gemini_output/out_5.png",
            "instruction": "put fork from basket to tray",
            "removed_objects": [
              "robotic arm",
              "gray basket"
            ],
            "expected_count": 2,
            "description": "The instruction mentions manipulation of fork, basket, and tray, but the model removed robotic arm and gray basket which are unmentioned."
          },
          {
            "id": 6,
            "before": "lgie-eval-data/original_image/oi_6.png",
            "after": "lgie-eval-data/r3_gemini_output/out_6.png",
            "instruction": "put red bottle in sink",
            "removed_objects": [
              "carrot",
              "blue bowl"
            ],
            "expected_count": 2,
            "description": "The instruction mentions manipulation of red bottle and sink, but the model removed carrot and blue bowl which are unmentioned."
          },
          {
            "id": 7,
            "before": "lgie-eval-data/original_image/oi_7.png",
            "after": "lgie-eval-data/r3_gemini_output/out_7.png",
            "instruction": "pour almonds in pot",
            "removed_objects": [
              "red bell pepper"
            ],
            "expected_count": 1,
            "description": "The instruction mentions manipulation of almonds and pot, but the model removed red bell pepper which is unmentioned."
          },
          {
            "id": 8,
            "before": "lgie-eval-data/original_image/oi_8.png",
            "after": "lgie-eval-data/r3_gemini_output/out_8.png",
            "instruction": "turn lever vertical to front",
            "removed_objects": [
              "metal pot",
              "red bell pepper",
              "green vegetable"
            ],
            "expected_count": 3,
            "description": "The instruction mentions manipulation of lever, but the model removed metal pot, red bell pepper, and green vegetable which are unmentioned."
          },
          {
            "id": 9,
            "before": "lgie-eval-data/original_image/oi_9.png",
            "after": "lgie-eval-data/r3_gemini_output/out_9.png",
            "instruction": "put spoon on plate",
            "removed_objects": [
              "fork",
              "knife"
            ],
            "expected_count": 2,
            "description": "The instruction mentions manipulation of spoon and plate, but the model removed fork and knife which are unmentioned."
          },
          {
            "id": 10,
            "before": "lgie-eval-data/original_image/oi_10.png",
            "after": "lgie-eval-data/r3_gemini_output/out_10.png",
            "instruction": "pick up bit holder",
            "removed_objects": [
              "big screwdriver on right",
              "bit holder"
            ],
            "expected_count": 2,
            "description": "The instruction mentions manipulation of bit holder, but the model removed big screwdriver on right and bit holder which are unmentioned."
          },
          {
            "id": 11,
            "before": "lgie-eval-data/original_image/oi_11.png",
            "after": "lgie-eval-data/r3_gemini_output/out_11.png",
            "instruction": "close microwave",
            "removed_objects": [
              "red sushi"
            ],
            "expected_count": 1,
            "description": "The instruction mentions manipulation of microwave, but the model removed red sushi which is unmentioned."
          },
          {
            "id": 12,
            "before": "lgie-eval-data/original_image/oi_12.png",
            "after": "lgie-eval-data/r3_gemini_output/out_12.png",
            "instruction": "put banana in colander",
            "removed_objects": [
              "green cutting board"
            ],
            "expected_count": 1,
            "description": "The instruction mentions manipulation of banana and colander, but the model removed green cutting board which is unmentioned."
          },
          {
            "id": 13,
            "before": "lgie-eval-data/original_image/oi_13.png",
            "after": "lgie-eval-data/r3_gemini_output/out_13.png",
            "instruction": "lever vertical to front",
            "removed_objects": [],
            "expected_count": 0,
            "description": "The instruction mentions manipulation of lever, and no unmentioned objects were removed."
          },
          {
            "id": 14,
            "before": "lgie-eval-data/original_image/oi_14.png",
            "after": "lgie-eval-data/r3_gemini_output/out_14.png",
            "instruction": "flip pot upright in sink",
            "removed_objects": [
              "red bell pepper",
              "orange pumpkin"
            ],
            "expected_count": 2,
            "description": "The instruction mentions manipulation of pot and sink, but the model removed red bell pepper and orange pumpkin which are unmentioned."
          },
          {
            "id": 15,
            "before": "lgie-eval-data/original_image/oi_15.png",
            "after": "lgie-eval-data/r3_gemini_output/out_15.png",
            "instruction": "put fork from basket to tray",
            "removed_objects": [],
            "expected_count": 0,
            "description": "The instruction mentions manipulation of fork, basket, and tray, and no unmentioned objects were removed."
          },
          {
            "id": 16,
            "before": "lgie-eval-data/original_image/oi_16.png",
            "after": "lgie-eval-data/r3_gemini_output/out_16.png",
            "instruction": "flip pot upright in sink",
            "removed_objects": [
              "corn",
              "yellow bell pepper"
            ],
            "expected_count": 2,
            "description": "The instruction mentions manipulation of pot and sink, but the model removed corn and yellow bell pepper which are unmentioned."
          },
          {
            "id": 17,
            "before": "lgie-eval-data/original_image/oi_17.png",
            "after": "lgie-eval-data/r3_gemini_output/out_17.png",
            "instruction": "put corn in pot which is in sink",
            "removed_objects": [
              "green vegetable"
            ],
            "expected_count": 1,
            "description": "The instruction mentions manipulation of corn, pot, and sink, but the model removed green vegetable which is unmentioned."
          },
          {
            "id": 18,
            "before": "lgie-eval-data/original_image/oi_18.png",
            "after": "lgie-eval-data/r3_gemini_output/out_18.png",
            "instruction": "put broccoli in bowl",
            "removed_objects": [
              "red sauce bottle"
            ],
            "expected_count": 1,
            "description": "The instruction mentions manipulation of broccoli and bowl, but the model removed red sauce bottle which is unmentioned."
          },
          {
            "id": 19,
            "before": "lgie-eval-data/original_image/oi_19.png",
            "after": "lgie-eval-data/r3_gemini_output/out_19.png",
            "instruction": "pick up closest rainbow Allen key set",
            "removed_objects": [],
            "expected_count": 0,
            "description": "The instruction mentions manipulation of rainbow Allen key set, and no unmentioned objects were removed."
          },
          {
            "id": 20,
            "before": "lgie-eval-data/original_image/oi_20.png",
            "after": "lgie-eval-data/r3_gemini_output/out_20.png",
            "instruction": "open oven",
            "removed_objects": [],
            "expected_count": 0,
            "description": "The instruction mentions manipulation of oven, and no unmentioned objects were removed."
          },
          {
            "id": 21,
            "before": "lgie-eval-data/original_image/oi_21.png",
            "after": "lgie-eval-data/r3_gemini_output/out_21.png",
            "instruction": "put pot in sink",
            "removed_objects": [
              "green cutting board",
              "green plate",
              "yellow plastic knife",
              "yellow bell pepper",
              "corn",
              "red bell pepper",
              "lettuce",
              "red chilly",
              "orange pumpkin"
            ],
            "expected_count": 9,
            "description": "The instruction mentions manipulation of pot and sink, but the model removed green cutting board, green plate, yellow plastic knife, yellow bell pepper, corn, red bell pepper, lettuce, red chilly, and orange pumpkin which are unmentioned."
          },
          {
            "id": 22,
            "before": "lgie-eval-data/original_image/oi_22.png",
            "after": "lgie-eval-data/r3_gemini_output/out_22.png",
            "instruction": "take clothes out of laundry machine",
            "removed_objects": [
              "red cloth",
              "blue cloth inside spin tub"
            ],
            "expected_count": 2,
            "description": "The instruction mentions manipulation of clothes and laundry machine, but the model removed red cloth and blue cloth inside spin tub which are unmentioned."
          }

    ]
    
    # Load models
    models = load_all_models()
    
    # Run evaluation
    run_evaluation(models, test_cases, verbose=True)
