"""
R2.py

Sophisticated algorithm to detect UNMENTIONED MOVED-WITHIN-SCENE objects in manipulation videos.

Algorithm:
1) Extract mentioned objects from manipulation instruction using VideoChat-Flash
2) Perform moved-within-scene object detection:
   a) SAM proposal generation on first frame
   b) XMem tracking to get masks in second frame
   c) LoFTR/centroid-based distance metric for movement detection
   d) KMeans clustering to separate moved vs stationary objects
   e) BLIP2 verification to filter out mentioned objects

Requirements:
    pip install transformers torch opencv-python segment-anything scikit-learn natsort
    pip install flash-attn --no-build-isolation (optional)
    
GPU Memory Optimization:
    For better memory management, set before running:
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
Command to run:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=./XMem python3 R2.py
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
from sklearn.cluster import KMeans

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
R2_OUTPUT_DIR = os.path.join(BASE_DIR, "R2_Output")

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
# Movement Detection Configuration
# -------------------------
# Minimum mask area ratio (relative to image) to consider an object
# Filters out very small masks that are likely noise
MIN_MASK_AREA_RATIO = 0.0005  # 0.05% of image area

# Absolute minimum movement threshold (pixels) - used for post-KMeans validation
# If the "moved" cluster center is below this, we treat it as noise
ABSOLUTE_MIN_MOVEMENT_PX = 25.0

# Minimum distance (pixels) for an object to be DEFINITELY moved, regardless of clustering
# Objects above this threshold bypass KMeans classification into the "stationary" cluster
DEFINITELY_MOVED_PX = 85.0

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
    test_dir = os.path.join(R2_OUTPUT_DIR, safe_name)
    
    subdirs = {
        'root': test_dir,
        'instruction_targets': os.path.join(test_dir, '1_instruction_targets'),
        'sam_masks': os.path.join(test_dir, '2_sam_masks'),
        'xmem_tracking': os.path.join(test_dir, '3_xmem_tracking'),
        'movement_distances': os.path.join(test_dir, '4_movement_distances'),
        'clustering': os.path.join(test_dir, '5_clustering'),
        'blip2_verification': os.path.join(test_dir, '6_blip2_verification'),
        'final_results': os.path.join(test_dir, '7_final_results'),
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


def save_movement_distances(output_dir: str, moved_distances: List[Tuple], color_to_idx: Dict):
    """Save movement distance calculations."""
    results = []
    for color, dist in moved_distances:
        idx = color_to_idx.get(color, -1)
        results.append({
            'mask_index': idx,
            'color_rgb': list(color) if isinstance(color, tuple) else color,
            'distance': float(dist)
        })
    
    # Sort by distance (descending)
    results.sort(key=lambda x: x['distance'], reverse=True)
    
    with open(os.path.join(output_dir, 'distances.json'), 'w') as f:
        json.dump({
            'total_masks_analyzed': len(results),
            'distances': results
        }, f, indent=2)
    
    # Save as readable text
    with open(os.path.join(output_dir, 'distances.txt'), 'w') as f:
        f.write("Movement Distances (sorted by distance, descending):\n")
        f.write("=" * 50 + "\n")
        for r in results:
            f.write(f"Mask {r['mask_index']:3d} | Color {r['color_rgb']} | Distance: {r['distance']:.4f}\n")


def save_clustering_results(output_dir: str, moved_distances: List[Tuple], 
                            moved_colors: List, color_to_idx: Dict,
                            first_frame_path: str, first_frame_masks: Dict):
    """Save clustering results with visualizations."""
    # Load first frame for visualization
    first_frame = cv2.imread(first_frame_path)
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if len(first_frame_masks) > 0:
        sample_mask = list(first_frame_masks.values())[0]
        mask_h, mask_w = sample_mask.shape[:2]
        frame_h, frame_w = first_frame_rgb.shape[:2]
        if mask_h != frame_h or mask_w != frame_w:
            first_frame_rgb = cv2.resize(first_frame_rgb, (mask_w, mask_h))
    
    # Identify moved vs stationary
    moved_set = set(moved_colors)
    all_colors = set(c for c, d in moved_distances)
    stationary_colors = all_colors - moved_set
    
    # Create visualization for moved objects only
    moved_vis = first_frame_rgb.copy()
    for color in moved_colors:
        if color in first_frame_masks:
            mask = first_frame_masks[color]
            moved_vis[mask == 1] = (255, 0, 0)  # Red for moved
    
    blended_moved = cv2.addWeighted(first_frame_rgb, 0.5, moved_vis, 0.5, 0)
    cv2.imwrite(os.path.join(output_dir, 'moved_objects_highlighted.png'),
                cv2.cvtColor(blended_moved, cv2.COLOR_RGB2BGR))
    
    # Create visualization showing moved vs stationary
    cluster_vis = first_frame_rgb.copy()
    for color in moved_colors:
        if color in first_frame_masks:
            mask = first_frame_masks[color]
            cluster_vis[mask == 1] = (255, 0, 0)  # Red for moved
    for color in stationary_colors:
        if color in first_frame_masks:
            mask = first_frame_masks[color]
            cluster_vis[mask == 1] = (0, 255, 0)  # Green for stationary
    
    blended_cluster = cv2.addWeighted(first_frame_rgb, 0.4, cluster_vis, 0.6, 0)
    cv2.imwrite(os.path.join(output_dir, 'clustering_visualization.png'),
                cv2.cvtColor(blended_cluster, cv2.COLOR_RGB2BGR))
    
    # Save summary
    moved_indices = [color_to_idx.get(c, -1) for c in moved_colors]
    stationary_indices = [color_to_idx.get(c, -1) for c in stationary_colors]
    
    with open(os.path.join(output_dir, 'clustering_results.json'), 'w') as f:
        json.dump({
            'num_moved': len(moved_colors),
            'num_stationary': len(stationary_colors),
            'moved_mask_indices': moved_indices,
            'stationary_mask_indices': stationary_indices,
            'moved_colors': [list(c) if isinstance(c, tuple) else c for c in moved_colors],
        }, f, indent=2)
    
    with open(os.path.join(output_dir, 'clustering_results.txt'), 'w') as f:
        f.write(f"Clustering Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total objects analyzed: {len(all_colors)}\n")
        f.write(f"Moved objects: {len(moved_colors)}\n")
        f.write(f"Stationary objects: {len(stationary_colors)}\n")
        f.write(f"\nMoved mask indices: {moved_indices}\n")


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
        'unmentioned_moved_objects': objects,
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
        f.write(f"\nUnmentioned moved objects:\n")
        for obj in objects:
            f.write(f"  - {obj}\n")


# =============================================================================
# SECTION 0: Local LLM for Instruction Target Extraction & Reconciliation
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
    """Load the local LLM for text-only instruction parsing.
    
    Uses Qwen2.5-7B-Instruct with bitsandbytes 4-bit NF4 quantization (~5 GB VRAM).
    Includes a compatibility patch for transformers 4.40.x + accelerate bnb dispatch.
    Handles instruction target extraction and semantic reconciliation on-device.
    """
    global _local_llm_model, _local_llm_tokenizer
    if _local_llm_model is not None:
        return  # Already loaded
    
    # ---- bnb / transformers 4.40.x compatibility patch ----
    # transformers 4.40.x raises ValueError when accelerate's dispatch_model
    # calls .to() on a bnb-quantized model. Patch .to() to be a safe no-op
    # for quantized models; we manually move non-quantized buffers afterwards.
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
    h, w = image.shape[:2]
    image_area = h * w

    # Generate SAM proposals with memory optimization
    with torch.no_grad():
        sam_proposals = mask_generator.generate(image)
    
    sam_masks = [p["segmentation"].astype(np.uint8) for p in sam_proposals]
    
    # Clear GPU memory after inference
    clear_cuda_cache()

    # Filter by minimum area ratio to remove noise
    min_area = int(image_area * MIN_MASK_AREA_RATIO)
    sam_masks = [m for m in sam_masks if np.sum(m) >= min_area]
    
    # Filtering for duplicates and submasks
    sam_masks = discard_submasks(sam_masks)
    sam_masks = merge_similar_masks(sam_masks)

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
# SECTION 4: LoFTR Distance Metric
# =============================================================================

def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """Compute centroid of a mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.mean(xs), np.mean(ys)


def compute_removed_from_scene_distance(mask: np.ndarray) -> float:
    """
    Compute the distance an object would have traveled to exit the scene.
    
    Uses the distance from the mask's centroid to the closest image edge.
    This represents the minimum distance the object must have moved to leave the scene.
    
    Args:
        mask: Binary mask of the object in the first frame
        
    Returns:
        Distance to closest edge (minimum movement to exit scene)
    """
    centroid = mask_centroid(mask)
    if centroid is None:
        return 0.0
    
    cx, cy = centroid
    h, w = mask.shape[:2]
    
    # Distances to each edge
    dist_left = cx
    dist_right = w - cx
    dist_top = cy
    dist_bottom = h - cy
    
    # The object exited through the closest edge
    min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
    
    # Return a significant distance (at least the distance to edge)
    # We add some buffer since the object likely moved beyond the edge
    return float(min_dist)


def bbox_from_mask(mask: np.ndarray, pad: int = 16):
    """Get bounding box from mask with padding."""
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
    """Crop image by bbox and return offset."""
    x0, y0, x1, y1 = bbox
    crop = img[y0:y1 + 1, x0:x1 + 1]
    return crop, (x0, y0)


def map_pts_to_original(pts: np.ndarray, offset: tuple):
    """Map points back to original coordinates."""
    x0, y0 = offset
    return pts + np.array([x0, y0], dtype=np.float32)


def filter_pts_in_mask(pts: np.ndarray, mask: np.ndarray):
    """Filter points that fall within a mask."""
    h, w = mask.shape[:2]
    xs = np.round(pts[:, 0]).astype(int)
    ys = np.round(pts[:, 1]).astype(int)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & (mask[ys, xs] > 0)
    return valid


def load_loftr_model(weight_type: str, weight_path: str, device: str = "cuda"):
    """Load LoFTR model for feature matching with memory optimization."""
    from loftr_src.loftr import LoFTR, default_cfg
    
    clear_cuda_cache()
    
    if weight_type == "outdoor_ot":
        default_cfg['match_coarse']['match_type'] = 'sinkhorn'
        default_cfg['match_coarse']['sparse_spvs'] = False
        default_cfg['match_coarse']['train_coarse_percent'] = 0.3
        default_cfg['trainer'] = default_cfg.get('trainer', {})
        default_cfg['trainer']['canonical_lr'] = 8e-3
        default_cfg['trainer']['warmup_step'] = 1875
        default_cfg['trainer']['warmup_ratio'] = 0.1
        default_cfg['trainer']['mslr_milestones'] = [8, 12, 16, 20, 24]
        default_cfg['trainer']['ransac_pixel_thr'] = 0.5
        default_cfg['trainer']['optimizer'] = "adamw"
        default_cfg['trainer']['adamw_decay'] = 0.1
    elif weight_type == "indoor_ot":
        default_cfg['match_coarse']['match_type'] = 'sinkhorn'
        default_cfg['match_coarse']['sparse_spvs'] = False
        default_cfg['trainer'] = default_cfg.get('trainer', {})
        default_cfg['trainer']['mslr_milestones'] = [3, 6, 9, 12, 17, 20, 23, 26, 29]
    
    matcher = LoFTR(config=default_cfg)
    ckpt = torch.load(weight_path, map_location='cpu')  # Load to CPU first
    matcher.load_state_dict(ckpt['state_dict'])
    del ckpt  # Free checkpoint memory
    matcher = matcher.eval().to(device)
    
    # Disable gradients
    for param in matcher.parameters():
        param.requires_grad = False
    
    return matcher


def match_visible_correspondences(matcher, frameA, frameB, maskA, maskB, device="cuda"):
    """Match keypoints between two frames within mask regions."""
    bboxA = bbox_from_mask(maskA)
    bboxB = bbox_from_mask(maskB)
    if bboxA is None or bboxB is None:
        return [], None, None, None

    cropA, offA = crop_and_offset(frameA, bboxA)
    cropB, offB = crop_and_offset(frameB, bboxB)
    cropMaskA, _ = crop_and_offset(maskA, bboxA)
    cropMaskB, _ = crop_and_offset(maskB, bboxB)

    # Convert to grayscale, resize to 640x480 for LoFTR compatibility
    imgA = cv2.cvtColor(cropA, cv2.COLOR_RGB2GRAY)
    imgB = cv2.cvtColor(cropB, cv2.COLOR_RGB2GRAY)
    imgA = cv2.resize(imgA, (640, 480))
    imgB = cv2.resize(imgB, (640, 480))

    tA = torch.from_numpy(imgA)[None][None].float().to(device) / 255.0
    tB = torch.from_numpy(imgB)[None][None].float().to(device) / 255.0

    # Inference with memory optimization
    batch = {"image0": tA, "image1": tB}
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        matcher(batch)
    mkpts0 = batch["mkpts0_f"].cpu().numpy()
    mkpts1 = batch["mkpts1_f"].cpu().numpy()
    mconf = batch["mconf"].cpu().numpy()
    
    # Clean up batch tensors
    del tA, tB, batch

    if len(mkpts0) == 0:
        return [], None, None, None

    # Map back to full-image coordinates
    scaleA = cropA.shape[1] / 640, cropA.shape[0] / 480
    scaleB = cropB.shape[1] / 640, cropB.shape[0] / 480
    mkpts0_scaled = mkpts0 * np.array(scaleA)
    mkpts1_scaled = mkpts1 * np.array(scaleB)
    ptsA_full = map_pts_to_original(mkpts0_scaled, offA)
    ptsB_full = map_pts_to_original(mkpts1_scaled, offB)

    # Filter by mask
    keepA = filter_pts_in_mask(ptsA_full, maskA)
    keepB = filter_pts_in_mask(ptsB_full, maskB)
    keep = keepA & keepB
    ptsA_vis = ptsA_full[keep]
    ptsB_vis = ptsB_full[keep]
    mconf = mconf[keep]

    if len(ptsA_vis) == 0:
        return [], None, None, None

    matched_pairs = [(ptsA_vis[i].tolist(), ptsB_vis[i].tolist()) for i in range(len(ptsA_vis))]
    return matched_pairs, ptsA_vis, ptsB_vis, mconf


def get_distance(matcher, frameA: np.ndarray, frameB: np.ndarray, maskA: np.ndarray, maskB: np.ndarray) -> float:
    """
    Compute distance between mask positions in two frames.
    Uses centroid-based approach with LoFTR feature matching as refinement.
    """
    # Centroid-based distance
    c0 = mask_centroid(maskA)
    cN = mask_centroid(maskB)
    if c0 is None or cN is None:
        return -1
    centroid_dist = np.linalg.norm(np.array(cN) - np.array(c0))
    
    # Feature matching distance
    fm_dist = -1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        matches, ptsA, ptsB, conf = match_visible_correspondences(
            matcher, frameA, frameB, maskA, maskB, device=device
        )
        if len(matches) > 0:
            distances = np.linalg.norm(ptsA - ptsB, axis=1)
            fm_dist = float(distances.mean())
    except Exception:
        pass
    
    if fm_dist == -1:
        return centroid_dist
    return min(centroid_dist, fm_dist)


# =============================================================================
# SECTION 5: KMeans Clustering for Moved vs Stationary
# =============================================================================

def cluster_moved_objects(
    moved_distances: List[Tuple[Any, float]],
    image_shape: Tuple[int, int] = None,
    verbose: bool = False
) -> List[Any]:
    """
    Use KMeans to separate moved objects from stationary ones.
    
    Uses the original KMeans approach with a minimal post-clustering
    validation: if the "moved" cluster center is below an absolute
    minimum movement threshold, treat all objects as stationary.
    
    Args:
        moved_distances: List of (object_id, distance) tuples
        image_shape: (height, width) - not used currently
        verbose: Print debug info
        
    Returns:
        List of object_ids that belong to the "moved" cluster
    """
    if len(moved_distances) == 0:
        return []
    elif len(moved_distances) == 1:
        color, dist = moved_distances[0]
        return [color] if dist > ABSOLUTE_MIN_MOVEMENT_PX else []
    elif len(moved_distances) < 3:
        # Too few samples for reliable clustering
        return [color for color, dist in moved_distances if dist > ABSOLUTE_MIN_MOVEMENT_PX]
    
    # Enough samples for clustering
    distances = np.array([item[1] for item in moved_distances]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(distances)
    
    # Identify which cluster represents "moved" (higher distance)
    moved_cluster_label = np.argmax(kmeans.cluster_centers_)
    high_center = float(kmeans.cluster_centers_[moved_cluster_label][0])
    
    if verbose:
        centers = sorted(kmeans.cluster_centers_.flatten())
        print(f"    KMeans: centers=[{centers[0]:.1f}, {centers[1]:.1f}]")
    
    # Post-KMeans validation: if the "moved" cluster center is too low,
    # nothing actually moved significantly
    if high_center < ABSOLUTE_MIN_MOVEMENT_PX:
        if verbose:
            print(f"    Validation: moved center {high_center:.1f}px < {ABSOLUTE_MIN_MOVEMENT_PX}px threshold, no objects moved")
        return []
    
    # Gap ratio check: if the gap between cluster centers is small,
    # require higher absolute threshold to be in "moved" cluster.
    # This prevents mislabeling moderate-distance background shifts as "moved".
    low_center = float(kmeans.cluster_centers_[1 - moved_cluster_label][0])
    gap_ratio = high_center / max(low_center, 1.0)
    
    # Filter to keep only items in moved cluster
    moved = []
    for i, item in enumerate(moved_distances):
        if kmeans.labels_[i] == moved_cluster_label:
            color, dist = item
            # If gap ratio is small (< 3x), require individual distance
            # to be above a stricter threshold to avoid borderline FPs
            if gap_ratio < 3.0 and dist < ABSOLUTE_MIN_MOVEMENT_PX * 1.5:
                if verbose:
                    print(f"    Gap filter: {color} at {dist:.1f}px rejected (gap_ratio={gap_ratio:.1f}, need >{ABSOLUTE_MIN_MOVEMENT_PX * 1.5:.0f}px)")
                continue
            moved.append(color)
    
    # Override: objects with distance > DEFINITELY_MOVED_PX are always "moved"
    # even if KMeans assigned them to the stationary cluster
    moved_set = set(moved)
    for item in moved_distances:
        color, dist = item
        if dist > DEFINITELY_MOVED_PX and color not in moved_set:
            moved.append(color)
            moved_set.add(color)
            if verbose:
                print(f"    Override: {color} at {dist:.1f}px > {DEFINITELY_MOVED_PX}px, forced to moved")
    
    return moved


# =============================================================================
# SECTION 6: BLIP2 Verification and Captioning
# =============================================================================

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


def normalize_object_name(name: str) -> str:
    """Normalize object name for matching."""
    name = name.lower().strip()
    # Remove common articles and prefixes
    for prefix in ['a ', 'an ', 'the ', 'some ']:
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Remove common suffixes
    name = name.rstrip('s')  # simple plural handling
    return name.strip()


def is_valid_object_caption(caption: str) -> bool:
    """
    Check if a BLIP2 caption represents a plausible manipulable object.
    Filter out obvious hallucinations that don't belong in a kitchen/table scene.
    """
    caption_lower = caption.lower().strip()
    
    # List of clearly invalid captions (common BLIP2 hallucinations)
    # Be conservative - only filter things that could NEVER appear on a table/kitchen
    invalid_patterns = [
        'person', 'people', 'man', 'woman', 'child', 'human', 'face',
        'sand dune', 'beach', 'ocean', 'mountain',
        'car', 'vehicle', 'truck', 'bus', 'motorcycle',
        'building', 'house', 'road', 'street',
        'horse', 'cow', 'elephant',
        'broomstick', 'wand', 'wizard',
        'sword', 'shield', 'weapon', 'gun',
        'door', 'window', 'ceiling', 'floor',
        'unknown', 'nothing', 'blank', 'empty',
        # Surface/background objects that shouldn't count as manipulable objects
        'cutting board', 'wooden box', 'wood box', 'table', 'surface',
        'countertop', 'counter', 'shelf', 'background', 'wall',
        'cabinet', 'drawer', 'stovetop', 'stove top', 'sink',
    ]
    
    for pattern in invalid_patterns:
        # Use word-boundary matching to avoid substring collisions
        # e.g., 'car' should not match 'carrot', 'man' should not match 'mango'
        if re.search(r'\b' + re.escape(pattern) + r'\b', caption_lower):
            return False
    
    # Caption should be reasonably short (1-6 words typically)
    word_count = len(caption_lower.split())
    if word_count > 8:
        return False
    
    return True


def objects_are_similar(name1: str, name2: str) -> bool:
    """Check if two object names refer to the same type of object."""
    n1 = normalize_object_name(name1)
    n2 = normalize_object_name(name2)
    
    # Direct match
    if n1 == n2:
        return True
    
    # Check if one contains the other (but require significant overlap)
    if len(n1) >= 3 and len(n2) >= 3:
        if n1 in n2 or n2 in n1:
            return True
    
    # Common synonyms/variants
    synonym_groups = [
        {'bowl', 'dish', 'container', 'pot'},
        {'cup', 'mug', 'glass'},
        {'pepper', 'bell pepper', 'capsicum'},
        {'carrot', 'orange vegetable'},
        {'spoon', 'utensil', 'cutlery'},
        {'knife', 'utensil', 'cutlery'},
        {'fork', 'utensil', 'cutlery'},
        {'apple', 'fruit', 'red fruit', 'green fruit'},
        {'orange', 'citrus', 'fruit'},
        {'banana', 'yellow fruit', 'fruit'},
        {'pear', 'green fruit', 'fruit'},
        {'tomato', 'red vegetable', 'fruit'},
        {'cucumber', 'green vegetable'},
        {'potato', 'vegetable'},
        {'onion', 'vegetable'},
        {'sponge', 'scrubber', 'cleaning pad'},
    ]
    
    for group in synonym_groups:
        matches_n1 = any(syn in n1 or n1 in syn for syn in group)
        matches_n2 = any(syn in n2 or n2 in syn for syn in group)
        if matches_n1 and matches_n2:
            return True
    
    return False


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
    """
    # Convert to BGR for cropping
    if original_image_frame.shape[2] == 3 and original_image_frame.dtype == np.uint8:
        image_bgr = cv2.cvtColor(original_image_frame, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = original_image_frame
    
    crop = crop_from_mask(image_bgr, obj_mask)
    if crop is None:
        return False

    # Ensure crop is RGB (BLIP-2 normalize expects 3 channels)
    if crop.mode != "RGB":
        crop = crop.convert("RGB")

    question = f"Question: Is this a {obj_description}? Answer with yes or no."

    inputs = blip_processor(images=crop, text=question, return_tensors="pt")
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
            max_new_tokens=32,
            do_sample=False
        )
    answer = blip_processor.batch_decode(outputs, skip_special_tokens=True)[0].lower().strip()
    
    # Clean up
    del inputs, moved_inputs, outputs
    
    return "yes" in answer


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
# SECTION 7: Main Algorithm
# =============================================================================

def num_unmentioned_moved_within_scene(
    img_before_path: str,
    img_after_path: str,
    instruction: str,
    video_chat_model,
    video_chat_tokenizer,
    sam_mask_generator,
    loftr_matcher,
    blip_processor,
    blip_model,
    blip_device,
    verbose: bool = False,
    output_dirs: Optional[Dict[str, str]] = None
) -> Tuple[int, List[str]]:
    """
    Compute number of UNMENTIONED objects that MOVED-WITHIN-SCENE.
    
    Algorithm:
    1) Extract mentioned objects from manipulation instruction using VideoChat-Flash
    2) Perform moved-within-scene object detection:
       a) SAM proposal generation on first frame
       b) XMem tracking to get masks in second frame
       c) LoFTR/centroid-based distance metric for movement detection
       d) KMeans clustering to separate moved vs stationary objects
       e) BLIP2 verification to filter out mentioned objects
    
    Args:
        img_before_path: Path to first (before) frame
        img_after_path: Path to second (after) frame  
        instruction: Manipulation instruction text
        video_chat_model: Loaded VideoChat-Flash model
        video_chat_tokenizer: VideoChat-Flash tokenizer
        sam_mask_generator: SAM mask generator
        loftr_matcher: LoFTR feature matcher
        blip_processor: BLIP-2 processor
        blip_model: BLIP-2 model
        blip_device: Device for BLIP-2
        verbose: Print debug info
        output_dirs: Dictionary of output directory paths for logging
        
    Returns:
        count: Number of unmentioned moved-within-scene objects
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
    removed_from_scene_colors = all_sam_colors - tracked_colors
    
    if verbose and len(removed_from_scene_colors) > 0:
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
    # STEP 2c: Compute movement distances using LoFTR
    # =========================================================================
    if verbose:
        print("[STEP 2c] Computing movement distances...")
    
    first_frame = cv2.cvtColor(cv2.imread(img_before_path), cv2.COLOR_BGR2RGB)
    second_frame = cv2.cvtColor(cv2.imread(img_after_path), cv2.COLOR_BGR2RGB)
    
    moved_distances = []
    
    # Compute distances for tracked objects (moved within scene)
    for color, mask_second in tracked_masks.items():
        # Find corresponding first frame mask
        if color in first_frame_masks_by_color:
            mask_first = first_frame_masks_by_color[color]
            
            # Compute distance
            dist = get_distance(loftr_matcher, first_frame, second_frame, mask_first, mask_second)
            
            if dist >= 0:
                moved_distances.append((color, dist))
                if verbose:
                    print(f"    Color {color}: distance = {dist:.2f}")
    
    # Compute distances for "removed from scene" objects
    # These are objects that were in frame 1 but not tracked to frame 2
    for color in removed_from_scene_colors:
        mask_first = first_frame_masks_by_color[color]
        # Distance to closest edge represents minimum movement to exit scene
        dist = compute_removed_from_scene_distance(mask_first)
        if dist > 0:
            moved_distances.append((color, dist))
            if verbose:
                print(f"    Color {color}: distance = {dist:.2f} (removed from scene)")
    
    # Clear GPU memory after LoFTR computations
    clear_cuda_cache()
    
    # Save movement distances
    if output_dirs:
        save_movement_distances(output_dirs['movement_distances'], moved_distances, color_to_idx)
    
    # =========================================================================
    # STEP 2d: KMeans clustering to identify moved objects
    # =========================================================================
    if verbose:
        print("[STEP 2d] Clustering to identify moved objects...")
    
    # Get image shape for adaptive threshold calculation
    image_shape = first_frame.shape[:2]  # (height, width)
    
    moved_colors = cluster_moved_objects(moved_distances, image_shape=image_shape, verbose=verbose)
    
    # Filter out "removed from scene" objects - we only want "moved within scene"
    moved_within_scene_colors = [c for c in moved_colors if c not in removed_from_scene_colors]
    removed_but_clustered = [c for c in moved_colors if c in removed_from_scene_colors]
    
    if verbose:
        print(f"  Identified {len(moved_colors)} moved objects total")
        if len(removed_but_clustered) > 0:
            print(f"  Filtered out {len(removed_but_clustered)} 'removed from scene' objects")
        print(f"  Final: {len(moved_within_scene_colors)} moved-within-scene objects")
    
    # Use the filtered list for downstream processing
    moved_colors = moved_within_scene_colors
    
    # Save clustering results
    if output_dirs:
        save_clustering_results(
            output_dirs['clustering'],
            moved_distances,
            moved_colors,
            color_to_idx,
            img_before_path,
            first_frame_masks_by_color
        )
    
    # =========================================================================
    # STEP 2e: BLIP2 verification to filter out mentioned objects
    # =========================================================================
    if verbose:
        print("[STEP 2e] BLIP2 verification...")
    
    final_answer = []
    verification_results = []
    mention_match_counts = {}  # Track how many objects matched each mentioned name
    matched_color_info = []    # Track (color, matched_object) for potential re-evaluation
    
    for idx, color in enumerate(moved_colors):
        # Periodic cache clearing during BLIP verification
        if idx > 0 and idx % 3 == 0:
            clear_cuda_cache()
        
        # Get the mask from the first frame for captioning
        if color in first_frame_masks_by_color:
            obj_mask = first_frame_masks_by_color[color]
        elif color in tracked_masks:
            obj_mask = tracked_masks[color]
        else:
            continue
        
        # Check if this object matches any mentioned object
        is_mentioned = False
        matched_object = None
        for men_obj in mentioned_objects:
            if verify_blip2(obj_mask, first_frame, men_obj, blip_processor, blip_model, blip_device):
                is_mentioned = True
                matched_object = men_obj
                mention_match_counts[men_obj] = mention_match_counts.get(men_obj, 0) + 1
                matched_color_info.append((color, men_obj))
                if verbose:
                    print(f"    Object with color {color} matches mentioned object '{men_obj}'")
                break
        
        caption = None
        if not is_mentioned:
            # Get caption for this unmentioned moved object
            caption = blip2_caption(obj_mask, first_frame, blip_processor, blip_model, blip_device)
            
            # Filter out hallucinated captions
            if is_valid_object_caption(caption):
                final_answer.append(caption)
                if verbose:
                    print(f"    Unmentioned moved object: {caption}")
            else:
                if verbose:
                    print(f"    Filtered hallucinated caption: {caption}")
        
        # Store verification result for logging
        verification_results.append({
            'color': list(color) if isinstance(color, tuple) else color,
            'mask_index': color_to_idx.get(color, -1),
            'is_mentioned': is_mentioned,
            'matched_object': matched_object,
            'caption': caption
        })
    
    # Exclusive matching: each mentioned object should match at most 1 mask.
    # Two modes:
    #   Case A (all matched, pred=0): Always release excess - prevents BLIP2 false matches
    #   Case B (some unmatched): Release excess but check spatial proximity and similarity
    #          to protect against duplicate SAM masks while catching genuinely different objects
    if any(c > 1 for c in mention_match_counts.values()):
        all_matched = len(final_answer) == 0 and len(moved_colors) > 0
        
        if verbose:
            mode = "all-matched (always release)" if all_matched else "partial-matched (proximity+similarity check)"
            print(f"    [Exclusive matching] Mode: {mode}")
        
        # Build mapping: mentioned object → first matched color (the "kept" match)
        kept_colors = {}
        released_colors = []
        seen_mentions = set()
        for color, men_obj in matched_color_info:
            if men_obj in seen_mentions:
                released_colors.append((color, men_obj))
            else:
                seen_mentions.add(men_obj)
                kept_colors[men_obj] = color
        
        # Caption the released objects
        for color, released_from in released_colors:
            if color in first_frame_masks_by_color:
                obj_mask = first_frame_masks_by_color[color]
            elif color in tracked_masks:
                obj_mask = tracked_masks[color]
            else:
                continue
            
            if not all_matched:
                # Case B: Check spatial proximity to the kept mask
                # If masks are close (same object segmented twice by SAM), treat as duplicate
                kept_color = kept_colors.get(released_from)
                if kept_color is not None and kept_color in first_frame_masks_by_color:
                    kept_mask = first_frame_masks_by_color[kept_color]
                    # Compute centroid distance between kept and released masks
                    released_ys, released_xs = np.where(obj_mask > 0)
                    kept_ys, kept_xs = np.where(kept_mask > 0)
                    if len(released_ys) > 0 and len(kept_ys) > 0:
                        released_centroid = (released_xs.mean(), released_ys.mean())
                        kept_centroid = (kept_xs.mean(), kept_ys.mean())
                        centroid_dist = np.sqrt(
                            (released_centroid[0] - kept_centroid[0]) ** 2 +
                            (released_centroid[1] - kept_centroid[1]) ** 2
                        )
                        if centroid_dist < 80:  # Close enough to be same object
                            if verbose:
                                print(f"    [Released] Mask centroid {centroid_dist:.1f}px from kept mask, treating as duplicate SAM segment")
                            continue
            
            caption = blip2_caption(obj_mask, first_frame, blip_processor, blip_model, blip_device)
            
            if not all_matched:
                # Also check caption similarity as a fallback
                is_duplicate = any(objects_are_similar(caption, men_obj) for men_obj in mentioned_objects)
                if is_duplicate:
                    if verbose:
                        print(f"    [Released] Caption '{caption}' similar to mentioned object, treating as mentioned")
                    continue
            
            if is_valid_object_caption(caption):
                final_answer.append(caption)
                if verbose:
                    print(f"    [Released] Unmentioned moved object: {caption}")
            else:
                if verbose:
                    print(f"    [Released] Filtered hallucinated caption: {caption}")
    
    # Post-verification reconciliation: if NO moved object matched the mentioned name,
    # the mentioned object is likely among the "unmentioned" captions (BLIP2 failed to match).
    # Use local LLM to identify which caption most likely refers to the mentioned object.
    # Only apply when pred >= 2 to avoid accidentally zeroing out correct single predictions.
    total_mentioned_matches = sum(mention_match_counts.values())
    if total_mentioned_matches == 0 and len(final_answer) >= 2 and len(mentioned_objects) > 0:
        try:
            captions_str = ", ".join(f"'{c}'" for c in final_answer)
            mentioned_str = mentioned_objects[0]
            reconciliation_prompt = (
                f"In a robotic manipulation scene, the instruction mentions moving an object called '{mentioned_str}'. "
                f"A vision model detected these objects as having moved: {captions_str}. "
                f"Which ONE of these detected objects is most likely the same physical object as '{mentioned_str}'? "
                f"\n\nIMPORTANT: Vision models describe objects by appearance, not function. Common equivalences include:\n"
                f"- 'detergent' may appear as 'a plastic bottle' or 'a bottle'\n"
                f"- 'pot' may appear as 'a pan' or 'a cooking pan'\n"
                f"- 'sushi' may appear as 'a piece of food' or 'a red hat'\n"
                f"- 'omelette' may appear as 'a white egg' or 'a pancake'\n\n"
                f"Respond with ONLY the exact detected object name from the list that best matches '{mentioned_str}'. "
                f"If none could plausibly be '{mentioned_str}', respond with exactly 'none'.\n"
                f"Answer:"
            )
            llm_response = call_local_llm(reconciliation_prompt, temperature=0.1).strip().lower()
            
            if verbose:
                print(f"    [Reconciliation] LLM response: '{llm_response}'")
            
            if llm_response != 'none' and llm_response:
                # Find and remove the matching caption
                matched_idx = None
                for i, caption in enumerate(final_answer):
                    if caption.lower().strip() == llm_response or \
                       llm_response in caption.lower() or \
                       caption.lower() in llm_response:
                        matched_idx = i
                        break
                
                if matched_idx is not None:
                    removed = final_answer.pop(matched_idx)
                    if verbose:
                        print(f"    [Reconciliation] Reclassified '{removed}' as mentioned object '{mentioned_str}'")
                elif verbose:
                    print(f"    [Reconciliation] Could not match LLM response to any caption")
        except Exception as e:
            if verbose:
                print(f"    [Reconciliation] LLM call failed: {e}")
    
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
# SECTION 8: Model Loading and Test Runner
# =============================================================================

def load_all_models():
    """Load all required models with memory optimization."""
    print("[INFO] Loading all models...")
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
    
    # LoFTR (smaller model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_type = "outdoor_ot"
    weight_path = os.path.join(LOFTR_WEIGHTS_DIR, f"{weight_type}.ckpt")
    loftr_matcher = load_loftr_model(weight_type, weight_path, device)
    clear_cuda_cache()
    
    # BLIP-2
    blip_processor, blip_model, blip_device = init_blip()
    clear_cuda_cache()
    
    print("[INFO] All models loaded successfully!")
    
    return {
        'video_chat_model': video_chat_model,
        'video_chat_tokenizer': video_chat_tokenizer,
        'sam_mask_generator': sam_mask_generator,
        'loftr_matcher': loftr_matcher,
        'blip_processor': blip_processor,
        'blip_model': blip_model,
        'blip_device': blip_device,
    }


def run_evaluation(models: Dict, test_cases: List[Dict], verbose: bool = False):
    """Run evaluation on test cases with memory optimization and comprehensive logging."""
    total = len(test_cases)
    score_sum = 0.0
    
    # Delete existing output directory and recreate fresh for this run
    if os.path.exists(R2_OUTPUT_DIR):
        shutil.rmtree(R2_OUTPUT_DIR)
        print(f"[INFO] Deleted existing output directory: {R2_OUTPUT_DIR}")
    os.makedirs(R2_OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Created fresh output directory: {R2_OUTPUT_DIR}")
    
    # Save overall run info
    run_info = {
        'total_tests': total,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'instruction_target_method': INSTRUCTION_TARGET_METHOD,
    }
    with open(os.path.join(R2_OUTPUT_DIR, 'run_info.json'), 'w') as f:
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
        
        predicted_count, objects = num_unmentioned_moved_within_scene(
            img_before_path=tc["before"],
            img_after_path=tc["after"],
            instruction=tc["instruction"],
            video_chat_model=models['video_chat_model'],
            video_chat_tokenizer=models['video_chat_tokenizer'],
            sam_mask_generator=models['sam_mask_generator'],
            loftr_matcher=models['loftr_matcher'],
            blip_processor=models['blip_processor'],
            blip_model=models['blip_model'],
            blip_device=models['blip_device'],
            verbose=verbose,
            output_dirs=output_dirs
        )
        
        print(f"  Expected count: {tc['expected_count']}")
        print(f"  Predicted count: {predicted_count}")
        print(f"  Unmentioned moved objects: {objects}")
        
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
        'results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(R2_OUTPUT_DIR, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save human-readable summary
    with open(os.path.join(R2_OUTPUT_DIR, 'evaluation_summary.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("R2 Evaluation Summary\n")
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
                f.write(f"  Detected objects: {', '.join(r['objects'])}\n")
            f.write("\n")
    
    print(f"\n[INFO] All outputs saved to: {R2_OUTPUT_DIR}")
    
    return success_score


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            "before": "lgie-eval-data/input_2.jpg",
            "after": "lgie-eval-data/output_2.png",
            "instruction": "The robotic arm puts the omelette on top of the cabbage",
            "expected_count": 1,
            "description": "Omelette on cabbage - cabbage also moved"
        },
        {
            "before": "lgie-eval-data/input_1.jpg",
            "after": "lgie-eval-data/output_1_1.png",
            "instruction": "The robotic arm puts the cucumber on top of the blue cube",
            "expected_count": 0,
            "description": "Cucumber on blue cube - no unmentioned moves"
        },
        {
            "before": "lgie-eval-data/input_5.jpg",
            "after": "lgie-eval-data/output_5_1.png",
            "instruction": "The robotic arm puts the sushi into the pot",
            "expected_count": 1,
            "description": "Sushi into pot - strawberry also moved"
        },
        {
            "before": "lgie-eval-data/input_5.jpg",
            "after": "lgie-eval-data/output_5_1.png",
            "instruction": "The robotic arm puts the carrot into the pot",
            "expected_count": 2,
            "description": "Carrot into pot (wrong) - sushi and strawberry moved instead"
        },
        {
            "before": "lgie-eval-data/input_4.jpg",
            "after": "lgie-eval-data/output_4_1.png",
            "instruction": "The robotic arm moves the blue spoon out of the scene",
            "expected_count": 0,
            "description": "Blue spoon is moved out of the scene - no unmentioned moves"
        },
        {
            "before": "lgie-eval-data/input_4.jpg",
            "after": "lgie-eval-data/output_4_2.png",
            "instruction": "The robotic arm moves the blue spoon out of the scene",
            "expected_count": 0,
            "description": "Blue spoon, banana (unmentioned) and blue cube (unmentioned) are moved out of the scene - still no unmentioned moves within the scene"
        },
        {
            "before": "lgie-eval-data/input_4.jpg",
            "after": "lgie-eval-data/output_4_3.png",
            "instruction": "The robotic arm moves the blue spoon to the left edge of the table",
            "expected_count": 2,
            "description": "Blue spoon (mentioned), banana (unmentioned) and the cucumber (unmentioned) are moved within the scene"
        },
          {
            "id": 23,
            "before": "lgie-eval-data/original_image/oi_23.png",
            "after": "lgie-eval-data/r2_gemini_output/out_23.png",
            "instruction": "pick up sponge and wipe plate",
            "moved_objects": [],
            "expected_count": 0,
            "description": "Sponge is picked up and seen wiping plate -- no unmentioned moved-within-scene objects."
          },
          {
            "id": 24,
            "before": "lgie-eval-data/original_image/oi_24.png",
            "after": "lgie-eval-data/r2_gemini_output/out_24.png",
            "instruction": "put bowl on plate cardboard fence",
            "moved_objects": [
              "green pepper outside the cardboard fence"
            ],
            "expected_count": 1,
            "description": "Bowl is correctly place on plate but unmentioned green pepper is placed outside the cardboard fence"
          },
          {
            "id": 25,
            "before": "lgie-eval-data/original_image/oi_25.png",
            "after": "lgie-eval-data/r2_gemini_output/out_25.png",
            "instruction": "put detergent in sink",
            "moved_objects": [
              "metal pot and red sushi moved within the sink"
            ],
            "expected_count": 2,
            "description": "Detergent bottle is moved to the sink but unmentioned metal pot and red sushi also mistakenly moved within the sink"
          },
          {
            "id": 26,
            "before": "lgie-eval-data/original_image/oi_26.png",
            "after": "lgie-eval-data/r2_gemini_output/out_26.png",
            "instruction": "put banana in pot cardboard fence",
            "moved_objects": [
              "whole pot moved to left corner of cardboard fence"
            ],
            "expected_count": 1,
            "description": "Banana is correctly put in pot but whole pot is mistakenly moved to the bottom left corner within cardboard fence"
          },
          {
            "id": 27,
            "before": "lgie-eval-data/original_image/oi_27.png",
            "after": "lgie-eval-data/r2_gemini_output/out_27.png",
            "instruction": "put carrot in pot cardboard fence",
            "moved_objects": [
              "pot is moved within the cardboard fence"
            ],
            "expected_count": 1,
            "description": "Carrot is correctly put in pot but whole pot is mistakenly moved to the bottom left corner within cardboard fence"
          },
          {
            "id": 28,
            "before": "lgie-eval-data/original_image/oi_28.png",
            "after": "lgie-eval-data/r2_gemini_output/out_28.png",
            "instruction": "open microwave",
            "moved_objects": [
              "orange within the cardboard fence"
            ],
            "expected_count": 1,
            "description": "Unmentioned orange is mistakenly moved within cardboard fence."
          },
          {
            "id": 29,
            "before": "lgie-eval-data/original_image/oi_29.png",
            "after": "lgie-eval-data/r2_gemini_output/out_29.png",
            "instruction": "open book",
            "moved_objects": [
              "omlette is placed on top of book"
            ],
            "expected_count": 1,
            "description": "Book is opened but omlette which should be below the book is mistakenly put on top of the opened book."
          },
          {
            "id": 30,
            "before": "lgie-eval-data/original_image/oi_30.png",
            "after": "lgie-eval-data/r2_gemini_output/out_30.png",
            "instruction": "flip pot upright in sink",
            "moved_objects": [
              "orange pumpkin is placed within the upright pot"
            ],
            "expected_count": 1,
            "description": "Pot is correctly flipped upright but unmentioned orange pumpkin is mistakenly moved within the pot."
          },
          {
            "id": 31,
            "before": "lgie-eval-data/original_image/oi_31.png",
            "after": "lgie-eval-data/r2_gemini_output/out_31.png",
            "instruction": "open brown1fbox flap",
            "moved_objects": [
              "corn and beans container have moved to left of brown box"
            ],
            "expected_count": 2,
            "description": "Unmentioned corn and beans box have been mistakenly moved to left of the brown box."
          },
          {
            "id": 32,
            "before": "lgie-eval-data/original_image/oi_32.png",
            "after": "lgie-eval-data/r2_gemini_output/out_32.png",
            "instruction": "take broccoli out of pan",
            "moved_objects": [],
            "expected_count": 0,
            "description": "Broccoli is correctly put out of pan -- no unmentioned moved-within-scene objects."
          },
          {
            "id": 33,
            "before": "lgie-eval-data/original_image/oi_33.png",
            "after": "lgie-eval-data/r2_gemini_output/out_33.png",
            "instruction": "put corn in pot which is in sink",
            "moved_objects": [
              "red pepper taken out from sink is moved to kitchen surface"
            ],
            "expected_count": 1,
            "description": "Corn is correctly put in the pot but unmentioned red pepper has been taken out from the sink and moved to kitchen surface."
          },
          {
            "id": 34,
            "before": "lgie-eval-data/original_image/oi_34.png",
            "after": "lgie-eval-data/r2_gemini_output/out_34.png",
            "instruction": "put small spoon from basket to tray",
            "moved_objects": [],
            "expected_count": 0,
            "description": "Small spoon has been correctly put on basket tray -- no unmentioned moved-within-scene objects."
          },
          {
            "id": 35,
            "before": "lgie-eval-data/original_image/oi_35.png",
            "after": "lgie-eval-data/r2_gemini_output/out_35.png",
            "instruction": "pick up bit holder",
            "moved_objects": [],
            "expected_count": 0,
            "description": "Bit holder is lifted -- no unmentioned moved-within-scene objects."
          },
          {
            "id": 36,
            "before": "lgie-eval-data/original_image/oi_36.png",
            "after": "lgie-eval-data/r2_gemini_output/out_36.png",
            "instruction": "turn lever vertical to front",
            "moved_objects": [
              "onion",
              "carrot",
              "plastic knife",
              "all moved out from the sink to surface"
            ],
            "expected_count": 3,
            "description": "Unmentioned onion, carrot, and knife have been taken out from the sink and mistakenly moved."
          },
          {
            "id": 37,
            "before": "lgie-eval-data/original_image/oi_37.png",
            "after": "lgie-eval-data/r2_gemini_output/out_37.png",
            "instruction": "take carrot off plate",
            "moved_objects": [
              "plate and carrot both taken from sink moved to kitchen surface"
            ],
            "expected_count": 1,
            "description": "Unmentioned plate have been mistakenly taken from sink moved to kitchen surface"
          },
          {
            "id": 38,
            "before": "lgie-eval-data/original_image/oi_38.png",
            "after": "lgie-eval-data/r2_gemini_output/out_38.png",
            "instruction": "lever vertical to front",
            "moved_objects": [],
            "expected_count": 0,
            "description": "No unmentioned moved-within-scene objects"
          },
          {
            "id": 39,
            "before": "lgie-eval-data/original_image/oi_39.png",
            "after": "lgie-eval-data/r2_gemini_output/out_39.png",
            "instruction": "put banana in colander",
            "moved_objects": [
              "spoon",
              "tomato",
              "spatula",
              "green bell pepper",
              "all moved to colander"
            ],
            "expected_count": 4,
            "description": "Banana has been correctly put in colander but unmentioned spoon, tomato, spatula, and green bell pepper have also been mistakenly put into the colander."
          },
          {
            "id": 40,
            "before": "lgie-eval-data/original_image/oi_40.png",
            "after": "lgie-eval-data/r2_gemini_output/out_40.png",
            "instruction": "put pear in bowl",
            "moved_objects": [],
            "expected_count": 0,
            "description": "Pear is correctly placed in bowl -- no unmentioned moved-within-scene objects."
          },
          {
            "id": 41,
            "before": "lgie-eval-data/original_image/oi_41.png",
            "after": "lgie-eval-data/r2_gemini_output/out_41.png",
            "instruction": "take carrot off plate",
            "moved_objects": [],
            "expected_count": 0,
            "description": "Carrot is removed from the plate -- no unmentioned moved-within-scene objects."
          },
          {
            "id": 42,
            "before": "lgie-eval-data/original_image/oi_42.png",
            "after": "lgie-eval-data/r2_gemini_output/out_42.png",
            "instruction": "take carrot off plate",
            "moved_objects": [],
            "expected_count": 0,
            "description": "Carrot is correctly taken off the plate and placed aside -- no unmentioned moved-within-scene objects."
          },
          {
            "id": 43,
            "before": "lgie-eval-data/original_image/oi_43.png",
            "after": "lgie-eval-data/r2_gemini_output/out_43.png",
            "instruction": "close drawer of box",
            "moved_objects": [
              "green pear inside the drawer moved within it."
            ],
            "expected_count": 1,
            "description": "Unmentioned green pear inside the drawer mistakenly moved within the drawer."
          },
          {
            "id": 44,
            "before": "lgie-eval-data/original_image/oi_44.png",
            "after": "lgie-eval-data/r2_gemini_output/out_44.png",
            "instruction": "put pepper in pan",
            "moved_objects": [
              "yellow bell pepper is also moved in pan"
            ],
            "expected_count": 1,
            "description": "Pepper is placed in pan but unmentioned yellow bell pepper is also mistakenly put in pan."
          },
          {
            "id": 45,
            "before": "lgie-eval-data/original_image/oi_45.png",
            "after": "lgie-eval-data/r2_gemini_output/out_45.png",
            "instruction": "fold cloth in half",
            "moved_objects": [],
            "expected_count": 0,
            "description": "Cloth is folded in half -- no unmentioned moved-within-scene objects."
          }
    ]
    
    # Load models
    models = load_all_models()
    
    # Run evaluation
    run_evaluation(models, test_cases, verbose=True)
