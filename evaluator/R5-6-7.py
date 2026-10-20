import os
import gc
import re
import atexit
import tempfile
import traceback
from typing import Tuple, Dict, List
from os import path
from contextlib import contextmanager

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Install dependencies (uncomment on first run)
# os.system("pip install transformers==4.40.1 av imageio decord opencv-python timm natsort segment-anything")
# os.system("pip install flash-attn --no-build-isolation")

"""
R5,R6,R7 - Object Change Detection (color, texture, shape)
Object-Centric Approach using SAM + XMem + VideoChat-Flash

Pipeline:
1. SAM: Detect all objects in Frame 1 (before image) with filtering
2. XMem: Track objects to get masks in Frame 2 (after image)
3. Filter: Keep only objects visible in both frames (significant area in both)
4. For each visible object: Create isolated 2-frame video (object crop with black background)
5. VideoChat-Flash: Detect color/texture/shape changes for each object

Memory Optimization:
- Models are loaded one at a time
- Each model is unloaded before the next one is loaded
- Aggressive CUDA cache clearing between model swaps
"""

# -------------------------
# Config
# -------------------------
MODEL_NAME = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"
_SCRIPT_DIR = path.dirname(path.abspath(__file__))
SAM_CHECKPOINT = path.join(_SCRIPT_DIR, "sam_vit_h_4b8939.pth")
XMEM_MODEL_PATH = path.join(_SCRIPT_DIR, "XMem", "saves", "XMem.pth")
XMEM_DIR = path.join(_SCRIPT_DIR, "XMem")
MAX_NUM_FRAMES = 32
GEN_CONFIG = dict(do_sample=False, temperature=0.0, max_new_tokens=256, top_p=0.1, num_beams=1)

# -------------------------
# GPU Memory Optimization Utilities
# -------------------------
def clear_cuda_cache():
    """Aggressively clear CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

@contextmanager
def gpu_memory_manager():
    """Context manager for GPU memory-intensive operations."""
    try:
        yield
    finally:
        clear_cuda_cache()

def unload_model(model, name="model"):
    """Completely unload a model from GPU/memory."""
    if model is None:
        return
    try:
        if hasattr(model, 'to'):
            model.to('cpu')
        if hasattr(model, 'cpu'):
            model.cpu()
        del model
    except Exception as e:
        print(f"[WARN] Error unloading {name}: {e}")
    clear_cuda_cache()

# Color palette for visualization
COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128)
]

# =========================================================================
# SAM Mask Filtering Functions (from your codebase)
# =========================================================================
def get_iou_between_masks(mask1, mask2):
    """Calculate IoU between two binary masks."""
    intersection = np.sum(mask1 * mask2)
    union = np.sum(np.maximum(mask1, mask2))
    return intersection / union if union > 0 else 0

def is_submask(mask1, mask2, iou_thresh):
    """Returns True if mask1 is a submask of mask2."""
    if get_iou_between_masks(mask1, mask1 * mask2) >= iou_thresh:
        return True
    return False

def dfs(node, master_parent, visited, adj, is_master_parent):
    """DFS helper for submask detection."""
    visited[node] = True
    for i in range(len(adj[node])):
        if visited[adj[node][i]]:
            if adj[node][i] != master_parent and is_master_parent[adj[node][i]]:
                is_master_parent[adj[node][i]] = False
        else:
            dfs(adj[node][i], master_parent, visited, adj, is_master_parent)

def discard_submasks(masks, iou_thresh=0.82):
    """Remove masks that are submasks of larger masks."""
    n = len(masks)
    if n == 0:
        return masks
    visited = [False for i in range(n)]
    is_master_parent = {i: False for i in range(n)}
    adj = [[] for i in range(n)]
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

def merge_similar_masks(masks, sim_thresh=0.82):
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
    deleted = [False for i in range(n)]
    if not pairwise_iou:
        return masks
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
                if (i, smaller_node) in pairwise_iou:
                    del pairwise_iou[(i, smaller_node)]
                if (smaller_node, i) in pairwise_iou:
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

def filter_by_area(masks, min_area_ratio=0.001):
    """Filter masks by area - remove only very small masks (noise).
    Large masks are kept as they often represent surfaces where texture changes occur.
    """
    if len(masks) == 0:
        return masks
    h, w = masks[0].shape
    total_area = h * w
    filtered = []
    for mask in masks:
        area_ratio = np.sum(mask) / total_area
        if area_ratio >= min_area_ratio:
            filtered.append(mask)
    return filtered

# =========================================================================
# SAM Initialization and Inference
# =========================================================================
def init_sam_model(sam_checkpoint=SAM_CHECKPOINT, model_type="vit_h"):
    """Initialize SAM model and mask generator."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    clear_cuda_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator, sam

def unload_sam_model(sam_model, mask_generator):
    """Completely unload SAM model from GPU."""
    if mask_generator is not None:
        del mask_generator
    if sam_model is not None:
        sam_model.to('cpu')
        del sam_model
    clear_cuda_cache()
    print("  SAM unloaded from GPU")

def run_sam_inference(image: np.ndarray, mask_generator) -> List[np.ndarray]:
    """Run SAM to get filtered object masks."""
    # Generate SAM proposals
    sam_proposals = mask_generator.generate(image)
    sam_masks = [p["segmentation"].astype(np.uint8) for p in sam_proposals]
    
    # Apply filtering functions (filter_by_area last to remove noise after merging)
    sam_masks = discard_submasks(sam_masks)
    sam_masks = merge_similar_masks(sam_masks)
    sam_masks = filter_by_area(sam_masks)
    
    return sam_masks

# =========================================================================
# XMem Tracking
# =========================================================================
def init_xmem_model(xmem_model_path=XMEM_MODEL_PATH):
    """Initialize XMem model for object tracking."""
    # Add XMem to path if needed
    import sys
    if XMEM_DIR not in sys.path:
        sys.path.insert(0, XMEM_DIR)
    
    from model.network import XMem
    from inference.inference_core import InferenceCore
    
    clear_cuda_cache()
    
    config = {
        'top_k': 30,
        'mem_every': 5,
        'deep_update_every': -1,
        'enable_long_term': True,
        'enable_long_term_count_usage': False,
        'max_mid_term_frames': 10,
        'min_mid_term_frames': 5,
        'max_long_term_elements': 10000,
        'num_prototypes': 128,
    }
    
    network = XMem(config, xmem_model_path).cuda().eval()
    model_weights = torch.load(xmem_model_path)
    network.load_weights(model_weights, init_as_zero_if_needed=True)
    
    return network, config

def unload_xmem_model(xmem_network):
    """Completely unload XMem model from GPU."""
    if xmem_network is not None:
        xmem_network.to('cpu')
        del xmem_network
    clear_cuda_cache()
    print("  XMem unloaded from GPU")

def track_single_mask_with_xmem(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray, 
                                 xmem_network, xmem_config) -> np.ndarray:
    """
    Track a single mask from frame 1 to frame 2 using XMem.
    Returns the tracked mask in frame 2.
    """
    from inference.inference_core import InferenceCore
    
    h, w = mask.shape
    
    # Resize img2 to match img1 if sizes differ
    img2_resized = img2
    if img2.shape[:2] != (h, w):
        img2_resized = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    
    # Create mask with label 1
    labeled_mask = mask.astype(np.uint8)
    
    # Initialize processor for single object
    processor = InferenceCore(xmem_network, config=xmem_config)
    processor.set_all_labels([1])
    
    # Process frame 1 with mask
    rgb1 = torch.from_numpy(img1).permute(2, 0, 1).float().cuda() / 255.0
    msk1 = torch.from_numpy(labeled_mask).long().cuda()
    
    # Convert mask to one-hot format (just 1 channel for single object)
    msk1_onehot = msk1.unsqueeze(0).float()
    
    with torch.amp.autocast('cuda', enabled=True):
        # First frame with mask
        processor.step(rgb1, msk1_onehot, [1])
        
        # Second frame without mask (tracking)
        rgb2 = torch.from_numpy(img2_resized).permute(2, 0, 1).float().cuda() / 255.0
        prob = processor.step(rgb2, None, None, end=True)
    
    # prob shape is [num_objects+1, H, W] where channel 0 is background, channel 1 is object
    # Use argmax to get the label with highest probability
    out_labels = torch.argmax(prob, dim=0).cpu().numpy().astype(np.uint8)
    # Object has label 1
    out_mask = (out_labels == 1).astype(np.uint8)
    
    # Cleanup
    del processor, rgb1, rgb2, msk1, msk1_onehot, prob, out_labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return out_mask


def track_masks_with_xmem(img1: np.ndarray, img2: np.ndarray, masks_frame1: List[np.ndarray], 
                          xmem_network, xmem_config) -> List[np.ndarray]:
    """
    Track masks from frame 1 to frame 2 using XMem.
    Processes one object at a time to avoid OOM.
    Returns masks in frame 2 corresponding to each mask in frame 1.
    """
    if len(masks_frame1) == 0:
        return []
    
    h, w = masks_frame1[0].shape
    
    # Resize img2 to match img1 if sizes differ
    img2_resized = img2
    if img2.shape[:2] != (h, w):
        img2_resized = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    
    masks_frame2 = []
    
    # Track each mask individually to avoid OOM with many objects
    for mask in masks_frame1:
        tracked_mask = track_single_mask_with_xmem(
            img1, img2_resized, mask, xmem_network, xmem_config
        )
        masks_frame2.append(tracked_mask)
    
    return masks_frame2

# Note: XMem is required - no fallback. If XMem fails, we raise an error.

# =========================================================================
# Object Crop and Video Creation
# =========================================================================
def create_object_crop(image: np.ndarray, mask: np.ndarray, padding: int = 50) -> np.ndarray:
    """Create a cropped image of the object with black background.
    
    Crops to the object's bounding box with generous padding to help the model
    focus on the object while still seeing its context and relative size.
    Non-masked pixels are blacked out.
    
    Args:
        image: RGB image array
        mask: Binary mask of the object
        padding: Pixels of padding around bounding box (default 50)
    
    Returns:
        Cropped image with object visible, rest blacked out
    """
    # Find bounding box of mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        # Empty mask - return small black image
        return np.zeros((64, 64, 3), dtype=np.uint8)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Add generous padding
    h, w = mask.shape
    rmin = max(0, rmin - padding)
    rmax = min(h, rmax + padding + 1)
    cmin = max(0, cmin - padding)
    cmax = min(w, cmax + padding + 1)
    
    # Crop image and mask
    crop = image[rmin:rmax, cmin:cmax].copy()
    mask_crop = mask[rmin:rmax, cmin:cmax]
    
    # Apply mask (black background for non-object pixels)
    mask_3ch = np.stack([mask_crop, mask_crop, mask_crop], axis=-1)
    crop = crop * mask_3ch
    
    # Ensure minimum size for video encoding
    min_size = 64
    if crop.shape[0] < min_size or crop.shape[1] < min_size:
        new_h = max(crop.shape[0], min_size)
        new_w = max(crop.shape[1], min_size)
        padded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded
    
    return crop

def create_object_video(crop1: np.ndarray, crop2: np.ndarray, output_path: str, fps: int = 1) -> str:
    """Create a 2-frame video from two object crops (before and after).
    
    Args:
        crop1: RGB image array for frame 1 (before)
        crop2: RGB image array for frame 2 (after)
        output_path: Path to save the video
        fps: Frames per second (default 1)
    """
    # Convert RGB to BGR for OpenCV
    frame1 = cv2.cvtColor(crop1, cv2.COLOR_RGB2BGR)
    frame2 = cv2.cvtColor(crop2, cv2.COLOR_RGB2BGR)
    
    h, w = frame1.shape[:2]
    
    # Resize frame2 to match frame1 if needed
    if frame2.shape[:2] != (h, w):
        frame2 = cv2.resize(frame2, (w, h), interpolation=cv2.INTER_AREA)
    
    # Ensure dimensions are even (required by many video codecs)
    if h % 2 != 0:
        h = h + 1
        frame1 = cv2.resize(frame1, (w, h), interpolation=cv2.INTER_AREA)
        frame2 = cv2.resize(frame2, (w, h), interpolation=cv2.INTER_AREA)
    if w % 2 != 0:
        w = w + 1
        frame1 = cv2.resize(frame1, (w, h), interpolation=cv2.INTER_AREA)
        frame2 = cv2.resize(frame2, (w, h), interpolation=cv2.INTER_AREA)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    out.write(frame1)
    out.write(frame2)
    out.release()
    return output_path

# =========================================================================
# VideoChat-Flash Helpers
# =========================================================================
def load_videochat_model_and_tokenizer(model_name=MODEL_NAME):
    """Load VideoChat-Flash model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.mm_llm_compress = False
    return tokenizer, model

def _safe_chat_call(video_chat_model, verbose: bool = False, **kwargs):
    """Safe wrapper for model chat calls with robust retry logic."""
    call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if "video_path" not in call_kwargs:
        call_kwargs["video_path"] = ""
    
    max_retries = 5
    last_error = None
    for attempt in range(max_retries):
        try:
            with torch.no_grad():
                return video_chat_model.chat(**call_kwargs)
        except Exception as e:
            last_error = e
            print(f"[WARN] chat attempt {attempt+1}/{max_retries} failed: {repr(e)}")
            # Clear CUDA cache between retries to recover from OOM
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt < max_retries - 1:
                import time as _time
                _time.sleep(1)  # brief pause before retry
    raise RuntimeError(f"Model chat failed after {max_retries} retries. Last error: {last_error}")

def _parse_yes_no(text: str) -> str:
    """Extract Yes or No from model output."""
    t = text.lower().strip()
    
    # Direct single-word answers
    if t in ['yes', 'yes.', 'yes!', 'yes,']:
        return "Yes"
    if t in ['no', 'no.', 'no!', 'no,']:
        return "No"
    
    # Check first word
    first_word = t.split()[0] if t.split() else ""
    if first_word in ['yes', 'yes.', 'yes,']:
        return "Yes"
    if first_word in ['no', 'no.', 'no,']:
        return "No"
    
    # Check last word
    last_word = t.replace('.', ' ').replace(',', ' ').split()[-1] if t else ""
    if 'yes' in last_word:
        return "Yes"
    if 'no' in last_word:
        return "No"
    
    # Count occurrences
    yes_count = t.count('yes')
    no_count = t.count(' no') + (1 if t.startswith('no') else 0)
    
    if yes_count > no_count:
        return "Yes"
    if no_count > yes_count:
        return "No"
    
    # Default
    return "No"

# =========================================================================
# Memory Cleanup Utilities
# =========================================================================
def cleanup_model(model, name="model"):
    """Move model to CPU and free GPU memory."""
    try:
        if hasattr(model, "to"):
            model.to("cpu")
        elif hasattr(model, "cpu"):
            model.cpu()
    except Exception:
        print(f"[WARN] Couldn't move {name} to CPU cleanly.")

def cleanup_cuda():
    """Free CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

# =========================================================================
# Logging Utilities
# =========================================================================
def save_mask_visualization(masks: List[np.ndarray], image: np.ndarray, save_path: str):
    """Save visualization of masks overlaid on image."""
    overlay = image.copy()
    for idx, mask in enumerate(masks):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def save_individual_masks(masks: List[np.ndarray], save_dir: str, prefix: str = "mask"):
    """Save individual binary masks as images."""
    os.makedirs(save_dir, exist_ok=True)
    for idx, mask in enumerate(masks):
        mask_img = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_{idx:03d}.png"), mask_img)

def save_object_crops(crops: List[np.ndarray], save_dir: str, prefix: str = "crop"):
    """Save object crops as images."""
    os.makedirs(save_dir, exist_ok=True)
    for idx, crop in enumerate(crops):
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_{idx:03d}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

# =========================================================================
# Full-Scene and Programmatic Analysis Helpers
# =========================================================================

def create_full_scene_video(img_before_rgb, img_after_rgb, output_path):
    """Create a 2-frame video from full scene images (RGB numpy arrays)."""
    frame1 = cv2.cvtColor(img_before_rgb, cv2.COLOR_RGB2BGR)
    frame2 = cv2.cvtColor(img_after_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame1.shape[:2]
    if frame2.shape[:2] != (h, w):
        frame2 = cv2.resize(frame2, (w, h), interpolation=cv2.INTER_AREA)
    if h % 2 != 0:
        h += 1
        frame1 = cv2.resize(frame1, (w, h))
        frame2 = cv2.resize(frame2, (w, h))
    if w % 2 != 0:
        w += 1
        frame1 = cv2.resize(frame1, (w, h))
        frame2 = cv2.resize(frame2, (w, h))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 1, (w, h))
    out.write(frame1)
    out.write(frame2)
    out.release()
    return output_path


def compute_mean_lab(img_rgb, mask):
    """Compute mean LAB color of masked region. Returns (L, a, b) or None."""
    if np.sum(mask) < 10:
        return None
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = lab[mask > 0]
    return np.mean(pixels, axis=0)


def compute_sharpness_laplacian(img_rgb, mask):
    """Compute Laplacian variance (sharpness) of masked region."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    lap_values = laplacian[mask > 0]
    if len(lap_values) < 10:
        return 0.0
    return np.var(lap_values)


def compute_mask_shape_distance(mask1, mask2):
    """Compare shapes using Hu moments. Returns distance (0=identical)."""
    contours1, _ = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours1 or not contours2:
        return 0.0
    c1 = max(contours1, key=cv2.contourArea)
    c2 = max(contours2, key=cv2.contourArea)
    if cv2.contourArea(c1) < 10 or cv2.contourArea(c2) < 10:
        return 0.0
    return cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I2, 0.0)


# =========================================================================
# Comprehensive Object Change Analysis (Stage 3 of pipeline)
# =========================================================================
def analyze_object_changes_vlm(
    img_before: np.ndarray,
    img_after: np.ndarray,
    masks_frame1: List[np.ndarray],
    masks_frame2: List[np.ndarray],
    instruction: str,
    video_chat_model,
    video_chat_tokenizer,
    test_log_dir: str = None,
    verbose: bool = False
) -> Dict[str, str]:
    """
    Comprehensive change detection combining three tracks:
    Track 1: Full-scene VLM analysis (catches global changes)
    Track 2: Programmatic pixel-level analysis (LAB color, sharpness, contours)
    Track 3: Per-object VLM analysis (top N largest objects only, conservative)
    """
    results = {"Color": "No", "Texture": "No", "Shape": "No"}
    temp_files = []
    
    if len(masks_frame1) == 0:
        print("[WARN] No objects to analyze")
        return results
    
    # Logging setup
    if test_log_dir:
        log_path = os.path.join(test_log_dir, "analysis_log.txt")
        log = open(log_path, "w", encoding="utf-8")
        log.write(f"Instruction: {instruction}\n")
        log.write(f"Masks: {len(masks_frame1)}\n\n")
    else:
        log = None
    
    # ================================================================
    # TRACK 1: Programmatic Analysis (Global + Per-Object)
    # ================================================================
    if verbose:
        print(f"  [Track 1] Programmatic analysis ({len(masks_frame1)} masks)...")

    prog_color = False
    prog_texture = False
    # Shape: programmatic disabled (too many FPs from tracking artifacts)

    IMAGE_AREA = img_before.shape[0] * img_before.shape[1]
    PROG_MIN_AREA = 500
    COLOR_DE_PER_OBJ = 24.0        # dE_ab threshold per-object (conservative)
    CHROMA_THRESHOLD = 20.0         # dC threshold for colored<->white transitions
    COLOR_MIN_AREA_FRAC = 0.005    # Object must be > 0.5% of image for color check
    GLOBAL_HUE_THRESHOLD = 3.0     # dE_ab for global hue shift detection (lowered from 4.0 to catch subtle shifts)
    GLOBAL_LIGHTNESS_THRESHOLD = 12.0  # dL threshold for global lightness change (gray↔white)
    SHARPNESS_RATIO_THRESH = 0.45  # Per-object sharpness drop ratio
    GLOBAL_SHARP_RATIO = 0.55      # Global sharpness ratio threshold
    TEXTURE_MIN_OBJECTS = 3        # Need 3+ objects to confirm per-object texture change
    EDGE_MIN_OBJECTS = 5             # Need 5+ objects for edge density change (conservative)
    EDGE_LARGE_OBJ_MIN = 1         # Only 1 large object needed for edge texture on large objects
    EDGE_LARGE_OBJ_AREA = 40000    # Area threshold for "large object" edge tier
    EDGE_DROP_THRESH = 0.40        # Edge density drop ratio per-object
    EDGE_RISE_THRESH = 2.5         # Edge density rise ratio per-object
    SURFACE_MIN_AREA = 80000       # Min area for surface-level logging (~10% of image)

    # --- Global hue shift ---
    lab1_full = cv2.cvtColor(img_before, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab2_full = cv2.cvtColor(img_after, cv2.COLOR_RGB2LAB).astype(np.float32)
    global_da = np.mean(lab2_full[:, :, 1]) - np.mean(lab1_full[:, :, 1])
    global_db = np.mean(lab2_full[:, :, 2]) - np.mean(lab1_full[:, :, 2])
    global_de_ab = np.sqrt(global_da ** 2 + global_db ** 2)

    # Brightness filter: suppress small hue shifts caused by brightness changes
    global_dL = abs(float(np.mean(lab2_full[:, :, 0]) - np.mean(lab1_full[:, :, 0])))
    brightness_artifact = (global_dL > 10.0 and global_de_ab < 8.0)

    if global_de_ab > GLOBAL_HUE_THRESHOLD and not brightness_artifact:
        prog_color = True
        if verbose:
            print(f"    PROG: Global hue shift dE_ab={global_de_ab:.1f} > {GLOBAL_HUE_THRESHOLD}")
    elif brightness_artifact and global_de_ab > GLOBAL_HUE_THRESHOLD:
        if verbose:
            print(f"    PROG: Hue {global_de_ab:.1f} suppressed (brightness artifact, dL={global_dL:.1f})")

    # Lightness-based color detection: catches gray↔white, darkening/brightening
    # Only trigger when hue shift is minimal (pure lightness change, not hue+brightness)
    if global_dL > GLOBAL_LIGHTNESS_THRESHOLD and global_de_ab < 3.0:
        prog_color = True
        if verbose:
            print(f"    PROG: Global lightness change dL={global_dL:.1f} > {GLOBAL_LIGHTNESS_THRESHOLD} (hue stable dE={global_de_ab:.1f})")

    if log:
        log.write(f"[PROG] Global hue dE_ab={global_de_ab:.1f} dL={global_dL:.1f} (brightness_artifact={brightness_artifact})\n")

    # --- Global sharpness ---
    gray1 = cv2.cvtColor(img_before, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img_after, cv2.COLOR_RGB2GRAY)
    g_sharp1 = np.var(cv2.Laplacian(gray1.astype(np.float64), cv2.CV_64F))
    g_sharp2 = np.var(cv2.Laplacian(gray2.astype(np.float64), cv2.CV_64F))
    g_sharp_ratio = g_sharp2 / g_sharp1 if g_sharp1 > 0 else 1.0

    GLOBAL_NOISE_RATIO = 5.0        # Only very high global sharpness increase = noise (avoids FPs)
    if g_sharp_ratio < GLOBAL_SHARP_RATIO:
        prog_texture = True
        if verbose:
            print(f"    PROG: Global blur detected sharpness_ratio={g_sharp_ratio:.2f} < {GLOBAL_SHARP_RATIO}")
    if g_sharp_ratio > GLOBAL_NOISE_RATIO:
        prog_texture = True
        if verbose:
            print(f"    PROG: Global noise detected sharpness_ratio={g_sharp_ratio:.2f} > {GLOBAL_NOISE_RATIO}")
    if log:
        log.write(f"[PROG] Global sharpness ratio={g_sharp_ratio:.2f} (blur_thresh={GLOBAL_SHARP_RATIO}, noise_thresh={GLOBAL_NOISE_RATIO})\n")

    # Pre-compute Canny edge maps for edge density texture comparison
    canny1_map = cv2.Canny(gray1, 50, 150)
    canny2_map = cv2.Canny(gray2, 50, 150)

    # --- Per-object programmatic (color + texture only, NO shape) ---
    prog_color_hits = 0
    prog_texture_hits = 0
    prog_severe_blur_count = 0
    prog_noise_hits = 0
    prog_edge_texture_hits = 0
    prog_edge_large_hits = 0         # Track large-object edge changes separately
    prog_surface_change_count = 0
    SEVERE_BLUR_THRESH = 0.34       # Extreme blur threshold for any single object
    SEVERE_BLUR_MIN_AREA = 2000     # Minimum area for severe blur detection
    PER_OBJ_NOISE_THRESH = 2.5     # If object sharpness increases 2.5x+, noise
    prog_objects_analyzed = 0
    color_min_area = int(IMAGE_AREA * COLOR_MIN_AREA_FRAC)

    for idx, (m1, m2) in enumerate(zip(masks_frame1, masks_frame2)):
        a1 = int(np.sum(m1))
        a2 = int(np.sum(m2))
        if a1 < PROG_MIN_AREA or a2 < PROG_MIN_AREA:
            continue
        prog_objects_analyzed += 1

        # Color: LAB chrominance distance + chroma difference (only for large objects)
        if a1 >= color_min_area and a2 >= color_min_area:
            lab_mean1 = compute_mean_lab(img_before, m1)
            lab_mean2 = compute_mean_lab(img_after, m2)
            if lab_mean1 is not None and lab_mean2 is not None:
                da = float(lab_mean2[1] - lab_mean1[1])
                db = float(lab_mean2[2] - lab_mean1[2])
                de_ab = np.sqrt(da ** 2 + db ** 2)
                # Chroma difference: catches colored<->white/gray transitions
                chroma1 = np.sqrt((float(lab_mean1[1]) - 128)**2 + (float(lab_mean1[2]) - 128)**2)
                chroma2 = np.sqrt((float(lab_mean2[1]) - 128)**2 + (float(lab_mean2[2]) - 128)**2)
                dC = abs(chroma1 - chroma2)
                # Full dE (including lightness) for desaturation detection
                dL_obj = float(lab_mean2[0]) - float(lab_mean1[0])
                dE_full = np.sqrt(dL_obj**2 + da**2 + db**2)
                # Desaturation: object went colored→white/gray (only when scene brightness stable)
                desaturated = (global_dL < 8.0 and dE_full > 35 and
                              chroma1 > 8 and chroma2 < chroma1 * 0.4)
                if de_ab > COLOR_DE_PER_OBJ or dC > CHROMA_THRESHOLD or desaturated:
                    prog_color_hits += 1
                    if verbose:
                        print(f"    PROG: Obj {idx} color dE_ab={de_ab:.1f} dC={dC:.1f} dE_full={dE_full:.1f} (area={a1}){' DESAT' if desaturated else ''}")
                    if log:
                        log.write(f"[PROG] Obj {idx}: Color dE_ab={de_ab:.1f} dC={dC:.1f} dE_full={dE_full:.1f} area={a1}{' DESAT' if desaturated else ''}\\n")

        # Texture: sharpness comparison
        sharp1 = compute_sharpness_laplacian(img_before, m1)
        sharp2 = compute_sharpness_laplacian(img_after, m2)
        if sharp1 > 20:
            s_ratio = sharp2 / sharp1
            if s_ratio < SHARPNESS_RATIO_THRESH:
                prog_texture_hits += 1
                if verbose:
                    print(f"    PROG: Obj {idx} texture sharpness_ratio={s_ratio:.2f} (area={a1})")
                if log:
                    log.write(f"[PROG] Obj {idx}: Sharpness ratio={s_ratio:.2f} area={a1}\n")
                # Track severe blur (extreme sharpness drop on large objects)
                if s_ratio < SEVERE_BLUR_THRESH and a1 >= SEVERE_BLUR_MIN_AREA:
                    prog_severe_blur_count += 1
                    if verbose:
                        print(f"    PROG: Obj {idx} SEVERE BLUR (ratio={s_ratio:.2f}, area={a1})")
            elif s_ratio > PER_OBJ_NOISE_THRESH:
                prog_noise_hits += 1
                if verbose:
                    print(f"    PROG: Obj {idx} noise sharpness_ratio={s_ratio:.2f} (area={a1})")
                if log:
                    log.write(f"[PROG] Obj {idx}: Noise ratio={s_ratio:.2f} area={a1}\n")

        # Edge density comparison (Canny)
        edge_pixels1 = int(np.sum(canny1_map[m1 > 0] > 0))
        edge_pixels2 = int(np.sum(canny2_map[m2 > 0] > 0))
        ed1 = edge_pixels1 / max(a1, 1)
        ed2 = edge_pixels2 / max(a2, 1)
        if ed1 > 0.01:  # Minimum edge content to compare
            edge_ratio = ed2 / ed1
            if edge_ratio < EDGE_DROP_THRESH or edge_ratio > EDGE_RISE_THRESH:
                prog_edge_texture_hits += 1
                # Track large-object edge changes separately
                if a1 >= EDGE_LARGE_OBJ_AREA:
                    prog_edge_large_hits += 1
                if verbose:
                    print(f"    PROG: Obj {idx} edge_density_ratio={edge_ratio:.2f} (area={a1}){' [LARGE]' if a1 >= EDGE_LARGE_OBJ_AREA else ''}")
                if log:
                    log.write(f"[PROG] Obj {idx}: Edge ratio={edge_ratio:.2f} area={a1}\n")
            # Surface-level: large object with moderate edge or sharpness change
            if a1 >= SURFACE_MIN_AREA:
                sharp_change = (sharp1 > 20 and (sharp2 / sharp1 < 0.50 or sharp2 / sharp1 > 2.0))
                edge_change = (edge_ratio < 0.50 or edge_ratio > 2.0)
                if sharp_change or edge_change:
                    prog_surface_change_count += 1
                    if verbose:
                        print(f"    PROG: Obj {idx} SURFACE CHANGE (s_ratio={sharp2/sharp1 if sharp1>0 else 0:.2f}, e_ratio={edge_ratio:.2f}, area={a1})")
                    if log:
                        log.write(f"[PROG] Obj {idx}: Surface change s_ratio={sharp2/sharp1 if sharp1>0 else 0:.2f} e_ratio={edge_ratio:.2f} area={a1}\n")

    if prog_color_hits > 0:
        prog_color = True
    # Two-tier texture detection:
    # Tier 1: Widespread blur (3+ objects show sharpness drop)
    if prog_texture_hits >= TEXTURE_MIN_OBJECTS:
        prog_texture = True
    # Tier 2: Severe blur on any single large object (extreme sharpness drop)
    if prog_severe_blur_count > 0:
        prog_texture = True
    # Note: Per-object noise hits logged but NOT used for texture decision
    # (too many FPs from minor tracking/focus differences)
    # Tier 3a: Edge density widespread change (5+ objects)
    if prog_edge_texture_hits >= EDGE_MIN_OBJECTS:
        prog_texture = True
    # Tier 3b: Edge density change WITHOUT blur (4+ edge but 0 blur hits)
    # Rationale: pattern/material change creates edge differences without blurring
    if prog_edge_texture_hits >= 4 and prog_texture_hits == 0:
        prog_texture = True
    # Tier 3c: Large-object edge change (1+ large object with significant edge density change)
    if prog_edge_large_hits >= EDGE_LARGE_OBJ_MIN:
        prog_texture = True
    # Note: Surface detection logged but NOT used for texture decision (too many FPs)

    if verbose:
        print(f"    PROG Summary: Color={prog_color} (global_hue={global_de_ab:.1f}, per_obj={prog_color_hits}), Texture={prog_texture} (blur={prog_texture_hits}, edge={prog_edge_texture_hits}, edge_large={prog_edge_large_hits}, surface={prog_surface_change_count}, severe={prog_severe_blur_count}), analyzed={prog_objects_analyzed}")
    if log:
        log.write(f"[PROG] Summary: color={prog_color} (hits={prog_color_hits}), texture={prog_texture} (blur={prog_texture_hits}, edge={prog_edge_texture_hits}, surface={prog_surface_change_count}), objects={prog_objects_analyzed}\n\n")

    # ================================================================
    # TRACK 2: Per-Object VLM Analysis (top N largest objects)
    # ================================================================
    if verbose:
        print("  [Track 2] Per-object VLM analysis...")

    obj_color_votes = 0
    obj_texture_votes = 0
    obj_shape_votes = 0
    shape_cooccurrence_count = 0  # Track shape votes with color/texture co-occurrence

    MIN_AREA_VLM = max(IMAGE_AREA * 0.005, 1000)  # 0.5% of image
    MAX_OBJECTS_VLM = 7

    # Sort by area (largest first), select top N
    # Filter out objects with extreme area changes (tracking failures)
    VLM_AREA_RATIO_LO = 0.2   # Min area_ratio (frame2/frame1)
    VLM_AREA_RATIO_HI = 5.0   # Max area_ratio
    mask_by_area = []
    for i, (m1, m2) in enumerate(zip(masks_frame1, masks_frame2)):
        a1 = int(np.sum(m1))
        a2 = int(np.sum(m2))
        if a1 >= MIN_AREA_VLM and a2 >= PROG_MIN_AREA:
            ar = a2 / a1 if a1 > 0 else 1.0
            if VLM_AREA_RATIO_LO <= ar <= VLM_AREA_RATIO_HI:
                mask_by_area.append((i, a1, m1, m2))
            elif verbose:
                print(f"    Skipping obj {i}: area_ratio={ar:.2f} (tracking failure)")
    mask_by_area.sort(key=lambda x: x[1], reverse=True)
    vlm_objects = mask_by_area[:MAX_OBJECTS_VLM]

    if verbose:
        print(f"    Checking {len(vlm_objects)} objects (min_area={MIN_AREA_VLM:.0f})")

    for obj_idx, area1, mask1, mask2 in vlm_objects:
        area2 = int(np.sum(mask2))

        # Create crops and video
        crop1 = create_object_crop(img_before, mask1)
        crop2 = create_object_crop(img_after, mask2)

        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        obj_video_path = tmpf.name
        tmpf.close()
        temp_files.append(obj_video_path)
        create_object_video(crop1, crop2, obj_video_path)

        # Save crops for logging
        if test_log_dir:
            crops_dir = os.path.join(test_log_dir, "object_crops")
            os.makedirs(crops_dir, exist_ok=True)
            cv2.imwrite(os.path.join(crops_dir, f"obj_{obj_idx:03d}_before.png"),
                        cv2.cvtColor(crop1, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(crops_dir, f"obj_{obj_idx:03d}_after.png"),
                        cv2.cvtColor(crop2, cv2.COLOR_RGB2BGR))

        # Track per-object results for co-occurrence analysis
        obj_c = False
        obj_t = False
        obj_s = False

        # --- COLOR ---
        color_prompt = """Compare this object in Frame 1 (before) and Frame 2 (after).
Has this object's COLOR clearly changed? (e.g., changed from one distinct color to another,
became a different color, went from multi-colored to single-colored)

Ignore: brightness differences, lighting changes, shadows, viewing angle differences.
Only answer "Yes" for CLEAR, OBVIOUS color changes.

Answer ONLY "Yes" or "No"."""

        color_resp = _safe_chat_call(
            video_chat_model, verbose=False, video_path=obj_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=color_prompt, return_history=False,
            max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
        )
        if _parse_yes_no(color_resp) == "Yes":
            obj_color_votes += 1
            obj_c = True

        # --- TEXTURE ---
        texture_prompt = """Compare this object in Frame 1 (before) and Frame 2 (after).
Has this object's SURFACE TEXTURE been clearly altered?

Answer "Yes" ONLY if you see clear evidence of:
- Surface became noticeably blurry, smeared, or lost fine detail
- Surface became noticeably noisy, grainy, or pixelated
- Surface pattern fundamentally changed (textured became smooth, striped became plain)
- Surface material or finish clearly changed
- Surface quality clearly degraded (looks processed or AI-generated)

Answer "No" if:
- The surface texture looks essentially the same in both frames
- Differences are only in lighting, brightness, or viewing angle
- The object is partially hidden but visible parts have same texture
- You are unsure or the change is very subtle

Answer ONLY "Yes" or "No"."""

        texture_resp = _safe_chat_call(
            video_chat_model, verbose=False, video_path=obj_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=texture_prompt, return_history=False,
            max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
        )
        if _parse_yes_no(texture_resp) == "Yes":
            obj_texture_votes += 1
            obj_t = True

        # --- SHAPE ---
        shape_prompt = """Compare this object in Frame 1 (before) and Frame 2 (after).

CRITICAL: Has this object been PHYSICALLY DEFORMED or WARPED?

This means the actual physical object has been permanently altered - NOT just viewed
from a different angle, partially hidden, or moved to a new position.

Answer "Yes" ONLY if you see UNDENIABLE evidence of:
- The object is bent, crushed, warped, stretched, or physically deformed
- The object has a completely different physical shape (square became round, half became whole)
- Parts have clearly broken off or been physically added
- The object looks visibly enlarged, shrunk, or distorted in shape

Answer "No" if ANY of these apply:
- Object just moved to a new position (same shape, different location)
- Object is partially hidden behind something (that is occlusion, NOT deformation)
- Object looks slightly different due to camera angle or perspective change
- You are not absolutely certain the shape physically changed
- The difference could be explained by viewing angle, partial visibility, or lighting

IMPORTANT: When in doubt, answer "No". Only say "Yes" for OBVIOUS physical deformation.

Answer ONLY "Yes" or "No"."""

        shape_resp = _safe_chat_call(
            video_chat_model, verbose=False, video_path=obj_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=shape_prompt, return_history=False,
            max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
        )
        if _parse_yes_no(shape_resp) == "Yes":
            obj_shape_votes += 1
            obj_s = True

        # Track co-occurrence: shape vote with color or texture vote on same object
        if obj_s and (obj_c or obj_t):
            shape_cooccurrence_count += 1

        if verbose:
            c = _parse_yes_no(color_resp)
            t = _parse_yes_no(texture_resp)
            s = _parse_yes_no(shape_resp)
            print(f"    Obj {obj_idx}: C={c} T={t} S={s} (area={area1})")
        if log:
            log.write(f"[OBJ VLM] Obj {obj_idx} (area={area1}):\n")
            log.write(f"  Color: {_parse_yes_no(color_resp)} - {color_resp.strip()[:80]}\n")
            log.write(f"  Texture: {_parse_yes_no(texture_resp)} - {texture_resp.strip()[:80]}\n")
            log.write(f"  Shape: {_parse_yes_no(shape_resp)} - {shape_resp.strip()[:80]}\n\n")

    obj_color = obj_color_votes >= 2  # Require 2+ (all single-vote color is FP; all TPs have programmatic backing)
    obj_texture = obj_texture_votes >= 2  # Require 2+ objects for texture (all single-vote texture is FP)
    # Shape: require 2+ votes, OR accept 1 vote if that object also showed
    # color or texture change (co-occurrence indicates genuine alteration)
    if obj_shape_votes >= 2:
        obj_shape = True
    elif obj_shape_votes == 1 and shape_cooccurrence_count >= 1:
        obj_shape = True
    else:
        obj_shape = False

    # ================================================================
    # COMBINE SIGNALS
    # ================================================================
    # Color: programmatic (global hue + per-object dE_ab + chroma) OR VLM
    results["Color"] = "Yes" if (prog_color or obj_color) else "No"
    # Texture: programmatic (sharpness/noise) OR per-object VLM
    results["Texture"] = "Yes" if (prog_texture or obj_texture) else "No"
    # Shape: VLM only (programmatic shape disabled due to tracking noise)
    results["Shape"] = "Yes" if obj_shape else "No"

    if verbose:
        print(f"\n  === COMBINED RESULTS ===")
        print(f"  Programmatic: C={'Y' if prog_color else 'N'}  T={'Y' if prog_texture else 'N'}  S=disabled")
        print(f"  Object VLM:   C={'Y' if obj_color else 'N'} ({obj_color_votes})  T={'Y' if obj_texture else 'N'} ({obj_texture_votes})  S={'Y' if obj_shape else 'N'} ({obj_shape_votes})")
        print(f"  FINAL:        Color={results['Color']}  Texture={results['Texture']}  Shape={results['Shape']}")

    if log:
        log.write(f"\n=== COMBINATION ===\n")
        log.write(f"Programmatic: C={'Y' if prog_color else 'N'}  T={'Y' if prog_texture else 'N'}  S=disabled\n")
        log.write(f"Object VLM:   C={'Y' if obj_color else 'N'}  T={'Y' if obj_texture else 'N'}  S={'Y' if obj_shape else 'N'}\n")
        log.write(f"FINAL: {results}\n")
        log.close()

    return results

# =========================================================================
# Run Tests
# =========================================================================
if __name__ == "__main__":
    test_cases = [
        {
            "before": path.join(_SCRIPT_DIR, "lgie-eval-data", "input_5.jpg"),
            "after": path.join(_SCRIPT_DIR, "lgie-eval-data", "output_5_2.png"),
            "instruction": "The robotic arm puts the corn into the pot",
            "expected_unrelated_color_changes": "No",
            "expected_unrelated_texture_changes": "No",
            "expected_unrelated_shape_changes": "Yes",
            "description": "The corn is initially half but becomes a full corn in the final frame -- shape change. The robotic arm also changes shape across frames."
        },
        {
            "before": path.join(_SCRIPT_DIR, "lgie-eval-data", "input_5.jpg"),
            "after": path.join(_SCRIPT_DIR, "lgie-eval-data", "output_5_3.png"),
            "instruction": "The robotic arm puts the corn into the pot",
            "expected_unrelated_color_changes": "No",
            "expected_unrelated_texture_changes": "No",
            "expected_unrelated_shape_changes": "No",
            "description": "No color, texture or shape changes occur in any of the objects in the scene. Some objects might have become partially/fully occluded in the final/edited scene but that's not to be counted as color/texture/shape change."
        },
        {
            "before": path.join(_SCRIPT_DIR, "lgie-eval-data", "input_5.jpg"),
            "after": path.join(_SCRIPT_DIR, "lgie-eval-data", "output_5_4.png"),
            "instruction": "The robotic arm puts the steel pot on top of the stove",
            "expected_unrelated_color_changes": "No",
            "expected_unrelated_texture_changes": "No",
            "expected_unrelated_shape_changes": "No",
            "description": "No color, texture or shape changes occur in any of the objects in the scene. Some objects (like the stove) might have become partially/fully occluded in the final/edited scene but that's not to be counted as color/texture/shape change. Some objects (like the orange slices) might have appeared newly in the final/edited frame -- but this is not to be detected as color/texture/shape change since we do not know how these objects looked like in the first frame"
        },
        {
            "before": path.join(_SCRIPT_DIR, "lgie-eval-data", "input_1.jpg"),
            "after": path.join(_SCRIPT_DIR, "lgie-eval-data", "output_1_1.png"),
            "instruction": "The robotic arm puts the cucumber on top of the blue cube",
            "expected_unrelated_color_changes": "No",
            "expected_unrelated_texture_changes": "No",
            "expected_unrelated_shape_changes": "No",
            "description": "No background changes -- as per the instruction"
        },
        {
            "before": path.join(_SCRIPT_DIR, "lgie-eval-data", "input_1.jpg"),
            "after": path.join(_SCRIPT_DIR, "lgie-eval-data", "output_1_2.png"),
            "instruction": "The robotic arm puts the cucumber on top of the blue cube",
            "expected_unrelated_color_changes": "Yes",
            "expected_unrelated_texture_changes": "Yes",
            "expected_unrelated_shape_changes": "Yes",
            "description": "The blue spoon no longer has the grey scoop -- it's completely blue in the final frame. Almost every object changed shape. Surfaces changed texture."
        },
        

        {
          "before": "lgie-eval-data/original_image/oi_92.png",
          "after": "lgie-eval-data/r567_gemini_output/out_92.png",
          "instruction": "open microwave",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "No",
          "description": "No background changes -- as per the instruction",
          "id": 92
        },
        {
          "before": "lgie-eval-data/original_image/oi_93.png",
          "after": "lgie-eval-data/r567_gemini_output/out_93.png",
          "instruction": "put eggplant in pot or pan",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "Yes",
          "expected_unrelated_shape_changes": "No",
          "description": "The eggplant is placed in pot but it's texture is altered.",
          "id": 93
        },
        {
          "before": "lgie-eval-data/original_image/oi_94.png",
          "after": "lgie-eval-data/r567_gemini_output/out_94.png",
          "instruction": "take carrot off plate",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "Yes",
          "expected_unrelated_shape_changes": "No",
          "description": "Instruction is followed but kitchen top's surface texture is altered.",
          "id": 94
        },
        {
          "before": "lgie-eval-data/original_image/oi_95.png",
          "after": "lgie-eval-data/r567_gemini_output/out_95.png",
          "instruction": "open small4fbox flaps",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "Yes",
          "description": "Box flaps are open and box shape is dramatically wider than before.",
          "id": 95
        },
        {
          "before": "lgie-eval-data/original_image/oi_96.png",
          "after": "lgie-eval-data/r567_gemini_output/out_96.png",
          "instruction": "put clothes in laundry machine",
          "expected_unrelated_color_changes": "Yes",
          "expected_unrelated_texture_changes": "Yes",
          "expected_unrelated_shape_changes": "No",
          "description": "Laundry machine texture is unexpectedly altered to be more shiny and reflective, along with it's color which has become greenish.",
          "id": 96
        },
        {
          "before": "lgie-eval-data/original_image/oi_97.png",
          "after": "lgie-eval-data/r567_gemini_output/out_97.png",
          "instruction": "put spatula on cutting board",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "Yes",
          "description": "Spatula has been placed on cutting board but table's shape is altered and is round in shape.",
          "id": 97
        },
        {
          "before": "lgie-eval-data/original_image/oi_98.png",
          "after": "lgie-eval-data/r567_gemini_output/out_98.png",
          "instruction": "take carrot off plate",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "No",
          "description": "Instruction isn't followed, and scene has become brighter, but no background or foreground objects' change -- as per the instruction",
          "id": 98
        },
        {
          "before": "lgie-eval-data/original_image/oi_99.png",
          "after": "lgie-eval-data/r567_gemini_output/out_99.png",
          "instruction": "open microwave",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "No",
          "description": "Viewing angle has changed, but no background or foreground objects' change -- as per the instruction.",
          "id": 99
        },
        {
          "before": "lgie-eval-data/original_image/oi_100.png",
          "after": "lgie-eval-data/r567_gemini_output/out_100.png",
          "instruction": "take carrot off plate",
          "expected_unrelated_color_changes": "Yes",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "No",
          "description": "Kitchen top's surface color is clearly changed.",
          "id": 100
        },
        {
          "before": "lgie-eval-data/original_image/oi_101.png",
          "after": "lgie-eval-data/r567_gemini_output/out_101.png",
          "instruction": "close brown1fbox flap",
          "expected_unrelated_color_changes": "Yes",
          "expected_unrelated_texture_changes": "Yes",
          "expected_unrelated_shape_changes": "No",
          "description": "The background surface is unexpectedly altered and looks like cardboard texture.",
          "id": 101
        },
        {
          "before": "lgie-eval-data/original_image/oi_102.png",
          "after": "lgie-eval-data/r567_gemini_output/out_102.png",
          "instruction": "put spoon on plate",
          "expected_unrelated_color_changes": "Yes",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "No",
          "description": "Stationery plate has changed color and has become white from gray.",
          "id": 102
        },
        {
          "before": "lgie-eval-data/original_image/oi_103.png",
          "after": "lgie-eval-data/r567_gemini_output/out_103.png",
          "instruction": "flip pot upright in sink",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "No",
          "description": "Instruction is adhered and image became brighter but no background or foreground objects' change -- as per the instruction.",
          "id": 103
        },
        {
          "before": "lgie-eval-data/original_image/oi_104.png",
          "after": "lgie-eval-data/r567_gemini_output/out_104.png",
          "instruction": "flip pot upright in sink",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "Yes",
          "expected_unrelated_shape_changes": "No",
          "description": "Kitchen surface's texture is unexpectedly altered.",
          "id": 104
        },
        {
          "before": "lgie-eval-data/original_image/oi_105.png",
          "after": "lgie-eval-data/r567_gemini_output/out_105.png",
          "instruction": "open white1fbox flap",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "Yes",
          "expected_unrelated_shape_changes": "No",
          "description": "The table's texture has noticeably become blurry.",
          "id": 105
        },
        {
          "before": "lgie-eval-data/original_image/oi_106.png",
          "after": "lgie-eval-data/r567_gemini_output/out_106.png",
          "instruction": "right pepper shaker",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "Yes",
          "expected_unrelated_shape_changes": "No",
          "description": "Everything including the surface has become noisy, surface has also become blurry.",
          "id": 106
        },
        {
          "before": "lgie-eval-data/original_image/oi_107.png",
          "after": "lgie-eval-data/r567_gemini_output/out_107.png",
          "instruction": "put banana in pot cardboard fence",
          "expected_unrelated_color_changes": "Yes",
          "expected_unrelated_texture_changes": "Yes",
          "expected_unrelated_shape_changes": "No",
          "description": "Table color has changed and texture is gone",
          "id": 107
        },  
        {
          "before": "lgie-eval-data/original_image/oi_107.png",
          "after": "lgie-eval-data/r567_gemini_output/out_107_1.png",
          "instruction": "put banana in pot cardboard fence",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "Yes",
          "description": "Table edges have been warped",
          "id": 108
        },  
        {
          "before": "lgie-eval-data/original_image/oi_107.png",
          "after": "lgie-eval-data/r567_gemini_output/out_107_2.png",
          "instruction": "put banana in pot cardboard fence",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "Yes",
          "description": "Red strawberry and orange carrot shapes have changed.",
          "id": 109
        },
        {
          "before": "lgie-eval-data/original_image/oi_110.png",
          "after": "lgie-eval-data/r567_gemini_output/out_110.png",
          "instruction": "put carrot on plate",
          "expected_unrelated_color_changes": "Yes",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "No",
          "description": "Lighting hue turned purple-ish.",
          "id": 110
        },
        {
          "before": "lgie-eval-data/original_image/oi_111.png",
          "after": "lgie-eval-data/r567_gemini_output/out_111.png",
          "instruction": "put bowl on plate cardboard fence",
          "expected_unrelated_color_changes": "Yes",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "No",
          "description": "Unexpected lighting hue shift detected.",
          "id": 111
        },
        {
          "before": "lgie-eval-data/original_image/oi_112.png",
          "after": "lgie-eval-data/r567_gemini_output/out_112.png",
          "instruction": "put knife on cutting board cardboard fence",
          "expected_unrelated_color_changes": "Yes",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "No",
          "description": "Background wall has changed color to green from white.",
          "id": 112
        },
        {
          "before": "lgie-eval-data/original_image/oi_113.png",
          "after": "lgie-eval-data/r567_gemini_output/out_113.png",
          "instruction": "turn faucet front to left",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "No",
          "description": "Hallucinated objects have appeared but no background or foreground objects' change -- as per the instruction",
          "id": 113
        },
        {
          "before": "lgie-eval-data/original_image/oi_114.png",
          "after": "lgie-eval-data/r567_gemini_output/out_114.png",
          "instruction": "turn lever vertical to front",
          "expected_unrelated_color_changes": "Yes",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "No",
          "description": "The objects: orange pumpkin and yellow bell pepper have been distorted.",
          "id": 114
        },
        {
          "before": "lgie-eval-data/original_image/oi_114.png",
          "after": "lgie-eval-data/r567_gemini_output/out_114_1.png",
          "instruction": "turn lever vertical to front",
          "expected_unrelated_color_changes": "No",
          "expected_unrelated_texture_changes": "No",
          "expected_unrelated_shape_changes": "Yes",
          "description": "Hallucinated pumpkin has appeared but yellow bell pepper has been distorted, it's shape seems enlarged.",
          "id": 115
        }

    ]

    # =========================================================================
    # STAGED PIPELINE to avoid OOM:
    # Stage 1: SAM - generate masks for all test images
    # Stage 2: XMem - track masks for all test pairs
    # Stage 3: VideoChat-Flash - analyze changes
    # =========================================================================
    print("=" * 60)
    print("STAGED PIPELINE - Memory-Optimized")
    print("=" * 60)
    
    # Create main log directory
    LOG_DIR = path.join(_SCRIPT_DIR, "object_change_detection_logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"\nLogging intermediate data to: {LOG_DIR}")
    
    total = len(test_cases)
    
    # Data structures to hold intermediate results
    all_images_before = []
    all_images_after = []
    all_masks_frame1 = []
    all_masks_frame2 = []
    
    # =========================================================================
    # STAGE 1: Load SAM, process all images, save masks, unload SAM
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: SAM Mask Generation")
    print("=" * 60)
    
    print("\n[1/3] Loading SAM...")
    try:
        sam_mask_generator, sam_model = init_sam_model()
        print("  SAM loaded successfully")
    except Exception as e:
        print(f"  [ERROR] Failed to load SAM: {e}")
        raise RuntimeError(f"SAM initialization failed: {e}")
    
    for i, tc in enumerate(test_cases, start=1):
        print(f"\n  Processing test {i}/{total}: {tc['description'][:40]}...")
        
        test_id = f"test_{i:03d}"
        test_log_dir = os.path.join(LOG_DIR, test_id)
        os.makedirs(test_log_dir, exist_ok=True)
        
        # Resolve paths relative to script dir if needed
        before_path = tc["before"] if path.isabs(tc["before"]) else path.join(_SCRIPT_DIR, tc["before"])
        after_path = tc["after"] if path.isabs(tc["after"]) else path.join(_SCRIPT_DIR, tc["after"])
        
        # Load images
        img_before = cv2.imread(before_path)
        img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
        img_after = cv2.imread(after_path)
        img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
        
        # Resize to common resolution
        h, w = img_before.shape[:2]
        target_h = min(max(h, img_after.shape[0]), 1024)
        target_w = min(max(w, img_after.shape[1]), 1024)
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        if (new_h, new_w) != img_before.shape[:2]:
            img_before = cv2.resize(img_before, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        if img_after.shape[:2] != (new_h, new_w):
            img_after = cv2.resize(img_after, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        all_images_before.append(img_before)
        all_images_after.append(img_after)
        
        # Log input images
        cv2.imwrite(os.path.join(test_log_dir, "input_before.png"), cv2.cvtColor(img_before, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(test_log_dir, "input_after.png"), cv2.cvtColor(img_after, cv2.COLOR_RGB2BGR))
        with open(os.path.join(test_log_dir, "instruction.txt"), "w") as f:
            f.write(tc["instruction"])
        
        # Run SAM
        masks_frame1 = run_sam_inference(img_before, sam_mask_generator)
        all_masks_frame1.append(masks_frame1)
        
        print(f"    SAM detected {len(masks_frame1)} objects")
        
        # Log SAM masks
        sam_log_dir = os.path.join(test_log_dir, "sam_masks_frame1")
        save_individual_masks(masks_frame1, sam_log_dir, prefix="sam_mask")
        save_mask_visualization(masks_frame1, img_before, os.path.join(test_log_dir, "sam_overlay_frame1.png"))
        
        clear_cuda_cache()
    
    # Unload SAM
    print("\n  Unloading SAM...")
    unload_sam_model(sam_model, sam_mask_generator)
    sam_model = None
    sam_mask_generator = None
    
    # =========================================================================
    # STAGE 2: Load XMem, track masks for all test pairs, unload XMem
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: XMem Mask Tracking")
    print("=" * 60)
    
    print("\n[2/3] Loading XMem...")
    try:
        xmem_network, xmem_config = init_xmem_model()
        print("  XMem loaded successfully")
    except Exception as e:
        print(f"  [ERROR] Failed to load XMem: {e}")
        raise RuntimeError(f"XMem initialization failed: {e}")
    
    for i, (img_before, img_after, masks_f1) in enumerate(zip(all_images_before, all_images_after, all_masks_frame1), start=1):
        print(f"\n  Tracking test {i}/{total}...")
        
        test_id = f"test_{i:03d}"
        test_log_dir = os.path.join(LOG_DIR, test_id)
        
        if len(masks_f1) == 0:
            all_masks_frame2.append([])
            print("    Skipped (no masks)")
            continue
        
        # Track masks
        masks_frame2 = track_masks_with_xmem(img_before, img_after, masks_f1, xmem_network, xmem_config)
        all_masks_frame2.append(masks_frame2)
        
        print(f"    Tracked {len(masks_frame2)} masks")
        
        # Log XMem masks
        xmem_log_dir = os.path.join(test_log_dir, "xmem_masks_frame2")
        save_individual_masks(masks_frame2, xmem_log_dir, prefix="xmem_mask")
        save_mask_visualization(masks_frame2, img_after, os.path.join(test_log_dir, "xmem_overlay_frame2.png"))
        
        clear_cuda_cache()
    
    # Unload XMem
    print("\n  Unloading XMem...")
    unload_xmem_model(xmem_network)
    xmem_network = None
    
    # =========================================================================
    # STAGE 3: Load VideoChat-Flash, analyze all objects
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 3: VideoChat-Flash Analysis")
    print("=" * 60)
    
    print("\n[3/3] Loading VideoChat-Flash...")
    tokenizer, vcf_model = load_videochat_model_and_tokenizer()
    print("  VideoChat-Flash loaded successfully")
    
    score_sum = 0.0
    
    for i, tc in enumerate(test_cases, start=1):
        print(f"\n--- Test {i}: {tc['description'][:50]}... ---")
        
        test_id = f"test_{i:03d}"
        test_log_dir = os.path.join(LOG_DIR, test_id)
        
        img_before = all_images_before[i-1]
        img_after = all_images_after[i-1]
        masks_f1 = all_masks_frame1[i-1]
        masks_f2 = all_masks_frame2[i-1]
        instruction = tc["instruction"]
        
        # Analyze changes
        preds = analyze_object_changes_vlm(
            img_before, img_after, masks_f1, masks_f2, instruction,
            vcf_model, tokenizer, test_log_dir, verbose=True
        )
        
        # Normalize for comparison
        p_color = preds.get("Color", "No").strip().title()
        p_texture = preds.get("Texture", "No").strip().title()
        p_shape = preds.get("Shape", "No").strip().title()

        e_color = tc["expected_unrelated_color_changes"].strip().title()
        e_texture = tc["expected_unrelated_texture_changes"].strip().title()
        e_shape = tc["expected_unrelated_shape_changes"].strip().title()

        # Compute per-test correctness
        correct = 0
        correct += 1 if p_color == e_color else 0
        correct += 1 if p_texture == e_texture else 0
        correct += 1 if p_shape == e_shape else 0

        per_case_score = correct / 3.0
        score_sum += per_case_score

        print(
            f"\nTest {i}: {tc['description'][:60]} -> "
            f"Predictions: Color={p_color}, Texture={p_texture}, Shape={p_shape} | "
            f"Expected: Color={e_color}, Texture={e_texture}, Shape={e_shape} | "
            f"score={per_case_score:.3f}"
        )

        # GPU memory cleanup after each test
        clear_cuda_cache()

    success_score = score_sum / total
    print(f"\n{'=' * 60}")
    print(f"Final success_score = {success_score:.6f} ({success_score * 100:.2f}%)")
    print("=" * 60)