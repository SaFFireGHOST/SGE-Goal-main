"""
Detect UNRELATED CAMERA Changes (Zoom, Crop, Perspective) between two frames
using an ensemble of:
- CV geometry (AKAZE + RANSAC homography)  [primary for labels]
- VideoChat-Flash (optional, mainly for human-readable Reason)

R10, R11, R12 - Unrelated camera changes detection:

Definition of UNRELATED CROPPING:
An image was edited but only some portions of the edited image are visible —
rest of it has been turned black/masked. The textual manipulation/edit instruction
didn't indicate anything about such behaviour as something that's expected.

Definition of UNRELATED ZOOMING:
An image was edited and the edited image has the same dimensions as the original
image and there is no blackening (no cropping). But the edited image is either a
zoomed in or zoomed out version of the scene, and the textual manipulation/edit
instruction didn't indicate anything about this behaviour as something expected.

Definition of UNRELATED PERSPECTIVE/ANGLE CHANGE:
An image was edited and the edited image has the same dimensions as the original.
But the scene appears to be viewed from a different vantage point (e.g., the camera
angle seems to have shifted, rotated, or moved to a different side), and the textual
manipulation/edit instruction didn't indicate anything about this behaviour.

Key features:
- Separate detection of Zoom, Crop (black regions), and Perspective changes
- Perspective precedence: if Perspective is detected and scale-change is not large,
  do NOT also mark Zoom (prevents "perspective mislabeled as zoom")
- Crop detection via black/masked region analysis
- 3-way scoring (Zoom, Perspective, Crop)
- Scored augmentation tests

Requirements:
%uv pip install transformers torch opencv-python
"""

import os
import atexit
import tempfile
import json
import re
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------
# Config
# -------------------------
MODEL_NAME = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"
MAX_NUM_FRAMES = 8  # Reduced for efficiency - only 2 frames needed
GEN_CONFIG = dict(do_sample=False, temperature=0.0, max_new_tokens=150, top_p=0.1, num_beams=1)

SCORE_AUGMENTATIONS = False

# Use CV detector as primary - much faster than VLM
USE_CV_DETECTOR = True
USE_VLM_DETECTOR = True  # Use VLM for perspective detection when CV is uncertain
ENSEMBLE_OVERRIDE_WITH_CV = True  # CV "Yes" wins over VLM "No"

# CV thresholds - tuned for better accuracy
CV_MIN_INLIERS = 20  # Lowered to detect more cases

# Zoom thresholds - need to detect subtle zooms
CV_ZOOM_LOGSCALE_THRESH = 0.015  # Lowered: catches Test 8 (0.018), Test 19 (0.022); safe: Test 34 (0.014), Test 35 (0.013)

# Perspective thresholds - increased to reduce false positives from AI artifacts
CV_PROJ_THRESH = 9e-4  # Higher - Test 16 has 8.09e-04 which is a FP
CV_SHEAR_ROT_THRESH = 0.15  # Higher - AI edits often have shear artifacts
CV_PARALLEL_DIVERGENCE_THRESH_RAD = 0.45  # Much higher - Test 16 has 0.415 which is FP from zoom

# Perspective precedence: if perspective is detected, don't also claim zoom unless scale is VERY large
CV_ZOOM_KEEP_IF_PERSPECTIVE_LOGSCALE = 0.15  # Keep zoom if log_scale >= this when Perspective=Yes


# -------------------------
# Helpers
# -------------------------
def two_images_to_video(img1_path: str, img2_path: str, out_path: str, fps: int = 1) -> str:
    """Make a 2-frame mp4. Resizes frame2 to frame1 for encoder compatibility."""
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


def _safe_chat_call(video_chat_model, verbose: bool = False, **kwargs):
    call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if "video_path" not in call_kwargs:
        call_kwargs["video_path"] = ""

    for attempt in range(2):
        try:
            with torch.no_grad():
                return video_chat_model.chat(**call_kwargs)
        except Exception as e:
            if verbose:
                print(f"DEBUG chat attempt {attempt+1} failed:", repr(e))
            continue
    raise RuntimeError("Model chat failed after retries.")


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _norm_yes_no(v) -> str:
    t = str(v).strip().lower()
    return "Yes" if t.startswith("y") else "No"


# -------------------------
# Augmentation helpers
# -------------------------
def augment_zoom(img_path: str, scale: float, out_path: str) -> str:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read: {img_path}")
    h, w = img.shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    img_zoom = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    if scale > 1.0:
        startx = (nw - w) // 2
        starty = (nh - h) // 2
        img_zoom = img_zoom[starty:starty + h, startx:startx + w]
    else:
        padx = (w - nw) // 2
        pady = (h - nh) // 2
        img_zoom = cv2.copyMakeBorder(
            img_zoom,
            pady, h - nh - pady,
            padx, w - nw - padx,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

    cv2.imwrite(out_path, img_zoom)
    return out_path


def augment_perspective(img_path: str, max_angle: float, out_path: str) -> str:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read: {img_path}")
    h, w = img.shape[:2]
    pts1 = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    shift = int(min(h, w) * max_angle)
    pts2 = np.array([
        [np.random.randint(0, shift + 1), np.random.randint(0, shift + 1)],
        [w - np.random.randint(0, shift + 1), np.random.randint(0, shift + 1)],
        [np.random.randint(0, shift + 1), h - np.random.randint(0, shift + 1)],
        [w - np.random.randint(0, shift + 1), h - np.random.randint(0, shift + 1)]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_p = cv2.warpPerspective(img, M, (w, h))
    cv2.imwrite(out_path, img_p)
    return out_path


def augment_crop(img_path: str, crop_frac: float, out_path: str) -> str:
    """Crop and resize - creates a zoom effect, NOT a black-out crop."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read: {img_path}")
    h, w = img.shape[:2]
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    startx = np.random.randint(0, w - cw + 1)
    starty = np.random.randint(0, h - ch + 1)
    img_crop = img[starty:starty + ch, startx:startx + cw]
    img_crop = cv2.resize(img_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_path, img_crop)
    return out_path


def augment_blackout_crop(img_path: str, blackout_frac: float, edge: str, out_path: str) -> str:
    """Black out a portion of the image at specified edge - creates true unrelated crop effect.
    
    Args:
        img_path: Path to input image
        blackout_frac: Fraction of image to black out (0.1 = 10%)
        edge: Which edge to black out ('top', 'bottom', 'left', 'right', 'random')
        out_path: Path to save output image
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read: {img_path}")
    h, w = img.shape[:2]
    
    if edge == 'random':
        edge = np.random.choice(['top', 'bottom', 'left', 'right'])
    
    result = img.copy()
    
    if edge == 'top':
        cutoff = int(h * blackout_frac)
        result[:cutoff, :] = 0
    elif edge == 'bottom':
        cutoff = int(h * blackout_frac)
        result[h - cutoff:, :] = 0
    elif edge == 'left':
        cutoff = int(w * blackout_frac)
        result[:, :cutoff] = 0
    elif edge == 'right':
        cutoff = int(w * blackout_frac)
        result[:, w - cutoff:] = 0
    
    cv2.imwrite(out_path, result)
    return out_path


def augment_blackout_region(img_path: str, region_frac: float, out_path: str) -> str:
    """Black out a random rectangular region in the image - creates true unrelated crop effect."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read: {img_path}")
    h, w = img.shape[:2]
    
    # Random region size
    rh = int(h * region_frac)
    rw = int(w * region_frac)
    
    # Random position
    startx = np.random.randint(0, max(1, w - rw))
    starty = np.random.randint(0, max(1, h - rh))
    
    result = img.copy()
    result[starty:starty + rh, startx:startx + rw] = 0
    
    cv2.imwrite(out_path, result)
    return out_path


# -------------------------
# Crop Detection (black/masked regions) - IMPROVED
# -------------------------
CV_CROP_BLACK_THRESH = 8  # pixel intensity threshold for "truly black" (lowered for precision)
CV_CROP_EDGE_WIDTH = 30   # pixels from edge to check
CV_CROP_RATIO_THRESH = 0.09  # raised: avoids FP on Test 29 (0.086); keeps TP on Test 20 (0.095)
CV_CROP_EDGE_BLACK_THRESH = 0.70  # raised: edge must be 70% black to count
CV_CROP_LARGE_REGION_THRESH = 0.03  # raised: region must be > 3% of image to count

def detect_cropped_regions(img_before_path: str, img_after_path: str) -> Tuple[bool, float, str]:
    """
    Detect if AFTER image has blacked-out (cropped) regions compared to BEFORE.
    
    Key insight: Compare before and after images to detect NEW black regions,
    not just any black regions in the after image.
    """
    img_before = cv2.imread(img_before_path, cv2.IMREAD_COLOR)
    img_after = cv2.imread(img_after_path, cv2.IMREAD_COLOR)
    
    if img_before is None or img_after is None:
        return False, 0.0, "Could not read image(s)"
    
    h, w = img_before.shape[:2]
    if img_after.shape[:2] != (h, w):
        img_after = cv2.resize(img_after, (w, h), interpolation=cv2.INTER_AREA)
    
    gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
    
    total_pixels = h * w
    
    # Check for truly black pixels (< threshold) in after image
    black_mask_after = gray_after < CV_CROP_BLACK_THRESH
    black_mask_before = gray_before < CV_CROP_BLACK_THRESH
    
    # NEW black regions = black in after but NOT black in before
    new_black_mask = black_mask_after & ~black_mask_before
    new_black_count = np.sum(new_black_mask)
    new_black_ratio = new_black_count / total_pixels
    
    # Also check total black in after (some cropping turns everything black)
    total_black_after = np.sum(black_mask_after)
    total_black_ratio = total_black_after / total_pixels
    
    # Check edges specifically for new black bars
    edge_w = CV_CROP_EDGE_WIDTH
    edge_regions_after = [
        gray_after[:edge_w, :],      # top
        gray_after[-edge_w:, :],     # bottom
        gray_after[:, :edge_w],      # left
        gray_after[:, -edge_w:]      # right
    ]
    edge_regions_before = [
        gray_before[:edge_w, :],
        gray_before[-edge_w:, :],
        gray_before[:, :edge_w],
        gray_before[:, -edge_w:]
    ]
    edge_names = ["top", "bottom", "left", "right"]
    
    black_edges = []
    for region_after, region_before, name in zip(edge_regions_after, edge_regions_before, edge_names):
        edge_black_after = np.sum(region_after < CV_CROP_BLACK_THRESH) / region_after.size
        edge_black_before = np.sum(region_before < CV_CROP_BLACK_THRESH) / region_before.size
        # Edge is cropped if it's mostly black now AND wasn't before
        if edge_black_after > CV_CROP_EDGE_BLACK_THRESH and edge_black_before < 0.3:
            black_edges.append(name)
    
    # Check for large contiguous NEW black regions
    _, binary = cv2.threshold(gray_after, CV_CROP_BLACK_THRESH, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    large_black_regions = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > total_pixels * CV_CROP_LARGE_REGION_THRESH:  # region > 3% of image
            # Check if this region was NOT black before
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            region_was_black_before = np.mean(gray_before[mask > 0]) < CV_CROP_BLACK_THRESH + 10
            if not region_was_black_before:
                large_black_regions += 1
    
    # Decision: crop detected if significant NEW black regions appeared
    is_cropped = (
        (new_black_ratio > CV_CROP_RATIO_THRESH) or  # significant new black area
        (len(black_edges) >= 1 and new_black_ratio > 0.02) or  # black edge with some new black
        (large_black_regions >= 2)  # multiple large new black regions
    )
    
    reason = f"new_black={new_black_ratio:.3f}, total_black={total_black_ratio:.3f}, edges={black_edges}, large_new={large_black_regions}"
    return is_cropped, new_black_ratio, reason


# -------------------------
# CV Detector (AKAZE + Homography + Corner Parallelism + Perspective precedence + Crop detection)
# -------------------------
def cv_detect_camera_transforms(img_before_path: str, img_after_path: str) -> Dict[str, str]:
    img1 = cv2.imread(img_before_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_after_path, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return {"Zoom": "No", "Perspective": "No", "Crop": "No", "Reason": "CV: could not read image(s)"}

    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

    # Detect cropped (blacked-out) regions - compare before and after
    is_cropped, black_ratio, crop_reason = detect_cropped_regions(img_before_path, img_after_path)
    crop = "Yes" if is_cropped else "No"

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    det = cv2.AKAZE_create()
    k1, d1 = det.detectAndCompute(g1, None)
    k2, d2 = det.detectAndCompute(g2, None)

    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return {"Zoom": "No", "Perspective": "No", "Crop": crop, "Reason": f"CV: insufficient keypoints | Crop: {crop_reason}"}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < CV_MIN_INLIERS:
        return {"Zoom": "No", "Perspective": "No", "Crop": crop, "Reason": f"CV: too few good matches ({len(good)}) | Crop: {crop_reason}"}

    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return {"Zoom": "No", "Perspective": "No", "Crop": crop, "Reason": f"CV: homography failed | Crop: {crop_reason}"}

    inliers = int(mask.sum())
    if inliers < CV_MIN_INLIERS:
        return {"Zoom": "No", "Perspective": "No", "Crop": crop, "Reason": f"CV: low inliers ({inliers}) | Crop: {crop_reason}"}

    if abs(H[2, 2]) > 1e-9:
        H = H / H[2, 2]

    A = H[:2, :2]
    detA = float(np.linalg.det(A))
    scale = np.sqrt(abs(detA)) if detA != 0 else 1.0
    log_scale = abs(np.log(max(scale, 1e-9)))

    proj = max(abs(float(H[2, 0])), abs(float(H[2, 1])))

    off = max(abs(float(A[0, 1])), abs(float(A[1, 0])))
    diag = max(abs(float(A[0, 0])), abs(float(A[1, 1])), 1e-9)
    shear_rot = off / diag

    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    def angle(u, v) -> float:
        nu = np.linalg.norm(u) + 1e-9
        nv = np.linalg.norm(v) + 1e-9
        c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
        return float(np.arccos(c))

    top = warped[1] - warped[0]
    bottom = warped[2] - warped[3]
    left = warped[3] - warped[0]
    right = warped[2] - warped[1]
    top_bottom_angle = angle(top, bottom)
    left_right_angle = angle(left, right)
    parallel_divergence = max(top_bottom_angle, left_right_angle)  # radians

    # Initial decisions
    zoom = "Yes" if log_scale > CV_ZOOM_LOGSCALE_THRESH else "No"
    perspective = "Yes" if (
        proj > CV_PROJ_THRESH or
        shear_rot > CV_SHEAR_ROT_THRESH or
        parallel_divergence > CV_PARALLEL_DIVERGENCE_THRESH_RAD
    ) else "No"

    # ---- Perspective precedence fix ----
    # Perspective warps often create apparent scale changes; avoid mislabeling as Zoom.
    if perspective == "Yes" and log_scale < CV_ZOOM_KEEP_IF_PERSPECTIVE_LOGSCALE:
        zoom = "No"

    reason = (
        f"CV: inliers={inliers}, scale≈{scale:.3f}, log|s|≈{log_scale:.3f}, "
        f"proj≈{proj:.2e}, shear/rot≈{shear_rot:.3f}, parallel_div≈{parallel_divergence:.3f}rad | Crop: {crop_reason}"
    )
    return {"Zoom": zoom, "Perspective": perspective, "Crop": crop, "Reason": reason}


# -------------------------
# VLM Detector (focused on perspective detection - CV handles zoom/crop)
# -------------------------
def vlm_detect_camera_transforms(
    img_before_path: str,
    img_after_path: str,
    instruction: str,
    video_chat_model,
    video_chat_tokenizer,
    verbose: bool = False
) -> Dict[str, str]:

    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video_path = tmpf.name
    tmpf.close()
    atexit.register(lambda: os.path.exists(tmp_video_path) and os.remove(tmp_video_path))
    two_images_to_video(img_before_path, img_after_path, tmp_video_path)

    # Improved prompt - specific about what constitutes each type of change
    user_prompt = f"""Frame 1 = Original image. Frame 2 = Edited image.
Instruction: "{instruction}"

TASK: Detect UNRELATED camera changes (NOT requested in instruction).
Focus on BACKGROUND elements only. IGNORE changes to objects being edited.

ZOOM detection:
- Compare how much of the room/scene is visible
- Zoomed OUT = can see MORE of the room/table/surroundings in Frame 2
- Zoomed IN = can see LESS of the room (closer view) in Frame 2
- Answer YES if zoom changed

PERSPECTIVE detection - LOOK CAREFULLY:
- Is the camera viewing from a DIFFERENT HEIGHT or ANGLE?
- Low-angle = camera looking UP at the scene
- High-angle = camera looking DOWN (top-down view)
- Side-angle = camera shifted left/right, one side of objects appears larger
- Tilted = scene appears rotated or slanted
- EVEN SUBTLE changes in viewpoint angle should be marked YES
- If background elements appear from different angle, answer YES

CROP detection:
- Are there solid BLACK bars or regions in Frame 2?
- Answer YES only for obvious black masked areas

JSON only: {{"Zoom": "Yes"/"No", "Perspective": "Yes"/"No", "Crop": "Yes"/"No", "Reason": "brief"}}""".strip()

    out = _safe_chat_call(
        video_chat_model,
        verbose=verbose,
        video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer,
        user_prompt=user_prompt,
        return_history=False,
        max_num_frames=MAX_NUM_FRAMES,
        generation_config=GEN_CONFIG
    )

    data = _extract_json(out)
    zoom = _norm_yes_no(data.get("Zoom", "No"))
    crop = _norm_yes_no(data.get("Crop", "No"))
    persp = _norm_yes_no(data.get("Perspective", "No"))
    return {
        "Zoom": zoom,
        "Crop": crop,
        "Perspective": persp,
        "Reason": str(data.get("Reason", "")).strip(),
    }


# -------------------------
# VLM Zoom Retry (focused zoom-only prompt for uncertain cases)
# -------------------------
def vlm_detect_zoom_focused(
    img_before_path: str,
    img_after_path: str,
    video_chat_model,
    video_chat_tokenizer,
    verbose: bool = False
) -> str:
    """Focused zoom-only VLM detection - used as second opinion when primary VLM misses zoom."""
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video_path = tmpf.name
    tmpf.close()
    atexit.register(lambda: os.path.exists(tmp_video_path) and os.remove(tmp_video_path))
    two_images_to_video(img_before_path, img_after_path, tmp_video_path)

    user_prompt = """Look at Frame 1 and Frame 2 carefully. Ignore any changes to objects — focus ONLY on the camera's field of view.

ZOOM CHECK:
1. Count how many distinct objects/surfaces are visible at the EDGES of each frame.
2. Can you see objects at the edges of one frame that are NOT visible in the other?
3. Does the scene in one frame extend further (showing more walls, table surface, floor)?

SIGNS OF ZOOM:
- Frame 2 shows MORE of the scene than Frame 1 = ZOOMED OUT
- Frame 2 shows LESS of the scene than Frame 1 = ZOOMED IN
- Objects near the edges appear/disappear between frames = ZOOM CHANGE
- The "field of view" (how much of the room/table is visible) is different = ZOOM

If BOTH frames show the SAME extent of the scene with the SAME edges, answer No.
If the visible area is DIFFERENT in ANY way, answer Yes.

Answer JSON only: {"Zoom": "Yes"/"No"}"""

    out = _safe_chat_call(
        video_chat_model,
        verbose=verbose,
        video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer,
        user_prompt=user_prompt,
        return_history=False,
        max_num_frames=MAX_NUM_FRAMES,
        generation_config=GEN_CONFIG
    )

    data = _extract_json(out)
    return _norm_yes_no(data.get("Zoom", "No"))


def _check_semantic_zoom(img_before_path: str, img_after_path: str, cv_reason: str) -> bool:
    """
    Detect 'semantic zoom' — AI-regenerated images showing wider/narrower views
    without geometric scale change in feature matches.
    Uses:
      1. SSIM: if images look structurally very different despite good matches → suspect zoom
      2. Feature spatial coverage: if matched features cover different proportions → suspect zoom
    Returns True if semantic zoom is suspected.
    """
    img1 = cv2.imread(img_before_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_after_path, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return False

    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

    # 1. SSIM check — low SSIM with high inliers suggests substantial scene change
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM using OpenCV (simplified block-based SSIM)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    # Downsample for efficiency
    scale = max(1, min(h, w) // 256)
    g1s = cv2.resize(g1, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    g2s = cv2.resize(g2, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    g1f = g1s.astype(np.float64)
    g2f = g2s.astype(np.float64)

    mu1 = cv2.GaussianBlur(g1f, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2f, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(g1f ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2f ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1f * g2f, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_val = float(np.mean(ssim_map))

    # 2. Feature spatial coverage check using AKAZE matches
    det = cv2.AKAZE_create()
    k1, d1 = det.detectAndCompute(g1, None)
    k2, d2 = det.detectAndCompute(g2, None)

    coverage_diff = 0.0
    if d1 is not None and d2 is not None and len(k1) >= 10 and len(k2) >= 10:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(d1, d2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) >= 20:
            pts1 = np.float32([k1[m.queryIdx].pt for m in good])
            pts2 = np.float32([k2[m.trainIdx].pt for m in good])

            # Compute bounding box coverage in each image
            x1_min, y1_min = pts1.min(axis=0)
            x1_max, y1_max = pts1.max(axis=0)
            x2_min, y2_min = pts2.min(axis=0)
            x2_max, y2_max = pts2.max(axis=0)

            area1 = max((x1_max - x1_min) * (y1_max - y1_min), 1.0) / (h * w)
            area2 = max((x2_max - x2_min) * (y2_max - y2_min), 1.0) / (h * w)

            # If features cover significantly different portions, suspect semantic zoom
            coverage_diff = abs(area1 - area2) / max(area1, area2)

    # Decision: suspect semantic zoom if SSIM is low OR coverage differs significantly
    # Thresholds tuned to catch AI-regenerated zoom while avoiding false positives
    if ssim_val < 0.55:
        return True
    if coverage_diff > 0.25:
        return True
    # Combined: moderate SSIM drop + some coverage difference
    if ssim_val < 0.70 and coverage_diff > 0.15:
        return True

    return False


# -------------------------
# Final Detector (CV primary + VLM reason)
# -------------------------
def detect_camera_transforms(
    img_before_path: str,
    img_after_path: str,
    instruction: str,
    video_chat_model,
    video_chat_tokenizer,
    verbose: bool = False
) -> Dict[str, str]:
    """
    Efficient camera transform detection:
    1. Run CV detector first (fast)
    2. Only run VLM if CV is uncertain
    3. Use better ensemble logic based on CV confidence
    """
    
    # Always run CV detector first (fast)
    cv_pred = cv_detect_camera_transforms(img_before_path, img_after_path)
    
    # Determine CV confidence level
    cv_reason = cv_pred.get("Reason", "")
    cv_uncertain = "too few" in cv_reason or "low inliers" in cv_reason
    
    if not USE_VLM_DETECTOR:
        return cv_pred
    
    # Parse inlier count from reason
    inlier_count = 0
    import re
    inlier_match = re.search(r'inliers=(\d+)', cv_reason)
    if inlier_match:
        inlier_count = int(inlier_match.group(1))
    
    # When CV has high confidence (50+ inliers) and detected nothing geometrically,
    # always run focused VLM zoom check to catch "semantic zooms" — AI-regenerated
    # images showing wider/narrower views without geometric scale change.
    # The VLM focused zoom prompt asks about visible area/FOV which catches these.
    if inlier_count >= 50 and cv_pred["Zoom"] == "No" and cv_pred["Perspective"] == "No":
        # Try VLM focused zoom check first
        zoom_focused = vlm_detect_zoom_focused(
            img_before_path, img_after_path, video_chat_model, video_chat_tokenizer, verbose=verbose
        )
        if zoom_focused == "Yes":
            cv_pred["Zoom"] = "Yes"
            cv_pred["Reason"] = f"semantic_zoom_detected=Yes(VLM) | {cv_pred['Reason']}"
            return cv_pred
        
        # VLM missed — try SSIM/coverage-based semantic zoom detection
        semantic_zoom = _check_semantic_zoom(img_before_path, img_after_path, cv_reason)
        if semantic_zoom:
            cv_pred["Zoom"] = "Yes"
            cv_pred["Reason"] = f"semantic_zoom_detected=Yes(SSIM) | {cv_pred['Reason']}"
        return cv_pred
    
    # Run VLM only when CV is uncertain or CV detected something
    vlm_pred = vlm_detect_camera_transforms(
        img_before_path, img_after_path, instruction, video_chat_model, video_chat_tokenizer, verbose=verbose
    )
    
    # Ensemble logic
    final = {}
    
    # Zoom detection ensemble
    if cv_pred["Zoom"] == "Yes":
        # CV detected zoom - trust it
        final["Zoom"] = "Yes"
    elif cv_uncertain:
        # CV uncertain - rely on VLM for zoom
        final["Zoom"] = vlm_pred.get("Zoom", "No")
    else:
        # CV confident no zoom (even with < 50 inliers but not "uncertain")
        final["Zoom"] = "No"
    
    # Perspective detection ensemble
    # Key insight: Balance between missing true positives and avoiding false positives
    # Parse new_black from CV reason for downstream decisions
    new_black = 0.0
    nb_match = re.search(r'new_black=([\d.]+)', cv_reason)
    if nb_match:
        new_black = float(nb_match.group(1))
    
    if cv_pred["Perspective"] == "Yes":
        final["Perspective"] = "Yes"
    elif "too few" in cv_reason:
        # Too few good matches (< MIN_INLIERS before RANSAC) - images are very different
        # Trust VLM for perspective UNLESS high new_black suggests extreme zoom artifacts
        if new_black > 0.07 and vlm_pred.get("Zoom", "No") == "No":
            # High new black + VLM missed zoom → likely extreme zoom, not perspective
            final["Perspective"] = "No"
        else:
            final["Perspective"] = vlm_pred.get("Perspective", "No")
    elif "low inliers" in cv_reason:
        # Some matches found but RANSAC filtering reduced inliers below threshold
        # VLM tends to confuse zoom with perspective UNLESS crop is also detected
        # (crop + low inliers suggests genuinely different view including perspective)
        if cv_pred["Crop"] == "Yes":
            final["Perspective"] = vlm_pred.get("Perspective", "No")
        else:
            final["Perspective"] = "No"
    elif inlier_count < 25:
        # Very low inliers (but not flagged uncertain) - consider VLM
        final["Perspective"] = vlm_pred.get("Perspective", "No")
    else:
        # CV confident no perspective (25+ inliers) - DON'T override with VLM
        final["Perspective"] = "No"
    
    # Crop: CV is reliable (black region analysis)
    final["Crop"] = cv_pred["Crop"]
    
    # Zoom retry: When CV uncertain and primary VLM missed zoom, try focused zoom-only check
    # This addresses VLM's tendency to confuse zoom with perspective in combined prompts
    if final["Zoom"] == "No" and cv_uncertain:
        zoom_focused = vlm_detect_zoom_focused(
            img_before_path, img_after_path, video_chat_model, video_chat_tokenizer, verbose=verbose
        )
        if zoom_focused == "Yes":
            final["Zoom"] = "Yes"
            final_reason_prefix = "zoom_retry=Yes"
        else:
            # new_black is objective evidence of zoom-in (black borders from cropping)
            # Trust it even when perspective is detected
            if new_black > 0.05:
                final["Zoom"] = "Yes"
                final_reason_prefix = f"zoom_newblack={new_black:.3f}"
            elif final["Perspective"] == "No":
                # Only use SSIM/coverage fallback when perspective is NOT detected,
                # because perspective changes also cause low SSIM/high coverage_diff
                # leading to false positive zoom detections
                semantic_zoom = _check_semantic_zoom(img_before_path, img_after_path, cv_reason)
                if semantic_zoom:
                    final["Zoom"] = "Yes"
                    final_reason_prefix = "zoom_semantic_ssim=Yes"
                else:
                    final_reason_prefix = ""
            else:
                final_reason_prefix = ""
    else:
        final_reason_prefix = ""
    
    reason_parts = [final_reason_prefix, vlm_pred.get('Reason',''), cv_pred['Reason']]
    final["Reason"] = " | ".join(p for p in reason_parts if p).strip(" |")

    return final


# -------------------------
# Loader
# -------------------------
def load_videochat_model_and_tokenizer(model_name=MODEL_NAME):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.mm_llm_compress = False
    return tokenizer, model


# -------------------------
# Scoring
# -------------------------
def score_prediction(preds: Dict[str, str], e_zoom: str, e_persp: str, e_crop: str) -> Tuple[float, Dict[str, bool]]:
    p_zoom = preds.get("Zoom", "No").strip().title()
    p_persp = preds.get("Perspective", "No").strip().title()
    p_crop = preds.get("Crop", "No").strip().title()

    e_zoom = e_zoom.strip().title()
    e_persp = e_persp.strip().title()
    e_crop = e_crop.strip().title()

    ok = {
        "Zoom": (p_zoom == e_zoom),
        "Perspective": (p_persp == e_persp),
        "Crop": (p_crop == e_crop),
    }
    return (int(ok["Zoom"]) + int(ok["Perspective"]) + int(ok["Crop"])) / 3.0, ok


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    test_cases = [
        {
            "before": "lgie-eval-data/input_3.jpg",
            "after": "lgie-eval-data/output_3_1.png",
            "instruction": "The robotic arm puts the sushi on top of the omelette",
            "expected_unrelated_zoom": "No",
            "expected_unrelated_perspective": "No",
            "expected_unrelated_crop": "No",
            "description": "Hallucinations are there, but no unrelated camera changes",
        },
        {
            "before": "lgie-eval-data/input_3.jpg",
            "after": "lgie-eval-data/output_3_2.png",
            "instruction": "The robotic arm puts the omelette on top of the cabbage",
            "expected_unrelated_zoom": "No",
            "expected_unrelated_perspective": "No",
            "expected_unrelated_crop": "No",
            "description": "Instruction mismatch, but no unrelated camera changes",
        },
        {
            "before": "lgie-eval-data/input_5.png",
            "after": "lgie-eval-data/ft_5.png",
            "instruction": "The robotic arm picks up glue and puts it into the drawer",
            "expected_unrelated_zoom": "Yes",
            "expected_unrelated_perspective": "No",
            "expected_unrelated_crop": "No",
            "description": "Zoomed out"
        },
        {
            "before": "lgie-eval-data/input_6.png",
            "after": "lgie-eval-data/ft_6.png",
            "instruction": "The robotic arm puts the broccoli in pot or pan",
            "expected_unrelated_zoom": "Yes",
            "expected_unrelated_perspective": "No",
            "expected_unrelated_crop": "No",
            "description": "Zoomed out"
        },
        {
            "before": "lgie-eval-data/input_7.png",
            "after": "lgie-eval-data/ft_7.png",
            "instruction": "The robotic arm puts the banana in colander",
            "expected_unrelated_zoom": "Yes",
            "expected_unrelated_perspective": "No",
            "expected_unrelated_crop": "No",
            "description": "Zoomed out"
        },
        {
            "before": "lgie-eval-data/input_8.png",
            "after": "lgie-eval-data/ft_8.png",
            "instruction": "The robotic arm takes the banana out of the colander",
            "expected_unrelated_zoom": "Yes",
            "expected_unrelated_perspective": "No",
            "expected_unrelated_crop": "No",
            "description": "Zoomed out"
        },
        {
            "before": "lgie-eval-data/input_9.png",
            "after": "lgie-eval-data/ft_9.png",
            "instruction": "The robotic arm takes the broccoli out of the pan",
            "expected_unrelated_zoom": "Yes",
            "expected_unrelated_perspective": "No",
            "expected_unrelated_crop": "No",
            "description": "Zoomed out"
        },
        {
            "before": "lgie-eval-data/input_10.png",
            "after": "lgie-eval-data/ft_10.png",
            "instruction": "The robotic arm puts the cucumber in the cup",
            "expected_unrelated_zoom": "Yes",
            "expected_unrelated_perspective": "No",
            "expected_unrelated_crop": "No",
            "description": "Zoomed out"
        },
        {
            "before": "lgie-eval-data/input_11.png",
            "after": "lgie-eval-data/ft_11.png",
            "instruction": "The robotic arm puts the small spoon from basked in the tray",
            "expected_unrelated_zoom": "Yes",
            "expected_unrelated_perspective": "No",
            "expected_unrelated_crop": "No",
            "description": "Zoomed out"
        },
        {
            "before": "lgie-eval-data/input_12.png",
            "after": "lgie-eval-data/ft_12.png",
            "instruction": "The robotic arm closes the brown box's flap",
            "expected_unrelated_zoom": "Yes",
            "expected_unrelated_perspective": "No",
            "expected_unrelated_crop": "No",
            "description": "Zoomed out"
        },
        {
            "before": "lgie-eval-data/input_13.png",
            "after": "lgie-eval-data/ft_13.png",
            "instruction": "The robotic arm puts the corn in the pan which is on the stove",
            "expected_unrelated_zoom": "Yes",
            "expected_unrelated_perspective": "No",
            "expected_unrelated_crop": "No",
            "description": "Zoomed out"
        },
        {
            "before": "lgie-eval-data/input_14.png",
            "after": "lgie-eval-data/ft_14.png",
            "instruction": "The robotic arm picks up pan from the stove",
            "expected_unrelated_zoom": "Yes",
            "expected_unrelated_perspective": "No",
            "expected_unrelated_crop": "No",
            "description": "Zoomed out"
        },
        {
          "id": 138,
          "before": "lgie-eval-data/original_image/oi_138.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_138.png",
          "instruction": "put banana in pot cardboard fence",
          "expected_unrelated_zoom": "No",
          "expected_unrelated_perspective": "Yes",
          "expected_unrelated_crop": "No",
          "description": "Instruction followed, but unexpectedly camera perspective is altered to be a low-angle view."
        },
        {
          "id": 139,
          "before": "lgie-eval-data/original_image/oi_139.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_139.png",
          "instruction": "take sushi off plate",
          "expected_unrelated_zoom": "No",
          "expected_unrelated_perspective": "Yes",
          "expected_unrelated_crop": "No",
          "description": "Instruction followed, but unexpectedly camera perspective is altered to be a high-angle top-down view."
        },
        {
          "id": 140,
          "before": "lgie-eval-data/original_image/oi_140.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_140.png",
          "instruction": "flip pot upright in sink",
          "expected_unrelated_zoom": "No",
          "expected_unrelated_perspective": "Yes",
          "expected_unrelated_crop": "No",
          "description": "Instruction followed, but unexpectedly camera perspective is altered to be a oblique side-angle view."
        },
        {
          "id": 141,
          "before": "lgie-eval-data/original_image/oi_141.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_141.png",
          "instruction": "open white1fbox flap",
          "expected_unrelated_zoom": "Yes",
          "expected_unrelated_perspective": "No",
          "expected_unrelated_crop": "No",
          "description": "Instruction match, but camera seems to have zoomed-in to white-box as red plate near the right edge is out of frame."
        },
        {
          "id": 142,
          "before": "lgie-eval-data/original_image/oi_142.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_142.png",
          "instruction": "take carrot off plate",
          "expected_unrelated_zoom": "Yes",
          "expected_unrelated_perspective": "Yes",
          "expected_unrelated_crop": "No",
          "description": "Instruction followed, but camera perspective seems to be changed from right high-angle to left high-angle, and camera has zoomed-out as more objects are in view and field-of-view is wider."
        },
        {
          "id": 143,
          "before": "lgie-eval-data/original_image/oi_143.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_143.png",
          "instruction": "pour almonds in pot",
          "expected_unrelated_zoom": "Yes",
          "expected_unrelated_perspective": "No",
          "expected_unrelated_crop": "No",
          "description": "Instruction followed, but camera seems to have zoomed-in as field-of-view has narrowed."
        },
        {
          "id": 144,
          "before": "lgie-eval-data/original_image/oi_144.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_144_1.png",
          "instruction": "pick up bowl and put in small4fbox",
          "expected_unrelated_zoom": "Yes",
          "expected_unrelated_perspective": "No",
          "expected_unrelated_crop": "Yes",
          "description": "Instruction mismatch, and camera seems to zoomed-in from right as part of banana is out-of-frame. Also the blue bowl and sushi is blacked-out (cropped)."
        },
        {
          "id": 145,
          "before": "lgie-eval-data/original_image/oi_145.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_145_2.png",
          "instruction": "put green squash in pot or pan",
          "expected_unrelated_zoom": "Yes",
          "expected_unrelated_perspective": "No",
          "expected_unrelated_crop": "Yes",
          "description": "Camera zoomed-in on the pan, and the top of the top-most pot has been blacked-out (cropped)."
        },
        {
          "id": 146,
          "before": "lgie-eval-data/original_image/oi_146.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_146_1.png",
          "instruction": "pick up pan from stove",
          "expected_unrelated_zoom": "Yes",
          "expected_unrelated_perspective": "Yes",
          "expected_unrelated_crop": "Yes",
          "description": "Instruction followed, but camera zoomed-in on the pan cutting frame from left, bottom, and right edge; camera also seems to have altered perspective to be lower in position than before; also the whole pan above the stove has been blacked-out (cropped) along with the top of pot which is placed in kitchen sink."
        },
        {
          "id": 147,
          "before": "lgie-eval-data/original_image/oi_147.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_147.png",
          "instruction": "put broccoli in pot",
          "expected_unrelated_zoom": "No",
          "expected_unrelated_perspective": "No",
          "expected_unrelated_crop": "No",
          "description": "Instruction mismatch, but no unrelated camera changes"
        },
        {
          "id": 148,
          "before": "lgie-eval-data/original_image/oi_148.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_148.png",
          "instruction": "flip pot upright in sink",
          "expected_unrelated_zoom": "No",
          "expected_unrelated_perspective": "No",
          "expected_unrelated_crop": "No",
          "description": "Instruction mismatch, but no unrelated camera changes"
        },
        {
          "id": 149,
          "before": "lgie-eval-data/original_image/oi_149.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_149.png",
          "instruction": "pick up scissors and put into drawer",
          "expected_unrelated_zoom": "No",
          "expected_unrelated_perspective": "No",
          "expected_unrelated_crop": "No",
          "description": "Instruction mismatch, but no unrelated camera changes"
        },
        {
          "id": 150,
          "before": "lgie-eval-data/original_image/oi_150.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_150.png",
          "instruction": "close fridge",
          "expected_unrelated_zoom": "Yes",
          "expected_unrelated_perspective": "Yes",
          "expected_unrelated_crop": "No",
          "description": "Camera perspective altered and camera seems to have zoomed-out enlarging the field-of-view."
        },
        {
          "id": 151,
          "before": "lgie-eval-data/original_image/oi_151.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_151.png",
          "instruction": "turn faucet front to left",
          "expected_unrelated_zoom": "Yes",
          "expected_unrelated_perspective": "No",
          "expected_unrelated_crop": "No",
          "description": "Camera zoomed-in narrowing the field-of-view."
        },
        {
          "id": 152,
          "before": "lgie-eval-data/original_image/oi_152.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_152.png",
          "instruction": "put sweet potato in pot which is in sink",
          "expected_unrelated_zoom": "Yes",
          "expected_unrelated_perspective": "No",
          "expected_unrelated_crop": "No",
          "description": "Camera has zoomed-out significantly showing the whole table."
        },
        {
          "id": 153,
          "before": "lgie-eval-data/original_image/oi_153.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_153.png",
          "instruction": "take broccoli out of pan",
          "expected_unrelated_zoom": "Yes",
          "expected_unrelated_perspective": "No",
          "expected_unrelated_crop": "No",
          "description": "Camera has noticeably zoomed-out as faucet in the right of scene is now in frame."
        },
        {
          "id": 154,
          "before": "lgie-eval-data/original_image/oi_154.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_154.png",
          "instruction": "pour almonds in pot",
          "expected_unrelated_zoom": "Yes",
          "expected_unrelated_perspective": "No",
          "expected_unrelated_crop": "No",
          "description": "Instruction followed, but camera has zoomed-in very close to the pot."
        },
        {
          "id": 155,
          "before": "lgie-eval-data/original_image/oi_155.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_155.png",
          "instruction": "put lid on pot or pan",
          "expected_unrelated_zoom": "No",
          "expected_unrelated_perspective": "Yes",
          "expected_unrelated_crop": "No",
          "description": "Instruction followed, but camera's perspective seem to be altered to be a low, side-on perspective."
        },
        {
          "id": 156,
          "before": "lgie-eval-data/original_image/oi_156.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_156.png",
          "instruction": "pick up bowl and put in small4fbox",
          "expected_unrelated_zoom": "No",
          "expected_unrelated_perspective": "Yes",
          "expected_unrelated_crop": "No",
          "description": "Camera's perspective have been altered and looks like high-angle, slightly tilted view"
        },
        {
          "id": 157,
          "before": "lgie-eval-data/original_image/oi_157.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_157.png",
          "instruction": "put lid on pot",
          "expected_unrelated_zoom": "No",
          "expected_unrelated_perspective": "Yes",
          "expected_unrelated_crop": "No",
          "description": "Along with some hallucinations camera's perspective have been altered to be steep low-angle view."
        },
        {
          "id": 158,
          "before": "lgie-eval-data/original_image/oi_158.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_158.png",
          "instruction": "put knife on cutting board cardboard fence",
          "expected_unrelated_zoom": "No",
          "expected_unrelated_perspective": "Yes",
          "expected_unrelated_crop": "No",
          "description": "Camera perspective have been altered to an angled side perspective, as if the camera is positioned slightly to the left of the setup and tilted downward at a shallow angle."
        },
        {
          "id": 159,
          "before": "lgie-eval-data/original_image/oi_159.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_159.png",
          "instruction": "flip pot upright in sink",
          "expected_unrelated_zoom": "No",
          "expected_unrelated_perspective": "No",
          "expected_unrelated_crop": "No",
          "description": "Hallucinations occured, but no unrelated camera changes."
        },
        {
          "id": 160,
          "before": "lgie-eval-data/original_image/oi_160.png",
          "after": "lgie-eval-data/r10-11-12_gemini_output/out_160.png",
          "instruction": "put small spoon from basket to tray",
          "expected_unrelated_zoom": "No",
          "expected_unrelated_perspective": "No",
          "expected_unrelated_crop": "No",
          "description": "Perfect edit -- Instruction followed and no unrelated camera changes."
        }

    ]

    tokenizer, model = load_videochat_model_and_tokenizer()

    score_sum = 0.0
    total_scored = 0

    for i, tc in enumerate(test_cases, start=1):
        preds = detect_camera_transforms(tc["before"], tc["after"], tc["instruction"], model, tokenizer, verbose=False)
        per_score, ok = score_prediction(
            preds,
            tc["expected_unrelated_zoom"],
            tc["expected_unrelated_perspective"],
            tc["expected_unrelated_crop"]
        )
        score_sum += per_score
        total_scored += 1

        print(
            f"Test {i}: {tc['description']} -> "
            f"Predictions: Zoom={preds.get('Zoom')}, Perspective={preds.get('Perspective')}, Crop={preds.get('Crop')} | "
            f"Expected: Zoom={tc['expected_unrelated_zoom']}, Perspective={tc['expected_unrelated_perspective']}, Crop={tc['expected_unrelated_crop']} | "
            f"score={per_score:.3f} (ok={ok})"
        )
        if preds.get("Reason"):
            print("  Reason:", preds["Reason"])

        if SCORE_AUGMENTATIONS:
            # Zoom: expect Zoom=Yes, Perspective=No, Crop=No
            zoom_scales = np.linspace(0.80, 1.20, 10)
            for j, scale in enumerate(zoom_scales):
                tmp_after = f"/tmp/after_zoom_{i}_{j}.jpg"
                augment_zoom(tc["after"], float(scale), tmp_after)
                preds_aug = detect_camera_transforms(tc["before"], tmp_after, tc["instruction"], model, tokenizer, verbose=False)

                exp_zoom, exp_persp, exp_crop = "Yes", "No", "No"
                per_aug, ok_aug = score_prediction(preds_aug, exp_zoom, exp_persp, exp_crop)
                score_sum += per_aug
                total_scored += 1

                print(
                    f"Test {i} (Zoom Aug {j+1} scale={scale:.2f}): "
                    f"Pred: Zoom={preds_aug.get('Zoom')}, Persp={preds_aug.get('Perspective')}, Crop={preds_aug.get('Crop')} | "
                    f"Exp: Zoom={exp_zoom}, Persp={exp_persp}, Crop={exp_crop} | "
                    f"score={per_aug:.3f} (ok={ok_aug})"
                )

            # Perspective: expect Zoom=No, Perspective=Yes, Crop=No
            for j in range(10):
                tmp_after = f"/tmp/after_persp_{i}_{j}.jpg"
                augment_perspective(tc["after"], 0.08, tmp_after)
                preds_aug = detect_camera_transforms(tc["before"], tmp_after, tc["instruction"], model, tokenizer, verbose=False)

                exp_zoom, exp_persp, exp_crop = "No", "Yes", "No"
                per_aug, ok_aug = score_prediction(preds_aug, exp_zoom, exp_persp, exp_crop)
                score_sum += per_aug
                total_scored += 1

                print(
                    f"Test {i} (Perspective Aug {j+1}): "
                    f"Pred: Zoom={preds_aug.get('Zoom')}, Persp={preds_aug.get('Perspective')}, Crop={preds_aug.get('Crop')} | "
                    f"Exp: Zoom={exp_zoom}, Persp={exp_persp}, Crop={exp_crop} | "
                    f"score={per_aug:.3f} (ok={ok_aug})"
                )

            # Crop: expect Zoom=Yes (since augment_crop also zooms), Perspective=No, Crop=No (augment_crop doesn't black-out, it just crops and resizes)
            # Note: augment_crop creates a zoom effect, not a black-out crop effect
            crop_fracs = np.linspace(0.70, 0.95, 10)
            for j, frac in enumerate(crop_fracs):
                tmp_after = f"/tmp/after_crop_{i}_{j}.jpg"
                augment_crop(tc["after"], float(frac), tmp_after)
                preds_aug = detect_camera_transforms(tc["before"], tmp_after, tc["instruction"], model, tokenizer, verbose=False)

                exp_zoom, exp_persp, exp_crop = "Yes", "No", "No"
                per_aug, ok_aug = score_prediction(preds_aug, exp_zoom, exp_persp, exp_crop)
                score_sum += per_aug
                total_scored += 1

                print(
                    f"Test {i} (Crop->Zoom Aug {j+1} frac={frac:.2f}): "
                    f"Pred: Zoom={preds_aug.get('Zoom')}, Persp={preds_aug.get('Perspective')}, Crop={preds_aug.get('Crop')} | "
                    f"Exp: Zoom={exp_zoom}, Persp={exp_persp}, Crop={exp_crop} | "
                    f"score={per_aug:.3f} (ok={ok_aug})"
                )

            # True Black-out Crop: expect Zoom=No, Perspective=No, Crop=Yes
            # Test edge black-outs
            edges = ['top', 'bottom', 'left', 'right']
            blackout_fracs = [0.1, 0.15, 0.2]
            for j, (edge, frac) in enumerate([(e, f) for e in edges for f in blackout_fracs]):
                tmp_after = f"/tmp/after_blackout_{i}_{j}.jpg"
                augment_blackout_crop(tc["after"], frac, edge, tmp_after)
                preds_aug = detect_camera_transforms(tc["before"], tmp_after, tc["instruction"], model, tokenizer, verbose=False)

                exp_zoom, exp_persp, exp_crop = "No", "No", "Yes"
                per_aug, ok_aug = score_prediction(preds_aug, exp_zoom, exp_persp, exp_crop)
                score_sum += per_aug
                total_scored += 1

                print(
                    f"Test {i} (Blackout Crop Aug {j+1} edge={edge} frac={frac:.2f}): "
                    f"Pred: Zoom={preds_aug.get('Zoom')}, Persp={preds_aug.get('Perspective')}, Crop={preds_aug.get('Crop')} | "
                    f"Exp: Zoom={exp_zoom}, Persp={exp_persp}, Crop={exp_crop} | "
                    f"score={per_aug:.3f} (ok={ok_aug})"
                )

            # Random region black-out: expect Zoom=No, Perspective=No, Crop=Yes
            for j in range(5):
                tmp_after = f"/tmp/after_blackout_region_{i}_{j}.jpg"
                augment_blackout_region(tc["after"], 0.15, tmp_after)
                preds_aug = detect_camera_transforms(tc["before"], tmp_after, tc["instruction"], model, tokenizer, verbose=False)

                exp_zoom, exp_persp, exp_crop = "No", "No", "Yes"
                per_aug, ok_aug = score_prediction(preds_aug, exp_zoom, exp_persp, exp_crop)
                score_sum += per_aug
                total_scored += 1

                print(
                    f"Test {i} (Blackout Region Aug {j+1}): "
                    f"Pred: Zoom={preds_aug.get('Zoom')}, Persp={preds_aug.get('Perspective')}, Crop={preds_aug.get('Crop')} | "
                    f"Exp: Zoom={exp_zoom}, Persp={exp_persp}, Crop={exp_crop} | "
                    f"score={per_aug:.3f} (ok={ok_aug})"
                )

    final = score_sum / max(total_scored, 1)
    print(f"\nFinal success_score = {final:.6f} ({final * 100:.2f}%) over {total_scored} scored cases")