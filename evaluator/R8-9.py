"""
Minimal end-to-end script: Detect Unrelated LIGHTING Changes (Brighter/Darker)
using a hybrid approach: CV-based histogram analysis + VideoChat-Flash confirmation.

Logic:
1. Compute histogram-based brightness metrics (mean, median, percentiles) for both images.
2. Compare global and regional brightness changes using CV analysis.
3. Use VideoChat-Flash for confirmation when CV analysis is ambiguous.
4. Ensemble CV + VLM results for final decision.

Requirements:
%uv pip install transformers torch opencv-python numpy
"""

import os
import atexit
import tempfile
from typing import Tuple, Dict, List
import numpy as np

import cv2
import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------
# Config
# -------------------------
MODEL_NAME = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"
MAX_NUM_FRAMES = 32

# Configs - Low temperature for deterministic analysis
GEN_CONFIG = dict(do_sample=False, temperature=0.0, max_new_tokens=200, top_p=0.1, num_beams=1)

# CV Detection Thresholds
# These thresholds control sensitivity of brightness change detection
CV_BRIGHTNESS_MEAN_DIFF_THRESH = 4.5     # Mean brightness difference threshold (lowered for sensitivity)
CV_BRIGHTNESS_RATIO_THRESH = 0.04        # Percentage change threshold (4%)
CV_BRIGHTNESS_PERCENTILE_DIFF = 6.0      # Percentile-based difference threshold
CV_REGION_CHANGE_THRESH = 0.10           # Threshold for regional changes

# Ensemble mode: "cv_primary" uses CV as primary, "vlm_primary" uses VLM as primary
ENSEMBLE_MODE = "cv_primary"

# -------------------------
# Helpers
# -------------------------
def two_images_to_video(img1_path: str, img2_path: str, out_path: str, fps: int = 1) -> str:
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None: raise ValueError(f"Could not read: {img1_path}")
    if img2 is None: raise ValueError(f"Could not read: {img2_path}")

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
    if "video_path" not in call_kwargs: call_kwargs["video_path"] = ""
    
    for attempt in range(2):
        try:
            with torch.no_grad():
                return video_chat_model.chat(**call_kwargs)
        except Exception as e:
            if verbose: print(f"DEBUG chat attempt {attempt+1} failed:", repr(e))
            continue
    raise RuntimeError("Model chat failed after retries.")

def _parse_yes_no(text: str) -> str:
    """Helper to extract a strict Yes or No from the model output."""
    t = text.lower()
    if "verdict: yes" in t: return "Yes"
    if "verdict: no" in t: return "No"
    
    # Fallback heuristic
    if "yes" in t and "no" not in t: return "Yes"
    if "no" in t and "yes" not in t: return "No"
    return "No"

# Augmentation helpers
# -------------------------
def augment_brightness(img_path: str, factor: float, out_path: str) -> str:
    img = cv2.imread(img_path)
    if img is None: raise ValueError(f"Could not read: {img_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype('float32')
    hsv[...,2] = hsv[...,2] * factor
    hsv[...,2] = hsv[...,2].clip(0,255)
    hsv = hsv.astype('uint8')
    img_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(out_path, img_bright)
    return out_path


# -------------------------
# CV-Based Brightness Analysis
# -------------------------
def compute_brightness_metrics(img: np.ndarray) -> Dict[str, float]:
    """
    Compute various brightness metrics for an image.
    Returns dict with mean, median, std, and percentile values.
    """
    # Convert to LAB color space for perceptual brightness (L channel)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    
    # Also compute grayscale brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # HSV Value channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2].astype(np.float32)
    
    return {
        "l_mean": np.mean(l_channel),
        "l_median": np.median(l_channel),
        "l_std": np.std(l_channel),
        "l_p10": np.percentile(l_channel, 10),
        "l_p90": np.percentile(l_channel, 90),
        "gray_mean": np.mean(gray),
        "gray_median": np.median(gray),
        "v_mean": np.mean(v_channel),
        "v_median": np.median(v_channel),
    }


def compute_regional_brightness(img: np.ndarray, grid_size: int = 3) -> np.ndarray:
    """
    Divide image into grid and compute mean brightness for each region.
    Returns a grid_size x grid_size array of brightness values.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    
    region_h = h // grid_size
    region_w = w // grid_size
    
    regional_brightness = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            region = gray[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
            regional_brightness[i, j] = np.mean(region)
    
    return regional_brightness


def compute_histogram_features(img: np.ndarray) -> Dict[str, float]:
    """
    Compute histogram-based features for brightness analysis.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize
    
    # Compute cumulative histogram
    cumhist = np.cumsum(hist)
    
    # Find quartile positions
    q25_idx = np.searchsorted(cumhist, 0.25)
    q50_idx = np.searchsorted(cumhist, 0.50)
    q75_idx = np.searchsorted(cumhist, 0.75)
    
    # Dark and bright pixel ratios
    dark_ratio = cumhist[64] if len(cumhist) > 64 else 0  # Pixels < 64
    bright_ratio = 1.0 - (cumhist[192] if len(cumhist) > 192 else 1)  # Pixels > 192
    
    return {
        "q25": q25_idx,
        "q50": q50_idx,
        "q75": q75_idx,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "hist_std": np.std(np.arange(256) * hist),  # Weighted std of histogram
    }


def compute_block_brightness_change(img1: np.ndarray, img2: np.ndarray, block_size: int = 16) -> Tuple[float, float]:
    """
    Compute brightness change by comparing local blocks.
    Returns (brighter_score, darker_score) between 0 and 1.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    h, w = gray1.shape
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size
    
    brighter_count = 0
    darker_count = 0
    significant_count = 0
    total_blocks = n_blocks_h * n_blocks_w
    
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block1 = gray1[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            block2 = gray2[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            
            mean1 = np.mean(block1)
            mean2 = np.mean(block2)
            diff = mean2 - mean1
            
            # Only count significant changes (>3 intensity levels)
            if abs(diff) > 3:
                significant_count += 1
                if diff > 0:
                    brighter_count += 1
                else:
                    darker_count += 1
    
    if significant_count == 0:
        return 0.0, 0.0
    
    # Ratio of blocks that became brighter/darker among significantly changed blocks
    brighter_score = brighter_count / total_blocks
    darker_score = darker_count / total_blocks
    
    return brighter_score, darker_score


def cv_detect_brightness_changes(img_before_path: str, img_after_path: str, 
                                  verbose: bool = False) -> Dict[str, any]:
    """
    Use CV-based analysis to detect brightness changes between two images.
    Returns dict with detected changes and confidence metrics.
    """
    img1 = cv2.imread(img_before_path)
    img2 = cv2.imread(img_after_path)
    
    if img1 is None:
        raise ValueError(f"Could not read: {img_before_path}")
    if img2 is None:
        raise ValueError(f"Could not read: {img_after_path}")
    
    # Resize img2 to match img1 if needed
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    
    # Compute global brightness metrics
    metrics1 = compute_brightness_metrics(img1)
    metrics2 = compute_brightness_metrics(img2)
    
    # Compute histogram features
    hist1 = compute_histogram_features(img1)
    hist2 = compute_histogram_features(img2)
    
    # Compute block-based brightness analysis
    block_brighter, block_darker = compute_block_brightness_change(img1, img2)
    
    # Compute regional brightness
    regional1 = compute_regional_brightness(img1)
    regional2 = compute_regional_brightness(img2)
    
    # Compute differences
    l_mean_diff = metrics2["l_mean"] - metrics1["l_mean"]
    gray_mean_diff = metrics2["gray_mean"] - metrics1["gray_mean"]
    v_mean_diff = metrics2["v_mean"] - metrics1["v_mean"]
    
    # Percentage change
    l_pct_change = (l_mean_diff / (metrics1["l_mean"] + 1e-6)) * 100
    gray_pct_change = (gray_mean_diff / (metrics1["gray_mean"] + 1e-6)) * 100
    
    # Regional changes
    regional_diff = regional2 - regional1
    max_regional_increase = np.max(regional_diff)
    max_regional_decrease = np.min(regional_diff)
    
    # Count regions that significantly changed
    bright_regions = np.sum(regional_diff > CV_BRIGHTNESS_MEAN_DIFF_THRESH * 1.2)
    dark_regions = np.sum(regional_diff < -CV_BRIGHTNESS_MEAN_DIFF_THRESH * 1.2)
    
    # Percentile-based analysis (robust to object movements)
    p90_diff = metrics2["l_p90"] - metrics1["l_p90"]
    p10_diff = metrics2["l_p10"] - metrics1["l_p10"]
    
    # Histogram-based analysis
    q50_diff = hist2["q50"] - hist1["q50"]  # Median brightness shift
    dark_ratio_diff = hist2["dark_ratio"] - hist1["dark_ratio"]  # More dark pixels?
    bright_ratio_diff = hist2["bright_ratio"] - hist1["bright_ratio"]  # More bright pixels?
    
    # Decision logic with multiple criteria
    became_brighter = False
    became_darker = False
    brighter_confidence = 0.0
    darker_confidence = 0.0
    
    # Check for brighter - need at least 2 signals
    brighter_signals = 0
    if l_mean_diff > CV_BRIGHTNESS_MEAN_DIFF_THRESH:
        brighter_signals += 1
    if gray_mean_diff > CV_BRIGHTNESS_MEAN_DIFF_THRESH:
        brighter_signals += 1
    if l_pct_change > CV_BRIGHTNESS_RATIO_THRESH * 100:
        brighter_signals += 1
    if p90_diff > CV_BRIGHTNESS_PERCENTILE_DIFF:
        brighter_signals += 1
    if bright_regions >= 3:  # At least 3 regions significantly brighter
        brighter_signals += 1
    if q50_diff > 8:  # Histogram median shifted brighter
        brighter_signals += 1
    if bright_ratio_diff > 0.05:  # More bright pixels
        brighter_signals += 1
    if block_brighter > 0.25:  # >25% of blocks became brighter
        brighter_signals += 1
    
    # Check for darker - need at least 2 signals
    darker_signals = 0
    if l_mean_diff < -CV_BRIGHTNESS_MEAN_DIFF_THRESH:
        darker_signals += 1
    if gray_mean_diff < -CV_BRIGHTNESS_MEAN_DIFF_THRESH:
        darker_signals += 1
    if l_pct_change < -CV_BRIGHTNESS_RATIO_THRESH * 100:
        darker_signals += 1
    if p10_diff < -CV_BRIGHTNESS_PERCENTILE_DIFF:
        darker_signals += 1
    if dark_regions >= 3:  # At least 3 regions significantly darker
        darker_signals += 1
    if q50_diff < -8:  # Histogram median shifted darker
        darker_signals += 1
    if dark_ratio_diff > 0.05:  # More dark pixels
        darker_signals += 1
    if block_darker > 0.25:  # >25% of blocks became darker
        darker_signals += 1
    
    # Compound subtle dimming: if multiple metrics all point darker by small amounts,
    # count as an additional signal (catches subtle but consistent dimming)
    subtle_dark_indicators = 0
    if l_mean_diff < -2.0: subtle_dark_indicators += 1
    if gray_mean_diff < -2.0: subtle_dark_indicators += 1
    if p10_diff < -3.0: subtle_dark_indicators += 1
    if q50_diff < -4: subtle_dark_indicators += 1
    if block_darker > 0.15: subtle_dark_indicators += 1
    if subtle_dark_indicators >= 3:
        darker_signals += 1
    
    # Compound subtle brightening (symmetric)
    subtle_bright_indicators = 0
    if l_mean_diff > 2.0: subtle_bright_indicators += 1
    if gray_mean_diff > 2.0: subtle_bright_indicators += 1
    if p90_diff > 3.0: subtle_bright_indicators += 1
    if q50_diff > 4: subtle_bright_indicators += 1
    if block_brighter > 0.15: subtle_bright_indicators += 1
    if subtle_bright_indicators >= 3:
        brighter_signals += 1
    
    # Need at least 2 signals for strong detection, or 1 signal with high magnitude
    if brighter_signals >= 2:
        became_brighter = True
        brighter_confidence = min(1.0, brighter_signals / 5.0)
    elif brighter_signals == 1 and (l_mean_diff > CV_BRIGHTNESS_MEAN_DIFF_THRESH * 2 or 
                                     l_pct_change > CV_BRIGHTNESS_RATIO_THRESH * 200):
        # Single strong signal can trigger detection
        became_brighter = True
        brighter_confidence = 0.5
    
    if darker_signals >= 2:
        became_darker = True
        darker_confidence = min(1.0, darker_signals / 5.0)
    elif darker_signals == 1 and (l_mean_diff < -CV_BRIGHTNESS_MEAN_DIFF_THRESH * 2 or 
                                   l_pct_change < -CV_BRIGHTNESS_RATIO_THRESH * 200):
        # Single strong signal can trigger detection
        became_darker = True
        darker_confidence = 0.5
    
    if verbose:
        print(f"=== CV Brightness Analysis ===")
        print(f"L mean: {metrics1['l_mean']:.2f} -> {metrics2['l_mean']:.2f} (diff: {l_mean_diff:.2f})")
        print(f"Gray mean: {metrics1['gray_mean']:.2f} -> {metrics2['gray_mean']:.2f} (diff: {gray_mean_diff:.2f})")
        print(f"L % change: {l_pct_change:.2f}%")
        print(f"P90 diff: {p90_diff:.2f}, P10 diff: {p10_diff:.2f}")
        print(f"Histogram Q50 diff: {q50_diff:.2f}, Dark ratio diff: {dark_ratio_diff:.4f}, Bright ratio diff: {bright_ratio_diff:.4f}")
        print(f"Bright regions: {bright_regions}, Dark regions: {dark_regions}")
        print(f"Brighter signals: {brighter_signals}, Darker signals: {darker_signals}")
        print(f"CV Decision - Brighter: {became_brighter}, Darker: {became_darker}")
    
    return {
        "Brighter": "Yes" if became_brighter else "No",
        "Darker": "Yes" if became_darker else "No",
        "brighter_confidence": brighter_confidence,
        "darker_confidence": darker_confidence,
        "l_mean_diff": l_mean_diff,
        "gray_mean_diff": gray_mean_diff,
        "l_pct_change": l_pct_change,
        "brighter_signals": brighter_signals,
        "darker_signals": darker_signals,
    }

# -------------------------
# Core Logic
# -------------------------
def detect_lighting_changes(
    img_before_path: str,
    img_after_path: str,
    instruction: str,
    video_chat_model,
    video_chat_tokenizer,
    verbose: bool = False
) -> Dict[str, str]:
    
    # Step 1: CV-based brightness analysis (fast and objective)
    cv_results = cv_detect_brightness_changes(img_before_path, img_after_path, verbose=verbose)
    
    # Step 2: Setup video for VLM
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video_path = tmpf.name
    tmpf.close()
    atexit.register(lambda: os.path.exists(tmp_video_path) and os.remove(tmp_video_path))
    two_images_to_video(img_before_path, img_after_path, tmp_video_path)

    results = {}
    
    # Step 3: Get VLM confirmation with improved prompts
    # -------------------------------------------------------------------------
    # QUESTION 1: Brighter / Overexposure
    # -------------------------------------------------------------------------
    q1 = f"""TASK: Detect if the scene became BRIGHTER between Frame 1 and Frame 2.
Instruction: "{instruction}"

Compare the OVERALL ILLUMINATION and AMBIENT LIGHTING between:
- Frame 1 (Original image, before edit)
- Frame 2 (Edited image, after edit)

FOCUS ON:
1. Background brightness - Is the background more lit up?
2. Surface reflections - Do surfaces appear more reflective/washed out?
3. Ambient light level - Does the overall scene look more illuminated?
4. Shadow intensity - Are shadows lighter/less visible?
5. Highlight areas - Are bright spots more prominent or overexposed?

IGNORE (do NOT count as lighting changes):
- Robot arm appearance/brightness
- Object movement (a white object moving is NOT a lighting change)
- Color changes (focus only on luminosity/brightness)

Did the scene become noticeably BRIGHTER in Frame 2 compared to Frame 1?
(This means the ambient light level increased, not just an object moved)

Reasoning: [Analyze the specific brightness indicators listed above]
Verdict: Yes or No"""

    out1 = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q1, return_history=False,
        max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
    )
    vlm_brighter = _parse_yes_no(out1)
    if verbose: 
        print(f"--- VLM Brighter Check ---\nResponse: {out1.strip()}\nParsed: {vlm_brighter}\n")

    # -------------------------------------------------------------------------
    # QUESTION 2: Darker / Underexposure
    # -------------------------------------------------------------------------
    q2 = f"""TASK: Detect if the scene became DARKER between Frame 1 and Frame 2.
Instruction: "{instruction}"

Compare the OVERALL ILLUMINATION and AMBIENT LIGHTING between:
- Frame 1 (Original image, before edit)
- Frame 2 (Edited image, after edit)

FOCUS ON:
1. Background darkness - Is the background less lit?
2. Shadow intensity - Are shadows darker/more prominent?
3. Ambient light level - Does the overall scene look less illuminated?
4. Detail visibility - Are details harder to see in dark areas?
5. Overall dimness - Does the scene look like the lights were turned down?

IGNORE (do NOT count as lighting changes):
- Robot arm appearance
- Object removal (if an object leaves, it's not a lighting change)
- Cast shadows from object movement

Did the scene become noticeably DARKER in Frame 2 compared to Frame 1?
(This means the ambient light level decreased, not just an object was removed)

Reasoning: [Analyze the specific darkness indicators listed above]
Verdict: Yes or No"""

    out2 = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q2, return_history=False,
        max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_CONFIG
    )
    vlm_darker = _parse_yes_no(out2)
    if verbose: 
        print(f"--- VLM Darker Check ---\nResponse: {out2.strip()}\nParsed: {vlm_darker}\n")

    # Step 4: Ensemble CV + VLM results
    # CV is more reliable for detecting actual pixel-level brightness changes
    # VLM can help with context but is less sensitive to subtle changes
    
    if ENSEMBLE_MODE == "cv_primary":
        # CV primary: CV "Yes" wins, VLM can confirm CV "No" cases
        results["Brighter"] = cv_results["Brighter"]
        results["Darker"] = cv_results["Darker"]
        
        # If CV says No but VLM says Yes with high confidence signal, reconsider
        # (but we trust CV more since it's objective)
        if cv_results["Brighter"] == "No" and vlm_brighter == "Yes":
            # Only override if CV was close to threshold
            if cv_results.get("brighter_signals", 0) >= 1:
                results["Brighter"] = "Yes"
        
        if cv_results["Darker"] == "No" and vlm_darker == "Yes":
            if cv_results.get("darker_signals", 0) >= 1:
                results["Darker"] = "Yes"
                
    elif ENSEMBLE_MODE == "vlm_primary":
        # VLM primary with CV override
        results["Brighter"] = vlm_brighter
        results["Darker"] = vlm_darker
        
        # CV override: if CV confidently detects change, trust it
        if cv_results["brighter_confidence"] >= 0.5:
            results["Brighter"] = cv_results["Brighter"]
        if cv_results["darker_confidence"] >= 0.5:
            results["Darker"] = cv_results["Darker"]
    
    else:
        # Default: majority vote / OR logic
        results["Brighter"] = "Yes" if (cv_results["Brighter"] == "Yes" or vlm_brighter == "Yes") else "No"
        results["Darker"] = "Yes" if (cv_results["Darker"] == "Yes" or vlm_darker == "Yes") else "No"
    
    if verbose:
        print(f"=== Final Ensemble Decision ===")
        print(f"CV: Brighter={cv_results['Brighter']}, Darker={cv_results['Darker']}")
        print(f"VLM: Brighter={vlm_brighter}, Darker={vlm_darker}")
        print(f"Final: Brighter={results['Brighter']}, Darker={results['Darker']}")

    return results

# -------------------------
# Loader
# -------------------------
def load_videochat_model_and_tokenizer(model_name=MODEL_NAME):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, 
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.mm_llm_compress = False
    return tokenizer, model

# -------------------------
# Run Tests
# -------------------------
if __name__ == "__main__":
    
    test_cases = [
        {
            "before": "lgie-eval-data/input_3.jpg",
            "after": "lgie-eval-data/output_3_1.png",
            "instruction": "The robotic arm puts the sushi on top of the omelette",
            "expected_became_brighter": "No",
            "expected_became_darker": "No",
            "description": "Hallucinations are there, but no lighting changes"
        },
        {
            "before": "lgie-eval-data/input_3.jpg",
            "after": "lgie-eval-data/output_3_2.png",
            "instruction": "The robotic arm puts the omelette on top of the cabbage",
            "expected_became_brighter": "No",
            "expected_became_darker": "No",
            "description": "Instruction mismatch, but no lighting changes"
        },
        
        {
          "before": "lgie-eval-data/original_image/oi_115.png",
          "after": "lgie-eval-data/r89_gemini_output/out_115.png",
          "instruction": "take broccoli out of pan",
          "expected_became_brighter": "Yes",
          "expected_became_darker": "No",
          "description": "Instruction followed but unexpectedly scene became noticeably brighter",
          "id": 115
        },
        {
          "before": "lgie-eval-data/original_image/oi_116.png",
          "after": "lgie-eval-data/r89_gemini_output/out_116.png",
          "instruction": "take lid off pot or pan",
          "expected_became_brighter": "No",
          "expected_became_darker": "Yes",
          "description": "Instruction followed but unexpectedly scene became significantly darker",
          "id": 116
        },
        {
          "before": "lgie-eval-data/original_image/oi_117.png",
          "after": "lgie-eval-data/r89_gemini_output/out_117.png",
          "instruction": "turn lever vertical to front",
          "expected_became_brighter": "No",
          "expected_became_darker": "Yes",
          "description": "Instruction mismatch, and scene became dimmer.",
          "id": 117
        },
        {
          "before": "lgie-eval-data/original_image/oi_118.png",
          "after": "lgie-eval-data/r89_gemini_output/out_118.png",
          "instruction": "put carrot on plate",
          "expected_became_brighter": "No",
          "expected_became_darker": "No",
          "description": "Instruction mismatch, but no lighting changes",
          "id": 118
        },
        {
          "before": "lgie-eval-data/original_image/oi_119.png",
          "after": "lgie-eval-data/r89_gemini_output/out_119.png",
          "instruction": "put banana in pot cardboard fence",
          "expected_became_brighter": "No",
          "expected_became_darker": "No",
          "description": "Perfect edit -- Instruction followed, and no lighting changes",
          "id": 119
        },
        {
          "before": "lgie-eval-data/original_image/oi_120.png",
          "after": "lgie-eval-data/r89_gemini_output/out_120.png",
          "instruction": "put corn in pot which is in sink",
          "expected_became_brighter": "No",
          "expected_became_darker": "Yes",
          "description": "Instruction followed, but no environment lighting becamer dimmer",
          "id": 120
        },
        {
          "before": "lgie-eval-data/original_image/oi_121.png",
          "after": "lgie-eval-data/r89_gemini_output/out_121.png",
          "instruction": "pour almonds in pot",
          "expected_became_brighter": "No",
          "expected_became_darker": "Yes",
          "description": "Instruction followes, but scene became darker.",
          "id": 121
        },
        {
          "before": "lgie-eval-data/original_image/oi_122.png",
          "after": "lgie-eval-data/r89_gemini_output/out_122.png",
          "instruction": "put pepper in pot or pan",
          "expected_became_brighter": "Yes",
          "expected_became_darker": "No",
          "description": "Lighting became noticeably brighter and more illuminated.",
          "id": 122
        },
        {
          "before": "lgie-eval-data/original_image/oi_123.png",
          "after": "lgie-eval-data/r89_gemini_output/out_123.png",
          "instruction": "put broccoli in pot",
          "expected_became_brighter": "Yes",
          "expected_became_darker": "No",
          "description": "Instruction followed, but Lighting became dramatically brighter and more illuminated.",
          "id": 123
        },
        {
          "before": "lgie-eval-data/original_image/oi_124.png",
          "after": "lgie-eval-data/r89_gemini_output/out_124.png",
          "instruction": "put broccoli in pot",
          "expected_became_brighter": "Yes",
          "expected_became_darker": "No",
          "description": "Instruction mismatch, and lighting brighter.",
          "id": 124
        },
        {
          "before": "lgie-eval-data/original_image/oi_125.png",
          "after": "lgie-eval-data/r89_gemini_output/out_125.png",
          "instruction": "open small4fbox flaps",
          "expected_became_brighter": "No",
          "expected_became_darker": "No",
          "description": "Instruction mismatch, but no lighting changes",
          "id": 125
        },
        {
          "before": "lgie-eval-data/original_image/oi_126.png",
          "after": "lgie-eval-data/r89_gemini_output/out_126.png",
          "instruction": "close large4fbox flaps",
          "expected_became_brighter": "No",
          "expected_became_darker": "No",
          "description": "Instruction mismatch, but no lighting changes",
          "id": 126
        },
        {
          "before": "lgie-eval-data/original_image/oi_127.png",
          "after": "lgie-eval-data/r89_gemini_output/out_127.png",
          "instruction": "fold cloth in half",
          "expected_became_brighter": "Yes",
          "expected_became_darker": "No",
          "description": "Instruction mismatch, and lighting became dramatically bright.",
          "id": 127
        },
        {
          "before": "lgie-eval-data/original_image/oi_128.png",
          "after": "lgie-eval-data/r89_gemini_output/out_128.png",
          "instruction": "close brown1fbox flap",
          "expected_became_brighter": "Yes",
          "expected_became_darker": "No",
          "description": "Instruction mismatch, and lighting became dramatically bright.",
          "id": 128
        },
        {
          "before": "lgie-eval-data/original_image/oi_129.png",
          "after": "lgie-eval-data/r89_gemini_output/out_129.png",
          "instruction": "flip pot upright in sink",
          "expected_became_brighter": "Yes",
          "expected_became_darker": "No",
          "description": "Instruction mismatch, and lighting became dramatically bright.",
          "id": 129
        },
        {
          "before": "lgie-eval-data/original_image/oi_130.png",
          "after": "lgie-eval-data/r89_gemini_output/out_130.png",
          "instruction": "put bowl on plate",
          "expected_became_brighter": "Yes",
          "expected_became_darker": "No",
          "description": "Instruction mismatch, and lighting became noticeably brighter.",
          "id": 130
        },
        {
          "before": "lgie-eval-data/original_image/oi_131.png",
          "after": "lgie-eval-data/r89_gemini_output/out_131.png",
          "instruction": "pick up scissors and put into drawer",
          "expected_became_brighter": "No",
          "expected_became_darker": "Yes",
          "description": "Instruction mismatch, and lighting became dimmer.",
          "id": 131
        },
        {
          "before": "lgie-eval-data/original_image/oi_132.png",
          "after": "lgie-eval-data/r89_gemini_output/out_132.png",
          "instruction": "turn faucet front to left",
          "expected_became_brighter": "No",
          "expected_became_darker": "Yes",
          "description": "Instruction mismatch, and lighting became noticeably darker.",
          "id": 132
        },
        {
          "before": "lgie-eval-data/original_image/oi_133.png",
          "after": "lgie-eval-data/r89_gemini_output/out_133.png",
          "instruction": "put lid on pot or pan",
          "expected_became_brighter": "No",
          "expected_became_darker": "Yes",
          "description": "Instruction mismatch, and lighting became extremely dim.",
          "id": 133
        },
        {
          "before": "lgie-eval-data/original_image/oi_134.png",
          "after": "lgie-eval-data/r89_gemini_output/out_134.png",
          "instruction": "put lid on pot",
          "expected_became_brighter": "No",
          "expected_became_darker": "Yes",
          "description": "Instruction mismatch, and lighting became extremely dim.",
          "id": 134
        },
        {
          "before": "lgie-eval-data/original_image/oi_135.png",
          "after": "lgie-eval-data/r89_gemini_output/out_135.png",
          "instruction": "open oven",
          "expected_became_brighter": "No",
          "expected_became_darker": "Yes",
          "description": "Instruction mismatch, and lighting became dimmer..",
          "id": 135
        },
        {
          "before": "lgie-eval-data/original_image/oi_136.png",
          "after": "lgie-eval-data/r89_gemini_output/out_136.png",
          "instruction": "pick up pan from stove",
          "expected_became_brighter": "Yes",
          "expected_became_darker": "No",
          "description": "Instruction mismatch, and lighting became dramatically bright.",
          "id": 136
        },
        {
          "before": "lgie-eval-data/original_image/oi_137.png",
          "after": "lgie-eval-data/r89_gemini_output/out_137.png",
          "instruction": "turn lever vertical to front",
          "expected_became_brighter": "Yes",
          "expected_became_darker": "No",
          "description": "Instruction mismatch, and lighting became noticeably bright.",
          "id": 137
        }

    ]

    # Load model & tokenizer once (heavy)
    tokenizer, model = load_videochat_model_and_tokenizer()

    total = len(test_cases)
    score_sum = 0.0

    for i, tc in enumerate(test_cases, start=1):
        preds = detect_lighting_changes(
            tc["before"], tc["after"], tc["instruction"],
            model, tokenizer,
            verbose=False
        )

        # normalize for comparison (case-insensitive)
        p_brighter = preds.get("Brighter", "No").strip().title()
        p_darker = preds.get("Darker", "No").strip().title()

        e_brighter = tc["expected_became_brighter"].strip().title()
        e_darker = tc["expected_became_darker"].strip().title()

        # compute per-test correctness (each question worth 1/2)
        correct = 0
        correct += 1 if p_brighter == e_brighter else 0
        correct += 1 if p_darker == e_darker else 0

        per_case_score = correct / 2.0
        score_sum += per_case_score

        print(
            f"Test {i}: {tc['description']} -> "
            f"Predictions: Brighter={p_brighter}, Darker={p_darker} | "
            f"Expected: Brighter={e_brighter}, Darker={e_darker} | "
            f"score={per_case_score:.3f}"
        )

        # --- Augmentation tests ---
        # Create brighter and darker versions of before/after images
        tmp_bright_before = f"/tmp/bright_before_{i}.jpg"
        tmp_bright_after = f"/tmp/bright_after_{i}.jpg"
        tmp_dark_before = f"/tmp/dark_before_{i}.jpg"
        tmp_dark_after = f"/tmp/dark_after_{i}.jpg"
        augment_brightness(tc["before"], 1.3, tmp_bright_before)
        augment_brightness(tc["after"], 1.3, tmp_bright_after)
        augment_brightness(tc["before"], 0.7, tmp_dark_before)
        augment_brightness(tc["after"], 0.7, tmp_dark_after)

        # Test on brighter images
        preds_bright = detect_lighting_changes(
            tmp_bright_before, tmp_bright_after, tc["instruction"],
            model, tokenizer, verbose=False)
        print(
            f"Test {i} (Brighter Aug): "
            f"Predictions: Brighter={preds_bright.get('Brighter','No')}, Darker={preds_bright.get('Darker','No')}")

        # Test on darker images
        preds_dark = detect_lighting_changes(
            tmp_dark_before, tmp_dark_after, tc["instruction"],
            model, tokenizer, verbose=False)
        print(
            f"Test {i} (Darker Aug): "
            f"Predictions: Brighter={preds_dark.get('Brighter','No')}, Darker={preds_dark.get('Darker','No')}")

    success_score = score_sum / total
    print(f"\nFinal success_score = {success_score:.6f} ({success_score * 100:.2f}%)")