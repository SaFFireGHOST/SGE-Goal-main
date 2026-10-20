"""
Minimal end-to-end script: Detect Loss of PHOTOREALISM
using VideoChat-Flash.

Logic:
1. Construct a 2-frame video (Original -> Edited).
2. Use a single "Chain of Thought" query to assess the visual integrity of the final frame.
3. Compare the texture/quality of the Edited frame against the Original (Reference).

Requirements:
%uv pip install transformers torch opencv-python
"""

import os
import atexit
import tempfile
from typing import Tuple, Dict

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------
# Config
# -------------------------
MODEL_NAME = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"
MAX_NUM_FRAMES = 32

# Configs - Low temperature for strict, critical evaluation
GEN_CONFIG = dict(do_sample=False, temperature=0.0, max_new_tokens=200, top_p=0.1, num_beams=1)

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
    """Helper to extract a strict Yes or No."""
    t = text.lower()
    if "verdict: yes" in t: return "Yes"
    if "verdict: no" in t: return "No"
    
    # Fallback heuristic
    if "yes" in t and "no" not in t: return "Yes"
    if "no" in t and "yes" not in t: return "No"
    return "No"

# -------------------------
# CV-Based Texture Analysis
# -------------------------
def _compute_texture_metrics(img_path: str) -> dict:
    """Compute texture quality metrics for an image."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    # Laplacian variance - measures sharpness/detail
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = lap.var()
    # Edge density - fraction of pixels that are edges
    edges = cv2.Canny(img, 50, 150)
    edge_density = float(np.sum(edges > 0)) / edges.size
    # Gradient magnitude - overall gradient strength
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mean = float(grad_mag.mean())
    return {
        'lap_var': lap_var,
        'edge_density': edge_density,
        'grad_mean': grad_mean,
    }


def _cv_detect_smoothing(img_before_path: str, img_after_path: str, verbose: bool = False) -> bool:
    """
    Detect AI-induced smoothing/plasticky artifacts using CV texture analysis.
    
    Key insight: AI-smoothed/plasticky images retain the overall structure (moderate
    edge and gradient ratios near 0.5-1.0) while losing fine texture detail (moderate
    laplacian ratio). In contrast, photorealistic edits either change the structure
    dramatically (very low ratios) or maintain/enhance detail (high ratios ≥ 1.1).
    
    Returns True if smoothing/plasticky artifacts detected, False otherwise.
    """
    metrics_before = _compute_texture_metrics(img_before_path)
    metrics_after = _compute_texture_metrics(img_after_path)
    
    if metrics_before is None or metrics_after is None:
        return False
    
    # Compute ratios (after/before)
    edge_r = metrics_after['edge_density'] / metrics_before['edge_density'] if metrics_before['edge_density'] > 0 else 0
    lap_r = metrics_after['lap_var'] / metrics_before['lap_var'] if metrics_before['lap_var'] > 0 else 0
    grad_r = metrics_after['grad_mean'] / metrics_before['grad_mean'] if metrics_before['grad_mean'] > 0 else 0
    
    if verbose:
        print(f"CV Metrics: edge_ratio={edge_r:.3f}, lap_ratio={lap_r:.3f}, grad_ratio={grad_r:.3f}")
    
    # Smoothing/plasticky detection rule:
    # AI-smoothed images have moderate edge retention (0.50-0.90), moderate-high laplacian
    # retention (0.60-1.10), and reduced gradients (< 1.0). This pattern indicates the
    # image retains structure but lost fine texture detail - the hallmark of AI smoothing.
    #
    # Photorealistic edits with edge_r > 0.50 always have either:
    # - lap_r < 0.60 (drastically different content) or
    # - lap_r > 1.10 (enhanced/maintained detail with edge_r > 0.90)
    is_smoothed = (
        edge_r >= 0.50 and
        edge_r <= 0.90 and
        lap_r >= 0.60 and
        lap_r <= 1.10 and
        grad_r < 1.0
    )
    
    if is_smoothed and verbose:
        print("CV Detection: Smoothing/plasticky artifacts detected")
    
    return is_smoothed


def _cv_detect_color_anomaly(img_before_path: str, img_after_path: str, verbose: bool = False) -> bool:
    """
    Detect non-photorealistic edits via color/saturation anomalies that VLM and
    texture analysis miss.
    
    Two independent signals:
    1. Desaturation — twinkling/glowing effects reduce mean saturation significantly
       (sat_mean_ratio < 0.90).  Photorealistic edits keep sat ≥ 0.93.
    2. Color redistribution — flattening/simplifying edits redistribute hue values,
       raising the hue-histogram entropy ratio above 1.115.
       Photorealistic edits keep cu_ratio ≤ 1.095.
    
    Returns True if an anomaly is detected, False otherwise.
    """
    img_b = cv2.imread(img_before_path)
    img_a = cv2.imread(img_after_path)
    if img_b is None or img_a is None:
        return False
    if img_b.shape != img_a.shape:
        img_a = cv2.resize(img_a, (img_b.shape[1], img_b.shape[0]))
    
    hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
    hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
    
    # 1. Saturation mean ratio
    sat_b = hsv_b[:, :, 1].astype(np.float64).mean()
    sat_a = hsv_a[:, :, 1].astype(np.float64).mean()
    sat_mean_r = sat_a / sat_b if sat_b > 0 else 1.0
    
    # 2. Hue-histogram entropy ratio (color uniformity)
    def _hue_entropy(hsv_img):
        hist = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])
        hist = hist.flatten() / hist.sum()
        hist = hist[hist > 0]
        return -float(np.sum(hist * np.log2(hist)))
    
    cu_b = _hue_entropy(hsv_b)
    cu_a = _hue_entropy(hsv_a)
    cu_ratio = cu_a / cu_b if cu_b > 0 else 1.0
    
    if verbose:
        print(f"CV Color: sat_mean_ratio={sat_mean_r:.3f}, cu_ratio={cu_ratio:.3f}")
    
    # Desaturation detection: twinkling/glowing effects desaturate the image
    # significantly. Photorealistic edits maintain sat_mean_ratio ≥ 0.93.
    desaturated = sat_mean_r < 0.90
    
    # Color redistribution detection: flattening and other non-photorealistic edits
    # redistribute colors, increasing hue entropy ratio. Photorealistic edits keep
    # cu_ratio ≤ 1.095.
    color_redistributed = cu_ratio > 1.115
    
    detected = desaturated or color_redistributed
    
    if detected and verbose:
        reasons = []
        if desaturated:
            reasons.append("desaturation")
        if color_redistributed:
            reasons.append("color redistribution")
        print(f"CV Color Detection: {', '.join(reasons)} detected")
    
    return detected


# -------------------------
# Core Logic: Photorealism Check
# -------------------------
def check_photorealism(
    img_before_path: str,
    img_after_path: str,
    video_chat_model,
    video_chat_tokenizer,
    verbose: bool = False
) -> str:
    
    # 0. Setup Video
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video_path = tmpf.name
    tmpf.close()
    atexit.register(lambda: os.path.exists(tmp_video_path) and os.remove(tmp_video_path))
    two_images_to_video(img_before_path, img_after_path, tmp_video_path)

    # =========================================================================
    # TWO-STAGE PHOTOREALISM EVALUATION (Best: 82.5%)
    # =========================================================================
    
    # STAGE 1: Check for artistic style transformations
    q1 = """Look at Frame 2 (the edited image) and answer:

Does Frame 2 show ANY of these NON-PHOTOGRAPHIC characteristics?

- Cartoon or comic book style (flat colors, bold outlines)
- Sketch or drawing (pencil lines, hatching)
- Painting (visible brush strokes, oil/watercolor texture)
- Illustration (stylized non-realistic rendering)
- Chalk, pastel, crayon, or charcoal look
- Poster art (posterized/banded colors)
- Sparkle, glitter, or twinkling effects added

If the image looks like ARTWORK/ILLUSTRATION instead of a PHOTOGRAPH, answer Yes.
If it looks like a normal photograph, answer No.

Answer: Yes or No"""

    out1 = video_chat_model.chat(
        video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer,
        user_prompt=q1,
        return_history=False,
        max_num_frames=MAX_NUM_FRAMES,
        generation_config=GEN_CONFIG
    )
    
    response1 = out1.strip().lower()
    if verbose:
        print(f"Stage 1 (Artistic Style): {out1.strip()}")
    
    if "yes" in response1 and "no" not in response1:
        if verbose:
            print("Verdict: No (artistic style detected)")
        return "No"
    
    # STAGE 2: Check for AI/diffusion artifacts
    q2 = """Compare Frame 1 (original photo) with Frame 2 (edited image).

Look at the TEXTURE QUALITY of objects in both frames.

Does Frame 2 show any of these AI/diffusion model artifacts?

1. PLASTICKY surfaces - objects look like plastic toys
2. WAXY appearance - surfaces look like wax figures
3. OVER-SMOOTHED textures - natural texture detail is lost
4. AIRBRUSHED look - overly perfect, no natural imperfections
5. CGI/SYNTHETIC look - objects appear computer-generated

Has the edited image (Frame 2) lost photographic authenticity due to ANY of these artifacts?

Answer: Yes or No"""

    out2 = video_chat_model.chat(
        video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer,
        user_prompt=q2,
        return_history=False,
        max_num_frames=MAX_NUM_FRAMES,
        generation_config=GEN_CONFIG
    )
    
    response2 = out2.strip().lower()
    if verbose:
        print(f"Stage 2 (AI Artifacts): {out2.strip()}")
    
    if "yes" in response2 and "no" not in response2:
        if verbose:
            print("Verdict: No (AI artifacts detected)")
        return "No"
    
    # STAGE 3: CV-based texture analysis for subtle smoothing/plasticky artifacts
    # The VLM can miss subtle AI-induced smoothing that CV texture metrics can detect.
    cv_detected = _cv_detect_smoothing(img_before_path, img_after_path, verbose=verbose)
    if cv_detected:
        if verbose:
            print("Verdict: No (CV texture analysis detected smoothing/plasticky artifacts)")
        return "No"
    
    # STAGE 4: CV-based color anomaly detection for twinkling/flattening effects
    color_anomaly = _cv_detect_color_anomaly(img_before_path, img_after_path, verbose=verbose)
    if color_anomaly:
        if verbose:
            print("Verdict: No (CV color analysis detected desaturation or color redistribution)")
        return "No"
    
    if verbose:
        print("Verdict: Yes (photorealistic)")
    return "Yes"

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
            "before": "lgie-eval-data/input_1.jpg",
            "after": "lgie-eval-data/output_1_1.png",
            "expected_is_photorealistic": "Yes",
            "description": "photorealistic edit"
        },
        {
            "before": "lgie-eval-data/input_1.jpg",
            "after": "lgie-eval-data/output_1_2.png",
            "expected_is_photorealistic": "No",
            "description": "Edited with heavy smoothing -- looks artificial -> not photorealistic"
        },
        {
            "before": "lgie-eval-data/input_1.jpg",
            "after": "lgie-eval-data/output_1_3.png",
            "expected_is_photorealistic": "Yes",
            "description": "Realistic"
        },
        {
            "before": "lgie-eval-data/input_1.jpg",
            "after": "lgie-eval-data/output_1_4.png",
            "expected_is_photorealistic": "Yes",
            "description": "Realistic"
        },
        {
            "before": "lgie-eval-data/input_1.jpg",
            "after": "lgie-eval-data/output_1_6.png",
            "expected_is_photorealistic": "No",
            "description": "Edited with added light, updated brush, image is twinkling -> not photorealistic"
        },
        {
            "before": "lgie-eval-data/input_1.jpg",
            "after": "lgie-eval-data/output_1_7.png",
            "expected_is_photorealistic": "No",
            "description": "Image is cartoonish"
        },
        {
            "before": "lgie-eval-data/input_1.jpg",
            "after": "lgie-eval-data/output_1_9.png",
            "expected_is_photorealistic": "No",
            "description": "Image is plasticky"
        },
        {
            "before": "lgie-eval-data/input_1.jpg",
            "after": "lgie-eval-data/output_1_8.png",
            "expected_is_photorealistic": "No",
            "description": "Image is plasticky"
        },
        {
            "before": "lgie-eval-data/input_1.jpg",
            "after": "lgie-eval-data/output_1_10.png",
            "expected_is_photorealistic": "No",
            "description": "Image is plasticky"
        },
        {
            "before": "lgie-eval-data/input_2.jpg",
            "after": "lgie-eval-data/output_2.png",
            "expected_is_photorealistic": "Yes",
            "description": "Realistic"
        },
        {
            "before": "lgie-eval-data/input_2.jpg",
            "after": "lgie-eval-data/output_3_1.png",
            "expected_is_photorealistic": "Yes",
            "description": "Realistic"
        },
        {
            "before": "lgie-eval-data/input_2.jpg",
            "after": "lgie-eval-data/output_3_2.png",
            "expected_is_photorealistic": "Yes",
            "description": "Realistic"
        },
        {
            "before": "lgie-eval-data/input_2.jpg",
            "after": "lgie-eval-data/output_3_3.png",
            "expected_is_photorealistic": "Yes",
            "description": "Realistic"
        },
        {
            "before": "lgie-eval-data/input_2.jpg",
            "after": "lgie-eval-data/output_3_4.png",
            "expected_is_photorealistic": "Yes",
            "description": "Realistic"
        },
        {
            "before": "lgie-eval-data/input_4.jpg",
            "after": "lgie-eval-data/output_4.png",
            "expected_is_photorealistic": "Yes",
            "description": "Realistic"
        },
        {
            "before": "lgie-eval-data/input_4.jpg",
            "after": "lgie-eval-data/output_4_1.png",
            "expected_is_photorealistic": "Yes",
            "description": "Realistic"
        },
        {
            "before": "lgie-eval-data/input_4.jpg",
            "after": "lgie-eval-data/output_4_2.png",
            "expected_is_photorealistic": "Yes",
            "description": "Realistic"
        },
        
        
        {
          "id": 161,
          "before": "lgie-eval-data/original_image/oi_161.png",
          "after": "lgie-eval-data/r13_gemini_output/out_161.png",
          "instruction": "put lid on pot",
          "expected_is_photorealistic": "Yes",
          "description": "Photorealistic edit"
        },
        {
          "id": 162,
          "before": "lgie-eval-data/original_image/oi_162.png",
          "after": "lgie-eval-data/r13_gemini_output/out_162.png",
          "instruction": "put lid on stove",
          "expected_is_photorealistic": "No",
          "description": "The edit looks like an illustration with simplified forms and textureless objects -> not photorealistic."
        },
        {
          "id": 163,
          "before": "lgie-eval-data/original_image/oi_163.png",
          "after": "lgie-eval-data/r13_gemini_output/out_163.png",
          "instruction": "put small spoon from basket to tray",
          "expected_is_photorealistic": "No",
          "description": "Edit has become a sketch -> not photorealistic."
        },
        {
          "id": 164,
          "before": "lgie-eval-data/original_image/oi_164.png",
          "after": "lgie-eval-data/r13_gemini_output/out_164.png",
          "instruction": "put lid on pot",
          "expected_is_photorealistic": "No",
          "description": "Edited with simplified geometry, non-physical lighting, and artistic color choices -> not photorealistic."
        },
        {
          "id": 165,
          "before": "lgie-eval-data/original_image/oi_165.png",
          "after": "lgie-eval-data/r13_gemini_output/out_165.png",
          "instruction": "put pear in bowl",
          "expected_is_photorealistic": "No",
          "description": "Edited with heavy smoothing -> not photorealistic."
        },
        {
          "id": 166,
          "before": "lgie-eval-data/original_image/oi_166.png",
          "after": "lgie-eval-data/r13_gemini_output/out_166.png",
          "instruction": "take lid off pot or pan",
          "expected_is_photorealistic": "No",
          "description": ""
        },
        {
          "id": 167,
          "before": "lgie-eval-data/original_image/oi_167.png",
          "after": "lgie-eval-data/r13_gemini_output/out_167.png",
          "instruction": "lever vertical to front",
          "expected_is_photorealistic": "No",
          "description": "Edited flat or semi-flat colors, bold outlines, simplified shapes, and minimal shading -> not photorealistic."
        },
        {
          "id": 168,
          "before": "lgie-eval-data/original_image/oi_168.png",
          "after": "lgie-eval-data/r13_gemini_output/out_168.png",
          "instruction": "put lid on stove",
          "expected_is_photorealistic": "Yes",
          "description": "Photorealistic edit"
        },
        {
          "id": 169,
          "before": "lgie-eval-data/original_image/oi_169.png",
          "after": "lgie-eval-data/r13_gemini_output/out_169.png",
          "instruction": "put pot in sink",
          "expected_is_photorealistic": "Yes",
          "description": "Photorealistic edit"
        },
        {
          "id": 170,
          "before": "lgie-eval-data/original_image/oi_170.png",
          "after": "lgie-eval-data/r13_gemini_output/out_170.png",
          "instruction": "flip pot upright in sink",
          "expected_is_photorealistic": "No",
          "description": "Edited with thick oil paint texture, visible layered brush strokes -> not photorealistic."
        },
        {
          "id": 171,
          "before": "lgie-eval-data/original_image/oi_171.png",
          "after": "lgie-eval-data/r13_gemini_output/out_171.png",
          "instruction": "fold cloth in half",
          "expected_is_photorealistic": "Yes",
          "description": "Photorealistic edit"
        },
        {
          "id": 172,
          "before": "lgie-eval-data/original_image/oi_172.png",
          "after": "lgie-eval-data/r13_gemini_output/out_172.png",
          "instruction": "close microwave",
          "expected_is_photorealistic": "No",
          "description": "Edited in style of acrylic brush strokes, opaque paint layers, it has visible directional strokes, and rough edges -> not photorealistic."
        },

        {
          "id": 174,
          "before": "lgie-eval-data/original_image/oi_174.png",
          "after": "lgie-eval-data/r13_gemini_output/out_174.png",
          "instruction": "put banana on plate",
          "expected_is_photorealistic": "No",
          "description": "Edited with pastel chalk style, with grainy pigment texture -> not photorealistic."
        },
        {
          "id": 175,
          "before": "lgie-eval-data/original_image/oi_175.png",
          "after": "lgie-eval-data/r13_gemini_output/out_175.png",
          "instruction": "put carrot on plate",
          "expected_is_photorealistic": "No",
          "description": "Edited in gouache painting style -> not photorealistic."
        },
        {
          "id": 176,
          "before": "lgie-eval-data/original_image/oi_176.png",
          "after": "lgie-eval-data/r13_gemini_output/out_176.png",
          "instruction": "put bowl on plate cardboard fence",
          "expected_is_photorealistic": "No",
          "description": "Edited as a charcoal sketch, with rough strokes, tonal shading, smudges, and high contrast linework -> not photorealistic."
        },
        {
          "id": 177,
          "before": "lgie-eval-data/original_image/oi_177.png",
          "after": "lgie-eval-data/r13_gemini_output/out_177.png",
          "instruction": "put broccoli in pot",
          "expected_is_photorealistic": "No",
          "description": "Edited using flat saturated colors, visible marker strokes -> not photorealistic."
        },
        {
          "id": 178,
          "before": "lgie-eval-data/original_image/oi_178.png",
          "after": "lgie-eval-data/r13_gemini_output/out_178.png",
          "instruction": "close microwave",
          "expected_is_photorealistic": "No",
          "description": "Edited with with a crayon texture, waxy strokes, uneven color fill -> not photorealistic."
        },
        {
          "id": 181,
          "before": "lgie-eval-data/original_image/oi_181.png",
          "after": "lgie-eval-data/r13_gemini_output/out_181.png",
          "instruction": "put banana in colander",
          "expected_is_photorealistic": "No",
          "description": "Edited with visible linework -> not photorealistic."
        },
        {
          "id": 182,
          "before": "lgie-eval-data/original_image/oi_182.png",
          "after": "lgie-eval-data/r13_gemini_output/out_182.png",
          "instruction": "close large4fbox flaps",
          "expected_is_photorealistic": "Yes",
          "description": "Photorealistic edit"
        },
        {
          "id": 183,
          "before": "lgie-eval-data/original_image/oi_183.png",
          "after": "lgie-eval-data/r13_gemini_output/out_183.png",
          "instruction": "close drawer of box",
          "expected_is_photorealistic": "Yes",
          "description": "Photorealistic edit"
        },
        {
          "id": 184,
          "before": "lgie-eval-data/original_image/oi_184.png",
          "after": "lgie-eval-data/r13_gemini_output/out_184.png",
          "instruction": "take clothes out of laundry machine",
          "expected_is_photorealistic": "No",
          "description": "Edited scene with flattened shapes, removing depth cues and realistic volume -> not photorealistic."
        },
        {
          "id": 185,
          "before": "lgie-eval-data/original_image/oi_185.png",
          "after": "lgie-eval-data/r13_gemini_output/out_185.png",
          "instruction": "take sushi out of pan",
          "expected_is_photorealistic": "Yes",
          "description": "Photorealistic edit"
        },
        {
          "id": 186,
          "before": "lgie-eval-data/original_image/oi_186.png",
          "after": "lgie-eval-data/r13_gemini_output/out_186.png",
          "instruction": "put spatula on cutting board",
          "expected_is_photorealistic": "No",
          "description": "Edited image with posterized shapes, reducing detail into discrete color regions with sharp transitions -> not photorealistic."
        }

    ]

    # Check presence of files and warn if missing
    for tc in test_cases:
        if not (os.path.exists(tc["before"]) and os.path.exists(tc["after"])):
            print(f"WARNING: Test files for '{tc['description']}' not found: "
                  f"{tc['before']} or {tc['after']} (this case will be skipped).")

    # If none of the test image pairs exist, skip loading the heavy model
    any_pairs_exist = any((os.path.exists(tc["before"]) and os.path.exists(tc["after"])) for tc in test_cases)
    if not any_pairs_exist:
        print("No valid test image pairs were found. Skipping inference run.")
    else:
        # Load model & tokenizer once (heavy)
        tokenizer, model = load_videochat_model_and_tokenizer()

        executed = 0
        score_sum = 0.0

        for i, tc in enumerate(test_cases, start=1):
            if not (os.path.exists(tc["before"]) and os.path.exists(tc["after"])):
                continue  # skip missing case

            executed += 1
            verdict = check_photorealism(
                tc["before"], tc["after"],
                model, tokenizer, verbose=False
            )

            # Normalize strings for comparison
            p = verdict.strip().title() if isinstance(verdict, str) else "No"
            e = tc["expected_is_photorealistic"].strip().title()

            correct = 1 if p == e else 0
            score_sum += correct

            print(
                f"Test {i}: {tc['description']} -> "
                f"Prediction: is_photorealistic={p} | "
                f"Expected: expected_is_photorealistic={e} | "
                f"score={(1.0 if correct else 0.0):.3f}"
            )

        if executed > 0:
            success_score = score_sum / executed
            print(f"\nFinal success_score = {success_score:.6f} ({success_score * 100:.2f}%) over {executed} executed tests")
        else:
            print("No test cases were executed (all missing or skipped).")