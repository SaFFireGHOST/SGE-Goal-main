import os

# install dependencies
os.system("pip install transformers==4.40.1")
os.system("pip install av")
os.system("pip install imageio")
os.system("pip install decord")
os.system("pip install opencv-python")
os.system("pip install timm")
os.system("pip install natsort")
# optional
os.system("pip install flash-attn --no-build-isolation")

"""
Enhanced end-to-end script: compute number of HALLUCINATED objects
using VideoChat-Flash.

The key hallucination pattern in robotic manipulation edits:
- "Duplicate Object Hallucination": The VLM creates a COPY of an object to perform
  the manipulation, while leaving the ORIGINAL object in its initial position.

Logic:
1. Extract instruction objects (objects being manipulated).
2. For EACH target object, specifically check if it appears DUPLICATED 
   (original still at initial position + copy at new position).
3. Also scan for any random/unexpected new objects.
4. Carefully verify each candidate to minimize false positives.

Requirements:
  pip install transformers torch opencv-python
"""

import os
import re
import atexit
import tempfile
from typing import Tuple, List, Dict

import cv2
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForZeroShotObjectDetection

# -------------------------
# Deterministic settings to reduce VLM stochasticity
# NOTE: These are kept but weakened to allow natural variance
# while maintaining some reproducibility  
# -------------------------
# torch.backends.cudnn.deterministic = True  # Disabled: locks into suboptimal path
# torch.backends.cudnn.benchmark = False      # Disabled: locks into suboptimal path
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# -------------------------
# Config
# -------------------------
MODEL_NAME = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"
MAX_NUM_FRAMES = 32

# Generation configs - deterministic for consistency
GEN_SHORT = dict(do_sample=False, temperature=0.0, max_new_tokens=100, top_p=0.1, num_beams=1)
GEN_MED = dict(do_sample=False, temperature=0.0, max_new_tokens=200, top_p=0.1, num_beams=1)
GEN_LONG = dict(do_sample=False, temperature=0.0, max_new_tokens=300, top_p=0.1, num_beams=1)

# -------------------------
# Grounding DINO Config
# -------------------------
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GDINO_BOX_THRESHOLD = 0.30
GDINO_TEXT_THRESHOLD = 0.25

# Global Grounding DINO model references
_gdino_processor = None
_gdino_model = None


def load_grounding_dino():
    """Load Grounding DINO model for object detection/counting."""
    global _gdino_processor, _gdino_model
    if _gdino_processor is None:
        print(f"Loading Grounding DINO ({GDINO_MODEL_ID})...")
        _gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
        _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GDINO_MODEL_ID
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        _gdino_model.eval()
        print("Grounding DINO loaded successfully.")
    return _gdino_processor, _gdino_model


def gdino_detect(image_path: str, text_query: str, box_threshold: float = None, text_threshold: float = None):
    """Run Grounding DINO detection on an image.
    
    Args:
        image_path: Path to the image
        text_query: Object description(s), e.g. "banana." or "banana. apple."
        box_threshold: Confidence threshold for boxes (default: GDINO_BOX_THRESHOLD)
        text_threshold: Confidence threshold for text (default: GDINO_TEXT_THRESHOLD)
    
    Returns:
        List of dicts with keys: 'label', 'score', 'box' (x0,y0,x1,y1)
    """
    if box_threshold is None:
        box_threshold = GDINO_BOX_THRESHOLD
    if text_threshold is None:
        text_threshold = GDINO_TEXT_THRESHOLD
    
    processor, model = load_grounding_dino()
    image = Image.open(image_path).convert("RGB")
    
    # Ensure query ends with period
    if not text_query.strip().endswith("."):
        text_query = text_query.strip() + "."
    
    inputs = processor(images=image, text=text_query.lower(), return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )
    
    detections = []
    if results and len(results) > 0:
        result = results[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        # Handle both old ("labels") and new ("text_labels") API
        labels = result.get("text_labels", result.get("labels", []))
        
        for box, score, label in zip(boxes, scores, labels):
            box_list = [round(x, 2) for x in box.tolist()] if hasattr(box, 'tolist') else box
            score_val = round(score.item(), 4) if hasattr(score, 'item') else score
            label_str = str(label) if not isinstance(label, str) else label
            detections.append({
                'label': label_str,
                'score': score_val,
                'box': box_list
            })
    
    return detections


def gdino_count(image_path: str, object_name: str, box_threshold: float = None) -> int:
    """Count instances of an object in an image using Grounding DINO.
    
    Returns the number of detected instances above the threshold.
    """
    detections = gdino_detect(image_path, object_name, box_threshold=box_threshold)
    return len(detections)


def gdino_count_both_frames(img_before: str, img_after: str, object_name: str, 
                             box_threshold: float = None) -> tuple:
    """Count objects in both frames and return (count_f1, count_f2)."""
    c1 = gdino_count(img_before, object_name, box_threshold)
    c2 = gdino_count(img_after, object_name, box_threshold)
    return c1, c2


# -------------------------
# Helpers
# -------------------------
def two_images_to_video(img1_path: str, img2_path: str, out_path: str, fps: int = 1) -> str:
    """Creates a 2-frame video from before/after images."""
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
    """Wrapper for model chat with retry logic."""
    call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if "video_path" not in call_kwargs:
        call_kwargs["video_path"] = ""

    for attempt in range(3):
        try:
            with torch.no_grad():
                return video_chat_model.chat(**call_kwargs)
        except Exception as e:
            if verbose:
                print(f"DEBUG chat attempt {attempt + 1} failed:", repr(e))
            continue
    raise RuntimeError("Model chat failed after retries.")


def _clean_list(text: str, strict_robot_filter: bool = True) -> List[str]:
    """Splits comma/and separated strings into a clean list.
    
    Args:
        text: Raw text containing object names
        strict_robot_filter: If True, filters all robot terms (for instruction parsing)
                             If False, allows "extra/another robotic arm" for hallucination detection
    """
    text = text.lower()
    # Remove common non-object words
    for word in ["none", "nothing", "n/a", "no objects", "no new"]:
        text = text.replace(word, "")
    text = text.replace(".", "").replace("the ", "").replace("a ", "")
    items = re.split(r",|\band\b|\n", text)
    cleaned = []
    for x in items:
        x = x.strip().strip("-").strip("•").strip("*").strip()
        # Filter out noise
        if x and len(x) > 1 and not _is_robot(x, strict=strict_robot_filter) and not _is_noise(x):
            cleaned.append(x)
    return cleaned


def _is_robot(name: str, strict: bool = True) -> bool:
    """Filter to exclude robotic arm references.
    
    When strict=True: filters out all robot-related terms (used for instruction parsing)
    When strict=False: only filters out obvious non-hallucination robot terms
                       (allows "extra robotic arm", "another robot arm" to pass through)
    """
    n = name.lower()
    
    if strict:
        # For instruction object extraction - filter all robot references
        robot_terms = ["robot", "arm", "gripper", "manipulator", "claw", "end-effector", "effector"]
        return any(x in n for x in robot_terms)
    else:
        # For hallucination detection - only filter the expected single robot arm
        # Allow through: "extra", "another", "second", "duplicate" robot arms
        if any(prefix in n for prefix in ["extra ", "another ", "second ", "duplicate ", "new "]):
            return False  # Don't filter - this could be a hallucinated robot arm
        # Only filter simple "robot arm" or "robotic arm" references
        robot_terms = ["robot arm", "robotic arm", "gripper", "end-effector", "effector"]
        return any(x in n for x in robot_terms)


def _is_noise(name: str) -> bool:
    """Filter out common noise/non-object terms."""
    n = name.lower().strip()
    noise_terms = [
        "frame", "image", "video", "scene", "background", "table", "surface",
        "light", "lighting", "reflection", "blur", "pixel", "artifact",
        "change", "movement", "motion", "position", "location", "spot",
        "same", "similar", "identical", "original", "initial", "final"
    ]
    # Very short or very long strings are likely noise
    if len(n) < 2 or len(n) > 50:
        return True
    # Filter out numbered yes/no responses like "1 no", "2 yes", "1. no"
    if re.match(r'^\d+\.?\s*(yes|no|n/a)$', n):
        return True
    # Filter out standalone yes/no
    if n in ['yes', 'no', 'n/a', 'yes/no']:
        return True
    # Filter out meta-text like "new objects are:", "the following:", etc.
    meta_patterns = ["objects are", "following", "listed", "include", "are:"]
    if any(pat in n for pat in meta_patterns):
        return True
    return any(x == n or n == x for x in noise_terms)


def _normalize_object_name(name: str) -> str:
    """Normalize object names for comparison."""
    n = name.lower().strip()
    # Remove articles and common modifiers
    for prefix in ["the ", "a ", "an ", "new ", "second ", "another ", "copy of ", "duplicate "]:
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n.strip()


def _is_human_body_part(name: str) -> bool:
    """Detect if the object is a human body part (shouldn't appear in robot manipulation)."""
    n = name.lower().strip()
    human_parts = ["human hand", "hand", "finger", "fingers", "palm", "human arm", 
                   "person", "human", "wrist", "forearm"]
    # Check if it's clearly a human hand/body part
    for part in human_parts:
        if part in n and "handle" not in n and "second hand" not in n:
            return True
    return False


def _is_special_hallucination(name: str) -> bool:
    """Detect special hallucination types that should always be counted."""
    n = name.lower().strip()
    # Human hands appearing in robot manipulation are always hallucinations
    if _is_human_body_part(n):
        return True
    # Extra robotic arms (when explicitly mentioned as extra/another)
    if any(prefix in n for prefix in ["extra ", "another ", "second ", "duplicate "]):
        if any(term in n for term in ["robot", "arm", "robotic"]):
            return True
    return False


def _extract_objects_from_text(instruction: str) -> List[str]:
    """Extract objects from instruction text using pattern matching.
    Fallback when model returns empty object list."""
    instruction_lower = instruction.lower().strip()
    objects = []
    
    # Common object words to look for
    common_objects = [
        "pan", "pot", "bowl", "plate", "cup", "mug", "knife", "fork", "spoon",
        "spatula", "ladle", "cutting board", "cardboard", "fence",
        "apple", "banana", "orange", "tomato", "carrot", "broccoli", "lettuce",
        "onion", "pepper", "corn", "potato", "sweet potato", "cucumber", "sushi",
        "omelette", "egg", "bread", "cheese",
        "lid", "handle", "detergent", "sponge", "cloth", "clothes",
        "pen", "marker", "key", "allen key", "screwdriver", "tool",
        "ball", "cube", "block", "box", "container",
        "oven", "stove", "sink", "drawer", "cabinet", "machine", "laundry",
    ]
    
    for obj in common_objects:
        if obj in instruction_lower:
            objects.append(obj)
    
    # Also try common patterns
    patterns = [
        r"put\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:in|on|into|onto)",
        r"pick\s+up\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
        r"take\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:off|out|from)",
        r"move\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:to|into|onto)",
        r"flip\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
        r"close\s+(?:the\s+)?(\w+)",
        r"open\s+(?:the\s+)?(\w+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, instruction_lower)
        if match:
            obj = match.group(1).strip()
            if obj and len(obj) > 1 and obj not in objects:
                objects.append(obj)
    
    return objects


def _objects_match(name1: str, name2: str) -> bool:
    """Check if two object names refer to the same object type."""
    n1 = _normalize_object_name(name1)
    n2 = _normalize_object_name(name2)
    # Direct match
    if n1 == n2:
        return True
    # One contains the other
    if n1 in n2 or n2 in n1:
        return True
    return False


def _are_potential_duplicates(name1: str, name2: str) -> bool:
    """Quick check if two object names MIGHT refer to the same physical object.
    This is a pre-filter before asking the model to verify."""
    n1 = _normalize_object_name(name1)
    n2 = _normalize_object_name(name2)
    
    # Direct/substring match - definitely potential duplicates
    if n1 == n2 or n1 in n2 or n2 in n1:
        return True
    
    # Check if one is a modifier-extended version of the other
    # e.g., "detergent" and "detergent fluid", "spoon" and "metal spoon"
    words1 = set(n1.split())
    words2 = set(n2.split())
    if words1 and words2:
        overlap = words1 & words2
        if overlap and (len(overlap) >= len(words1) - 1 or len(overlap) >= len(words2) - 1):
            return True
    
    # Check for similar visual characteristics that might cause confusion
    # These are candidates for model verification, not automatic deduplication
    # NOTE: Keep shape-distinct objects separate (ball ≠ tomato, ball ≠ cube)
    potential_confusion_pairs = [
        # Similar colors/shapes that model might confuse
        ({"egg", "omelette", "omelet"}, {"egg", "omelette", "omelet"}),
        ({"tomato", "cherry"}, {"tomato", "cherry"}),  # Small red round fruits
        ({"red ball", "red sphere"}, {"red ball", "red sphere"}),  # Spherical objects
        ({"blue ball", "blue sphere"}, {"blue ball", "blue sphere"}),  # Spherical objects
        ({"orange", "orange ball"}, {"orange", "orange ball"}),
        # Yellow elongated objects - banana and yellow brush can be confused
        ({"banana", "yellow brush", "brush"}, {"banana", "yellow brush", "brush"}),
    ]
    
    for group1, group2 in potential_confusion_pairs:
        n1_matches = any(term in n1 or n1 in term for term in group1)
        n2_matches = any(term in n2 or n2 in term for term in group2)
        if n1_matches and n2_matches:
            return True
    
    return False


def _model_verify_same_object(obj1: str, obj2: str, video_path: str, 
                               model, tokenizer, verbose: bool = False) -> bool:
    """Use the video chat model to verify if two detected objects are the same physical object."""
    
    query = f"""TASK: Determine if two detected objects refer to the SAME physical object in the scene.

Detected object 1: "{obj1}"
Detected object 2: "{obj2}"

Look at Frame 2 (the after image) carefully.

Question: In Frame 2, are "{obj1}" and "{obj2}" referring to the SAME physical object, 
or are they TWO DIFFERENT physical objects in the scene?

Consider:
- If there's only ONE object that could be called either "{obj1}" or "{obj2}", they are the SAME.
- If there are TWO SEPARATE objects (one that is "{obj1}" and another that is "{obj2}"), they are DIFFERENT.

Answer with exactly one of:
SAME_OBJECT - They refer to the same physical object in the scene
DIFFERENT_OBJECTS - They are two distinct physical objects in the scene

ANSWER: [SAME_OBJECT or DIFFERENT_OBJECTS]"""

    out = _safe_chat_call(
        model, verbose=verbose, video_path=video_path,
        tokenizer=tokenizer, user_prompt=query, return_history=False,
        max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_SHORT
    )
    
    reply = out.strip().lower()
    if verbose:
        print(f"  [Dedup check] '{obj1}' vs '{obj2}': {out.strip()}")
    
    # Check for SAME_OBJECT verdict
    if "same_object" in reply or "same object" in reply:
        return True
    if "different_object" in reply or "different object" in reply:
        return False
    
    # Default: if unclear, assume different (don't merge)
    return False


def _deduplicate_hallucinations_with_model(hallucinations: List[str], instruction_objects: List[str],
                                            video_path: str, model, tokenizer, 
                                            verbose: bool = False) -> List[str]:
    """Remove duplicates that refer to the same physical object, using model verification."""
    if len(hallucinations) <= 1:
        return hallucinations
    
    # Sort: instruction objects first, then by specificity (longer names often more specific)
    def priority(obj):
        is_instruction = any(_objects_match(obj, inst) for inst in instruction_objects)
        return (0 if is_instruction else 1, -len(obj))
    
    sorted_halls = sorted(hallucinations, key=priority)
    
    # Keep track of which objects to keep
    kept = []
    
    for obj in sorted_halls:
        is_duplicate = False
        
        for kept_obj in kept:
            # First quick check: are these potential duplicates?
            if _are_potential_duplicates(obj, kept_obj):
                # Ask the model to verify
                if _model_verify_same_object(obj, kept_obj, video_path, model, tokenizer, verbose):
                    is_duplicate = True
                    if verbose:
                        print(f"  [Dedup] Removing '{obj}' - same object as '{kept_obj}'")
                    break
        
        if not is_duplicate:
            kept.append(obj)
    
    return kept


# -------------------------
# Core Logic
# -------------------------
def num_unmentioned_hallucinations(
    img_before_path: str,
    img_after_path: str,
    instruction: str,
    video_chat_model,
    video_chat_tokenizer,
    verbose: bool = False
) -> Tuple[int, List[str]]:
    """
    Detects hallucinated objects in an edited image.
    
    Returns:
        Tuple of (count of hallucinations, list of hallucinated object names)
    """
    
    # 0. Setup Video
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video_path = tmpf.name
    tmpf.close()
    atexit.register(lambda: os.path.exists(tmp_video_path) and os.remove(tmp_video_path))
    two_images_to_video(img_before_path, img_after_path, tmp_video_path)

    # Track hallucinated objects with their confidence level
    # HIGH confidence = count explicitly increased, skip final verification
    # LOW confidence = detected via other signals, needs verification
    hallucinated_objects: dict = {}  # {object_name: 'HIGH' or 'LOW'}
    step2_counts: dict = {}  # {object_name: (count_f1, count_f2)} from Step 2
    step3a_counts: dict = {}  # {object_name: (count_f1, count_f2)} from Step 3a (background objects)
    inventory_contradictions: set = set()  # objects where count=0 in F1 but found in inventory

    # Parse instruction to identify destination objects
    # In "put X in/on Y", Y is the destination and shouldn't be duplicated
    destination_objects: set = set()
    instr_lower = instruction.lower()
    dest_preps = [' in ', ' on ', ' into ', ' onto ', ' inside ', ' on top of ']
    for prep in dest_preps:
        if prep in instr_lower:
            after_prep = instr_lower.split(prep)[-1].strip()
            # Remove trailing punctuation and common suffixes
            after_prep = re.sub(r'[.,;!?]+$', '', after_prep).strip()
            # Remove articles
            after_prep = re.sub(r'^(the|a|an)\s+', '', after_prep).strip()
            if after_prep:
                destination_objects.add(after_prep)

    # =========================================================================
    # STEP 1: Extract Instruction Objects (ALL objects involved)
    # =========================================================================
    q1 = f"""TASK: Extract ALL objects mentioned in this robotic manipulation instruction.

Instruction: "{instruction}"

List ALL physical objects mentioned in the instruction, including:
- Objects being moved/picked up
- Objects being placed onto or into (destination objects)
- Any other objects referenced

Do NOT include: the robot, arm, gripper, or abstract locations like "left" or "right".

For example:
- "put the apple on the plate" → apple, plate
- "move the cup next to the bowl" → cup, bowl
- "place sushi on top of omelette" → sushi, omelette

Output format: comma-separated list of object names only.
If no specific objects mentioned, output: none"""

    out1 = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q1, return_history=False,
        max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_SHORT
    )
    
    instruction_objects = _clean_list(out1.strip())
    
    # Fallback: If model returns empty, try to extract objects from instruction text directly
    if not instruction_objects:
        instruction_objects = _extract_objects_from_text(instruction)
        if verbose:
            print(f"Step 1 - Using fallback text extraction: {instruction_objects}")
    
    if verbose:
        print(f"Step 1 - Instruction Objects: {instruction_objects}")

    # =========================================================================
    # STEP 1b: Inventory ALL objects in Frame 1 (for duplication detection)
    # =========================================================================
    q1b = """TASK: Complete Object Inventory of Frame 1

Look at Frame 1 (the BEFORE image) and list ALL distinct objects you can see.

Include EVERYTHING:
- Food items (fruits, vegetables, prepared food, etc.)
- Utensils (spoons, forks, knives, etc.)
- Containers (bowls, plates, cups, boxes, etc.)
- Toys (balls, blocks, cubes, etc.)
- Tools and other items
- Background objects

Do NOT include:
- The robotic arm
- The table/surface itself
- Walls/floor

Output format: comma-separated list of all objects visible in Frame 1."""

    out1b = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q1b, return_history=False,
        max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_MED
    )
    
    frame1_objects = _clean_list(out1b.strip())
    if verbose:
        print(f"Step 1b - Frame 1 Object Inventory: {frame1_objects}")

    # =========================================================================
    # STEP 2: Check for DUPLICATE OBJECT Hallucinations (Primary Check)
    # This is the most common hallucination pattern in robotic manipulation edits
    # =========================================================================
    
    for target in instruction_objects:
        q2 = f"""TASK: Duplicate Object Detection

You are comparing Frame 1 (BEFORE) and Frame 2 (AFTER) of a robotic manipulation.
The instruction was: "{instruction}"
Focus on this object: "{target}"

CRITICAL QUESTION: In Frame 2 (the AFTER image), do you see TWO or MORE instances of "{target}"?

Look VERY carefully at Frame 2 and answer:
1. How many "{target}" objects do you see in Frame 1?
2. How many "{target}" objects do you see in Frame 2?
3. Is there still a "{target}" at its ORIGINAL position from Frame 1 in Frame 2?

A DUPLICATION HALLUCINATION occurs when:
- The original "{target}" is STILL at its original position in Frame 2
- AND there is ALSO a "{target}" at a new position (where it was moved to)
- Result: You see 2 copies of the same object in Frame 2

A CORRECT EDIT means:
- The "{target}" moved from its original position to a new position
- The original position is now EMPTY
- Result: You see only 1 "{target}" in Frame 2, just at a different location

ANSWER FORMAT:
Count_in_Frame1: [number]
Count_in_Frame2: [number]
Original_position_empty_in_Frame2: [YES or NO]
Verdict: [DUPLICATED or NOT_DUPLICATED]"""

        out2 = _safe_chat_call(
            video_chat_model, verbose=verbose, video_path=tmp_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=q2, return_history=False,
            max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_MED
        )
        
        reply = out2.strip().lower()
        if verbose:
            print(f"Step 2 - Duplicate check for '{target}':\n{out2.strip()}")

        # Parse all fields first
        lines = reply.split("\n")
        count_frame1 = None
        count_frame2 = None
        original_empty = None
        verdict_text = None
        
        for line in lines:
            line_lower = line.lower().strip()
            if "count_in_frame1" in line_lower or "count in frame 1" in line_lower:
                # Extract number after colon to avoid picking up "1" from "Frame1"
                if ":" in line:
                    value_part = line.split(":")[-1].strip()
                    nums = re.findall(r'\d+', value_part)
                    if nums:
                        count_frame1 = int(nums[0])
            elif "count_in_frame2" in line_lower or "count in frame 2" in line_lower:
                # Extract number after colon to avoid picking up "2" from "Frame2"
                if ":" in line:
                    value_part = line.split(":")[-1].strip()
                    nums = re.findall(r'\d+', value_part)
                    if nums:
                        count_frame2 = int(nums[0])
            elif "original_position_empty" in line_lower or "original position empty" in line_lower:
                original_empty = "yes" in line_lower
            elif "verdict:" in line_lower:
                verdict_text = line_lower.split("verdict:")[-1].strip()
        
        # Determine duplication with clear priority:
        # 1. STRONGEST: Count explicitly increased (count_frame2 > count_frame1)
        # 2. MEDIUM: Verdict says DUPLICATED (not NOT_DUPLICATED)
        # 3. If verdict says NOT_DUPLICATED - trust it
        
        # Store Step 2 counts for use in Step 4 verification
        if count_frame1 is not None and count_frame2 is not None:
            step2_counts[target] = (count_frame1, count_frame2)

        is_duplicated = False
        confidence = 'LOW'
        
        # Priority 1: Trust explicit count increase (strongest signal)
        if count_frame1 is not None and count_frame2 is not None:
            if count_frame2 > count_frame1:
                is_duplicated = True
                confidence = 'HIGH'
                # Inventory contradiction check: if count_frame1=0 but the object
                # was found in the Frame 1 inventory, the model's counting is wrong.
                # The object exists in Frame 1 (inventory saw it) but the count says 0.
                # Downgrade to LOW confidence.
                if count_frame1 == 0:
                    obj_in_inventory = any(_objects_match(target, inv_obj) for inv_obj in frame1_objects)
                    if obj_in_inventory:
                        confidence = 'LOW'
                        inventory_contradictions.add(target)
                        if verbose:
                            print(f"  [DEBUG] Count says 0 in F1 but '{target}' found in inventory — downgrading to LOW")
                if verbose and confidence == 'HIGH':
                    print(f"  [DEBUG] Count increased {count_frame1}->{count_frame2}, HIGH confidence")
        
        # Priority 2: Check verdict (only if count didn't already determine)
        if not is_duplicated and verdict_text:
            # Check for explicit NOT_DUPLICATED first
            if "not_duplicated" in verdict_text or "not duplicated" in verdict_text:
                is_duplicated = False  # Explicitly not duplicated
                if verbose:
                    print(f"  [DEBUG] Verdict says NOT_DUPLICATED, trusting it")
            elif "duplicated" in verdict_text:
                # Verdict says duplicated but count didn't increase
                # This is lower confidence - might be model confusion
                is_duplicated = True
                confidence = 'LOW'
                if verbose:
                    print(f"  [DEBUG] Verdict says DUPLICATED but count didn't increase, LOW confidence")
        
        # GDINO CROSS-VALIDATION: Use Grounding DINO to get ground-truth counts
        gdino_f1, gdino_f2 = gdino_count_both_frames(img_before_path, img_after_path, target)
        if verbose:
            print(f"  [GDINO] '{target}': Frame1={gdino_f1}, Frame2={gdino_f2}")
        
        if is_duplicated:
            # GDINO disagrees → downgrade confidence
            if gdino_f2 <= gdino_f1:
                if confidence == 'HIGH':
                    confidence = 'LOW'  # Downgrade, don't reject — VLM might be right
                    if verbose:
                        print(f"  [GDINO override] Count did NOT increase ({gdino_f1}->{gdino_f2}), downgrading to LOW")
                # If already LOW and GDINO disagrees, still keep as LOW for verification
            elif gdino_f2 > gdino_f1 and confidence == 'LOW':
                # GDINO confirms duplication → upgrade to HIGH
                # BUT: only trust GDINO when counts are reliable (max ≤ 5).
                # High counts (6+) mean GDINO is matching wrong objects (noise).
                if max(gdino_f1, gdino_f2) <= 5:
                    confidence = 'HIGH'
                    if verbose:
                        print(f"  [GDINO confirm] Count increased ({gdino_f1}->{gdino_f2}), upgrading to HIGH")
                else:
                    if verbose:
                        print(f"  [GDINO unreliable] Counts too high ({gdino_f1}->{gdino_f2}), keeping LOW")
        else:
            # VLM said no duplication, but GDINO sees more objects in Frame 2
            # Only rescue if GDINO increase is significant: >= 2, OR small count (F1 <= 2)
            # Also rescue marginal increases if counts are in moderate range (max <= 5)
            # BUT: for in-place manipulation verbs (flip, close, open, turn),
            # marginal increases are likely repositioning noise, not real duplication
            if gdino_f2 > gdino_f1 and gdino_f1 > 0:
                gdino_increase = gdino_f2 - gdino_f1
                in_place_verbs = ['flip', 'close', 'open', 'turn', 'rotate', 'raise', 'lower', 'upright']
                is_in_place = any(v in instruction.lower() for v in in_place_verbs)
                # For in-place verbs, require >= 2 increase or very small counts (f1 <= 2)
                # For placement verbs, also allow moderate counts (max <= 5)
                if gdino_increase >= 2 or gdino_f1 <= 2:
                    is_duplicated = True
                    confidence = 'LOW'
                    if verbose:
                        print(f"  [GDINO rescue] VLM missed duplication! Count {gdino_f1}->{gdino_f2}, adding as LOW")
                elif not is_in_place and max(gdino_f1, gdino_f2) <= 5:
                    is_duplicated = True
                    confidence = 'LOW'
                    if verbose:
                        print(f"  [GDINO rescue] Moderate increase {gdino_f1}->{gdino_f2}, adding as LOW")
                else:
                    if verbose:
                        print(f"  [GDINO rescue SKIPPED] Marginal increase {gdino_f1}->{gdino_f2} (noise), not rescuing")
        
        if is_duplicated:
            hallucinated_objects[target] = confidence
            if verbose:
                print(f"  -> HALLUCINATION DETECTED: '{target}' is duplicated (confidence: {confidence})")

    # =========================================================================
    # STEP 2b: Check for STRUCTURAL HALLUCINATIONS on instruction objects
    # =========================================================================
    # Sometimes the object is placed correctly but its structure/appearance
    # changes dramatically (e.g., spoon becomes deformed, different shape)
    
    for target in instruction_objects:
        # Skip if already detected as duplicated
        if target in hallucinated_objects:
            continue
            
        q2b = f"""TASK: Structural/Appearance Change Detection

Compare the "{target}" in Frame 1 (BEFORE) vs Frame 2 (AFTER).

The instruction was: "{instruction}"

Focus on the PHYSICAL STRUCTURE and SHAPE of "{target}":

1. SHAPE: Does "{target}" have the SAME shape in both frames?
   - Same length/width/height?
   - Same curvature/angles?
   - Same overall silhouette?

2. STRUCTURE: Does "{target}" have the SAME structure?
   - Same number of parts?
   - Parts in same arrangement?
   - Nothing bent, broken, or deformed?

3. APPEARANCE: Does "{target}" look like the SAME object?
   - Could this be a different "{target}" swapped in?
   - Does it look significantly altered?

IMPORTANT: Position changes are OK. We're looking for SHAPE/STRUCTURE changes.

A STRUCTURAL HALLUCINATION means:
- The object's shape changed significantly (bent, stretched, shrunk, deformed)
- Parts are missing or added
- It looks like a DIFFERENT version of the object

Structural_Change: [NONE, MINOR, or MAJOR]
Description: [what structural changes do you see, if any]
Verdict: [STRUCTURAL_HALLUCINATION or NO_STRUCTURAL_CHANGE]"""

        out2b = _safe_chat_call(
            video_chat_model, verbose=verbose, video_path=tmp_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=q2b, return_history=False,
            max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_MED
        )
        
        reply = out2b.strip().lower()
        if verbose:
            print(f"Step 2b - Structural check for '{target}':\n{out2b.strip()}")
        
        # Check for structural hallucination
        is_structural_hallucination = False
        
        if "verdict:" in reply:
            verdict_part = reply.split("verdict:")[-1].strip().split("\n")[0]
            if "structural_hallucination" in verdict_part or "structural hallucination" in verdict_part:
                is_structural_hallucination = True
        
        # Also check for MAJOR structural change
        if "structural_change:" in reply or "structural change:" in reply:
            change_marker = "structural_change:" if "structural_change:" in reply else "structural change:"
            change_part = reply.split(change_marker)[-1].strip().split("\n")[0]
            if "major" in change_part:
                is_structural_hallucination = True
        
        # Check for deformation keywords in description
        # BUT only if the verdict does NOT explicitly say NO_STRUCTURAL_CHANGE
        verdict_says_no = False
        if "verdict:" in reply:
            last_verdict = reply.split("verdict:")[-1].strip().split("\n")[0]
            if "no_structural_change" in last_verdict or "no structural change" in last_verdict or "no_change" in last_verdict:
                verdict_says_no = True
        
        if not verdict_says_no:
            deformation_keywords = ["bent", "deformed", "stretched", "shrunk", "broken", 
                                    "different shape", "changed shape", "significantly", 
                                    "altered", "modified", "warped", "distorted"]
            if any(kw in reply for kw in deformation_keywords):
                # If description mentions major changes, flag it
                if "description:" in reply:
                    desc_part = reply.split("description:")[-1].strip().split("\n")[0]
                    if any(kw in desc_part for kw in deformation_keywords):
                        is_structural_hallucination = True
        
        if is_structural_hallucination:
            hallucinated_objects[target + " (structural)"] = 'LOW'
            if verbose:
                print(f"  -> STRUCTURAL HALLUCINATION DETECTED: '{target}'")

    # =========================================================================
    # STEP 2c: Check for OBJECT SUBSTITUTION Hallucinations
    # =========================================================================
    # This catches cases where the wrong object is placed instead of the target
    # (e.g., corn placed instead of sweet potato)
    # ONLY applies when instruction involves "put X in/on Y" type actions
    
    # Check if instruction is a "put/place" type that could have substitution
    instruction_lower = instruction.lower()
    is_placement_instruction = any(verb in instruction_lower for verb in 
                                   ["put ", "place ", "move ", "pick up", "take "])
    
    # Only check substitution for the SOURCE object (the object being moved)
    # Not for destination objects
    # Pattern captures up to 3 words for objects like "sweet potato", "blue pen", "corn on cob"
    source_object = None
    for pattern in [
        r"put\s+(?:the\s+)?(.+?)\s+(?:in|on|into|onto)\s+",
        r"place\s+(?:the\s+)?(.+?)\s+(?:in|on|into|onto)\s+",
        r"move\s+(?:the\s+)?(.+?)\s+(?:to|into|onto)\s+",
        r"pick\s+up\s+(?:the\s+)?(.+?)(?:\s+and|\s*$)",
    ]:
        match = re.search(pattern, instruction_lower)
        if match:
            candidate = match.group(1).strip()
            # Limit to first 3 words to avoid grabbing too much
            words = candidate.split()
            if len(words) <= 3:
                source_object = candidate
            else:
                source_object = " ".join(words[:3])
            break
    
    if is_placement_instruction and source_object and len(source_object) > 1:
        # Only check substitution for the source object
        target = source_object
        
        # Skip if already detected as duplicated (using fuzzy matching)
        if not any(_objects_match(target, hobj) for hobj in hallucinated_objects):
            # First: Find out what the destination is from instruction
            destination = None
            for dest_pattern in [
                r"(?:in|on|into|onto)\s+(?:the\s+)?(.+?)(?:\s+which|\s*$)",
                r"(?:in|on|into|onto)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
            ]:
                dest_match = re.search(dest_pattern, instruction_lower)
                if dest_match:
                    destination = dest_match.group(1).strip()
                    break
            
            q2c = f"""TASK: Object Substitution Detection

The instruction says: "{instruction}"
The instruction wants to place/move: "{target}"

Look at Frame 2 (AFTER image) carefully at the destination location:

1. What object is NOW at the destination (the {destination if destination else 'target location'})?
2. Is this the same type of object as "{target}", or is it a DIFFERENT object type?

Examples of SUBSTITUTION:
- Instruction: "put sweet potato in pan" but a CORN is in the pan = SUBSTITUTION
- Instruction: "put red apple in bowl" but a GREEN BALL is in the bowl = SUBSTITUTION

Examples of NO SUBSTITUTION:
- The "{target}" is at the destination = NO SUBSTITUTION
- The destination is empty = NO SUBSTITUTION
- The "{target}" moved but looks the same = NO SUBSTITUTION

Answer format:
Object_at_destination: [what object is at the destination in Frame 2]
Is_it_the_same_as_target: [YES if same type as "{target}", NO if different]
Was_substituted: [YES if a different object was placed instead, NO otherwise]"""

            out2c = _safe_chat_call(
                video_chat_model, verbose=verbose, video_path=tmp_video_path,
                tokenizer=video_chat_tokenizer, user_prompt=q2c, return_history=False,
                max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_MED
            )
            
            reply = out2c.strip().lower()
            if verbose:
                print(f"Step 2c - Substitution check for '{target}':\n{out2c.strip()}")
            
            # Check for substitution - multiple parsing strategies
            is_substitution = False
            substituted_object = None

            # Strategy 1: Check substitution-related keys (various model response formats)
            vlm_explicitly_said_no = False
            for marker in ["was_substituted:", "was_different_object_substituted:",
                           "is_substitution:", "substitution:", "substituted:"]:
                if marker in reply:
                    sub_part = reply.split(marker)[-1].strip().split("\n")[0]
                    if "yes" in sub_part and "no" not in sub_part:
                        is_substitution = True
                    elif "no" in sub_part:
                        vlm_explicitly_said_no = True
                    break

            # Strategy 2: Check "same as target" type keys (reversed logic - NO means substitution)
            vlm_says_same = False
            if not is_substitution:
                for marker in ["is_it_the_same_as_target:", "same_as_target:",
                               "same_type_as_target:", "same type as target:",
                               "is it the same as target:", "same as target:"]:
                    if marker in reply:
                        same_part = reply.split(marker)[-1].strip().split("\n")[0]
                        if "no" in same_part and "yes" not in same_part:
                            is_substitution = True
                        elif "yes" in same_part:
                            vlm_says_same = True
                        break

            # Extract what object is at destination
            obj_at_dest_raw = None
            for marker in ["object_at_destination:", "object at destination:",
                           "if_yes_what_object:", "object_at_dest:", "what_object:"]:
                if marker in reply:
                    obj_part = reply.split(marker)[-1].strip().split("\n")[0]
                    obj_part = obj_part.strip().strip("[]")
                    if obj_part and len(obj_part) > 1 and len(obj_part) < 50:
                        obj_part = obj_part.replace("(", "").replace(")", "").strip()
                        obj_part = re.sub(r'^(the|a|an)\s+', '', obj_part).strip()
                        if "empty" not in obj_part.lower() and "none" not in obj_part.lower():
                            obj_at_dest_raw = obj_part
                    break

            if is_substitution and obj_at_dest_raw:
                substituted_object = obj_at_dest_raw

            # Strategy 3: Even if model said NO substitution, check if named object differs from target
            # BUT respect the VLM's explicit answers: if VLM says "same as target" or
            # "not substituted", trust it over text matching (e.g., soap ≈ detergent)
            if not is_substitution and obj_at_dest_raw:
                if not vlm_explicitly_said_no and not vlm_says_same:
                    if not _objects_match(obj_at_dest_raw, target):
                        is_substitution = True
                        substituted_object = obj_at_dest_raw
            
            if is_substitution and substituted_object:
                # Extra check: make sure the substituted object isn't the same as the target
                if not _objects_match(substituted_object, target):
                    hallucinated_objects[substituted_object] = 'SUBSTITUTION'
                    if verbose:
                        print(f"  -> SUBSTITUTION HALLUCINATION: '{substituted_object}' placed instead of '{target}'")

    # =========================================================================
    # STEP 3: Comprehensive Background/Random Object Hallucination Check
    # =========================================================================
    # This step looks for:
    # A) Duplication of background objects (objects NOT in instruction)
    # B) Completely new objects that appeared from nowhere
    
    # Step 3a: Check for duplication of ANY object from Frame 1 inventory
    # USE GDINO as primary counter — much more reliable than VLM for counting
    if verbose:
        print(f"Step 3 - Checking for duplications of all Frame 1 objects...")
    
    for obj in frame1_objects:
        # Skip instruction objects (already handled in Step 2)
        # BUT: if the inventory object has a color/type modifier that the instruction
        # object lacks (e.g., "green plate" vs "plate"), treat as different —
        # a colored variant could be a new hallucinated object
        obj_norm = _normalize_object_name(obj)
        is_exact_instruction_obj = False
        for inst_obj in instruction_objects:
            inst_norm = _normalize_object_name(inst_obj)
            if obj_norm == inst_norm:
                is_exact_instruction_obj = True
                break
            # Only skip if they match AND the inventory name doesn't have extra words
            # "green plate" has extra word "green" vs "plate" → don't skip
            if _objects_match(obj, inst_obj):
                obj_words = obj_norm.split()
                inst_words = inst_norm.split()
                if len(obj_words) <= len(inst_words):
                    is_exact_instruction_obj = True
                    break
                # Same length but different modifiers → treat as different
        if is_exact_instruction_obj:
            continue
        
        # Skip if already detected
        if obj in hallucinated_objects:
            continue
        
        # GDINO-first approach: count with Grounding DINO before VLM
        gdino_f1, gdino_f2 = gdino_count_both_frames(img_before_path, img_after_path, obj)
        if verbose:
            print(f"Step 3a - GDINO count for '{obj}': Frame1={gdino_f1}, Frame2={gdino_f2}")
        
        # Store GDINO counts as step3a counts
        step3a_counts[obj] = (gdino_f1, gdino_f2)
        
        # If GDINO says count increased, flag as candidate
        # But sanity check: if counts are suspiciously high (>3), GDINO may be
        # matching wrong objects (e.g., "orange" matching any orange-colored item)
        need_vlm_fallback = False
        if gdino_f2 > gdino_f1 and gdino_f1 > 0:
            if max(gdino_f1, gdino_f2) <= 3:
                hallucinated_objects[obj] = 'LOW'
                if verbose:
                    print(f"  -> BACKGROUND DUPLICATION CANDIDATE (GDINO): '{obj}' count {gdino_f1}->{gdino_f2}")
            else:
                # High counts — unreliable, fall back to VLM
                need_vlm_fallback = True
                if verbose:
                    print(f"  -> GDINO counts suspiciously high ({gdino_f1}->{gdino_f2}), falling back to VLM...")
        elif gdino_f2 > gdino_f1 and gdino_f1 == 0:
            need_vlm_fallback = True
        elif gdino_f1 == gdino_f2 and gdino_f1 > 3:
            # Both counts equal and suspiciously high → GDINO may be overcounting
            # and masking a real +1 increase. Fall back to VLM.
            need_vlm_fallback = True
            if verbose:
                print(f"  -> GDINO counts equal but high ({gdino_f1}->{gdino_f2}), falling back to VLM...")
        
        # Special case: if this object partially matches an instruction object
        # (e.g., "green plate" matches "plate"), GDINO may be confusing them.
        # Even if GDINO says equal counts, do a VLM check.
        if not need_vlm_fallback and gdino_f2 == gdino_f1:
            has_partial_instr_match = False
            for inst_obj in instruction_objects:
                inst_norm = _normalize_object_name(inst_obj)
                obj_norm_check = _normalize_object_name(obj)
                obj_words = set(obj_norm_check.split())
                inst_words = set(inst_norm.split())
                # Check if they share a base word but are NOT identical
                if obj_words & inst_words and obj_norm_check != inst_norm:
                    has_partial_instr_match = True
                    break
            if has_partial_instr_match:
                need_vlm_fallback = True
                if verbose:
                    print(f"  -> Partial match with instruction object, falling back to VLM...")
        
        if need_vlm_fallback:
            q3_dup = f"""TASK: Check if "{obj}" was DUPLICATED

Compare Frame 1 (BEFORE) and Frame 2 (AFTER).
Focus on: "{obj}"

Count carefully:
1. How many "{obj}" do you see in Frame 1?
2. How many "{obj}" do you see in Frame 2?

If there are MORE "{obj}" in Frame 2 than in Frame 1, it means a duplicate was created (HALLUCINATION).

Count_in_Frame1: [number]
Count_in_Frame2: [number]
Verdict: [DUPLICATED if count increased, NOT_DUPLICATED if same or less]"""

            out3_dup = _safe_chat_call(
                video_chat_model, verbose=verbose, video_path=tmp_video_path,
                tokenizer=video_chat_tokenizer, user_prompt=q3_dup, return_history=False,
                max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_SHORT
            )
            
            reply = out3_dup.strip().lower()
            if verbose:
                print(f"Step 3a - VLM fallback for '{obj}' (GDINO F1=0):\n{out3_dup.strip()}")
            
            count_frame1 = None
            count_frame2 = None
            for line in reply.split("\n"):
                line_lower = line.lower().strip()
                if "count_in_frame1" in line_lower or "count in frame 1" in line_lower:
                    if ":" in line:
                        value_part = line.split(":")[-1].strip()
                        nums = re.findall(r'\d+', value_part)
                        if nums:
                            count_frame1 = int(nums[0])
                elif "count_in_frame2" in line_lower or "count in frame 2" in line_lower:
                    if ":" in line:
                        value_part = line.split(":")[-1].strip()
                        nums = re.findall(r'\d+', value_part)
                        if nums:
                            count_frame2 = int(nums[0])
            
            if count_frame1 is not None and count_frame2 is not None:
                step3a_counts[obj] = (count_frame1, count_frame2)  # Override with VLM counts
                if count_frame2 > count_frame1:
                    hallucinated_objects[obj] = 'LOW'
                    if verbose:
                        print(f"  -> BACKGROUND DUPLICATION CANDIDATE (VLM fallback): '{obj}' count {count_frame1}->{count_frame2}")
            
            if "verdict:" in reply:
                verdict_part = reply.split("verdict:")[-1].strip().split("\n")[0]
                if "duplicated" in verdict_part and "not_duplicated" not in verdict_part and "not duplicated" not in verdict_part:
                    if obj not in hallucinated_objects:
                        hallucinated_objects[obj] = 'LOW'
                        if verbose:
                            print(f"  -> BACKGROUND DUPLICATION CANDIDATE (VLM verdict): '{obj}'")
        # If GDINO says no increase AND counts are reasonable, skip VLM (saves VLM calls!)
        if not need_vlm_fallback and gdino_f2 <= gdino_f1:
            if verbose:
                print(f"  -> GDINO: No duplication for '{obj}' ({gdino_f1}->{gdino_f2}), skipping VLM")
    
    # Step 3b: Scan for completely NEW objects not in Frame 1
    q3b = f"""TASK: Find Completely NEW Objects

Compare Frame 1 (BEFORE) and Frame 2 (AFTER).

Objects that existed in Frame 1: {frame1_objects}

Look for objects in Frame 2 that:
1. Were NOT present ANYWHERE in Frame 1
2. Are real physical objects (not shadows, reflections, or blur)
3. Are NOT the robotic arm

Check CAREFULLY for:
- Any NEW balls (red ball, blue ball, etc.)
- Any NEW fruits or vegetables
- Any NEW utensils or tools
- Any objects that appear "out of nowhere"

List any NEW objects that appear in Frame 2 but did NOT exist in Frame 1.
Be thorough - check all areas: foreground, background, corners, edges.

Output: comma-separated list of new objects, or "none" if no new objects found."""

    out3b = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q3b, return_history=False,
        max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_MED
    )
    
    new_objects_raw = out3b.strip().lower()
    if verbose:
        print(f"Step 3b - New objects scan: {new_objects_raw}")
    
    # Step 3c: More aggressive scan with specific prompting
    q3c = """TASK: Detailed New Object Detection

Look VERY carefully at Frame 2 (AFTER image) and compare with Frame 1 (BEFORE image).

I need you to find ANY object that exists in Frame 2 but NOT in Frame 1.

Specifically check for:
- RED BALL - is there a red ball in Frame 2 that wasn't in Frame 1?
- BLUE BALL - is there a blue ball in Frame 2 that wasn't in Frame 1?
- Any colored ball that is NEW?
- Any fruit (banana, apple, orange) that is NEW?
- Any utensil that is NEW?
- ANY object at all that is NEW?

Scan the ENTIRE image including:
- Far left and far right edges
- Top and bottom areas
- Behind or beside other objects
- Areas that might be partially hidden

List ALL new objects you find (be specific about colors), or "none" if Frame 2 has no new objects."""

    out3c = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q3c, return_history=False,
        max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_MED
    )
    
    new_objects_raw_c = out3c.strip().lower()
    if verbose:
        print(f"Step 3c - Detailed new objects scan: {new_objects_raw_c}")
    
    # Step 3d: Direct question about specific new object types
    q3d = """TASK: Direct New Object Check

Answer YES or NO for each:

1. Is there a RED BALL in Frame 2 that was NOT in Frame 1? [YES/NO]
2. Is there a NEW BALL (any color) in Frame 2 that was NOT in Frame 1? [YES/NO]
3. Is there any completely NEW object in Frame 2 that was NOT in Frame 1? [YES/NO]

If you answered YES to any, list the new objects:
New objects found: [list or none]"""

    out3d = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q3d, return_history=False,
        max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_SHORT
    )
    
    new_objects_raw_d = out3d.strip().lower()
    if verbose:
        print(f"Step 3d - Direct new object check: {new_objects_raw_d}")
    
    # Step 3e: Check for HUMAN HANDS and EXTRA ROBOTIC ARMS (common hallucinations in robot manipulation)
    q3e = f"""TASK: Detect HUMAN HANDS or EXTRA ROBOTIC ARMS

Look at Frame 2 (AFTER image) VERY carefully.

1. HUMAN HAND/ARM CHECK - Look for ANY of these in Frame 2:
   - Human fingers (5 fingers, skin color)
   - Human palm or hand
   - Human arm or forearm (skin visible, not metal)
   - A hand that looks like it belongs to a person, not a robot
   - IMPORTANT: If you see an object being held or placed by what looks like human fingers, answer YES

2. EXTRA ROBOT ARM CHECK:
   - Count robotic arms/grippers in Frame 1
   - Count robotic arms/grippers in Frame 2
   - If Frame 2 has MORE robot arms than Frame 1, answer YES

Answer format (answer YES or NO for each):
Human_hand_visible: [YES if any human hand/fingers/arm visible, NO otherwise]
Number_of_robot_arms_frame1: [count]
Number_of_robot_arms_frame2: [count]
Extra_arm_appeared: [YES if more robot arms in Frame 2, NO otherwise]
List_of_anomalies: [list any human body parts or extra arms found, or None]"""

    out3e = _safe_chat_call(
        video_chat_model, verbose=verbose, video_path=tmp_video_path,
        tokenizer=video_chat_tokenizer, user_prompt=q3e, return_history=False,
        max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_MED
    )
    
    anomaly_raw = out3e.strip().lower()
    if verbose:
        print(f"Step 3e - Human hand/extra arm check: {out3e.strip()}")
    
    # If step 3e didn't detect hand, do a focused hand-only check for common hallucination cases
    step3e_detected_hand = False
    if "human_hand_visible:" in anomaly_raw:
        hand_part = anomaly_raw.split("human_hand_visible:")[-1].strip().split("\n")[0]
        if "yes" in hand_part and "no" not in hand_part:
            step3e_detected_hand = True
    
    if not step3e_detected_hand:
        # Do a very focused hand check
        q3e_hand = """TASK: Human Hand Detection (CRITICAL)

Look ONLY at Frame 2 (AFTER image).

Is there ANY human body part visible in Frame 2?
- Human hand
- Human fingers (skin-colored, 5 fingers)
- Human arm (skin visible)

Note: Robotic grippers are NOT human hands. Look for SKIN, not metal.

Answer with ONLY: YES or NO"""

        out3e_hand = _safe_chat_call(
            video_chat_model, verbose=verbose, video_path=tmp_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=q3e_hand, return_history=False,
            max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_SHORT
        )
        
        hand_check = out3e_hand.strip().lower()
        if verbose:
            print(f"Step 3e (focused hand check): {hand_check}")
        
        if "yes" in hand_check and "no" not in hand_check:
            step3e_detected_hand = True

    # Third attempt: indirect hand detection - ask what is performing the action
    if not step3e_detected_hand:
        q3e_who = f"""Look at Frame 2 (AFTER image) carefully.

Something is holding or placing an object in this scene.
What is performing the manipulation?

A) A robotic gripper/arm (metal, mechanical, with joints)
B) A human hand (skin-colored, with fingers and fingernails)
C) No entity is holding anything
D) Cannot determine

Answer with ONLY the letter (A, B, C, or D):"""

        out3e_who = _safe_chat_call(
            video_chat_model, verbose=verbose, video_path=tmp_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=q3e_who, return_history=False,
            max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_SHORT
        )

        who_reply = out3e_who.strip().lower()
        if verbose:
            print(f"Step 3e (indirect hand check): {who_reply}")

        # Check if answer indicates human hand (option B)
        if re.match(r'^b[\s\.\)\:]', who_reply) or who_reply.strip() == "b" or "human hand" in who_reply or "human fingers" in who_reply:
            step3e_detected_hand = True
            if verbose:
                print(f"  -> Indirect check: HUMAN HAND detected")

    # Parse Step 3e for human hands and extra arms
    step3e_objects = set()
    
    # Add human hand if detected (from either check)
    if step3e_detected_hand:
        step3e_objects.add("human hand")
        if verbose:
            print(f"  -> Detected HUMAN HAND in Frame 2")
    
    # Check for extra robotic arm
    if "extra_arm_appeared:" in anomaly_raw or "extra arm appeared:" in anomaly_raw:
        for marker in ["extra_arm_appeared:", "extra arm appeared:"]:
            if marker in anomaly_raw:
                arm_part = anomaly_raw.split(marker)[-1].strip().split("\n")[0]
                if "yes" in arm_part and "no" not in arm_part:
                    step3e_objects.add("extra robotic arm")
                break
    
    # Also check arm counts
    arm_f1 = None
    arm_f2 = None
    for line in anomaly_raw.split("\n"):
        if "robot_arms_frame1" in line or "robot arms frame1" in line or "robot_arms_frame 1" in line:
            nums = re.findall(r'\d+', line)
            if nums:
                arm_f1 = int(nums[-1])
        if "robot_arms_frame2" in line or "robot arms frame2" in line or "robot_arms_frame 2" in line:
            nums = re.findall(r'\d+', line)
            if nums:
                arm_f2 = int(nums[-1])
    
    if arm_f1 is not None and arm_f2 is not None and arm_f2 > arm_f1:
        step3e_objects.add("extra robotic arm")
        if verbose:
            print(f"  -> Detected extra robotic arm (count {arm_f1} -> {arm_f2})")

    # Focused arm check if not yet detected
    if "extra robotic arm" not in step3e_objects:
        q3e_arm = """TASK: Count robotic arms carefully

A robotic arm is a mechanical arm with joints, usually metallic, with a gripper at the end.

Frame 1: How many SEPARATE robotic arms are visible? [number]
Frame 2: How many SEPARATE robotic arms are visible? [number]

Does Frame 2 have MORE robotic arms than Frame 1?
Answer: [YES or NO]"""

        out3e_arm = _safe_chat_call(
            video_chat_model, verbose=verbose, video_path=tmp_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=q3e_arm, return_history=False,
            max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_SHORT
        )

        arm_reply = out3e_arm.strip().lower()
        if verbose:
            print(f"Step 3e (focused arm check): {arm_reply}")

        af1_val = None
        af2_val = None
        for line in arm_reply.split("\n"):
            ll = line.lower().strip()
            if ("frame 1" in ll or "frame1" in ll) and ":" in line:
                nums = re.findall(r'\d+', line.split(":")[-1])
                if nums:
                    af1_val = int(nums[0])
            elif ("frame 2" in ll or "frame2" in ll) and ":" in line:
                nums = re.findall(r'\d+', line.split(":")[-1])
                if nums:
                    af2_val = int(nums[0])

        if af1_val is not None and af2_val is not None and af2_val > af1_val:
            step3e_objects.add("extra robotic arm")
            if verbose:
                print(f"  -> Focused arm check: extra arm ({af1_val} -> {af2_val})")

        # Also check direct YES/NO answer
        if "extra robotic arm" not in step3e_objects and "answer:" in arm_reply:
            ans_part = arm_reply.split("answer:")[-1].strip().split("\n")[0]
            if "yes" in ans_part and "no" not in ans_part:
                step3e_objects.add("extra robotic arm")
                if verbose:
                    print(f"  -> Focused arm check: YES more arms in Frame 2")

    # Check list of anomalies
    if "list_of_anomalies:" in anomaly_raw or "list of anomalies:" in anomaly_raw:
        for marker in ["list_of_anomalies:", "list of anomalies:"]:
            if marker in anomaly_raw:
                anomaly_list = anomaly_raw.split(marker)[-1].strip().split("\n")[0]
                if "none" not in anomaly_list and anomaly_list.strip():
                    # Parse items, allow robot terms through
                    items = _clean_list(anomaly_list, strict_robot_filter=False)
                    step3e_objects.update(items)
                break
    
    if verbose and step3e_objects:
        print(f"  -> Step 3e detected (VLM): {step3e_objects}")

    # GDINO-based hand detection (more reliable than VLM for detecting human hands)
    # Check for human hand in Frame 2 only (high threshold to avoid FP)
    gdino_hand_f2 = gdino_detect(img_after_path, "human hand. human fingers. person hand.", box_threshold=0.40)
    if gdino_hand_f2:
        step3e_objects.add("human hand")
        if verbose:
            print(f"  -> GDINO detected human hand in Frame 2: {len(gdino_hand_f2)} detections (scores: {[round(d['score'], 3) for d in gdino_hand_f2]})")
    
    # GDINO arm detection: delta-based approach with high threshold
    # Only count as extra arm if Frame 2 has more arms than Frame 1
    gdino_arm_f1 = gdino_detect(img_before_path, "robotic arm. robot arm. mechanical arm.", box_threshold=0.45)
    gdino_arm_f2 = gdino_detect(img_after_path, "robotic arm. robot arm. mechanical arm.", box_threshold=0.45)
    if len(gdino_arm_f2) >= 2 and len(gdino_arm_f1) <= 1:
        step3e_objects.add("extra robotic arm")
        if verbose:
            print(f"  -> GDINO detected extra robotic arm: F1={len(gdino_arm_f1)}, F2={len(gdino_arm_f2)}")
    elif verbose:
        print(f"  -> GDINO arm check: F1={len(gdino_arm_f1)}, F2={len(gdino_arm_f2)} (no extra)")
    
    if verbose and step3e_objects:
        print(f"  -> Step 3e final detected: {step3e_objects}")

    # Parse Step 3d carefully - only extract objects from YES responses
    step3d_objects = set()
    # Check for explicit YES responses to the questions
    if "1." in new_objects_raw_d or "1:" in new_objects_raw_d:
        # Check if answer to Q1 (red ball) is YES
        if re.search(r'1\.?\s*:?\s*yes', new_objects_raw_d):
            step3d_objects.add("red ball")
    if "2." in new_objects_raw_d or "2:" in new_objects_raw_d:
        # Check if answer to Q2 (new ball) is YES  
        if re.search(r'2\.?\s*:?\s*yes', new_objects_raw_d):
            step3d_objects.add("ball")
    # Check for "New objects found:" section
    if "new objects found:" in new_objects_raw_d:
        found_section = new_objects_raw_d.split("new objects found:")[-1].strip()
        if "none" not in found_section and found_section:
            found_items = _clean_list(found_section)
            step3d_objects.update(found_items)
    # Also catch explicit mentions of red ball with yes context
    if "red ball" in new_objects_raw_d and "yes" in new_objects_raw_d:
        step3d_objects.add("red ball")
    
    # Step 3f: Check for unexpected objects at/in the destination (for placement/flip instructions)
    step3f_objects = set()
    _instr_lower = instruction.lower()
    _has_action_verb = any(v in _instr_lower for v in ["put ", "place ", "move ", "flip ", "close ", "open "])
    if _has_action_verb:
        _dest3f = None
        for _dp in [
            r"(?:in|on|into|onto)\s+(?:the\s+)?(.+?)(?:\s+which|\s+that|\s*$)",
            r"(?:in|on|into|onto)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
        ]:
            _dm = re.search(_dp, _instr_lower)
            if _dm:
                _dest3f = _dm.group(1).strip()
                _words = _dest3f.split()
                _dest3f = " ".join(_words[:2])
                break

        # For flip/close instructions, check inside the instruction object itself
        if "flip " in _instr_lower or "close " in _instr_lower:
            for _iobj in instruction_objects:
                if _iobj.lower() in ["pot", "pan", "bowl", "oven", "drawer", "box", "container", "sink"]:
                    _dest3f = _iobj
                    break

        if _dest3f and len(_dest3f) > 1:
            q3f = f"""TASK: Check for unexpected objects at destination

Compare Frame 1 (BEFORE) and Frame 2 (AFTER).
Focus on the {_dest3f}.

1. List ALL objects inside/on the {_dest3f} in Frame 1 (BEFORE):
2. List ALL objects inside/on the {_dest3f} in Frame 2 (AFTER):

Are there any EXTRA objects in/on the {_dest3f} in Frame 2 that were NOT inside/on it in Frame 1?
(The object from the instruction "{instruction}" is expected - only list UNEXPECTED extras)

New_unexpected_objects: [list any extra unexpected objects, or NONE]"""

            out3f = _safe_chat_call(
                video_chat_model, verbose=verbose, video_path=tmp_video_path,
                tokenizer=video_chat_tokenizer, user_prompt=q3f, return_history=False,
                max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_MED
            )

            reply3f = out3f.strip().lower()
            if verbose:
                print(f"Step 3f - Destination content check ({_dest3f}):\n{out3f.strip()}")

            for _mk in ["new_unexpected_objects:", "new unexpected objects:"]:
                if _mk in reply3f:
                    _new_objs = reply3f.split(_mk)[-1].strip().split("\n")[0]
                    if "none" not in _new_objs and _new_objs.strip() and len(_new_objs.strip()) > 1:
                        _items = _clean_list(_new_objs)
                        for _item in _items:
                            if not any(_objects_match(_item, io) for io in instruction_objects):
                                step3f_objects.add(_item)
                                if verbose:
                                    print(f"  -> NEW object at {_dest3f}: '{_item}'")
                            else:
                                # Step 3f found an instruction object at destination — 
                                # this may indicate a duplication that Step 2 missed.
                                # Add it back as a LOW candidate if not already detected.
                                if _item not in hallucinated_objects and not any(_objects_match(_item, ho) for ho in hallucinated_objects):
                                    hallucinated_objects[_item] = 'LOW'
                                    if verbose:
                                        print(f"  -> INSTRUCTION OBJ at destination '{_dest3f}': '{_item}' — added as duplication candidate")
                    break

    # Step 3g: Check for AMORPHOUS/FLUID hallucinations at destination
    # (e.g., liquid spilled, fluid appeared, residue, powder)
    # These are not discrete objects so GDINO and counting approaches miss them.
    if _has_action_verb and _dest3f and len(_dest3f) > 1:
        q3g = f"""TASK: Check for AMORPHOUS hallucinations (spills, liquid, residue, etc.)

Compare Frame 1 (BEFORE) and Frame 2 (AFTER).
Focus on: {_dest3f}

Look for any NEW substances that appeared in/on the {_dest3f} in Frame 2:
- Liquid, fluid, or spilled substance
- Powder, residue, or coating
- Any amorphous material that wasn't there before

IMPORTANT: Only flag substances that are NOT part of the intended instruction "{instruction}".

Is there any NEW amorphous substance (liquid, fluid, residue, etc.) in/on the {_dest3f} in Frame 2 that was NOT there in Frame 1?

Answer: [YES - describe what appeared] or [NO - no new substances]"""

        out3g = _safe_chat_call(
            video_chat_model, verbose=verbose, video_path=tmp_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=q3g, return_history=False,
            max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_SHORT
        )
        
        reply3g = out3g.strip().lower()
        if verbose:
            print(f"Step 3g - Amorphous hallucination check ({_dest3f}): {out3g.strip()}")
        
        if "yes" in reply3g and "no" not in reply3g.split("yes")[0][-5:]:
            # Extract what appeared
            substance_desc = reply3g.split("yes")[-1].strip().strip("-:").strip()
            if substance_desc and len(substance_desc) > 2:
                # Clean up the description
                substance_desc = re.sub(r'^[\s,\-:]+', '', substance_desc).strip()
                substance_desc = substance_desc.split(".")[0].strip()  # Take first sentence
                if len(substance_desc) > 50:
                    substance_desc = substance_desc[:50]
                if substance_desc and not any(_objects_match(substance_desc, io) for io in instruction_objects):
                    step3f_objects.add(substance_desc)
                    if verbose:
                        print(f"  -> AMORPHOUS hallucination detected: '{substance_desc}'")

    # Combine candidates from steps 3b and 3c (but NOT 3d raw output)
    all_new_candidates = set()
    for raw_output in [new_objects_raw, new_objects_raw_c]:
        if "none" not in raw_output and raw_output.strip():
            candidates = _clean_list(raw_output, strict_robot_filter=False)  # Allow robot terms for hallucination detection
            all_new_candidates.update(candidates)
    
    # Add properly parsed Step 3d objects
    all_new_candidates.update(step3d_objects)
    
    # Add Step 3e objects (human hands and extra robot arms)
    all_new_candidates.update(step3e_objects)

    # Add Step 3f objects (unexpected objects at destination)
    all_new_candidates.update(step3f_objects)

    if verbose:
        print(f"Step 3 - Combined new object candidates: {list(all_new_candidates)}")

    # Verify each candidate
    for obj in all_new_candidates:
        # Skip if it matches any instruction object (these are handled in Step 2)
        if any(_objects_match(obj, inst_obj) for inst_obj in instruction_objects):
            if verbose:
                print(f"  -> Skipping '{obj}' - matches instruction object (handled in Step 2)")
            continue
        
        # Skip obvious non-hallucination terms
        definite_skip = ["shadow", "lighting", "blur", "reflection", "glare"]
        if any(term in obj.lower() for term in definite_skip):
            if verbose:
                print(f"  -> Skipping '{obj}' - visual artifact")
            continue
        
        # Special case: Human hands and extra robotic arms are ALWAYS hallucinations in robot manipulation
        if _is_special_hallucination(obj):
            hallucinated_objects[obj] = 'SPECIAL_HALLUCINATION'
            if verbose:
                print(f"  -> SPECIAL HALLUCINATION (auto-confirmed): '{obj}'")
            continue
            
        # Thorough verification for each candidate
        q4 = f"""TASK: Verify if "{obj}" is a Random Hallucination

Object found in Frame 2: "{obj}"
Manipulation instruction: "{instruction}"
Instruction objects: {instruction_objects}

Please verify by answering:

Q1: Is "{obj}" visible ANYWHERE in Frame 1 (before image)?
- Look carefully at the entire Frame 1
- Answer: [YES_IN_FRAME1 or NO_NOT_IN_FRAME1]

Q2: Is "{obj}" one of the instruction objects ({instruction_objects}) or directly related to them?
- Answer: [YES_RELATED or NO_UNRELATED]

Q3: Is "{obj}" a shadow, reflection, or part of the robot arm?
- Answer: [YES_ARTIFACT or NO_REAL_OBJECT]

FINAL DETERMINATION:
- If Q1=NO_NOT_IN_FRAME1 AND Q2=NO_UNRELATED AND Q3=NO_REAL_OBJECT → HALLUCINATION
- Otherwise → NOT_HALLUCINATION

Q1_ANSWER: [YES_IN_FRAME1 or NO_NOT_IN_FRAME1]
Q2_ANSWER: [YES_RELATED or NO_UNRELATED]
Q3_ANSWER: [YES_ARTIFACT or NO_REAL_OBJECT]
FINAL: [HALLUCINATION or NOT_HALLUCINATION]"""

        out4 = _safe_chat_call(
            video_chat_model, verbose=verbose, video_path=tmp_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=q4, return_history=False,
            max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_MED
        )
        
        reply = out4.strip().lower()
        if verbose:
            print(f"Step 3 Verify '{obj}':\n{out4.strip()}")

        # Parse structured response
        is_hallucination = False
        
        # Check FINAL verdict
        if "final:" in reply:
            final_part = reply.split("final:")[-1].strip().split("\n")[0]
            if "not_hallucination" not in final_part and "not hallucination" not in final_part:
                if "hallucination" in final_part:
                    is_hallucination = True
        
        # Also check individual answers for more robust detection
        q1_not_in_frame1 = "no_not_in_frame1" in reply or "not_in_frame1" in reply
        q2_unrelated = "no_unrelated" in reply
        q3_real_object = "no_real_object" in reply
        
        if q1_not_in_frame1 and q2_unrelated and q3_real_object:
            is_hallucination = True
        
        if is_hallucination:
            # Before confirming as new object, check if it might be a misidentification
            # of an existing Frame 1 object (e.g., "blue ball" might be "blue cube")
            is_misidentification = False
            for f1_obj in frame1_objects:
                if _are_potential_duplicates(obj, f1_obj):
                    # Ask model if this "new object" is actually the existing object
                    if _model_verify_same_object(obj, f1_obj, tmp_video_path, 
                                                  video_chat_model, video_chat_tokenizer, verbose):
                        is_misidentification = True
                        if verbose:
                            print(f"  -> '{obj}' is likely a misidentification of existing '{f1_obj}' - skipping")
                        break
            
            if not is_misidentification:
                hallucinated_objects[obj] = 'NEW_OBJECT'  # Mark as new object (not duplication)
                if verbose:
                    print(f"  -> RANDOM HALLUCINATION CONFIRMED: '{obj}'")

    # =========================================================================
    # STEP 4: Final Verification Pass (Reduce False Positives)
    # =========================================================================
    
    final_hallucinations = []
    
    for obj, confidence in hallucinated_objects.items():
        # HIGH confidence detections for INSTRUCTION objects only are auto-confirmed
        # Background object duplications still need verification due to unreliable counting
        clean_obj = obj.replace(" (structural)", "")
        is_instruction_obj = any(_objects_match(clean_obj, inst_obj) for inst_obj in instruction_objects)
        
        if confidence == 'HIGH' and is_instruction_obj:
            final_hallucinations.append(clean_obj)
            if verbose:
                print(f"Step 4 - Auto-confirmed '{clean_obj}' (HIGH confidence instruction object)")
            continue
        
        # Special hallucinations (human hands, extra robot arms) are auto-confirmed
        if confidence == 'SPECIAL_HALLUCINATION':
            final_hallucinations.append(clean_obj)
            if verbose:
                print(f"Step 4 - Auto-confirmed '{clean_obj}' (special hallucination - shouldn't exist in robot manipulation)")
            continue
        
        # Substitution hallucinations (wrong object placed) need verification
        if confidence == 'SUBSTITUTION':
            # Verify the substitution claim
            q5_sub = f"""VERIFY SUBSTITUTION: Is "{clean_obj}" truly a hallucinated substitution?

The instruction mentioned placing a different object, but "{clean_obj}" appears at the destination instead.

Verify:
1. Is "{clean_obj}" visible in Frame 2 at a destination location?
2. Was "{clean_obj}" present in Frame 1?
3. Is this a NEW appearance of "{clean_obj}" (not just it moving)?

If "{clean_obj}" appeared NEW in Frame 2 as a substitution for the intended object → CONFIRMED_SUBSTITUTION
If "{clean_obj}" was already in Frame 1 or this is not a substitution → FALSE_POSITIVE

VERDICT: [CONFIRMED_SUBSTITUTION or FALSE_POSITIVE]"""

            out_sub = _safe_chat_call(
                video_chat_model, verbose=verbose, video_path=tmp_video_path,
                tokenizer=video_chat_tokenizer, user_prompt=q5_sub, return_history=False,
                max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_SHORT
            )
            
            sub_reply = out_sub.strip().lower()
            if verbose:
                print(f"Step 4 - Substitution verify '{clean_obj}': {out_sub.strip()}")
            
            if "confirmed_substitution" in sub_reply or "confirmed substitution" in sub_reply:
                final_hallucinations.append(clean_obj)
                if verbose:
                    print(f"  -> CONFIRMED (substitution): '{clean_obj}'")
            else:
                if verbose:
                    print(f"  -> REJECTED (false positive substitution): '{clean_obj}'")
            continue
        
        # All other detections need verification
        if verbose:
            print(f"Step 4 - Verifying '{obj}' (needs confirmation)")
        
        # Different verification based on hallucination type
        is_new_object = (confidence == 'NEW_OBJECT')
        
        if "(structural)" in obj:
            clean_obj = obj.replace(" (structural)", "")
            q5 = f"""FINAL VERIFICATION for structural change of: "{clean_obj}"

Instruction: "{instruction}"

Compare the SHAPE and STRUCTURE of "{clean_obj}" between Frame 1 and Frame 2.

Look for SIGNIFICANT structural differences:
- Is the shape drastically different (bent, deformed, stretched)?
- Are parts missing or added?
- Does it look like a DIFFERENT "{clean_obj}" was substituted?

Position/location changes are NORMAL and expected.
Lighting/angle differences are NORMAL.
But actual SHAPE changes to the object itself are HALLUCINATIONS.

FINAL_VERDICT: [STRUCTURAL_HALLUCINATION or NO_SIGNIFICANT_CHANGE]"""
        elif is_new_object:
            # For NEW objects that weren't in Frame 1 at all
            q5 = f"""FINAL VERIFICATION for NEW object: "{obj}"

Instruction: "{instruction}"

TASK: Verify if "{obj}" is a NEW object that appeared in Frame 2 but was NOT in Frame 1.

1. Look at Frame 1 (BEFORE) - Is there ANY "{obj}" visible? Search the entire frame.
2. Look at Frame 2 (AFTER) - Is there a "{obj}" visible?

If "{obj}" is in Frame 2 but NOT in Frame 1 → CONFIRMED_HALLUCINATION (new object appeared)
If "{obj}" is in BOTH frames → FALSE_POSITIVE (it was already there)
If "{obj}" is in NEITHER frame → FALSE_POSITIVE (it doesn't exist)

Is "{obj}" in Frame 1: [YES or NO]
Is "{obj}" in Frame 2: [YES or NO]

FINAL_VERDICT: [CONFIRMED_HALLUCINATION or FALSE_POSITIVE]"""
        else:
            # For background object duplications or LOW confidence instruction objects, be more skeptical
            # Ask for explicit counts
            q5 = f"""FINAL VERIFICATION for object: "{obj}"

Instruction: "{instruction}"

TASK: CAREFULLY verify if "{obj}" truly INCREASED in count between frames.

STEP 1: Count ALL instances of "{obj}" in Frame 1 (BEFORE image)
STEP 2: Count ALL instances of "{obj}" in Frame 2 (AFTER image)

BE PRECISE - only count clear, distinct, complete instances of "{obj}".
Don't count partial views, shadows, reflections, or similar-looking but different objects.

REQUIRED FORMAT - Fill in ALL fields:
Count_in_Frame1: [put exact number here]
Count_in_Frame2: [put exact number here]
Count_increased: [YES if Frame2 > Frame1, NO otherwise]

FINAL_VERDICT: [CONFIRMED_HALLUCINATION if count increased, FALSE_POSITIVE if count same or decreased]"""

        out5 = _safe_chat_call(
            video_chat_model, verbose=verbose, video_path=tmp_video_path,
            tokenizer=video_chat_tokenizer, user_prompt=q5, return_history=False,
            max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_MED
        )
        
        reply = out5.strip().lower()
        reply_original = out5.strip()  # Keep original case for display
        if verbose:
            print(f"Step 4 - Final verify '{obj}':\n{reply_original}")

        # Check for confirmed hallucination - multiple detection methods
        is_confirmed = False
        is_rejected = False
        
        clean_obj = obj.replace(" (structural)", "")
        
        # Method 1: Check for explicit verdict keywords anywhere in response
        if "confirmed_hallucination" in reply or "confirmed hallucination" in reply:
            is_confirmed = True
        if "structural_hallucination" in reply or "structural hallucination" in reply:
            is_confirmed = True
        if "false_positive" in reply or "false positive" in reply:
            is_rejected = True
        if "no_significant_change" in reply or "no significant change" in reply:
            is_rejected = True
        
        # Method 2: Check for FINAL_VERDICT: or similar patterns
        for marker in ["final_verdict:", "final verdict:", "verdict:", "final:"]:
            if marker in reply:
                verdict_part = reply.split(marker)[-1].strip().split("\n")[0]
                if "confirmed" in verdict_part or ("hallucination" in verdict_part and "not" not in verdict_part and "false" not in verdict_part):
                    is_confirmed = True
                elif "false" in verdict_part or "not" in verdict_part:
                    is_rejected = True
                break
        
        # Method 3: Check for duplication evidence language
        duplication_phrases = [
            "two instances", "two copies", "appears twice", "duplicated", 
            "duplicate", "both positions", "original still", "two " + clean_obj,
            "can see two", "see both", "2 instances", "2 copies"
        ]
        for phrase in duplication_phrases:
            if phrase in reply:
                # Check it's not negated in nearby context
                idx = reply.find(phrase)
                context_before = reply[max(0, idx-30):idx]
                if "not" not in context_before and "no " not in context_before and "don't" not in context_before and "only one" not in context_before:
                    is_confirmed = True
                    break
        
        # Method 4: Parse counts from response - BUT only use for instruction objects
        # Background objects have unreliable counts, so don't use count parsing to confirm them
        clean_obj_for_check = obj.replace(" (structural)", "")
        is_instr_obj = any(_objects_match(clean_obj_for_check, inst_obj) for inst_obj in instruction_objects)
        
        if is_instr_obj:
            # For instruction objects, we can trust count parsing more
            count_f1 = None
            count_f2 = None
            count_increased = None
            
            for line in reply.split("\n"):
                line_lower = line.lower().strip()
                # Try both formats for count parsing
                if "count_in_frame1:" in line_lower or "count in frame 1:" in line_lower:
                    if ":" in line:
                        nums = re.findall(r'\d+', line.split(":")[-1])
                        if nums:
                            count_f1 = int(nums[0])
                elif "count_in_frame2:" in line_lower or "count in frame 2:" in line_lower:
                    if ":" in line:
                        nums = re.findall(r'\d+', line.split(":")[-1])
                        if nums:
                            count_f2 = int(nums[0])
                elif "count" in line_lower and "frame 1" in line_lower:
                    if ":" in line:
                        nums = re.findall(r'\d+', line.split(":")[-1])
                        if nums:
                            count_f1 = int(nums[0])
                elif "count" in line_lower and "frame 2" in line_lower:
                    if ":" in line:
                        nums = re.findall(r'\d+', line.split(":")[-1])
                        if nums:
                            count_f2 = int(nums[0])
                elif "count_increased:" in line_lower or "count increased:" in line_lower:
                    if ":" in line:
                        inc_part = line.split(":")[-1].strip().lower()
                        if "yes" in inc_part and "no" not in inc_part:
                            count_increased = True
                        elif "no" in inc_part:
                            count_increased = False
            
            if verbose and (count_f1 is not None or count_f2 is not None or count_increased is not None):
                print(f"  [Step 4 counts] F1:{count_f1} F2:{count_f2} Increased:{count_increased}")
            
            # If count increased, it's a hallucination
            if count_f1 is not None and count_f2 is not None and count_f2 > count_f1:
                is_confirmed = True
            # If count is 0 in frame 1 and > 0 in frame 2, it's a new object hallucination
            if count_f1 == 0 and count_f2 is not None and count_f2 > 0:
                is_confirmed = True
            # Also trust explicit count_increased: YES
            if count_increased == True:
                is_confirmed = True
            # Note: We do NOT reject based on counts alone since model counting is unreliable.
            # The verb-based check handles non-duplication verbs (flip, close, open etc.)
        
        # Final decision: For background objects, require explicit CONFIRMED_HALLUCINATION verdict
        # For instruction objects, confirmed wins unless explicitly rejected
        if is_instr_obj:
            if is_confirmed and not is_rejected:
                # For LOW confidence instruction objects with non-duplication verbs,
                # require visual verification of seeing two separate instances.
                # This catches false positives where an object was merely flipped/turned
                # but the model incorrectly says it was duplicated.
                # NOTE: Only use verbs that imply in-place manipulation (no location change).
                # Do NOT include put/place/move — those can produce genuine duplications.
                action_verbs_non_dup = ['flip', 'close', 'open', 'turn', 'rotate', 'raise', 'lower',
                                        'upright']
                needs_extra_verify = (confidence == 'LOW' and 
                                      any(v in instruction.lower() for v in action_verbs_non_dup))
                # Also force extra verify if there was an inventory contradiction
                # (count=0 in F1 but object was in inventory — model counting is unreliable)
                if clean_obj in inventory_contradictions:
                    needs_extra_verify = True
                # Also force extra verify for destination objects in placement instructions
                # ("put X in/on Y" — Y shouldn't be duplicated, only X can be)
                if confidence == 'LOW' and destination_objects:
                    is_dest = any(_objects_match(clean_obj, dest) for dest in destination_objects)
                    if is_dest:
                        needs_extra_verify = True
                
                # GDINO cross-check for LOW confidence instruction objects:
                # If GDINO says no increase, the VLM confirmation is likely wrong
                gdino_rejected = False
                gdino_rejected_high_count = False  # Flag: rejection due to high noisy counts
                if confidence == 'LOW' and not needs_extra_verify:
                    gd_f1, gd_f2 = gdino_count_both_frames(img_before_path, img_after_path, clean_obj)
                    if verbose:
                        print(f"  [GDINO cross-check] '{clean_obj}': F1={gd_f1}, F2={gd_f2}")
                    if (gd_f1 > 0 or gd_f2 > 0) and gd_f2 == gd_f1 and max(gd_f1, gd_f2) <= 3:
                        # GDINO reliably counts SAME number in both frames → no increase
                        gdino_rejected = True
                        if verbose:
                            print(f"  [GDINO override] Same count ({gd_f1}=={gd_f2}), rejecting LOW-confidence detection")
                    elif max(gd_f1, gd_f2) > 5 and (gd_f2 - gd_f1) <= 1:
                        # Suspiciously high GDINO counts with marginal increase → noise
                        gdino_rejected = True
                        gdino_rejected_high_count = True
                        if verbose:
                            print(f"  [GDINO override] High counts ({gd_f1}->{gd_f2}), marginal increase is noise")
                
                if gdino_rejected and gdino_rejected_high_count:
                    # High-count rejection: GDINO is matching too many wrong objects.
                    # Don't attempt visual rescue — both GDINO and VLM counting are unreliable.
                    if verbose:
                        print(f"  -> REJECTED (high GDINO noise): '{clean_obj}'")
                elif gdino_rejected:
                    # Instead of rejecting outright, give it a visual rescue chance.
                    # GDINO might miss objects it can't detect (e.g., colored plates).
                    q5_gd_rescue = f"""SECOND OPINION: Is "{clean_obj}" duplicated?

Look at Frame 2 (AFTER image) ONLY.
Search the ENTIRE frame carefully for "{clean_obj}".

Can you find TWO separate, distinct instances of "{clean_obj}" visible at different locations in Frame 2?

Instance 1: [location, or "not found"]
Instance 2: [location, or "not found"]

Answer: [TWO_FOUND if you found two separate instances, ONE_OR_NONE otherwise]"""
                    out_gd_rescue = _safe_chat_call(
                        video_chat_model, verbose=verbose, video_path=tmp_video_path,
                        tokenizer=video_chat_tokenizer, user_prompt=q5_gd_rescue, return_history=False,
                        max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_SHORT
                    )
                    gd_rescue_reply = out_gd_rescue.strip().lower()
                    if verbose:
                        print(f"  [Visual rescue after GDINO override] '{clean_obj}': {out_gd_rescue.strip()}")
                    gd_rescue_ok = ("two_found" in gd_rescue_reply or "two found" in gd_rescue_reply or 
                                    "two separate" in gd_rescue_reply or
                                    (re.search(r'\btwo\b', gd_rescue_reply) and 
                                     "one_or_none" not in gd_rescue_reply and "one or none" not in gd_rescue_reply and
                                     "not found" not in gd_rescue_reply))
                    if gd_rescue_ok:
                        final_hallucinations.append(clean_obj)
                        if verbose:
                            print(f"  -> RESCUED (visual override of GDINO): '{clean_obj}'")
                    else:
                        if verbose:
                            print(f"  -> REJECTED (GDINO override + visual confirms): '{clean_obj}'")
                elif needs_extra_verify:
                    q5_dbl = f"""STRICT VERIFICATION: Is "{clean_obj}" duplicated in Frame 2?

Look at Frame 2 ONLY. Can you find TWO or more distinct, separate instances of "{clean_obj}" at different locations?

If you see only ONE "{clean_obj}", it is NOT duplicated.

Describe each instance:
Instance 1: [location]
Instance 2: [location or N/A if only one exists]

VERDICT: [TWO_OR_MORE or ONLY_ONE]"""
                    out_dbl = _safe_chat_call(
                        video_chat_model, verbose=verbose, video_path=tmp_video_path,
                        tokenizer=video_chat_tokenizer, user_prompt=q5_dbl, return_history=False,
                        max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_SHORT
                    )

                    dbl_reply = out_dbl.strip().lower()
                    if verbose:
                        print(f"  [Extra verify] '{clean_obj}': {out_dbl.strip()}")

                    if "two_or_more" in dbl_reply or "two or more" in dbl_reply or (re.search(r'\btwo\b', dbl_reply) and "only_one" not in dbl_reply and "only one" not in dbl_reply):
                        final_hallucinations.append(clean_obj)
                        if verbose:
                            print(f"  -> CONFIRMED (extra-verified): '{clean_obj}'")
                    else:
                        if verbose:
                            print(f"  -> REJECTED (only one found): '{clean_obj}'")
                else:
                    final_hallucinations.append(clean_obj)
                    if verbose:
                        print(f"  -> CONFIRMED: '{clean_obj}'")
            elif is_rejected and confidence == 'LOW':
                # Standard Step 4 rejected this LOW confidence detection.
                # Give it a visual rescue: ask if there are genuinely two instances.
                # BUT: skip rescue for destination objects — destinations are just the target
                # location and shouldn't be duplicated. If both Step 2 and Step 4 rejected,
                # trust that verdict.
                is_dest_obj = destination_objects and any(_objects_match(clean_obj, dest) for dest in destination_objects)
                # Exception: if Step 2 GDINO confirmed this object's count actually increased,
                # allow rescue even for destination objects (GDINO is deterministic evidence)
                gdino_confirmed_increase = False
                if clean_obj in step2_counts:
                    s2_f1, s2_f2 = step2_counts[clean_obj]
                elif any(_objects_match(clean_obj, k) for k in step2_counts):
                    matched_key = next(k for k in step2_counts if _objects_match(clean_obj, k))
                    s2_f1, s2_f2 = step2_counts[matched_key]
                else:
                    s2_f1, s2_f2 = None, None
                # Re-check GDINO for this specific object
                gd_rescue_f1, gd_rescue_f2 = gdino_count_both_frames(img_before_path, img_after_path, clean_obj)
                # For destination objects: require significant GDINO increase (≥2)
                # Marginal increases (+1) can be GDINO noise, especially with overlapping
                # objects (sushi ON omelette counted as 2 omelettes)
                if gd_rescue_f2 - gd_rescue_f1 >= 2 and max(gd_rescue_f1, gd_rescue_f2) <= 3:
                    gdino_confirmed_increase = True
                    if verbose:
                        print(f"  [GDINO dest rescue] '{clean_obj}': {gd_rescue_f1}->{gd_rescue_f2}, allowing rescue")
                if is_dest_obj and not gdino_confirmed_increase:
                    if verbose:
                        print(f"  -> REJECTED (destination object, no rescue): '{clean_obj}'")
                else:
                    q5_rescue = f"""SECOND OPINION: Is "{clean_obj}" duplicated?

Look at Frame 2 (AFTER image) ONLY.
Search the ENTIRE frame carefully for "{clean_obj}".

Can you find TWO separate, distinct instances of "{clean_obj}" visible at different locations in Frame 2?

Instance 1: [location, or "not found"]
Instance 2: [location, or "not found"]

Answer: [TWO_FOUND if you found two separate instances, ONE_OR_NONE otherwise]"""
                    out_rescue = _safe_chat_call(
                        video_chat_model, verbose=verbose, video_path=tmp_video_path,
                        tokenizer=video_chat_tokenizer, user_prompt=q5_rescue, return_history=False,
                        max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_SHORT
                    )

                    rescue_reply = out_rescue.strip().lower()
                    if verbose:
                        print(f"  [Visual rescue] '{clean_obj}': {out_rescue.strip()}")

                    rescue_confirmed = ("two_found" in rescue_reply or "two found" in rescue_reply or "two separate" in rescue_reply or (re.search(r'\btwo\b', rescue_reply) and "one_or_none" not in rescue_reply and "one or none" not in rescue_reply and "not found" not in rescue_reply))
                    # GDINO sanity check on rescue: if GDINO reliably says same count, 
                    # reject ONLY for destination objects (GDINO may miss specific objects
                    # like "blue pen" that the VLM correctly identifies)
                    is_dest_for_rescue = False
                    if destination_objects:
                        clean_norm = _normalize_object_name(clean_obj)
                        for dest in destination_objects:
                            dest_norm = _normalize_object_name(dest)
                            if _objects_match(clean_obj, dest):
                                is_dest_for_rescue = True
                                break
                            # Also check if any word in clean_obj appears in dest or vice versa
                            # (e.g., "blue plate" shares "plate" with "plate cardboard fence")
                            clean_words = set(clean_norm.split())
                            dest_words = set(dest_norm.split())
                            if clean_words & dest_words:
                                is_dest_for_rescue = True
                                break
                    if rescue_confirmed and is_dest_for_rescue:
                        gd_resc_f1, gd_resc_f2 = gdino_count_both_frames(img_before_path, img_after_path, clean_obj)
                        if (gd_resc_f1 > 0 or gd_resc_f2 > 0) and gd_resc_f2 == gd_resc_f1 and max(gd_resc_f1, gd_resc_f2) <= 3:
                            rescue_confirmed = False
                            if verbose:
                                print(f"  [GDINO rescue override] '{clean_obj}': same count ({gd_resc_f1}=={gd_resc_f2}), blocking rescue")
                    if rescue_confirmed:
                        final_hallucinations.append(clean_obj)
                        if verbose:
                            print(f"  -> RESCUED: '{clean_obj}' (visual check found two instances)")
                    else:
                        if verbose:
                            print(f"  -> REJECTED (rescue failed): '{clean_obj}'")
            else:
                if verbose:
                    print(f"  -> REJECTED (false positive): '{clean_obj}'")
        elif is_new_object:
            # For NEW objects (not duplications), just check verdict - no count increase needed
            verdict_confirmed = False
            
            for marker in ["final_verdict:", "final verdict:", "verdict:", "final:"]:
                if marker in reply:
                    verdict_part = reply.split(marker)[-1].strip().split("\n")[0]
                    if "confirmed_hallucination" in verdict_part or "confirmed hallucination" in verdict_part:
                        verdict_confirmed = True
                    break
            
            if not verdict_confirmed:
                if "confirmed_hallucination" in reply or "confirmed hallucination" in reply:
                    if "not confirmed" not in reply and "not_confirmed" not in reply:
                        verdict_confirmed = True
            
            # For new objects, also check if Frame 2 has YES and Frame 1 has NO
            frame1_no = False
            frame2_yes = False
            for line in reply.split("\n"):
                line_lower = line.lower().strip()
                if "frame 1" in line_lower and ": no" in line_lower:
                    frame1_no = True
                if "frame 2" in line_lower and ": yes" in line_lower:
                    frame2_yes = True
            if frame1_no and frame2_yes:
                verdict_confirmed = True
            
            if verdict_confirmed and not is_rejected:
                final_hallucinations.append(clean_obj)
                if verbose:
                    print(f"  -> CONFIRMED (new object): '{clean_obj}'")
            else:
                if verbose:
                    print(f"  -> REJECTED (false positive): '{clean_obj}'")
        else:
            # For background objects, be VERY STRICT - require DOUBLE verification
            # This reduces false positives from unreliable model counting
            
            # First check: parse the verification response
            verdict_confirmed = False
            
            # Method 1: Check for marker-prefixed verdict
            for marker in ["final_verdict:", "final verdict:", "verdict:", "final:"]:
                if marker in reply:
                    verdict_part = reply.split(marker)[-1].strip().split("\n")[0]
                    if "confirmed_hallucination" in verdict_part or "confirmed hallucination" in verdict_part:
                        verdict_confirmed = True
                    break
            
            # Method 2: Check if response contains CONFIRMED_HALLUCINATION anywhere
            if not verdict_confirmed:
                if "confirmed_hallucination" in reply or "confirmed hallucination" in reply:
                    if "not confirmed" not in reply and "not_confirmed" not in reply:
                        verdict_confirmed = True
            
            # For background objects: REQUIRE a SECOND verification to reduce false positives
            # First, parse Step 4 counts and compare with Step 3a counts for consistency  
            step4_count_f1 = None
            step4_count_f2 = None
            for line in reply.split("\n"):
                line_lower = line.lower().strip()
                if "count_in_frame1:" in line_lower or "count in frame 1:" in line_lower:
                    if ":" in line:
                        nums = re.findall(r'\d+', line.split(":")[-1])
                        if nums:
                            step4_count_f1 = int(nums[0])
                elif "count_in_frame2:" in line_lower or "count in frame 2:" in line_lower:
                    if ":" in line:
                        nums = re.findall(r'\d+', line.split(":")[-1])
                        if nums:
                            step4_count_f2 = int(nums[0])
            
            # Cross-check: if Step 3a and Step 4 disagree on Frame 1 count by > 1,
            # the model is unreliable at counting this object → reject
            # EXCEPTION: if Step 4 sees FEWER objects in Frame 2 than Step 3a,
            # Step 4 may be undercounting (missing the hallucinated object) → give it a chance
            counts_unreliable = False
            if clean_obj in step3a_counts and step4_count_f1 is not None:
                s3a_f1, s3a_f2 = step3a_counts[clean_obj]
                if abs(s3a_f1 - step4_count_f1) > 1:
                    counts_unreliable = True
                    # Check if Step 4 undercounts Frame 2 vs Step 3a
                    # If so, Step 4 might be missing the new object → proceed to double-check
                    if step4_count_f2 is not None and s3a_f2 is not None and step4_count_f2 < s3a_f2:
                        counts_unreliable = False  # Override: let DEFINITELY_YES decide
                        if verbose:
                            print(f"  [Count mismatch override] Step3a=({s3a_f1},{s3a_f2}) Step4=({step4_count_f1},{step4_count_f2}) → Step4 undercounts F2, proceeding to double-check")
                    else:
                        if verbose:
                            print(f"  [Count mismatch] Step3a_F1={s3a_f1} vs Step4_F1={step4_count_f1}, F2 agrees → unreliable, rejecting")
            
            if counts_unreliable:
                if verbose:
                    print(f"  -> REJECTED (count disagreement): '{clean_obj}'")
            elif verdict_confirmed and not is_rejected:
                # Double-check background objects using GDINO + VLM fallback
                # Key insight: GDINO is deterministic, so re-checking with same threshold
                # always gives the same result → only trust large GDINO increases (>=2)
                # For marginal increases (+1) or GDINO-invisible objects, fall back to VLM
                gdino_verified = False
                gdino_f1_v, gdino_f2_v = gdino_count_both_frames(img_before_path, img_after_path, clean_obj)
                gdino_diff = gdino_f2_v - gdino_f1_v
                if verbose:
                    print(f"  [GDINO double-check] '{clean_obj}': F1={gdino_f1_v}, F2={gdino_f2_v}")
                
                if gdino_f1_v > 0 or gdino_f2_v > 0:
                    # GDINO can see this object
                    # Sanity check: if either count is suspiciously high (>3),
                    # GDINO may be matching the wrong objects → treat as unreliable
                    gdino_reliable = max(gdino_f1_v, gdino_f2_v) <= 3
                    
                    if gdino_diff >= 2 and gdino_reliable:
                        # Large increase with reliable counts
                        gdino_verified = True
                        if verbose:
                            print(f"  -> GDINO confirms large increase ({gdino_f1_v} → {gdino_f2_v})")
                    elif gdino_diff >= 1 and not gdino_reliable:
                        # High counts — GDINO likely matching wrong objects
                        if verbose:
                            print(f"  -> GDINO counts suspiciously high ({gdino_f1_v} → {gdino_f2_v}), asking VLM...")
                    elif gdino_diff == 1:
                        # Marginal increase — could be GDINO noise → ask VLM for confirmation
                        if verbose:
                            print(f"  -> GDINO marginal increase ({gdino_f1_v} → {gdino_f2_v}), asking VLM...")
                        # Fall through to VLM DEFINITELY_YES below
                    else:
                        # No increase or decrease — GDINO says no duplication
                        if verbose:
                            print(f"  -> GDINO says NO increase ({gdino_f1_v} → {gdino_f2_v})")
                
                # VLM fallback for: GDINO invisible (both 0), or marginal +1 increase
                if not gdino_verified and gdino_diff < 2:
                    q5_verify = f"""STRICT VERIFICATION: Is "{clean_obj}" truly duplicated?

In Frame 1 (BEFORE), count the exact number of "{clean_obj}" visible.
In Frame 2 (AFTER), count the exact number of "{clean_obj}" visible.

Are you 100% certain that Frame 2 has MORE instances of "{clean_obj}" than Frame 1?
This must be a genuinely new, separate object - not just the same object viewed differently.

Answer: [DEFINITELY_YES or NOT_CERTAIN]"""
                    
                    out_verify = _safe_chat_call(
                        video_chat_model, verbose=verbose, video_path=tmp_video_path,
                        tokenizer=video_chat_tokenizer, user_prompt=q5_verify, return_history=False,
                        max_num_frames=MAX_NUM_FRAMES, generation_config=GEN_SHORT
                    )
                    
                    verify_reply = out_verify.strip().lower()
                    if verbose:
                        print(f"  [VLM double-check] '{clean_obj}': {out_verify.strip()}")
                    
                    if "definitely_yes" in verify_reply or "definitely yes" in verify_reply:
                        gdino_verified = True
                
                if gdino_verified:
                    final_hallucinations.append(clean_obj)
                    if verbose:
                        print(f"  -> CONFIRMED (double-verified): '{clean_obj}'")
                else:
                    if verbose:
                        print(f"  -> REJECTED (failed double-check): '{clean_obj}'")
            else:
                # VLM said false positive for background object.
                # But if GDINO reliably shows a clear increase, trust GDINO.
                # GUARD: Only rescue if VLM at least SAW the object (step4 count > 0).
                # If VLM reports 0→0 (can't see ANY instances), GDINO is likely misdetecting.
                gdino_rescue_bg = False
                vlm_saw_object = (step4_count_f1 is not None and step4_count_f1 > 0) or \
                                 (step4_count_f2 is not None and step4_count_f2 > 0)
                if clean_obj in step3a_counts and vlm_saw_object:
                    s3a_f1, s3a_f2 = step3a_counts[clean_obj]
                    s3a_diff = s3a_f2 - s3a_f1
                    s3a_reliable = max(s3a_f1, s3a_f2) <= 3 and s3a_f1 > 0
                    # Require +2 increase for background rescue (not just +1)
                    # A single +1 increase is too unreliable for GDINO on background objects
                    # (easily caused by occlusion shifts, slight angle changes)
                    if s3a_diff >= 2 and s3a_reliable:
                        # GDINO Step 3a saw reliable increase — do a recheck
                        gd_bg_f1, gd_bg_f2 = gdino_count_both_frames(img_before_path, img_after_path, clean_obj)
                        gd_bg_reliable = max(gd_bg_f1, gd_bg_f2) <= 3 and gd_bg_f1 > 0
                        if gd_bg_f2 > gd_bg_f1 and gd_bg_reliable:
                            gdino_rescue_bg = True
                            final_hallucinations.append(clean_obj)
                            if verbose:
                                print(f"  -> GDINO RESCUE (background): '{clean_obj}' count {gd_bg_f1}->{gd_bg_f2} (VLM unreliable)")
                elif clean_obj in step3a_counts and not vlm_saw_object:
                    if verbose:
                        print(f"  -> GDINO RESCUE BLOCKED: VLM saw 0 instances of '{clean_obj}' (F1:{step4_count_f1}, F2:{step4_count_f2})")
                if not gdino_rescue_bg:
                    if verbose:
                        print(f"  -> REJECTED (false positive): '{clean_obj}'")
    # Use the model to visually verify if two detected hallucinations are the same object
    if verbose and len(final_hallucinations) > 1:
        print(f"Step 5 - Deduplication check for: {final_hallucinations}")
    final_hallucinations = _deduplicate_hallucinations_with_model(
        final_hallucinations, instruction_objects, 
        tmp_video_path, video_chat_model, video_chat_tokenizer, verbose
    )
    
    return len(final_hallucinations), final_hallucinations


# -------------------------
# Loader
# -------------------------
def load_videochat_model_and_tokenizer(model_name=MODEL_NAME):
    """Load the VideoChat-Flash model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.mm_llm_compress = False
    
    # Also load Grounding DINO (lightweight, ~2GB VRAM)
    load_grounding_dino()
    
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
            "expected_count": 2,
            "description": "A hallucinated sushi is put on top of a hallucinated omelette. The originals remain at their original places respectively"
        },
        {
            "before": "lgie-eval-data/input_3.jpg",
            "after": "lgie-eval-data/output_3_2.png",
            "instruction": "The robotic arm puts the sushi on top of the omelette",
            "expected_count": 0,
            "description": "The original sushi is put on top of the original omelette -- no hallucinated objects"
        },
        {
            "before": "lgie-eval-data/input_1.jpg",
            "after": "lgie-eval-data/output_1_hal.png",
            "instruction": "The robotic arm puts the blue spoon on top of the cube",
            "expected_count": 3,
            "description": "A new banana is hallucinated at the far left (background object duplication -- opposite to where the existing banana is). A new red ball is hallucinated (new object -- appears out of nowhere). The blue spoon is structurally changed significantly (thus we can say that the new spoon structure is a hallucinated one), but is placed correctly on top of the cube as mentioned in the given manipulation instruction."
        },
        {
          "id": 69,
          "before": "lgie-eval-data/original_image/oi_69.png",
          "after": "lgie-eval-data/r4_gemini_output/out_69.png",
          "instruction": "pick up violet Allen key",
          "hallucinated_objects": [
            "another violet allen key"
          ],
          "expected_count": 1,
          "description": "A hallucinated violet allen key appeared being held by robotic arm."
        },
        {
          "id": 70,
          "before": "lgie-eval-data/original_image/oi_70.png",
          "after": "lgie-eval-data/r4_gemini_output/out_70.png",
          "instruction": "put lid on stove",
          "hallucinated_objects": [
            "another robotic arm"
          ],
          "expected_count": 1,
          "description": "Lid is being held by a hallucinated robotic arm on the right of the scene whereas the original robotic arm stays above the stove on the top left."
        },
        {
          "id": 71,
          "before": "lgie-eval-data/original_image/oi_71.png",
          "after": "lgie-eval-data/r4_gemini_output/out_71.png",
          "instruction": "take carrot off plate",
          "hallucinated_objects": ["green plate"],
          "expected_count": 1,
          "description": "A hallucinated green plate has occured near the pot in the scene."
        },
        {
          "id": 72,
          "before": "lgie-eval-data/original_image/oi_72.png",
          "after": "lgie-eval-data/r4_gemini_output/out_72.png",
          "instruction": "put knife on cutting board cardboard fence",
          "hallucinated_objects": [
            "another knife"
          ],
          "expected_count": 1,
          "description": "Another knife is hallucinated by the model and is placed on cutting board."
        },
        {
          "id": 73,
          "before": "lgie-eval-data/original_image/oi_73.png",
          "after": "lgie-eval-data/r4_gemini_output/out_73.png",
          "instruction": "put pan in sink",
          "hallucinated_objects": [],
          "expected_count": 0,
          "description": "Pan is put in sink -- no hallucinated objects."
        },
        {
          "id": 74,
          "before": "lgie-eval-data/original_image/oi_74.png",
          "after": "lgie-eval-data/r4_gemini_output/out_74.png",
          "instruction": "put cucumber in cup",
          "hallucinated_objects": [],
          "expected_count": 0,
          "description": "Cucumber is put in cup -- no hallucinated_objects"
        },
        {
          "id": 75,
          "before": "lgie-eval-data/original_image/oi_75.png",
          "after": "lgie-eval-data/r4_gemini_output/out_75.png",
          "instruction": "put bowl on plate cardboard fence",
          "hallucinated_objects": [
            "another bowl"
          ],
          "expected_count": 1,
          "description": "Another blue bowl has been hallucinated by the model and is placed on its side on blue plate."
        },
        {
          "id": 76,
          "before": "lgie-eval-data/original_image/oi_76.png",
          "after": "lgie-eval-data/r4_gemini_output/out_76.png",
          "instruction": "close oven",
          "hallucinated_objects": [
            "A white, flat object that appears to be a toy plate is resting on the bottom rack of the oven"
          ],
          "expected_count": 1,
          "description": "A white, flat object that appears to be a toy plate which is resting on the bottom rack of the oven has been hallucinated by the model"
        },
        {
          "id": 77,
          "before": "lgie-eval-data/original_image/oi_77.png",
          "after": "lgie-eval-data/r4_gemini_output/out_77.png",
          "instruction": "put detergent in sink",
          "hallucinated_objects": [
            "detergent fluid spilled in sink"
          ],
          "expected_count": 1,
          "description": "Model has hallucinated detergent fluid spilled in sink."
        },
        {
          "id": 78,
          "before": "lgie-eval-data/original_image/oi_78.png",
          "after": "lgie-eval-data/r4_gemini_output/out_78.png",
          "instruction": "put knife on cutting board cardboard fence",
          "hallucinated_objects": [
            "another knife placed on cardboard fence"
          ],
          "expected_count": 1,
          "description": "An hallucinated knife is placed on perimeter of cardboard fence."
        },
        {
          "id": 79,
          "before": "lgie-eval-data/original_image/oi_79.png",
          "after": "lgie-eval-data/r4_gemini_output/out_79.png",
          "instruction": "put broccoli in pot",
          "hallucinated_objects": [],
          "expected_count": 0,
          "description": "Broccolis is put in pan -- no hallucinated objects."
        },
        {
          "id": 80,
          "before": "lgie-eval-data/original_image/oi_80.png",
          "after": "lgie-eval-data/r4_gemini_output/out_80.png",
          "instruction": "put clothes in laundry machine",
          "hallucinated_objects": [
            "red, green, blue clothes"
          ],
          "expected_count": 1,
          "description": "Hallucinated red, green, blue clothes have appeared in the spinning tub of laundry machine."
        },
        {
          "id": 81,
          "before": "lgie-eval-data/original_image/oi_81.png",
          "after": "lgie-eval-data/r4_gemini_output/out_81.png",
          "instruction": "put corn in pan which is on stove",
          "hallucinated_objects": [
            "human hand",
            "extra corn"
          ],
          "expected_count": 2,
          "description": "A hallucinated hand is placing an hallucinated corn in the scene."
        },
        {
          "id": 82,
          "before": "lgie-eval-data/original_image/oi_82.png",
          "after": "lgie-eval-data/r4_gemini_output/out_82.png",
          "instruction": "put sweet potato in pot which is in sink",
          "hallucinated_objects": [
            "yellow bell pepper"
          ],
          "expected_count": 1,
          "description": "The model misterpreted yellow bell pepper as sweet potato and hallucinated another yellow pepper in the pot."
        },
        {
          "id": 83,
          "before": "lgie-eval-data/original_image/oi_83.png",
          "after": "lgie-eval-data/r4_gemini_output/out_83.png",
          "instruction": "flip pot upright in sink",
          "hallucinated_objects": [
            "corn",
            "green vegetable"
          ],
          "expected_count": 2,
          "description": "The model hallucinated a corn and a green vegetable inside the pot."
        },
        {
          "id": 84,
          "before": "lgie-eval-data/original_image/oi_84.png",
          "after": "lgie-eval-data/r4_gemini_output/out_84.png",
          "instruction": "put carrot on cutting board",
          "hallucinated_objects": [],
          "expected_count": 0,
          "description": "Carrot has been put on cutting board -- no hallucinated objects."
        },
        {
          "id": 85,
          "before": "lgie-eval-data/original_image/oi_85.png",
          "after": "lgie-eval-data/r4_gemini_output/out_85.png",
          "instruction": "take sushi off plate",
          "hallucinated_objects": [],
          "expected_count": 0,
          "description": "Sushi has been taken off the plate -- no hallucinated objects."
        },
        {
          "id": 86,
          "before": "lgie-eval-data/original_image/oi_86.png",
          "after": "lgie-eval-data/r4_gemini_output/out_86.png",
          "instruction": "put broccoli in pot or pan",
          "hallucinated_objects": [
            "human hand"
          ],
          "expected_count": 1,
          "description": "A hallucinated hand is placing the broccoli in the pot."
        },
        {
          "id": 87,
          "before": "lgie-eval-data/original_image/oi_87.png",
          "after": "lgie-eval-data/r4_gemini_output/out_87.png",
          "instruction": "put sweet potato in pan which is on stove",
          "hallucinated_objects": [
            "yellow corn"
          ],
          "expected_count": 1,
          "description": "Instead of red sweet potato model hallucinated a corn and put it in pan."
        },
        {
          "id": 88,
          "before": "lgie-eval-data/original_image/oi_88.png",
          "after": "lgie-eval-data/r4_gemini_output/out_88.png",
          "instruction": "pick up blue pen and put into drawer",
          "hallucinated_objects": [
            "another blue pen",
            "another robotic arm"
          ],
          "expected_count": 2,
          "description": "Model has hallucinated an extra robotic arm and an extra blue pen."
        },
        {
          "id": 89,
          "before": "lgie-eval-data/original_image/oi_89.png",
          "after": "lgie-eval-data/r4_gemini_output/out_89.png",
          "instruction": "flip pot upright in sink",
          "hallucinated_objects": [],
          "expected_count": 0,
          "description": "Pot is upright in sink -- no hallucinated objects."
        },
        {
          "id": 90,
          "before": "lgie-eval-data/original_image/oi_90.png",
          "after": "lgie-eval-data/r4_gemini_output/out_90.png",
          "instruction": "pick up pot from sink",
          "hallucinated_objects": [],
          "expected_count": 0,
          "description": "Pot is raised a little -- no hallucinated objects."
        },
        {
          "id": 91,
          "before": "lgie-eval-data/original_image/oi_91.png",
          "after": "lgie-eval-data/r4_gemini_output/out_91.png",
          "instruction": "put corn in pan which is on stove",
          "hallucinated_objects": [
            "another corn"
          ],
          "expected_count": 1,
          "description": "A hallucinated corn has appeared in the pan."
        }

    ]

    # Load model & tokenizer once
    tokenizer, model = load_videochat_model_and_tokenizer()

    total = len(test_cases)
    score_sum = 0.0

    for i, tc in enumerate(test_cases, start=1):
        predicted_count, objects = num_unmentioned_hallucinations(
            tc["before"], tc["after"], tc["instruction"],
            model, tokenizer,
            verbose=True  # Enable verbose for debugging
        )

        expected = tc["expected_count"]
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Test {i}: {tc['description']}")
        print(f"  Expected count: {expected}")
        print(f"  Predicted count: {predicted_count}")
        print(f"  Hallucinated objects: {objects}")
        
        # Calculate score
        case_score = 2 ** (-abs(predicted_count - expected))
        score_sum += case_score
        
        status = "✓ PASS" if predicted_count == expected else "✗ FAIL"
        print(f"  Score: {case_score:.4f} {status}")
        print(f"{'='*60}\n")

    success_score = score_sum / total
    print(f"\n{'#'*60}")
    print(f"Final success_score = {success_score:.6f} ({success_score * 100:.2f}%)")
    print(f"{'#'*60}")